#!/usr/bin/env python3
"""g1-interactive.py — VR-driven world-model rollout for the UWM G1 denoiser.

Pico VR controller + SMPL → GMR retarget → 22-D action → AutoRegressiveForward
Dynamics(mode='wm') → tokenizer decode → H.264 → PicoVideoRelay → headset.

End-to-end pipeline is the same as wm_pico_eval_kvcache.py from creo-g1-teleop;
the only differences are the model wrapper (this repo's `AutoRegressiveForward
Dynamics` over the UWM two-stream denoiser instead of the original DreamerV4
encoder/dynamics/decoder trio) and the seeding source (this repo's
`G1ChunkDataset` picks a random training window instead of a manual h5 slice).

Run alongside `pico-vr-bridge --port 5580` and point the Pico app at this
PC's IP : --video-port.
"""

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

# ── pico_vr / gmr_retargeter / pico_manager (live in creo-g1-teleop) ────
CREO = Path("/home/mim-server/robot/GR00T-WholeBodyControl/creo-g1-teleop")
GROOT = CREO.parent
for p in (str(GROOT),
          str(CREO / "pico_vr" / "src"),
          str(CREO / "gmr_retargeter" / "src"),
          str(GROOT / "gear_sonic" / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pico_vr import PicoVR
from gmr_retargeter import GMRRetargeter
from pico_manager.video_relay import PicoVideoRelay

# ── this repo ───────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hydra import initialize_config_dir, compose

from dreamerv4uwm.datasets import G1ChunkDataset
from dreamerv4uwm.models.utils import load_denoiser, load_tokenizer
from dreamerv4uwm.sampling import AutoRegressiveForwardDynamics


# ── Action assembly (parity with creo-g1-teleop/wm_pico_eval_kvcache.py) ──
ACTION_DIM = 22
MAX_LINEAR_VEL = 0.3
MAX_LINEAR_LAT_VEL = 0.4
MAX_YAW_RATE = 1.5
JOYSTICK_DEADZONE = 0.05
UPPER_BODY_INDICES = np.array(
    [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.intp,
)


def apply_deadzone(v: float, dz: float) -> float:
    if abs(v) < dz:
        return 0.0
    s = 1.0 if v > 0 else -1.0
    return s * (abs(v) - dz) / (1.0 - dz)


def build_action(ctrl, qpos_29: np.ndarray) -> np.ndarray:
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    if ctrl is None:
        a[3:20] = qpos_29[UPPER_BODY_INDICES]
        return a
    a[0] = apply_deadzone(ctrl.left_axis[1], JOYSTICK_DEADZONE) * MAX_LINEAR_VEL
    a[1] = apply_deadzone(-ctrl.left_axis[0], JOYSTICK_DEADZONE) * MAX_LINEAR_LAT_VEL
    a[2] = apply_deadzone(-ctrl.right_axis[0], JOYSTICK_DEADZONE) * MAX_YAW_RATE
    a[3:20] = qpos_29[UPPER_BODY_INDICES]
    a[20] = float(ctrl.left_trigger)
    a[21] = float(ctrl.right_trigger)
    return a


# ── Background GMR retargeter ───────────────────────────────────────────
class RetargetWorker:
    def __init__(self, human_height: float):
        self._retargeter = GMRRetargeter(human_height=human_height)
        self._lock = threading.Lock()
        self._latest_qpos = np.zeros(29, dtype=np.float32)
        self._latest_smpl: np.ndarray | None = None
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="gmr-worker")
        self._thread.start()

    def submit(self, body_joints_pose: np.ndarray):
        with self._cv:
            self._latest_smpl = body_joints_pose
            self._cv.notify()

    def latest_qpos(self) -> np.ndarray:
        with self._lock:
            return self._latest_qpos.copy()

    def _run(self):
        while not self._stop.is_set():
            with self._cv:
                while self._latest_smpl is None and not self._stop.is_set():
                    self._cv.wait(timeout=0.1)
                smpl = self._latest_smpl
                self._latest_smpl = None
            if smpl is None:
                continue
            try:
                res = self._retargeter.retarget(smpl)
                with self._lock:
                    self._latest_qpos = res.joint_pos.astype(np.float32)
            except Exception as e:
                print(f"[gmr] {e}")

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()


# ── H.264 encoder + Pico relay ──────────────────────────────────────────
class H264Encoder:
    def __init__(self, w: int, h: int, fps: int, bitrate_kbps: int = 4000,
                 keyint: int | None = None):
        if keyint is None:
            keyint = max(2, int(round(fps)))
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-color_range", "tv",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-x264-params",
            f"keyint={keyint}:min-keyint={keyint}:scenecut=0",
            "-b:v", f"{bitrate_kbps}k",
            "-f", "h264",
            "pipe:1",
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0,
        )
        self.relay_queue: "queue.Queue | None" = None
        self._buf = b""
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name="h264-reader")
        self._reader.start()

    def attach_queue(self, q: "queue.Queue"):
        self.relay_queue = q

    def push_rgb(self, rgb_uint8: np.ndarray):
        try:
            self.proc.stdin.write(rgb_uint8.tobytes())
        except BrokenPipeError:
            pass

    def _read_loop(self):
        SC = b"\x00\x00\x00\x01"
        while True:
            data = self.proc.stdout.read(4096)
            if not data:
                if self.relay_queue is not None and self._buf:
                    try:
                        self.relay_queue.put_nowait(self._buf)
                    except queue.Full:
                        pass
                return
            self._buf += data
            while True:
                i = self._buf.find(SC)
                if i < 0:
                    break
                j = self._buf.find(SC, i + 4)
                if j < 0:
                    self._buf = self._buf[i:]
                    break
                nal = self._buf[i:j]
                self._buf = self._buf[j:]
                if self.relay_queue is not None:
                    try:
                        self.relay_queue.put_nowait(nal)
                    except queue.Full:
                        try:
                            self.relay_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.relay_queue.put_nowait(nal)
                        except queue.Full:
                            pass

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass


# ── Model loading via Hydra (mirror notebooks/g1-sampling.ipynb) ────────
def load_models(config_path: str, config_name: str, dynamics_ckpt: str,
                tokenizer_ckpt: str, max_num_forward_steps: int, device):
    cfg_dir = Path(config_path).resolve()
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=config_name)
    cfg.dynamics_ckpt = dynamics_ckpt
    cfg.tokenizer_ckpt = tokenizer_ckpt
    denoiser = load_denoiser(cfg, device, max_num_forward_steps=max_num_forward_steps)
    tokenizer = load_tokenizer(cfg, device, max_num_forward_steps=max_num_forward_steps)
    return cfg, denoiser.eval().to(device), tokenizer.eval().to(device)


def seed_from_g1_dataset(data_dir: str, seed_len: int, n_act_cfg: int,
                         split: str, train_fraction: float, split_seed: int,
                         window_idx: int | None, device):
    dataset = G1ChunkDataset(
        data_dir=data_dir,
        window_size=seed_len,
        stride=seed_len,
        split=split,
        train_fraction=train_fraction,
        split_seed=split_seed,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No windows of size {seed_len} found in {data_dir}")
    idx = window_idx if window_idx is not None else int(torch.randint(len(dataset), (1,)).item())
    idx = idx % len(dataset)
    print(f"[g1-interactive] seeding from G1ChunkDataset[{idx}] (seed_len={seed_len})")
    batch = dataset[idx]
    imgs = batch["image"][:seed_len].to(device=device)[None]               # (1, T_ctx, 3, H, W)
    actions = batch["action"][:seed_len, :n_act_cfg].to(device=device)[None]  # (1, T_ctx, n_act)
    return imgs, actions


# ── Main ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── model ─────────────────────────────────────────────────────────
    p.add_argument("--config-path", default=str(_REPO / "scripts" / "config"),
                   help="Hydra config directory.")
    p.add_argument("--config-name", default="dynamics/g1-large.yaml",
                   help="Hydra config file (relative to --config-path).")
    p.add_argument("--dynamics-ckpt",
                   default="/home/mim-server/projects/rooholla/dreamerV4-UWM/checkpoints/dynamics/G1/86504.pt")
    p.add_argument("--tokenizer-ckpt",
                   default="/home/mim-server/projects/rooholla/dreamerV4-UWM/checkpoints/tokenizer/g1_reference.pt")
    # ── seeding ───────────────────────────────────────────────────────
    p.add_argument("--seed-data-dir",
                   default="/home/mim-server/robot/GR00T-WholeBodyControl/creo-g1-teleop/new_wm_h5_segmented_small/world_model",
                   help="G1ChunkDataset root (directory of chunk_*.h5).")
    p.add_argument("--seed-len", type=int, default=16,
                   help="Number of context frames to encode + prime the cache with.")
    p.add_argument("--seed-split", default="train", choices=("train", "test"))
    p.add_argument("--seed-train-fraction", type=float, default=0.9)
    p.add_argument("--seed-split-seed", type=int, default=123)
    p.add_argument("--seed-window-idx", type=int, default=None,
                   help="If set, pick this specific window index; else random.")
    # ── sampling ──────────────────────────────────────────────────────
    p.add_argument("--num-steps", type=int, default=4,
                   help="Flow-matching Euler steps per generated frame "
                        "(passed as denoising_step_count). Must be a power of two.")
    p.add_argument("--context-cond-tau", type=float, default=0.99,
                   help="Conditioning noise level for cached context frames.")
    p.add_argument("--max-session-frames", type=int, default=512,
                   help="Hard cap on the rollout length (KV cache size). The "
                        "session stops once this many predicted frames have "
                        "been emitted. Bumping this grows VRAM linearly.")
    # ── VR + retarget ─────────────────────────────────────────────────
    p.add_argument("--vr-host", default="localhost")
    p.add_argument("--vr-port", type=int, default=5580)
    p.add_argument("--human-height", type=float, default=1.5)
    # ── streaming ─────────────────────────────────────────────────────
    p.add_argument("--video-port", type=int, default=13579)
    p.add_argument("--target-fps", type=float, default=10.0)
    p.add_argument("--bitrate-kbps", type=int, default=4000)
    p.add_argument("--out-resolution", default="1440x720",
                   help="Side-by-side stereo geometry the headset expects.")
    # ── misc ──────────────────────────────────────────────────────────
    p.add_argument("--device", default="cuda")
    p.add_argument("--debug-dir", default=None,
                   help="If set, save the first --debug-frames PNGs here.")
    p.add_argument("--debug-frames", type=int, default=10)
    args = p.parse_args()

    device = torch.device(args.device)
    cfg, denoiser, tokenizer = load_models(
        args.config_path, args.config_name,
        args.dynamics_ckpt, args.tokenizer_ckpt,
        max_num_forward_steps=args.max_session_frames + args.seed_len + 8,
        device=device,
    )
    n_act_cfg = int(cfg.denoiser.n_actions)
    if n_act_cfg != ACTION_DIM:
        print(f"[g1-interactive] WARNING: cfg.denoiser.n_actions={n_act_cfg} "
              f"!= ACTION_DIM={ACTION_DIM}; action vector will be cropped.")

    imgs_ctx, actions_ctx = seed_from_g1_dataset(
        args.seed_data_dir, args.seed_len, n_act_cfg,
        args.seed_split, args.seed_train_fraction, args.seed_split_seed,
        args.seed_window_idx, device,
    )

    ar = AutoRegressiveForwardDynamics(
        denoiser=denoiser,
        tokenizer=tokenizer,
        mode='wm',
        context_length=args.max_session_frames + args.seed_len + 8,
        max_forward_steps=args.max_session_frames + args.seed_len + 8,
        context_cond_tau=args.context_cond_tau,
        denoising_step_count=args.num_steps,
        device=device,
        dtype=torch.float32,
    )

    print("[g1-interactive] priming caches with seed context ...")
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        ar.reset(imgs_ctx, actions_ctx)

    # ── VR + retargeter ───────────────────────────────────────────────
    print(f"[g1-interactive] connecting to pico-vr-bridge {args.vr_host}:{args.vr_port}")
    vr = PicoVR(host=args.vr_host, port=args.vr_port)
    vr.start()
    retargeter = RetargetWorker(human_height=args.human_height)

    # ── output stream ─────────────────────────────────────────────────
    out_w, out_h = (int(x) for x in args.out_resolution.lower().split("x"))
    if out_w % 2:
        raise ValueError("--out-resolution width must be even (stereo split)")
    eye_w = out_w // 2

    print(f"[g1-interactive] PicoVideoRelay listening on TCP :{args.video_port}")
    relay_q: "queue.Queue[bytes]" = queue.Queue(maxsize=64)
    relay = PicoVideoRelay(command_port=args.video_port, relay_queue=relay_q)
    relay.start()
    encoder = H264Encoder(out_w, out_h, int(round(args.target_fps)),
                          bitrate_kbps=args.bitrate_kbps)
    encoder.attach_queue(relay_q)

    debug_dir = None
    if args.debug_dir is not None:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image as _Image  # noqa: F401
        except ImportError:
            print("[g1-interactive] PIL missing; --debug-dir ignored")
            debug_dir = None

    target_dt = 1.0 / args.target_fps
    print(f"[g1-interactive] streaming at {args.target_fps:.1f} Hz; "
          f"point Pico app at this PC's IP, port {args.video_port}. Ctrl-C to stop.")

    next_t = time.monotonic()
    step = 0
    t_loop = time.monotonic()
    try:
        while step < args.max_session_frames:
            now = time.monotonic()
            if next_t > now:
                time.sleep(next_t - now)
            next_t += target_dt
            if time.monotonic() > next_t + target_dt:
                next_t = time.monotonic() + target_dt

            ctrl = vr.get_controller_state()
            smpl = vr.get_smpl_state()
            if smpl is not None:
                retargeter.submit(smpl.body_joints_pose)
            qpos = retargeter.latest_qpos()
            a = build_action(ctrl, qpos)[:n_act_cfg]
            a_t = torch.from_numpy(a).to(device).view(1, n_act_cfg)

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                img_t = ar.step(actions_t=a_t)
                if isinstance(img_t, tuple):  # reward-head checkpoints
                    img_t = img_t[0]

            recon_f = img_t[0].to(torch.float32).cpu().numpy()  # (C, H, W)
            img = np.clip(recon_f, 0.0, 1.0).transpose(1, 2, 0)
            img = np.ascontiguousarray((img * 255).astype(np.uint8))

            # Stereo upscale to headset geometry
            eye = cv2.resize(img, (eye_w, out_h), interpolation=cv2.INTER_LINEAR)
            stereo = np.concatenate([eye, eye], axis=1)
            encoder.push_rgb(stereo)

            if debug_dir is not None and step < args.debug_frames:
                from PIL import Image as _Image
                _Image.fromarray(img).save(debug_dir / f"frame_{step:04d}.png")

            step += 1
            if step % 30 == 0:
                fps = 30.0 / (time.monotonic() - t_loop)
                t_loop = time.monotonic()
                print(f"[g1-interactive] step={step} fps={fps:.1f} "
                      f"qsz={relay_q.qsize()}")
        print(f"[g1-interactive] hit --max-session-frames={args.max_session_frames}; stopping")
    except KeyboardInterrupt:
        print("\n[g1-interactive] shutting down")
    finally:
        encoder.close()
        relay.stop()
        vr.stop()
        retargeter.stop()


if __name__ == "__main__":
    main()
