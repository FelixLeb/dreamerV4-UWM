"""Interactive WM-mode rollout for pushT with a live reward window.

Drives the UWM denoiser in world-model mode over a primed pushT context.
At each frame the right joystick supplies a 2-D action; the AR sampler
returns (image, rewards) where `rewards` is the reward head's L-step MTP
prediction read off the agent token at the *last* denoising step. KV-caching
keeps the per-frame cost to a single denoising pass + one cache commit.

Requires a checkpoint trained with `denoiser.train_reward_model=true`
(e.g. via scripts/train_reward_only.py + scripts/config/dynamics/pushT-reward-only.yaml).

Usage:
    python scripts/interactive-play-pushT.py \\
        --config-path config --config-name dynamics/pushT-reward-only \\
        dynamics_ckpt=/path/to/ckpt.pt \\
        tokenizer_ckpt=/path/to/tokenizer.pt \\
        dataset.data_dir=/path/to/sharded
"""
import contextlib
import time

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn.functional import interpolate
from tqdm import tqdm

from dreamerv4uwm.datasets import ShardedHDF5Dataset
from dreamerv4uwm.models.utils import load_denoiser, load_tokenizer
from dreamerv4uwm.sampling import AutoRegressiveForwardDynamics
from dreamerv4uwm.utils.joy import XBoxController


# -------------------------------------------------------------------------
# Configurable constants
# -------------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
JOYSTICK_ID = 0
ACTION_SCALE = 0.1
NUM_INIT_FRAMES = 8
NUM_FORWARD_STEPS = 1000
CONTEXT_LEN = 64
DENOISING_STEPS = 4
DATASET_WINDOW_INDEX = 500
TARGET_FPS = 5.0  # one frame ≈ 1/TARGET_FPS seconds (pushT default dt=0.2)

# Reward-panel display range (real-space, post symlog inverse).
REWARD_DISPLAY_MIN = -1.0
REWARD_DISPLAY_MAX = 25.0


# -------------------------------------------------------------------------
# Initial latent extraction from a pushT episode
# -------------------------------------------------------------------------
def get_initial_frames(data_dir, resolution, device, window_size=CONTEXT_LEN, idx=DATASET_WINDOW_INDEX):
    dataset = ShardedHDF5Dataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=1,
        split='train',
        train_fraction=0.9,
        split_seed=123,
        shuffle_windows=False,
    )
    idx = min(idx, len(dataset) - 1)
    batch = dataset[idx]
    imgs = batch["image"]      # (T, C, H, W) in [0, 1]
    actions = batch["action"]  # (T, n_act)
    imgs = interpolate(imgs, resolution).to(device=device)[None]   # (1, T, C, H, W)
    actions = actions.to(device=device)[None]                      # (1, T, n_act)
    return imgs, actions


# -------------------------------------------------------------------------
# Reward visualization
# -------------------------------------------------------------------------
def render_reward_panel(rewards_np, mtp_length, panel_size=(420, 720)):
    """Render an L-bar chart of the MTP reward predictions.

    rewards_np: (L,) array of decoded scalar rewards for r_{t+0..t+L-1}.
    """
    H, W = panel_size
    img = np.full((H, W, 3), 32, dtype=np.uint8)

    # Title.
    cv2.putText(img, "Reward MTP head (last denoise step)", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, f"r(t+0..t+{mtp_length-1})", (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    margin_x, top, bottom = 50, 80, 40
    plot_W = W - 2 * margin_x
    plot_H = H - top - bottom

    # Axis bounds.
    lo, hi = REWARD_DISPLAY_MIN, REWARD_DISPLAY_MAX
    span = hi - lo
    if span <= 0:
        span = 1.0

    def y_for(v):
        v_clip = float(np.clip(v, lo, hi))
        frac = (v_clip - lo) / span
        return int(top + plot_H - frac * plot_H)

    # Zero-line if 0 in range.
    if lo <= 0.0 <= hi:
        y0 = y_for(0.0)
        cv2.line(img, (margin_x, y0), (W - margin_x, y0), (90, 90, 90), 1)
        cv2.putText(img, "0", (margin_x - 22, y0 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

    # Y-axis ticks at lo/hi.
    cv2.putText(img, f"{hi:.0f}", (margin_x - 28, top + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(img, f"{lo:.0f}", (margin_x - 28, top + plot_H + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

    # Bars.
    L = len(rewards_np)
    bar_slot = plot_W / max(L, 1)
    bar_w = max(int(bar_slot * 0.7), 4)
    for i, r in enumerate(rewards_np):
        cx = int(margin_x + (i + 0.5) * bar_slot)
        x0 = cx - bar_w // 2
        x1 = cx + bar_w // 2
        baseline_y = y_for(0.0) if lo <= 0.0 <= hi else top + plot_H
        bar_y = y_for(r)
        y_top, y_bot = (bar_y, baseline_y) if r >= 0 else (baseline_y, bar_y)
        color = (80, 200, 110) if r >= 0 else (90, 90, 220)
        cv2.rectangle(img, (x0, y_top), (x1, y_bot), color, -1)
        cv2.putText(img, f"{r:+.2f}", (x0 - 4, top + plot_H + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"+{i}", (x0 + 2, top + plot_H + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

    return img


# -------------------------------------------------------------------------
# Joystick control loop
# -------------------------------------------------------------------------
@hydra.main(config_path="config", config_name="dynamics/pushT-reward-only", version_base=None)
def main(cfg: DictConfig):
    assert cfg.denoiser.train_reward_model, (
        "interactive-play-pushT requires a denoiser with train_reward_model=true; "
        "use scripts/config/dynamics/pushT-reward-only.yaml or override "
        "denoiser.train_reward_model=true."
    )

    print("Loading models")
    denoiser = load_denoiser(cfg, DEVICE, max_num_forward_steps=NUM_FORWARD_STEPS).eval()
    tokenizer = load_tokenizer(cfg, DEVICE, max_num_forward_steps=NUM_FORWARD_STEPS).eval()

    print("Extracting initial frames from dataset...")
    imgs, actions = get_initial_frames(
        cfg.dataset.data_dir,
        tuple(cfg.dataset.resolution),
        device=DEVICE,
    )

    world = AutoRegressiveForwardDynamics(
        denoiser, tokenizer,
        mode='wm',
        context_length=CONTEXT_LEN,
        denoising_step_count=DENOISING_STEPS,
        max_forward_steps=NUM_FORWARD_STEPS,
        device=DEVICE,
        dtype=DTYPE,
    )

    use_cuda = (DEVICE.type == "cuda")
    ctx = torch.autocast("cuda", dtype=DTYPE) if use_cuda else contextlib.nullcontext()
    with ctx:
        world.reset(imgs[:, :NUM_INIT_FRAMES], actions[:, :NUM_INIT_FRAMES])

    action_t = torch.zeros(1, 1, actions.shape[-1]).to(DEVICE, dtype=DTYPE)
    joy = XBoxController(JOYSTICK_ID)

    mtp_length = denoiser.model.cfg.mtp_length
    period = 1.0 / TARGET_FPS

    print("Starting joystick-controlled UWM rollout (mode=wm, with reward head)...")
    print("Right stick = action; press Q in either window to quit.")

    n_steps = NUM_FORWARD_STEPS - cfg.denoiser.context_length
    for _ in tqdm(range(n_steps)):
        tic = time.time()
        states = joy.getStates()
        cmd_right = states["right_joy"]
        action_t[..., 0] = -cmd_right[1] * ACTION_SCALE
        action_t[..., 1] = cmd_right[0] * ACTION_SCALE

        with ctx:
            img, rewards = world.step(action_t)

        img_np = (img[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_np = cv2.resize(img_np, (720, 720))
        cv2.imshow("uwm pushT (wm mode)", img_np)

        rewards_np = rewards[0].float().cpu().numpy()
        cv2.imshow("reward (MTP)", render_reward_panel(rewards_np, mtp_length))

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        # pace to TARGET_FPS without blocking the joystick polling thread
        while time.time() - tic < period:
            time.sleep(0.001)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
