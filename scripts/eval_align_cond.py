"""Evaluate AdaLN-Zero class conditioning of an aligned UWM dynamics model.

Rolls out the KV-cached `AutoRegressiveForwardDynamics` sampler under each
conditioning class and compares them against the *unconditioned* model
(`cond_class=None`, which — with a frozen backbone — reproduces the base
pretrained model). For a random subset of windows drawn from BOTH the play and
demo datasets, it writes:

  * action plots   — every class's generated action trajectory overlaid on the
                     ground-truth action, one subplot per action dim;
  * snapshot grids — evenly spaced frames, one row per source (GT + each class);
  * rollout videos — GT tiled above every class's rollout, side by side;
  * metrics        — per-class pixel PSNR / latent MSE / action RMSE, plus a
                     summary.json aggregating across samples and sources.

The model architecture + tokenizer are taken from the `config.yaml` that the
training run saved next to the checkpoint, so you normally only pass the
checkpoint and an output dir.

Usage (run once per checkpoint):

    python scripts/eval_align_cond.py \
        --dynamics-ckpt /scratch/.../checkpoints/align/pushT/adaln/75000.pt \
        --output-dir    /scratch/.../eval/adaln \
        --mode policy --num-samples 4 --num-context 8 --num-predict 40 \
        --num-diffusion-steps 8

    python scripts/eval_align_cond.py \
        --dynamics-ckpt /scratch/.../checkpoints/align/pushT/adaln-blockcausal/75000.pt \
        --output-dir    /scratch/.../eval/adaln-blockcausal

Notes:
  * `--mode policy` (default) generates BOTH actions and frames autoregressively,
    so the action plots and the video both reflect the *conditioned policy*.
    `--mode wm` feeds ground-truth actions and only generates frames (isolates
    dynamics; action plots then just echo the GT actions and are skipped).
  * Class names map to the training indices: null=0, play=1, demo=2; the special
    name `none` runs the sampler with cond_class=None (unconditioned / base).
  * Single-process, one GPU; no torch.distributed.
"""

import argparse
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


# Class-name → training index. `none` is the special unconditioned pass.
CLASS_TO_IDX = {"none": None, "null": 0, "play": 1, "demo": 2}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dynamics-ckpt", required=True,
                   help="Path to the aligned checkpoint (e.g. .../adaln/75000.pt).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config", default=None,
                   help="Path to a config.yaml. Default: config.yaml next to the checkpoint.")
    p.add_argument("--tokenizer-ckpt", default=None,
                   help="Override tokenizer ckpt. Default: cfg.tokenizer_ckpt from the config.")
    p.add_argument("--play-data-dir", default=None,
                   help="Override play data dir. Default: cfg.dataset.play_data_dir.")
    p.add_argument("--demo-data-dir", default=None,
                   help="Override demo data dir. Default: cfg.dataset.demo_data_dir.")
    p.add_argument("--classes", default="none,null,play,demo",
                   help="Comma-separated subset of {none,null,play,demo}.")
    p.add_argument("--mode", default="policy", choices=["policy", "wm"],
                   help="AR sampler mode. policy=generate actions+frames; wm=GT actions, frames only.")
    p.add_argument("--num-samples", type=int, default=4,
                   help="Random windows PER dataset source (play and demo each).")
    p.add_argument("--num-context", type=int, default=8)
    p.add_argument("--num-predict", type=int, default=40)
    p.add_argument("--num-diffusion-steps", type=int, default=8,
                   help="Euler denoising steps per frame (power of two).")
    p.add_argument("--context-cond-tau", type=float, default=0.9)
    p.add_argument("--sources", default="play,demo",
                   help="Comma-separated subset of {play,demo}.")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--train-fraction", type=float, default=0.9)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--num-snapshots", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(args):
    cfg_path = args.config or os.path.join(os.path.dirname(args.dynamics_ckpt), "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"No config found at {cfg_path}. Pass --config explicitly (the training "
            f"run saves config.yaml next to its checkpoints)."
        )
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    print(f"[eval] loaded config: {cfg_path}")
    return cfg


def load_models(cfg, args, device):
    from dreamerv4uwm.models.utils import load_denoiser, load_tokenizer

    if not bool(cfg.denoiser.get("cond_adaln", False)):
        warnings.warn(
            "config has denoiser.cond_adaln=False — the model is NOT class-conditioned, "
            "so all class passes will be identical to `none`. Did you point at a "
            "conditioned checkpoint/config?"
        )
    cfg.dynamics_ckpt = args.dynamics_ckpt
    if args.tokenizer_ckpt:
        cfg.tokenizer_ckpt = args.tokenizer_ckpt
    assert cfg.get("tokenizer_ckpt"), (
        "tokenizer_ckpt is not in the config; pass --tokenizer-ckpt explicitly."
    )

    # Size the temporal RoPE tables to the full rollout length so long rollouts
    # (num_context + num_predict can exceed max_sequence_length) don't index past
    # the table. RoPE buffers are non-persistent, so enlarging them does NOT
    # affect checkpoint loading; and since the KV caches roll over a
    # context_length window, the *relative* positions stay in-distribution.
    window = int(args.num_context + args.num_predict)
    max_steps = max(int(cfg.denoiser.max_sequence_length), window)
    print(f"[eval] building models with temporal capacity {max_steps} frames "
          f"(max_sequence_length={int(cfg.denoiser.max_sequence_length)}, rollout window={window}).")
    # strict=False tolerates harmless extras; we explicitly check the conditioning
    # weights actually loaded below so a base ckpt can't silently masquerade.
    denoiser = load_denoiser(cfg, device=device, max_num_forward_steps=max_steps, strict=False)
    tokenizer = load_tokenizer(cfg, device=device, max_num_forward_steps=max_steps)
    denoiser.eval(); tokenizer.eval()
    for m in (denoiser, tokenizer):
        for pp in m.parameters():
            pp.requires_grad_(False)

    # Guardrail: confirm the conditioning weights are non-trivial (i.e. the
    # checkpoint actually contains trained AdaLN params, not a zero-init base).
    if bool(cfg.denoiser.get("cond_adaln", False)):
        adaln_abs = 0.0
        for n, prm in denoiser.named_parameters():
            if "adaln" in n:
                adaln_abs += prm.abs().sum().item()
        if adaln_abs == 0.0:
            warnings.warn(
                "All AdaLN params are exactly zero — conditioning will be a no-op "
                "(every class == base). Is this a freshly-initialized / base checkpoint?"
            )
        else:
            print(f"[eval] AdaLN conditioning weights present (sum|w|={adaln_abs:.3e}).")
    return cfg, denoiser, tokenizer


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataset(cfg, data_dir, window_size, args):
    from dreamerv4uwm.datasets import ShardedHDF5Dataset, G1ChunkDataset
    kind = str(cfg.dataset.get("kind", "sharded_hdf5"))
    common = dict(
        data_dir=data_dir, window_size=window_size, stride=1, split=args.split,
        train_fraction=args.train_fraction, split_seed=args.split_seed,
        shuffle_windows=False,
    )
    if kind == "g1_chunked":
        return G1ChunkDataset(**common)
    if kind == "sharded_hdf5":
        return ShardedHDF5Dataset(**common)
    raise ValueError(f"unsupported dataset kind for eval: {kind}")


def sample_windows(dataset, num_samples, n_actions, rng):
    """Random subset of `num_samples` windows. Returns
    images (B,T,C,H,W) float[0,1], actions (B,T,n_actions) float."""
    n_total = len(dataset)
    if n_total == 0:
        raise RuntimeError("dataset produced 0 windows (window_size too large?)")
    n = min(num_samples, n_total)
    idxs = rng.choice(n_total, size=n, replace=False)
    imgs, acts = [], []
    for i in idxs:
        s = dataset[int(i)]
        imgs.append(s["image"])
        acts.append(s["action"][..., :n_actions])
    return torch.stack(imgs, 0), torch.stack(acts, 0), [int(i) for i in idxs]


def encode_chunked(tokenizer, images, max_T):
    """Encode (B,T,C,H,W) -> (B,T,N_lat,D_lat) in temporal chunks of <= max_T.

    The video tokenizer's temporal RoPE table is sized to its max sequence
    length, so a one-shot encode of T > max_T overflows it. Chunking keeps
    each encode within range; boundary effects are negligible for the GT
    reference latents used only by the latent-MSE metric.
    """
    T = images.shape[1]
    if T <= max_T:
        return tokenizer.encode(images)
    parts = [tokenizer.encode(images[:, s:s + max_T]) for s in range(0, T, max_T)]
    return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Visualization / IO (adapted from scripts/eval_block_causal.py)
# ---------------------------------------------------------------------------

def to_uint8_thwc(frames):
    """(T,C,H,W) float[0,1] -> (T,H,W,C) uint8 numpy."""
    x = frames.detach().to(torch.float32).clamp(0, 1).cpu().numpy()
    x = np.transpose(x, (0, 2, 3, 1))
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_video(frames_thwc, path, fps):
    """Write (T,H,W,C) uint8. mp4 via ffmpeg if present, else GIF via PIL."""
    import shutil
    import subprocess

    path = Path(path)
    frames = np.ascontiguousarray(frames_thwc)
    T, H, W, C = frames.shape
    if H % 2 or W % 2:
        Hp, Wp = H + (H % 2), W + (W % 2)
        padded = np.zeros((T, Hp, Wp, C), dtype=np.uint8)
        padded[:, :H, :W] = frames
        frames, H, W = padded, Hp, Wp

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        mp4 = path.with_suffix(".mp4")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
               "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(mp4)]
        try:
            pr = subprocess.run(cmd, input=frames.tobytes(),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if pr.returncode == 0 and mp4.exists():
                return str(mp4)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"ffmpeg mp4 write failed ({e}); falling back to gif")

    from PIL import Image
    gif = path.with_suffix(".gif")
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1)), loop=0)
    return str(gif)


def _load_font(size):
    """A scalable TrueType font at `size` px (DejaVuSans, shipped with
    matplotlib), falling back to PIL's bitmap default."""
    import os
    from PIL import ImageFont
    candidates = []
    try:
        import matplotlib
        candidates.append(os.path.join(os.path.dirname(matplotlib.__file__),
                                       "mpl-data", "fonts", "ttf", "DejaVuSans.ttf"))
    except Exception:
        pass
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            if os.path.exists(c):
                return ImageFont.truetype(c, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size)  # PIL >= 10
    except Exception:
        return ImageFont.load_default()


def tiled_video(panels_tchw, labels, path, fps, sep=4):
    """Tile several (T,C,H,W) rollouts (GT + each class) side by side
    HORIZONTALLY with thin separators and a single readable label header row on
    top, then save as one video. Landscape layout fits a screen better than a
    tall vertical stack."""
    T = min(p.shape[0] for p in panels_tchw)
    ups = [to_uint8_thwc(p[:T]) for p in panels_tchw]   # each (T,H,W,C)
    H, W, C = ups[0].shape[1:]
    vbar = np.zeros((T, H, sep, C), dtype=np.uint8); vbar[..., 0] = 255  # red column sep
    pieces = []
    for j, u in enumerate(ups):
        pieces.append(u)
        if j != len(ups) - 1:
            pieces.append(vbar)
    body = np.concatenate(pieces, axis=2)               # (T, H, total_W, C)
    # Header sized to the frame so labels are legible (font ~ H/10, min 14px).
    font_size = max(14, H // 10)
    header = _header_strip(labels, W, sep, body.shape[2], C, T, font_size)
    out = np.concatenate([header, body], axis=1)        # (T, strip_h+H, total_W, C)
    return save_video(out, path, fps)


def _header_strip(labels, panel_w, sep, total_w, C, T, font_size):
    """A (T, strip_h, total_w, C) black header with each label drawn (in a
    readable TrueType font) centered over its column."""
    from PIL import Image, ImageDraw
    font = _load_font(font_size)
    strip_h = font_size + 8
    img = Image.new("RGB", (total_w, strip_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for j, lab in enumerate(labels):
        x0 = j * (panel_w + sep)
        try:
            tw = draw.textlength(lab, font=font)
        except Exception:  # older PIL
            tw = 0.6 * font_size * len(lab)
        draw.text((x0 + max(2, int((panel_w - tw) / 2)), 3), lab,
                  fill=(255, 255, 0), font=font)  # yellow = high contrast on frames
    arr = np.array(img)[..., :C]                        # (strip_h, total_w, C)
    return np.broadcast_to(arr[None], (T, strip_h, total_w, C)).copy()


def save_snapshot_grid(frames_by_label, path, n_snap):
    """frames_by_label: list of (label, (T,C,H,W)). One row per label, columns =
    evenly spaced frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [l for l, _ in frames_by_label]
    T = min(f.shape[0] for _, f in frames_by_label)
    n_snap = min(n_snap, T)
    idx = np.linspace(0, T - 1, n_snap, dtype=int)
    nrows = len(frames_by_label)
    fig, axes = plt.subplots(nrows, n_snap, figsize=(2.0 * n_snap, 2.0 * nrows),
                             squeeze=False)
    for r, (label, f) in enumerate(frames_by_label):
        u = to_uint8_thwc(f[idx])
        for c, t in enumerate(idx):
            axes[r][c].imshow(u[c]); axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(f"t={t}", fontsize=8)
        axes[r][0].set_ylabel(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_action_plot_multi(series_by_label, gt_act, path):
    """series_by_label: list of (label, (T,A)) generated-action arrays.
    gt_act: (T,A). One subplot per action dim; GT bold, each class overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = gt_act.shape[1]
    ncols = min(4, A)
    nrows = math.ceil(A / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.3 * nrows),
                             squeeze=False)
    Tg = gt_act.shape[0]
    for a in range(A):
        ax = axes[a // ncols][a % ncols]
        ax.plot(np.arange(Tg), gt_act[:, a], label="GT", lw=2.0, color="black")
        for label, ser in series_by_label:
            ax.plot(np.arange(ser.shape[0]), ser[:, a], lw=1.3, ls="--", label=label)
        ax.set_title(f"act[{a}]", fontsize=8); ax.tick_params(labelsize=7)
    for a in range(A, nrows * ncols):
        axes[a // ncols][a % ncols].axis("off")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_frame_metrics(pred_img, gt_img, pred_z, gt_z, pred_act, gt_act):
    pf = lambda x: x.detach().to(torch.float32).cpu()
    pred_img, gt_img = pf(pred_img), pf(gt_img)
    pix_mse = (pred_img - gt_img).pow(2).mean(dim=(1, 2, 3)).numpy()
    psnr = 10.0 * np.log10(1.0 / np.clip(pix_mse, 1e-10, None))
    out = dict(pixel_mse=pix_mse, psnr=psnr)
    if pred_z is not None and gt_z is not None:
        out["latent_mse"] = (pf(pred_z) - pf(gt_z)).pow(2).mean(dim=(1, 2)).numpy()
    if pred_act is not None and gt_act is not None:
        out["action_rmse"] = (pf(pred_act) - pf(gt_act)).pow(2).mean(dim=1).sqrt().numpy()
    return out


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_class(denoiser, tokenizer, cfg, imgs_ctx, actions_ctx, gt_actions_future,
                  cond_idx, mode, n_pred, num_diffusion_steps, context_cond_tau,
                  device, dtype, autocast_dev):
    """Run an AR rollout under one conditioning class. Returns
    (pred_images (n_pred,C,H,W), pred_latents (n_pred,N,D), pred_actions (n_pred,A) or None)
    for a SINGLE sample (B is collapsed by the caller passing B=1 slices)."""
    from dreamerv4uwm.sampling import AutoRegressiveForwardDynamics

    ar = AutoRegressiveForwardDynamics(
        denoiser=denoiser, tokenizer=tokenizer, mode=mode,
        context_length=int(cfg.denoiser.context_length),
        context_cond_tau=context_cond_tau,
        denoising_step_count=num_diffusion_steps,
        device=device, dtype=dtype, cond_class=cond_idx,
    )
    frames, acts, lats = [], [], []
    with torch.no_grad(), torch.autocast(device_type=autocast_dev, dtype=dtype):
        ar.reset(imgs_ctx, actions_ctx)
        for t in range(n_pred):
            if mode == "wm":
                out = ar.step(actions_t=gt_actions_future[:, t])
                frame = out[0] if isinstance(out, tuple) else out
                act_t = None
            else:  # policy
                out = ar.step()
                frame, act_t = out[0], out[1]
            frames.append(frame.detach().to(torch.float32).cpu())
            lats.append(ar.current_z.detach().to(torch.float32).cpu())  # (B,1,N,D)
            if act_t is not None:
                acts.append(act_t.detach().to(torch.float32).cpu())
    pred_images = torch.stack(frames, dim=1)             # (B,n_pred,C,H,W)
    pred_latents = torch.cat(lats, dim=1)                # (B,n_pred,N,D)
    pred_actions = torch.stack(acts, dim=1) if acts else None  # (B,n_pred,A)
    return pred_images, pred_latents, pred_actions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    autocast_dev = "cuda" if "cuda" in str(device) else "cpu"
    dtype = torch.bfloat16 if autocast_dev == "cuda" else torch.float32

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    for c in classes:
        if c not in CLASS_TO_IDX:
            raise ValueError(f"unknown class '{c}'; choose from {list(CLASS_TO_IDX)}")
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_root / ".mplcache"))
    (out_root / ".mplcache").mkdir(parents=True, exist_ok=True)

    cfg = load_config(args)
    cfg, denoiser, tokenizer = load_models(cfg, args, device)
    n_actions = int(cfg.denoiser.n_actions)
    tok_max = int(cfg.denoiser.max_sequence_length)
    ctx_len = int(cfg.denoiser.context_length)
    T_ctx, n_pred = args.num_context, args.num_predict
    window_size = T_ctx + n_pred
    print(f"[eval] mode={args.mode} classes={classes} sources={sources} "
          f"n_actions={n_actions} window={window_size} "
          f"(trained max_sequence_length={tok_max}, context_length={ctx_len})")
    if n_pred > ctx_len:
        print(f"[eval]   note: num_predict ({n_pred}) > context_length ({ctx_len}); "
              f"the KV caches roll over a {ctx_len}-frame window, so this is a long "
              f"autoregressive rollout — relative positions stay in-distribution, but "
              f"autoregressive error can accumulate later in the rollout.")

    src_dirs = {
        "play": args.play_data_dir or cfg.dataset.get("play_data_dir"),
        "demo": args.demo_data_dir or cfg.dataset.get("demo_data_dir"),
    }

    summary = {
        "dynamics_ckpt": args.dynamics_ckpt, "mode": args.mode,
        "classes": classes, "sources": sources,
        "num_context": T_ctx, "num_predict": n_pred,
        "num_diffusion_steps": args.num_diffusion_steps,
        "num_samples_per_source": args.num_samples,
        "metrics": {},  # source -> class -> scalar dict
    }

    for source in sources:
        data_dir = src_dirs.get(source)
        if not data_dir:
            warnings.warn(f"no data dir for source '{source}'; skipping.")
            continue
        print(f"\n[eval] ===== source: {source} ({data_dir}) =====")
        dataset = build_dataset(cfg, data_dir, window_size, args)
        images, actions, idxs = sample_windows(dataset, args.num_samples, n_actions, rng)
        B = images.shape[0]
        images, actions = images.to(device), actions.to(device)
        print(f"[eval]   {B} random windows (idx={idxs}), image {tuple(images.shape)}")

        gt_future_images = images[:, T_ctx:T_ctx + n_pred].contiguous()
        gt_future_actions = actions[:, T_ctx:T_ctx + n_pred].contiguous()
        imgs_ctx = images[:, :T_ctx].contiguous()
        actions_ctx = actions[:, :T_ctx].contiguous()
        # GT future latents (for the latent-MSE metric only). Encoded in
        # temporal chunks <= the tokenizer's max sequence length, since the
        # video tokenizer's temporal RoPE is sized to that and a longer
        # one-shot encode overflows it (n_pred can exceed it for long rollouts).
        with torch.no_grad(), torch.autocast(device_type=autocast_dev, dtype=dtype):
            gt_future_latents = encode_chunked(tokenizer, gt_future_images.to(dtype), tok_max)

        # Roll out every class (batched over the B windows at once).
        per_class = {}  # class -> (pred_images, pred_latents, pred_actions)
        for c in classes:
            print(f"[eval]   rollout class={c} ...")
            per_class[c] = rollout_class(
                denoiser, tokenizer, cfg, imgs_ctx, actions_ctx, gt_future_actions,
                CLASS_TO_IDX[c], args.mode, n_pred, args.num_diffusion_steps,
                args.context_cond_tau, device, dtype, autocast_dev,
            )

        # Per-sample outputs + metrics.
        src_dir = out_root / source
        agg = {c: {} for c in classes}
        for i in range(B):
            s_dir = src_dir / f"sample_{i}_win{idxs[i]}"
            s_dir.mkdir(parents=True, exist_ok=True)

            # Snapshot grid: GT + each class.
            rows = [("GT", gt_future_images[i])]
            rows += [(c, per_class[c][0][i]) for c in classes]
            save_snapshot_grid(rows, s_dir / "snapshots.png", args.num_snapshots)

            # Tiled rollout video: GT on top, each class below.
            tiled_video([gt_future_images[i]] + [per_class[c][0][i] for c in classes],
                        ["GT"] + classes, s_dir / "rollout", args.fps)

            # Action plot (policy mode only — wm uses GT actions as input).
            if args.mode == "policy":
                series = [(c, per_class[c][2][i].numpy()) for c in classes
                          if per_class[c][2] is not None]
                if series:
                    save_action_plot_multi(
                        series, gt_future_actions[i].to(torch.float32).cpu().numpy(),
                        s_dir / "actions.png")

            # Metrics per class.
            for c in classes:
                pi, pz, pa = per_class[c]
                mt = per_frame_metrics(
                    pi[i], gt_future_images[i], pz[i], gt_future_latents[i],
                    (pa[i] if pa is not None else None), gt_future_actions[i],
                )
                for k, v in mt.items():
                    agg[c].setdefault(k, []).append(v)

        # Aggregate per-class scalars for this source.
        summary["metrics"][source] = {}
        for c in classes:
            scal = {}
            for k, lst in agg[c].items():
                arr = np.stack(lst, 0)  # (B, n_pred)
                scal[f"{k}_mean"] = float(np.mean(arr))
                scal[f"{k}_final"] = float(np.mean(arr[:, -1]))
            summary["metrics"][source][c] = scal
            np.savez(src_dir / f"metrics_{c}.npz",
                     **{k: np.stack(v, 0) for k, v in agg[c].items()})
            print(f"[eval]   {source}/{c}: "
                  + " ".join(f"{k}={v:.4f}" for k, v in scal.items()))

    _summary_bars(summary, classes, sources, out_root)
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] done. Results in {out_root}")


def _summary_bars(summary, classes, sources, out_root):
    """Grouped bar charts: per-source PSNR and action RMSE across classes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _metric_present(metric):
        return any(f"{metric}_mean" in summary["metrics"].get(s, {}).get(c, {})
                   for s in sources for c in classes)

    metrics = [m for m in ("psnr", "action_rmse", "latent_mse") if _metric_present(m)]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 3.8),
                             squeeze=False)
    x = np.arange(len(classes))
    width = 0.8 / max(1, len(sources))
    for mi, metric in enumerate(metrics):
        ax = axes[0][mi]
        for si, s in enumerate(sources):
            vals = [summary["metrics"].get(s, {}).get(c, {}).get(f"{metric}_mean", np.nan)
                    for c in classes]
            ax.bar(x + si * width, vals, width, label=s)
        ax.set_xticks(x + width * (len(sources) - 1) / 2)
        ax.set_xticklabels(classes)
        ax.set_title(f"mean {metric}"); ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "summary_by_class.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
