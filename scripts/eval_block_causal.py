"""Evaluate block-causal inference of a UWM dynamics model across block sizes.

Block-causal inference is exactly what `inference.hybrid_chunk.HybridChunkSampler`
does: a chunk of `m` frames is denoised with bidirectional attention *within* the
chunk and causal attention to the cached (clean) context, then the whole chunk is
committed to the KV cache as a context block. So a "block size" m maps to
`chunk_size = m, commit_per_chunk = m`. This script sweeps several block sizes,
runs a policy-mode rollout (state AND action are generated jointly — what the
block-causal sanity protocol trains), and writes for each block size:

  * rollout videos (ground-truth on top, prediction on bottom),
  * snapshot strips (evenly spaced frames, GT vs pred),
  * action plots (generated vs ground-truth action, per dimension),
  * per-frame metrics (pixel PSNR/MSE, latent MSE, action RMSE),

plus cross-block summary plots and a `summary.json`.

Usage:
    python scripts/eval_block_causal.py \
        --config-name dynamics/g1-large \
        --dynamics-ckpt /path/to/dynamics.pt \
        --tokenizer-ckpt /path/to/tokenizer.pt \
        --data-dir /path/to/data \
        --dataset-kind g1_chunked \
        --output-dir /path/to/out \
        --block-sizes 2,4,8,16 \
        --num-context 8 --num-predict 48 --num-diffusion-steps 8 --num-samples 4

Notes:
  * `--dataset-kind` is one of {g1_chunked, sharded_hdf5}.
  * The model config (`--config-name`) determines architecture / tokenizer
    defaults; pass the same config the checkpoint was trained with.
  * Runs single-process on one GPU; no torch.distributed needed.
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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="dynamics/g1-large",
                   help="Hydra config name under --config-path (e.g. dynamics/g1-large).")
    p.add_argument("--config-path",
                   default=os.path.join(os.path.dirname(__file__), "config"),
                   help="Directory holding the hydra configs.")
    p.add_argument("--dynamics-ckpt", required=True)
    p.add_argument("--tokenizer-ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--dataset-kind", default="g1_chunked",
                   choices=["g1_chunked", "sharded_hdf5"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--block-sizes", default="2,4,8,16",
                   help="Comma-separated block sizes to evaluate.")
    p.add_argument("--num-context", type=int, default=8,
                   help="Number of clean context frames seeded into the cache.")
    p.add_argument("--num-predict", type=int, default=48,
                   help="Number of frames to roll out.")
    p.add_argument("--num-diffusion-steps", type=int, default=8,
                   help="Euler denoising steps per chunk (block).")
    p.add_argument("--num-samples", type=int, default=4,
                   help="Number of eval windows (batch size).")
    p.add_argument("--action-noise-std", type=float, default=1.0)
    p.add_argument("--commit-noise-n", type=float, default=0.0,
                   help="Noise level added to committed K/V for long-rollout stability.")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--train-fraction", type=float, default=0.9)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--num-snapshots", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataset(kind, data_dir, window_size, split, train_fraction, split_seed):
    from dreamerv4uwm.datasets import ShardedHDF5Dataset, G1ChunkDataset
    common = dict(
        data_dir=data_dir, window_size=window_size, stride=1, split=split,
        train_fraction=train_fraction, split_seed=split_seed, shuffle_windows=False,
    )
    if kind == "g1_chunked":
        return G1ChunkDataset(**common)
    elif kind == "sharded_hdf5":
        return ShardedHDF5Dataset(**common)
    raise ValueError(f"unknown dataset kind: {kind}")


def collect_windows(dataset, num_samples, n_actions):
    """Grab the first `num_samples` windows. Returns
    images (B,T,C,H,W) float[0,1], actions (B,T,n_actions) float."""
    n = min(num_samples, len(dataset))
    if n == 0:
        raise RuntimeError("dataset produced 0 windows (window_size too large?)")
    imgs, acts = [], []
    for i in range(n):
        s = dataset[i]
        imgs.append(s["image"])
        acts.append(s["action"][..., :n_actions])
    return torch.stack(imgs, 0), torch.stack(acts, 0)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_models(args, device):
    from hydra import compose, initialize_config_dir
    from dreamerv4uwm.models.utils import load_denoiser, load_tokenizer

    cfg_name = args.config_name
    if cfg_name.endswith(".yaml"):
        cfg_name = cfg_name[:-5]
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath(args.config_path)):
        cfg = compose(config_name=cfg_name)

    OmegaConf.set_struct(cfg, False)
    cfg.dynamics_ckpt = args.dynamics_ckpt
    cfg.tokenizer_ckpt = args.tokenizer_ckpt
    cfg.dataset.data_dir = args.data_dir

    max_steps = int(cfg.denoiser.max_sequence_length)
    denoiser = load_denoiser(cfg, device=device, max_num_forward_steps=max_steps, strict=False)
    tokenizer = load_tokenizer(cfg, device=device, max_num_forward_steps=max_steps)
    denoiser.eval()
    tokenizer.eval()
    for m in (denoiser, tokenizer):
        for pp in m.parameters():
            pp.requires_grad_(False)
    return cfg, denoiser, tokenizer


# ---------------------------------------------------------------------------
# Visualization / IO helpers (defensive imports)
# ---------------------------------------------------------------------------

def to_uint8_thwc(frames):
    """(T,C,H,W) float[0,1] tensor -> (T,H,W,C) uint8 numpy."""
    x = frames.detach().to(torch.float32).clamp(0, 1).cpu().numpy()
    x = np.transpose(x, (0, 2, 3, 1))
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_video(frames_thwc, path, fps):
    """Write a (T,H,W,C) uint8 array. mp4 via the ffmpeg binary if present,
    else GIF via PIL. (The training container ships ffmpeg + PIL but not
    imageio/mediapy, so we avoid those deps.)"""
    import shutil
    import subprocess

    path = Path(path)
    frames = np.ascontiguousarray(frames_thwc)
    T, H, W, C = frames.shape
    # libx264 + yuv420p needs even dimensions; pad if necessary.
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
            p = subprocess.run(cmd, input=frames.tobytes(),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if p.returncode == 0 and mp4.exists():
                return str(mp4)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"ffmpeg mp4 write failed ({e}); falling back to gif")

    from PIL import Image
    gif = path.with_suffix(".gif")
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1)), loop=0)
    return str(gif)


def comparison_video(gt_tchw, pred_tchw, path, fps, sep=4):
    """Stack GT (top) over prediction (bottom) with a separator and save."""
    T = min(gt_tchw.shape[0], pred_tchw.shape[0])
    gt = to_uint8_thwc(gt_tchw[:T])
    pr = to_uint8_thwc(pred_tchw[:T])
    H, W, C = gt.shape[1:]
    bar = np.zeros((T, sep, W, C), dtype=np.uint8)
    bar[..., 0] = 255  # red separator
    stacked = np.concatenate([gt, bar, pr], axis=1)
    return save_video(stacked, path, fps)


def save_snapshot_strip(gt_tchw, pred_tchw, path, n_snap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = min(gt_tchw.shape[0], pred_tchw.shape[0])
    n_snap = min(n_snap, T)
    idx = np.linspace(0, T - 1, n_snap, dtype=int)
    gt = to_uint8_thwc(gt_tchw[idx])
    pr = to_uint8_thwc(pred_tchw[idx])
    fig, axes = plt.subplots(2, n_snap, figsize=(2.2 * n_snap, 4.6))
    if n_snap == 1:
        axes = axes.reshape(2, 1)
    for j, t in enumerate(idx):
        axes[0, j].imshow(gt[j]); axes[0, j].set_title(f"t={t}", fontsize=8)
        axes[1, j].imshow(pr[j])
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("GT", fontsize=10)
    axes[1, 0].set_ylabel("pred", fontsize=10)
    fig.suptitle(Path(path).parent.name, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_action_plot(pred_act, gt_act, path):
    """pred_act, gt_act: (T, A) numpy. One subplot per action dim."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, A = pred_act.shape
    ncols = min(4, A)
    nrows = math.ceil(A / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows),
                             squeeze=False)
    t = np.arange(T)
    for a in range(A):
        ax = axes[a // ncols][a % ncols]
        ax.plot(t, gt_act[:, a], label="GT", lw=1.5)
        ax.plot(t, pred_act[:, a], label="pred", lw=1.5, ls="--")
        ax.set_title(f"act[{a}]", fontsize=8)
        ax.tick_params(labelsize=7)
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
    """All inputs (T, ...) torch tensors. Returns dict of (T,) numpy arrays."""
    pf = lambda x: x.detach().to(torch.float32).cpu()
    pred_img, gt_img = pf(pred_img), pf(gt_img)
    pred_z, gt_z = pf(pred_z), pf(gt_z)
    pred_act, gt_act = pf(pred_act), pf(gt_act)

    pix_mse = (pred_img - gt_img).pow(2).mean(dim=(1, 2, 3)).numpy()      # (T,)
    psnr = 10.0 * np.log10(1.0 / np.clip(pix_mse, 1e-10, None))
    lat_mse = (pred_z - gt_z).pow(2).mean(dim=(1, 2)).numpy()             # (T,)
    act_rmse = (pred_act - gt_act).pow(2).mean(dim=1).sqrt().numpy()      # (T,)
    return dict(pixel_mse=pix_mse, psnr=psnr, latent_mse=lat_mse, action_rmse=act_rmse)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    autocast_dev = "cuda" if "cuda" in str(device) else "cpu"
    # bf16 on GPU (matches training); fp32 on CPU (robust for dry-runs).
    dtype = torch.bfloat16 if autocast_dev == "cuda" else torch.float32

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    # Matplotlib needs a writable config dir; the container has no writable $HOME.
    os.environ.setdefault("MPLCONFIGDIR", str(out_root / ".mplcache"))
    (out_root / ".mplcache").mkdir(parents=True, exist_ok=True)

    block_sizes = [int(b) for b in args.block_sizes.split(",") if b.strip()]
    T_ctx, n_pred = args.num_context, args.num_predict
    window_size = T_ctx + n_pred

    print(f"[eval] loading model from config {args.config_name}")
    cfg, denoiser, tokenizer = load_models(args, device)
    n_actions = int(cfg.denoiser.n_actions)
    context_length = int(cfg.denoiser.context_length)
    print(f"[eval] n_actions={n_actions} context_length={context_length} "
          f"horizon_aware={cfg.denoiser.get('horizon_aware')}")

    bad = [m for m in block_sizes if m >= context_length]
    if bad:
        raise ValueError(f"block sizes {bad} must be < context_length ({context_length})")
    if T_ctx > context_length - max(block_sizes):
        warnings.warn(
            f"num_context ({T_ctx}) is large relative to cache capacity "
            f"(context_length - block = {context_length - max(block_sizes)}); "
            f"context may be truncated for the largest block."
        )

    # --- Data ---
    print(f"[eval] loading {args.dataset_kind} windows (window={window_size}) "
          f"from {args.data_dir}")
    dataset = build_dataset(args.dataset_kind, args.data_dir, window_size,
                            args.split, args.train_fraction, args.split_seed)
    images, actions = collect_windows(dataset, args.num_samples, n_actions)
    B = images.shape[0]
    images = images.to(device)
    actions = actions.to(device)
    print(f"[eval] got B={B} windows, image shape {tuple(images.shape)}, "
          f"action shape {tuple(actions.shape)}")

    # --- Encode once ---
    with torch.no_grad(), torch.autocast(device_type=autocast_dev, dtype=dtype):
        latents = tokenizer.encode(images.to(dtype))                  # (B,T,N,D)
    ctx_latents = latents[:, :T_ctx].contiguous()
    ctx_actions = actions[:, :T_ctx].contiguous()
    gt_future_latents = latents[:, T_ctx:T_ctx + n_pred].contiguous()
    gt_future_images = images[:, T_ctx:T_ctx + n_pred].contiguous()
    gt_future_actions = actions[:, T_ctx:T_ctx + n_pred].contiguous()

    from dreamerv4uwm.inference.hybrid_chunk import HybridChunkSampler

    summary = {
        "config_name": args.config_name,
        "dynamics_ckpt": args.dynamics_ckpt,
        "tokenizer_ckpt": args.tokenizer_ckpt,
        "dataset_kind": args.dataset_kind,
        "num_context": T_ctx, "num_predict": n_pred,
        "num_diffusion_steps": args.num_diffusion_steps,
        "num_samples": B, "block_sizes": block_sizes,
        "per_block": {},
    }
    curves = {}  # block -> mean per-frame metric arrays

    for m in block_sizes:
        print(f"\n[eval] ===== block size m={m} =====")
        block_dir = out_root / f"block_{m}"
        block_dir.mkdir(parents=True, exist_ok=True)

        # HybridChunkSampler IS the block-causal sampler: chunk_size=m,
        # commit_per_chunk=m commits the whole block as context. The
        # horizon_aware warning's premise ("False => purely causal") is a
        # false positive for block-causal checkpoints, which ARE trained for
        # bidir-within-block; flip the flag *for the sampler view only* (the
        # model is already built) to silence the misleading message.
        real_ha = cfg.denoiser.get("horizon_aware", False)
        cfg.denoiser.horizon_aware = True
        sampler = HybridChunkSampler(
            denoiser=denoiser, cfg=cfg,
            chunk_size=m, commit_per_chunk=m,
            num_diffusion_steps=args.num_diffusion_steps,
            action_noise_std=args.action_noise_std,
            commit_noise_n=args.commit_noise_n,
            clean_commit_pass=True, device=device, dtype=dtype,
        )
        cfg.denoiser.horizon_aware = real_ha

        with torch.no_grad(), torch.autocast(device_type=autocast_dev, dtype=dtype):
            sampler.init_cache(batch_size=B)
            sampler.warm_up_cache(ctx_latents, ctx_actions)
            pred_z, pred_a = sampler.generate(n_frames=n_pred)         # (B,n_pred,..)
            pred_images = tokenizer.decode(pred_z)                     # (B,n_pred,C,H,W)

        # --- Per-sample outputs + metrics ---
        agg = {k: [] for k in ("pixel_mse", "psnr", "latent_mse", "action_rmse")}
        for i in range(B):
            s_dir = block_dir / f"sample_{i}"
            s_dir.mkdir(parents=True, exist_ok=True)
            comparison_video(gt_future_images[i], pred_images[i],
                             s_dir / "rollout", args.fps)
            save_snapshot_strip(gt_future_images[i], pred_images[i],
                                s_dir / "snapshots.png", args.num_snapshots)
            save_action_plot(pred_a[i].to(torch.float32).cpu().numpy(),
                             gt_future_actions[i].to(torch.float32).cpu().numpy(),
                             s_dir / "actions.png")
            mt = per_frame_metrics(pred_images[i], gt_future_images[i],
                                   pred_z[i], gt_future_latents[i],
                                   pred_a[i], gt_future_actions[i])
            for k in agg:
                agg[k].append(mt[k])

        # Mean per-frame curves across samples.
        mean_curves = {k: np.stack(agg[k], 0).mean(0) for k in agg}
        curves[m] = mean_curves
        np.savez(block_dir / "metrics.npz", **{k: np.stack(agg[k], 0) for k in agg})

        scalar = {
            "psnr_mean": float(np.mean(mean_curves["psnr"])),
            "pixel_mse_mean": float(np.mean(mean_curves["pixel_mse"])),
            "latent_mse_mean": float(np.mean(mean_curves["latent_mse"])),
            "action_rmse_mean": float(np.mean(mean_curves["action_rmse"])),
            "psnr_final": float(mean_curves["psnr"][-1]),
            "action_rmse_final": float(mean_curves["action_rmse"][-1]),
        }
        summary["per_block"][str(m)] = scalar
        with open(block_dir / "metrics.json", "w") as f:
            json.dump(scalar, f, indent=2)
        print(f"[eval] m={m}: PSNR {scalar['psnr_mean']:.2f} dB | "
              f"latent MSE {scalar['latent_mse_mean']:.4f} | "
              f"action RMSE {scalar['action_rmse_mean']:.4f}")

    # --- Cross-block summary plots ---
    _summary_plots(curves, out_root)
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] done. Results in {out_root}")


def _summary_plots(curves, out_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _overlay(metric, ylabel, fname, logy=False):
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in sorted(curves):
            y = curves[m][metric]
            ax.plot(np.arange(len(y)), y, label=f"block={m}", lw=1.6)
        ax.set_xlabel("rollout frame"); ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_root / fname, dpi=120); plt.close(fig)

    _overlay("psnr", "PSNR (dB)", "psnr_vs_time.png")
    _overlay("latent_mse", "latent MSE", "latent_mse_vs_time.png", logy=True)
    _overlay("action_rmse", "action RMSE", "action_rmse_vs_time.png")

    # Bar: mean PSNR & action RMSE vs block size.
    ms = sorted(curves)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.bar([str(m) for m in ms], [float(np.mean(curves[m]["psnr"])) for m in ms])
    a1.set_title("mean PSNR vs block size"); a1.set_xlabel("block size"); a1.set_ylabel("PSNR (dB)")
    a2.bar([str(m) for m in ms], [float(np.mean(curves[m]["action_rmse"])) for m in ms])
    a2.set_title("mean action RMSE vs block size"); a2.set_xlabel("block size"); a2.set_ylabel("RMSE")
    fig.tight_layout(); fig.savefig(out_root / "summary_vs_blocksize.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
