#!/usr/bin/env python
"""
Tokenizer noise-robustness evaluation.

Sweeps tau in [0.0, 0.1, ..., 1.0] (fraction of clean latent vs N(0,1) noise)
across 5 random snippets of the SOAR dataset, for four tokenizers:

  - DreamerV4 UWM tokenizer (checkpoint)
  - SD3 VAE                 (per-frame image tokenizer)
  - Wan2.1 VAE              (spatiotemporal: 8x spatial, 4x temporal)
  - Wan2.1 VAE              (per-frame mode, NUM_FRAMES=1)

Outputs (written to <repo>/notebooks/results/tokenizer-eval/):
  ground_truth/snippet{i}.{gif,mp4}
  <method>/snippet{i}_tau{X.X}.{gif,mp4}
  psnr.png, lpips.png, metrics.json
"""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL, AutoencoderKLWan
from hydra import compose, initialize
from torch.nn.functional import interpolate

from dreamerv4uwm.datasets import ShardedHDF5Dataset
from dreamerv4uwm.models.utils import load_tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = "/media/mim-server/5a9b3378-c509-41de-b07f-544b25e6a481/soar_data_sharded"
DEFAULT_TOK_CKPT = str(REPO_ROOT / "checkpoints/tokenizer_ckpts/soar.pt")
DEFAULT_OUT = REPO_ROOT / "notebooks/results/tokenizer-eval"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DEFAULT_DATA)
    p.add_argument("--tokenizer_ckpt", default=DEFAULT_TOK_CKPT)
    p.add_argument("--config_name", default="dynamics/pushT.yaml",
                   help="Hydra config providing tokenizer architecture")
    p.add_argument("--num_snippets", type=int, default=5)
    p.add_argument("--num_frames", type=int, default=33,
                   help="Frames per snippet (must satisfy 4k+1 for Wan-VAE).")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def to_uint8(video01):
    v = video01.detach().cpu().to(torch.float32).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return (v * 255).astype(np.uint8)


def save_video(video01, path_no_ext, fps):
    arr = to_uint8(video01)
    iio.imwrite(str(path_no_ext) + ".mp4", arr, fps=fps, codec="libx264")
    iio.imwrite(str(path_no_ext) + ".gif", arr, duration=int(1000 / fps), loop=0)


def psnr01(a, b):
    mse = ((a.float() - b.float()) ** 2).mean()
    return float(-10.0 * torch.log10(mse + 1e-12))


def lpips_metric(net, a01, b01):
    with torch.no_grad():
        d = net(a01 * 2 - 1, b01 * 2 - 1)
    return float(d.mean())


# ------------------------- per-method runners -------------------------

@torch.no_grad()
def run_dreamer(tokenizer, frames01, taus, generator):
    """frames01: (T, 3, H, W) on cuda, [0,1]. Returns dict[tau] -> (T,3,H,W) [0,1]."""
    imgs = frames01.unsqueeze(0)  # (1, T, C, H, W)
    out = {}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        latents = tokenizer.encode(imgs)
        noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        for tau in taus:
            mix = tau * latents + (1 - tau) * noise
            recon = tokenizer.decode(mix)  # (1, T, 3, H, W) [0,1]
            out[tau] = recon[0].float().cpu()
    return out


@torch.no_grad()
def run_sd3(sd3_vae, frames01, taus, generator):
    frames = (frames01 * 2 - 1).to(dtype=torch.float16)
    latent = sd3_vae.encode(frames).latent_dist.mode()
    noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
    out = {}
    for tau in taus:
        mix = tau * latent + (1 - tau) * noise
        recon = sd3_vae.decode(mix).sample
        out[tau] = ((recon.clamp(-1, 1) + 1) / 2).float().cpu()
    return out


@torch.no_grad()
def run_wan_temporal(wan_vae, frames01, taus, generator):
    """frames01: (T, 3, H, W) where T = 4k+1."""
    video = (frames01 * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0).to(dtype=torch.float32)  # (1,3,T,H,W)
    latent = wan_vae.encode(video).latent_dist.mode()
    noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
    out = {}
    for tau in taus:
        mix = tau * latent + (1 - tau) * noise
        recon = wan_vae.decode(mix, return_dict=False)[0]  # (1,3,T,H,W) in [-1,1]
        recon = (recon.clamp(-1, 1) + 1) / 2
        out[tau] = recon[0].permute(1, 0, 2, 3).float().cpu()  # (T, 3, H, W)
    return out


@torch.no_grad()
def run_wan_perframe(wan_vae, frames01, taus, generator, chunk=8):
    """Encode each frame as its own 1-frame video; latent T=1 each."""
    T = frames01.shape[0]
    # (T, 3, 1, H, W) — batch of single-frame videos
    video = (frames01 * 2 - 1).unsqueeze(2).to(dtype=torch.float32)
    # Encode in chunks to keep VRAM bounded.
    latents = []
    for s in range(0, T, chunk):
        latents.append(wan_vae.encode(video[s:s + chunk]).latent_dist.mode())
    latent = torch.cat(latents, dim=0)  # (T, 16, 1, h, w)
    noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
    out = {}
    for tau in taus:
        mix = tau * latent + (1 - tau) * noise
        recons = []
        for s in range(0, T, chunk):
            r = wan_vae.decode(mix[s:s + chunk], return_dict=False)[0]  # (chunk,3,1,H,W)
            recons.append(r)
        recon = torch.cat(recons, dim=0)
        recon = (recon.clamp(-1, 1) + 1) / 2
        out[tau] = recon.squeeze(2).float().cpu()  # (T, 3, H, W)
    return out


# ------------------------- main -------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_root = Path(args.out_dir)
    method_dirs = ["ground_truth", "dreamer_uwm", "sd3_vae", "wan_vae_temporal", "wan_vae_perframe"]
    for m in method_dirs:
        (out_root / m).mkdir(parents=True, exist_ok=True)

    device = "cuda"
    taus = [round(0.1 * i, 1) for i in range(11)]
    res = (args.resolution, args.resolution)

    # ---- Snippets ----
    print("Loading dataset…")
    dataset = ShardedHDF5Dataset(
        data_dir=args.data_path,
        window_size=64, stride=1, split="train",
        train_fraction=0.9, split_seed=123,
    )
    indices = sorted(rng.choice(len(dataset), size=args.num_snippets, replace=False).tolist())
    print(f"Selected snippet indices: {indices}")

    snippets = []
    for i, idx in enumerate(indices):
        batch = dataset[idx]
        frames = batch["image"]  # (T, 3, H, W) [0,1]
        frames = interpolate(frames, res)[: args.num_frames]
        snippets.append(frames.contiguous())
        save_video(frames, out_root / "ground_truth" / f"snippet{i}", args.fps)
    print(f"Saved {args.num_snippets} ground-truth snippets to {out_root/'ground_truth'}")

    # ---- LPIPS ----
    print("Loading LPIPS (alex)…")
    lpips_net = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    metrics = {m: {tau: [] for tau in taus} for m in method_dirs if m != "ground_truth"}

    def eval_method(name, runner):
        print(f"\n=== {name} ===")
        for i, frames_cpu in enumerate(snippets):
            frames = frames_cpu.to(device)
            gen = torch.Generator(device=device).manual_seed(args.seed + i)
            recons = runner(frames, gen)
            for tau, recon in recons.items():
                T_recon = recon.shape[0]
                gt_dev = frames[:T_recon]
                rec_dev = recon.to(device)
                p = psnr01(gt_dev, rec_dev)
                l = lpips_metric(lpips_net, gt_dev, rec_dev)
                metrics[name][tau].append((p, l))
                save_video(recon, out_root / name / f"snippet{i}_tau{tau:.1f}", args.fps)
            print(f"  snippet {i}: " + ", ".join(
                f"τ={t:.1f} P={metrics[name][t][-1][0]:.2f} L={metrics[name][t][-1][1]:.3f}"
                for t in (0.0, 0.5, 1.0)
            ))
            del recons
            torch.cuda.empty_cache()

    # ---- 1) DreamerV4 UWM ----
    print("\nLoading DreamerV4 UWM tokenizer…")
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config_name)
    cfg.tokenizer_ckpt = args.tokenizer_ckpt
    dreamer_tok = load_tokenizer(cfg, device, max_num_forward_steps=300).eval().cuda()
    eval_method("dreamer_uwm", lambda fr, g: run_dreamer(dreamer_tok, fr, taus, g))
    del dreamer_tok
    torch.cuda.empty_cache()

    # ---- 2) SD3 VAE ----
    print("\nLoading SD3 VAE…")
    sd3_vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="vae", torch_dtype=torch.float16,
    ).to(device).eval()
    eval_method("sd3_vae", lambda fr, g: run_sd3(sd3_vae, fr, taus, g))
    del sd3_vae
    torch.cuda.empty_cache()

    # ---- 3) Wan2.1 VAE — temporal & per-frame ----
    print("\nLoading Wan2.1 VAE…")
    wan_vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae", torch_dtype=torch.float32,
    ).to(device).eval()
    wan_vae.enable_tiling()
    eval_method("wan_vae_temporal", lambda fr, g: run_wan_temporal(wan_vae, fr, taus, g))
    eval_method("wan_vae_perframe", lambda fr, g: run_wan_perframe(wan_vae, fr, taus, g))
    del wan_vae
    torch.cuda.empty_cache()

    # ---- Aggregate + save ----
    summary = {}
    for name in metrics:
        psnrs = [[t[0] for t in metrics[name][tau]] for tau in taus]
        lps = [[t[1] for t in metrics[name][tau]] for tau in taus]
        summary[name] = {
            "tau": taus,
            "psnr_mean": [float(np.mean(x)) for x in psnrs],
            "psnr_std":  [float(np.std(x))  for x in psnrs],
            "lpips_mean": [float(np.mean(x)) for x in lps],
            "lpips_std":  [float(np.std(x))  for x in lps],
        }
    with open(out_root / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_root/'metrics.json'}")

    # ---- Plots ----
    pretty = {
        "dreamer_uwm": "DreamerV4 UWM",
        "sd3_vae": "SD3 VAE (per-frame)",
        "wan_vae_temporal": "Wan2.1 VAE (spatiotemporal)",
        "wan_vae_perframe": "Wan2.1 VAE (per-frame)",
    }
    colors = {
        "dreamer_uwm": "tab:blue",
        "sd3_vae": "tab:orange",
        "wan_vae_temporal": "tab:green",
        "wan_vae_perframe": "tab:red",
    }
    markers = {"dreamer_uwm": "o", "sd3_vae": "s", "wan_vae_temporal": "^", "wan_vae_perframe": "D"}

    def make_plot(metric_key, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for name in summary:
            ax.plot(
                summary[name]["tau"], summary[name][metric_key],
                label=pretty[name], color=colors[name], marker=markers[name],
                linewidth=2.8, markersize=7,
            )
        ax.set_xlabel(r"$\tau$  (clean-latent fraction; 0 = pure noise, 1 = clean)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, frameon=True, loc="best")
        ax.tick_params(labelsize=11)
        ax.set_xticks(taus)
        fig.tight_layout()
        fig.savefig(out_root / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_root/fname}")

    make_plot("psnr_mean", "PSNR (dB)  ↑",
              "Reconstruction PSNR vs noise interpolation level", "psnr.png")
    make_plot("lpips_mean", "LPIPS  ↓",
              "Reconstruction LPIPS vs noise interpolation level", "lpips.png")

    print(f"\nAll outputs written under {out_root}")


if __name__ == "__main__":
    main()
