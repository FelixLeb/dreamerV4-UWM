#!/usr/bin/env python3
"""
viz_fdpo_pairs.py

Visualize (positive, negative) trajectory pairs produced by
gen_fdpo_negatives.py (schema 'fdpo_pairs_v1'). For each pair, saves a PNG with:

  - Top: two strips of frame thumbnails (positive on top, negative below) at
    fixed timestamps. A red separator marks the ctx/horizon boundary.
  - Bottom: 2-D action trajectories over time (positive solid, negative dashed)
    with a vertical line at the boundary.

Usage:
    conda run -n dreamerv4 python scripts/preprocessing/viz_fdpo_pairs.py \\
        --pairs_dir  /tmp/fdpo_pairs_smoke \\
        --output_dir /tmp/fdpo_viz
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _list_shards(d: Path):
    return sorted(d.glob("shard_*.h5"))


def _frame_strip(ax, frames_thwc_u8, ts, label, boundary_t):
    """Plot a row of thumbnails at the given timesteps."""
    n = len(ts)
    H, W = frames_thwc_u8.shape[1], frames_thwc_u8.shape[2]
    composite = np.zeros((H, W * n, 3), dtype=np.uint8)
    for i, t in enumerate(ts):
        composite[:, i * W:(i + 1) * W] = frames_thwc_u8[t]
    ax.imshow(composite)
    ax.set_xticks([i * W + W // 2 for i in range(n)])
    ax.set_xticklabels([f"t={t}" for t in ts], fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel(label, fontsize=10)
    # Red boundary line between the last ctx tick and the first horizon tick
    last_ctx_i = max([i for i, t in enumerate(ts) if t < boundary_t], default=-1)
    if 0 <= last_ctx_i < n - 1:
        x_line = (last_ctx_i + 1) * W
        ax.axvline(x_line - 0.5, color="red", lw=2)


def _action_plot(ax, demo_acts, neg_acts, boundary_t):
    T, A = demo_acts.shape
    t_axis = np.arange(T)
    for d in range(A):
        ax.plot(t_axis, demo_acts[:, d], "-",  color=f"C{d}", lw=1.5,
                label=f"demo a[{d}]")
        ax.plot(t_axis, neg_acts[:, d],  "--", color=f"C{d}", lw=1.5,
                label=f"neg  a[{d}]")
    ax.axvline(boundary_t - 0.5, color="red", lw=1.5, alpha=0.7)
    ax.set_xlabel("frame")
    ax.set_ylabel("action")
    ax.legend(loc="upper right", fontsize=7, ncols=2)
    ax.grid(alpha=0.3)


def _build_figure(pos_imgs, pos_acts, neg_imgs, neg_acts, boundary_t,
                  title: str, ts):
    fig, axes = plt.subplots(3, 1, figsize=(14, 6),
                             gridspec_kw={"height_ratios": [1, 1, 1.4]})
    _frame_strip(axes[0], pos_imgs, ts, "ξ⁺ (demo)", boundary_t)
    _frame_strip(axes[1], neg_imgs, ts, "ξ⁻ (gen) ", boundary_t)
    _action_plot(axes[2], pos_acts, neg_acts, boundary_t)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir",  type=str, required=True,
                   help="Output dir of gen_fdpo_negatives.py (schema fdpo_pairs_v1).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_pairs",  type=int, default=None)
    p.add_argument("--frame_count", type=int, default=9,
                   help="Number of thumbnails per row (frames evenly spaced).")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = Path(args.pairs_dir)

    meta_path = pairs_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    schema = meta.get("schema")
    if schema != "fdpo_pairs_v1":
        print(f"WARNING: metadata schema={schema!r}, expected 'fdpo_pairs_v1'")
    ctx_frames = int(meta.get("ctx_frames", 32))
    horizon_frames = int(meta.get("horizon_frames", 32))
    window = ctx_frames + horizon_frames

    n_done = 0
    for shard_path in _list_shards(pairs_dir):
        with h5py.File(shard_path, "r") as f:
            n_pairs = int(f.attrs.get("num_pairs", f["pos_images"].shape[0]))
            for i in range(n_pairs):
                pos_imgs = f["pos_images"][i]
                pos_acts = f["pos_actions"][i]
                neg_imgs = f["neg_images"][i]
                neg_acts = f["neg_actions"][i]

                ts = np.linspace(0, window - 1, args.frame_count).astype(int).tolist()
                title = (f"{shard_path.name} | pair {i} | "
                         f"frames [0,{ctx_frames}) shared context | "
                         f"[{ctx_frames},{window}) demo (ξ⁺) vs ref rollout (ξ⁻)")
                fig = _build_figure(pos_imgs, pos_acts, neg_imgs, neg_acts,
                                    boundary_t=ctx_frames, title=title, ts=ts)
                out_path = out_dir / f"{shard_path.stem}_pair{i:03d}.png"
                fig.savefig(out_path, dpi=110, bbox_inches="tight")
                plt.close(fig)
                n_done += 1
                if args.max_pairs is not None and n_done >= args.max_pairs:
                    break
        if args.max_pairs is not None and n_done >= args.max_pairs:
            break
    print(f"Wrote {n_done} viz PNGs -> {out_dir}")


if __name__ == "__main__":
    main()
