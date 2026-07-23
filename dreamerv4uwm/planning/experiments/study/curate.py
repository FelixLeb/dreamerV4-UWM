"""Descriptor-assisted proposer for a diverse curated init set.

Samples many candidate ``(window_idx, t0)`` decision points, scores each REAL
decision-frame image with the T-pose detector (no world model needed — we segment
the ground-truth frame, not a decoded latent), then farthest-point-selects a subset
that spans pose × start-reward space. Writes:
  * a YAML init file (``config/inits/pushT_curated.yaml``) consumed by run_sweep, and
  * a contact-sheet PNG of the selected decision frames (annotated) for pruning.

Run (no GPU required):
    python -m dreamerv4uwm.planning.experiments.study.curate \
        --config dreamerv4uwm/planning/experiments/study/config/sweep.yaml \
        --n-candidates 800 --n-select 40
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.nn.functional import interpolate
from omegaconf import OmegaConf

from ...reward import (score_t_centered, score_t_centered_straight,
                       score_t_centered_angle, _t_heading)
from .data import make_dataset


# ---------------------------------------------------------------------------
# selection helpers (pure; unit-tested without a dataset)
# ---------------------------------------------------------------------------

def farthest_point_sample(X: np.ndarray, k: int, seed: int = 0) -> List[int]:
    """Greedy farthest-point sampling over rows of standardized X."""
    n = len(X)
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    start = int(rng.integers(n))
    sel = [start]
    d = np.linalg.norm(X - X[start], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return sel


def _facing(heading_deg: float) -> str:
    """Coarse which-way-up tag from the T's crossbar heading (image y-down):
    -90=up (upright), +90=down (inverted), 0=right, +/-180=left."""
    targets = {"up": -90.0, "down": 90.0, "right": 0.0, "left": 180.0}
    circ = lambda a, b: abs((a - b + 180.0) % 360.0 - 180.0)
    return min(targets, key=lambda k: circ(heading_deg, targets[k]))


def auto_label(cx: float, cy: float, heading_deg: float, start_reward: float) -> str:
    horiz = "left" if cx < 0.4 else "right" if cx > 0.6 else "center"
    vert = "top" if cy < 0.4 else "bottom" if cy > 0.6 else "mid"
    rb = "near-goal" if start_reward >= 0.7 else "mid" if start_reward >= 0.35 else "far"
    return f"{vert}-{horiz}, {rb}, T-{_facing(heading_deg)} ({int(round(heading_deg))}deg)"


# ---------------------------------------------------------------------------
# candidate scoring (real frames)
# ---------------------------------------------------------------------------

def _decision_rgb(dataset, widx, t0, Tc, resolution=(256, 256)) -> np.ndarray:
    fr = dataset[widx]["image"][t0 + Tc - 1].float()        # (3,h,w)
    if fr.max() > 1.5:                                       # uint8-style [0,255] -> [0,1]
        fr = fr / 255.0
    fr = interpolate(fr[None], resolution)[0]
    return (fr.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


def make_reward_scorer(reward_cfg):
    """Return ``(score_fn(rgb) -> (score, dbg), kind)`` matching ``reward.kind`` on a
    single frame — the frame-level counterpart of ``run_sweep.build_reward`` (which builds
    the latent-space reward *class*). ``start_reward`` is this ``score``; the ``dbg`` it
    returns still carries ``center``/``centroid``/``mask``/``found`` for the fingerprint and
    the T-pose feature."""
    kind = str(reward_cfg.get("kind", "center")).lower()
    seg = dict(center_xy=tuple(reward_cfg.center_xy), sigma=float(reward_cfg.sigma))
    if kind == "center":
        return (lambda rgb: score_t_centered(rgb, **seg)), kind
    kw = dict(seg)
    if "combine" in reward_cfg:
        kw["combine"] = str(reward_cfg.combine)
    for k in ("w_center", "w_orient"):
        if k in reward_cfg:
            kw[k] = float(reward_cfg[k])
    if kind == "straight":
        for k in ("target_theta_deg", "sigma_theta_deg"):
            if k in reward_cfg:
                kw[k] = float(reward_cfg[k])
        return (lambda rgb: score_t_centered_straight(rgb, **kw)), kind
    if kind == "angle":
        for k in ("target_heading_deg", "sigma_heading_deg"):
            if k in reward_cfg:
                kw[k] = float(reward_cfg[k])
        return (lambda rgb: score_t_centered_angle(rgb, **kw)), kind
    raise ValueError(f"unknown reward.kind={kind!r} (expected center|straight|angle)")


def score_candidates(dataset, cands, Tc, score_fn, orient_weight: float = 0.5) -> list:
    """For each (widx, t0): pose (cx, cy, w*cos(phi), w*sin(phi), sqrt_area_norm),
    ``start_reward`` (the ``reward.kind`` score) and ``start_center_fingerprint`` (the center-only
    score, kept as a decode-robust, kind-independent dataset fingerprint) + found.
    Orientation is the up/down-resolved HEADING (same :func:`_t_heading` as
    THeadingDescriptor), circular-encoded so the curated set spans which-way-up too."""
    out = []
    for j, (widx, t0) in enumerate(cands):
        rgb = _decision_rgb(dataset, widx, t0, Tc)
        H, W = rgb.shape[:2]
        score, dbg = score_fn(rgb)
        center = float(dbg.get("center", score))          # center component (== score for kind=center)
        if dbg.get("found"):
            cx, cy = dbg["centroid"]
            phi = _t_heading(dbg["mask"])
            feat = [cx / W, cy / H, orient_weight * float(np.cos(phi)),
                    orient_weight * float(np.sin(phi)), float(np.sqrt(dbg["area"])) / np.sqrt(H * W)]
            out.append(dict(window_idx=int(widx), t0=int(t0), start_reward=float(score),
                            start_center_fingerprint=center, found=True, feat=feat,
                            heading_deg=float(np.degrees(phi)), rgb=rgb, dbg=dbg))
        else:
            out.append(dict(window_idx=int(widx), t0=int(t0), start_reward=float(score),
                            start_center_fingerprint=center, found=False, feat=None,
                            heading_deg=float("nan"), rgb=rgb, dbg=dbg))
        if (j + 1) % 100 == 0:
            print(f"  scored {j + 1}/{len(cands)}", flush=True)
    return out


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def write_yaml(path: Path, dataset_cfg: dict, Tc: int, chosen: list, reward_kind: str,
               reward_combine: str = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    L = ["# Curated initial contexts for the MCTS diagnostics sweep (generated by curate.py).",
         "# An init = a decision point: Tc context frames ending at the decision frame.",
         "# window_idx is a ShardedHDF5Dataset index (NOT a random seed).",
         "# Changing any dataset field below remaps window_idx -> different frames.",
         "# start_reward = the reward_kind score (below); start_center_fingerprint = center-only score,",
         "# the kind-independent, decode-robust fingerprint the loader checks for drift.",
         "", "dataset:"]
    for k in ("data_dir", "window_size", "stride", "split", "train_fraction",
              "split_seed", "shuffle_windows"):
        L.append(f"  {k}: {dataset_cfg[k]}")
    L += [f"Tc: {Tc}", f"reward_kind: {reward_kind}"]
    if reward_combine is not None:            # orientation combine mode (straight/angle only)
        L.append(f"reward_combine: {reward_combine}")
    L += ["", "inits:"]
    for i, c in enumerate(chosen):
        f = c["feat"]
        L.append(f"  - {{id: {i}, window_idx: {c['window_idx']}, t0: {c['t0']}, "
                 f"start_reward: {c['start_reward']:.3f}, start_center_fingerprint: {c['start_center_fingerprint']:.3f}, "
                 f"label: \"{auto_label(f[0], f[1], c['heading_deg'], c['start_reward'])}\"}}")
    path.write_text("\n".join(L) + "\n")


def contact_sheet(path: Path, chosen: list, ncol: int = 8) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ...reward import annotate_t
    n = len(chosen)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.7 * ncol, 1.9 * nrow), squeeze=False)
    for i, c in enumerate(chosen):
        ax = axes[i // ncol][i % ncol]
        ax.imshow(annotate_t(c["rgb"], c["dbg"]))
        ax.set_title(f"#{i} w{c['window_idx']} t{c['t0']}\nR={c['start_reward']:.2f}", fontsize=6)
        ax.axis("off")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="sweep.yaml (for data + reward params)")
    ap.add_argument("--n-candidates", type=int, default=800)
    ap.add_argument("--n-select", type=int, default=40)
    ap.add_argument("--tc", type=int, default=None, help="override Tc (else cfg.Tc or 8)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output yaml (default config/inits/pushT_curated.yaml)")
    ap.add_argument("overrides", nargs="*",
                    help="OmegaConf dotlist overrides, e.g. reward.combine=weighted_sum")
    args = ap.parse_args(argv)

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg_dir = Path(args.config).resolve().parent
    Tc = args.tc or int(cfg.get("Tc", 8))
    score_fn, reward_kind = make_reward_scorer(cfg.reward)
    reward_combine = str(cfg.reward.get("combine", "product")) if reward_kind != "center" else None
    orient_weight = float(cfg.get("descriptor", {}).get("orient_weight", 0.5))
    dcfg = dict(data_dir=cfg.data.data_dir, window_size=int(cfg.data.get("window_size", 64)),
                stride=1, split="train", train_fraction=0.9, split_seed=123, shuffle_windows=False)

    dataset = make_dataset(dcfg["data_dir"], window_size=dcfg["window_size"], stride=dcfg["stride"],
                           split=dcfg["split"], train_fraction=dcfg["train_fraction"],
                           split_seed=dcfg["split_seed"], shuffle_windows=dcfg["shuffle_windows"])
    print(f"[curate] dataset windows={len(dataset)}  Tc={Tc}  reward.kind={reward_kind}", flush=True)

    rng = np.random.default_rng(args.seed)
    t_hi = dcfg["window_size"] - Tc                     # t0 in [0, t_hi]
    cands = list(zip(rng.integers(0, len(dataset), args.n_candidates).tolist(),
                     rng.integers(0, t_hi + 1, args.n_candidates).tolist()))
    scored = score_candidates(dataset, cands, Tc, score_fn, orient_weight=orient_weight)

    found = [c for c in scored if c["found"]]
    print(f"[curate] {len(found)}/{len(scored)} candidates have a valid T", flush=True)
    if len(found) < args.n_select:
        raise RuntimeError(f"only {len(found)} valid candidates < n_select={args.n_select}; "
                           f"raise --n-candidates")

    F = np.array([c["feat"] + [c["start_reward"]] for c in found], float)
    Fs = (F - F.mean(0)) / (F.std(0) + 1e-9)
    sel = farthest_point_sample(Fs, args.n_select, seed=args.seed)
    chosen = [found[i] for i in sel]
    # order by start_reward for a readable contact sheet / file
    chosen.sort(key=lambda c: c["start_reward"])

    out = Path(args.out) if args.out else cfg_dir / "inits" / "pushT_curated.yaml"
    write_yaml(out, dcfg, Tc, chosen, reward_kind, reward_combine)
    sheet = out.with_suffix(".png")
    contact_sheet(sheet, chosen)
    rr = np.array([c["start_reward"] for c in chosen])
    print(f"[curate] wrote {len(chosen)} inits -> {out}  (reward.kind={reward_kind})\n"
          f"         contact sheet -> {sheet}\n"
          f"         start_reward span [{rr.min():.2f}, {rr.max():.2f}] mean {rr.mean():.2f}", flush=True)


if __name__ == "__main__":
    main()