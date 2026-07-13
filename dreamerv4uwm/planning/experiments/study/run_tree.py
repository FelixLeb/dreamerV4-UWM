"""Build ONE MCTS tree and return a single flat metrics row (one CSV line).

Deterministic given ``plan_seed`` + initial context. This is the unit the sweep
(``run_sweep.py``) maps over; keeping it pure and side-effect-free makes the sweep
trivially parallel and resumable.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Optional

import torch

from ...mcts import MCTS, PlanConfig
from .metrics import compute_tree_metrics
from .baselines import compute_baselines


def _plan_peak(reward_fn, out) -> float:
    """Best single-frame reward over the frames of the returned plan."""
    pz = out.get("plan_z")
    if pz is None:
        return float("nan")
    N, D = pz.shape[-2], pz.shape[-1]
    frames = pz.reshape(-1, N, D)
    return float(reward_fn(frames).max().item())


def _cfg_row(cfg: PlanConfig) -> dict:
    row = {k: v for k, v in asdict(cfg).items() if k != "dtype"}
    row["dtype"] = str(cfg.dtype).replace("torch.", "")
    return row


@torch.no_grad()
def run_one_tree(*, denoiser, reward_fn, descriptor, plan_cfg: PlanConfig,
                 ctx_z: torch.Tensor, ctx_a: torch.Tensor, plan_seed: int,
                 meta: Optional[dict] = None, baselines: bool = True,
                 n_random: int = 16) -> dict:
    """Return one flat row: meta + config + timing + outcome + all tree metrics."""
    t0 = time.time()
    planner = MCTS(denoiser, reward_fn, plan_cfg, seed=plan_seed, trace=True)
    out = planner.plan(ctx_z, ctx_a)
    plan_secs = time.time() - t0

    tree_peak = _plan_peak(reward_fn, out)
    metrics = compute_tree_metrics(planner, descriptor, result=out)

    row = dict(meta or {})
    row.update(_cfg_row(plan_cfg))
    row["plan_seed"] = plan_seed
    row["plan_secs"] = plan_secs
    row["tree_peak"] = tree_peak
    if baselines:
        row.update(compute_baselines(denoiser, reward_fn, ctx_z, ctx_a, plan_cfg,
                                     tree_peak=tree_peak, n_random=n_random,
                                     seed=plan_seed + 100003))
    row.update(metrics)
    return row
