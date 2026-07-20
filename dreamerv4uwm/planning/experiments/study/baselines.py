"""In-model outcome baselines (Family F). No environment executor.

Since there is no closed-loop simulator, "did planning help?" is measured
*relative* to cheaper in-model alternatives at comparable budget:

* ``g_shootN``   = tree peak − best peak of ``n_random`` undirected rollouts.
* ``g_1shot`` = tree peak − peak of a single policy-prior unroll (no search).

All peaks are "best reward over any frame of the rollout" (MPC-style, matching the
tree's best-prefix objective). Rollouts use the same edge generator (``R.imagine``)
and the tree's own diversity knobs, so the comparison is apples-to-apples.
"""
from __future__ import annotations

from typing import Optional

import torch

from ... import rollout as R


@torch.no_grad()
def _rollout_peak(denoiser, reward_fn, ctx_z, ctx_a, *, H, B, K, ctx_noise,
                  ctx_noise_honest, action_temp, dtype, gen) -> float:
    """Best single-frame reward over B rollouts of length H."""
    z, _ = R.imagine(denoiser, ctx_z, ctx_a, H, B=B, K=K, ctx_noise=ctx_noise,
                     ctx_noise_honest=ctx_noise_honest, action_temp=action_temp,
                     dtype=dtype, generator=gen)
    r = reward_fn(z)                       # (B, H)
    return float(r.max().item())


@torch.no_grad()
def compute_baselines(denoiser, reward_fn, ctx_z, ctx_a, cfg, *, tree_peak: float,
                      n_random: int = 16, seed: int = 12345) -> dict:
    """Return ``{root_reward, g_shootN, g_1shot, shootN_peak, oneshot_peak}``.
    ``tree_peak`` is the planner's achieved peak reward (computed in run_tree)."""
    dev = ctx_z.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    root_reward = float(reward_fn(ctx_z[:, -1:]).reshape(-1)[0].item())

    shootN_peak = _rollout_peak(
        denoiser, reward_fn, ctx_z, ctx_a, H=cfg.sim_horizon, B=n_random, K=cfg.K_steps,
        ctx_noise=cfg.ctx_noise, ctx_noise_honest=cfg.ctx_noise_honest,
        action_temp=cfg.action_temp, dtype=cfg.dtype, gen=gen)
    # policy prior = a single unroll, no search (low-temp = the prior's own mode-ish action)
    oneshot_peak = _rollout_peak(
        denoiser, reward_fn, ctx_z, ctx_a, H=cfg.sim_horizon, B=1, K=cfg.K_steps,
        ctx_noise=cfg.ctx_noise, ctx_noise_honest=cfg.ctx_noise_honest,
        action_temp=cfg.action_temp, dtype=cfg.dtype, gen=gen)
    return dict(
        root_reward=root_reward,
        shootN_peak=shootN_peak, oneshot_peak=oneshot_peak,
        g_shootN=tree_peak - shootN_peak,
        g_1shot=tree_peak - oneshot_peak,
        delta_over_root=tree_peak - root_reward,
    )
