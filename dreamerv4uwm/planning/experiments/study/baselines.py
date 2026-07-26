"""In-model outcome baselines (Family F). No environment executor.

Since there is no closed-loop simulator, "did planning help?" is measured
*relative* to cheaper in-model alternatives. Two families of baseline:

* **flat** (``_rollout_peak``) — a single ``sim_horizon``-frame rollout. Cheap, but
  it looks only ``sim_horizon`` frames ahead while the plan reaches up to
  ``horizon*max_depth`` frames, so it gives the tree a free lookahead advantage and
  its length tracks ``sim_horizon`` (confounding that sweep).
* **fair** (``_rollout_fair_peak``) — the honest control: built the SAME WAY the tree
  builds a plan (``max_depth`` edges of ``horizon`` frames, re-conditioning the
  context after each edge, exactly like MCTS ``_advance_ctx``), just with no search.
  Same lookahead (``horizon*max_depth``) and same edge-by-edge, re-anchored
  construction as the plan, so ``g`` credits *search*, not horizon or roll-out style.

Outputs (per tree):
  flat: ``g_shootN``, ``g_1shot``       (baselines ``shootN_peak``, ``oneshot_peak``)
  fair: ``g_shootN_fair``, ``g_1shot_fair`` (baselines ``shootN_fair_peak``, ``oneshot_fair_peak``)

All peaks are "best reward over any frame of the rollout" (MPC-style, matching the
tree's best-prefix objective; the start/context frame is excluded, as in ``tree_peak``).
The ``shootN`` variants take the best of ``n_random`` rollouts (random shooting); the
``1shot`` variants use a single rollout (act on the prior once, no search).
"""
from __future__ import annotations

import torch

from ... import rollout as R


@torch.no_grad()
def _rollout_peak(denoiser, reward_fn, ctx_z, ctx_a, *, H, B, K, ctx_noise,
                  ctx_noise_honest, action_temp, action_prior, dtype, gen) -> float:
    """Best single-frame reward over B rollouts of length H (one flat rollout each)."""
    z, _ = R.imagine(denoiser, ctx_z, ctx_a, H, B=B, K=K, ctx_noise=ctx_noise,
                     ctx_noise_honest=ctx_noise_honest, action_temp=action_temp,
                     action_prior=action_prior, dtype=dtype, generator=gen)
    r = reward_fn(z)                       # (B, H)
    return float(r.max().item())


@torch.no_grad()
def _rollout_fair_peak(denoiser, reward_fn, ctx_z, ctx_a, *, depth, H, B, K, ctx_noise,
                       ctx_noise_honest, action_temp, action_prior, dtype, max_ctx, gen) -> float:
    """Best single-frame reward over B *no-search* rollouts built like the tree's plan.

    Each of the B rollouts runs ``depth`` edges of ``H`` frames, and after every edge
    the context window is advanced with the imagined states/actions and truncated to
    ``max_ctx`` — mirroring MCTS ``_advance_ctx``. This matches the plan's lookahead
    (``depth*H`` frames) AND its edge-by-edge, re-anchored construction, so the only
    thing the tree adds over this baseline is the *search* (selecting which edge to
    extend), not a longer or single-shot rollout.
    """
    cz = ctx_z.expand(B, -1, -1, -1).contiguous()   # (B, Tc, N, D)
    ca = ctx_a.expand(B, -1, -1).contiguous()        # (B, Tc, n_act)
    best = float("-inf")
    for _ in range(max(int(depth), 1)):
        z, a = R.imagine(denoiser, cz, ca, H, B=B, K=K, ctx_noise=ctx_noise,
                         ctx_noise_honest=ctx_noise_honest, action_temp=action_temp,
                         action_prior=action_prior, dtype=dtype, generator=gen)   # (B,H,N,D), (B,H,n_act)
        best = max(best, float(reward_fn(z).max().item()))    # exclude start frame, like tree_peak
        cz = torch.cat([cz, z], dim=1)[:, -max_ctx:].contiguous()   # advance ctx (== _advance_ctx)
        ca = torch.cat([ca, a], dim=1)[:, -max_ctx:].contiguous()
    return best


@torch.no_grad()
def compute_baselines(denoiser, reward_fn, ctx_z, ctx_a, cfg, *, tree_peak: float,
                      n_random: int = 16, seed: int = 12345, fair: bool = True,
                      flat: bool = True) -> dict:
    """Return the baseline peaks and planning gains for one tree.

    ``tree_peak`` is the planner's achieved peak reward (computed in run_tree).
    ``fair``/``flat`` toggle the two baseline families (both on by default). The fair
    family costs ~``max_depth``x more decodes than the flat one — disable ``flat`` if
    you only want the fair comparison.
    """
    dev = ctx_z.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    root_reward = float(reward_fn(ctx_z[:, -1:]).reshape(-1)[0].item())
    kw = dict(K=cfg.K_steps, ctx_noise=cfg.ctx_noise, ctx_noise_honest=cfg.ctx_noise_honest,
              action_temp=cfg.action_temp, action_prior=cfg.action_prior, dtype=cfg.dtype, gen=gen)
    out = {"root_reward": root_reward, "delta_over_root": tree_peak - root_reward}

    if flat:  # single sim_horizon rollout (cheap, unequal lookahead)
        shootN_peak = _rollout_peak(denoiser, reward_fn, ctx_z, ctx_a,
                                    H=cfg.sim_horizon, B=n_random, **kw)
        oneshot_peak = _rollout_peak(denoiser, reward_fn, ctx_z, ctx_a,
                                     H=cfg.sim_horizon, B=1, **kw)
        out.update(shootN_peak=shootN_peak, oneshot_peak=oneshot_peak,
                   g_shootN=tree_peak - shootN_peak, g_1shot=tree_peak - oneshot_peak)

    if fair:  # depth-deep, re-conditioned rollout matching the plan's construction
        shootN_fair = _rollout_fair_peak(denoiser, reward_fn, ctx_z, ctx_a,
                                         depth=cfg.max_depth, H=cfg.horizon,
                                         B=n_random, max_ctx=cfg.max_ctx, **kw)
        oneshot_fair = _rollout_fair_peak(denoiser, reward_fn, ctx_z, ctx_a,
                                          depth=cfg.max_depth, H=cfg.horizon,
                                          B=1, max_ctx=cfg.max_ctx, **kw)
        out.update(shootN_fair_peak=shootN_fair, oneshot_fair_peak=oneshot_fair,
                   g_shootN_fair=tree_peak - shootN_fair, g_1shot_fair=tree_peak - oneshot_fair)

    return out
