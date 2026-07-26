"""In-model outcome baselines (Family F). No environment executor.

Since there is no closed-loop simulator, "did planning help?" is measured *relative* to
cheaper in-model alternatives. **Both** baselines are built the SAME WAY the tree builds a
plan (``max_depth`` edges of ``horizon`` frames, re-conditioning the context after each edge
and truncating to ``max_ctx``, exactly like MCTS ``_advance_ctx``) — so they share the plan's
lookahead (``horizon*max_depth``) and edge-by-edge construction, and ``g`` credits *search*,
not horizon or roll-out style. They differ only in how many rollouts they draw:

* **random** (``_rollout_random_peak``) — a SINGLE depth-deep rollout: one sample per edge,
  no selection. The undirected control ("act on the prior once, roll forward").
* **greedy** (``_rollout_greedy_peak``) — the best of ``n_random`` depth-deep rollouts
  (random shooting): draw N and keep the best. The "greedily pick the best of N" control.

**Caveat (edge_mode).** Both baselines build edges with the ``imagine`` sampler (they respect
``ctx_noise`` / ``action_temp`` / ``action_prior`` but not ``edge_mode`` or the
autoregressive-only ``action_noise``). So under ``edge_mode='two_stage'`` / ``'autoregressive'``
the baselines still *imagine* each edge, and ``g_random`` / ``g_greedy`` then also reflect the
tree's edge-sampler choice, not search alone. To isolate search under a non-default sampler,
the baselines would need to build edges with that same sampler.

Outputs (per tree): ``g_random`` (baseline ``random_peak``), ``g_greedy`` (baseline
``greedy_peak``), and ``delta_over_root``. All peaks are "best reward over any frame of the
rollout" (MPC-style, matching the tree's best-prefix objective; the start/context frame is
excluded, as in ``tree_peak``).
"""
from __future__ import annotations

import torch

from ... import rollout as R


@torch.no_grad()
def _rollout_random_peak(denoiser, reward_fn, ctx_z, ctx_a, *, depth, H, K, ctx_noise,
                         ctx_noise_honest, action_temp, action_prior, dtype, max_ctx, gen) -> float:
    """Best single-frame reward over ONE no-search rollout built like the tree's plan.

    A single depth-deep trajectory: ``depth`` edges of ``H`` frames, re-conditioning the
    context after each edge (imagined states/actions appended, truncated to ``max_ctx`` —
    mirroring MCTS ``_advance_ctx``). One rollout per edge, no selection, so this is the
    *undirected* control at the plan's lookahead (``depth*H`` frames): the tree adds *search*
    over it.
    """
    cz, ca = ctx_z, ctx_a                                       # B = 1 (single rollout)
    best = float("-inf")
    for _ in range(max(int(depth), 1)):
        z, a = R.imagine(denoiser, cz, ca, H, B=1, K=K, ctx_noise=ctx_noise,
                         ctx_noise_honest=ctx_noise_honest, action_temp=action_temp,
                         action_prior=action_prior, dtype=dtype, generator=gen)   # (1,H,N,D)
        best = max(best, float(reward_fn(z).max().item()))         # exclude start frame, like tree_peak
        cz = torch.cat([cz, z], dim=1)[:, -max_ctx:].contiguous()  # advance ctx (== _advance_ctx)
        ca = torch.cat([ca, a], dim=1)[:, -max_ctx:].contiguous()
    return best


@torch.no_grad()
def _rollout_greedy_peak(denoiser, reward_fn, ctx_z, ctx_a, *, depth, H, B, K, ctx_noise,
                         ctx_noise_honest, action_temp, action_prior, dtype, max_ctx, gen) -> float:
    """Best single-frame reward over B *no-search* depth-deep rollouts — random shooting.

    Runs B independent depth-deep trajectories in parallel (each ``depth`` edges of ``H``
    frames, re-conditioned on its OWN imagination, truncated to ``max_ctx`` like MCTS
    ``_advance_ctx``) and keeps the best reward over all of them. Same lookahead and
    construction as the plan; the only thing the tree adds is the *search* (which edge to
    extend) beyond greedily taking the best of B shots.
    """
    cz = ctx_z.expand(B, -1, -1, -1).contiguous()   # (B, Tc, N, D)
    ca = ctx_a.expand(B, -1, -1).contiguous()        # (B, Tc, n_act)
    best = float("-inf")
    for _ in range(max(int(depth), 1)):
        z, a = R.imagine(denoiser, cz, ca, H, B=B, K=K, ctx_noise=ctx_noise,
                         ctx_noise_honest=ctx_noise_honest, action_temp=action_temp,
                         action_prior=action_prior, dtype=dtype, generator=gen)   # (B,H,N,D)
        best = max(best, float(reward_fn(z).max().item()))
        cz = torch.cat([cz, z], dim=1)[:, -max_ctx:].contiguous()
        ca = torch.cat([ca, a], dim=1)[:, -max_ctx:].contiguous()
    return best


@torch.no_grad()
def compute_baselines(denoiser, reward_fn, ctx_z, ctx_a, cfg, *, tree_peak: float,
                      n_random: int = 16, seed: int = 12345, random: bool = True,
                      greedy: bool = True) -> dict:
    """Return the baseline peaks and planning gains for one tree.

    ``tree_peak`` is the planner's achieved peak reward (computed in run_tree). Two
    matched-lookahead, no-search controls (both on by default):
      ``random`` -> a single depth-deep re-conditioned rollout (undirected);
      ``greedy`` -> best of ``n_random`` depth-deep rollouts (random shooting).
    ``greedy`` costs ~``n_random``x more decodes than ``random`` — disable it if you only want
    the cheap single-rollout comparison.
    """
    dev = ctx_z.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    root_reward = float(reward_fn(ctx_z[:, -1:]).reshape(-1)[0].item())
    kw = dict(K=cfg.K_steps, ctx_noise=cfg.ctx_noise, ctx_noise_honest=cfg.ctx_noise_honest,
              action_temp=cfg.action_temp, action_prior=cfg.action_prior, dtype=cfg.dtype, gen=gen)
    out = {"root_reward": root_reward, "delta_over_root": tree_peak - root_reward}

    if random:  # one depth-deep re-conditioned rollout (undirected)
        random_peak = _rollout_random_peak(denoiser, reward_fn, ctx_z, ctx_a,
                                           depth=cfg.max_depth, H=cfg.horizon,
                                           max_ctx=cfg.max_ctx, **kw)
        out.update(random_peak=random_peak, g_random=tree_peak - random_peak)

    if greedy:  # best of n_random depth-deep rollouts (random shooting)
        greedy_peak = _rollout_greedy_peak(denoiser, reward_fn, ctx_z, ctx_a,
                                           depth=cfg.max_depth, H=cfg.horizon,
                                           B=n_random, max_ctx=cfg.max_ctx, **kw)
        out.update(greedy_peak=greedy_peak, g_greedy=tree_peak - greedy_peak)

    return out
