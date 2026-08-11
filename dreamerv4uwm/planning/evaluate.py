"""In-model outcome baselines (Family F). No environment executor.

Since there is no closed-loop simulator, "did planning help?" is measured *relative* to
cheaper in-model alternatives. **Both** baselines are built the SAME WAY the tree builds a
plan (``max_depth`` edges of ``horizon`` frames, re-conditioning the context after each edge
and truncating to ``max_ctx``, exactly like MCTS ``_advance_ctx``) — so they share the plan's
lookahead (``horizon*max_depth``) and edge-by-edge construction, and ``g`` credits *search*,
not horizon or roll-out style. They differ only in how many rollouts they draw:

* **random** (``_rollout_random_peak`` / ``_rollout_random_last``) — a SINGLE depth-deep
  rollout: one sample per edge, no selection. The undirected control ("act on the prior
  once, roll forward").
* **greedy** (``_rollout_greedy_peak`` / ``_rollout_greedy_last``) — the best of
  ``n_random`` depth-deep rollouts (random shooting): draw N and keep the best. The
  "greedily pick the best of N" control.

Each control is read out under **two objectives**, because they answer different questions:

* ``*_peak`` — best reward over ANY frame of the rollout (MPC-style, matching the tree's
  best-prefix back-up), and
* ``*_last`` — reward of the **final** state only ("where did you end up"), the natural
  objective when the task is to *reach and hold* a configuration rather than to pass
  through a good one. A plan that shoves the T across the centre and out again scores
  well on ``peak`` and badly on ``last``.

Both are computed from the SAME rollouts and the same reward evaluation, so the ``last``
readout costs no extra decodes (the reward is already evaluated on every frame).

**Caveat (edge_mode).** Both baselines build edges with the ``imagine`` sampler (they respect
``ctx_noise`` / ``action_temp`` / ``action_prior`` / ``state_prior`` but not ``edge_mode`` or
the autoregressive-only ``action_noise``). So under ``edge_mode='two_stage'`` / ``'autoregressive'``
the baselines still *imagine* each edge, and ``g_random_peak`` / ``g_greedy_peak`` then also reflect the
tree's edge-sampler choice, not search alone. To isolate search under a non-default sampler,
the baselines would need to build edges with that same sampler.

Outputs (per tree): the baseline readouts ``random_peak`` / ``greedy_peak`` /
``random_last`` / ``greedy_last``, the peak gains ``g_random_peak`` / ``g_greedy_peak`` (vs
``tree_peak``), the terminal gains ``g_random_last`` / ``g_greedy_last`` (vs ``tree_last``),
and ``delta_over_root``. The start/context frame is excluded from every readout, as in
``tree_peak``.

The tree-side numbers these are scored against — ``tree_peak`` and ``tree_last`` — come from
:func:`plan_peak` / :func:`plan_last`, which read them off the planner's returned ``plan_z``.
So the whole comparison for one tree is::

    out = MCTS(denoiser, reward_fn, cfg, seed=s).plan(ctx_z, ctx_a)
    row = compute_baselines(denoiser, reward_fn, ctx_z, ctx_a, cfg,
                            tree_peak=plan_peak(reward_fn, out),
                            tree_last=plan_last(reward_fn, out))

**Compare like with like.** A ``*_last`` baseline is only meaningful against the plan's
*final* state (``tree_last``), never against ``tree_peak`` — mixing the two would score the
planner on its best moment and the baseline on its last one.
"""
from __future__ import annotations

from typing import Tuple

import torch

from . import rollout as R


@torch.no_grad()
def _rollout_peak_and_last(denoiser, reward_fn, ctx_z, ctx_a, *, depth, H, B, K, ctx_noise,
                           ctx_noise_honest, action_temp, action_prior, state_prior, dtype,
                           max_ctx, gen) -> Tuple[float, float]:
    """Run ``B`` no-search depth-deep rollouts; return ``(peak, last)``.

    ``B`` independent depth-deep trajectories in parallel: ``depth`` edges of ``H`` frames,
    each re-conditioned on its OWN imagination after every edge (states/actions appended,
    truncated to ``max_ctx`` — mirroring MCTS ``_advance_ctx``). Same lookahead
    (``depth*H``) and same edge-by-edge construction as the plan; no selection happens
    between edges, so the only thing the tree adds on top is the *search*.

    Two readouts off the same rollouts and the same reward evaluation:
      ``peak`` — best reward over ANY frame, any edge, any of the ``B`` rollouts;
      ``last`` — best reward over the ``B`` **final** states (last frame of the last edge).

    ``B=1`` is the undirected "random" control; ``B=n_random`` is the "greedy" random-shooting
    control. ``peak >= last`` always holds, since the terminal frames are a subset of all
    frames.
    """
    cz = ctx_z.expand(B, -1, -1, -1).contiguous()    # (B, Tc, N, D)
    ca = ctx_a.expand(B, -1, -1).contiguous()        # (B, Tc, n_act)
    peak, last = float("-inf"), float("nan")
    for _ in range(max(int(depth), 1)):
        z, a = R.imagine(denoiser, cz, ca, H, B=B, K=K, ctx_noise=ctx_noise,
                         ctx_noise_honest=ctx_noise_honest, action_temp=action_temp,
                         action_prior=action_prior, state_prior=state_prior,
                         dtype=dtype, generator=gen)                              # (B,H,N,D)
        r = reward_fn(z)                                           # (B,H) — the only decode
        peak = max(peak, float(r.max().item()))                    # excludes the start frame, like tree_peak
        last = float(r[:, -1].max().item())                        # this edge's terminal states; final edge wins
        cz = torch.cat([cz, z], dim=1)[:, -max_ctx:].contiguous()  # advance ctx (== _advance_ctx)
        ca = torch.cat([ca, a], dim=1)[:, -max_ctx:].contiguous()
    return peak, last


# --- the four named controls -------------------------------------------------------
# Thin readouts over :func:`_rollout_peak_and_last`. Each call re-runs its own rollouts,
# so use them when you want ONE number; ``compute_baselines`` instead takes both readouts
# from a single run per control (half the model calls).

def _rollout_random_peak(denoiser, reward_fn, ctx_z, ctx_a, **kw) -> float:
    """Best reward over ANY frame of ONE no-search depth-deep rollout (undirected control)."""
    return _rollout_peak_and_last(denoiser, reward_fn, ctx_z, ctx_a, B=1, **kw)[0]


def _rollout_random_last(denoiser, reward_fn, ctx_z, ctx_a, **kw) -> float:
    """Reward of the FINAL state of ONE no-search depth-deep rollout (undirected control)."""
    return _rollout_peak_and_last(denoiser, reward_fn, ctx_z, ctx_a, B=1, **kw)[1]


def _rollout_greedy_peak(denoiser, reward_fn, ctx_z, ctx_a, **kw) -> float:
    """Best reward over ANY frame of ``B`` no-search depth-deep rollouts (random shooting)."""
    return _rollout_peak_and_last(denoiser, reward_fn, ctx_z, ctx_a, **kw)[0]


def _rollout_greedy_last(denoiser, reward_fn, ctx_z, ctx_a, **kw) -> float:
    """Best reward over the FINAL states of ``B`` no-search depth-deep rollouts."""
    return _rollout_peak_and_last(denoiser, reward_fn, ctx_z, ctx_a, **kw)[1]


# --- the tree side of the comparison -----------------------------------------------
# The two numbers the baselines above are scored against, read straight off the planner's
# returned ``plan_z`` (the state plan root -> best node, shape ``(n_edges, H, N, D)``).
# Both return NaN when the planner produced no plan, so a degenerate tree yields NaN gains
# rather than raising.

@torch.no_grad()
def plan_peak(reward_fn, out: dict) -> float:
    """Best single-frame reward over ANY frame of the returned plan.

    The tree-side counterpart of the ``*_peak`` baselines. ``out`` is the dict returned by
    ``MCTS.plan``. Excludes the start/context frame (it is not part of ``plan_z``), matching
    :func:`_rollout_peak_and_last`.
    """
    pz = out.get("plan_z")                       # (n_edges, H, N, D)
    if pz is None:
        return float("nan")
    N, D = pz.shape[-2], pz.shape[-1]
    return float(reward_fn(pz.reshape(-1, N, D)).max().item())


@torch.no_grad()
def plan_last(reward_fn, out: dict) -> float:
    """Reward of the plan's FINAL state — the last frame of its last edge.

    The terminal-objective counterpart of :func:`plan_peak` ("where the plan ends up" vs
    "the best moment it passes through"), to be compared against the ``*_last`` baselines.
    """
    pz = out.get("plan_z")                       # (n_edges, H, N, D)
    if pz is None:
        return float("nan")
    return float(reward_fn(pz[-1, -1:]).reshape(-1)[0].item())


@torch.no_grad()
def compute_baselines(denoiser, reward_fn, ctx_z, ctx_a, cfg, *, tree_peak: float,
                      tree_last: float, n_random: int = 16, seed: int = 12345,
                      random: bool = True, greedy: bool = True) -> dict:
    """Return the baseline readouts and planning gains for one tree.

    ``tree_peak`` is the planner's achieved peak reward and ``tree_last`` the reward of its
    plan's FINAL state — get both from :func:`plan_peak` / :func:`plan_last`. Both are
    **required**: each gain is only meaningful against its own objective (``g_*`` off
    ``tree_peak``, ``g_*_last`` off ``tree_last``), so there is no sensible default for
    either.

    Two matched-lookahead, no-search controls (both on by default):
      ``random`` -> a single depth-deep re-conditioned rollout (undirected);
      ``greedy`` -> best of ``n_random`` depth-deep rollouts (random shooting).
    Each control is run ONCE and read out twice (``*_peak`` and ``*_last``, see the module
    docstring), so the terminal-objective baselines are free.

    ``greedy`` costs ~``n_random``x more decodes than ``random`` — disable it if you only want
    the cheap single-rollout comparison.
    """
    dev = ctx_z.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    root_reward = float(reward_fn(ctx_z[:, -1:]).reshape(-1)[0].item())
    kw = dict(depth=cfg.max_depth, H=cfg.horizon, K=cfg.K_steps, ctx_noise=cfg.ctx_noise,
              ctx_noise_honest=cfg.ctx_noise_honest, action_temp=cfg.action_temp,
              action_prior=cfg.action_prior, state_prior=cfg.state_prior,
              dtype=cfg.dtype, max_ctx=cfg.max_ctx, gen=gen)
    out = {"root_reward": root_reward, "delta_over_root": tree_peak - root_reward}

    if random:  # one depth-deep re-conditioned rollout (undirected)
        peak, last = _rollout_peak_and_last(denoiser, reward_fn, ctx_z, ctx_a, B=1, **kw)
        out.update(random_peak=peak, g_random_peak=tree_peak - peak,
                   random_last=last, g_random_last=tree_last - last)

    if greedy:  # best of n_random depth-deep rollouts (random shooting)
        peak, last = _rollout_peak_and_last(denoiser, reward_fn, ctx_z, ctx_a, B=n_random, **kw)
        out.update(greedy_peak=peak, g_greedy_peak=tree_peak - peak,
                   greedy_last=last, g_greedy_last=tree_last - last)

    return out
