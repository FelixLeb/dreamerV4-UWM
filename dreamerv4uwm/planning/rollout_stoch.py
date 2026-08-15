"""Stochastic (churned) edge samplers — Family A of ``edge-diversity.tex``.

Why
---
:mod:`rollout` integrates the probability-flow ODE with a **deterministic** Euler step, so
for a fixed context the sampler is a pure function of the prior draw,
``Phi_ctx : eps -> a_{1:H}``. All sibling diversity is inherited from ``eps`` through
``Phi_ctx``, and ODE samplers are empirically *contractive*: they concentrate samples
locally even when the posterior they are meant to represent is wide (the "collapse errors"
of arXiv:2508.16154; the "striking lack of set-level diversity" of arXiv:2510.09060).

The probability-flow ODE and the reverse SDE share marginals, but only the SDE actually
*samples* them. This module re-introduces that stochasticity. Unlike ``ctx_noise`` /
``action_temp``, it does **not** push the model off-distribution: the target distribution is
unchanged, we just stop collapsing onto one of its modes.

The mechanism: noise refreshment
--------------------------------
On the linear interpolant ``x_tau = (1-tau)*eps + tau*x_1``, a point ``x`` at cleanness
``tau`` together with the model's clean estimate ``x_hat`` implies a noise component
``eps_hat = (x - tau*x_hat) / (1-tau)``. We partially resample it::

    eps_new = sqrt(1 - churn^2) * eps_hat  +  churn * xi,      xi ~ N(0, I)
    x       = (1-tau) * eps_new + tau * x_hat

If ``eps_hat`` and ``xi`` are standard normal and independent, so is ``eps_new`` — the
marginal is preserved while the *particular* noise realisation is partially forgotten. This
is a restart / churn step (Karras et al.'s ``S_churn``, Xu et al.'s restart sampling)
expressed so that it:

* stays exactly on the interpolant manifold (no off-path excursion);
* keeps the ``K``-step schedule and therefore **exactly the same number of denoiser
  forwards** as :mod:`rollout` — the compute budget is unchanged, which is what makes this a
  clean ablation against the deterministic sampler;
* vanishes automatically as ``tau -> 1`` (the ``(1-tau)`` factor), so the final steps are
  un-churned and the sample lands back on the data manifold;
* reduces to the **identity** at ``churn=0``.

``churn=0`` reproduces :mod:`rollout` bit-for-bit, RNG stream included (no random numbers
are drawn when churn is inactive). That is asserted by ``notebooks/E0-churn-diagnostic``.

Orthogonal churn
----------------
``churn_orthogonal=True`` projects ``xi`` orthogonal to the local flow direction
``v = x_hat - x`` before mixing. The idea (arXiv:2510.09060) is that a perturbation
geometrically decoupled from the mode-seeking direction adds spread without fighting the
quality-seeking dynamics, so diversity rises at lower fidelity cost.

What to sweep
-------------
``churn`` in ``[0, 1]`` is the one knob that matters; ``churn_targets`` isolates whether the
action-side or state-side prior is what actually differentiates siblings.

Both samplers here inherit :mod:`rollout`'s unconditional mode (``ctx_z=ctx_a=None``, plus a
``device=``), which gives the churn sweep its ceiling: the spread with no context at all.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ..sampling_new import make_is_horizon, _quantize_tau_to_idx
from .rollout import (_dims, _autocast, _expand_ctx, _build_context, _scaled_noise,
                      _ctx_len, _resolve_device)


# ---------------------------------------------------------------------------
# the churn step
# ---------------------------------------------------------------------------

def _proj_out(xi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Remove from ``xi`` its component along ``v``, per batch element.

    Both are ``(B, ...)``; the inner products are taken over every non-batch dim, so each
    rollout gets its own projection against its own local flow direction."""
    dims = tuple(range(1, xi.dim()))
    vv = (v * v).sum(dim=dims, keepdim=True).clamp_min(1e-12)
    coef = (xi * v).sum(dim=dims, keepdim=True) / vv
    return xi - coef * v


def _refresh(x: torch.Tensor, x_hat: torch.Tensor, tau: float, churn: float, *,
             orthogonal: bool, generator: Optional[torch.Generator]) -> torch.Tensor:
    """Partially resample the noise component of ``x`` at cleanness ``tau``.

    ``x`` is the current iterate, ``x_hat`` the model's clean-sample estimate. Returns a
    point at the *same* cleanness whose implied noise has been mixed with a fresh draw by
    ``churn`` in ``[0, 1]``. ``churn=0`` is the exact identity, ``churn=1`` fully forgets the
    noise realisation drawn so far.
    """
    keep = max(1.0 - tau, 1e-5)                      # the (1 - tau) factor, guarded
    eps_hat = (x - tau * x_hat) / keep               # noise currently implied by (x, x_hat)
    xi = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    if orthogonal:
        xi = _proj_out(xi, x_hat - x)                # decouple from the flow direction
    eps_new = math.sqrt(max(1.0 - churn * churn, 0.0)) * eps_hat + churn * xi
    return keep * eps_new + tau * x_hat


def _churn_at(tau: float, churn: float, churn_tmax: float) -> float:
    """Churn amount in force at cleanness ``tau`` (0 disables the step entirely)."""
    return churn if (churn > 0.0 and tau <= churn_tmax) else 0.0


# ---------------------------------------------------------------------------
# joint imagination with churn  :  the stochastic edge generator
# ---------------------------------------------------------------------------

@torch.no_grad()
def imagine_stoch(
    denoiser,
    ctx_z: Optional[torch.Tensor],  # (1|B, Tc, N_lat, D_lat), or None
    ctx_a: Optional[torch.Tensor],  # (1|B, Tc, n_act),        or None
    H: int,
    B: int = 1,
    K: int = 12,
    *,
    churn: float = 0.0,           # in [0,1]: how much of the noise to resample each step
    churn_tmax: float = 1.0,      # only churn while cleanness <= this (1.0 = always)
    churn_orthogonal: bool = False,   # project the fresh noise orthogonal to the flow
    churn_targets: str = "both",  # "both" | "action" | "state"
    ctx_noise: float = 0.0,
    ctx_noise_honest: bool = True,
    action_temp: float = 1.0,
    action_prior: str = "normal",
    state_prior: str = "normal",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`rollout.imagine` with noise refreshment. Same signature plus the churn knobs,
    same ``K`` denoiser forwards, same returns ``(z_hor (B,H,N,D), a_hor (B,H,n_act))``.

    At ``churn=0`` this is :func:`rollout.imagine` exactly — same arithmetic and same RNG
    stream, since no extra draws are taken.

    ``churn_targets`` selects which modality gets refreshed: ``"action"`` alone answers
    "does action-side stochasticity drive sibling diversity?", ``"state"`` alone the
    converse, ``"both"`` (default) is the full sampler.

    ``ctx_z=ctx_a=None`` samples unconditionally (``Tc=0``), as in :func:`rollout.imagine`
    — the churn measurement without any conditioning, i.e. how much spread the sampler has
    when nothing is pinning it down. Pass ``device=`` since there is no context to read it
    from (defaults to the denoiser's).
    """
    if churn_targets not in ("both", "action", "state"):
        raise ValueError(f"unknown churn_targets={churn_targets!r} "
                         "(expected 'both' | 'action' | 'state')")
    if not 0.0 <= churn <= 1.0:
        raise ValueError(f"churn={churn} must lie in [0, 1]")

    N, N_lat, D_lat, n_act = _dims(denoiser)
    device = _resolve_device(denoiser, ctx_z, device=device)
    ctx_z, ctx_a = _expand_ctx(ctx_z, ctx_a, B)
    Tc = _ctx_len(ctx_z)
    T = Tc + H

    z_ctx, a_ctx, ctx_obs_idx = _build_context(
        ctx_z, ctx_a, ctx_noise, ctx_noise_honest, N, generator)

    z = torch.empty(B, T, N_lat, D_lat, device=device)
    a = torch.empty(B, T, n_act, device=device)
    if Tc:
        z[:, :Tc], a[:, :Tc] = z_ctx, a_ctx
    z[:, Tc:] = _scaled_noise((B, H, N_lat, D_lat), scale=1.0, dist=state_prior,
                              device=device, generator=generator)
    a[:, Tc:] = _scaled_noise((B, H, n_act), scale=action_temp, dist=action_prior,
                              device=device, generator=generator)

    step_idx = torch.zeros((B, T), dtype=torch.long, device=device)
    is_hor = make_is_horizon(T, ctx_len=Tc, device=device)
    obs_sigma = torch.empty((B, T), dtype=torch.long, device=device)
    act_sigma = torch.empty((B, T), dtype=torch.long, device=device)

    do_a = churn_targets in ("both", "action")
    do_z = churn_targets in ("both", "state")

    cur, dt = 0.0, 1.0 / K
    for _ in range(K):
        cur_idx = _quantize_tau_to_idx(cur, N)
        obs_sigma[:, :Tc] = ctx_obs_idx
        act_sigma[:, :Tc] = N - 1
        obs_sigma[:, Tc:] = cur_idx
        act_sigma[:, Tc:] = cur_idx
        with _autocast(device, dtype):
            z_hat, a_hat, _ = denoiser(
                noisy_act=a, noisy_obs=z,
                obs_sigma_idx=obs_sigma, obs_step_idx=step_idx,
                act_sigma_idx=act_sigma, act_step_idx=step_idx, is_horizon=is_hor)
        z_hat = z_hat.float()
        a_hat = a_hat.squeeze(-2).float()
        denom = max(1.0 - cur, 1e-5)
        z[:, Tc:] = z[:, Tc:] + (z_hat[:, Tc:] - z[:, Tc:]) / denom * dt
        a[:, Tc:] = a[:, Tc:] + (a_hat[:, Tc:] - a[:, Tc:]) / denom * dt
        cur += dt

        eta = _churn_at(cur, churn, churn_tmax)      # refresh at the NEW cleanness
        if eta > 0.0:
            if do_z:
                z[:, Tc:] = _refresh(z[:, Tc:], z_hat[:, Tc:], cur, eta,
                                     orthogonal=churn_orthogonal, generator=generator)
            if do_a:
                a[:, Tc:] = _refresh(a[:, Tc:], a_hat[:, Tc:], cur, eta,
                                     orthogonal=churn_orthogonal, generator=generator)
    return z[:, Tc:], a[:, Tc:]


# ---------------------------------------------------------------------------
# policy mode with churn
# ---------------------------------------------------------------------------

@torch.no_grad()
def policy_stoch(
    denoiser,
    ctx_z: Optional[torch.Tensor],
    ctx_a: Optional[torch.Tensor],
    H: int,
    B: int = 1,
    K: int = 12,
    *,
    churn: float = 0.0,
    churn_tmax: float = 1.0,
    churn_orthogonal: bool = False,
    ctx_noise: float = 0.0,
    ctx_noise_honest: bool = True,
    action_temp: float = 1.0,
    action_prior: str = "normal",
    state_prior: str = "normal",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """:func:`rollout.policy` with noise refreshment on the action. Returns ``(B,H,n_act)``.

    Only the action is churned here — the horizon state is held at pure noise and never
    integrated, so it has no clean estimate to refresh against. ``churn=0`` reproduces
    :func:`rollout.policy` exactly. Needed by ``two_stage`` / ``autoregressive`` edge modes;
    the default ``imagine`` edge mode only needs :func:`imagine_stoch`.

    ``ctx_z=ctx_a=None`` samples the unconditional action prior ``p(a_{1:H})``; pass
    ``device=`` (defaults to the denoiser's).
    """
    if not 0.0 <= churn <= 1.0:
        raise ValueError(f"churn={churn} must lie in [0, 1]")

    N, N_lat, D_lat, n_act = _dims(denoiser)
    device = _resolve_device(denoiser, ctx_z, device=device)
    ctx_z, ctx_a = _expand_ctx(ctx_z, ctx_a, B)
    Tc = _ctx_len(ctx_z)
    T = Tc + H

    z_ctx, a_ctx, ctx_obs_idx = _build_context(
        ctx_z, ctx_a, ctx_noise, ctx_noise_honest, N, generator)

    z = torch.empty(B, T, N_lat, D_lat, device=device)
    a = torch.empty(B, T, n_act, device=device)
    if Tc:
        z[:, :Tc], a[:, :Tc] = z_ctx, a_ctx
    z[:, Tc:] = _scaled_noise((B, H, N_lat, D_lat), scale=1.0, dist=state_prior,
                              device=device, generator=generator)
    a[:, Tc:] = _scaled_noise((B, H, n_act), scale=action_temp, dist=action_prior,
                              device=device, generator=generator)

    step_idx = torch.zeros((B, T), dtype=torch.long, device=device)
    is_hor = make_is_horizon(T, ctx_len=Tc, device=device)
    obs_sigma = torch.empty((B, T), dtype=torch.long, device=device)
    act_sigma = torch.empty((B, T), dtype=torch.long, device=device)

    cur, dt = 0.0, 1.0 / K
    for _ in range(K):
        obs_sigma[:, :Tc] = ctx_obs_idx
        act_sigma[:, :Tc] = N - 1
        obs_sigma[:, Tc:] = 0                                  # horizon state: full noise
        act_sigma[:, Tc:] = _quantize_tau_to_idx(cur, N)
        with _autocast(device, dtype):
            _, a_hat, _ = denoiser(
                noisy_act=a, noisy_obs=z,
                obs_sigma_idx=obs_sigma, obs_step_idx=step_idx,
                act_sigma_idx=act_sigma, act_step_idx=step_idx, is_horizon=is_hor)
        a_hat = a_hat.squeeze(-2).float()
        denom = max(1.0 - cur, 1e-5)
        a[:, Tc:] = a[:, Tc:] + (a_hat[:, Tc:] - a[:, Tc:]) / denom * dt
        cur += dt

        eta = _churn_at(cur, churn, churn_tmax)
        if eta > 0.0:
            a[:, Tc:] = _refresh(a[:, Tc:], a_hat[:, Tc:], cur, eta,
                                 orthogonal=churn_orthogonal, generator=generator)
    return a[:, Tc:]


# ---------------------------------------------------------------------------
# planner adapter
# ---------------------------------------------------------------------------

def make_edge_sampler(denoiser, cfg, *, churn: float, churn_tmax: float = 1.0,
                      churn_orthogonal: bool = False, churn_targets: str = "both",
                      generator: Optional[torch.Generator] = None):
    """Build an ``edge_sampler`` closure for ``MCTS(edge_sampler=...)``.

    Lets a full tree run on the churned sampler without touching :mod:`mcts`. ``cfg`` is a
    :class:`~.mcts.PlanConfig`, whose diversity knobs are forwarded unchanged, so a churn
    sweep is directly comparable against the same config at ``churn=0``.

    Note the planner counts a custom sampler as one primitive call per edge, which is
    correct here: churn adds no denoiser forwards, so ``n_denoiser_calls`` stays comparable
    with the built-in ``imagine`` edge mode.
    """
    def sample(ctx_z, ctx_a, H, B):
        return imagine_stoch(denoiser, ctx_z, ctx_a, H, B=B, K=cfg.K_steps,
                             churn=churn, churn_tmax=churn_tmax,
                             churn_orthogonal=churn_orthogonal, churn_targets=churn_targets,
                             ctx_noise=cfg.ctx_noise, ctx_noise_honest=cfg.ctx_noise_honest,
                             action_temp=cfg.action_temp, action_prior=cfg.action_prior,
                             state_prior=cfg.state_prior, dtype=cfg.dtype,
                             generator=generator)
    return sample
