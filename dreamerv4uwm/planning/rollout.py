"""Batched rollout primitives for planning on top of the UWM denoiser.

These wrap a single, proven flow-matching idea — re-attach a (optionally
noised) clean context, then Euler-integrate the horizon from its noise prior to
clean — into the three operating points a planner needs:

* :func:`policy`     — ``p(a_{t:t+H} | o_{<=t})``. Horizon **state** held at pure
  noise (never integrated), horizon **action** integrated. The "policy mode".
* :func:`transition` — world-model step. Horizon **action** given & held clean,
  horizon **state** integrated. ``s, a -> s'``.
* :func:`imagine`    — **joint** short rollout ``(a, o') ~ p(. | o_{<=t})``: both
  action and state integrated together. This is the MCTS *edge generator*: one 
  denoiser pass per Euler step yields a full action+state rollout.

All three share the same diversity knobs, which is the whole point of keeping
them in one place — the policy-diversity experiments and the planner pull the
same levers:

* ``ctx_noise`` in [0, 1] — noise level mixed into the **observation** context
  (``o <- (1-tau)*eps + tau*o``, ``tau = 1 - ctx_noise``). 0 = clean.
* ``ctx_noise_honest`` — if True, the context's ``sigma_idx`` is set to the
  *true* (noised) cleanness so the model knows the state is uncertain (a
  principled widening of the posterior). If False, the context is corrupted but
  announced clean (mismatched / OOD — a contrast condition).
* ``action_temp`` — std of the action noise prior. Flow matching transports the
  prior ``N(0, I)`` to ``p(a|o)``; scaling the prior std is a temperature-like
  control. ``!= 1`` is mildly OOD (the model only ever saw std-1 priors).
* ``action_prior`` — shape of that action noise prior: ``"normal"`` (default,
  ``action_temp * N(0, I)``) or ``"uniform"`` (``U(-a, a)`` with ``a = action_temp *
  sqrt(3)``, i.e. std-matched to the normal case). Uniform is a flat, bounded prior —
  more OOD than the Gaussian one, an alternative lever on action diversity. Only affects
  action-sampling rollouts (``policy`` / ``imagine``), not ``transition`` (given actions).
* ``K`` — number of Euler integration steps.

Convention reminder (matches ``sampling_new.py``): ``n`` is *noise level*
(0 = clean, 1 = pure noise); ``tau = 1 - n`` is *cleanness* (the index the
embeddings expect). Args here use ``ctx_noise`` (an ``n``) and integrate in
``tau``.

Everything runs under ``bf16`` autocast by default — fp32 forwards OOM at the
batch sizes a tree search needs and are ~3.5x slower on this checkpoint, while
the bf16 world-model reconstruction error is within noise of fp32.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional, Tuple

import torch

from ..sampling_new import make_is_horizon, _quantize_tau_to_idx


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _dims(denoiser):
    d = denoiser.cfg.denoiser
    return (int(d.num_noise_levels), int(d.num_latent_tokens),
            int(d.latent_dim), int(d.n_actions))


def _autocast(device, dtype):
    if dtype is None or (isinstance(device, torch.device) and device.type != "cuda"):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _expand_ctx(ctx_z, ctx_a, B):
    """Broadcast a (1, Tc, ...) context to (B, Tc, ...) (no-op if already B)."""
    if ctx_z.shape[0] != B:
        ctx_z = ctx_z.expand(B, -1, -1, -1).contiguous()
        ctx_a = ctx_a.expand(B, -1, -1).contiguous()
    return ctx_z, ctx_a


def _build_context(ctx_z, ctx_a, ctx_noise, ctx_noise_honest, N, gen):
    """Return (z_ctx, a_ctx, ctx_obs_sigma_idx) with optional honest/mismatched
    observation-noise injection. Actions are always kept clean."""
    tau_ctx = 1.0 - float(ctx_noise)
    if ctx_noise > 0.0:
        eps = torch.randn(ctx_z.shape, device=ctx_z.device, dtype=ctx_z.dtype, generator=gen)
        z_ctx = (1.0 - tau_ctx) * eps + tau_ctx * ctx_z
    else:
        z_ctx = ctx_z
    obs_idx = _quantize_tau_to_idx(tau_ctx if ctx_noise_honest else 1.0, N)
    return z_ctx, ctx_a, obs_idx


def _action_prior(B, H, n_act, *, action_temp, action_prior, device, generator):
    """Sample the horizon **action** noise prior the flow integrates from.

    * ``"normal"``  — ``action_temp * N(0, I)`` (std ``action_temp``); the prior the
      denoiser was trained with. ``action_temp != 1`` is mildly OOD.
    * ``"uniform"`` — ``U(-a, a)`` with ``a = action_temp * sqrt(3)``, so its std still
      equals ``action_temp`` (temperature-matched to the normal case — a flat, bounded
      prior instead of a Gaussian one). More OOD than the Gaussian prior (the flow only
      ever saw ``N(0, I)``), but a different way to inject action diversity.
    """
    if action_prior == "normal":
        return action_temp * torch.randn(B, H, n_act, device=device, generator=generator)
    if action_prior == "uniform":
        a = action_temp * math.sqrt(3.0)
        u = torch.rand(B, H, n_act, device=device, generator=generator)   # U(0, 1)
        return a * (2.0 * u - 1.0)                                         # U(-a, a), std = action_temp
    raise ValueError(f"unknown action_prior={action_prior!r} (expected 'normal' or 'uniform')")


# ---------------------------------------------------------------------------
# policy mode :  p(a_{t:t+H} | o_{<=t})
# ---------------------------------------------------------------------------

@torch.no_grad()
def policy(
    denoiser,
    ctx_z: torch.Tensor,          # (1|B, Tc, N_lat, D_lat) clean obs context
    ctx_a: torch.Tensor,          # (1|B, Tc, n_act)        clean action context
    H: int,                       # horizon length (number of steps to roll out)
    B: int = 1,                   # batch size (number of rollouts to sample)
    K: int = 12,                  # number of Euler integration steps
    *,                            # keyword-only args
    ctx_noise: float = 0.0,
    ctx_noise_honest: bool = True,
    action_temp: float = 1.0,
    action_prior: str = "normal",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample ``B`` action rollouts of length ``H`` from the policy marginal
    ``p(a_{t:t+H} | o_{<=t})``. The horizon **state** is held at pure noise
    throughout (we don't claim to know the future observations), so only the
    actions are meaningful. Returns ``a_hor`` of shape ``(B, H, n_act)``.
    """
    N, N_lat, D_lat, n_act = _dims(denoiser)
    device = ctx_z.device
    ctx_z, ctx_a = _expand_ctx(ctx_z, ctx_a, B)
    Tc, T = ctx_z.shape[1], ctx_z.shape[1] + H

    z_ctx, a_ctx, ctx_obs_idx = _build_context(
        ctx_z, ctx_a, ctx_noise, ctx_noise_honest, N, generator)

    z = torch.empty(B, T, N_lat, D_lat, device=device)
    a = torch.empty(B, T, n_act, device=device)
    z[:, :Tc], a[:, :Tc] = z_ctx, a_ctx
    z[:, Tc:] = torch.randn(B, H, N_lat, D_lat, device=device, generator=generator)  # held at noise
    a[:, Tc:] = _action_prior(B, H, n_act, action_temp=action_temp, action_prior=action_prior,
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
    return a[:, Tc:]


# ---------------------------------------------------------------------------
# world-model transition :  s, a -> s'
# ---------------------------------------------------------------------------

@torch.no_grad()
def transition(
    denoiser,
    ctx_z: torch.Tensor,          # (1|B, Tc, N_lat, D_lat)
    ctx_a: torch.Tensor,          # (1|B, Tc, n_act)
    actions: torch.Tensor,        # (B, H, n_act) clean action sequence to apply
    K: int = 12,
    *,
    ctx_noise: float = 0.0,
    ctx_noise_honest: bool = True,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Roll the world model forward under a **given** clean action sequence.
    Horizon actions are held clean; horizon state integrates from noise to
    clean. Returns predicted ``z_hor`` of shape ``(B, H, N_lat, D_lat)``.
    """
    N, N_lat, D_lat, n_act = _dims(denoiser)
    device = ctx_z.device
    B, H = actions.shape[0], actions.shape[1]
    ctx_z, ctx_a = _expand_ctx(ctx_z, ctx_a, B)
    Tc, T = ctx_z.shape[1], ctx_z.shape[1] + H

    z_ctx, a_ctx, ctx_obs_idx = _build_context(
        ctx_z, ctx_a, ctx_noise, ctx_noise_honest, N, generator)

    z = torch.empty(B, T, N_lat, D_lat, device=device)
    a = torch.empty(B, T, n_act, device=device)
    z[:, :Tc], a[:, :Tc] = z_ctx, a_ctx
    z[:, Tc:] = torch.randn(B, H, N_lat, D_lat, device=device, generator=generator)
    a[:, Tc:] = actions.to(device=device, dtype=a.dtype)      # given, held clean

    step_idx = torch.zeros((B, T), dtype=torch.long, device=device)
    is_hor = make_is_horizon(T, ctx_len=Tc, device=device)
    obs_sigma = torch.empty((B, T), dtype=torch.long, device=device)
    act_sigma = torch.empty((B, T), dtype=torch.long, device=device)

    cur, dt = 0.0, 1.0 / K
    for _ in range(K):
        obs_sigma[:, :Tc] = ctx_obs_idx
        obs_sigma[:, Tc:] = _quantize_tau_to_idx(cur, N)
        act_sigma[:] = N - 1                                  # all actions clean
        with _autocast(device, dtype):
            z_hat, _, _ = denoiser(
                noisy_act=a, noisy_obs=z,
                obs_sigma_idx=obs_sigma, obs_step_idx=step_idx,
                act_sigma_idx=act_sigma, act_step_idx=step_idx, is_horizon=is_hor)
        z_hat = z_hat.float()
        denom = max(1.0 - cur, 1e-5)
        z[:, Tc:] = z[:, Tc:] + (z_hat[:, Tc:] - z[:, Tc:]) / denom * dt
        cur += dt
    return z[:, Tc:]


# ---------------------------------------------------------------------------
# joint imagination :  (a, o') ~ p(. | o)      <-- the MCTS edge generator
# ---------------------------------------------------------------------------

@torch.no_grad()
def imagine(
    denoiser,
    ctx_z: torch.Tensor,          # (1|B, Tc, N_lat, D_lat)
    ctx_a: torch.Tensor,          # (1|B, Tc, n_act)
    H: int,
    B: int = 1,
    K: int = 12,
    *,
    ctx_noise: float = 0.0,
    ctx_noise_honest: bool = True,
    action_temp: float = 1.0,
    action_prior: str = "normal",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample ``B`` **joint** short rollouts of length ``H`` from the model:
    actions and the states they induce are denoised together, both from their
    noise priors to clean, conditioned on the (optionally noised) context.

    This is the planner's edge generator ``pi_prior``: a single object that is
    simultaneously a stochastic policy and its own forward model. Returns
    ``(z_hor, a_hor)`` with shapes ``(B, H, N_lat, D_lat)`` and ``(B, H, n_act)``.
    """
    N, N_lat, D_lat, n_act = _dims(denoiser)
    device = ctx_z.device
    ctx_z, ctx_a = _expand_ctx(ctx_z, ctx_a, B)
    Tc, T = ctx_z.shape[1], ctx_z.shape[1] + H

    z_ctx, a_ctx, ctx_obs_idx = _build_context(
        ctx_z, ctx_a, ctx_noise, ctx_noise_honest, N, generator)

    z = torch.empty(B, T, N_lat, D_lat, device=device)
    a = torch.empty(B, T, n_act, device=device)
    z[:, :Tc], a[:, :Tc] = z_ctx, a_ctx
    z[:, Tc:] = torch.randn(B, H, N_lat, D_lat, device=device, generator=generator)
    a[:, Tc:] = _action_prior(B, H, n_act, action_temp=action_temp, action_prior=action_prior,
                              device=device, generator=generator)

    step_idx = torch.zeros((B, T), dtype=torch.long, device=device)
    is_hor = make_is_horizon(T, ctx_len=Tc, device=device)
    obs_sigma = torch.empty((B, T), dtype=torch.long, device=device)
    act_sigma = torch.empty((B, T), dtype=torch.long, device=device)

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
    return z[:, Tc:], a[:, Tc:]