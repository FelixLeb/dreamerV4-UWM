"""Batched rollout primitives for planning on top of the UWM denoiser.

These wrap a single, proven flow-matching idea — re-attach a (optionally
noised) clean context, then Euler-integrate the horizon from its noise prior to
clean — into the three operating points a planner needs:

* :func:`policy`     — ``p(a_{t:t+H} | o_{<=t})``. Horizon **state** held at pure
  noise (never integrated), horizon **action** integrated. The "policy mode".
* :func:`transition` — world-model step. Horizon **action** given & held clean,
  horizon **state** integrated. ``s, a -> s'``.
* :func:`imagine`    — **joint** short rollout ``(a, o') ~ p(. | o_{<=t})``: both
  action and state integrated together. This is the default MCTS *edge generator*: one
  denoiser pass per Euler step yields a full action+state rollout.
* :func:`autoregressive` — the edge **one frame at a time**: ``policy`` (H=1) →
  optional added action noise → ``transition`` (H=1), re-conditioning on the realized
  state each step. Causally consistent (each action sees the true previous state), at
  ``2*H`` primitive calls per edge.

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
  more OOD than the Gaussian one, an alternative lever on action diversity. Affects
  action-sampling rollouts (``policy`` / ``imagine`` / ``autoregressive``), not
  ``transition`` (given actions).
* ``state_prior`` — the same shape choice for the **observation** noise prior the
  horizon state starts from: ``"normal"`` (default, ``N(0, I)`` — what the denoiser was
  trained to transport) or ``"uniform"`` (``U(-sqrt(3), sqrt(3))``, std-matched). There is
  no ``state_temp`` counterpart to ``action_temp``: the state prior is always unit-scale,
  so this knob changes the prior's *shape* only. It affects every primitive — in
  ``transition`` / ``imagine`` the state prior is integrated to clean, and in ``policy``
  the horizon state is *held* at this prior, so it is what the model reads as "the future
  is unknown". Note the latent is high-dimensional (``N_lat*D_lat``, 8192 here), where a
  Gaussian and a std-matched uniform have near-identical norm concentration; expect a
  weaker effect than the action-side counterpart, and read a null result as such.
* ``action_noise`` / ``action_noise_dist`` — (``autoregressive`` only) magnitude and shape
  (``"normal"`` / ``"uniform"``) of extra noise **added** to each policy action, on top of
  the policy's own stochasticity. ``0`` = none.
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


def _scaled_noise(shape, *, scale, dist, device, generator):
    """``scale`` x std-matched noise of shape ``shape``. Used for every noise prior in
    this module: the action **prior** the flow integrates from (``policy`` / ``imagine``,
    scaled by ``action_temp``), the **state** prior (``transition`` / ``imagine``, always
    unit-scale — there is no ``state_temp``), and the **additive** action-exploration
    noise of ``autoregressive``.

    * ``"normal"``  — ``scale * N(0, I)`` (std ``scale``); the prior the denoiser was
      trained with (as a prior, ``scale != 1`` is mildly OOD).
    * ``"uniform"`` — ``U(-a, a)`` with ``a = scale * sqrt(3)``, so its std still equals
      ``scale`` (temperature-matched to the normal case — a flat, bounded shape instead of
      a Gaussian one).
    """
    if dist == "normal":
        return scale * torch.randn(shape, device=device, generator=generator)
    if dist == "uniform":
        a = scale * math.sqrt(3.0)
        u = torch.rand(shape, device=device, generator=generator)          # U(0, 1)
        return a * (2.0 * u - 1.0)                                         # U(-a, a), std = scale
    raise ValueError(f"unknown noise dist={dist!r} (expected 'normal' or 'uniform')")


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
    state_prior: str = "normal",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample ``B`` action rollouts of length ``H`` from the policy marginal
    ``p(a_{t:t+H} | o_{<=t})``. The horizon **state** is held at pure noise
    throughout (we don't claim to know the future observations), so only the
    actions are meaningful. Returns ``a_hor`` of shape ``(B, H, n_act)``.

    ``state_prior`` shapes that held-at-noise horizon state; it is never integrated here,
    but it *is* fed to the denoiser at every step, so it still perturbs the sampled
    actions.
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
    z[:, Tc:] = _scaled_noise((B, H, N_lat, D_lat), scale=1.0, dist=state_prior,   # held at noise
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
    state_prior: str = "normal",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Roll the world model forward under a **given** clean action sequence.
    Horizon actions are held clean; horizon state integrates from noise to
    clean. Returns predicted ``z_hor`` of shape ``(B, H, N_lat, D_lat)``.

    ``state_prior`` selects the shape of the noise prior that integration starts from
    (``"normal"`` | std-matched ``"uniform"``). There is no action-prior knob here — the
    actions are given.
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
    z[:, Tc:] = _scaled_noise((B, H, N_lat, D_lat), scale=1.0, dist=state_prior,
                              device=device, generator=generator)
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
    state_prior: str = "normal",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample ``B`` **joint** short rollouts of length ``H`` from the model:
    actions and the states they induce are denoised together, both from their
    noise priors to clean, conditioned on the (optionally noised) context.

    This is the planner's edge generator ``pi_prior``: a single object that is
    simultaneously a stochastic policy and its own forward model. Returns
    ``(z_hor, a_hor)`` with shapes ``(B, H, N_lat, D_lat)`` and ``(B, H, n_act)``.

    Both priors are selectable here: ``action_prior`` (scaled by ``action_temp``) and
    ``state_prior`` (unit-scale), each ``"normal"`` or std-matched ``"uniform"``.
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


# ---------------------------------------------------------------------------
# autoregressive imagination :  step-by-step policy -> world model
# ---------------------------------------------------------------------------

@torch.no_grad()
def autoregressive(
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
    state_prior: str = "normal",
    action_noise: float = 0.0,
    action_noise_dist: str = "normal",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the edge **one frame at a time**, autoregressively.

    For each of ``H`` steps: sample a single action from the policy
    ``p(a_t | o_{<=t})`` (:func:`policy` with horizon 1), optionally add exploration
    noise to it, step the world model ``s_t, a_t -> s_{t+1}`` (:func:`transition` with
    horizon 1), then **append** ``(s_{t+1}, a_t)`` to the context and repeat. Every action
    is therefore conditioned on the *actually realized* previous state — unlike
    :func:`imagine` (which denoises the whole horizon jointly) and unlike ``two_stage``
    (policy then world-model over the whole horizon at once).

    ``action_noise`` adds noise ON TOP of the policy's own stochasticity (which
    ``action_temp`` / ``action_prior`` already control): ``a_t <- a_t +
    scaled_noise(action_noise, action_noise_dist)``, with ``action_noise_dist`` in
    ``{'normal', 'uniform'}`` (std-matched, see :func:`_scaled_noise`). ``0`` = none.

    ``action_prior`` / ``state_prior`` are forwarded unchanged to both sub-calls, so the
    prior shapes apply at every one of the ``H`` steps.

    Cost: ``2*H`` primitive rollout calls per edge (an ``H``-frame ``policy`` + ``H``-frame
    ``transition`` would be ``2``), so this is the most expensive edge sampler. ``ctx_noise``
    is applied at every step to the current (growing) context; for a clean-memory rollout
    set ``ctx_noise=0`` and drive diversity through ``action_temp`` / ``action_noise``.
    Returns ``(z_hor (B,H,N,D), a_hor (B,H,n_act))``.
    """
    _, N_lat, D_lat, n_act = _dims(denoiser)
    device = ctx_z.device
    cz, ca = _expand_ctx(ctx_z, ctx_a, B)
    cz, ca = cz.contiguous(), ca.contiguous()
    z_out = torch.empty(B, H, N_lat, D_lat, device=device)
    a_out = torch.empty(B, H, n_act, device=device)
    for h in range(H):
        # 1. one action from the policy (horizon 1; the policy's own future state is noise)
        a1 = policy(denoiser, cz, ca, 1, B=B, K=K, ctx_noise=ctx_noise,
                    ctx_noise_honest=ctx_noise_honest, action_temp=action_temp,
                    action_prior=action_prior, state_prior=state_prior,
                    dtype=dtype, generator=generator)   # (B,1,n_act)
        # 2. optional additive exploration noise on the sampled action
        if action_noise and action_noise > 0.0:
            a1 = a1 + _scaled_noise((B, 1, n_act), scale=action_noise, dist=action_noise_dist,
                                    device=device, generator=generator)
        # 3. world-model step to the next state given that action (horizon 1)
        z1 = transition(denoiser, cz, ca, a1, K=K, ctx_noise=ctx_noise,
                        ctx_noise_honest=ctx_noise_honest, state_prior=state_prior,
                        dtype=dtype, generator=generator)  # (B,1,N,D)
        z_out[:, h], a_out[:, h] = z1[:, 0], a1[:, 0]
        # 4. append the realized (state, action) and advance the context
        cz = torch.cat([cz, z1], dim=1)
        ca = torch.cat([ca, a1], dim=1)
    return z_out, a_out