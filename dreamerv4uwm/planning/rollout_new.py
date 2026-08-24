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
* :func:`progressive` — the edge as a **rolling noise ramp**: the frames in flight carry
  *per-frame, monotonically increasing* noise levels, and each denoiser pass advances all
  of them one notch, emitting the frame that reached clean and appending a fresh pure-noise
  frame at the back. Interpolates between the two extremes above at a fraction of the
  autoregressive cost. See "Progressive autoregressive rollout" below.

All of them share the same diversity knobs, which is the whole point of keeping
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

Unconditional generation
------------------------
Every primitive accepts ``ctx_z=ctx_a=None``, i.e. **no context at all** (``Tc = 0``, so
``T = H`` and ``is_horizon`` is all-ones — fully bidirectional). The model then samples
from its own prior over trajectories rather than a posterior given a history: ``imagine``
gives ``p(a_{1:H}, o_{1:H})``, ``policy`` gives ``p(a_{1:H})``, ``transition`` gives
``p(o_{1:H} | a_{1:H})`` (a state sequence invented to match the given actions), and
``autoregressive`` generates its first frame from nothing and conditions on its own output
thereafter. ``ctx_noise`` / ``ctx_noise_honest`` are inert in this mode — there is no
context to corrupt. Since the device can no longer be read off the context, pass
``device=`` (it defaults to the denoiser's own).

This is the natural upper bound for the edge-diversity question: the spread of
unconditional samples is the most the sampler can produce, so it bounds what any
context-conditioned expansion can hope for.

Progressive autoregressive rollout
----------------------------------
:func:`progressive` implements the PA-VDM schedule (Xie et al., *Progressive
Autoregressive Video Diffusion Models*, arXiv:2410.08151) on this denoiser. Every other
primitive here gives the whole horizon **one shared** noise level and walks it to clean
together (``imagine``), or denoises exactly one frame at a time to clean before looking at
the next (``autoregressive``). PA-VDM is the continuum between them: hold ``window``
frames in flight at *different* noise levels — a ramp from nearly-clean at the front to
pure noise at the back — and per denoiser pass advance every frame by one notch, **save**
the front frame once it reaches clean, **shift** the window forward, and **append** a new
pure-noise frame at the back.

Three reasons this is a natural fit here rather than a port:

* **No model change.** ``obs_sigma_idx`` / ``act_sigma_idx`` are already ``(B, T)`` and
  feed a per-frame embedding lookup (``models/dynamics.py``), so per-frame noise levels
  are natively expressible. The paper's one required modification (per-frame timestep
  embedding) is already how this denoiser is written.
* **In-distribution, not training-free.** ``loss.py``'s ``forcing`` mode trains on exactly
  this shape — "Progressive Temporal Denoising: linearly decreasing tau across T, frame 0
  cleanest, frame T-1 noisiest", both streams sharing the ramp — and ``loss_new.py``'s
  ``progressive`` r-profile subsumes it. A checkpoint whose mode mix included those has
  *seen* per-frame ramps; one trained only on ``step`` profiles (clean ctx + single-level
  horizon) has not, and should be read as the paper's training-free regime. Check the mode
  weights of the checkpoint before interpreting a result.
* **Cost sits in the gap.** ``ceil(K/window)*(H-1) + K`` denoiser passes
  (:func:`progressive_steps`) against ``K`` for ``imagine`` and ``2*H*K`` for
  ``autoregressive``. At the default ``window=K`` that is ``H + K - 1`` — e.g. 17 passes
  for ``H=12, K=6``, versus 6 and 144.

``window`` is the dial: ``1`` reduces to strict frame-by-frame autoregression (each frame
fully denoised, conditioned on the realized previous ones — the joint counterpart of
:func:`autoregressive`), and larger windows overlap more denoising, so a frame is still
being refined while its successors are born. Note the two ends differ in *what a frame
gets to see*: a frame's later neighbours are present (as noise) throughout its own
denoising, which is exactly the "better propagation from earlier to later frames" the
paper credits for temporal coherence.

Two paper techniques are handled differently here:

* **Overlapped conditioning** (their Sec. 3.4 — keep a chunk of already-clean frames in
  the attention window, else frame-to-frame jitter) is what ``max_ctx`` controls: emitted
  frames are appended to the context and remain visible. The default keeps all of them.
* **Chunked frames** (their Sec. 3.3 — a 3D-VAE chunk of ``C`` latent frames must share a
  noise level or the video diverges) is a **no-op here** and is not implemented: this
  tokenizer is purely spatial (``(B,T,C,H,W) -> (B,T,N,D)``, ``T`` preserved), so its chunk
  size is ``C = 1`` and every latent frame is already independently addressable in time.

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


def _ctx_len(ctx_z) -> int:
    """Number of context frames; ``0`` for the unconditional case (``ctx_z is None``)."""
    return 0 if ctx_z is None else int(ctx_z.shape[1])


def _resolve_device(denoiser, *tensors, device=None):
    """Device to build the rollout buffers on.

    Explicit ``device`` wins; otherwise the first tensor given (normally the context);
    otherwise the denoiser's own device — the fallback the unconditional path relies on,
    since with ``ctx_z=None`` there is nothing else to read it from.
    """
    if device is not None:
        return torch.device(device)
    for t in tensors:
        if t is not None:
            return t.device
    return next(denoiser.parameters()).device


def _expand_ctx(ctx_z, ctx_a, B):
    """Broadcast a (1, Tc, ...) context to (B, Tc, ...) (no-op if already B).

    ``(None, None)`` passes straight through: no context at all (unconditional)."""
    if ctx_z is None:
        return None, None
    if ctx_z.shape[0] != B:
        ctx_z = ctx_z.expand(B, -1, -1, -1).contiguous()
        ctx_a = ctx_a.expand(B, -1, -1).contiguous()
    return ctx_z, ctx_a


def _build_context(ctx_z, ctx_a, ctx_noise, ctx_noise_honest, N, gen):
    """Return (z_ctx, a_ctx, ctx_obs_sigma_idx) with optional honest/mismatched
    observation-noise injection. Actions are always kept clean.

    With ``ctx_z is None`` (unconditional) there is nothing to noise: returns
    ``(None, None, idx)``, the index being an unused placeholder — every context slice
    it would be written into is empty."""
    tau_ctx = 1.0 - float(ctx_noise)
    obs_idx = _quantize_tau_to_idx(tau_ctx if ctx_noise_honest else 1.0, N)
    if ctx_z is None:
        return None, None, obs_idx
    if ctx_noise > 0.0:
        eps = torch.randn(ctx_z.shape, device=ctx_z.device, dtype=ctx_z.dtype, generator=gen)
        z_ctx = (1.0 - tau_ctx) * eps + tau_ctx * ctx_z
    else:
        z_ctx = ctx_z
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
    ctx_z: Optional[torch.Tensor],  # (1|B, Tc, N_lat, D_lat) clean obs context, or None
    ctx_a: Optional[torch.Tensor],  # (1|B, Tc, n_act)        clean action context, or None
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
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample ``B`` action rollouts of length ``H`` from the policy marginal
    ``p(a_{t:t+H} | o_{<=t})``. The horizon **state** is held at pure noise
    throughout (we don't claim to know the future observations), so only the
    actions are meaningful. Returns ``a_hor`` of shape ``(B, H, n_act)``.

    ``state_prior`` shapes that held-at-noise horizon state; it is never integrated here,
    but it *is* fed to the denoiser at every step, so it still perturbs the sampled
    actions.

    Pass ``ctx_z=ctx_a=None`` to sample **unconditionally** (``Tc=0``): the marginal
    ``p(a_{1:H})`` with no observed history. ``ctx_noise`` / ``ctx_noise_honest`` are then
    inert, and ``device`` (defaulting to the denoiser's) says where to build the buffers.
    """
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
    ctx_z: Optional[torch.Tensor],  # (1|B, Tc, N_lat, D_lat), or None
    ctx_a: Optional[torch.Tensor],  # (1|B, Tc, n_act),        or None
    actions: torch.Tensor,        # (B, H, n_act) clean action sequence to apply
    K: int = 12,
    *,
    ctx_noise: float = 0.0,
    ctx_noise_honest: bool = True,
    state_prior: str = "normal",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Roll the world model forward under a **given** clean action sequence.
    Horizon actions are held clean; horizon state integrates from noise to
    clean. Returns predicted ``z_hor`` of shape ``(B, H, N_lat, D_lat)``.

    ``state_prior`` selects the shape of the noise prior that integration starts from
    (``"normal"`` | std-matched ``"uniform"``). There is no action-prior knob here — the
    actions are given.

    With ``ctx_z=ctx_a=None`` (``Tc=0``) this becomes ``p(o_{1:H} | a_{1:H})``: the states
    an action sequence induces with no observed starting state — the model is free to
    invent one consistent with the actions.
    """
    N, N_lat, D_lat, n_act = _dims(denoiser)
    B, H = actions.shape[0], actions.shape[1]
    device = _resolve_device(denoiser, ctx_z, actions, device=device)
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
    ctx_z: Optional[torch.Tensor],  # (1|B, Tc, N_lat, D_lat), or None
    ctx_a: Optional[torch.Tensor],  # (1|B, Tc, n_act),        or None
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
    device: Optional[torch.device] = None,
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

    With ``ctx_z=ctx_a=None`` (``Tc=0``) this samples the **unconditional** joint
    ``p(a_{1:H}, o_{1:H})`` — trajectories from nowhere, the model's own prior over
    episodes. Useful as the diversity ceiling to compare context-conditioned siblings
    against: whatever spread survives conditioning cannot exceed this.
    """
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
    ctx_z: Optional[torch.Tensor],  # (1|B, Tc, N_lat, D_lat), or None
    ctx_a: Optional[torch.Tensor],  # (1|B, Tc, n_act),        or None
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
    device: Optional[torch.device] = None,
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

    With ``ctx_z=ctx_a=None`` the **first** frame is generated unconditionally (``Tc=0``);
    from then on the context is the rollout's own realized frames, so step ``h`` conditions
    on ``h`` self-generated ones. That makes this the pure autoregressive prior over
    episodes, ``prod_h p(a_h | o_{<h}) p(o_h | o_{<h}, a_h)``.
    """
    _, N_lat, D_lat, n_act = _dims(denoiser)
    device = _resolve_device(denoiser, ctx_z, device=device)
    cz, ca = _expand_ctx(ctx_z, ctx_a, B)
    if cz is not None:
        cz, ca = cz.contiguous(), ca.contiguous()
    z_out = torch.empty(B, H, N_lat, D_lat, device=device)
    a_out = torch.empty(B, H, n_act, device=device)
    for h in range(H):
        # 1. one action from the policy (horizon 1; the policy's own future state is noise)
        a1 = policy(denoiser, cz, ca, 1, B=B, K=K, ctx_noise=ctx_noise,
                    ctx_noise_honest=ctx_noise_honest, action_temp=action_temp,
                    action_prior=action_prior, state_prior=state_prior,
                    dtype=dtype, device=device, generator=generator)   # (B,1,n_act)
        # 2. optional additive exploration noise on the sampled action
        if action_noise and action_noise > 0.0:
            a1 = a1 + _scaled_noise((B, 1, n_act), scale=action_noise, dist=action_noise_dist,
                                    device=device, generator=generator)
        # 3. world-model step to the next state given that action (horizon 1)
        z1 = transition(denoiser, cz, ca, a1, K=K, ctx_noise=ctx_noise,
                        ctx_noise_honest=ctx_noise_honest, state_prior=state_prior,
                        dtype=dtype, device=device, generator=generator)  # (B,1,N,D)
        z_out[:, h], a_out[:, h] = z1[:, 0], a1[:, 0]
        # 4. append the realized (state, action) and advance the context
        #    (from nothing, on the unconditional first step)
        cz = z1 if cz is None else torch.cat([cz, z1], dim=1)
        ca = a1 if ca is None else torch.cat([ca, a1], dim=1)
    return z_out, a_out


# ---------------------------------------------------------------------------
# progressive autoregressive imagination :  a rolling per-frame noise ramp
# ---------------------------------------------------------------------------

def progressive_steps(H: int, K: int = 12, window: Optional[int] = None) -> int:
    """Denoiser passes one :func:`progressive` rollout costs — computable from the config
    alone, so a planner can budget an edge before running it.

    A frame needs ``K`` passes to go from its noise prior to clean, and consecutive frames
    enter the window ``stride = ceil(K / window)`` passes apart, so the last of ``H`` frames
    starts at pass ``(H-1)*stride`` and finishes ``K`` later::

        steps = (H - 1) * ceil(K / window) + K

    At the default ``window = K`` (the paper's ``F = S``) this is ``H + K - 1``. At
    ``window = 1`` it is ``H * K``, i.e. one full ``imagine`` per frame. Compare ``K`` for
    :func:`imagine` and ``2*H*K`` for :func:`autoregressive`.

    ``ceil`` rather than exact division because a frame can only be born on a pass
    boundary. When ``window`` does not divide ``K`` the stride rounds *up*, so the window
    is never exceeded but is left under-filled: the frames actually in flight number
    ``ceil(K / ceil(K/window))``, not ``window`` — ``K=6, window=4`` really runs a window
    of 3, indistinguishable from ``window=3``. Prefer a ``window`` that divides ``K`` so
    the knob means what it says (with the default ``window=K``, every value divides).
    """
    W = int(K if window is None else window)
    if not 1 <= W <= int(K):
        raise ValueError(f"window={W} must satisfy 1 <= window <= K={K}")
    return (int(H) - 1) * -(-int(K) // W) + int(K)


@torch.no_grad()
def progressive(
    denoiser,
    ctx_z: Optional[torch.Tensor],  # (1|B, Tc, N_lat, D_lat), or None
    ctx_a: Optional[torch.Tensor],  # (1|B, Tc, n_act),        or None
    H: int,
    B: int = 1,
    K: int = 12,
    *,
    window: Optional[int] = None,
    ctx_noise: float = 0.0,
    ctx_noise_honest: bool = True,
    action_temp: float = 1.0,
    action_prior: str = "normal",
    state_prior: str = "normal",
    max_ctx: Optional[int] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll the edge out as a **progressive autoregressive** denoising process (PA-VDM).

    Instead of giving the whole horizon one shared noise level (:func:`imagine`) or
    finishing one frame before starting the next (:func:`autoregressive`), keep ``window``
    frames in flight at *staggered* cleanness — a ramp from nearly-clean at the front to
    pure noise at the back — and repeat::

        denoise once  ->  advance every in-flight frame by one Euler notch
                      ->  the front frame reached clean: save it, append it to the context
                      ->  shift the window, append a fresh pure-noise frame at the back

    Both streams of a frame (its latent **and** its action) share that frame's noise level,
    so this is the joint ``(a, o')`` sampler of :func:`imagine` re-scheduled in time, not a
    policy/world-model split.

    Parameters beyond the shared diversity knobs
    --------------------------------------------
    ``window`` (``W``)
        Frames in flight, i.e. how much the denoising of neighbouring frames overlaps.
        Defaults to ``K`` (the paper's ``F = S``: shift every pass, maximum overlap).
        ``1`` is strict frame-by-frame autoregression — each frame walks its full ``K``
        steps to clean, conditioned only on realized frames, and costs ``H*K`` passes.
        Must satisfy ``1 <= window <= K``; consecutive frames are born
        ``ceil(K/window)`` passes apart, so a ``window`` dividing ``K`` keeps it exactly
        full (see :func:`progressive_steps`).
    ``max_ctx``
        How many **clean** frames stay in the attention window — the paper's *overlapped
        conditioning*. Emitted frames are appended to the context (and are therefore what
        later frames condition on); ``None`` (default) keeps every one of them plus the
        original context, ``n`` keeps only the most recent ``n``, and ``0`` keeps none at
        all — the paper's no-overlap ablation, which they report causes frame-to-frame
        discontinuity, and which here also drops the given context once the first frame
        lands. Mirrors the planner's own ``PlanConfig.max_ctx`` truncation.

    Notes
    -----
    * ``ctx_noise`` is re-applied to the **whole** clean block on every pass — the given
      context *and* the frames emitted so far — matching :func:`autoregressive` rather than
      the paper (which conditions on genuinely clean frames). ``ctx_noise=0`` is the
      faithful PA-VDM setting; anything above it deliberately corrupts the model's own
      realized history, which is the point when this is used as a diversity knob.
    * The per-frame noise levels are the only thing that differs from ``imagine`` at the
      model interface. Whether the checkpoint has *seen* such ramps depends on its training
      mode mix (``forcing`` / ``progressive``) — see the module docstring.
    * Cost is ``progressive_steps(H, K, window)`` denoiser passes over a window of
      ``len(context) + up to `window`` frames, versus ``K`` passes over ``Tc + H`` frames
      for :func:`imagine`.

    Pass ``ctx_z=ctx_a=None`` to generate **unconditionally**: the first frames are born
    with no history at all, and the rollout conditions on its own emitted frames from then
    on (subject to ``max_ctx``).

    Returns ``(z_hor (B,H,N,D), a_hor (B,H,n_act))``, the ``H`` frames in emission order.
    """
    N, N_lat, D_lat, n_act = _dims(denoiser)
    H, K = int(H), int(K)
    if H < 1:
        raise ValueError(f"H={H} must be >= 1")
    W = K if window is None else int(window)
    n_steps = progressive_steps(H, K, W)          # also validates 1 <= W <= K
    stride = -(-K // W)                           # ceil(K/W): passes between two births
    dt = 1.0 / K

    device = _resolve_device(denoiser, ctx_z, device=device)
    cz, ca = _expand_ctx(ctx_z, ctx_a, B)         # clean block; grows as frames are emitted
    if cz is not None:
        cz, ca = cz.contiguous(), ca.contiguous()

    z_out = torch.empty(B, H, N_lat, D_lat, device=device)
    a_out = torch.empty(B, H, n_act, device=device)

    # the window, slot 0 = oldest/cleanest. `taus` holds each slot's cleanness (0 = pure
    # noise, 1 = clean), so it *is* the progressive ramp: taus[0] > taus[1] > ... > taus[-1]
    z_win = torch.empty(B, 0, N_lat, D_lat, device=device)
    a_win = torch.empty(B, 0, n_act, device=device)
    taus: list = []
    n_born = n_emitted = 0

    for step in range(n_steps):
        # --- 1. append: a fresh pure-noise frame enters the back of the window ----------
        if n_born < H and step % stride == 0:
            z_win = torch.cat([z_win, _scaled_noise((B, 1, N_lat, D_lat), scale=1.0,
                                                    dist=state_prior, device=device,
                                                    generator=generator)], dim=1)
            a_win = torch.cat([a_win, _scaled_noise((B, 1, n_act), scale=action_temp,
                                                    dist=action_prior, device=device,
                                                    generator=generator)], dim=1)
            taus.append(0.0)
            n_born += 1
        w = len(taus)

        # --- 2. assemble [clean block | ramped window] ---------------------------------
        z_ctx, a_ctx, ctx_obs_idx = _build_context(
            cz, ca, ctx_noise, ctx_noise_honest, N, generator)
        Tc = _ctx_len(z_ctx)
        T = Tc + w
        z = z_win if z_ctx is None else torch.cat([z_ctx, z_win], dim=1)
        a = a_win if a_ctx is None else torch.cat([a_ctx, a_win], dim=1)

        step_idx = torch.zeros((B, T), dtype=torch.long, device=device)
        is_hor = make_is_horizon(T, ctx_len=Tc, device=device)   # emitted frames are context
        obs_sigma = torch.empty((B, T), dtype=torch.long, device=device)
        act_sigma = torch.empty((B, T), dtype=torch.long, device=device)
        obs_sigma[:, :Tc] = ctx_obs_idx
        act_sigma[:, :Tc] = N - 1
        # the ramp: one noise level PER FRAME, which is the whole point
        ramp = torch.tensor([_quantize_tau_to_idx(t, N) for t in taus],
                            dtype=torch.long, device=device)
        obs_sigma[:, Tc:] = ramp
        act_sigma[:, Tc:] = ramp

        with _autocast(device, dtype):
            z_hat, a_hat, _ = denoiser(
                noisy_act=a, noisy_obs=z,
                obs_sigma_idx=obs_sigma, obs_step_idx=step_idx,
                act_sigma_idx=act_sigma, act_step_idx=step_idx, is_horizon=is_hor)
        z_hat = z_hat[:, Tc:].float()
        a_hat = a_hat.squeeze(-2)[:, Tc:].float()

        # --- 3. one Euler notch, PER SLOT (each sits at its own tau, so its own step) ---
        # same rule as everywhere else here — move dt/(1-tau) of the way to the predicted
        # clean — but vectorised over the ramp. The clamp makes the final notch land
        # exactly on the prediction instead of overshooting past it.
        tau_t = torch.tensor(taus, device=device).view(1, w, 1, 1)
        coef = (dt / (1.0 - tau_t).clamp_min(1e-5)).clamp_max(1.0)
        z_win = z_win + (z_hat - z_win) * coef
        a_win = a_win + (a_hat - a_win) * coef.view(1, w, 1)
        taus = [t + dt for t in taus]

        # --- 4. save & shift: whatever reached clean leaves the window as an output -----
        while taus and taus[0] >= 1.0 - 1e-6:
            z_out[:, n_emitted], a_out[:, n_emitted] = z_win[:, 0], a_win[:, 0]
            # overlapped conditioning: the emitted frame becomes context for its successors
            cz = z_win[:, :1] if cz is None else torch.cat([cz, z_win[:, :1]], dim=1)
            ca = a_win[:, :1] if ca is None else torch.cat([ca, a_win[:, :1]], dim=1)
            if max_ctx is not None:
                cz, ca = (None, None) if max_ctx <= 0 else (cz[:, -max_ctx:], ca[:, -max_ctx:])
            z_win, a_win = z_win[:, 1:], a_win[:, 1:]
            taus.pop(0)
            n_emitted += 1

    return z_out, a_out