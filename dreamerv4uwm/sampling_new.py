"""
Diagnostic samplers and probes for the horizon-aware unified-loss UWM model.

Self-contained — does not import from `sampling.py`. The companion eval
notebook `notebooks/horizon-aware-eval.ipynb` is the primary consumer.

Two utilities are exposed, both designed for diagnostic use against a model
trained with `train_dynamics_uwm_new.py` + `loss_new.py`:

1. `ray_sampler` — flow-matching Euler integrator that starts at an arbitrary
   point `(n_act_start, n_state_start)` on the unit noise square and
   integrates along the straight ray to clean origin. Per-modality dt is
   scaled so both modalities arrive at clean after the same number of
   Euler steps. Subsumes WM / ID / policy / mode-3 / any θ in between via
   the choice of starting point and `is_horizon`.

2. `grid_loss_probe` — non-integrating probe. For each `(n_act, n_state)`
   on a user-supplied grid, constructs `x_tau` at that noise level using
   the known clean horizon signal (for diagnostic init), queries the model
   once, and returns the x-prediction MSE. Sweeping the grid produces the
   `(n_act, n_state)` loss-surface heatmap (Tier 1.a).

Convention reminder: `n` is *noise level* (0 = clean, 1 = pure noise — the
user-facing convention used in our design discussion). `tau` is *cleanness*
(τ = 1 - n — the codebase convention). Args expose `n`; internal math uses τ.

`is_horizon` is the per-frame `(T,)` flag consumed by `models/dynamics.py`'s
horizon-aware temporal mask: 0 = causal-attend-from, 1 = bidirectional within
the horizon block. When the loaded checkpoint has `cfg.denoiser.horizon_aware=False`,
the flag is silently ignored by the denoiser.
"""
from typing import Optional, Tuple

import torch


# ============================================================================
# is_horizon helpers
# ============================================================================

def make_is_horizon(
    T: int,
    ctx_len: Optional[int] = None,
    all_bidir: bool = False,
    all_causal: bool = False,
    device='cpu',
) -> torch.Tensor:
    """Build the `(T,)` is_horizon flag for the temporal mask.

    Exactly one of three patterns:
      - `all_causal=True`           → `zeros(T)` (pure causal).
      - `all_bidir=True`            → `ones(T)`  (fully bidirectional).
      - `ctx_len=C` (default-ish)   → `[0]*C + [1]*(T-C)` (causal into ctx,
                                       bidirectional within horizon).

    `ctx_len` is ignored if either `all_*` flag is set.
    """
    if all_bidir and all_causal:
        raise ValueError("Pick at most one of all_bidir / all_causal")
    if all_causal:
        return torch.zeros(T, dtype=torch.long, device=device)
    if all_bidir:
        return torch.ones(T, dtype=torch.long, device=device)
    if ctx_len is None:
        raise ValueError("Provide ctx_len or one of all_bidir / all_causal")
    ih = torch.zeros(T, dtype=torch.long, device=device)
    ih[ctx_len:] = 1
    return ih


def _quantize_tau_to_idx(tau: float, N: int) -> int:
    """Map continuous τ ∈ [0, 1] to a valid embedding index in [0, N-1]."""
    return max(0, min(N - 1, int(round(tau * N))))


# ============================================================================
# Ray sampler — start anywhere on the unit square, integrate to clean
# ============================================================================

@torch.no_grad()
def ray_sampler(
    denoiser,
    ctx_latents: torch.Tensor,         # (B, T_ctx, N_lat, D_lat) — clean obs context
    ctx_actions: torch.Tensor,         # (B, T_ctx, n_act)        — clean action context
    horizon_latents: torch.Tensor,     # (B, T_hor, N_lat, D_lat) — clean state ground truth
    horizon_actions: torch.Tensor,     # (B, T_hor, n_act)        — clean action ground truth
    n_act_start: float,                # noise level at integration start, ∈ [0, 1]
    n_state_start: float,              # noise level at integration start, ∈ [0, 1]
    num_diffusion_steps: int,
    is_horizon: Optional[torch.Tensor] = None,  # (T_total,) long; auto-built from ctx_len if None
    context_cond_tau: float = 0.99,
    action_noise_std: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate from `(n_act_start, n_state_start)` along the straight ray
    to `(0, 0)` on the unit square. Returns `(z_hor, a_hor)` — the final
    horizon predictions, shapes `(B, T_hor, N_lat, D_lat)` and `(B, T_hor, n_act)`.

    Horizon clean signal (`horizon_latents`, `horizon_actions`) is used only
    to initialize `x_tau` at the chosen starting noise level — this is a
    *diagnostic* protocol that gives the model a partially-corrupted ground
    truth and asks it to finish the denoising. For pure-prior generation,
    set `n_act_start = n_state_start = 1.0` so the initial x_tau collapses
    to pure noise and the clean tensors are unused (only their shapes matter).

    A modality with `n_X_start == 0` is held at its clean horizon signal
    throughout — no Euler step is taken on that stream. This recovers WM
    (state-only integration) and ID (action-only) cleanly.

    Per-modality dt is `n_X_start / num_diffusion_steps`, so each modality
    travels exactly its required distance in K steps regardless of starting
    point — i.e., both modalities arrive at fully clean at step K.
    """
    device, dtype = ctx_latents.device, ctx_latents.dtype
    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    T_hor = horizon_latents.shape[1]
    T_total = T_ctx + T_hor
    n_act_dim = ctx_actions.shape[-1]
    N = denoiser.cfg.denoiser.num_noise_levels

    if is_horizon is None:
        is_horizon = make_is_horizon(T_total, ctx_len=T_ctx, device=device)
    else:
        is_horizon = is_horizon.to(device=device, dtype=torch.long)

    # Cleanness (τ) at integration start, per modality.
    tau_state_start = 1.0 - float(n_state_start)   # τ_state at step 0
    tau_act_start   = 1.0 - float(n_act_start)
    tau_cond_idx    = _quantize_tau_to_idx(context_cond_tau, N)
    step_idx_t      = torch.zeros((B, T_total), dtype=torch.long, device=device)

    # Per-Euler dt in τ — positive (τ increases as we denoise).
    K = int(num_diffusion_steps)
    dt_tau_state = float(n_state_start) / K       # = 0 if held clean → no movement
    dt_tau_act   = float(n_act_start)   / K

    # --- Initialize buffers ----------------------------------------------------
    z = torch.empty(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
    a = torch.empty(B, T_total, n_act_dim,     device=device, dtype=dtype)

    # Context: nearly-clean mix at context_cond_tau.
    z[:, :T_ctx] = (
        (1.0 - context_cond_tau) * torch.randn_like(ctx_latents)
        + context_cond_tau * ctx_latents
    )
    a[:, :T_ctx] = (
        (1.0 - context_cond_tau) * action_noise_std * torch.randn_like(ctx_actions)
        + context_cond_tau * ctx_actions
    )

    # Horizon: mix clean signal with random noise at the chosen start τ.
    # At n=1 (full noise) → τ_start=0 → x_tau = noise (clean signal unused).
    # At n=0 (clean)      → τ_start=1 → x_tau = clean (noise unused, no integration).
    z_noise = torch.randn(B, T_hor, N_lat, D_lat, device=device, dtype=dtype)
    z[:, T_ctx:] = (1.0 - tau_state_start) * z_noise + tau_state_start * horizon_latents
    a_noise = action_noise_std * torch.randn(B, T_hor, n_act_dim, device=device, dtype=dtype)
    a[:, T_ctx:] = (1.0 - tau_act_start) * a_noise + tau_act_start * horizon_actions

    # --- Euler integration -----------------------------------------------------
    cur_tau_state = tau_state_start
    cur_tau_act   = tau_act_start

    for _ in range(K):
        # sigma_idx tensors: context at tau_cond_idx, horizon at current per-modality τ.
        obs_sigma_idx = torch.full((B, T_total), tau_cond_idx, dtype=torch.long, device=device)
        act_sigma_idx = torch.full((B, T_total), tau_cond_idx, dtype=torch.long, device=device)
        obs_sigma_idx[:, T_ctx:] = _quantize_tau_to_idx(cur_tau_state, N)
        act_sigma_idx[:, T_ctx:] = _quantize_tau_to_idx(cur_tau_act, N)

        z_hat, a_hat, _ = denoiser(
            noisy_act=a,
            noisy_obs=z,
            obs_sigma_idx=obs_sigma_idx,
            obs_step_idx=step_idx_t,
            act_sigma_idx=act_sigma_idx,
            act_step_idx=step_idx_t,
            is_horizon=is_horizon,
        )
        a_hat = a_hat.squeeze(-2)  # (B, T_total, n_act)

        # Per-modality Euler step on horizon only; skip if held clean (dt = 0).
        if dt_tau_state > 0.0:
            denom_state = max(1.0 - cur_tau_state, 1e-5)
            z[:, T_ctx:] = z[:, T_ctx:] + ((z_hat - z) / denom_state * dt_tau_state)[:, T_ctx:]
            cur_tau_state += dt_tau_state
        if dt_tau_act > 0.0:
            denom_act = max(1.0 - cur_tau_act, 1e-5)
            a[:, T_ctx:] = a[:, T_ctx:] + ((a_hat - a) / denom_act * dt_tau_act)[:, T_ctx:]
            cur_tau_act += dt_tau_act

    return z[:, T_ctx:], a[:, T_ctx:]


# ============================================================================
# Progressive ray sampler — per-frame varying starting noise
# ============================================================================

@torch.no_grad()
def progressive_ray_sampler(
    denoiser,
    ctx_latents: torch.Tensor,         # (B, T_ctx, N_lat, D_lat)
    ctx_actions: torch.Tensor,         # (B, T_ctx, n_act)
    horizon_latents: torch.Tensor,     # (B, T_hor, N_lat, D_lat)
    horizon_actions: torch.Tensor,     # (B, T_hor, n_act)
    n_act_start_per_frame,             # scalar OR 1-D tensor of length T_hor
    n_state_start_per_frame,           # scalar OR 1-D tensor of length T_hor
    num_diffusion_steps: int,
    is_horizon: Optional[torch.Tensor] = None,
    context_cond_tau: float = 0.99,
    action_noise_std: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-frame ray integration: each horizon frame starts at its own
    `(n_act, n_state)` and integrates its own ray to clean origin. Used
    to mimic the `progressive` r-profile inference (per-frame staircase of
    noise levels across the horizon).

    All frames take the same number K of Euler steps; per-frame dt scales
    so each frame arrives at clean exactly at step K regardless of its
    individual starting noise.

    Scalar args are broadcast — calling this with both args = 1.0 reproduces
    `ray_sampler(n_X_start=1.0)`. The function is kept as a separate entry
    point rather than collapsed into `ray_sampler` because the per-frame
    bookkeeping (tensor dt / cur_tau) is heavier than the scalar code path.
    """
    device, dtype = ctx_latents.device, ctx_latents.dtype
    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    T_hor = horizon_latents.shape[1]
    T_total = T_ctx + T_hor
    n_act_dim = ctx_actions.shape[-1]
    N = denoiser.cfg.denoiser.num_noise_levels

    def _to_per_frame(x, T):
        if isinstance(x, (int, float)):
            return torch.full((T,), float(x), dtype=torch.float32)
        x = torch.as_tensor(x, dtype=torch.float32)
        assert x.shape == (T,), f"expected scalar or shape ({T},), got {tuple(x.shape)}"
        return x

    n_act_pf   = _to_per_frame(n_act_start_per_frame,   T_hor).to(device)
    n_state_pf = _to_per_frame(n_state_start_per_frame, T_hor).to(device)

    if is_horizon is None:
        is_horizon = make_is_horizon(T_total, ctx_len=T_ctx, device=device)
    else:
        is_horizon = is_horizon.to(device=device, dtype=torch.long)

    K = int(num_diffusion_steps)
    tau_cond_idx = _quantize_tau_to_idx(context_cond_tau, N)
    step_idx_t   = torch.zeros((B, T_total), dtype=torch.long, device=device)

    # Per-frame τ start and dt.
    tau_state_pf = 1.0 - n_state_pf            # (T_hor,)
    tau_act_pf   = 1.0 - n_act_pf
    dt_state_pf  = n_state_pf / K              # frames with n=0 have dt=0 (held clean)
    dt_act_pf    = n_act_pf   / K

    # Buffers.
    z = torch.empty(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
    a = torch.empty(B, T_total, n_act_dim,     device=device, dtype=dtype)

    # Context init (nearly clean).
    z[:, :T_ctx] = (
        (1.0 - context_cond_tau) * torch.randn_like(ctx_latents)
        + context_cond_tau * ctx_latents
    )
    a[:, :T_ctx] = (
        (1.0 - context_cond_tau) * action_noise_std * torch.randn_like(ctx_actions)
        + context_cond_tau * ctx_actions
    )

    # Horizon init: per-frame partial-noise mix.
    z_noise = torch.randn(B, T_hor, N_lat, D_lat, device=device, dtype=dtype)
    a_noise = action_noise_std * torch.randn(B, T_hor, n_act_dim, device=device, dtype=dtype)
    tau_state_b = tau_state_pf.view(1, T_hor, 1, 1)
    tau_act_b   = tau_act_pf.view(1, T_hor, 1)
    z[:, T_ctx:] = (1.0 - tau_state_b) * z_noise + tau_state_b * horizon_latents
    a[:, T_ctx:] = (1.0 - tau_act_b)   * a_noise + tau_act_b   * horizon_actions

    cur_tau_state_pf = tau_state_pf.clone()
    cur_tau_act_pf   = tau_act_pf.clone()
    dt_state_b = dt_state_pf.view(1, T_hor, 1, 1)
    dt_act_b   = dt_act_pf.view(1, T_hor, 1)

    for _ in range(K):
        # Per-frame sigma_idx (quantize current τ to the grid).
        state_idx_hor = (cur_tau_state_pf * N).round().long().clamp(0, N - 1)
        act_idx_hor   = (cur_tau_act_pf   * N).round().long().clamp(0, N - 1)
        obs_sigma_idx = torch.full((B, T_total), tau_cond_idx, dtype=torch.long, device=device)
        act_sigma_idx = torch.full((B, T_total), tau_cond_idx, dtype=torch.long, device=device)
        obs_sigma_idx[:, T_ctx:] = state_idx_hor.unsqueeze(0).expand(B, -1)
        act_sigma_idx[:, T_ctx:] = act_idx_hor.unsqueeze(0).expand(B, -1)

        z_hat, a_hat, _ = denoiser(
            noisy_act=a,
            noisy_obs=z,
            obs_sigma_idx=obs_sigma_idx,
            obs_step_idx=step_idx_t,
            act_sigma_idx=act_sigma_idx,
            act_step_idx=step_idx_t,
            is_horizon=is_horizon,
        )
        a_hat = a_hat.squeeze(-2)

        # Per-frame Euler step (frames with dt=0 contribute zero update).
        denom_state = (1.0 - cur_tau_state_pf).clamp_min(1e-5).view(1, T_hor, 1, 1)
        denom_act   = (1.0 - cur_tau_act_pf  ).clamp_min(1e-5).view(1, T_hor, 1)
        z[:, T_ctx:] = z[:, T_ctx:] + (z_hat[:, T_ctx:] - z[:, T_ctx:]) / denom_state * dt_state_b
        a[:, T_ctx:] = a[:, T_ctx:] + (a_hat[:, T_ctx:] - a[:, T_ctx:]) / denom_act   * dt_act_b
        cur_tau_state_pf = cur_tau_state_pf + dt_state_pf
        cur_tau_act_pf   = cur_tau_act_pf   + dt_act_pf

    return z[:, T_ctx:], a[:, T_ctx:]


# ============================================================================
# Marginal samplers — multi-frame pretraining inference (Tier 3)
# ============================================================================
#
# Mirror inference protocol of `VideoPretrainingForwardProcess` /
# `ActionPretrainingForwardProcess` in `loss_new.py`:
#   • Active modality: per-frame Diffusion-Forcing-style noise that integrates
#     to clean over K Euler steps (per-frame `dt = n_active_start / K`).
#   • Inactive modality: input held at pure noise throughout (no integration,
#     model receives `sigma_idx = 0` on the inactive horizon stream).
#
# Distinct from `ray_sampler` because that one only supports "integrate from
# n_start" or "hold clean" — neither matches "hold at pure noise."


@torch.no_grad()
def marginal_state_sampler(
    denoiser,
    ctx_latents: torch.Tensor,
    ctx_actions: torch.Tensor,
    horizon_latents: torch.Tensor,     # (B, T_hor, N_lat, D_lat) — shape; values used iff n<1
    horizon_actions: torch.Tensor,     # (B, T_hor, n_act) — shape only; action held at noise
    n_state_start_per_frame,           # scalar or (T_hor,) — state noise per frame
    num_diffusion_steps: int,
    is_horizon: Optional[torch.Tensor] = None,
    context_cond_tau: float = 0.99,
    action_noise_std: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-edge (`n_act=1` throughout) inference for video-pretraining-style
    state generation. Returns `(z_hor, a_hor)`; `a_hor` is the unchanged pure-
    noise action input and is returned for shape symmetry with other samplers.
    """
    device, dtype = ctx_latents.device, ctx_latents.dtype
    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    T_hor = horizon_latents.shape[1]
    T_total = T_ctx + T_hor
    n_act_dim = ctx_actions.shape[-1]
    N = denoiser.cfg.denoiser.num_noise_levels

    if isinstance(n_state_start_per_frame, (int, float)):
        n_state_pf = torch.full((T_hor,), float(n_state_start_per_frame), dtype=torch.float32)
    else:
        n_state_pf = torch.as_tensor(n_state_start_per_frame, dtype=torch.float32)
        assert n_state_pf.shape == (T_hor,)
    n_state_pf = n_state_pf.to(device)

    if is_horizon is None:
        is_horizon = make_is_horizon(T_total, ctx_len=T_ctx, device=device)
    else:
        is_horizon = is_horizon.to(device=device, dtype=torch.long)

    K = int(num_diffusion_steps)
    tau_cond_idx = _quantize_tau_to_idx(context_cond_tau, N)
    step_idx_t   = torch.zeros((B, T_total), dtype=torch.long, device=device)

    tau_state_pf = 1.0 - n_state_pf
    dt_state_pf  = n_state_pf / K
    dt_state_b   = dt_state_pf.view(1, T_hor, 1, 1)

    # Buffers.
    z = torch.empty(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
    a = torch.empty(B, T_total, n_act_dim,     device=device, dtype=dtype)

    # Context: nearly clean.
    z[:, :T_ctx] = (
        (1.0 - context_cond_tau) * torch.randn_like(ctx_latents)
        + context_cond_tau * ctx_latents
    )
    a[:, :T_ctx] = (
        (1.0 - context_cond_tau) * action_noise_std * torch.randn_like(ctx_actions)
        + context_cond_tau * ctx_actions
    )

    # Horizon state: partial-noise mix per frame.
    z_noise = torch.randn(B, T_hor, N_lat, D_lat, device=device, dtype=dtype)
    tau_state_b = tau_state_pf.view(1, T_hor, 1, 1)
    z[:, T_ctx:] = (1.0 - tau_state_b) * z_noise + tau_state_b * horizon_latents
    # Horizon action: pure noise throughout (held; never integrated).
    a[:, T_ctx:] = action_noise_std * torch.randn(B, T_hor, n_act_dim, device=device, dtype=dtype)

    cur_tau_state_pf = tau_state_pf.clone()

    for _ in range(K):
        state_idx_hor = (cur_tau_state_pf * N).round().long().clamp(0, N - 1)
        obs_sigma_idx = torch.full((B, T_total), tau_cond_idx, dtype=torch.long, device=device)
        act_sigma_idx = torch.full((B, T_total), tau_cond_idx, dtype=torch.long, device=device)
        obs_sigma_idx[:, T_ctx:] = state_idx_hor.unsqueeze(0).expand(B, -1)
        act_sigma_idx[:, T_ctx:] = 0  # horizon action: full noise (matches training)

        z_hat, _, _ = denoiser(
            noisy_act=a,
            noisy_obs=z,
            obs_sigma_idx=obs_sigma_idx,
            obs_step_idx=step_idx_t,
            act_sigma_idx=act_sigma_idx,
            act_step_idx=step_idx_t,
            is_horizon=is_horizon,
        )

        denom_state = (1.0 - cur_tau_state_pf).clamp_min(1e-5).view(1, T_hor, 1, 1)
        z[:, T_ctx:] = z[:, T_ctx:] + (z_hat[:, T_ctx:] - z[:, T_ctx:]) / denom_state * dt_state_b
        cur_tau_state_pf = cur_tau_state_pf + dt_state_pf
        # Action input: unchanged (still pure noise).

    return z[:, T_ctx:], a[:, T_ctx:]


@torch.no_grad()
def marginal_action_sampler(
    denoiser,
    ctx_latents: torch.Tensor,
    ctx_actions: torch.Tensor,
    horizon_latents: torch.Tensor,     # (B, T_hor, N_lat, D_lat) — shape only; state held at noise
    horizon_actions: torch.Tensor,     # (B, T_hor, n_act) — shape; values used iff n<1
    n_act_start_per_frame,             # scalar or (T_hor,) — action noise per frame
    num_diffusion_steps: int,
    is_horizon: Optional[torch.Tensor] = None,
    context_cond_tau: float = 0.99,
    action_noise_std: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-edge (`n_state=1` throughout) inference for action-pretraining-style
    action generation. Returns `(z_hor, a_hor)`; `z_hor` is the unchanged pure-
    noise state input and is returned for shape symmetry.
    """
    device, dtype = ctx_latents.device, ctx_latents.dtype
    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    T_hor = horizon_actions.shape[1]
    T_total = T_ctx + T_hor
    n_act_dim = ctx_actions.shape[-1]
    N = denoiser.cfg.denoiser.num_noise_levels

    if isinstance(n_act_start_per_frame, (int, float)):
        n_act_pf = torch.full((T_hor,), float(n_act_start_per_frame), dtype=torch.float32)
    else:
        n_act_pf = torch.as_tensor(n_act_start_per_frame, dtype=torch.float32)
        assert n_act_pf.shape == (T_hor,)
    n_act_pf = n_act_pf.to(device)

    if is_horizon is None:
        is_horizon = make_is_horizon(T_total, ctx_len=T_ctx, device=device)
    else:
        is_horizon = is_horizon.to(device=device, dtype=torch.long)

    K = int(num_diffusion_steps)
    tau_cond_idx = _quantize_tau_to_idx(context_cond_tau, N)
    step_idx_t   = torch.zeros((B, T_total), dtype=torch.long, device=device)

    tau_act_pf = 1.0 - n_act_pf
    dt_act_pf  = n_act_pf / K
    dt_act_b   = dt_act_pf.view(1, T_hor, 1)

    z = torch.empty(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
    a = torch.empty(B, T_total, n_act_dim,     device=device, dtype=dtype)

    # Context.
    z[:, :T_ctx] = (
        (1.0 - context_cond_tau) * torch.randn_like(ctx_latents)
        + context_cond_tau * ctx_latents
    )
    a[:, :T_ctx] = (
        (1.0 - context_cond_tau) * action_noise_std * torch.randn_like(ctx_actions)
        + context_cond_tau * ctx_actions
    )

    # Horizon state: pure noise throughout (held).
    z[:, T_ctx:] = torch.randn(B, T_hor, N_lat, D_lat, device=device, dtype=dtype)
    # Horizon action: partial-noise mix per frame.
    a_noise = action_noise_std * torch.randn(B, T_hor, n_act_dim, device=device, dtype=dtype)
    tau_act_b = tau_act_pf.view(1, T_hor, 1)
    a[:, T_ctx:] = (1.0 - tau_act_b) * a_noise + tau_act_b * horizon_actions

    cur_tau_act_pf = tau_act_pf.clone()

    for _ in range(K):
        act_idx_hor = (cur_tau_act_pf * N).round().long().clamp(0, N - 1)
        obs_sigma_idx = torch.full((B, T_total), tau_cond_idx, dtype=torch.long, device=device)
        act_sigma_idx = torch.full((B, T_total), tau_cond_idx, dtype=torch.long, device=device)
        obs_sigma_idx[:, T_ctx:] = 0  # horizon state: full noise (matches training)
        act_sigma_idx[:, T_ctx:] = act_idx_hor.unsqueeze(0).expand(B, -1)

        _, a_hat, _ = denoiser(
            noisy_act=a,
            noisy_obs=z,
            obs_sigma_idx=obs_sigma_idx,
            obs_step_idx=step_idx_t,
            act_sigma_idx=act_sigma_idx,
            act_step_idx=step_idx_t,
            is_horizon=is_horizon,
        )
        a_hat = a_hat.squeeze(-2)

        denom_act = (1.0 - cur_tau_act_pf).clamp_min(1e-5).view(1, T_hor, 1)
        a[:, T_ctx:] = a[:, T_ctx:] + (a_hat[:, T_ctx:] - a[:, T_ctx:]) / denom_act * dt_act_b
        cur_tau_act_pf = cur_tau_act_pf + dt_act_pf
        # State input: unchanged (still pure noise).

    return z[:, T_ctx:], a[:, T_ctx:]


# ============================================================================
# Grid loss probe — (n_act, n_state) heatmap for Tier 1.a
# ============================================================================

@torch.no_grad()
def grid_loss_probe(
    denoiser,
    ctx_latents: torch.Tensor,
    ctx_actions: torch.Tensor,
    target_latents: torch.Tensor,      # (B, T_hor, N_lat, D_lat) — clean ground truth
    target_actions: torch.Tensor,      # (B, T_hor, n_act)
    n_act_grid,                        # 1-D iterable in [0, 1]
    n_state_grid,                      # 1-D iterable in [0, 1]
    is_horizon: Optional[torch.Tensor] = None,
    context_cond_tau: float = 0.99,
    action_noise_std: float = 1.0,
    shared_noise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For each `(n_act, n_state)` in the Cartesian product of the two grids,
    build `x_tau` at that noise level using the known clean horizon signal,
    run a single denoiser forward pass, and compute the x-prediction MSE on
    the horizon. No integration.

    Returns
    -------
    obs_loss_grid : (G_act, G_state) tensor — state x-prediction MSE.
    act_loss_grid : (G_act, G_state) tensor — action x-prediction MSE.

    `shared_noise=True` reuses one Gaussian draw for the noise term across all
    grid points (keeps per-point MSE differences attributable to the noise
    level, not to RNG). Set False if you want IID noise per point.
    """
    device, dtype = ctx_latents.device, ctx_latents.dtype
    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    T_hor = target_latents.shape[1]
    T_total = T_ctx + T_hor
    n_act_dim = ctx_actions.shape[-1]
    N = denoiser.cfg.denoiser.num_noise_levels

    if is_horizon is None:
        is_horizon = make_is_horizon(T_total, ctx_len=T_ctx, device=device)
    else:
        is_horizon = is_horizon.to(device=device, dtype=torch.long)

    tau_cond_idx = _quantize_tau_to_idx(context_cond_tau, N)
    step_idx_t   = torch.zeros((B, T_total), dtype=torch.long, device=device)

    # Pre-build context.
    z_ctx = (
        (1.0 - context_cond_tau) * torch.randn_like(ctx_latents)
        + context_cond_tau * ctx_latents
    )
    a_ctx = (
        (1.0 - context_cond_tau) * action_noise_std * torch.randn_like(ctx_actions)
        + context_cond_tau * ctx_actions
    )

    # Optionally fix the noise tensors across grid points.
    if shared_noise:
        z_noise = torch.randn(B, T_hor, N_lat, D_lat, device=device, dtype=dtype)
        a_noise = action_noise_std * torch.randn(B, T_hor, n_act_dim, device=device, dtype=dtype)

    G_act = len(n_act_grid)
    G_state = len(n_state_grid)
    obs_loss_grid = torch.zeros(G_act, G_state)
    act_loss_grid = torch.zeros(G_act, G_state)

    for i, n_act in enumerate(n_act_grid):
        for j, n_state in enumerate(n_state_grid):
            tau_act   = 1.0 - float(n_act)
            tau_state = 1.0 - float(n_state)

            if not shared_noise:
                z_noise = torch.randn(B, T_hor, N_lat, D_lat, device=device, dtype=dtype)
                a_noise = action_noise_std * torch.randn(B, T_hor, n_act_dim, device=device, dtype=dtype)

            # Mix clean horizon with noise at the chosen τ.
            z_hor_tau = (1.0 - tau_state) * z_noise + tau_state * target_latents
            a_hor_tau = (1.0 - tau_act)   * a_noise + tau_act   * target_actions
            z = torch.cat([z_ctx, z_hor_tau], dim=1)
            a = torch.cat([a_ctx, a_hor_tau], dim=1)

            obs_sigma_idx = torch.full((B, T_total), tau_cond_idx, dtype=torch.long, device=device)
            act_sigma_idx = torch.full((B, T_total), tau_cond_idx, dtype=torch.long, device=device)
            obs_sigma_idx[:, T_ctx:] = _quantize_tau_to_idx(tau_state, N)
            act_sigma_idx[:, T_ctx:] = _quantize_tau_to_idx(tau_act,   N)

            z_hat, a_hat, _ = denoiser(
                noisy_act=a,
                noisy_obs=z,
                obs_sigma_idx=obs_sigma_idx,
                obs_step_idx=step_idx_t,
                act_sigma_idx=act_sigma_idx,
                act_step_idx=step_idx_t,
                is_horizon=is_horizon,
            )
            a_hat = a_hat.squeeze(-2)

            obs_loss_grid[i, j] = (z_hat[:, T_ctx:] - target_latents).pow(2).mean().item()
            act_loss_grid[i, j] = (a_hat[:, T_ctx:] - target_actions).pow(2).mean().item()

    return obs_loss_grid, act_loss_grid
