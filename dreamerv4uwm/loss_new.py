"""
Unified flow-matching forward process and loss for the dual-channel UWM.

This is an alternative to `loss.py`'s discrete mode-mixture sampler. It
replaces the {policy, wm, id, video, forcing, action_only} modes with a
single ray-based sampler over the (action_noise, state_noise) unit square
and a causality-aware per-modality loss weighting.

Design (see in-repo discussion logs for rationale):

  1. Per-batch ray angle θ ∈ [0, π/2] picks the *causality direction*:
       θ = 0    → inverse dynamics (action noisy | state clean)
       θ = π/4  → policy           (joint noise)
       θ = π/2  → world model      (state noisy | action clean)
     θ is drawn from a mixture {ID, policy, WM, continuum-Uniform[0, π/2]}.

  2. Per-frame ray progress r(t) ∈ [0, 1] picks the *denoising stage*:
       r = 0 → clean (origin), r = 1 → unit-square boundary along θ.
     The per-frame r-profile shape decides context conditioning:
       step        → clean ctx + noisy horizon (subsumes policy / wm / id)
       progressive → linear ramp (subsumes forcing)
       constant    → noisy everywhere (mode 3 / random action proposer)

  3. Loss weighting is causality-aware. Standard noise convention x = 1-τ:
       w_act ∝ x_act · (1 - x_state)   — action loss matters when action is
                                          noisy AND state is clean
       w_obs ∝ x_state · (1 - x_act)   — symmetric
     Mixed with a floor (uniform or legacy ramp) via `alpha`.

Convention note: this module produces (state_tau, action_tau) in the
codebase τ convention (τ=1 clean, τ=0 pure noise), with forward mixing
    x_τ = (1 - τ) · x0 + τ · x_clean
matching `loss.py`. The standard noise convention x = 1 - τ is used only
in the (θ, r) sampling layer.

The marginal single-frame (image-branch) objective lives in a separate
class+function pair in this module — `ImageForwardProcess` and
`compute_image_loss` — because its target (p(state) without action
conditioning, no temporal structure) is semantically distinct from the
joint state-action denoiser the unified loss trains.
"""

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .models.dynamics import DreamerV4Denoiser


# ============================================================================
# Reward MTP helpers (duplicated from loss.py for self-containment)
# ============================================================================

_REWARD_SYMLOG_MIN = -20.0
_REWARD_SYMLOG_MAX = 20.0


def _two_hot_targets(rewards: torch.Tensor, num_buckets: int):
    y = torch.sign(rewards) * torch.log(torch.abs(rewards) + 1.0)
    width = (_REWARD_SYMLOG_MAX - _REWARD_SYMLOG_MIN) / (num_buckets - 1)
    indices = (y - _REWARD_SYMLOG_MIN) / width
    indices = indices.clamp(0, num_buckets - 1)
    low = indices.floor().long()
    high = indices.ceil().long()
    low_w = high.float() - indices
    high_w = indices - low.float()
    edge = (low == high)
    low_w = torch.where(edge, torch.ones_like(low_w), low_w)
    high_w = torch.where(edge, torch.zeros_like(high_w), high_w)
    return low, low_w, high, high_w


def compute_reward_mtp_loss(pred_rewards: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    B, T, L, K = pred_rewards.shape
    if T == 0:
        return pred_rewards.new_zeros(())
    padded = F.pad(rewards, (0, L - 1))
    targets = padded.unfold(dimension=1, size=L, step=1)[:, :T]
    t_idx = torch.arange(T, device=rewards.device).view(1, T, 1)
    l_idx = torch.arange(L, device=rewards.device).view(1, 1, L)
    valid = ((t_idx + l_idx) < T).expand(B, T, L).float()
    low, low_w, high, high_w = _two_hot_targets(targets, num_buckets=K)
    logp = F.log_softmax(pred_rewards.float(), dim=-1)
    nll = -(
        low_w * logp.gather(-1, low.unsqueeze(-1)).squeeze(-1)
        + high_w * logp.gather(-1, high.unsqueeze(-1)).squeeze(-1)
    )
    return (nll * valid).sum() / valid.sum().clamp_min(1.0)


# ============================================================================
# RMS loss scaler (running per-name RMS, used to equalize obs/act magnitudes)
# ============================================================================

class RMSLossScaler:
    """Tracks per-name EMA of `loss²` and returns `loss / sqrt(EMA(loss²))`.

    Used to rebalance the obs/act flow-matching losses when the action stream
    isn't normalized to the same per-element variance as the latent stream —
    the causality-aware weighting in `compute_unified_uwm_loss` assumes the
    two terms are on comparable scale, which they aren't when raw actions
    have arbitrary units.

    The EMA divisor is no-grad and updates only *between* optimizer steps,
    so within any single step it's a constant scalar — the causality
    weighting's per-step relative semantics are fully preserved. Across
    many steps, the two scaled terms equalize to unit RMS in expectation.

    DDP-aware: `all_reduce` averages the per-rank `mean_sq` so the EMA is
    consistent across ranks.

    State is plain Python (not registered with the model). Resuming from
    a checkpoint resets the EMA; the first ~100 steps post-resume run at
    slightly mis-scaled magnitudes while it reconverges. Usually negligible.
    """

    def __init__(self, decay: float = 0.99, eps: float = 1e-8):
        self.decay = float(decay)
        self.eps = float(eps)
        self.ema_sq = {}  # name -> scalar tensor

    def __call__(self, name: str, value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean_sq = value.detach().pow(2).mean()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(mean_sq, op=dist.ReduceOp.AVG)
            if name not in self.ema_sq:
                self.ema_sq[name] = mean_sq
            else:
                self.ema_sq[name] = (
                    self.decay * self.ema_sq[name]
                    + (1.0 - self.decay) * mean_sq
                )
            rms = (self.ema_sq[name] + self.eps).sqrt()
        return value / rms


# ============================================================================
# Causality-aware loss weighting
# ============================================================================

def loss_weight_causal(
    act_tau: torch.Tensor,
    state_tau: torch.Tensor,
    eps: float = 1e-3,
    ramp_beta: float = 1.0,
):
    """Per-modality weight: causality term × per-frame ramp; returns (w_act, w_obs).

    **Causality term.** Each modality's weight is proportional to the
    *cleanness of the other* (the cause), then normalized so the two
    causal weights sum to 1:

        causal_act = (1 - n_state + ε) / ((1 - n_state + ε) + (1 - n_act + ε))
        causal_obs = (1 - n_act + ε)   / ((1 - n_state + ε) + (1 - n_act + ε))

    In the codebase τ convention (τ = 1 - n, so τ is cleanness):
        causal_act = (state_tau + ε) / ((state_tau + ε) + (act_tau + ε))
        causal_obs = (act_tau   + ε) / ((state_tau + ε) + (act_tau + ε))

    A modality's loss matters in proportion to how clean the OTHER modality
    (the cause) is. Normalizing keeps the *relative* attention budget
    constant across the noise square. ε regularizes the policy corner
    (both fully noisy) where the unregularized formula is 0/0; ε pushes
    that point to a 50/50 split (the natural "no clear cause/effect" default).

    **Per-frame ramp** (multiplied onto each modality):

        ramp_act = β + (1 - β) · (1 - n_act)   = β + (1 - β) · act_tau
        ramp_obs = β + (1 - β) · (1 - n_state) = β + (1 - β) · state_tau

    β = 1 → uniform (no ramp, recovers pure causality weighting).
    β = 0 → pure ramp (weight equals cleanness, down-weights the noisy end
            of each modality).

    Final: w_modality = causal_modality · ramp_modality.
    """
    cause_for_act = state_tau + eps   # state cleanness drives action loss
    cause_for_obs = act_tau + eps     # action cleanness drives state loss
    denom = cause_for_act + cause_for_obs

    causal_act = cause_for_act / denom
    causal_obs = cause_for_obs / denom

    # Per-modality, per-frame ramp. At β=1 this is identically 1 (no-op).
    ramp_act = ramp_beta + (1.0 - ramp_beta) * act_tau
    ramp_obs = ramp_beta + (1.0 - ramp_beta) * state_tau

    w_act = causal_act * ramp_act
    w_obs = causal_obs * ramp_obs
    return w_act, w_obs


# ============================================================================
# Unified forward process
# ============================================================================

class UnifiedForwardProcess(nn.Module):
    """Ray-based (θ, r) sampler over the (action_noise, state_noise) unit square.

    Returns the same info-dict shape `compute_unified_uwm_loss` consumes;
    structurally compatible with `loss.py`'s `UWMForwardProcess` so the
    train script's downstream code (forward pass, optimizer, logging) is
    unchanged.
    """

    PROFILE_STEP = 0
    PROFILE_PROGRESSIVE = 1
    PROFILE_CONSTANT = 2
    PROFILE_DIFFUSION_FORCING = 3
    PROFILE_REVERSE_STEP = 4

    THETA_ID = 0
    THETA_POLICY = 1
    THETA_WM = 2
    THETA_CONTINUUM = 3

    def __init__(
        self,
        max_diff_steps: int = 128,
        action_noise_std: float = 1.0,
        # θ-mixture (need not sum to 1; normalized internally)
        theta_id_prob: float = 0.15,
        theta_policy_prob: float = 0.15,
        theta_wm_prob: float = 0.15,
        theta_continuum_prob: float = 0.55,
        # r-profile mixture
        profile_step_prob: float = 0.5,
        profile_progressive_prob: float = 0.3,
        profile_constant_prob: float = 0.2,
        profile_diffusion_forcing_prob: float = 0.0,
        profile_reverse_step_prob: float = 0.0,
        # r distribution Beta(α, β) on [0, 1]; default uniform
        r_beta_alpha: float = 1.0,
        r_beta_beta: float = 1.0,
        # P(fully bidirectional attention) for the diffusion-forcing profile;
        # complementary mass goes to fully causal. Per-batch coin flip, same
        # mechanism as the pretraining marginals.
        diffusion_forcing_bidir_prob: float = 0.5,
        device='cpu',
    ):
        super().__init__()
        self.max_diff_steps = int(max_diff_steps)
        self.action_noise_std = float(action_noise_std)
        self.device = device

        theta_weights = torch.tensor(
            [theta_id_prob, theta_policy_prob, theta_wm_prob, theta_continuum_prob],
            dtype=torch.float32,
        )
        assert (theta_weights >= 0).all() and theta_weights.sum() > 0, \
            "θ-mixture weights must be non-negative with positive sum"
        self.theta_probs = (theta_weights / theta_weights.sum()).to(device)

        profile_weights = torch.tensor(
            [
                profile_step_prob,
                profile_progressive_prob,
                profile_constant_prob,
                profile_diffusion_forcing_prob,
                profile_reverse_step_prob,
            ],
            dtype=torch.float32,
        )
        assert (profile_weights >= 0).all() and profile_weights.sum() > 0, \
            "r-profile mixture weights must be non-negative with positive sum"
        self.profile_probs = (profile_weights / profile_weights.sum()).to(device)

        assert r_beta_alpha > 0 and r_beta_beta > 0, "Beta params must be positive"
        self.r_beta_alpha = float(r_beta_alpha)
        self.r_beta_beta = float(r_beta_beta)
        self._r_is_uniform = (self.r_beta_alpha == 1.0 and self.r_beta_beta == 1.0)

        assert 0.0 <= diffusion_forcing_bidir_prob <= 1.0
        self.diffusion_forcing_bidir_prob = float(diffusion_forcing_bidir_prob)

    # ----- ray geometry -----
    @staticmethod
    def _theta_to_boundary(theta: torch.Tensor):
        """Map θ ∈ [0, π/2] → (x_max, y_max) on the unit-square boundary so
        that (s · x_max, s · y_max) for s ∈ [0, 1] traces the ray.

        Below the diagonal (θ ≤ π/4): ray exits through x=1 first.
        Above the diagonal (θ > π/4): ray exits through y=1 first.
        Endpoints are exact at θ=0 and θ=π/2 (no division involved on the
        chosen branch).
        """
        diagonal = math.pi / 4
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        below = theta <= diagonal
        x_max = torch.where(
            below,
            torch.ones_like(theta),
            cos_t / sin_t.clamp(min=1e-6),
        )
        y_max = torch.where(
            below,
            sin_t / cos_t.clamp(min=1e-6),
            torch.ones_like(theta),
        )
        return x_max, y_max

    # ----- samplers -----
    def _sample_r(self, n: int) -> torch.Tensor:
        if self._r_is_uniform:
            return torch.rand(n, device=self.device)
        beta_dist = torch.distributions.Beta(
            torch.tensor(self.r_beta_alpha, device=self.device),
            torch.tensor(self.r_beta_beta, device=self.device),
        )
        return beta_dist.sample((n,))

    def _sample_r_2d(self, B: int, T: int) -> torch.Tensor:
        """Per-(batch, frame) i.i.d. r ∈ [0, 1] from the same Beta(α, β)
        distribution used by `_sample_r`. Used by the diffusion-forcing
        profile, where each frame draws its own r independently along the
        per-batch ray θ."""
        if self._r_is_uniform:
            return torch.rand(B, T, device=self.device)
        beta_dist = torch.distributions.Beta(
            torch.tensor(self.r_beta_alpha, device=self.device),
            torch.tensor(self.r_beta_beta, device=self.device),
        )
        return beta_dist.sample((B, T))

    def _sample_theta(self, B: int) -> torch.Tensor:
        component = torch.multinomial(self.theta_probs, B, replacement=True)
        theta = torch.empty(B, device=self.device)
        theta[component == self.THETA_ID]     = 0.0
        theta[component == self.THETA_POLICY] = math.pi / 4
        theta[component == self.THETA_WM]     = math.pi / 2
        cont_mask = (component == self.THETA_CONTINUUM)
        n_cont = int(cont_mask.sum().item())
        if n_cont > 0:
            theta[cont_mask] = torch.rand(n_cont, device=self.device) * (math.pi / 2)
        return theta

    def _sample_r_profile(self, B: int, T: int):
        """Sample (profile_type, per-frame r, is_horizon).

        profile_type and (for step) ctx_length are shared across the batch —
        matches legacy `loss.py` (one mode + one ctx_length per batch). This
        keeps `is_horizon` shape (T,) so it's drop-in for the temporal-mask
        builder in `models/blocks.py`.

        Per-profile is_horizon policy:
          step               → zeros on ctx frames, ones on horizon frames
          progressive        → all-zeros (preserve causal/streaming character)
          constant           → all-ones (full bidir; matches mode-3 chunk semantics)
          diffusion_forcing  → per-batch coin flip (zeros or ones) at
                               `diffusion_forcing_bidir_prob`. No ctx/hor;
                               each frame draws its own r along the ray.
          reverse_step       → all-ones (full bidir is required so the noisy
                               past frames can attend to the clean future
                               frames for goal-conditioned grounding).
        """
        profile_idx = int(torch.multinomial(self.profile_probs, 1).item())
        profile_type = torch.full((B,), profile_idx, dtype=torch.long, device=self.device)

        if profile_idx == self.PROFILE_STEP:
            ctx_length = int(torch.randint(1, T - 1, (1,)).item()) if T > 2 else 0
            r_hor = self._sample_r(B)                                              # (B,)
            r = torch.zeros(B, T, device=self.device)
            r[:, ctx_length:] = r_hor.unsqueeze(-1)
            is_horizon = torch.zeros(T, dtype=torch.long, device=self.device)
            is_horizon[ctx_length:] = 1

        elif profile_idx == self.PROFILE_PROGRESSIVE:
            r0 = self._sample_r(B)
            r1 = self._sample_r(B)
            r_min = torch.minimum(r0, r1)
            r_max = torch.maximum(r0, r1)
            slope = torch.linspace(0, 1, T, device=self.device).unsqueeze(0)       # (1, T)
            r = r_min.unsqueeze(-1) + slope * (r_max - r_min).unsqueeze(-1)
            is_horizon = torch.zeros(T, dtype=torch.long, device=self.device)

        elif profile_idx == self.PROFILE_CONSTANT:
            r = self._sample_r(B).unsqueeze(-1).expand(B, T).contiguous()
            is_horizon = torch.ones(T, dtype=torch.long, device=self.device)

        elif profile_idx == self.PROFILE_DIFFUSION_FORCING:
            # Per-frame i.i.d. r along the (single per-batch) ray θ. No ctx/hor
            # split. is_horizon is a per-batch coin flip between fully causal
            # and fully bidirectional, exposing the model to both inference
            # regimes (mirrors the pretraining-marginal pattern).
            r = self._sample_r_2d(B, T)
            if torch.rand(1, device=self.device).item() < self.diffusion_forcing_bidir_prob:
                is_horizon = torch.ones(T, dtype=torch.long, device=self.device)
            else:
                is_horizon = torch.zeros(T, dtype=torch.long, device=self.device)

        elif profile_idx == self.PROFILE_REVERSE_STEP:
            # Time-reverse of step: noisy past + clean future. Forces full
            # bidirectional attention (is_horizon = ones(T)) so noisy past
            # frames can attend to the clean future frames they're being
            # conditioned on. Trains goal-conditioned / hindsight reasoning:
            # "what past trajectory is consistent with this future?"
            future_start = int(torch.randint(1, T - 1, (1,)).item()) if T > 2 else 0
            r_past = self._sample_r(B)
            r = torch.zeros(B, T, device=self.device)
            r[:, :future_start] = r_past.unsqueeze(-1)
            is_horizon = torch.ones(T, dtype=torch.long, device=self.device)

        else:
            raise ValueError(f"unknown profile_idx {profile_idx}")

        return profile_type, r, is_horizon

    # ----- main entry points -----
    def _quantize_tau(self, tau: torch.Tensor):
        """Quantize continuous τ ∈ [0, 1] to the discrete grid and re-derive
        the snapped τ value. Matches `loss.py`'s pattern where the embedding
        lookup and forward-mixing τ are always on the same grid."""
        tau_idx = (tau * self.max_diff_steps).long().clamp(max=self.max_diff_steps - 1)
        tau_q = tau_idx.float() / self.max_diff_steps
        return tau_q, tau_idx

    def sample_step_noise(
        self,
        batch_size: int,
        seq_len: int,
        force_theta: Optional[float] = None,
    ):
        B, T = int(batch_size), int(seq_len)

        if force_theta is not None:
            theta = torch.full((B,), float(force_theta), device=self.device)
        else:
            theta = self._sample_theta(B)                                          # (B,)
        x_max, y_max = self._theta_to_boundary(theta)                              # each (B,)
        profile_type, r, is_horizon = self._sample_r_profile(B, T)                 # (B,), (B, T), (T,)

        # Map (θ, r) → (x_act, x_state) in standard noise convention.
        x_act_noise = r * x_max.unsqueeze(-1)     # (B, T) ∈ [0, 1]
        x_state_noise = r * y_max.unsqueeze(-1)   # (B, T) ∈ [0, 1]

        # Convert to τ convention (τ = 1 - x) and quantize.
        state_tau_q, state_tau_idx = self._quantize_tau(1.0 - x_state_noise)
        action_tau_q, action_tau_idx = self._quantize_tau(1.0 - x_act_noise)

        return (
            dict(tau=state_tau_q, tau_idx=state_tau_idx),
            dict(tau=action_tau_q, tau_idx=action_tau_idx),
            theta, r, profile_type, is_horizon,
        )

    def forward(
        self,
        z_clean: torch.Tensor,  # (B, T, N_lat, D_lat)
        a_clean: torch.Tensor,  # (B, T, 1, n_actions)
        force_theta: Optional[float] = None,
    ):
        B, T, N_lat, D_lat = z_clean.shape
        obs_diff, act_diff, theta, r, profile_type, is_horizon = self.sample_step_noise(
            B, T, force_theta=force_theta,
        )

        z0 = torch.randn_like(z_clean)
        obs_tau_b = obs_diff["tau"].unsqueeze(-1).unsqueeze(-1)                    # (B,T,1,1)
        z_tau = (1.0 - obs_tau_b) * z0 + obs_tau_b * z_clean

        a0 = self.action_noise_std * torch.randn_like(a_clean)
        act_tau_b = act_diff["tau"].unsqueeze(-1).unsqueeze(-1)                    # (B,T,1,1)
        a_tau = (1.0 - act_tau_b) * a0 + act_tau_b * a_clean

        return {
            "x": z_clean,
            "x0": z0,
            "x_tau": z_tau,
            "obs_tau": obs_diff["tau"],
            "obs_tau_idx": obs_diff["tau_idx"],
            "a": a_clean,
            "a0": a0,
            "a_tau": a_tau,
            "act_tau": act_diff["tau"],
            "act_tau_idx": act_diff["tau_idx"],
            "theta": theta,
            "r": r,
            "profile_type": profile_type,
            "is_horizon": is_horizon,  # (T,) long
        }


# ============================================================================
# Unified loss
# ============================================================================

def compute_unified_uwm_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
    device='cpu',
    causal_eps: float = 1e-3,
    ramp_beta: float = 1.0,
    rewards: Optional[torch.Tensor] = None,
    scaler: Optional[RMSLossScaler] = None,
):
    """Flow-matching loss with causality-aware per-modality weighting.

    No mode-based masking: every frame contributes, scaled by the causal
    weight. Clean frames (both modalities τ ≈ 1) get near-zero weight
    under the causal scheme and contribute negligibly without explicit
    slicing.

    Image-branch (single-frame, action-free) training is a separate
    objective served by `ImageForwardProcess` + `compute_image_loss`.

    Returns dict with {obs_flow_loss, act_flow_loss, reward_loss}.
    """
    x = info["x"]
    B, T, N_lat, D_lat = x.shape
    x_tau = info["x_tau"]
    obs_tau_idx = info["obs_tau_idx"]
    a = info["a"]
    a_tau = info["a_tau"]
    act_tau_idx = info["act_tau_idx"]

    step_idx = torch.zeros((B, T), dtype=torch.long, device=device)

    z_hat, a_hat, pred_rewards = denoiser(
        noisy_act=a_tau.squeeze(-2),
        noisy_obs=x_tau,
        obs_sigma_idx=obs_tau_idx,
        obs_step_idx=step_idx,
        act_sigma_idx=act_tau_idx,
        act_step_idx=step_idx,
        is_horizon=info.get("is_horizon"),
    )

    obs_flow_sq = (z_hat - x).pow(2).mean(dim=(-1, -2))  # (B, T)
    act_flow_sq = (a_hat - a).pow(2).mean(dim=(-1, -2))  # (B, T)

    w_act, w_obs = loss_weight_causal(
        act_tau=info['act_tau'],
        state_tau=info['obs_tau'],
        eps=causal_eps,
        ramp_beta=ramp_beta,
    )
    obs_flow_loss = (obs_flow_sq * w_obs).mean()
    act_flow_loss = (act_flow_sq * w_act).mean()

    # Rebalance obs vs act magnitudes if a scaler is supplied. Each term
    # divides by its own running RMS — equalizes long-run scale while
    # preserving the causality weighting's per-step relative dynamics.
    if scaler is not None:
        obs_flow_loss = scaler("obs", obs_flow_loss)
        act_flow_loss = scaler("act", act_flow_loss)

    reward_loss = None
    if pred_rewards is not None:
        if rewards is None:
            raise RuntimeError(
                "denoiser was built with train_reward_model=True but "
                "compute_unified_uwm_loss was called without `rewards`."
            )
        reward_loss = compute_reward_mtp_loss(pred_rewards, rewards)

    return {
        "obs_flow_loss": obs_flow_loss,
        "act_flow_loss": act_flow_loss,
        "reward_loss": reward_loss,
    }


# ============================================================================
# Image (marginal state) forward process and loss
# ============================================================================

class ImageForwardProcess(nn.Module):
    """Marginal state-flow training for cold-start frame generation.

    Targets p(state) — the single-frame distribution with no action
    conditioning and no temporal structure. Intended for the train script's
    `image` branch, where every frame is reshaped into an independent (T=1)
    sequence. Separate from `UnifiedForwardProcess` because the objective is
    semantically different (marginal vs joint) and the noise structure is
    1D (state τ only) vs 2D (state-action noise square).

    Forward: state τ uniform per batch element (broadcast over T); action τ
    pinned to 0 (pure noise). The action tensor is passed through unchanged
    to satisfy the denoiser's input shape, but the action stream is ignored
    in `compute_image_loss`. `is_horizon` is zeros(T) so the temporal mask
    stays purely causal (identical to None) and the `frame_id_embedder`
    row 0 still participates in the gradient (keeps DDP happy when
    cfg.denoiser.horizon_aware=True).

    Convention: τ=1 clean, τ=0 pure noise; forward mixing
    x_τ = (1 - τ)·x0 + τ·x_clean (matches `loss.py` / `UnifiedForwardProcess`).
    """

    def __init__(
        self,
        max_diff_steps: int = 128,
        action_noise_std: float = 1.0,
        device='cpu',
    ):
        super().__init__()
        self.max_diff_steps = int(max_diff_steps)
        self.action_noise_std = float(action_noise_std)
        self.device = device

    def forward(
        self,
        z_clean: torch.Tensor,  # (B, T, N_lat, D_lat) — T is 1 in the image branch
        a_clean: torch.Tensor,  # (B, T, 1, n_actions) — passed through, ignored in loss
    ):
        B, T, N_lat, D_lat = z_clean.shape
        device = z_clean.device

        # State τ: one uniform draw per batch element, broadcast over T,
        # quantized to the discrete grid (matches UnifiedForwardProcess'
        # quantize-then-restore pattern so embedding lookup and forward
        # mixing share the same τ value).
        state_tau_d = torch.randint(0, self.max_diff_steps, (B,), device=device)
        state_tau_idx = state_tau_d.unsqueeze(-1).expand(B, T).contiguous()        # (B, T) long
        state_tau = state_tau_idx.float() / self.max_diff_steps                    # (B, T)

        # Action τ: pinned to 0 (pure noise); action stream is ignored in loss.
        action_tau = torch.zeros(B, T, device=device)
        action_tau_idx = torch.zeros(B, T, dtype=torch.long, device=device)

        # State forward diffusion.
        z0 = torch.randn_like(z_clean)
        obs_tau_b = state_tau.unsqueeze(-1).unsqueeze(-1)                          # (B, T, 1, 1)
        z_tau = (1.0 - obs_tau_b) * z0 + obs_tau_b * z_clean

        # Action forward diffusion — purely noise (τ=0 ⇒ no clean mixing).
        a0 = self.action_noise_std * torch.randn_like(a_clean)
        a_tau = a0

        is_horizon = torch.zeros(T, dtype=torch.long, device=device)

        return {
            "x": z_clean,
            "x0": z0,
            "x_tau": z_tau,
            "obs_tau": state_tau,
            "obs_tau_idx": state_tau_idx,
            "a": a_clean,
            "a0": a0,
            "a_tau": a_tau,
            "act_tau": action_tau,
            "act_tau_idx": action_tau_idx,
            "is_horizon": is_horizon,
        }


def compute_image_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
    device='cpu',
    rewards: Optional[torch.Tensor] = None,
    scaler: Optional[RMSLossScaler] = None,
):
    """Marginal state-flow loss for the image (single-frame) pathway.

    State-only x-prediction MSE averaged across (B, T) with unit weight.
    The action stream is computed (so `action_projector` stays in the grad
    graph and DDP is happy) but its contribution is multiplied by zero —
    actions carry no information here.

    Returns {obs_flow_loss, act_flow_loss, reward_loss}. `act_flow_loss` is
    a zero-valued scalar (kept as a graph node, not a Python 0) so the train
    script's `total = obs + act` arithmetic is shape/dtype-uniform with the
    unified-loss path.
    """
    x = info["x"]
    B, T, N_lat, D_lat = x.shape
    x_tau = info["x_tau"]
    obs_tau_idx = info["obs_tau_idx"]
    a = info["a"]
    a_tau = info["a_tau"]
    act_tau_idx = info["act_tau_idx"]

    step_idx = torch.zeros((B, T), dtype=torch.long, device=device)

    z_hat, a_hat, pred_rewards = denoiser(
        noisy_act=a_tau.squeeze(-2),
        noisy_obs=x_tau,
        obs_sigma_idx=obs_tau_idx,
        obs_step_idx=step_idx,
        act_sigma_idx=act_tau_idx,
        act_step_idx=step_idx,
        is_horizon=info.get("is_horizon"),
    )

    obs_flow_loss = (z_hat - x).pow(2).mean()
    # Keep `action_projector` in the autograd graph — `* 0.0` zeros the
    # contribution but preserves the backward edge (mirrors the legacy
    # `*.mean()*0.` pattern in compute_uwm_loss for non-action modes).
    act_flow_loss = (a_hat - a).pow(2).mean() * 0.0

    # Scale only the meaningful (obs) term so the shared "obs" EMA isn't
    # polluted by the zeroed-out act term.
    if scaler is not None:
        obs_flow_loss = scaler("obs", obs_flow_loss)

    reward_loss = None
    if pred_rewards is not None:
        if rewards is None:
            raise RuntimeError(
                "denoiser was built with train_reward_model=True but "
                "compute_image_loss was called without `rewards`."
            )
        reward_loss = compute_reward_mtp_loss(pred_rewards, rewards)

    return {
        "obs_flow_loss": obs_flow_loss,
        "act_flow_loss": act_flow_loss,
        "reward_loss": reward_loss,
    }


# ============================================================================
# Multi-frame pretraining marginals (right edge: video, top edge: action)
# ============================================================================
#
# Asymmetric per-frame τ structure on the active modality:
#   • VideoPretraining   → per-frame i.i.d. (Diffusion Forcing): each frame
#                          draws its own state τ independently.
#   • ActionPretraining  → uniform across the sequence: one action τ sampled
#                          per batch element, broadcast across all T frames.
#
# In both, the inactive modality is pinned to pure noise everywhere. A
# per-batch coin flip selects either fully-causal (`is_horizon = zeros(T)`)
# or fully-bidirectional (`is_horizon = ones(T)`) attention.
#
# These are the multi-frame counterparts of `ImageForwardProcess`. They live
# alongside it rather than replacing it because the T=1 image branch and the
# T>1 long/short branches play different roles in the train script:
#   • image branch (T=1)        → `ImageForwardProcess` + `compute_image_loss`
#   • short/long video mode     → `VideoPretrainingForwardProcess`  + `compute_video_pretraining_loss`
#   • short/long action mode    → `ActionPretrainingForwardProcess` + `compute_action_pretraining_loss`


class VideoPretrainingForwardProcess(nn.Module):
    """Right-edge (n_act = 1) multi-frame training: state-only video flow.

    Per-frame state τ drawn independently (Diffusion Forcing). Action τ pinned
    to pure noise everywhere. `is_horizon` is a per-batch coin flip:
    `zeros(T)` (fully causal) or `ones(T)` (fully bidirectional), 50/50.
    """

    def __init__(
        self,
        max_diff_steps: int = 128,
        action_noise_std: float = 1.0,
        bidir_prob: float = 0.5,
        device='cpu',
    ):
        super().__init__()
        self.max_diff_steps = int(max_diff_steps)
        self.action_noise_std = float(action_noise_std)
        assert 0.0 <= bidir_prob <= 1.0
        self.bidir_prob = float(bidir_prob)
        self.device = device

    def forward(
        self,
        z_clean: torch.Tensor,  # (B, T, N_lat, D_lat)
        a_clean: torch.Tensor,  # (B, T, 1, n_actions) — passed through, loss ignores it
    ):
        B, T, N_lat, D_lat = z_clean.shape
        device = z_clean.device

        # Per-frame state τ (Diffusion Forcing).
        state_tau_idx = torch.randint(0, self.max_diff_steps, (B, T), device=device)
        state_tau = state_tau_idx.float() / self.max_diff_steps

        # Action τ = 0 (pure noise).
        action_tau = torch.zeros(B, T, device=device)
        action_tau_idx = torch.zeros(B, T, dtype=torch.long, device=device)

        # State forward diffusion.
        z0 = torch.randn_like(z_clean)
        obs_tau_b = state_tau.unsqueeze(-1).unsqueeze(-1)
        z_tau = (1.0 - obs_tau_b) * z0 + obs_tau_b * z_clean

        # Action stream is pure noise (τ = 0 ⇒ no clean mixing).
        a0 = self.action_noise_std * torch.randn_like(a_clean)
        a_tau = a0

        # Per-batch causality coin flip. Different ranks may pick differently;
        # DDP averages valid per-rank gradients so divergent masks are fine.
        if torch.rand(1, device=device).item() < self.bidir_prob:
            is_horizon = torch.ones(T, dtype=torch.long, device=device)
        else:
            is_horizon = torch.zeros(T, dtype=torch.long, device=device)

        return {
            "x": z_clean,
            "x0": z0,
            "x_tau": z_tau,
            "obs_tau": state_tau,
            "obs_tau_idx": state_tau_idx,
            "a": a_clean,
            "a0": a0,
            "a_tau": a_tau,
            "act_tau": action_tau,
            "act_tau_idx": action_tau_idx,
            "is_horizon": is_horizon,
        }


class ActionPretrainingForwardProcess(nn.Module):
    """Top-edge (n_state = 1) multi-frame training: action-only flow.

    Counterpart to `VideoPretrainingForwardProcess`, but with one action τ
    sampled per batch element and **broadcast across all T frames** (uniform
    across the sequence) — rather than per-frame i.i.d. State τ pinned to
    pure noise everywhere. Causality coin flip is the same per-batch
    50/50 mechanism (controlled by `bidir_prob`).
    """

    def __init__(
        self,
        max_diff_steps: int = 128,
        action_noise_std: float = 1.0,
        bidir_prob: float = 0.5,
        device='cpu',
    ):
        super().__init__()
        self.max_diff_steps = int(max_diff_steps)
        self.action_noise_std = float(action_noise_std)
        assert 0.0 <= bidir_prob <= 1.0
        self.bidir_prob = float(bidir_prob)
        self.device = device

    def forward(
        self,
        z_clean: torch.Tensor,  # (B, T, N_lat, D_lat) — passed through, loss ignores it
        a_clean: torch.Tensor,  # (B, T, 1, n_actions)
    ):
        B, T, N_lat, D_lat = z_clean.shape
        device = z_clean.device

        # Action τ: one uniform draw per batch element, broadcast across T
        # (sequence-uniform, not per-frame Diffusion Forcing). All frames in
        # a given sequence share the same action noise level.
        action_tau_d = torch.randint(0, self.max_diff_steps, (B,), device=device)
        action_tau_idx = action_tau_d.unsqueeze(-1).expand(B, T).contiguous()
        action_tau = action_tau_idx.float() / self.max_diff_steps

        # State τ = 0 (pure noise).
        state_tau = torch.zeros(B, T, device=device)
        state_tau_idx = torch.zeros(B, T, dtype=torch.long, device=device)

        # State stream is pure noise (τ = 0 ⇒ no clean mixing).
        z0 = torch.randn_like(z_clean)
        z_tau = z0

        # Action forward diffusion.
        a0 = self.action_noise_std * torch.randn_like(a_clean)
        act_tau_b = action_tau.unsqueeze(-1).unsqueeze(-1)
        a_tau = (1.0 - act_tau_b) * a0 + act_tau_b * a_clean

        if torch.rand(1, device=device).item() < self.bidir_prob:
            is_horizon = torch.ones(T, dtype=torch.long, device=device)
        else:
            is_horizon = torch.zeros(T, dtype=torch.long, device=device)

        return {
            "x": z_clean,
            "x0": z0,
            "x_tau": z_tau,
            "obs_tau": state_tau,
            "obs_tau_idx": state_tau_idx,
            "a": a_clean,
            "a0": a0,
            "a_tau": a_tau,
            "act_tau": action_tau,
            "act_tau_idx": action_tau_idx,
            "is_horizon": is_horizon,
        }


def compute_video_pretraining_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
    device='cpu',
    rewards: Optional[torch.Tensor] = None,
    scaler: Optional[RMSLossScaler] = None,
):
    """Multi-frame video-pretraining loss: state-only x-prediction.

    State loss is averaged across (B, T) with unit weight. Action loss is
    kept in the autograd graph via `* 0.0` (same pattern as compute_image_loss)
    so `action_projector` continues to receive gradient.
    """
    x = info["x"]
    B, T, N_lat, D_lat = x.shape
    x_tau = info["x_tau"]
    obs_tau_idx = info["obs_tau_idx"]
    a = info["a"]
    a_tau = info["a_tau"]
    act_tau_idx = info["act_tau_idx"]

    step_idx = torch.zeros((B, T), dtype=torch.long, device=device)

    z_hat, a_hat, pred_rewards = denoiser(
        noisy_act=a_tau.squeeze(-2),
        noisy_obs=x_tau,
        obs_sigma_idx=obs_tau_idx,
        obs_step_idx=step_idx,
        act_sigma_idx=act_tau_idx,
        act_step_idx=step_idx,
        is_horizon=info.get("is_horizon"),
    )

    obs_flow_loss = (z_hat - x).pow(2).mean()
    act_flow_loss = (a_hat - a).pow(2).mean() * 0.0

    # Scale only the meaningful (obs) term — same shared "obs" key as the
    # unified loss so the EMA pools obs-loss magnitudes across modes.
    if scaler is not None:
        obs_flow_loss = scaler("obs", obs_flow_loss)

    reward_loss = None
    if pred_rewards is not None:
        if rewards is None:
            raise RuntimeError(
                "denoiser was built with train_reward_model=True but "
                "compute_video_pretraining_loss was called without `rewards`."
            )
        reward_loss = compute_reward_mtp_loss(pred_rewards, rewards)

    return {
        "obs_flow_loss": obs_flow_loss,
        "act_flow_loss": act_flow_loss,
        "reward_loss": reward_loss,
    }


def compute_action_pretraining_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
    device='cpu',
    rewards: Optional[torch.Tensor] = None,
    scaler: Optional[RMSLossScaler] = None,
):
    """Multi-frame action-pretraining loss: action-only x-prediction.

    Symmetric to `compute_video_pretraining_loss`: action loss with unit
    weight, state loss multiplied by 0 to preserve the autograd graph.
    """
    x = info["x"]
    B, T, N_lat, D_lat = x.shape
    x_tau = info["x_tau"]
    obs_tau_idx = info["obs_tau_idx"]
    a = info["a"]
    a_tau = info["a_tau"]
    act_tau_idx = info["act_tau_idx"]

    step_idx = torch.zeros((B, T), dtype=torch.long, device=device)

    z_hat, a_hat, pred_rewards = denoiser(
        noisy_act=a_tau.squeeze(-2),
        noisy_obs=x_tau,
        obs_sigma_idx=obs_tau_idx,
        obs_step_idx=step_idx,
        act_sigma_idx=act_tau_idx,
        act_step_idx=step_idx,
        is_horizon=info.get("is_horizon"),
    )

    act_flow_loss = (a_hat - a).pow(2).mean()
    obs_flow_loss = (z_hat - x).pow(2).mean() * 0.0

    # Scale only the meaningful (act) term — same shared "act" key as the
    # unified loss so the EMA pools act-loss magnitudes across modes.
    if scaler is not None:
        act_flow_loss = scaler("act", act_flow_loss)

    reward_loss = None
    if pred_rewards is not None:
        if rewards is None:
            raise RuntimeError(
                "denoiser was built with train_reward_model=True but "
                "compute_action_pretraining_loss was called without `rewards`."
            )
        reward_loss = compute_reward_mtp_loss(pred_rewards, rewards)

    return {
        "obs_flow_loss": obs_flow_loss,
        "act_flow_loss": act_flow_loss,
        "reward_loss": reward_loss,
    }
