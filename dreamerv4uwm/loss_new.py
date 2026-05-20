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

`force_video=True` bypasses (θ, r) sampling and reproduces the legacy
video schedule (state τ uniform, broadcast over T; action τ pinned to 0,
loss masked to state). Kept so the image-branch of the existing train
script works without changes.
"""

import math
from typing import Optional

import torch
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
# Causality-aware loss weighting
# ============================================================================

def loss_weight_causal(
    act_tau: torch.Tensor,
    state_tau: torch.Tensor,
    alpha: float = 0.5,
    floor: str = 'uniform',
    floor_slope: float = 0.9,
):
    """Causality-aware per-modality weight; returns (w_act, w_obs).

    Standard noise convention (x = 1 - τ, x=0 clean):
        w_act ∝ x_act · (1 - x_state)   — action loss matters when action is
                                          noisy AND state is clean
        w_obs ∝ x_state · (1 - x_act)   — symmetric
    Translated to the codebase τ convention:
        causal_act = (1 - act_tau) · state_tau
        causal_obs = (1 - state_tau) · act_tau

    Final weight is convex combination with a floor:
        w = alpha · causal + (1 - alpha) · floor

    floor='uniform' → 1 baseline (recommended default).
    floor='ramp'    → floor_slope · τ + (1 - floor_slope) per-modality
                       (legacy ramp recovers at alpha=0, floor_slope=0.9).
    """
    causal_act = (1.0 - act_tau) * state_tau
    causal_obs = (1.0 - state_tau) * act_tau

    if floor == 'uniform':
        floor_act = torch.ones_like(act_tau)
        floor_obs = torch.ones_like(state_tau)
    elif floor == 'ramp':
        floor_act = floor_slope * act_tau + (1.0 - floor_slope)
        floor_obs = floor_slope * state_tau + (1.0 - floor_slope)
    else:
        raise ValueError(f"unknown floor scheme: {floor!r}")

    w_act = alpha * causal_act + (1.0 - alpha) * floor_act
    w_obs = alpha * causal_obs + (1.0 - alpha) * floor_obs
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
    PROFILE_VIDEO = -1  # sentinel for force_video=True

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
        # r distribution Beta(α, β) on [0, 1]; default uniform
        r_beta_alpha: float = 1.0,
        r_beta_beta: float = 1.0,
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
            [profile_step_prob, profile_progressive_prob, profile_constant_prob],
            dtype=torch.float32,
        )
        assert (profile_weights >= 0).all() and profile_weights.sum() > 0, \
            "r-profile mixture weights must be non-negative with positive sum"
        self.profile_probs = (profile_weights / profile_weights.sum()).to(device)

        assert r_beta_alpha > 0 and r_beta_beta > 0, "Beta params must be positive"
        self.r_beta_alpha = float(r_beta_alpha)
        self.r_beta_beta = float(r_beta_beta)
        self._r_is_uniform = (self.r_beta_alpha == 1.0 and self.r_beta_beta == 1.0)

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
        """Sample per-batch (profile_type, per-frame r) on [0, 1]."""
        profile_type = torch.multinomial(self.profile_probs, B, replacement=True)
        r = torch.zeros(B, T, device=self.device)

        # Step profile: r=0 on first C frames, r=r_hor on rest.
        step_mask = (profile_type == self.PROFILE_STEP)
        n_step = int(step_mask.sum().item())
        if n_step > 0:
            if T > 2:
                ctx_lengths = torch.randint(1, T - 1, (n_step,), device=self.device)
            else:
                ctx_lengths = torch.zeros(n_step, dtype=torch.long, device=self.device)
            r_hor = self._sample_r(n_step)
            frame_idx = torch.arange(T, device=self.device).unsqueeze(0)         # (1, T)
            is_horizon = (frame_idx >= ctx_lengths.unsqueeze(-1)).float()         # (n, T)
            r[step_mask] = is_horizon * r_hor.unsqueeze(-1)

        # Progressive profile: linear ramp r_min (frame 0) → r_max (frame T-1).
        prog_mask = (profile_type == self.PROFILE_PROGRESSIVE)
        n_prog = int(prog_mask.sum().item())
        if n_prog > 0:
            r0 = self._sample_r(n_prog)
            r1 = self._sample_r(n_prog)
            r_min = torch.minimum(r0, r1)
            r_max = torch.maximum(r0, r1)
            slope = torch.linspace(0, 1, T, device=self.device).unsqueeze(0)      # (1, T)
            r[prog_mask] = r_min.unsqueeze(-1) + slope * (r_max - r_min).unsqueeze(-1)

        # Constant profile: r broadcast across all frames.
        const_mask = (profile_type == self.PROFILE_CONSTANT)
        n_const = int(const_mask.sum().item())
        if n_const > 0:
            r[const_mask] = self._sample_r(n_const).unsqueeze(-1).expand(-1, T)

        return profile_type, r

    # ----- main entry points -----
    def _quantize_tau(self, tau: torch.Tensor):
        """Quantize continuous τ ∈ [0, 1] to the discrete grid and re-derive
        the snapped τ value. Matches `loss.py`'s pattern where the embedding
        lookup and forward-mixing τ are always on the same grid."""
        tau_idx = (tau * self.max_diff_steps).long().clamp(max=self.max_diff_steps - 1)
        tau_q = tau_idx.float() / self.max_diff_steps
        return tau_q, tau_idx

    def sample_step_noise(self, batch_size: int, seq_len: int, force_video: bool = False):
        B, T = int(batch_size), int(seq_len)

        if force_video:
            # Legacy video schedule: state τ uniform per batch (broadcast over T),
            # action τ = 0 (pure noise). Loss path masks the action stream.
            state_tau_d = torch.randint(0, self.max_diff_steps, (B,), device=self.device)
            state_tau = state_tau_d.float().unsqueeze(-1).expand(B, T) / self.max_diff_steps
            action_tau = torch.zeros(B, T, device=self.device)
            state_tau_q, state_tau_idx = self._quantize_tau(state_tau)
            action_tau_q, action_tau_idx = self._quantize_tau(action_tau)
            theta = torch.full((B,), math.pi / 2, device=self.device)             # diagnostic
            r = state_tau_q                                                        # diagnostic
            profile_type = torch.full(
                (B,), self.PROFILE_VIDEO, dtype=torch.long, device=self.device,
            )
            return (
                dict(tau=state_tau_q, tau_idx=state_tau_idx),
                dict(tau=action_tau_q, tau_idx=action_tau_idx),
                theta, r, profile_type,
            )

        theta = self._sample_theta(B)                                              # (B,)
        x_max, y_max = self._theta_to_boundary(theta)                              # each (B,)
        profile_type, r = self._sample_r_profile(B, T)                             # (B,), (B, T)

        # Map (θ, r) → (x_act, x_state) in standard noise convention.
        x_act_noise = r * x_max.unsqueeze(-1)     # (B, T) ∈ [0, 1]
        x_state_noise = r * y_max.unsqueeze(-1)   # (B, T) ∈ [0, 1]

        # Convert to τ convention (τ = 1 - x) and quantize.
        state_tau_q, state_tau_idx = self._quantize_tau(1.0 - x_state_noise)
        action_tau_q, action_tau_idx = self._quantize_tau(1.0 - x_act_noise)

        return (
            dict(tau=state_tau_q, tau_idx=state_tau_idx),
            dict(tau=action_tau_q, tau_idx=action_tau_idx),
            theta, r, profile_type,
        )

    def forward(
        self,
        z_clean: torch.Tensor,  # (B, T, N_lat, D_lat)
        a_clean: torch.Tensor,  # (B, T, 1, n_actions)
        force_video: bool = False,
    ):
        B, T, N_lat, D_lat = z_clean.shape
        obs_diff, act_diff, theta, r, profile_type = self.sample_step_noise(
            B, T, force_video=force_video,
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
            "is_video": bool(force_video),
        }


# ============================================================================
# Unified loss
# ============================================================================

def compute_unified_uwm_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
    device='cpu',
    causal_alpha: float = 0.5,
    weight_floor: str = 'uniform',
    floor_slope: float = 0.9,
    rewards: Optional[torch.Tensor] = None,
):
    """Flow-matching loss with causality-aware per-modality weighting.

    No mode-based masking: every frame contributes, scaled by the causal
    weight. Clean frames (both modalities τ ≈ 1) get near-zero weight
    under the causal scheme and contribute negligibly without explicit
    slicing.

    `info['is_video']=True` short-circuits to legacy video-mode behavior:
    state loss across the full sequence with unit weight, action loss
    zeroed.

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
    )

    obs_flow_sq = (z_hat - x).pow(2).mean(dim=(-1, -2))  # (B, T)
    act_flow_sq = (a_hat - a).pow(2).mean(dim=(-1, -2))  # (B, T)

    if info.get('is_video', False):
        obs_flow_loss = obs_flow_sq.mean()
        act_flow_loss = act_flow_sq.mean() * 0.0
    else:
        w_act, w_obs = loss_weight_causal(
            act_tau=info['act_tau'],
            state_tau=info['obs_tau'],
            alpha=causal_alpha,
            floor=weight_floor,
            floor_slope=floor_slope,
        )
        obs_flow_loss = (obs_flow_sq * w_obs).mean()
        act_flow_loss = (act_flow_sq * w_act).mean()

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
