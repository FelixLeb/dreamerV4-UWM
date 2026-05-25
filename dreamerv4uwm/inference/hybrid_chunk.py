"""Hybrid KV-cached chunk-based rollout sampler for UWM.

Inference-time consumer of the unified-loss `step` r-profile training
distribution: causal-back-to-context + bidirectional-within-horizon.

Per chunk:
  1. Allocate noise tensors for m new frames at the current cache tail.
  2. K denoising steps, each calling `denoiser.forward_chunk_step(...,
     commit_first_k=0)` and advancing τ via flow-matching Euler.
  3. Optional (recommended): one extra "commit pass" that calls
     `forward_chunk_step(... commit_first_k=k)` on the *clean* chunk so
     the K/V written to cache reflect clean tokens (not the slightly-noisy
     input from the last denoising step).
  4. Emit the first `k = commit_per_chunk` frames as committed; throw
     away the remaining `m - k` frames (they'll be re-denoised in the
     next chunk at fresh positions).

Two extreme operating modes via `commit_per_chunk`:
  - `commit_per_chunk = 1` (MPC-style): each committed frame benefits
    from bidir context to the entire remaining horizon. Costliest.
  - `commit_per_chunk = m` (chunk-wise): cheapest non-AR mode. Trades
    re-planning for throughput.

Cache sizing follows Option 1: `cache_capacity = context_length - m`,
so `cache_len + m ≤ context_length` always — the temporal attention can
use `mask=None, is_causal=False` (the same workaround as autoregressive
single-step decode, generalized to m). No custom mask, no flash-attn
silent-fallback risk.

Convention: `n` is noise level (0 = clean, 1 = pure noise), the user-
facing convention. `tau = 1 - n` is the codebase cleanness convention
that the model embeddings expect.
"""

import warnings
from typing import Optional, Tuple

import torch


def _quantize_tau_to_idx(tau: float, N: int) -> int:
    """Map continuous τ ∈ [0, 1] to a valid embedding index in [0, N-1]."""
    return max(0, min(N - 1, int(round(tau * N))))


class HybridChunkSampler:
    """Hybrid chunk-of-m KV-cached rollout sampler. See module docstring."""

    def __init__(
        self,
        denoiser,                          # DenoiserWrapper-like
        cfg,                                # full cfg (needs cfg.denoiser.{num_noise_levels,context_length})
        chunk_size: int,                    # m
        commit_per_chunk: int,              # k ∈ [1, m]
        num_diffusion_steps: int,           # K denoising steps per chunk
        action_noise_std: float = 1.0,
        commit_noise_n: float = 0.0,        # noise level (n ∈ [0, 1]) added to committed K/V for stability
        clean_commit_pass: bool = True,     # do an extra forward pass with the clean chunk to commit
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        assert 1 <= commit_per_chunk <= chunk_size, (
            f"commit_per_chunk={commit_per_chunk} must be in [1, chunk_size={chunk_size}]"
        )
        self.denoiser = denoiser
        self.cfg = cfg
        self.m = int(chunk_size)
        self.k = int(commit_per_chunk)
        self.K = int(num_diffusion_steps)
        self.action_noise_std = float(action_noise_std)
        self.commit_noise_n = float(commit_noise_n)
        self.clean_commit_pass = bool(clean_commit_pass)
        self.device = device
        self.dtype = dtype

        ctx_len = int(cfg.denoiser.context_length)
        # Cache capacity = context_length - m so cache_len + m ≤ context_length
        # at all times. Keeps the per-attention mask trivially full (no per-query
        # windowing required). See module docstring for the rationale.
        self.cache_capacity = ctx_len - self.m
        assert self.cache_capacity > 0, (
            f"chunk_size ({self.m}) must be strictly less than "
            f"cfg.denoiser.context_length ({ctx_len})"
        )

        self.N_noise = int(cfg.denoiser.num_noise_levels)
        self.N_lat = int(cfg.denoiser.num_latent_tokens)
        self.D_lat = int(cfg.denoiser.latent_dim)
        self.n_act = int(cfg.denoiser.n_actions)

        # Loud warning: this sampler assumes the model was trained with
        # horizon-aware temporal attention (bidir-within-horizon + causal-into-
        # context). If the checkpoint has horizon_aware=False, the model was
        # only trained on purely-causal temporal attention — using this sampler
        # against it puts the model in a regime it has NEVER seen at training
        # time. Predictions will be out-of-distribution and quality is unlikely
        # to be meaningful. Surface this explicitly rather than silently.
        ha = bool(getattr(cfg.denoiser, "horizon_aware", False))
        if not ha:
            msg = (
                "[HybridChunkSampler] cfg.denoiser.horizon_aware=False — the "
                "model was trained with purely-causal temporal attention. This "
                "sampler uses bidirectional attention within the active chunk, "
                "which the model has NOT been trained on. Predictions are "
                "out-of-distribution and quality may degrade significantly. "
                "To use this sampler as intended, train (or fine-tune) with "
                "horizon_aware=True. Override with cfg.denoiser.horizon_aware=True "
                "or set self.suppress_horizon_warning=True to silence."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            print(msg)  # also stdout — the warnings module is easy to miss
        self.suppress_horizon_warning = False

        # State
        self._cache_initialized = False
        self._cache_len = 0   # number of frames currently in cache
        self._batch_size = None

    # ----------------------------------------------------------------------
    # Cache setup
    # ----------------------------------------------------------------------
    def init_cache(self, batch_size: int):
        """Allocate KV caches in every temporal layer of the denoiser."""
        self.denoiser.init_cache(
            batch_size=batch_size,
            device=self.device,
            context_length=self.cache_capacity,
            dtype=self.dtype,
        )
        self._cache_initialized = True
        self._cache_len = 0
        self._batch_size = int(batch_size)

    @torch.no_grad()
    def warm_up_cache(
        self,
        seed_latents: torch.Tensor,           # (B, T_seed, N_lat, D_lat) clean tokenizer latents
        seed_actions: torch.Tensor,           # (B, T_seed, n_act)        clean actions
        is_horizon_per_seed_frame: Optional[torch.Tensor] = None,
    ):
        """Commit `T_seed` clean seed frames into the cache.

        Internally split into sub-chunks of size at most `m` so callers
        don't have to. Each sub-chunk is committed via a single
        `forward_chunk_step(commit_first_k = sub_T)` pass; later sub-chunks
        read the earlier ones from cache (causal grounding).

        `is_horizon_per_seed_frame` defaults to `zeros(T_seed)` (training-
        time context semantics — what `horizon_aware=True` saw for context
        frames in the step r-profile). If supplied, must be of length
        `T_seed` and matches per-frame.

        Distribution-shift caveat for m > 1: within each m-frame sub-chunk
        the seed frames temporarily get **bidirectional** attention (the
        hybrid path's only mode), even though they'll be used as **causal
        context** by later chunks. For `m = 1` this is moot (each sub-chunk
        has T=1, attention is degenerate single-key). For `m > 1` the
        committed K/V are slightly off from what a purely-causal seed
        computation would produce — small in practice, but worth knowing.
        If you need exact causal-seed K/V, run warm-up with a fresh
        sampler instance configured with `chunk_size = 1`.
        """
        assert self._cache_initialized, "Call init_cache(batch_size) first."
        B, T_seed, _, _ = seed_latents.shape
        assert self._cache_len + T_seed <= self.cache_capacity, (
            f"Seed would overflow cache: {self._cache_len} + {T_seed} > {self.cache_capacity}"
        )
        if is_horizon_per_seed_frame is not None:
            assert is_horizon_per_seed_frame.shape == (T_seed,), (
                f"is_horizon_per_seed_frame must be shape ({T_seed},), got "
                f"{tuple(is_horizon_per_seed_frame.shape)}"
            )

        seed_latents = seed_latents.to(device=self.device, dtype=self.dtype)
        seed_actions = seed_actions.to(device=self.device, dtype=self.dtype)
        clean_tau_idx = self.N_noise - 1  # τ ≈ 1 (clean)

        start = 0
        while start < T_seed:
            end = min(start + self.m, T_seed)
            sub_T = end - start
            z_sub = seed_latents[:, start:end]
            a_sub = seed_actions[:, start:end]

            if is_horizon_per_seed_frame is None:
                is_horizon = torch.zeros(sub_T, dtype=torch.long, device=self.device)
            else:
                is_horizon = is_horizon_per_seed_frame[start:end].to(
                    device=self.device, dtype=torch.long,
                )

            sigma_idx = torch.full(
                (B, sub_T), clean_tau_idx, dtype=torch.long, device=self.device,
            )
            step_idx = torch.zeros(
                (B, sub_T), dtype=torch.long, device=self.device,
            )

            self.denoiser.forward_chunk_step(
                noisy_act=a_sub,
                noisy_obs=z_sub,
                obs_sigma_idx=sigma_idx,
                obs_step_idx=step_idx,
                act_sigma_idx=sigma_idx,
                act_step_idx=step_idx,
                start_step_idx=self._cache_len,
                commit_first_k=sub_T,
                is_horizon=is_horizon,
            )
            self._cache_len += sub_T
            start = end

    # ----------------------------------------------------------------------
    # Single chunk
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Denoise one m-frame chunk and commit the first k frames.

        Returns:
            (committed_latents, committed_actions)
              shapes  (B, k, N_lat, D_lat), (B, k, n_act)
        """
        assert self._cache_initialized
        if self._cache_len + self.k > self.cache_capacity:
            # If commit would overflow, the cache's roll-and-insert logic
            # in `append_partial` will drop the oldest frames — but
            # cache_len doesn't track that drop on this side. Resync.
            overflow = (self._cache_len + self.k) - self.cache_capacity
            self._cache_len -= overflow

        B = self._batch_size
        m, K, N = self.m, self.K, self.N_noise
        device, dtype = self.device, self.dtype

        # --- Initialize chunk at full noise (τ = 0) ---
        z0 = torch.randn(B, m, self.N_lat, self.D_lat, device=device, dtype=dtype)
        a0 = self.action_noise_std * torch.randn(
            B, m, self.n_act, device=device, dtype=dtype,
        )
        z = z0.clone()
        a = a0.clone()

        # Per-frame step_idx (flow mode → all zeros), and the all-ones
        # is_horizon flag matching training-time step-profile horizon frames.
        step_idx = torch.zeros((B, m), dtype=torch.long, device=device)
        is_horizon = torch.ones(m, dtype=torch.long, device=device)

        # --- K-step Euler integration in τ space (τ from 0 → 1) ---
        cur_tau = 0.0
        dt_tau = 1.0 / K

        for k_step in range(K):
            sigma_idx = torch.full(
                (B, m), _quantize_tau_to_idx(cur_tau, N),
                dtype=torch.long, device=device,
            )
            z_hat, a_hat, _ = self.denoiser.forward_chunk_step(
                noisy_act=a,
                noisy_obs=z,
                obs_sigma_idx=sigma_idx,
                obs_step_idx=step_idx,
                act_sigma_idx=sigma_idx,
                act_step_idx=step_idx,
                start_step_idx=self._cache_len,
                commit_first_k=0,             # no commit during denoising
                is_horizon=is_horizon,
            )
            a_hat = a_hat.squeeze(-2)         # (B, m, n_act)

            denom = max(1.0 - cur_tau, 1e-5)
            z = z + (z_hat - z) / denom * dt_tau
            a = a + (a_hat - a) / denom * dt_tau
            cur_tau += dt_tau

        # At this point z ≈ z_clean, a ≈ a_clean (up to K-step integration error).

        # --- Optional commit-noise contamination (stability for long rollout) ---
        if self.commit_noise_n > 0.0:
            tau_commit = 1.0 - self.commit_noise_n
            commit_noise_z = torch.randn_like(z)
            commit_noise_a = self.action_noise_std * torch.randn_like(a)
            z_commit = (1.0 - tau_commit) * commit_noise_z + tau_commit * z
            a_commit = (1.0 - tau_commit) * commit_noise_a + tau_commit * a
            tau_commit_idx = _quantize_tau_to_idx(tau_commit, N)
        else:
            z_commit = z
            a_commit = a
            tau_commit_idx = N - 1   # clean

        # --- Commit the first k frames' K/V to cache ---
        if self.clean_commit_pass:
            # Extra forward pass with the (clean-ish) committed chunk so the
            # K/V written to cache reflect clean tokens. The forward output
            # is discarded; only the cache append matters.
            sigma_idx_commit = torch.full(
                (B, m), tau_commit_idx, dtype=torch.long, device=device,
            )
            self.denoiser.forward_chunk_step(
                noisy_act=a_commit,
                noisy_obs=z_commit,
                obs_sigma_idx=sigma_idx_commit,
                obs_step_idx=step_idx,
                act_sigma_idx=sigma_idx_commit,
                act_step_idx=step_idx,
                start_step_idx=self._cache_len,
                commit_first_k=self.k,
                is_horizon=is_horizon,        # ones; the model's frame_id row 1
                                              # was what it saw at training too
            )

        self._cache_len = min(self._cache_len + self.k, self.cache_capacity)
        return z[:, :self.k], a[:, :self.k]

    @torch.no_grad()
    def generate(self, n_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate `n_frames` total frames by repeatedly calling `step()`.

        Cache must already be initialized (and optionally warmed up). Returns
        `(latents, actions)` of shape `(B, n_frames, …)`.

        n_frames need not be a multiple of `commit_per_chunk`; the last
        chunk is truncated. Note: the truncated chunk still does a full
        K-step denoise — there's no fastpath for partial-emission.
        """
        assert self._cache_initialized
        out_z, out_a = [], []
        emitted = 0
        while emitted < n_frames:
            need = n_frames - emitted
            z_chunk, a_chunk = self.step()
            if need < z_chunk.shape[1]:
                z_chunk = z_chunk[:, :need]
                a_chunk = a_chunk[:, :need]
            out_z.append(z_chunk)
            out_a.append(a_chunk)
            emitted += z_chunk.shape[1]
        return torch.cat(out_z, dim=1), torch.cat(out_a, dim=1)
