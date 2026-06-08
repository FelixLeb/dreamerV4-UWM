# Architecture (denoiser)

A **flow-matching denoiser with two independent channels** — a
latent/state channel and an action channel — sharing one transformer. It does
**x-prediction** (regresses the clean signal): `forward(...)` returns
`(obs_output, act_output, pred_rewards|None)` where `obs_output` is the
predicted clean latent and `act_output` the predicted clean action. When needed ,the reward head is trained during post training where the base model parameters are kept fixed and only the reward head is trained. 

## Per-frame token sequence

Each timestep is expanded into a token set, concatenated along the token axis
(`dynamics.py:396`):

`[ Z (latent tokens) | Reg (register tokens) | IC (obs control) | AC (act control) | A (action token) ]`  (+ optional agent token).

- **Z** — `num_latent_tokens` (256 for 256x256 images) latents, projected `latent_dim (32)→model_dim`
  via `latent_projector`; output read back via `obs_projector` (`model_dim→latent_dim`).
- **Reg** — `num_register_tokens` (4) learned scratchpad tokens.
- **IC / AC** — the **independent per-frame, per-modality noise-control tokens**.
  Each is built by concatenating a **diffusion-τ embedding** and a **shortcut-step
  embedding** and projecting `2·model_dim→model_dim`. `obs_*`
  embedders drive IC, `act_*` drive AC — so state and action noise levels are set
  independently per frame. (`DiscreteEmbedder` = a learned embedding table; τ index
  ∈ `0..num_noise_levels-1`, shortcut index ∈ `0..log2(num_noise_levels)`.)
- **A** — one action token (`num_action_tokens` is 1), `action_input_proj`
  (`n_actions→model_dim`), output via `action_projector` (`model_dim→n_actions`).
  `n_actions`=22 (G1), 2 (pushT). 

τ convention: **τ=1 clean, τ=0 pure noise**, forward mix
`x_τ = (1−τ)·x0 + τ·x_clean`. The presence of both **diffusion** and **shortcut**
embedders means the architecture natively supports a shortcut/consistency
objective (dyadic step index); plain flow-matching just pins step index to 0.

## Transformer stack (axial: spatial + temporal)

`n_layers` blocks (`EfficientTransformerBlock`), each expanding to a sequence of
`EfficientTransformerLayer`s per `layer_types` (e.g. `[spatial, temporal,
spatial, temporal]`). Each layer is pre-norm:
`RMSNorm → AxialAttention → +residual`, then `RMSNorm → SwiGLU FFN → +residual`
(`blocks.py:618`).

- **AxialAttention** runs attention along one chosen axis of `(B, T, S, D)`:
  - **spatial** layers attend over the token axis `S` *within a frame* (`dim=2`),
  - **temporal** layers attend over the frame axis `T` *at each token position* (`dim=1`).
- Inner `Attention`: **GQA** (`n_kv_heads ≤ n_heads`), **RoPE** positions, optional
  QK-norm, SDPA (flash disabled by default → math/efficient backend, which is why
  arbitrary masks work).

## Attention masking

- **Spatial mask**: `None` (full intra-frame attention) unless the reward head is
  on, where an **agent-isolation mask** (`build_agent_isolation_mask`,
  `dynamics.py:160`) makes the agent token read-only.
- **Temporal mask** has three uncached paths, chosen at call time:
  1. causal rolling window (`context_length`) — this caps the attention span the past context_length frames and is used when training the model on longer than the context frames for avoiding overfitting to first frames. It is not used at the moment.;
  2. **horizon-aware** mask (causal-into-context + bidirectional-within-horizon),
     gated by `is_horizon` + `horizon_aware` and not used at the moment;
  3. a **caller-supplied `(T,T)` `temporal_attn_mask`** that overrides the others —
     used for **block-causal** (bidirectional within a block, strictly causal across
     blocks).
- **Cached inference**: `forward_step` (single/few frames, KV-cache, causality by
  cache construction) and `forward_chunk_step` (an `m`-frame chunk: bidirectional
  within the chunk + causal to the cache → block-causal rollout). `init_cache`
  allocates a per-temporal-layer `KVCache` (rolling buffer with
  `update`/`append_partial`/`no_update`).

## Optional reward head

When `train_reward_model=True`: a read-only **agent token** is appended per frame
and a **`RewardMTPHead`** predicts `mtp_length` future rewards as **symlog two-hot**
distributions (`SymlogTwoHotHead`). Off by default → no extra tokens/params,
checkpoints bit-compatible.
