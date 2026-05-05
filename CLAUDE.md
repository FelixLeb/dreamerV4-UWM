# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is (read this first)

This is **active research**, not a polished codebase. The repo extends Dreamer-V4 with a second independent flow-matching channel dedicated to robot actions, alongside the original latent/state channel. The single denoiser exposes **independent per-frame noise levels for state and action**, so inference-time choices of (τ_state, τ_action) select which of three target roles the same network plays:

1. **World model** — clean context + clean actions drive rollout of noisy states.
2. **Policy** — clean context → joint generation of (state, action) on the horizon.
3. **Random action proposer** — *noisy* context → joint (state, action) generation with high entropy to cover a search space for downstream planners.

The project is currently in the **architecture & training-protocol exploration phase**. Roohollah is running small-scale ablations on NYU HPC to find a training recipe that supports all three modes from a single checkpoint.

**Best recipe so far.** `mode_weights = {policy: 1, wm: 1}` gives solid modes 1 & 2 but fails mode 3. The current research effort targets mode 3 specifically — see `docs/experiments.md` for the live log and `memory/project_current_focus.md` for current hypotheses.

## Collaboration workflow

The loop is: **identify → brainstorm/research → design experiment → submit slurm → log result**. Stay inside that loop:

- **Propose, don't refactor.** Treat ideas as hypotheses for discussion. Don't restructure code, extract abstractions, or "clean up" files unless Rooholla explicitly asks. This codebase is in a quick-iteration state on purpose.
- **Log results to `docs/experiments.md`.** When a run finishes, a hypothesis is ruled out, or an insight crystallizes, offer to append a terse entry: *hypothesis · setup · result · takeaway*. One section per run. Don't create new ad-hoc notes files.
- **New experiments are cloned slurm scripts.** The pattern is: copy the nearest `hpc/slurms/dynamics/pushT/**/*.slurm`, rename, tweak hydra overrides (usually `train.mode_weights.*`, `train.loss_weighting`, `train.forcing_context_noise.*`, `train.shortcut.*`). Don't invent a new runner.
- **Always ask "what does this do to modes 1 and 2?"** The risk on any proposed fix for mode 3 is silent regression on world-model or policy behavior. Make regression-checking explicit.

## Architecture essentials

### Dual flow-matching channels
`dreamerv4uwm/models/dynamics.py:DreamerV4Denoiser` is a causal transformer over a per-frame token sequence `[Z, Reg, IC, AC, A]`:
- `Z` = latent tokens from the frozen tokenizer (`num_latent_tokens`, typically 256).
- `Reg` = learned register / scratchpad tokens.
- `IC` = **image control token** — a single token carrying (τ_state, shortcut_step_index) encoded by `obs_diffusion_embedder` + `obs_shortcut_embedder`.
- `AC` = **action control token** — same structure but for actions (`act_diffusion_embedder`, `act_shortcut_embedder`). **This is the key extension** — it lets state and action noise be controlled independently per frame.
- `A` = the noisy action token(s). `num_action_tokens` is enforced to 1; `n_actions` is the action-vector dimension.

Intra-frame attention follows `build_spatial_attention_mask` (see `dynamics.py`): `Z → Z, Reg, IC` (+ `AC, A` only when `latent_attends_action=true`); `A → everything`; `IC`/`AC` are write-only controls. In practice the current code applies no spatial mask (`spatial_mask=None` in `forward`) — the mask function is built but not used; worth keeping in mind if you reason from the docstring.

Temporal attention is causal with a window of `context_length`. Layer order is `cfg.denoiser.layer_types` (e.g. `[spatial, temporal, spatial, temporal]`).

### Noise-level conventions (read carefully — they're inverted)
- **τ = 1 is clean, τ = 0 is pure noise**, with forward mixing `x_tau = (1-τ) x0 + τ x_clean`. Flow matching in this repo is x-prediction (regress clean signal), not v-prediction.
- `loss_weight(τ, 'ramp')` returns `0.9τ + 0.1`, which **down-weights the noisy end** — the opposite of what you'd expect under the standard diffusion convention. Use `uniform` when you want gradient in the very-noisy regime.
- `num_noise_levels` is the grid size for τ; must be a power of 2 for the shortcut variant.

### Training modes (`loss.py`)
`UWMForwardProcess` samples one of `{policy, wm, id, video, forcing}` per batch with probabilities set by `train.mode_weights`. Each mode is a preset schedule over (state_τ, action_τ) on (context, horizon):

| Mode | Context state τ | Context action τ | Horizon state τ | Horizon action τ | Loss masked to |
|---|---|---|---|---|---|
| `policy` | clean | clean | sampled | sampled | horizon |
| `wm` | sampled (independent of hor) | clean | sampled | clean | full seq (state only) |
| `id` | clean | sampled | clean | sampled | full seq (action only) |
| `video` | sampled | noise | sampled (same τ as ctx) | noise | **full seq** (state only) |
| `forcing` | progressive linear | progressive linear | progressive linear | progressive linear (or noise if `forcing_mask_actions=true`) | full seq |

Additional knobs:
- `forcing_context_noise.{bias, alpha, beta}` — in `forcing` mode, re-samples context τ from `Beta(α, β)` with probability `bias`. `α < β` biases toward very-noisy contexts. Teaches generation from heavily-masked contexts.
- `forcing_mask_actions=true` — pins action τ to noise in `forcing` mode and zeroes its loss → video-only progressive forcing.
- `loss_weighting` — `ramp` (default) or `uniform`; see above for the τ-convention caveat.

### Shortcut variant (`train_dynamics_uwm_with_shortcut.py`)
`ShortcutUWMForwardProcess` + `compute_bootstrap_uwm_loss` add a dyadic shortcut-bootstrap objective. `step_index_raw = max_pow2` is the flow branch (direct x-prediction against clean target); lower step indices are bootstrap branches that target the mean of two half-step velocities from a teacher (EMA or the student in eval mode).
- `train.shortcut.flow_bias` — P(flow branch). Remaining mass is uniform over bootstrap rungs.
- `train.shortcut.ema_decay` — EMA decay for the teacher. `0.0` means teacher = student (self-bootstrap).

### Two denoiser training entrypoints (not one)
- `scripts/train_dynamics_uwm.py` — plain flow matching via `UWMForwardProcess` + `compute_uwm_loss`.
- `scripts/train_dynamics_uwm_with_shortcut.py` — shortcut variant with `update_ema` + `_unwrap` helpers for the EMA teacher.

Both are DDP + optional `torch.compile`. Config chain (Hydra, `@package _global_`): `dynamics/<name>.yaml` → `tokenizer/<name>.yaml` → `dataset/<name>.yaml`.

### Tokenizer
Trained separately with `scripts/train_tokenizer.py` (FSDP over `EfficientTransformerLayer`). During dynamics training the tokenizer is frozen in `eval` + `no_grad`; only the denoiser updates.

### Data
Sharded HDF5, one episode batch per shard: `metadata.json` + `shard_XXXX.h5` with `images (N,T,H,W,C) uint8` / `actions (N,T,A) float32` / `episode_lengths (N,) int32`. `scripts/preprocessing/{huggingface_to_hdf5.py, rlds_to_hdf5.py}` do conversion. `dreamerv4uwm/datasets.py:ShardedHDF5Dataset` is the loader.

## Commands

All training uses Hydra (`scripts/config/`) + `torchrun`. The tokenizer uses FSDP; dynamics use DDP.

### Dynamics — flow matching (primary research loop)
```bash
torchrun --standalone --nproc_per_node=<N> scripts/train_dynamics_uwm.py \
  --config-path scripts/config --config-name dynamics/pushT \
  dataset.data_dir=/path/to/sharded \
  tokenizer_ckpt=/path/to/tokenizer.pt \
  train.mode_weights.wm=1 train.mode_weights.policy=1 \
  train.loss_weighting=ramp \
  wandb.run_name=<experiment-name>
```

### Dynamics — with shortcut bootstrap
```bash
torchrun --standalone --nproc_per_node=<N> scripts/train_dynamics_uwm_with_shortcut.py \
  --config-path scripts/config --config-name dynamics/pushT \
  dataset.data_dir=... tokenizer_ckpt=... \
  train.shortcut.flow_bias=0.75 train.shortcut.ema_decay=0.999
```

### Tokenizer
```bash
torchrun --standalone --nproc_per_node=<N> scripts/train_tokenizer.py \
  --config-path scripts/config --config-name tokenizer/pushT \
  dataset.data_dir=/path/to/sharded \
  train.batch_per_gpu=1 train.grad_accum_steps=1
```

### Dataset conversion
- `scripts/preprocessing/huggingface_to_hdf5.py` — LeRobot / HF datasets
- `scripts/preprocessing/rlds_to_hdf5.py` — RLDS datasets

### SLURM
Submit via `sbatch hpc/slurms/dynamics/pushT/<no-shortcut|with-shortcut>/<experiment>.slurm`. Each slurm file is one experiment — clone the closest match and edit hydra overrides at the bottom of the file.

### Tests / lint
No test suite or lint config lives in this repo. Don't invent commands.

## Gotchas

- **No `play-*.py` scripts** in the current checkout, despite the README's legacy Quickstart section. If Rooholla asks to run a demo, confirm intent first.
- **Config key naming is inconsistent.** `train_dynamics_uwm.py` uses `train.accum_grad_steps`; `train_tokenizer.py` uses `train.grad_accum_steps`. Always check the specific script before copying a command.
- **`load_ddp_checkpoint` only restores model weights.** Optimizer + scheduler `load_state_dict` calls are commented out — resuming a run starts a fresh cosine LR cycle. Don't report "resumed optimizer state" if you're using this helper.
- **DDP + compile unwrap order.** `load_ddp_checkpoint` hard-codes `model.module._orig_mod.load_state_dict(...)`; if you change the DDP vs. `torch.compile` wrapping order in training scripts, update that call too.
- **Action tensor shape.** Dataset → `(B, T, A)`; denoiser → `(B, T, 1, n_actions)` via `actions[:, :, :cfg.denoiser.n_actions].unsqueeze(-2)`. `num_action_tokens=1` is assumed throughout loss code.

## Where to look

- **Live experiment log**: `docs/experiments.md`. Append here when runs complete or hypotheses change.
- **Older scratchpads**: `ongoing-runs.md` (repo root), `hpc/training-notes.md`. Migrate content to `docs/experiments.md` when refreshing; don't edit in place.
- **Slurm recipe bank**: `hpc/slurms/dynamics/pushT/{no-shortcut,with-shortcut}/*.slurm`. Each is a named experiment; file names describe the mode mix.
- **Claude memory** (cross-session context): user profile, project scope, current focus, and workflow feedback live in `~/.claude/projects/-scratch-rk4342-projects-dreamerV4-UWM/memory/`. Keep them in sync when focus shifts.
