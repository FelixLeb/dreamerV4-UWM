---
title: UWM (Unified World Model) — project context for external chat AIs
last_updated: 2026-04-17
---

# UWM project context

This document is a self-contained briefing on the **Unified World Model (UWM)** research project, intended to be pasted into an external chat AI as background context. It captures *what the project is, what we are currently trying to figure out, and the conventions a discussant needs to know to give useful feedback.* It is not a tutorial and not a code reference.

---

## 1. Project in one paragraph

UWM extends **Dreamer-V4** with a second, independent flow-matching channel dedicated to **robot actions**, alongside the original latent/state channel. A single causal-transformer denoiser is exposed to **independent per-frame noise levels** for state (τ_state) and action (τ_action). At inference time, the choice of (τ_state, τ_action) schedule selects which of three roles the same network plays. The research goal is to find a single training recipe that produces one checkpoint competent in **all three** roles simultaneously. The project is in the **architecture and training-protocol exploration phase**, run on small-scale ablations on NYU HPC.

## 2. The three target inference modes

| # | Mode | Context (τ_state, τ_action) | Horizon (τ_state, τ_action) | What it does |
|---|---|---|---|---|
| 1 | **World model** | clean, clean | noisy state, clean action | classical forward dynamics under given actions |
| 2 | **Policy** | clean, clean | noisy, noisy (joint) | model proposes both state and action on the horizon |
| 3 | **Random action proposer** | **noisy**, noisy | noisy, noisy (joint) | high-entropy joint generation; intended as a seeder for downstream planners / search |

(Convention: **τ = 1 means clean, τ = 0 means pure noise** — see §6.)

## 3. Current research status

**Best recipe so far.** `mode_weights = {policy: 1, wm: 1}` produces a model that is good at modes 1 and 2 but **degrades at mode 3** (noisy-context joint generation collapses temporal consistency / object permanence).

**Active focus.** Make mode 3 work without regressing modes 1 & 2. Current hypotheses being probed:

- **H1.** Adding the `video` training mode (unconditioned video generation from pure noise) teaches generation-from-noise, which should transfer to mode 3.
- **H1a.** Within video-mode training, *progressive per-frame τ* (frame 0 cleanest, last frame noisiest) beats a single τ shared across the sequence.
- **H1b.** *Uniform* loss weighting beats `ramp` weighting, because `ramp` (`0.9τ + 0.1`) systematically **down-weights the noisy end** under the inverted τ convention — i.e. it down-weights exactly the regime mode 3 cares about.
- **H2.** Whether the **shortcut bootstrap** variant helps any of the three modes, or only compresses inference-time cost.
- **H3.** Whether `latent_attends_action=true` (Z tokens seeing AC/A within the intra-frame mask) helps joint generation modes vs. keeping clean separation. Isolated ablation TBD.

**Mandatory regression check.** Any proposed fix for mode 3 must explicitly address: *what does this do to modes 1 and 2?* Silent regression on world-model or policy behavior is the main risk.

## 4. Architecture essentials

The denoiser (`dreamerv4uwm/models/dynamics.py:DreamerV4Denoiser`) is a **causal transformer over a per-frame token sequence**:

```
[ Z … Z   Reg … Reg   IC   AC   A ]
```

- `Z`: latent tokens from the **frozen tokenizer** (typically 256 tokens per frame).
- `Reg`: learned register / scratchpad tokens.
- `IC`: **Image Control** token — single token carrying `(τ_state, shortcut_step_index)` via diffusion + shortcut embedders.
- `AC`: **Action Control** token — same structure, but for actions (`τ_action`, action-shortcut step). **This is the key extension over vanilla Dreamer-V4.** It is what allows independent per-frame noise on state and action.
- `A`: the noisy action token(s). `num_action_tokens` is fixed to **1**; `n_actions` is the action-vector dimension.

**Intra-frame attention** (per `build_spatial_attention_mask`):
- `Z → {Z, Reg, IC}` (and additionally `AC, A` when `latent_attends_action=true`).
- `A → everything`.
- `IC, AC` are write-only control tokens (others read them; they read nothing of substance).

⚠️ **Implementation caveat.** The mask is built but currently passed as `spatial_mask=None` in `forward`, so in practice **no spatial mask is applied today**. Worth keeping in mind when reasoning from the docstring.

**Temporal attention** is causal with window `context_length`. Layer order is configurable: `cfg.denoiser.layer_types = [spatial, temporal, spatial, temporal, …]`.

## 5. Training modes (`dreamerv4uwm/loss.py`)

`UWMForwardProcess` samples one of these modes per batch with probabilities set by `train.mode_weights`. Each mode is a preset (state_τ, action_τ) schedule on (context, horizon) and a loss mask:

| Mode | ctx state τ | ctx action τ | hor state τ | hor action τ | Loss masked to |
|---|---|---|---|---|---|
| `policy` | clean | clean | sampled | sampled | horizon |
| `wm` | sampled (independent of hor) | clean | sampled | clean | full seq (state only) |
| `id` | clean | sampled | clean | sampled | full seq (action only) |
| `video` | sampled | noise | sampled (same τ as ctx) | noise | full seq (state only) |
| `forcing` | progressive linear | progressive linear | progressive linear | progressive linear (or noise if `forcing_mask_actions=true`) | full seq |

**Knobs that matter to current experiments:**

- `forcing_context_noise.{bias, alpha, beta}` — in `forcing` mode, with probability `bias`, re-samples context τ from `Beta(α, β)`. `α < β` biases context toward very-noisy. Used to teach generation from heavily-masked contexts.
- `forcing_mask_actions=true` — pins action τ to noise in `forcing` mode and zeroes its loss → video-only progressive forcing.
- `loss_weighting ∈ {ramp, uniform}` — see §6 for why this matters more than it looks.

## 6. Conventions to internalise (these trip people up)

- **τ = 1 clean, τ = 0 noisy.** Forward mixing is `x_τ = (1-τ) x_0 + τ x_clean`. **Opposite of standard diffusion convention.** Mentally invert when porting intuitions from elsewhere.
- **Flow matching here is x-prediction**, not v-prediction. The model regresses the clean signal.
- **`loss_weight(τ, 'ramp') = 0.9τ + 0.1`** therefore **down-weights the noisy end** — exactly opposite to what a standard diffusion practitioner would assume from the name. Switch to `uniform` if you want gradient signal in the noisy regime.
- **`num_noise_levels`** is the τ-grid size; must be a **power of 2** for the shortcut variant (dyadic bootstrap).
- **Action tensor shape.** Dataset → `(B, T, A)`; denoiser → `(B, T, 1, n_actions)` via `actions[:, :, :n_actions].unsqueeze(-2)`. `num_action_tokens = 1` is assumed throughout the loss code.
- **bf16 autocast** for both the (frozen) tokenizer encode and the (trainable) denoiser.

## 7. The shortcut variant (separate entrypoint)

`scripts/train_dynamics_uwm_with_shortcut.py` uses `ShortcutUWMForwardProcess` + `compute_bootstrap_uwm_loss`, adding a **dyadic shortcut-bootstrap** objective on top of flow matching:

- `step_index_raw = max_pow2` is the **flow branch** (direct x-prediction against the clean target).
- Lower step indices are **bootstrap branches** that target the mean of two half-step velocities from a teacher network (EMA copy of the student, or the student itself in eval mode if `ema_decay = 0`).
- `train.shortcut.flow_bias` — P(flow branch); the rest of the mass is uniform across bootstrap rungs.
- `train.shortcut.ema_decay = 0.0` ⇒ self-bootstrap (teacher = student).

## 8. Two training entrypoints (DDP + optional `torch.compile`)

- `scripts/train_dynamics_uwm.py` — plain flow matching (primary research loop).
- `scripts/train_dynamics_uwm_with_shortcut.py` — shortcut variant, with `update_ema` + `_unwrap` helpers.

Hydra config chain (every config has `@package _global_`): `dynamics/<name>.yaml` → `tokenizer/<name>.yaml` → `dataset/<name>.yaml`.

The **tokenizer** is trained separately by `scripts/train_tokenizer.py` (FSDP over `EfficientTransformerLayer`). During dynamics training the tokenizer is frozen (`eval` + `no_grad`).

## 9. Workflow & where things live

The iteration loop is: **identify → brainstorm/research → design experiment → submit slurm → log result.**

- **One experiment = one slurm script.** Located under `hpc/slurms/dynamics/pushT/{no-shortcut, with-shortcut}/`. Each `.slurm` file is a named experiment with its hydra overrides at the bottom. New experiments are produced by **cloning the nearest match and editing overrides** — typically `train.mode_weights.*`, `train.loss_weighting`, `train.forcing_context_noise.*`, `train.shortcut.*`. There is no general experiment runner.
- **Live experiment log:** `docs/experiments.md` — *the* source of truth for what has been tried. Format per run: **Hypothesis → Setup → Result → Takeaway.** Keep entries terse. Falsified hypotheses are labelled, not deleted.
- **Older scratchpads:** `ongoing-runs.md` (repo root) and `hpc/training-notes.md`. Migrate content into `docs/experiments.md` rather than editing them in place.
- **Code hotspots:** `dreamerv4uwm/models/dynamics.py` (denoiser), `dreamerv4uwm/loss.py` (forward processes + loss).

## 10. Data format

Sharded HDF5, one episode batch per shard:

```
<data_dir>/
  metadata.json
  shard_0000.h5
  shard_0001.h5
  ...
```

Each shard contains:
- `images`: `(N, T, H, W, C)` `uint8`
- `actions`: `(N, T, A)` `float32`
- `episode_lengths`: `(N,)` `int32`
- HDF5 attrs: `num_episodes`, `max_length`

Preprocessors: `scripts/preprocessing/{huggingface_to_hdf5.py, rlds_to_hdf5.py}`.

## 11. Operational gotchas

- **No `play-*.py` demo scripts** in the current checkout, despite the README's legacy Quickstart text.
- **Config key naming is inconsistent.** `train_dynamics_uwm.py` uses `train.accum_grad_steps`; `train_tokenizer.py` uses `train.grad_accum_steps`. Always check the specific script before copying a command line.
- **`load_ddp_checkpoint` only restores model weights.** Optimizer + LR-scheduler `load_state_dict` calls are commented out — resuming a run starts a **fresh cosine LR cycle**.
- **DDP + compile unwrap order is hard-coded.** `load_ddp_checkpoint` calls `model.module._orig_mod.load_state_dict(...)`. If the DDP vs. `torch.compile` wrapping order changes in training scripts, this call must be updated in lockstep.
- **`num_action_tokens = 1` is assumed** throughout the loss code; relaxing it touches both the dataset slice and the loss masks.

## 12. What kind of help is most useful

When discussing this project, useful contributions look like:
- Identifying *which training mode or schedule knob* is the right lever for a given symptom — and predicting its impact on **all three target modes**, not just the one being fixed.
- Catching when the inverted τ convention or the `ramp` weighting interpretation has been misapplied in a proposal.
- Suggesting concrete, *minimal-diff* experiments expressible as a hydra-override change to a cloned slurm script.
- Flagging silent-regression risk on modes 1 / 2 when proposing mode-3 fixes.

What is **not** useful: sweeping refactors, new abstractions, "cleaning up" the loss/dynamics files, or proposing alternative trainers. The codebase is in deliberate quick-iteration shape.
