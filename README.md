# Unified World Model (UWM) — Dreamer-V4 + action flow channel

> **Status: active research, exploratory phase.** This repository extends Dreamer-V4 with a second flow-matching channel dedicated to robot actions. We are currently iterating on the architecture and training protocol to find a single checkpoint that supports three inference-time roles from one model. Expect rough edges; see [`docs/experiments.md`](docs/experiments.md) for the running log.

## What's new vs. vanilla Dreamer-V4

Vanilla Dreamer-V4 denoises image latents conditioned on ground-truth actions. In UWM we add a **second independent flow-matching channel for the actions themselves**, controlled by its own per-frame noise-level embedding and control token inside the same transformer. The denoiser sees **independent (τ_state, τ_action) schedules per frame** and is trained to produce x-predictions for both streams.

This design makes the same trained checkpoint usable in three distinct operating modes, simply by choosing the right (τ_state, τ_action) schedule at inference time:

1. **World model.** Clean action stream drives the rollout of noisy states → classical forward dynamics.
2. **Policy.** Clean context, then *joint* generation of (state, action) on the horizon → action-conditioned rollouts where the model also proposes the actions.
3. **Random action proposer.** Noisy context → joint (state, action) generation. Intended as a high-entropy seeder for downstream planners / search methods.

The current research question is: **what training recipe makes a single model competent in all three modes simultaneously?**

## Repository status

- The **causal tokenizer** is trained and frozen; we reuse checkpoints across dynamics experiments.
- The **dynamics denoiser** (with the new dual channel) is the active area — see `dreamerv4uwm/models/dynamics.py` and `dreamerv4uwm/loss.py`.
- **Two training entrypoints**:
  - `scripts/train_dynamics_uwm.py` — plain flow matching (primary loop).
  - `scripts/train_dynamics_uwm_with_shortcut.py` — dyadic shortcut-bootstrap variant with EMA teacher.
- **Live experiment log**: [`docs/experiments.md`](docs/experiments.md). Each slurm file under `hpc/slurms/dynamics/pushT/{no-shortcut,with-shortcut}/` corresponds to a named experiment; the log captures hypothesis / setup / result / takeaway for each.
- **No interactive demo scripts are currently checked in.** Prior `play-*.py` scripts referenced in earlier documentation are not part of this branch.

## Training modes

The training loop (in `loss.py`) samples one mode per batch with probabilities set by `train.mode_weights`. Each mode is a preset (τ_state, τ_action) schedule that exercises a specific inference-time regime or teaches a prerequisite skill:

| Mode | What it teaches |
|---|---|
| `policy` | Joint (state, action) generation on horizon from clean context → target mode 2. |
| `wm` | State rollout under clean actions → target mode 1. |
| `id` | Inverse dynamics (actions from clean states) → auxiliary. |
| `video` | Unconditioned video generation from pure noise → prerequisite for target mode 3. |
| `forcing` | Progressive temporal denoising (per-frame linear τ) → grounding for long rollouts from corrupted context. |

Modes can be mixed; the best stable recipe so far is `{policy: 1, wm: 1}` (modes 1 & 2 work, mode 3 degrades). Current experiments are investigating how adding `video` and `forcing` training signal, along with different loss weightings (`ramp` vs. `uniform`) and progressive-noising schedules, affects mode 3 without regressing 1 & 2. See `docs/experiments.md` for current results.

## Installation

```bash
conda env create -f environment.yml
conda activate dreamerv4uwm
pip install -e .
```

## Training

All training scripts use **Hydra** (config root `scripts/config/`) and **torchrun**.

### Dynamics — flow matching (primary)

```bash
torchrun --standalone --nproc_per_node=<N> scripts/train_dynamics_uwm.py \
  --config-path scripts/config --config-name dynamics/pushT \
  dataset.data_dir=/path/to/sharded \
  tokenizer_ckpt=/path/to/tokenizer.pt \
  train.mode_weights.wm=1 train.mode_weights.policy=1 \
  train.loss_weighting=ramp
```

### Dynamics — shortcut bootstrap

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

(The tokenizer training script uses `grad_accum_steps`; the dynamics training scripts use `accum_grad_steps`. Keep them straight.)

### SLURM

Jobs are submitted one experiment at a time via `sbatch hpc/slurms/dynamics/pushT/<no-shortcut|with-shortcut>/<experiment>.slurm`. Cloning an existing file and tweaking hydra overrides is the standard way to launch a new experiment.

## Dataset format

Expected on-disk layout:

```
<data_dir>/
  metadata.json
  shard_0000.h5
  shard_0001.h5
  ...
```

Each shard contains:

- `images`: `(num_episodes, max_len, H, W, C)` `uint8`
- `actions`: `(num_episodes, max_len, action_dim)` `float32`
- `episode_lengths`: `(num_episodes,)` `int32`
- HDF5 attrs: `num_episodes`, `max_length`

Preprocessing scripts under `scripts/preprocessing/`:

- `huggingface_to_hdf5.py` — LeRobot / HuggingFace datasets
- `rlds_to_hdf5.py` — RLDS / TFDS datasets

See each script's `--help` for options (subsampling FPS, resize target, relative vs. absolute actions, etc.).

## Conventions & gotchas

- **τ = 1 clean, τ = 0 noisy**, with forward mixing `x_τ = (1-τ) x_0 + τ x_clean`. This is opposite to the usual diffusion convention. The default `ramp` loss weighting `w(τ) = 0.9τ + 0.1` therefore *down-weights* the noisy end of the schedule; use `uniform` when you specifically need gradient in the very-noisy regime.
- **bf16 autocast** is used for both the (frozen) tokenizer encode and the (trainable) denoiser.
- **Action shape.** Dataset provides `(B, T, A)`; the denoiser expects `(B, T, 1, n_actions)`. Training scripts slice and `unsqueeze(-2)` — keep `num_action_tokens=1`.
- **Checkpoint resume restores model weights only.** Optimizer and LR scheduler state are not reloaded; cosine schedule restarts.

## Contributing to the log

When a run finishes or a hypothesis is ruled out, append a terse entry to [`docs/experiments.md`](docs/experiments.md) in the format *hypothesis · setup (config + slurm file reference) · result · takeaway*. This is the only persistent record of what has been tried, so keep it current.

## Citation

Built on top of Dreamer-V4:

```
@article{hafner2025training,
  title={Training agents inside of scalable world models},
  author={Hafner, Danijar and Yan, Wilson and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2509.24527},
  year={2025}
}
```

## Acknowledgement

Computational resources from [NYU Torch](https://www.nyu.edu/life/information-technology/research-computing-services/high-performance-computing/high-performance-computing-nyu-it.html), LAAS-Gepetto, ANITI, and [Jean Zay](http://www.idris.fr/eng/jean-zay/jean-zay-presentation-eng.html). Developed at [Machines in Motion Lab (MiM)](https://www.machinesinmotion.org/), [NYU CREO](https://engineering.nyu.edu/research/centers/nyu-center-robotics-and-embodied-intelligence-creo), with equal contribution from [Joseph Amigo](https://scholar.google.com/citations?user=-PPor9IAAAAJ&hl=en) and [Rooholla Khorrambakht](https://scholar.google.com/citations?user=VdgZUjoAAAAJ&hl=en).
