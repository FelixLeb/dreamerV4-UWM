# MCTS planning-diagnostics study — runbook

Implements the sweep in [`../mcts_study_plan.md`](../mcts_study_plan.md). Builds many
MCTS trees over a factor grid, computes a diagnostic metric vector + in-model outcome
per tree, and writes CSV shards for analysis. Metrics/rationale: [`../../mcts_planning_diagnostics.tex`](../../mcts_planning_diagnostics.tex).

## Package
| file | role |
|---|---|
| `model.py` | load denoiser+tokenizer (bf16), decode fn |
| `data.py` | sample reproducible initial contexts from real PushT shards |
| `descriptors.py` | `StateDescriptor` — T-pose (PushT) + env-agnostic fallbacks |
| `metrics.py` | `compute_tree_metrics` → ~47 tree metrics (families A–E + tree-internal F) |
| `baselines.py` | in-model outcomes `g_rand`, `g_policy` |
| `run_tree.py` | `run_one_tree` → one flat CSV row |
| `run_sweep.py` | expand grid → job list → CSV shard (array-sliced, resumable) |
| `curate.py` | descriptor-assisted proposer → diverse curated init set (YAML + contact sheet) |
| `config/sweep.yaml` | full two-tier sweep; `config/smoke.yaml` tiny local check |
| `config/inits/pushT_curated.yaml` | 40 curated `(window_idx, t0)` decision points |

## Curate initial contexts (once, GPU-free)
Instead of random sampling, the sweep uses a fixed **curated** set (`inits.mode: file`) so
OFAT comparisons are *paired* across configs and deliberately span the situation space.
Regenerate it by farthest-point-selecting over real decision frames' T-pose + start-reward:
```bash
$PY -m dreamerv4uwm.planning.experiments.study.curate \
    --config .../config/sweep.yaml --n-candidates 800 --n-select 40
```
Writes `config/inits/pushT_curated.yaml` (with the dataset params pinned + a `start_reward`
fingerprint per init — the loader warns on drift) and a `.png` contact sheet to prune from.
`window_idx` is a dataset **index**, not a seed.

## Run locally
```bash
PY=/home/mim-server/miniconda3/envs/dreamerv4uwm/bin/python   # bf16; needs a free GPU
# tiny end-to-end check (3 configs x 2 inits):
$PY -m dreamerv4uwm.planning.experiments.study.run_sweep \
    --config dreamerv4uwm/planning/experiments/study/config/smoke.yaml --task-id 0 --num-tasks 1
# one task of the full sweep, OFAT only:
$PY -m ...run_sweep --config .../config/sweep.yaml --task-id 0 --num-tasks 1 \
    output_dir=/scratch/mcts_run1 sweep.random.n_samples=0
```
CLI overrides are OmegaConf dotlist (`key=value`), e.g. `n_inits=10 base_plan.horizon=28`.
Re-running the same task resumes (already-done `(config_id, init_id)` rows skipped).

## Run on NYU HPC (SLURM)
`../../hpc/slurms/mcts_sweep.slurm` — array job, one GPU/task, each task loads the model
once and processes a strided slice, appending to `shard_<task>.csv`.

1. Ship code + checkpoints + data to scratch (`hpc-transfer`); build/confirm the apptainer
   overlay (`hpc-overlay`).
2. **Fill in the `EDIT THESE` block** in the `.slurm`: `OVERLAY`, `SIF`, `ENV`, `PROJECT_DIR`,
   the cluster `DYN_CKPT`/`TOK_CKPT`/`DATA_DIR`, and `OUTPUT_DIR`. Keep `--array` size ==
   `NUM_TASKS`. *(These are the open items #3 in the plan — confirm before submitting.)*
3. Calibrate `--time` (below), then `sbatch mcts_sweep.slurm`.

### Resources & rationale
- **GPU:** 1/task, **80 GB** (a100/h100) recommended. bf16 is mandatory (fp32 OOMs). One tree at
  a time = modest VRAM, but 80 GB is the safe headroom for `pushT-large` denoiser+tokenizer + the
  reward/descriptor decode batches. 40 GB likely works — confirm on a single task.
- **CPU:** `--cpus-per-task=10`. The reward and the T-pose descriptor run **opencv segmentation on
  CPU**, one call per decoded frame — CPU is the throughput bottleneck, not the GPU.
- **RAM:** 48–64 GB.
- **Parallelism:** embarrassingly parallel, no inter-task comms. Array size = number of GPUs you
  want; the job list is split strided so each task gets a balanced mix of configs.

### Timing / budget
Measured: model load ≈ 8 s; a *tiny* tree (`horizon=6, n_iter=8`) ≈ 1 s steady-state. A full
`base_plan` tree (`horizon=20, n_iter=32, sim_horizon=20`) was **not** timeable locally (the shared
98 GB card was full from other jobs) — **run a 10-tree timing task first** and set `--time`
accordingly. Rough plan-doc estimate: Tier-1 OFAT (~30 cfg × 20 init ≈ 600 trees) + Tier-2 random
(`sweep.random.n_samples=500` × 20 ≈ 10k trees); at ~30 s/tree that's ~90 GPU-h → 16 tasks ≈ 6–7 h.

## Output
`shard_<task>.csv`, one row per tree: `config_id, config_tag, window_idx, t0, init_id`, all
`PlanConfig` factors, timing, outcome (`tree_peak, g_rand, g_policy, delta_over_root, …`), and every
tree metric. Concatenate shards for analysis.

## Analyse (after the sweep)
```bash
$PY -m dreamerv4uwm.planning.experiments.study.analysis.report \
    --input-dir /scratch/mcts_sweep/run1 --out-dir /scratch/mcts_sweep/run1/analysis
```
Produces `trees_all.parquet`, correlation tables (`metric_outcome.csv`, `factor_outcome.csv`,
`importance_*.csv`), `figures/` (response curves, correlation heatmap, mechanism scatter, per-link
importance bars), and `REPORT.md` (ranked early-warning metrics for `g_policy`/`g_rand`). The
`analysis/` modules also run standalone (`aggregate`, `correlate`, `plots`).

## Status / next
Built & verified: **M0–M3, M5** (see plan §9). **M4** SLURM is a template pending cluster paths.
Next: run the real sweep, then **M6** P1/P2 metrics (fidelity E1–E3 → unlocks the diversity×fidelity
2×2, `value_norm` treatment to test the L4 catch, centroid caching to drop the extra decode).
