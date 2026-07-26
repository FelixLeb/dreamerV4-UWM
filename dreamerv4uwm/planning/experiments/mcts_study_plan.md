# MCTS Planning Diagnostics — Experimental Study Plan

> **Purpose of this file.** This is the single source of truth for the study. When
> re-opened, it should tell me *exactly* what to build, in what order, and why.
> Companion theory doc: [`../mcts_planning_diagnostics.tex`](../mcts_planning_diagnostics.tex).
> Status legend: ☐ todo · ◐ in progress · ☑ done. Priorities: **P0** must-have,
> **P1** high-value, **P2** nice-to-have.

---

## 0. The question

> *Sometimes when the edges out of a node aren't diverse enough, MCTS planning
> "leads to nothing." Which factors produce good planning, and how do we detect a
> collapsing tree from cheap tree-level signals?*

Restated operationally: build **many trees** under varied conditions (initial
state, edge horizon, sim horizon, context noise, exploration constant, …),
compute a **vector of diagnostic metrics** on each tree plus an **outcome**, dump
everything to CSV, and then find **which metrics/knobs predict good outcomes** —
so we get (a) an early-warning signal for "this tree will plan to nothing" and (b)
a recipe for conditions that plan well.

**Design constraint (important):** the machinery must be **environment- and
reward-agnostic**. Today = PushT + `TCenterReward`, but everything is written
against pluggable interfaces (`RewardFn`, `StateDescriptor`, `Baselines`,
`Executor`) so a new env/reward is a config swap, not a rewrite.

---

## 1. Organizing principle — the causal chain

"Planning leads to nothing" is not one failure; it is a **break anywhere in a
5-link chain**. Each metric family tests one link. This is how we decide *which*
metrics matter and how we interpret results.

| Link | Question | Governing knobs | Metric family |
|---|---|---|---|
| **L1 Sampler** | Do the `B` sampled *action sequences* even differ? (pre-model) | `action_temp`, `ctx_noise` | Diversity — action (§4.D) |
| **L2 Model** | Do different actions → different *task-relevant outcomes*, not collapsed and not OOD fiction? | `horizon`, `ctx_noise`, `K_steps` | Diversity — outcome (§4.D) + Fidelity (§4.E) |
| **L3 Estimator** | Do those outcomes receive *distinguishable values*? | reward scale, `sim_horizon`, `gamma`, backup | Value quality (§4.C) |
| **L4 Selection** | Does UCT actually *act* on the value differences? | **`c_ucb` vs value scale** | Allocation (§4.B) |
| **L5 Decision** | Does the margin yield a confident, correct root choice? | budget `n_iterations` | Outcome (§4.F) |

**Prime suspect, independent of diversity (L4).** In [`mcts.py`](../mcts.py) the
simulation backs up `R = max over rollouts/prefixes of cumsum(γ^t·r)` — a
**cumulative sum** over up to `sim_horizon` frames, so `Q ∈ [0, ~20]` for
`sim_horizon≈28`. The UCB exploration bonus `c_ucb·√(ln N_p/N_c)` with
`c_ucb≈0.5` is `≈0.4–0.9`. Exploitation therefore dominates exploration by
**10–50×**: after each child's first visit, UCB1 is effectively **greedy** and
collapses onto the first lucky child — *even with perfectly diverse edges*. The
study must include `c_ucb` **and** a value-normalization treatment (§5) so we can
separate "L4 is broken" from "L1/L2 diversity is broken."

---

## 2. Metrics to compute

Not implementing everything in the `.tex`. This is the focused, high-value subset,
each tagged with the chain link, priority, and what it needs. **"Needs"** codes:
`tree`=final tree object · `trace`=per-iteration event log (`trace=True`) ·
`φ`=state descriptor · `M2`=per-node return sum-of-squares (small planner add) ·
`ref`=held-out real trajectories · `base`=baseline rollouts · `exec`=env executor.

### A. Structural / tree-shape — first-line collapse detection (cheap, always on)

| ID | Metric | How | Why (link) | Prio | Needs |
|---|---|---|---|---|---|
| A1 | `n_nodes`, `n_forward`, `expansion_efficiency` | counts; `(n_nodes-1)/n_forward` | budget spent on structure vs re-visits | P0 | tree |
| A2 | `max_depth`, `mean_visited_depth` | `Σ N(v)·depth(v)/Σ N(v)` | shallow-under-budget = search re-expands root, never looks ahead | P0 | tree |
| A3 | `eff_branching_factor` | realized mean visited children per internal node, vs nominal `B` | gap = "most generated edges never explored" = collapse signature | P0 | tree |
| A4 | `subtree_size_gini` | Gini over root-children subtree **sizes** (not just visits) | premature convergence; catches skew that *visit counts alone hide* ("second uncertainty") | P0 | tree |
| A5 | `width_profile` | node count per depth (store depths 1..K) | bushy-then-tapering (healthy) vs near-linear chain (degenerate) | P1 | tree |

### B. Search allocation / exploration–exploitation (L4–L5)

| ID | Metric | How | Why | Prio | Needs |
|---|---|---|---|---|---|
| B1 | `visit_entropy` (final + `entropy_auc`) | normalized `H(p)/ln k` over root children; AUC over iterations | healthy: high→concentrates; pathological: flat-high (indecision) or instant-0 (premature) | P0 | trace |
| B2 | `commit_top1` | max root-child visit share | cheap collapse scalar | P0 | tree |
| B3 | `visit_value_corr` | Spearman(`N_i`, `Q_i`) | **is the bandit working** — visits should flow to value | P0 | tree |
| B4 | **`exploit_explore_ratio`** | mean UCB explore term ÷ std of sibling `Q` | **the L4 catch** — ≪1 ⇒ effectively greedy; ≫1 ⇒ pure noise | **P0** | trace |
| B5 | `recommendation_agreement` | `argmax_i N_i == argmax_i Q_i` | disagreement ⇒ recommendation is tie-break luck | P1 | tree |
| B6 | `selection_depth_of_exploration` | deepest depth at which a non-greedy (non-argmax-Q) selection occurs | if exploration only fires at root, deep branches never diversify — key for short edges | P1 | trace |
| B7 | `best_action_switch_count`, `root_value_stability` | # argmax-root changes; var of root `Q` over last-k iters | still switching at budget end = under-converged (high simple-regret risk) | P1 | trace |

### C. Value quality (L3)

| ID | Metric | How | Why | Prio | Needs |
|---|---|---|---|---|---|
| C1 | `val_std`, `val_spread` | std / (max−mean) of root-children `Q` | `≈0` = crispest "planning is indifferent" | P0 | tree |
| C2 | `q_margin` | `(Q(1)−Q(2))` normalized by `val_std` | decision confidence; vanishing = inconclusive (distinct from collapse) | P0 | tree |
| C3 | `val_std_by_depth` | `val_std` over siblings at each interior depth | **discrimination decay** — does separation vanish deeper? | P1 | tree |
| C4 | `return_std` per node | needs running `M2` → `√(M2/N − Q²)` | separates "arms equal" from "arms unresolved" | P2 | M2 |

### D. Diversity (L1/L2) — the stated root cause, in **descriptor space** (not raw latent)

> **Lesson already learned:** raw latent `ℓ2` diversity is meaningless here
> (distance concentration in 8192-D + task-irrelevant nuisance variance). Measure
> diversity in a low-dim **descriptor** `φ(state)`. For `TCenterReward`, `φ` = the
> **T-pose** `(cₓ, c_y, θ, √area)` — *already computed inside the reward's
> segmentation and currently discarded*. For other rewards, `φ` falls back to
> (in order) a user-supplied descriptor → the scalar reward → a frozen
> vision-encoder embedding of the decoded frame. See `StateDescriptor` (§3).

| ID | Metric | How | Why (link) | Prio | Needs |
|---|---|---|---|---|---|
| D1 | `action_diversity` | mean pairwise `‖aᵢ−aⱼ‖` of sibling edges; `det(cov)` | **L1, pre-model** — is the sampler under-dispersed before the model even runs | P0 | tree |
| D2 | `outcome_diversity` | mean pairwise `‖φ(sᵢ)−φ(sⱼ)‖` over siblings; centroid hull area | **L2** — do edges reach genuinely different task states | P0 | tree, φ |
| D3 | `branch_collapse_index` (BCI) | `1 − N_eff/B`, `N_eff` = effective #clusters of sibling `φ` | the failure, as one number ∈[0,1] | P0 | tree, φ |
| D4 | `duplicate_rate` | fraction of sibling edges within ε of another in `φ` | crisp collapse scalar; **note: planner never dedups → duplicates burn budget** | P0 | tree, φ |
| D5 | `pose_vs_action_ratio` | D2 ÷ D1 | **localizes the break:** low-action=sampler; high-action/low-pose=model insensitive/OOD | P1 | tree, φ |
| D6 | `dpp_logdet` | `log det(K+εI)`, Gaussian kernel on `φ` | principled volume-spanned diversity (beyond BCI) | P2 | tree, φ |

All D-metrics computed **per expansion** and aggregated (mean over the tree) **and**
reported separately for the **root expansion** (where the decision lives).

### E. World-model / rollout fidelity (L2) — separate "search failed" from "model lied"

| ID | Metric | How | Why | Prio | Needs |
|---|---|---|---|---|---|
| E1 | `grounding_curve`, `h_star` | `‖ẑ_{t+h} − enc(o_{t+h})‖` teacher-forcing true actions on held-out real trajs; `h*` = tolerance-crossing horizon | **model ceiling** — planning beyond `h*` is fiction; caps useful `horizon`/`sim_horizon` | P1 | ref |
| E2 | `ood_rate` | fraction of rollout latents flagged OOD (Mahalanobis to a real-latent bank) | explains *diverse-but-useless* edges under high `ctx_noise` | P1 | (latent bank) |
| E3 | `reward_degenerate_rate` | fraction of high-value nodes where reward hit floor / tiny support (`TCenter`: no-T / area≈`min_area_frac`) | reward hacking / planner delusion; generalizes as "reward on degenerate input" | P1 | tree |
| E4 | `self_consistency` | `dist(long rollout ‖ composed short rollouts)` to same horizon | edges *are* composed short rollouts — does composition match the model's own long rollout? no ground truth needed | P2 | model |

E1/E4 are **model-level calibrations** (run once per checkpoint, not per tree) →
feed `h*` back as a recommended horizon ceiling. E2/E3 are **per-tree**.

### F. Outcome — the dependent variable(s)

> **Open dependency:** no closed-loop simulator was found in the notebook (initial
> contexts come from the real dataset). Until an `Executor` is available, outcomes
> are measured **in-model** via honest *relative* baselines. See §3 `Executor`.

| ID | Metric | How | Why | Prio | Needs |
|---|---|---|---|---|---|
| F1 | `peak_reward`, `delta_over_root` | best node reward on plan path; minus root reward | did the tree find anything better than the start | P0 | tree |
| F2 | `g_greedy` (gain vs shooting) | `tree_peak − peak(best of N depth-deep re-conditioned rollouts)` | **honest in-model test:** did *search* beat best-of-N random shooting | P0 | base |
| F3 | `g_random` (gain vs 1 rollout) | `tree_peak − (one depth-deep re-conditioned rollout, no search)` | did planning beat a single undirected rollout at matched lookahead | P0 | base |
| F4 | `seed_success_rate`, `seed_var` | fraction of inits with `peak_reward ≥ τ`; variance across seeds | reliability, not luck (aggregated per **condition**, not per tree) | P0 | (analysis) |
| F5 | `predicted_vs_realized_gap` | tree `peak_reward` − realized return of executing the plan | joint model+search error; **the real success signal** | P2 | exec |
| F6 | `budget_efficiency` | `g_greedy`/`q_margin` vs `n_iterations` | where does it saturate → min budget per condition | P2 | (sweep over budget) |

**Minimal high-value core** (if time-boxed, implement these first): A1–A4, B1–B4,
C1–C2, D1–D4, F1–F3 + F4. These span all five links and directly test the
hypothesis and the L4 catch.

---

## 3. Code architecture

New self-contained package: **`dreamerv4uwm/planning/experiments/study/`**. Keeps
the sweep isolated from the library; reuses `planning/{mcts,reward,rollout}.py`.

```
planning/experiments/study/
  __init__.py
  model.py          # load_world_model(cfg) -> (denoiser, tokenizer, rollout, dims); reuses models.utils, bf16
  data.py           # sample_initial_contexts(dataset_cfg, n, seed) -> list[(ctx_z, ctx_a)]  (from real shards)
  descriptors.py    # StateDescriptor protocol; TPoseDescriptor (reuses reward debug); RewardScalarDescriptor; EncoderEmbedDescriptor (fallback)
  baselines.py      # _rollout_random_peak(...), _rollout_greedy_peak(...)  -> g_random, g_greedy
  fidelity.py       # grounding_curve(), self_consistency(), fit_latent_bank(), ood_rate()  [model-level + per-tree]
  metrics.py        # compute_tree_metrics(tree, trace, phi, cfg) -> flat dict  (families A–E)
  run_tree.py       # build ONE tree for (plan_cfg, seed, init) -> metrics row (dict). Pure, deterministic given seed.
  run_sweep.py      # Hydra @main entrypoint: expand job list, load model once, loop rows, append CSV shard
  config/           # hydra configs (§4)
  analysis/
    aggregate.py    # concat shard CSVs -> trees.parquet
    correlate.py    # rank-corr / mutual-info / partial-corr / GBM feature importance of metrics vs outcome
    plots.py        # context-noise response curves, diversity×fidelity 2x2, per-link dashboards
```

**Key interfaces (env/reward-agnostic):**
- `RewardFn` — already exists (`reward.py` contract `z → r`). Sweep picks it by config.
- `StateDescriptor.__call__(node) -> np.ndarray` — the `φ` for diversity. `TPoseDescriptor`
  pulls the cached centroid/area; generic fallback decodes + embeds. **Prereq:** small
  reward change so `TCenterReward` can *return* its debug (centroid/area) instead of
  discarding it (see `.tex` §"what to add"), cached on the `Node` during `_expand`.
- `Baselines` — random + policy-prior peak reward at matched budget (in-model).
- `Executor` (optional) — `execute(plan) -> realized_return`; `None` today → F5 skipped.

**Planner instrumentation needed (small, backward-compatible):**
1. `trace=True` already exists — reuse its event log for B/entropy-trajectory/switches.
2. **P1 add:** optional `record_m2` (per-node `Σ R²`) for C4.
3. **P1 add:** optional `value_norm ∈ {none, mean, minmax_siblings}` hook in `_simulate`/`_ucb1`
   to test the L4 catch as a *treatment* (see §5). Guarded by config; default `none` = current behavior.
4. Cache `φ` inputs (centroid/area or terminal latent) on nodes at expansion.

---

## 4. Configuration (Hydra)

```
study/config/
  sweep.yaml           # top: defaults [env, reward, model, base_plan]; output_dir; sweep spec; execution
  base_plan.yaml       # a PlanConfig baseline (the fixed point OFAT perturbs)
  env/pushT.yaml       # dataset shards, init sampling, dims
  reward/tcenter.yaml  # TCenterReward kwargs (center_xy, sigma, min_area_frac, ...)
  model/pushT-large.yaml   # hydra compose target + ckpt paths + horizon_aware=false + bf16
```

- `model/pushT-large.yaml` reproduces the notebook load:
  `compose('dynamics/pushT-large', overrides=['denoiser.horizon_aware=false'])`,
  `dynamics_ckpt`, `tokenizer_ckpt`, `max_num_forward_steps=300`, device/bf16. Anchor
  the hydra `config_dir` explicitly (known resolution gotcha).
- The **sweep spec** lists factors + values (§5); `run_sweep.py` expands it to a job list
  (`config_id`, `seed`) and the SLURM array slices it by task id.

---

## 5. Sweep design

Full Cartesian is intractable (~10⁵ trees). Use a **two-tier** design:

**Tier 1 — OFAT (interpretable main effects).** Fix a `base_plan`, vary one factor at
a time across its grid. ~30 configs × 30 inits ≈ **900 trees**.

**Tier 2 — Latin-hypercube (interactions + predictive model).** Random-sample the joint
factor space. ~500 configs × 20 inits ≈ **10k trees**.

**Initial contexts.** A fixed **curated set** of 40 decision points
(`config/inits/pushT_curated.yaml`, built by `curate.py` via farthest-point selection over real
frames' T-pose × start-reward) — paired across configs, spanning the situation space
(reward 0.19→0.99, all quadrants, orientations). `inits.mode: random` remains available.

**Factors and grids (first pass — sign-off needed):**

| Factor | Grid | Tests link |
|---|---|---|
| `ctx_noise` | 0.0, 0.3, 0.5, 0.7, 0.9 | L1/L2 (diversity vs OOD — expect inverted-U) |
| `horizon` (edge) | 6, 12, 20, 28 | L2 (diversity vs `h*` fidelity ceiling) |
| `sim_horizon` | 8, 16, 28 | L3 (value scale & separation) |
| `branching` `B` | 3, 5, 8 | L1/L2 (fan-out width) |
| `action_temp` | 0.5, 1.0, 1.5 | L1 (sampler dispersion) |
| `c_ucb` | 0.5, 2.0, 8.0 | **L4 (the catch)** |
| **`value_norm`** | none, mean, minmax_siblings | **L4 (does normalizing fix collapse?)** |
| `max_depth` | 3, 5 | depth accumulation |
| `n_iterations` | 24, 48 | L5 / budget efficiency (F6) |

Seeds control **both** the initial state (sampled from real data) and the planner RNG;
log both `init_id` and `plan_seed` so seed variance (F4) is estimable.

---

## 6. Output schema

- **Primary:** `trees.csv` (one row per tree). Columns = `run_id`, all config factors,
  `init_id`, `plan_seed`, then every scalar metric (A–F). Flat → direct pandas/analysis.
  Written as per-task shards `shard_<task>.csv`, concatenated to `trees.parquet` in analysis.
- **Secondary (P1, opt-in):** raw MCTS `trace_<run_id>.json` per tree for deep dives / metric
  recompute without re-running the model.
- **Tertiary (P2):** `curves_<task>.parquet` — per-iteration series (entropy, root Q, switches)
  if we want them beyond the scalar summaries (AUC/var already in `trees.csv`).
- Idempotent: each row keyed by `(config_id, init_id, plan_seed)`; re-runs skip existing keys.

---

## 7. Cluster execution

**Compute model.** One tree at a time, **bf16** (fp32 OOMs). Model+tokenizer loaded
**once per task** and reused across that task's chunk (loading dominates otherwise).
Embarrassingly parallel — **no inter-GPU communication**.

**SLURM array (NYU HPC, primary).** `hpc/slurms/mcts_sweep.slurm`:
- `#SBATCH --array=0-N%K` — task `i` processes job-list slice `[i·chunk : (i+1)·chunk]`, writes its own shard.
- **Per task:** `--gres=gpu:1`, `--cpus-per-task=8–12` (opencv T-segmentation is **CPU**-bound —
  parallelize `score_t_centered` across frames), `--mem=48–64G`, `--time` from smoke-test timing.
- **GPU:** A100 80GB or H100 80GB (safe for `pushT-large` denoiser+tokenizer in bf16; 40GB likely
  ok but confirm). Runs inside the Singularity + conda overlay (see `run-python-in-container` memory).
- Launch: `sbatch mcts_sweep.slurm`; ship code via `hpc-transfer`; smoke-test via `hpc-smoke-test`.

**Alt: Brev 8×H100 (interactive / smaller sweeps).** No SLURM — a `launch_brev.sh` that pins one
worker per GPU (`CUDA_VISIBLE_DEVICES=0..7`) over 8 job-list slices via `xargs -P8`. Use the
`brev-gpu` skill. Good for Tier-1 / debugging before the full Tier-2 array.

**Throughput estimate (calibrate first).** ~15–60 s/tree (grows with `horizon·n_iterations`).
~11k trees ⇒ ~90–180 GPU-h ⇒ 16-way array ⇒ ~6–12 h wall. **First action on cluster:** a
20-tree timing smoke test to set `chunk`, `--time`, and confirm VRAM.

---

## 8. Analysis (`study/analysis/`)

1. `aggregate.py` — concat shards → `trees.parquet`; sanity checks (NaNs, degenerate trees, missing keys).
2. `correlate.py`:
   - Rank correlation (Spearman) of each **process** metric vs each **outcome** (F2/F3/F4).
   - **Mutual information** — to catch non-monotone effects (`ctx_noise` inverted-U).
   - **Partial correlation** / GBM + permutation importance — which factors matter *after*
     controlling for the diversity they buy (does `horizon` help beyond diversity?).
   - Report **per regime** (e.g. split by `c_ucb`, by `value_norm`) — a predictor in one regime
     may be irrelevant in another.
3. `plots.py`:
   - **Context-noise response curves** — diversity (D2/D3) and `ood_rate` (E2) vs `ctx_noise`;
     locate the knee (useful diversity before incoherence).
   - **Diversity × fidelity 2×2** — low-div+low-error ⇒ fix sampler/`c_ucb`/`value_norm`;
     high-div+high-OOD ⇒ cap `ctx_noise` at `h*`. The core decision rule.
   - **Per-link dashboard** — one panel per chain link (§1), each showing its metrics vs outcome,
     so a failure is *localized* to L1…L5.
   - Budget-efficiency curves (F6).

**Headline deliverable:** a ranked list of cheap tree-level early-warning signals for
"this tree will plan to nothing," + the conditions that avoid it, + which chain link each
condition fixes.

---

## 9. Build order (milestones)

- **M0 — plumbing.** ☑ `model.py` (`load_world_model`, `make_decode_fn`, `model_dims`),
  `data.py` (`sample_initial_contexts` from real shards). Reproduces the notebook load.
- **M1 — metrics core.** ☑ `descriptors.py` (`TPoseDescriptor` + fallbacks) + `metrics.py`
  (families A–E + tree-internal F, ~47 metrics) + `baselines.py` (F1–F3). Unit-checked on a
  hand-built tree (`scratchpad/test_metrics.py`, PASS).
- **M2 — single-tree runner.** ☑ `run_tree.py` → validated 74-col row; deterministic per seed.
  **End-to-end smoke test PASSES** (real model, ~6 s for a tiny tree; first tree already shows a
  clean, self-consistent *collapse* signature across all families).
- **M3 — sweep + config.** ☑ `run_sweep.py` + `config/{sweep,smoke}.yaml`. Verified locally: 6-tree
  smoke, disjoint 2-task array split, resume (0 rows on rerun), OFAT factor varies, metrics sane.
  The smoke already reproduces the hypothesis (ctx_noise↑ → bci↓, val_std↑, ratio↓, g_random→+).
- **M4 — cluster.** ◐ `hpc/slurms/mcts_sweep.slurm` (parameterized template) + `study/README.md`
  written. **Blocked on:** filling the container/overlay paths (open #3) and a cluster timing run
  to set `--time`. Local full-config timing was impossible (shared 98GB GPU saturated by other jobs).
- **M5 — analysis.** ☑ `analysis/{schema,aggregate,correlate,plots,report}.py`. Spearman + binned
  **mutual info** (non-monotone) + **partial** corr (vs knobs) + GBR/OLS **feature importance**,
  per-regime; response curves, correlation heatmap, mechanism scatter, per-link importance bars;
  one-shot `report.py` → `REPORT.md`. Validated on synthetic data with a known chain (`test_analysis.py`, PASS).
- **M6 — P1/P2 extras.** ☐ fidelity (E1/E2/E3), `value_norm` treatment (test the L4 catch),
  return-variance C4, executor hook (F5). Optimisation: cache the reward's T-centroid on nodes so
  the descriptor avoids a second decode.

> **Empirical note (M2 smoke, first tree, short-horizon config).** The metrics fired
> consistently: diversity collapsed (`bci=0.67`, `duplicate_rate=1.0`, `outcome_div≈0.006`),
> values flat (`val_std=0.026`), selection noise-dominated (`exploit_explore_ratio=17.3`,
> `visit_entropy=0.985`), planning gain ≈0 (`g_greedy≈g_random≈0`). The instrumentation
> distinguishes the two L4 regimes via the ratio: ≫1 = flat-value/noise (seen here), ≪1 = greedy.

---

## 10. Open questions

**Resolved (2026-07-10):**
1. ☑ **Outcome truth → in-model baselines.** No executor. Outcome = `g_greedy` + `g_random`
   (in-model, matched lookahead) + seed variance (F1–F4). **F5 deferred** behind the `Executor` hook.
2. ☑ **Cluster → NYU HPC (SLURM).** Write the array job; run in the Singularity+conda overlay.

**Still open:**
3. **Checkpoints/data on the cluster.** Confirm `dynamics_ckpt`, `tokenizer_ckpt`, and the PushT
   dataset shards are present on HPC scratch; else `hpc-transfer` them. (Verify at M4.)
4. **Factor grids (§5).** Current values are a first pass — revisit after Tier-1 main effects.
5. **Held-out real trajectories (E1).** Confirm we can pull `(obs, action)` sequences from the
   real dataset for the grounding calibration. (P1, needed at M6.)

**Current step: M0–M1 (build the metrics core).**
