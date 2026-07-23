# Reading the MCTS sweep results — metrics & figures guide

A reference for every column in the sweep data, every statistic the analysis prints, and every figure.
Definitions match the code exactly (`study/metrics.py`, `run_tree.py`, `baselines.py`,
`analysis/`). For **how to run** the code that produces these (build the parquet, make charts,
regenerate the deck), see the companion [`README.md`](README.md). Further companions in this directory:
[`metrics_reference.tex`](metrics_reference.tex) (full metric definitions) and
[`diagnostic_studies.tex`](diagnostic_studies.tex) (the chart/study catalogue); the sweep design is
[`../../mcts_study_plan.md`](../../mcts_study_plan.md).

---

## 1. Three kinds of columns

Every row of `trees_all.parquet` / `shard_*.csv` is **one MCTS tree**. Its columns are of three kinds:

- **Factors (knobs)** — the config that produced the tree: `horizon`, `ctx_noise`, `c_ucb`, … These
  are the *independent* variables you swept.
- **Process metrics** — properties measured *on the finished tree*: diversity, value spread, how
  visits were allocated, tree shape. These are the *candidate predictors* — cheap signals you could
  read online to tell if a tree is planning well.
- **Outcomes** — did planning actually help (`g_1shot`, `g_shootN`, …). The *dependent* variables.

The analysis asks: **which process metrics (and which knobs) predict good outcomes?**

> **Programmatic version:** `schema.data_dictionary(df)` returns this same reference as a
> table (`column, kind, link, dtype, description`) for the columns actually present in a run,
> built from `schema.DESCRIPTIONS`. It's rendered in `explore_results.ipynb` §1, and flags any
> column that lacks a description — so the code and this document can't silently drift apart.

Process metrics are organized by the **causal chain** (a break anywhere ⇒ "planning leads to nothing"):

| Link | Question | Metric family |
|---|---|---|
| L1 sampler | do the sampled *actions* differ? | action diversity |
| L2 model | do actions reach *different task outcomes*? | outcome/edge diversity |
| L3 estimator | do those outcomes get *distinguishable values*? | value spread |
| L4 selection | does UCT *act* on the value differences? | allocation / entropy |
| L5 decision | does it yield a confident root choice? | structure / margin |

---

## 2. The reward (what everything is measured against)

`TCenterReward` decodes a latent to an image, segments the red **T**, and returns
`R = exp(-½ (d/σ)²) ∈ [0,1]`, where `d` = distance of the T's centroid from the image centre, `σ=0.25`.
**R=1** ⇒ T perfectly centred (the goal); **R≈0** ⇒ T far from centre or not found. Every reward
number below is in this [0,1] scale.

---

## 3. Outcome metrics (the dependent variables)

Because absolute reward isn't comparable across start states, the real outcomes are *gains over
baselines* at equal budget. All computed in `run_tree.py` / `baselines.py`.

There are **two baseline families**. The *flat* baselines roll out a single `sim_horizon`-frame
trajectory; the ***fair*** baselines (`*_fair`) are built the **same way the plan is** —
`max_depth` edges of `horizon` frames, re-conditioning the context after each edge — so they match
the plan's lookahead *and* its construction. Prefer the fair ones (see the note below).

| Column | Definition | Read as |
|---|---|---|
| `tree_peak` | best single-frame reward over the frames of the **returned plan** | how good a state the plan reaches |
| `root_reward` | reward of the **start** state (last context frame) | the starting point |
| `oneshot_peak` / `shootN_peak` | best reward over one / `n_random` **flat** `sim_horizon` rollouts | flat no-search / random-shooting |
| `oneshot_fair_peak` / `shootN_fair_peak` | best reward over one / `n_random` **depth-deep re-conditioned** rollouts | fair no-search / random-shooting |
| **`g_1shot`** | `tree_peak − oneshot_peak` | planning gain vs a **flat** rollout |
| **`g_shootN`** | `tree_peak − shootN_peak` | search gain vs **flat** random shooting |
| **`g_1shot_fair`** | `tree_peak − oneshot_fair_peak` | **planning gain at matched lookahead — the fair headline** |
| **`g_shootN_fair`** | `tree_peak − shootN_fair_peak` | search gain vs **fair** random shooting |
| `delta_over_root` | `tree_peak − root_reward` | did the plan improve on the start at all? |
| `success_<gain>` (e.g. `success_g_1shot`) | `1 if <gain> > 0` | one binary label per gain outcome (derived by `dataset.build`) |
| `success_tree_peak` | `1 if tree_peak ≥ 0.7` | reached a well-centred T (derived) |

**`g_1shot_fair` is the main target.** `g ≤ 0` is the precise statement of "planning did nothing a
no-search rollout of the same lookahead couldn't do." The `_fair` variants are the honest ones: the
flat `g_1shot`/`g_shootN` give the tree a **free lookahead advantage** (the plan reaches `horizon×max_depth`
frames but the flat baseline only `sim_horizon`), and their baseline length **tracks `sim_horizon`** —
which confounds the `sim_horizon` sweep. `g_shootN(_fair) ≤ 0` is the stronger signal: the search
didn't even beat random shooting — the classic sign of *incoherent* edges (diverse but useless).

> The `*_fair` columns only exist for runs made after the fair-baseline change; `run1` predates it
> (re-run its baselines to get them). `schema.OUTCOMES` lists both, and the analysis skips whichever
> are absent.

---

## 4. Statistics vocabulary (how to read the correlation numbers)

These appear in `correlations.csv` (from `report.py`) and the optional `correlate.py` tables
(`metric_outcome.csv`, `factor_outcome.csv`, `importance_*.csv`).

| Term (aliases) | What it is | How to read |
|---|---|---|
| `spearman` (`rho`, ρ) | **rank** correlation of a metric with an outcome across all trees, ∈[−1,+1] | +1 metric↑⇒outcome↑ (monotone); −1 opposite; 0 no monotone link. Robust to outliers. |
| `p` | p-value: chance of seeing this correlation if the true one were 0 | small = reliable. **With n≈735 p is tiny for almost anything — judge strength by \|ρ\|, not p.** |
| `abs_spearman` | \|ρ\| | used only to rank metrics by strength regardless of sign |
| `mutual_info` (`mi`) | mutual information metric↔outcome (captures **any** dependency, incl. non-monotone) | **MI ≫ \|ρ\|** ⇒ a nonlinear / inverted-U link the correlation *understates* (e.g. `ctx_noise`). MI≈0 ⇒ little dependency of any kind. |
| `partial` (`partial_vs_factors`) | partial Spearman controlling for the swept **knobs** | **does the metric predict the outcome *beyond* the config?** partial≈raw ⇒ a real online signal; partial→0 ⇒ it was just a proxy for a knob. |
| `gbr_importance` | permutation importance from a gradient-boosted model using **all** metrics jointly | accuracy lost when this metric is shuffled — captures nonlinear + interaction value |
| `ols_beta_std` | standardized linear-regression coefficient | the metric's *linear* pull on the outcome, holding others fixed (sign + size in std units) |

Rough effect-size bands for \|ρ\|: **<0.1** negligible · **0.1–0.3** weak · **0.3–0.5** moderate · **>0.5** strong.

---

## 5. Process metrics, by family

Notation: root children (the "arms") `i` with visit counts `Nᵢ`, mean values `Qᵢ`, visit shares
`pᵢ = Nᵢ/ΣN`. "Sibling set" = the `B` children of one expansion. Diversity metrics are computed per
expansion and reported two ways: **`div_*_mean`** (averaged over the whole tree) and **`root_*`** (the
root expansion only, where the decision lives).

### A. Structural / tree shape
| Column | Definition | Healthy ↔ collapsed |
|---|---|---|
| `n_nodes` | total nodes in the tree | — (size/compute) |
| `n_forward` | batched world-model calls (cost proxy) | — |
| `n_expanded` | nodes that were given children | — |
| `expansion_efficiency` | `(n_nodes−1)/n_forward` | high = compute bought new structure |
| `max_depth` | deepest node depth reached | deeper = looked further ahead |
| `mean_visited_depth` | visit-weighted mean depth `ΣNᵥ·depthᵥ / ΣNᵥ` | grows with search ↔ **stuck near 1** (never commits deep) |
| `eff_branching_realized` | mean #children with ≥1 visit, per expanded node | vs nominal `branching`; gap = wasted edges |
| `eff_branching_geom` | `n_nodes^(1/max_depth)` | geometric branching estimate |
| `root_width` | #root children (= `branching`) | — |
| `subtree_size_gini` | Gini of the root children's **subtree sizes** ∈[0,1] | **high = selective** (mass on good lines) ↔ 0 = uniform bush (no discrimination) |

### B. Allocation / exploration–exploitation (the L4 selection link)
| Column | Definition | Healthy ↔ pathological |
|---|---|---|
| `visit_entropy` | normalized entropy of `{pᵢ}` ∈[0,1] | concentrates (low) once decided ↔ **stuck high = indecision** |
| `commit_top1` | `max pᵢ` (visit share of the top arm) | high = committed |
| `visit_value_corr` | Spearman(`Nᵢ`, `Qᵢ`) | **+** (visits flow to value) ↔ ~0 = bandit not tracking value |
| `exploit_explore_ratio` | mean UCB explore bonus ÷ `std(Qᵢ)` | **moderate**; ≫1 = exploration swamps flat values (noise); ≪1 = greedy |
| `mean_explore_term` | mean of `c_ucb·√(ln N_root / Nᵢ)` | the raw exploration bonus magnitude |
| `reco_agreement` | `1` if most-visited arm == highest-value arm | 1 = trustworthy recommendation ↔ 0 = tie-break luck |
| `selection_depth_exploration` | deepest depth where most-visited ≠ highest-value child (−1 = never) | exploration still active deep vs only at root |
| `entropy_auc` | mean of the visit-entropy **trajectory** over iterations (from the trace) | high throughout = never concentrated |
| `best_action_switches` | #times the recommended arm changed over iterations | **low** = converged ↔ high = under-budgeted |
| `root_value_stability` | std of root value over the last ~5 iterations | low = converged ↔ high = still moving |

### C. Value / return quality (the L3 estimator link)
| Column | Definition | Healthy ↔ collapsed |
|---|---|---|
| `val_std` | std of root-child values `Qᵢ` | **>0** (arms distinguishable) ↔ **→0 = planning indifferent** |
| `val_spread` | `max Qᵢ − mean Qᵢ` | larger = a clearly better arm exists |
| `q_margin` | `(Q₍₁₎ − Q₍₂₎)/std` — normalized gap best vs 2nd-best | large = confident decision ↔ ~0 = coin-flip |
| `val_std_depth1/2/3` | `val_std` among siblings at depth 1 / 2 / 3 | discrimination *decay* with depth; `depth3` often NaN (few deep sets) |

> **Scale caveat:** values are the cumulative-sum backup `Σγᵗr`, so their magnitude grows with
> `sim_horizon`. `val_std` is meaningful *within* a fixed config; comparing it *across* `sim_horizon`
> mixes real discrimination with this scale inflation (see §10).

### D. Diversity (the L1 sampler & L2 model links) — your core hypothesis
Descriptor `φ(state) = (cx, cy, θ, √area)` of the decoded T (pose). `edge_val` = a child edge's terminal reward.
| Column (`div_*_mean` and `root_*`) | Definition | Healthy ↔ collapsed |
|---|---|---|
| `edge_val_std` | std of the terminal **reward** across siblings | **>0** (reward-diverse outcomes) ↔ →0 |
| `best_edge_val_tree` | max `edge_val` over all nodes | best reward found anywhere |
| `action_div` (`div_action_div_mean`, `root_action_div`) | mean pairwise L2 of sibling **action** sequences | high = sampler proposes varied actions (L1) |
| `action_dmin` | min pairwise action distance | near-duplicate action detector |
| `outcome_div` (`div_outcome_div_mean`, `root_outcome_div`) | mean pairwise distance of sibling **T-poses** φ | high = edges reach different task states (L2) |
| `outcome_dmin` | min pairwise pose distance | near-duplicate outcome detector |
| **`bci`** (`div_bci_mean`, `root_bci`) | **Branch-Collapse Index** `= 1 − N_eff/B`; `N_eff` = #distinct sibling outcomes | **0 = all B edges distinct** ↔ **→1 = collapse to one outcome** |
| `duplicate_rate` | fraction of siblings with a near-duplicate (within ε) in pose space | 0 = all distinct ↔ 1 = all duplicated |
| `found_frac` (`div_found_frac_mean`, `root_found_frac`) | fraction of sibling states with a valid detected T | <1 ⇒ some outcomes have no T (off-manifold). `root_found_frac` was constant 1.0 here → dropped |
| `pose_action_ratio` | `outcome_div / action_div` | **which stage collapsed:** low ratio = model maps varied actions to same state (or OOD); high with high action = healthy |

### F. Plan (tree-internal outcome)
| Column | Definition |
|---|---|
| `best_node_value` | highest backed-up **mean value** among nodes (the recommendation's value; cumsum scale) |
| `best_edge_val_on_plan` | max `edge_val` along the returned plan path |
| `plan_len` | #edges in the returned plan |

---

## 6. Config / bookkeeping columns
`config_id`, `config_tag` (e.g. `base`, `ofat.ctx_noise=0.7`), `window_idx`+`t0`+`init_id` (which curated
start state), `plan_seed` (planner RNG), `plan_secs` (wall time). Plus every `PlanConfig` knob:
`horizon, branching, edge_mode, sim_horizon, sim_rollouts, n_iterations, c_ucb, gamma, n_min, max_depth,
K_steps, ctx_noise, ctx_noise_honest, action_temp, max_ctx`.

---

## 7. How to read the report outputs

`report.py` writes, per run, into `<run>/analysis/`:

- **`overview.txt`** — sanity: `rows / configs / inits`, the outcome's `success rate` (e.g.
  `g_1shot success rate`), and any heavily-NaN columns (a metric undefined for many trees).
- **`correlations.csv`** — the **headline**: each process metric and knob's Spearman correlation with
  the chosen outcome, sorted. Read strength by \|ρ\| (see §4); this is the table behind
  `figures/predictors.png`.
- **`figures/`** — the charts described in §8.

The optional `correlate.py` (run separately) adds three richer tables:

- **`factor_outcome.csv`** — Spearman + MI of each **knob** vs the outcome. ⚠️ **On OFAT data this is
  diluted** — for any one factor most rows sit at the baseline value, so a real main effect can look
  weak. **Read the knob-response figures for true main effects.**
- **`metric_outcome.csv`** — the **process metrics** most predictive of the outcome, each with `rho`
  (monotone strength), `mi` (any-shape strength), `partial` (strength beyond the knobs), and its causal
  `link`. *Which cheap tree signals foretell whether planning worked.* Prefer metrics whose `partial`
  stays high (genuine online signals) and whose `link` you can act on.
- **`importance_g_1shot.csv` / `importance_g_shootN.csv`** — gradient-boosted permutation importance +
  standardized OLS beta, using all metrics jointly (captures nonlinear + interaction value).

---

## 8. How to read the figures

**Report figures** (`report.py`, in `<run>/analysis/figures/`):

- **`knob_<k>.png`** (e.g. `knob_ctx_noise.png`) — the outcome's **mean ± SEM vs one swept knob**, over
  that knob's OFAT slice. Read the *shape*:
  - *monotone* (e.g. `bci` ↓ as `ctx_noise` ↑) = a clean lever;
  - *inverted-U* (e.g. `g_1shot` peaks mid-range) = a sweet spot — where non-monotone effects that
    correlations miss become visible;
  - *cliff* (e.g. `g_shootN` crashing at high noise) = a regime boundary.
  The most information-dense view on OFAT data.
- **`predictors.png`** — horizontal bars: Spearman of each metric with the outcome, **red = −, blue = +**,
  strongest \|corr\| on top. Who predicts the outcome and in which direction.
- **`mechanism.png`** — one dot per tree: `x = edge_val_std`, `y = the outcome`, **colour = `ctx_noise`**.
  Where the successes live — reward-diverse edges (right) vs the collapse corner (left).
- **`outcome.png`** — the distribution of the outcome across trees: how often planning helps, and by
  how much.

**Deck figures** (`plots.make_deck`, polished, in `../presentation/figures/`):

- **`fig_knob_effects`** — effect size (max − min mean `g_1shot` across a knob's settings) per knob,
  each bar labelled with its curve shape (`monotone` / `inverted-U` / `~flat`). Which knobs move planning.
- **`fig_knob_curves`** — the 2×2 response curves (`horizon`, `max_depth`, `ctx_noise`, `sim_horizon`).
- **`fig_ctxnoise`** — the headline story: `root_bci` rises with noise (more diverse edges) while
  `g_1shot` / `g_shootN` **peak then cliff** (usefulness has a sweet spot).
- **`fig_predictors`** — top per-tree predictors of `g_1shot`, bars **coloured by causal link (L1–L5)**,
  with the **partial** correlation (controlling for all knobs) overlaid as a diamond. Which *links*
  dominate, and which signals survive controlling for the config.
- **`fig_mechanism`** — `edge_val_std` → mean `g_1shot` (left) and → planning-**success rate** (right):
  reward-diverse edges give better *and* more reliable planning.

---

## 9. Cheat sheet — healthy vs collapsed tree

| Signal | Healthy planning | "Plans to nothing" |
|---|---|---|
| `edge_val_std`, `val_std`, `val_spread` | > 0 | ≈ 0 |
| `q_margin` | clearly > 0 | ≈ 0 |
| `bci` / `duplicate_rate` | low / →0 | high / →1 |
| `visit_entropy`, `entropy_auc` | concentrates (lower) | stuck high |
| `commit_top1`, `subtree_size_gini` | high | low (uniform) |
| `exploit_explore_ratio` | moderate | ≫1 (noise) or ≪1 (greedy) |
| `mean_visited_depth` | grows with budget | pinned ~1 |
| `g_1shot`, `g_shootN` | > 0 | ≤ 0 |

---

## 10. Pitfalls when interpreting

1. **OFAT dilutes pooled factor correlations.** A knob varied in only a few configs sits at baseline in
   most rows → weak pooled ρ even for a real effect. Use response curves / per-factor slices.
2. **Non-monotone effects hide from Spearman.** `ctx_noise` has an inverted-U on `g_1shot`; its ρ≈0 but
   the *curve* and MI reveal it. Always cross-check MI and the response figure.
3. **Cumsum value scale.** `val_std`/`Q` grow with `sim_horizon`; they mean "discrimination" only at
   fixed scale. Across `sim_horizon`, higher `val_std` can coincide with *worse* outcomes.
4. **Partial mechanical coupling.** `edge_val_std`/`val_std` are somewhat tied to `tree_peak` by
   construction, so they partly *define* the outcome. The **selection** metrics (`visit_entropy`,
   `commit_top1`, `exploit_explore_ratio`) are the cleaner scale-independent "is the search working"
   signals — trust those when they agree.
5. **Large-n p-values.** With hundreds of trees, `p` is tiny for trivial effects. Rank by \|ρ\| / MI /
   importance, not by `p`.
6. **`partial` was NaN in the first run** (a bug: constant OFAT factors broke the control matrix); fixed
   in `correlate.py` — recomputed values are ~equal to the raw ρ for the top metrics (they *are* real
   online signals).