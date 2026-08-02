# `analysis/` — the MCTS-sweep analysis code

This directory turns a finished MCTS sweep into a **dataframe**, a set of **charts**, and the
**deck figures**. One row of that dataframe = one MCTS tree (a config × a curated start state).

**Two docs, two jobs:**

| Doc | Answers |
|---|---|
| **`README.md`** (this file) | *what the code is and how to run it* — modules, data flow, commands, recipes |
| [`METRICS_GUIDE.md`](METRICS_GUIDE.md) | *what the numbers mean* — every column, the statistics vocabulary, how to read each figure, pitfalls |

If you want to know what `edge_val_std` or `partial` *means*, go to the guide. If you want to build a
parquet or add a chart, stay here.

---

## The data flow

```
   <run>/results/                dataset.build          charts.*  /  plots.*
   shard_0000.csv   ─┐          ┌───────────┐          ┌──────────────────────┐
   shard_0001.csv    ├─────────►│  one row  │─── df ──►│ response / scatter /  │──► PNG / notebook
        ...          │  concat  │  per tree │          │ hist / corr_bars ...  │
   shard_000N.csv   ─┘  + dedup └───────────┘          └──────────────────────┘
                              │  + success_<outcome>          plots.make_deck ──► deck PDFs
                              ▼
                       trees.parquet   ◄── everything downstream reads this
```

The sweep runner (`../run_sweep.py`) writes one **`shard_*.csv` per SLURM task**. `dataset.build`
concatenates them, drops duplicate `(config_id, init_id)` rows, adds the derived `success_<outcome>`
labels, and saves a single **`trees.parquet`**. All exploration and every figure read that parquet —
the raw shards are only needed once, to build it.

---

## Quickstart

All commands run from the **repo root**. The module prefix is
`dreamerv4uwm.planning.experiments.study.analysis` (abbreviated `…analysis` below).

```bash
# 1. build the parquet from a run's shards
python -m …analysis.dataset --input-dir <run>/results --out <run>/analysis/trees.parquet

# 2. one-shot standard report (parquet + correlations.csv + a folder of charts)
python -m …analysis.report --input-dir <run>/results --out-dir <run>/analysis

# 3. regenerate the five LaTeX-deck figures (default out-dir: study/presentation/figures/)
python -m …analysis.plots --parquet <run>/analysis/trees.parquet --deck
```

Interactive exploration is the notebook — open **[`explore_results.ipynb`](explore_results.ipynb)**,
point its first cell at a `trees.parquet`, and run down. It only uses `dataset` + `charts` + `schema`.

---

## The modules

Six small, single-purpose files. The everyday two are `dataset` and `charts`.

| File | Role | Key surface |
|---|---|---|
| **`dataset.py`** | build/load the parquet | `build(shards_dir, out=)`, `load(parquet)`, `overview(df, outcome)` |
| **`charts.py`** | simple, generic charts | `response`, `scatter`, `hist`, `corr_bars`/`correlations`, `heatmap`; `ofat`, `metric_cols` |
| **`schema.py`** | column taxonomy (single source of truth) | `data_dictionary(df)`, `add_success(df)`, `process_cols`, `link_of`, `OUTCOMES`/`FACTORS`/`LINKS`/`DESCRIPTIONS` |
| **`report.py`** | one-command standard report | `run(shards_dir, out_dir, outcome)` |
| **`plots.py`** | the 5 polished deck figures | `fig_*`, `make_deck(df, out_dir)` |
| **`correlate.py`** | optional advanced statistics | `metric_outcome_table`, `partial_spearman`, `feature_importance`, `analyze` |

### `dataset.py` — the parquet
`build` is the only thing that touches raw shards. It adds one binary label per outcome via
`schema.add_success`: `success_<gain>` = `gain > 0` for each gain outcome, and
`success_tree_peak` = `tree_peak ≥ 0.7`. `overview(df, outcome)` prints sizes + that outcome's
success rate + any mostly-NaN columns.

### `charts.py` — everyday charts
Every chart is `fn(df, cols…, ax=None, save=None)` and returns a matplotlib `Axes`:

```python
from …analysis import dataset, charts
df = dataset.load("trees.parquet")

charts.response(charts.ofat(df, "ctx_noise"), "ctx_noise", "g_random_peak")  # a knob sweep (mean ± SEM)
charts.scatter(df, "edge_val_std", "g_random_peak", color="ctx_noise")       # a metric vs the outcome
charts.hist(df, "root_bci")                                            # a distribution
charts.corr_bars(df, "g_random_peak")                                       # what predicts the outcome
```

Two conveniences worth knowing:
- **`charts.ofat(df, factor)`** returns just that factor's one-factor-at-a-time slice (`base` +
  `ofat.<factor>=…` configs), so a knob curve isn't diluted by rows sitting at the baseline value.
- **`charts.metric_cols(df)`** is the list of *candidate-predictor* columns — numeric, non-constant,
  with ids / outcomes / `success_*` / cost columns excluded. Pass it anywhere you want "all the real
  metrics."

`ax=` lets you tile charts into a subplot grid; `save=` also writes a PNG. That's the whole API — to
make a **new** chart, copy any function; it's a few lines of matplotlib on `df[col]`.

### `schema.py` — the single source of truth
Which columns are knobs / outcomes / metrics, the L1–L5 causal-link mapping, and a one-line
description of every column live here (and nowhere else). Adding a metric = adding it to `LINKS` and
`DESCRIPTIONS`; everything downstream (exclusions, the data dictionary, the deck's link colours)
follows automatically. `schema.data_dictionary(df)` renders the taxonomy for the columns actually
present and **flags any undocumented column**, so the code and the docs can't silently drift.

### `report.py` — the standard report
`report.run` is a few lines over `dataset` + `charts`: build the parquet, write `overview.txt` and
`correlations.csv`, and a `figures/` folder (predictors, outcome histogram, mechanism scatter, one
response curve per knob). It's meant to be **read and edited** — it's the template for "the charts I
always want."

### `plots.py` — the deck figures
The five polished figures for the Beamer deck (Okabe-Ito palette, annotations). `make_deck(df, out_dir)`
writes all five as PDF + PNG plus `key_numbers.txt`; the `--deck` CLI defaults its out-dir to
`study/presentation/figures/` (created on demand). Use `charts.py`, not this, for everyday exploration.

### `correlate.py` — advanced statistics (optional)
Rank correlation + **mutual information** (catches non-monotone effects), **partial** rank correlation
controlling for the swept knobs (is a metric a real online signal or just a knob proxy?), and
gradient-boosted **feature importance**. `partial_spearman` is reused by `plots.fig_predictors`.

```bash
python -m …analysis.correlate --input-dir <run>/results --out-dir <run>/analysis/stats
```

---

## Recipes

**Build a parquet, then explore in the notebook**
```bash
python -m …analysis.dataset --input-dir <run>/results --out <run>/analysis/trees.parquet
# open explore_results.ipynb, set df = dataset.load("<run>/analysis/trees.parquet")
```

**Make your own chart** — copy a template from `charts.py`, e.g. a knob curve for a different outcome:
```python
charts.response(charts.ofat(df, "sim_horizon"), "sim_horizon", "g_greedy_peak", save="my.png")
```

**A grid of every knob at once**
```python
import matplotlib.pyplot as plt
knobs = ["horizon", "max_depth", "ctx_noise", "sim_horizon", "branching", "c_ucb"]
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, k in zip(axes.ravel(), knobs):
    charts.response(charts.ofat(df, k), k, "g_random_peak", ax=ax)
fig.tight_layout()
```

**Add a new metric** — emit the column in `../metrics.py`, then list it in `schema.LINKS` (under its
causal link) and `schema.DESCRIPTIONS`. It now flows into `metric_cols`, the data dictionary, and the
deck colours with no other changes.

**Regenerate the deck after a new run**
```bash
python -m …analysis.plots --parquet <run>/analysis/trees.parquet --deck
```

---

## What the scripts write

| Command | Writes |
|---|---|
| `dataset` | `trees.parquet` |
| `report` | `trees.parquet`, `overview.txt`, `correlations.csv`, `figures/{predictors,outcome,mechanism,knob_<k>}.png` |
| `plots --deck` | `fig_{knob_effects,knob_curves,ctxnoise,predictors,mechanism}.{pdf,png}`, `key_numbers.txt` |
| `correlate --out-dir` | `metric_outcome.csv`, `factor_outcome.csv`, `importance_g_random_peak.csv`, `importance_g_greedy_peak.csv` |

See **`METRICS_GUIDE.md` §7–§8** for how to read each of these outputs.

---

## Also in this directory

- **`mcts_sweep/`** — the sweep runs (each `<run>/results/` holds the shards, `<run>/analysis/` the
  built parquet + figures).
- **`metrics_reference.tex`**, **`diagnostic_studies.tex`** — the LaTeX write-ups: the full metric
  definitions and the catalogue of studies/charts to build from them.
- The sweep **runbook** one level up — [`../README.md`](../README.md) — covers curating start states and
  running the sweep (locally / on NYU HPC) that produces the shards this code consumes; the sweep
  **design** is [`../../mcts_study_plan.md`](../../mcts_study_plan.md).
