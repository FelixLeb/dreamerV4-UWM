"""Build the dataset and save a standard set of charts + a correlation table.

    python -m ...analysis.report --input-dir <run>/results --out-dir <run>/analysis

Everything here is a few lines over ``dataset`` and ``charts``; edit freely.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless: this is a script

from . import dataset, charts

KNOBS = ["horizon", "max_depth", "ctx_noise", "sim_horizon", "branching", "c_ucb"]


def run(shards_dir, out_dir, outcome="g_random"):
    od = Path(out_dir); figs = od / "figures"; figs.mkdir(parents=True, exist_ok=True)

    df = dataset.build(shards_dir, out=od / "trees.parquet")
    (od / "overview.txt").write_text(dataset.overview(df, outcome) + "\n")
    print(dataset.overview(df, outcome))

    charts.correlations(df, outcome).to_csv(od / "correlations.csv", header=["spearman"])
    charts.corr_bars(df, outcome, save=figs / "predictors.png")
    charts.hist(df, outcome, save=figs / "outcome.png")
    charts.scatter(df, "edge_val_std", outcome, color="ctx_noise", save=figs / "mechanism.png")
    for k in KNOBS:
        if k in df.columns and df[k].nunique() > 1:
            charts.response(charts.ofat(df, k), k, outcome, save=figs / f"knob_{k}.png")

    print(f"wrote trees.parquet, correlations.csv, overview.txt and figures/ to {od}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="dir with shard_*.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--outcome", default="g_random")
    a = ap.parse_args(argv)
    run(a.input_dir, a.out_dir, a.outcome)


if __name__ == "__main__":
    main()
