"""One-shot report: aggregate shards -> sanity -> correlation tables -> figures ->
REPORT.md headline. Run:

    python -m dreamerv4uwm.planning.experiments.study.analysis.report \
        --input-dir /scratch/mcts_sweep/run1 --out-dir /scratch/mcts_sweep/run1/analysis
"""
from __future__ import annotations

from pathlib import Path

from .aggregate import load_shards, sanity_report, save
from .correlate import analyze, headline, factor_outcome_table
from . import plots


def run(input_dir: str, out_dir: str) -> None:
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    df = load_shards(input_dir)

    report = sanity_report(df)
    print(report)
    save(df, od / "trees_all.parquet")

    tables = analyze(df)
    for name, t in tables.items():
        t.to_csv(od / f"{name}.csv", index=False)

    figs = plots.make_all(df, od / "figures")

    head_pol = headline(tables, "g_1shot")
    head_rand = headline(tables, "g_shootN")
    factors = factor_outcome_table(df)

    lines = ["# MCTS sweep analysis report\n",
             "## Dataset\n```\n" + report + "\n```\n",
             "## Which knobs move the outcome (factor vs outcome, Spearman)\n```",
             factors[factors.outcome == "g_1shot"].head(12).to_string(index=False),
             "```\n",
             "## " + head_pol.splitlines()[0] + "\n```",
             "\n".join(head_pol.splitlines()[1:]), "```\n",
             "## " + head_rand.splitlines()[0] + "\n```",
             "\n".join(head_rand.splitlines()[1:]), "```\n",
             f"## Figures\n" + "\n".join(f"- `{Path(f).name}`" for f in figs)]
    (od / "REPORT.md").write_text("\n".join(lines))
    print(f"\nwrote tables, {len(figs)} figures, and REPORT.md to {od}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    run(args.input_dir, args.out_dir)


if __name__ == "__main__":
    main()
