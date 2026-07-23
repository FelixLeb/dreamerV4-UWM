"""Build and load the sweep dataset. One row = one MCTS tree.

    df = dataset.build("path/to/run/results", out="trees.parquet")   # shards -> parquet
    df = dataset.load("trees.parquet")                               # read it back
    print(dataset.overview(df))                                      # quick summary
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from . import schema


def build(shards_dir, out=None) -> pd.DataFrame:
    """Concatenate all ``shard_*.csv`` in ``shards_dir`` into one dataframe (one row per
    tree), drop duplicate ``(config_id, init_id)`` rows, add a ``success_<outcome>`` flag
    for each outcome (see ``schema.add_success``), and optionally save a parquet at ``out``."""
    files = sorted(glob.glob(str(Path(shards_dir) / "shard_*.csv")))
    if not files:
        raise FileNotFoundError(f"no shard_*.csv found in {shards_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["config_id", "init_id"]).reset_index(drop=True)
    schema.add_success(df)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    return df


def load(parquet) -> pd.DataFrame:
    """Load a previously-built parquet."""
    return pd.read_parquet(parquet)


def overview(df, outcome="g_1shot") -> str:
    """A short text summary: sizes, ``outcome`` success rate, and any mostly-empty columns."""
    lines = [f"{len(df)} trees  |  {df.config_id.nunique()} configs  |  {df.init_id.nunique()} inits"]
    sc = f"success_{outcome}"
    if sc in df.columns:
        lines.append(f"{outcome} success rate: {df[sc].mean():.1%}")
    na = df.isna().mean()
    heavy = na[na > 0.2]
    if len(heavy):
        lines.append("mostly-NaN columns: " + ", ".join(heavy.index))
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build the sweep parquet from shard CSVs.")
    ap.add_argument("--input-dir", required=True, help="dir containing shard_*.csv")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--outcome", default="g_1shot", help="which outcome's success rate to summarise")
    a = ap.parse_args(argv)
    df = build(a.input_dir, out=a.out)
    print(overview(df, a.outcome))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
