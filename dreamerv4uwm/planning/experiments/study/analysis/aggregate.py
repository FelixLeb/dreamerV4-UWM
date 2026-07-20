"""Aggregate sweep CSV shards into one dataframe, with sanity checks and derived
outcome columns.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import schema


def load_shards(input_dir: str, pattern: str = "shard_*.csv") -> pd.DataFrame:
    """Concatenate all shard CSVs, de-duplicate on (config_id, init_id), and add
    derived outcome columns (``success_1shot``, ``success_peak``)."""
    files = sorted(glob.glob(str(Path(input_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f"no shards matching {pattern} in {input_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = schema.apply_legacy_names(df)   # upgrade pre-rename runs (g_policy->g_1shot, ...)
    df = df.drop_duplicates(subset=["config_id", "init_id"], keep="last").reset_index(drop=True)
    return derive_outcomes(df)


def derive_outcomes(df: pd.DataFrame, peak_tau: float = 0.7) -> pd.DataFrame:
    """Binary success labels: planning beat the policy prior, and the plan reached a
    well-centred T."""
    if "g_1shot" in df:
        df["success_1shot"] = (df["g_1shot"] > 0).astype(int)
    if "tree_peak" in df:
        df["success_peak"] = (df["tree_peak"] >= peak_tau).astype(int)
    return df


def sanity_report(df: pd.DataFrame) -> str:
    """Human-readable summary: sizes, NaN-heavy columns, degenerate trees,
    constant metrics."""
    lines = []
    lines.append(f"rows={len(df)}  configs={df['config_id'].nunique()}  "
                 f"inits={df['init_id'].nunique()}  tags={df['config_tag'].nunique()}")
    if "success_1shot" in df:
        lines.append(f"success_1shot rate = {df['success_1shot'].mean():.3f}   "
                     f"(g_1shot>0 on {int(df['success_1shot'].sum())}/{len(df)})")
    # NaN-heavy columns
    frac = df.isna().mean().sort_values(ascending=False)
    heavy = frac[frac > 0.2]
    if len(heavy):
        lines.append("cols >20% NaN: " + ", ".join(f"{c}={v:.0%}" for c, v in heavy.items()))
    # degenerate trees
    if "n_nodes" in df and "branching" in df:
        degen = (df["n_nodes"] <= df["branching"] + 1).mean()
        lines.append(f"degenerate trees (n_nodes<=B+1): {degen:.1%}")
    # constant metrics (uninformative)
    proc = schema.process_cols(df)
    const = [c for c in df.columns if c not in proc and c not in schema.META
             and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= 1
             and c not in schema.FACTORS]
    if const:
        lines.append(f"constant metric cols (dropped): {', '.join(const)}")
    lines.append(f"usable process metrics: {len(proc)}")
    return "\n".join(lines)


def save(df: pd.DataFrame, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="dir with shard_*.csv")
    ap.add_argument("--out", default=None, help="write aggregated table (.csv/.parquet)")
    args = ap.parse_args(argv)
    df = load_shards(args.input_dir)
    print(sanity_report(df))
    if args.out:
        save(df, args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
