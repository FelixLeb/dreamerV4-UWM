"""Correlate process metrics with planning outcome.

Three complementary views (plan §8):
* **Rank correlation** (Spearman) + **mutual information** — MI catches non-monotone
  effects a monotone correlation misses (e.g. ctx_noise's inverted-U).
* **Partial rank correlation** controlling for the swept knobs — does a metric predict
  outcome *beyond* the factors that were dialed? (i.e. is it a real online signal, not
  just a proxy for a knob).
* **Feature importance** — a gradient-boosted model + permutation importance over the
  whole metric vector, plus standardized OLS betas as a linear complement.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import schema

MIN_N = 20


# ---------------------------------------------------------------------------
# pairwise stats
# ---------------------------------------------------------------------------

def _pair(df, a, b):
    d = df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
    return d[a].to_numpy(float), d[b].to_numpy(float)


def _mutual_info(x, y) -> float:
    from sklearn.feature_selection import mutual_info_regression
    if x.size < MIN_N or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(mutual_info_regression(x[:, None], y, random_state=0)[0])


def _residualize(v, Z):
    """Residuals of v regressed on Z (+ intercept)."""
    Z1 = np.column_stack([np.ones(len(v)), Z])
    beta, *_ = np.linalg.lstsq(Z1, v, rcond=None)
    return v - Z1 @ beta


def partial_spearman(df, x, y, controls) -> float:
    """Spearman(x, y | controls): rank everything, residualize x and y on the
    ranked controls, correlate the residuals."""
    controls = [c for c in controls if c in df.columns]
    cols = [x, y] + controls
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < MIN_N:
        return np.nan
    R = d.rank()
    rx, ry = R[x].to_numpy(float), R[y].to_numpy(float)
    if not controls:
        return float(stats.pearsonr(rx, ry)[0])
    Z = R[controls].to_numpy(float)
    Z = Z[:, Z.std(0) > 0]                       # drop constant controls (unswept factors in OFAT)
    if Z.shape[1] == 0:
        return float(stats.pearsonr(rx, ry)[0])
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(d)), Z])) < Z.shape[1] + 1:
        return np.nan
    ex, ey = _residualize(rx, Z), _residualize(ry, Z)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan
    return float(stats.pearsonr(ex, ey)[0])


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def metric_outcome_table(df, metrics=None, outcomes=None, controls=None) -> pd.DataFrame:
    metrics = metrics or schema.process_cols(df)
    outcomes = schema.present(outcomes or schema.OUTCOMES, df)
    controls = schema.present(controls or schema.FACTORS, df)
    rows = []
    for out in outcomes:
        for m in metrics:
            x, y = _pair(df, m, out)
            if x.size < MIN_N:
                continue
            sr, sp = stats.spearmanr(x, y)
            rows.append(dict(
                metric=m, link=schema.link_of(m), outcome=out,
                spearman=float(sr), p=float(sp), abs_spearman=abs(float(sr)),
                mutual_info=_mutual_info(x, y),
                partial_vs_factors=partial_spearman(df, m, out, controls),
                n=int(x.size),
            ))
    tab = pd.DataFrame(rows)
    return tab.sort_values(["outcome", "abs_spearman"], ascending=[True, False]).reset_index(drop=True)


def factor_outcome_table(df, factors=None, outcomes=None) -> pd.DataFrame:
    factors = schema.present(factors or schema.FACTORS, df)
    outcomes = schema.present(outcomes or schema.OUTCOMES, df)
    rows = []
    for out in outcomes:
        for f in factors:
            if df[f].nunique(dropna=True) < 2:
                continue
            x, y = _pair(df, f, out)
            if x.size < MIN_N:
                continue
            sr, sp = stats.spearmanr(x, y)
            rows.append(dict(factor=f, outcome=out, spearman=float(sr), p=float(sp),
                             abs_spearman=abs(float(sr)), mutual_info=_mutual_info(x, y),
                             n=int(x.size)))
    return pd.DataFrame(rows).sort_values(["outcome", "abs_spearman"],
                                          ascending=[True, False]).reset_index(drop=True)


def feature_importance(df, outcome="g_1shot", metrics=None, n_estimators=300) -> pd.DataFrame:
    """GBR permutation importance + standardized OLS beta for `outcome`."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    metrics = metrics or schema.process_cols(df)
    d = df[metrics + [outcome]].replace([np.inf, -np.inf], np.nan)
    y = d[outcome]
    d = d[y.notna()]
    y = d[outcome].to_numpy(float)
    X = d[metrics].to_numpy(float)
    # median-impute predictors to keep rows
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)
    if len(y) < MIN_N:
        return pd.DataFrame(columns=["metric", "link", "gbr_importance", "ols_beta_std"])
    gbr = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=3,
                                    subsample=0.8, random_state=0)
    gbr.fit(X, y)
    perm = permutation_importance(gbr, X, y, n_repeats=20, random_state=0)
    # standardized OLS betas
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ys)), Xs]), ys, rcond=None)
    tab = pd.DataFrame(dict(metric=metrics, link=[schema.link_of(m) for m in metrics],
                            gbr_importance=perm.importances_mean,
                            ols_beta_std=beta[1:]))
    tab["abs_beta"] = tab["ols_beta_std"].abs()
    return tab.sort_values("gbr_importance", ascending=False).reset_index(drop=True)


def per_regime(df, metric, outcome="g_1shot", regime="c_ucb", n_bins=3) -> pd.DataFrame:
    """Spearman(metric, outcome) within bins of a regime column."""
    d = df.copy()
    if d[regime].nunique() <= n_bins:
        d["_reg"] = d[regime]
    else:
        d["_reg"] = pd.qcut(d[regime], n_bins, duplicates="drop")
    rows = []
    for reg, g in d.groupby("_reg", observed=True):
        x, y = _pair(g, metric, outcome)
        if x.size < MIN_N:
            continue
        sr, sp = stats.spearmanr(x, y)
        rows.append(dict(regime=str(reg), metric=metric, outcome=outcome,
                         spearman=float(sr), p=float(sp), n=int(x.size)))
    return pd.DataFrame(rows)


def analyze(df) -> dict:
    return {
        "metric_outcome": metric_outcome_table(df),
        "factor_outcome": factor_outcome_table(df),
        "importance_g_1shot": feature_importance(df, "g_1shot"),
        "importance_g_shootN": feature_importance(df, "g_shootN"),
    }


def headline(tables: dict, outcome="g_1shot", k=12) -> str:
    mo = tables["metric_outcome"]
    top = mo[mo.outcome == outcome].head(k)
    lines = [f"Top {k} early-warning metrics for {outcome} (|Spearman|, MI, partial|factors):"]
    for _, r in top.iterrows():
        lines.append(f"  {r.metric:28s} [{r.link:12s}] rho={r.spearman:+.3f} "
                     f"mi={r.mutual_info:.3f} partial={r.partial_vs_factors:+.3f}")
    return "\n".join(lines)


def main(argv=None):
    import argparse
    from .dataset import build as load_shards
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    df = load_shards(args.input_dir)
    tables = analyze(df)
    print(headline(tables, "g_1shot"))
    if args.out_dir:
        from pathlib import Path
        od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
        for name, t in tables.items():
            t.to_csv(od / f"{name}.csv", index=False)
        print("wrote", len(tables), "tables to", od)


if __name__ == "__main__":
    main()
