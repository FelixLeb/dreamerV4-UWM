"""Plots for the sweep (plan §8): context-noise (and other factor) response curves,
metric-vs-outcome correlation heatmap, the diversity->value->outcome mechanism
scatter, and per-link importance bars. Saves PNGs; headless (Agg) backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import schema
from .correlate import metric_outcome_table

# metrics that tell the L1->L5 story, for the response-curve panel
STORY = ["root_bci", "div_outcome_div_mean", "div_action_div_mean", "val_std",
         "q_margin", "exploit_explore_ratio", "visit_entropy", "g_rand", "g_policy"]


def _agg(df, factor, col):
    g = df.groupby(factor)[col]
    m = g.mean()
    sem = g.std() / np.sqrt(g.count().clip(lower=1))
    return m.index.to_numpy(float), m.to_numpy(float), sem.to_numpy(float)


def response_curves(df, factor="ctx_noise", metrics=None, out=None):
    """mean ± SEM of each metric vs a swept factor (the headline inverted-U figure)."""
    metrics = schema.present(metrics or STORY, df)
    if df[factor].nunique() < 2:
        return None
    n = len(metrics)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for i, met in enumerate(metrics):
        ax = axes[i // ncol][i % ncol]
        x, m, s = _agg(df.replace([np.inf, -np.inf], np.nan), factor, met)
        ax.errorbar(x, m, yerr=s, marker="o", capsize=3)
        ax.set_title(met, fontsize=9)
        ax.set_xlabel(factor)
        ax.grid(alpha=0.3)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Response curves vs {factor}", fontsize=12)
    fig.tight_layout()
    return _save(fig, out)


def corr_heatmap(df, top=25, out=None):
    """Spearman(metric, outcome) heatmap for the top metrics by |correlation|."""
    tab = metric_outcome_table(df)
    if tab.empty:
        return None
    keep = (tab.groupby("metric")["abs_spearman"].max()
            .sort_values(ascending=False).head(top).index.tolist())
    piv = (tab[tab.metric.isin(keep)]
           .pivot(index="metric", columns="outcome", values="spearman")
           .reindex(keep))
    fig, ax = plt.subplots(figsize=(1.6 * piv.shape[1] + 3, 0.35 * len(keep) + 1.5))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(keep))); ax.set_yticklabels(keep, fontsize=8)
    for i in range(len(keep)):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Spearman")
    ax.set_title("Metric vs outcome (rank correlation)")
    fig.tight_layout()
    return _save(fig, out)


def mechanism_scatter(df, x="div_outcome_div_mean", y="val_std", hue="g_policy", out=None):
    """Diversity -> value separation, coloured by outcome: the L2->L3->outcome chain."""
    for c in (x, y, hue):
        if c not in df.columns:
            return None
    d = df[[x, y, hue]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(d[x], d[y], c=d[hue], cmap="viridis", s=25, alpha=0.8)
    fig.colorbar(sc, ax=ax, label=hue)
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(f"{x} → {y}  (colour = {hue})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out)


def link_importance_bars(df, outcome="g_policy", out=None):
    """|Spearman| of each metric with outcome, coloured by causal-chain link."""
    tab = metric_outcome_table(df)
    tab = tab[tab.outcome == outcome].sort_values("abs_spearman", ascending=True)
    if tab.empty:
        return None
    links = list(schema.LINKS) + ["other"]
    cmap = {lk: plt.cm.tab10(i) for i, lk in enumerate(links)}
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(tab))))
    ax.barh(tab.metric, tab.abs_spearman, color=[cmap[l] for l in tab.link])
    ax.set_xlabel(f"|Spearman| with {outcome}")
    ax.set_title(f"Metric importance for {outcome} (colour = causal link)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[l]) for l in links]
    ax.legend(handles, links, fontsize=7, loc="lower right")
    fig.tight_layout()
    return _save(fig, out)


def _save(fig, out):
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out
    return fig


def make_all(df, out_dir, factor="ctx_noise"):
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    made = []
    made.append(response_curves(df, factor=factor, out=od / f"response_{factor}.png"))
    made.append(corr_heatmap(df, out=od / "corr_heatmap.png"))
    made.append(mechanism_scatter(df, out=od / "mechanism_scatter.png"))
    made.append(link_importance_bars(df, "g_policy", out=od / "importance_g_policy.png"))
    made.append(link_importance_bars(df, "g_rand", out=od / "importance_g_rand.png"))
    return [m for m in made if m]


def main(argv=None):
    import argparse
    from .aggregate import load_shards
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--factor", default="ctx_noise")
    args = ap.parse_args(argv)
    df = load_shards(args.input_dir)
    made = make_all(df, args.out_dir, factor=args.factor)
    print(f"wrote {len(made)} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
