"""Simple charts from the sweep dataframe.

Every chart takes column NAMES and returns a matplotlib ``Axes``. Two optional args
on every chart:
    save="fig.png"  -> also save it to disk
    ax=my_axes      -> draw into an existing subplot (for a grid of charts)

That's it. Making a new chart is a few lines of matplotlib on ``df[col]`` --- copy one
of these as a template.

    charts.response(charts.ofat(df, "ctx_noise"), "ctx_noise", "g_1shot")   # a knob sweep
    charts.scatter(df, "edge_val_std", "g_1shot", color="ctx_noise")        # metric vs outcome
    charts.hist(df, "root_bci")                                             # a distribution
    charts.corr_bars(df, "g_1shot")                                        # what predicts the outcome
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

BLUE, RED, GREY = "#4c78a8", "#e45756", "#888888"

# columns to exclude when ranking "what predicts the outcome" (metric_cols):
_META = {"config_id", "config_tag", "window_idx", "t0", "init_id",   # ids / bookkeeping
         "plan_seed", "plan_secs", "dtype", "edge_mode"}
_OUTCOMES = {"g_1shot", "g_shootN", "g_1shot_fair", "g_shootN_fair",  # the outcomes themselves
             "delta_over_root", "tree_peak", "root_reward",
             "oneshot_peak", "shootN_peak", "oneshot_fair_peak", "shootN_fair_peak",
             "best_node_value", "best_edge_val_on_plan", "best_edge_val_tree"}
_OTHERS = {"n_forward"}                                               # compute cost, not a diagnostic
# (derived success_<outcome> flags are also excluded, by name prefix, in metric_cols)


# --- tiny helpers -----------------------------------------------------------

def metric_cols(df) -> list:
    """Numeric columns worth plotting/correlating: drops ids, outcomes, and constants.
    (Swept knobs are kept --- they are legitimate predictors.)"""
    skip = _META | _OUTCOMES | _OTHERS
    return [c for c in df.columns if c not in skip and not c.startswith("success")
            and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1]


def ofat(df, factor) -> pd.DataFrame:
    """Rows of the one-factor-at-a-time slice for ``factor``: the ``base`` config plus
    the ``ofat.<factor>=...`` configs. Wrap a knob sweep in this so the curve isn't
    diluted by configs that merely sit at the baseline value."""
    return df[df["config_tag"].str.startswith(f"ofat.{factor}=") | (df["config_tag"] == "base")]


def _ax(ax, figsize):
    if ax is not None:
        return ax, False
    _, ax = plt.subplots(figsize=figsize)
    return ax, True


def _done(ax, save, made):
    if made:
        ax.figure.tight_layout()
    if save is not None:
        ax.figure.savefig(save, dpi=150, bbox_inches="tight")
    return ax


# --- charts -----------------------------------------------------------------

def response(df, factor, y, ax=None, save=None):
    """Mean +/- SEM of ``y`` at each value of ``factor`` (a line with error bars)."""
    g = df.groupby(factor)[y]
    ax, made = _ax(ax, (5, 3.4))
    ax.errorbar(g.mean().index, g.mean().to_numpy(), yerr=g.sem().to_numpy(),
                marker="o", capsize=3, color=BLUE)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.set(xlabel=factor, ylabel=y, title=f"{y} vs {factor}")
    return _done(ax, save, made)


def scatter(df, x, y, color=None, ax=None, save=None):
    """One dot per tree; optionally colour by a third column."""
    d = df[[x, y] + ([color] if color else [])].replace([float("inf"), float("-inf")], pd.NA).dropna()
    ax, made = _ax(ax, (5, 4))
    sc = ax.scatter(d[x], d[y], c=(d[color] if color else BLUE), cmap="viridis", s=15, alpha=0.6)
    if color:
        ax.figure.colorbar(sc, ax=ax, label=color)
    ax.set(xlabel=x, ylabel=y, title=f"{y} vs {x}")
    return _done(ax, save, made)


def hist(df, col, bins=30, ax=None, save=None):
    """Distribution of one column across trees."""
    ax, made = _ax(ax, (5, 3.4))
    ax.hist(df[col].dropna(), bins=bins, color=BLUE)
    ax.set(xlabel=col, ylabel="trees", title=f"distribution of {col}")
    return _done(ax, save, made)


def correlations(df, y, cols=None) -> pd.Series:
    """Spearman correlation of each metric with ``y``, sorted (a Series, not a chart)."""
    cols = cols or [c for c in metric_cols(df) if c != y]
    r = df[cols + [y]].corr(method="spearman", numeric_only=True)[y].drop(y)
    return r.sort_values()


def corr_bars(df, y, cols=None, top=15, ax=None, save=None):
    """Horizontal bars of the metrics most correlated (Spearman) with ``y``."""
    r = correlations(df, y, cols)
    r = r.reindex(r.abs().sort_values().index).tail(top)          # keep the strongest |corr|
    ax, made = _ax(ax, (6, max(3, 0.35 * len(r))))
    ax.barh(r.index, r.to_numpy(), color=[RED if v < 0 else BLUE for v in r])
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set(xlabel=f"Spearman correlation with {y}", title=f"what predicts {y}")
    return _done(ax, save, made)


def heatmap(df, cols, ax=None, save=None):
    """Spearman correlation matrix of the given columns."""
    C = df[cols].corr(method="spearman", numeric_only=True)
    ax, made = _ax(ax, (1 + 0.5 * len(cols), 1 + 0.5 * len(cols)))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    ax.figure.colorbar(im, ax=ax, label="Spearman")
    return _done(ax, save, made)
