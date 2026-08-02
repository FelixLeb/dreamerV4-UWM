"""The five publication figures the LaTeX deck embeds (``presentation/``).

These are intentionally polished (Okabe-Ito palette, annotations) --- for everyday,
simple exploration use ``charts.py`` instead. Each ``fig_*(df, out=None)`` returns the
figure (inline) or saves it; ``make_deck`` writes all five as PDF+PNG.

    python -m ...analysis.plots --parquet <trees.parquet> --deck --out-dir presentation/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from . import dataset, schema
from .correlate import partial_spearman

# ---- style: Okabe-Ito (colourblind-safe) for the causal-chain links ----
INK, MUTED = "#222222", "#666666"
LINK_COLOR = {"L1_sampler": "#E69F00", "L2_diversity": "#009E73", "L3_value": "#0072B2",
              "L4_selection": "#D55E00", "L5_structure": "#CC79A7", "other": "#999999"}
LINK_LABEL = {"L1_sampler": "L1 sampler", "L2_diversity": "L2 diversity", "L3_value": "L3 value",
              "L4_selection": "L4 selection", "L5_structure": "L5 structure"}
ACCENT, ACCENT2 = "#0072B2", "#D55E00"

mpl.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#dddddd", lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def ofat(df, factor, col="g_random_peak"):
    """(x, mean, sem) of ``col`` at each ``factor`` setting, over that factor's OFAT
    slice (falls back to all rows when ``factor`` is not an OFAT axis). ``x`` is float for
    a numeric factor, else the categorical labels (strings) — so this never crashes on a
    string/bool knob like ``action_prior`` / ``edge_mode`` / ``ctx_noise_honest``."""
    m = df.config_tag.str.startswith(f"ofat.{factor}=") | (df.config_tag == "base")
    sub = df[m]
    if sub[factor].nunique() < 2:
        sub = df
    g = sub.replace([np.inf, -np.inf], np.nan).groupby(factor, observed=True)[col]
    xi = g.mean().index.to_numpy()
    x = xi.astype(float) if np.issubdtype(xi.dtype, np.number) else xi
    return x, g.mean().to_numpy(float), g.sem().to_numpy(float)


def _save(fig, out):
    if out is None:
        return fig
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    return str(out)


def fig_knob_effects(df, out=None):
    """Effect size (max-min mean g_random_peak across a knob's settings), sorted."""
    knobs = ["horizon", "sim_horizon", "max_depth", "ctx_noise", "branching", "action_temp", "c_ucb"]
    rows = []
    for k in knobs:
        _, m, _ = ofat(df, k)
        if len(m) < 2:
            continue
        eff = float(np.nanmax(m) - np.nanmin(m))
        peak_interior = 0 < int(np.nanargmax(m)) < len(m) - 1
        shape = ("~ flat" if eff < 0.05 else "inverted-U" if peak_interior
                 else "monotone up" if m[-1] >= m[0] else "monotone down")
        rows.append((k, eff, shape))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 3.8)); despine(ax)
    y = np.arange(len(rows)); vals = [r[1] for r in rows]
    ax.barh(y, vals, color=ACCENT, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    for i, (k, eff, shape) in enumerate(rows):
        ax.text(eff + 0.004, i, shape, va="center", ha="left", fontsize=9.5, color=INK)
    ax.set_xlabel("effect on planning gain  (max - min mean $g_{random}$ across settings)")
    ax.set_xlim(0, max(vals) * 1.35); ax.set_title("Which knobs move planning")
    return _save(fig, out)


def fig_knob_curves(df, out=None):
    knobs = [("horizon", "edge horizon"), ("max_depth", "max tree depth"),
             ("ctx_noise", "context noise"), ("sim_horizon", "simulation horizon")]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0))
    for ax, (k, lab) in zip(axes.ravel(), knobs):
        despine(ax)
        x, m, s = ofat(df, k)
        ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.errorbar(x, m, yerr=s, marker="o", ms=6, lw=2, capsize=3, color=ACCENT)
        ax.set_xlabel(lab); ax.set_ylabel("$g_{random}$")
    fig.suptitle("Planning gain vs each knob (one-factor-at-a-time)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out)


def fig_ctxnoise(df, out=None):
    """Headline: diversity rises monotonically, but usefulness peaks then cliffs."""
    x, bci, bci_s = ofat(df, "ctx_noise", "root_bci")
    _, gp, gp_s = ofat(df, "ctx_noise", "g_random_peak")
    _, gr, gr_s = ofat(df, "ctx_noise", "g_greedy_peak")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9)); despine(a1); despine(a2)
    a1.errorbar(x, bci, yerr=bci_s, marker="o", ms=6, lw=2, capsize=3, color=LINK_COLOR["L2_diversity"])
    a1.set_ylim(-0.03, 1.0); a1.set_xlabel("context noise"); a1.set_ylabel("branch-collapse index (root)")
    a1.set_title("More noise -> more diverse edges")
    a2.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    a2.errorbar(x, gp, yerr=gp_s, marker="o", ms=6, lw=2, capsize=3, color=ACCENT, label="$g_{random}$")
    a2.errorbar(x, gr, yerr=gr_s, marker="s", ms=6, lw=2, capsize=3, color=ACCENT2, label="$g_{greedy}$")
    ipk = int(np.argmax(gp))
    a2.annotate("sweet spot", (x[ipk], gp[ipk]), (x[ipk] - 0.34, gp[ipk] + 0.03),
                fontsize=9.5, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT))
    a2.annotate("incoherent /\nOOD diversity", (0.9, gr[-1]), (0.28, 0.05), fontsize=9.5,
                color=ACCENT2, ha="center", arrowprops=dict(arrowstyle="->", color=ACCENT2))
    a2.set_xlabel("context noise"); a2.set_ylabel("planning gain")
    a2.set_title("...but usefulness peaks, then collapses"); a2.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, out)


def fig_predictors(df, out=None):
    """Top per-tree predictors of g_random_peak, by |Spearman|, coloured by causal link;
    partial correlation (controlling for all knobs) overlaid as a marker."""
    mets = ["edge_val_std", "val_spread", "val_std", "q_margin", "visit_entropy",
            "commit_top1", "exploit_explore_ratio", "subtree_size_gini",
            "mean_visited_depth", "div_action_div_mean"]
    controls = [c for c in ["horizon", "sim_horizon", "branching", "action_temp", "c_ucb",
                            "max_depth", "n_iterations", "ctx_noise"] if df[c].nunique() > 1]
    rows = [(m, stats.spearmanr(df[m], df.g_random_peak, nan_policy="omit")[0],
             partial_spearman(df, m, "g_random_peak", controls), schema.link_of(m)) for m in mets]
    rows.sort(key=lambda r: abs(r[1]))
    fig, ax = plt.subplots(figsize=(7.8, 4.6)); despine(ax); ax.axvline(0, color=MUTED, lw=1)
    for i, (m, rho, par, lk) in enumerate(rows):
        ax.barh(i, rho, color=LINK_COLOR.get(lk, MUTED), height=0.6)
        ax.plot(par, i, "D", ms=6, color=INK, mfc="white", mew=1.4, zorder=5)
    ax.set_yticks(np.arange(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Spearman correlation with $g_{random}$")
    ax.set_title("Per-tree early-warning signals  (diamond = partial, controlling for all knobs)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=LINK_COLOR[k]) for k in LINK_LABEL]
    ax.legend(handles, list(LINK_LABEL.values()), frameon=False, loc="lower right", fontsize=9)
    return _save(fig, out)


def fig_mechanism(df, out=None):
    """Reward-diversity of the edges predicts whether planning helps."""
    import pandas as pd
    d = df[["edge_val_std", "g_random_peak", "success_g_random_peak"]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d.assign(bin=pd.qcut(d.edge_val_std, 6, duplicates="drop"))
    g = d.groupby("bin", observed=True)
    x = g.edge_val_std.mean().to_numpy()
    gp, gp_s = g.g_random_peak.mean().to_numpy(), g.g_random_peak.sem().to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9)); despine(a1); despine(a2)
    a1.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    a1.fill_between(x, gp - gp_s, gp + gp_s, color=LINK_COLOR["L2_diversity"], alpha=0.18)
    a1.plot(x, gp, marker="o", ms=6, lw=2, color=LINK_COLOR["L2_diversity"])
    a1.set_xlabel("edge reward diversity  (edge_val_std)"); a1.set_ylabel("mean $g_{random}$")
    a1.set_title("More reward-diverse edges -> better planning")
    a2.plot(x, 100 * g.success_g_random_peak.mean().to_numpy(), marker="o", ms=6, lw=2, color=LINK_COLOR["L2_diversity"])
    a2.set_xlabel("edge reward diversity  (edge_val_std)"); a2.set_ylabel("planning-success rate (%)")
    a2.set_ylim(0, 100); a2.set_title("...and more reliable planning")
    fig.tight_layout()
    return _save(fig, out)


def key_numbers(df, out=None):
    """Headline numbers quoted in the deck; writes key_numbers.txt if ``out`` is a dir."""
    L = [f"rows={len(df)}  configs={df.config_id.nunique()}  inits={df.init_id.nunique()}"]
    for k in ["horizon", "max_depth", "ctx_noise", "sim_horizon", "c_ucb"]:
        x, m, _ = ofat(df, k)
        L.append(f"{k}: g_random_peak " + ", ".join(f"{xi:g}->{mi:+.3f}" for xi, mi in zip(x, m)))
    text = "\n".join(L) + "\n"
    if out is not None:
        (Path(out) / "key_numbers.txt").write_text(text)
        print("  wrote key_numbers.txt")
    return text


DECK_FIGS = {"fig_knob_effects": fig_knob_effects, "fig_knob_curves": fig_knob_curves,
             "fig_ctxnoise": fig_ctxnoise, "fig_predictors": fig_predictors,
             "fig_mechanism": fig_mechanism}


def make_deck(df, out_dir):
    """Write all five figures (PDF + PNG) + key_numbers.txt for the LaTeX deck."""
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    for name, fn in DECK_FIGS.items():
        fig = fn(df, None)
        for ext in ("pdf", "png"):
            fig.savefig(od / f"{name}.{ext}", dpi=150)
        plt.close(fig); print("  wrote", name)
    key_numbers(df, od)
    return od


def main(argv=None):
    mpl.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", help="a trees.parquet")
    ap.add_argument("--input-dir", help="a dir of shard_*.csv (alternative to --parquet)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent
                    / "presentation" / "figures"))
    ap.add_argument("--deck", action="store_true", help="(default action) write the deck figures")
    a = ap.parse_args(argv)
    if not (a.parquet or a.input_dir):
        ap.error("give --parquet or --input-dir")
    df = dataset.load(a.parquet) if a.parquet else dataset.build(a.input_dir)
    make_deck(df, a.out_dir)
    print("wrote deck figures to", a.out_dir)


if __name__ == "__main__":
    main()
