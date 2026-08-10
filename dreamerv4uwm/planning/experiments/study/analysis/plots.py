"""The five publication figures the LaTeX deck embeds (``presentation/``).

These are intentionally polished (Okabe-Ito palette, annotations) --- for everyday,
simple exploration use ``charts.py`` instead. Each ``fig_*(df, out=None)`` returns the
figure (inline) or saves it; ``make_deck`` writes all five as PDF+PNG.

    python -m ...analysis.plots --parquet <trees.parquet> --deck --out-dir presentation/figures
"""
from __future__ import annotations

import argparse
import re
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


# ---- outcome plumbing -------------------------------------------------------
# Every figure takes the outcome column to plot, so the same deck renders against the PEAK
# objective (best frame anywhere) or the LAST one (the plan's final state) -- different
# questions, see baselines.py. Supported: the four gains, plus tree_peak / tree_last.

def _is_gain(oc):
    """True for the differenced outcomes, where 0 is the baseline and the sign is the story."""
    return oc.startswith("g_")


def _label(oc):
    """Axis label: gains render as math, absolute columns verbatim."""
    m = re.fullmatch(r"g_(random|greedy)_(peak|last)", oc)
    return rf"$g_{{{m.group(1)}}}^{{\mathrm{{{m.group(2)}}}}}$" if m else oc


def _companion(df, oc):
    """Second series for the two-series panel, or None if there is no natural counterpart.

    A gain pairs with the OTHER gain at the same objective (random <-> greedy). An absolute
    outcome (``tree_peak`` / ``tree_last``) pairs with its matched baseline (``greedy_peak`` /
    ``greedy_last``), so the vertical gap between the two curves IS the corresponding gain."""
    if "_random_" in oc:
        other = oc.replace("_random_", "_greedy_")
    elif "_greedy_" in oc:
        other = oc.replace("_greedy_", "_random_")
    elif oc.startswith("tree_"):
        other = oc.replace("tree_", "greedy_")
    else:
        other = None
    return other if (other is not None and other in df.columns) else None


def fig_knob_effects(df, out=None, outcome="g_random_peak"):
    """Effect size (max-min mean ``outcome`` across a knob's settings), sorted."""
    knobs = ["horizon", "sim_horizon", "max_depth", "ctx_noise", "branching", "action_temp", "c_ucb"]
    rows = []
    for k in knobs:
        _, m, _ = ofat(df, k, outcome)
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
    ax.set_xlabel(f"effect on {'planning gain' if _is_gain(outcome) else 'outcome'}  "
                  f"(max - min mean {_label(outcome)} across settings)")
    ax.set_xlim(0, max(vals) * 1.35); ax.set_title("Which knobs move planning")
    return _save(fig, out)


def fig_knob_curves(df, out=None, outcome="g_random_peak"):
    knobs = [("horizon", "edge horizon"), ("max_depth", "max tree depth"),
             ("ctx_noise", "context noise"), ("sim_horizon", "simulation horizon")]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0))
    for ax, (k, lab) in zip(axes.ravel(), knobs):
        despine(ax)
        x, m, s = ofat(df, k, outcome)
        if _is_gain(outcome):      # 0 = "no better than the baseline"; meaningless for tree_*
            ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.errorbar(x, m, yerr=s, marker="o", ms=6, lw=2, capsize=3, color=ACCENT)
        ax.set_xlabel(lab); ax.set_ylabel(_label(outcome))
    title = "Planning gain" if _is_gain(outcome) else _label(outcome)
    fig.suptitle(f"{title} vs each knob (one-factor-at-a-time)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out)


def fig_ctxnoise(df, out=None, outcome="g_random_peak"):
    """Headline: diversity rises monotonically, but usefulness peaks then cliffs.

    Right panel plots ``outcome`` plus its counterpart from :func:`_companion` — the other
    gain for a gain, or the matched baseline for ``tree_peak`` / ``tree_last`` (in which case
    the gap between the two curves is that objective's gain)."""
    x, bci, bci_s = ofat(df, "ctx_noise", "root_bci")
    _, p, p_s = ofat(df, "ctx_noise", outcome)
    comp = _companion(df, outcome)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9)); despine(a1); despine(a2)
    a1.errorbar(x, bci, yerr=bci_s, marker="o", ms=6, lw=2, capsize=3, color=LINK_COLOR["L2_diversity"])
    a1.set_ylim(-0.03, 1.0); a1.set_xlabel("context noise"); a1.set_ylabel("branch-collapse index (root)")
    a1.set_title("More noise -> more diverse edges")
    if _is_gain(outcome):
        a2.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    a2.errorbar(x, p, yerr=p_s, marker="o", ms=6, lw=2, capsize=3, color=ACCENT, label=_label(outcome))
    c = None
    if comp is not None:
        _, c, c_s = ofat(df, "ctx_noise", comp)
        a2.errorbar(x, c, yerr=c_s, marker="s", ms=6, lw=2, capsize=3, color=ACCENT2, label=_label(comp))
    # annotations placed off the data's own range, so they land correctly whatever the scale
    span = float(np.nanmax(p) - np.nanmin(p)) or 1.0
    ipk = int(np.nanargmax(p))
    a2.annotate("sweet spot", (x[ipk], p[ipk]), (x[ipk] - 0.34, p[ipk] + 0.35 * span),
                fontsize=9.5, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT))
    if c is not None and _is_gain(comp) and np.isfinite(c[-1]) and c[-1] < 0:
        a2.annotate("incoherent /\nOOD diversity", (x[-1], c[-1]),
                    (x[-1] - 0.62, c[-1] + 0.45 * span), fontsize=9.5,
                    color=ACCENT2, ha="center", arrowprops=dict(arrowstyle="->", color=ACCENT2))
    a2.set_xlabel("context noise")
    a2.set_ylabel("planning gain" if _is_gain(outcome) else "reward")
    a2.set_title("...but usefulness peaks, then collapses"); a2.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, out)


def fig_predictors(df, out=None, outcome="g_random_peak"):
    """Top per-tree predictors of ``outcome``, by |Spearman|, coloured by causal link;
    partial correlation (controlling for all knobs) overlaid as a marker."""
    mets = ["edge_val_std", "val_spread", "val_std", "q_margin", "visit_entropy",
            "commit_top1", "exploit_explore_ratio", "subtree_size_gini",
            "mean_visited_depth", "div_action_div_mean"]
    controls = [c for c in ["horizon", "sim_horizon", "branching", "action_temp", "c_ucb",
                            "max_depth", "n_iterations", "ctx_noise"] if df[c].nunique() > 1]
    rows = [(m, stats.spearmanr(df[m], df[outcome], nan_policy="omit")[0],
             partial_spearman(df, m, outcome, controls), schema.link_of(m)) for m in mets]
    rows.sort(key=lambda r: abs(r[1]))
    fig, ax = plt.subplots(figsize=(7.8, 4.6)); despine(ax); ax.axvline(0, color=MUTED, lw=1)
    for i, (m, rho, par, lk) in enumerate(rows):
        ax.barh(i, rho, color=LINK_COLOR.get(lk, MUTED), height=0.6)
        ax.plot(par, i, "D", ms=6, color=INK, mfc="white", mew=1.4, zorder=5)
    ax.set_yticks(np.arange(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(f"Spearman correlation with {_label(outcome)}")
    ax.set_title("Per-tree early-warning signals  (diamond = partial, controlling for all knobs)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=LINK_COLOR[k]) for k in LINK_LABEL]
    ax.legend(handles, list(LINK_LABEL.values()), frameon=False, loc="lower right", fontsize=9)
    return _save(fig, out)


def fig_mechanism(df, out=None, outcome="g_random_peak"):
    """Reward-diversity of the edges predicts whether planning helps."""
    import pandas as pd
    sc = f"success_{outcome}"                       # added by schema.add_success in dataset.build
    cols = ["edge_val_std", outcome] + ([sc] if sc in df.columns else [])
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    d = d.assign(bin=pd.qcut(d.edge_val_std, 6, duplicates="drop"))
    g = d.groupby("bin", observed=True)
    x = g.edge_val_std.mean().to_numpy()
    gp, gp_s = g[outcome].mean().to_numpy(), g[outcome].sem().to_numpy()
    n_ax = 2 if sc in d.columns else 1              # no success flag -> left panel only
    fig, axes = plt.subplots(1, n_ax, figsize=(9.6 if n_ax == 2 else 5.0, 3.9), squeeze=False)
    a1 = axes[0, 0]; despine(a1)
    if _is_gain(outcome):
        a1.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    a1.fill_between(x, gp - gp_s, gp + gp_s, color=LINK_COLOR["L2_diversity"], alpha=0.18)
    a1.plot(x, gp, marker="o", ms=6, lw=2, color=LINK_COLOR["L2_diversity"])
    a1.set_xlabel("edge reward diversity  (edge_val_std)")
    a1.set_ylabel(f"mean {_label(outcome)}")
    a1.set_title("More reward-diverse edges -> better planning")
    if n_ax == 2:
        a2 = axes[0, 1]; despine(a2)
        a2.plot(x, 100 * g[sc].mean().to_numpy(), marker="o", ms=6, lw=2, color=LINK_COLOR["L2_diversity"])
        a2.set_xlabel("edge reward diversity  (edge_val_std)"); a2.set_ylabel("planning-success rate (%)")
        a2.set_ylim(0, 100); a2.set_title("...and more reliable planning")
    fig.tight_layout()
    return _save(fig, out)


def key_numbers(df, out=None, outcome="g_random_peak"):
    """Headline numbers quoted in the deck; writes key_numbers.txt if ``out`` is a dir."""
    L = [f"outcome={outcome}",
         f"rows={len(df)}  configs={df.config_id.nunique()}  inits={df.init_id.nunique()}"]
    for k in ["horizon", "max_depth", "ctx_noise", "sim_horizon", "c_ucb"]:
        x, m, _ = ofat(df, k, outcome)
        L.append(f"{k}: {outcome} " + ", ".join(f"{xi}->{mi:+.3f}" for xi, mi in zip(x, m)))
    text = "\n".join(L) + "\n"
    if out is not None:
        (Path(out) / "key_numbers.txt").write_text(text)
        print("  wrote key_numbers.txt")
    return text


DECK_FIGS = {"fig_knob_effects": fig_knob_effects, "fig_knob_curves": fig_knob_curves,
             "fig_ctxnoise": fig_ctxnoise, "fig_predictors": fig_predictors,
             "fig_mechanism": fig_mechanism}


def make_deck(df, out_dir, outcome="g_random_peak"):
    """Write all five figures (PDF + PNG) + key_numbers.txt for the LaTeX deck."""
    if outcome not in df.columns:
        raise KeyError(f"outcome {outcome!r} is not a column. Present outcomes: "
                       f"{[c for c in schema.OUTCOMES if c in df.columns]}")
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    for name, fn in DECK_FIGS.items():
        fig = fn(df, None, outcome=outcome)
        for ext in ("pdf", "png"):
            fig.savefig(od / f"{name}.{ext}", dpi=150)
        plt.close(fig); print("  wrote", name)
    key_numbers(df, od, outcome)
    return od


def main(argv=None):
    mpl.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", help="a trees.parquet")
    ap.add_argument("--input-dir", help="a dir of shard_*.csv (alternative to --parquet)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent
                    / "presentation" / "figures"))
    ap.add_argument("--deck", action="store_true", help="(default action) write the deck figures")
    ap.add_argument("--outcome", nargs="+", default=["g_random_peak"],
                    help="outcome column(s) to render the deck against, e.g. "
                         "--outcome tree_last g_random_last g_greedy_last. With more than one, "
                         "each gets its own sub-directory under --out-dir (filenames would "
                         "otherwise collide).")
    a = ap.parse_args(argv)
    if not (a.parquet or a.input_dir):
        ap.error("give --parquet or --input-dir")
    df = dataset.load(a.parquet) if a.parquet else dataset.build(a.input_dir)
    multi = len(a.outcome) > 1
    for oc in a.outcome:
        od = Path(a.out_dir) / oc if multi else Path(a.out_dir)
        print(f"[deck] outcome={oc} -> {od}")
        make_deck(df, od, oc)
    print("wrote deck figures to", a.out_dir)


if __name__ == "__main__":
    main()
