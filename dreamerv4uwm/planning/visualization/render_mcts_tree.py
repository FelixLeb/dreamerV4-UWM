"""Static, scale-free picture of an MCTS trace written by ``make_mcts_trace_pushT.py``.

Unlike the manim scene (which animates the search into a fixed 16:9 frame and
therefore only fits a small tree), this renders **one image** whose canvas grows
with the tree: every leaf gets a fixed-width column and every depth a fixed-height
row, so a 5-deep / 200-node tree is simply a bigger PNG, not an unreadable one.

What you see:
    * one box per node, holding its decoded world-model state, border coloured by
      ``edge_val`` (red = bad, amber, green = the red T is centred);
    * edge thickness ∝ child visit count — where the search actually spent effort;
    * the recommended plan path in gold, the chosen best node ringed;
    * never-visited nodes (created by an expansion but never selected) dimmed;
    * a header with the run's knobs and its final diagnostics.

Usage:
    python render_mcts_tree.py mcts_trace_test-tree.json
    python render_mcts_tree.py mcts_trace_test-tree.json -o tree.pdf --style box
    python render_mcts_tree.py mcts_trace_test-tree.json --cell-w 0.9 --no-labels

Only needs numpy + matplotlib (Pillow to draw thumbnails); no manim, no GPU.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle

# same red -> amber -> green ramp the manim scene uses for node borders
CMAP = LinearSegmentedColormap.from_list("edge_val", ["#D64550", "#E8C547", "#3FB47F"])
NORM = Normalize(0.0, 1.0)

BG = "#14161A"
INK = "#ECECEC"
MUTE = "#9AA0A6"
GOLD = "#F4C430"
ROOT_COL = "#C9CDD2"

# matplotlib refuses to save a figure larger than 2**16 px on a side
MAX_PX = 30000


# ---------------------------------------------------------------------------
# layout (fallback only — make_mcts_trace_pushT.py normally bakes x/y into the
# JSON via its own layout_tree(); this keeps the script usable on a raw trace)
# ---------------------------------------------------------------------------
def ensure_layout(nodes):
    if all("x" in v and "y" in v for v in nodes.values()):
        return
    children = {nid: [] for nid in nodes}
    for nid, v in nodes.items():
        if v["parent"] is not None:
            children[v["parent"]].append(nid)
    for nid in children:
        children[nid].sort()
    cursor = [0.0]

    def rec(nid):
        kids = children[nid]
        if not kids:
            x = cursor[0]
            cursor[0] += 1.0
        else:
            for k in kids:
                rec(k)
            x = sum(nodes[k]["x"] for k in kids) / len(kids)
        nodes[nid]["x"] = x
        nodes[nid]["y"] = -float(nodes[nid]["depth"])

    rec(next(nid for nid, v in nodes.items() if v["parent"] is None))


# ---------------------------------------------------------------------------
def _header_text(meta, nodes):
    """Two lines: the knobs that produced this tree, then what came out of it."""
    def g(k, fmt="{}"):
        return fmt.format(meta[k]) if k in meta else "?"

    knobs = ("H={}  B={}  depth<={}  iters={}  c_ucb={}  gamma={}  "
             "sim=({}x{})  ctx_noise={}").format(
        g("horizon"), g("branching"), g("max_depth"), g("n_iterations"),
        g("c_ucb"), g("gamma"), g("sim_rollouts"), g("sim_horizon"), g("ctx_noise"))

    visited = sum(1 for v in nodes.values() if v.get("n_visit", 0) > 0)
    diag = ("{} nodes ({} visited)  ·  reached depth {}  ·  "
            "visit_entropy={}  val_spread={}  edge_val std={} in [{}, {}]").format(
        len(nodes), visited, g("max_reached_depth"), g("visit_entropy"),
        g("val_spread"), g("edge_val_std"), g("edge_val_min"), g("edge_val_max"))

    if "start_reward" in meta and "best_reward" in meta:
        diag += "  ·  reward {:.3f} -> {:.3f}".format(
            meta["start_reward"], meta["best_reward"])
    return knobs, diag


# inches per leaf column / depth row, per style — thumbnails need room, the
# compact styles don't. cell_w=None on the CLI means "use the style's default".
CELL_DEFAULTS = {"thumbs": (0.62, 1.25), "box": (0.18, 0.72), "dot": (0.13, 0.62)}


def render(trace_path, out_path=None, style="auto", cell_w=None, cell_h=None,
           thumb_frac=0.86, labels=True, dpi=140, dim_unvisited=True):
    trace_path = os.path.abspath(trace_path)
    base = os.path.dirname(trace_path)
    with open(trace_path) as f:
        tr = json.load(f)

    nodes = {int(k): v for k, v in tr["nodes"].items()}
    meta = tr.get("meta", {})
    ensure_layout(nodes)

    root = meta.get("root", min(nodes))
    plan = set(meta.get("plan_path", []))       # note: excludes the root
    best = meta.get("best_node")

    xs = np.array([v["x"] for v in nodes.values()])
    ys = np.array([v["y"] for v in nodes.values()])
    ncols = max(xs.max() - xs.min(), 1.0) + 1.0
    nrows = max(ys.max() - ys.min(), 1.0) + 1.0

    # ---- decide whether thumbnails are worth drawing ----------------------
    have_imgs = all(os.path.exists(os.path.join(base, v["img"]))
                    for v in nodes.values() if "img" in v) and \
        any("img" in v for v in nodes.values())
    if style == "auto":
        style = "thumbs" if have_imgs else "box"
    elif style == "thumbs" and not have_imgs:
        print("[warn] frames/ not found next to the trace — falling back to --style box",
              file=sys.stderr)
        style = "box"

    # ---- canvas: fixed inches per column/row, so the tree never gets cramped
    d_w, d_h = CELL_DEFAULTS[style]
    cell_w = d_w if cell_w is None else cell_w
    cell_h = d_h if cell_h is None else cell_h
    pad_x, pad_y, header_h = 0.5, 0.45, 0.95
    fig_w = ncols * cell_w + 2 * pad_x
    fig_h = nrows * cell_h + 2 * pad_y + header_h
    if max(fig_w, fig_h) * dpi > MAX_PX:
        dpi = int(MAX_PX / max(fig_w, fig_h))
        print(f"[info] tree is large — dropping dpi to {dpi} to stay under "
              f"{MAX_PX}px", file=sys.stderr)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([pad_x / fig_w, pad_y / fig_h,
                       1 - 2 * pad_x / fig_w,
                       1 - (2 * pad_y + header_h) / fig_h])
    ax.set_facecolor(BG)
    ax.set_axis_off()

    # node box size in *data* units, derived from the physical cell size so the
    # thumbnail and its border always agree regardless of tree dimensions
    w_data = thumb_frac
    h_data = thumb_frac * cell_w / cell_h
    ax.set_xlim(xs.min() - 1.15, xs.max() + 0.6)
    ax.set_ylim(ys.min() - 0.75, ys.max() + 0.45)

    # ---- edges (drawn first, under the nodes) -----------------------------
    visits = np.array([v.get("n_visit", 0) for v in nodes.values()], float)
    vmax = max(visits.max(), 1.0)
    plain, gold = [], []
    plain_lw, gold_lw = [], []
    for nid, v in nodes.items():
        p = v["parent"]
        if p is None:
            continue
        seg = [(nodes[p]["x"], nodes[p]["y"]), (v["x"], v["y"])]
        lw = 0.7 + 3.3 * (v.get("n_visit", 0) / vmax)
        if nid in plan:
            gold.append(seg)
            gold_lw.append(lw)
        else:
            plain.append(seg)
            plain_lw.append(lw)
    if plain:
        ax.add_collection(LineCollection(plain, colors="#4A4F57",
                                         linewidths=plain_lw, zorder=1))
    if gold:
        ax.add_collection(LineCollection(gold, colors=GOLD,
                                         linewidths=[lw + 1.6 for lw in gold_lw],
                                         zorder=2))

    # ---- nodes ------------------------------------------------------------
    px_per_col = cell_w * dpi
    fs = float(np.clip(px_per_col * 0.115, 3.5, 9.0))     # label size follows cell size
    for nid, v in nodes.items():
        x, y = v["x"], v["y"]
        is_root = nid == root
        n_visit = v.get("n_visit", 0)
        alpha = 0.42 if (dim_unvisited and n_visit == 0 and not is_root) else 1.0
        col = ROOT_COL if is_root else CMAP(NORM(v.get("edge_val", 0.0)))

        if style == "thumbs":
            arr = plt.imread(os.path.join(base, v["img"]))
            # OffsetImage's dpi_cor scales by dpi/72, so undo that to land on an
            # exact pixel width — otherwise thumbnails overflow their column
            zoom = (px_per_col * thumb_frac) / arr.shape[1] * (72.0 / dpi)
            ab = AnnotationBbox(OffsetImage(arr, zoom=zoom, alpha=alpha),
                                (x, y), frameon=False, pad=0.0, zorder=3)
            ax.add_artist(ab)
        elif style == "box":
            ax.add_patch(Rectangle((x - w_data / 2, y - h_data / 2), w_data, h_data,
                                   facecolor=col, edgecolor="none",
                                   alpha=alpha * 0.85, zorder=3))
        else:                                              # dot
            ax.plot(x, y, "o", ms=max(2.0, px_per_col * 0.09), color=col,
                    alpha=alpha, zorder=3)

        if style in ("thumbs", "box"):
            on_plan = nid in plan or is_root
            ax.add_patch(Rectangle(
                (x - w_data / 2, y - h_data / 2), w_data, h_data,
                facecolor="none", edgecolor=GOLD if on_plan else col,
                linewidth=(2.6 if on_plan else 1.5) * min(1.0, px_per_col / 60),
                alpha=alpha, zorder=4))
        if nid == best:
            ax.add_patch(Rectangle(
                (x - w_data * 0.62, y - h_data * 0.62), w_data * 1.24, h_data * 1.24,
                facecolor="none", edgecolor=GOLD, linewidth=1.2, linestyle=":",
                zorder=5))

        if labels and px_per_col >= 26:
            val = v.get("value")
            # stacked, not inline: an "n=.. V=.." one-liner is wider than a
            # column and collides with the siblings on either side
            txt = f"n={n_visit}" + ("" if val is None else f"\nV={val:.1f}")
            ax.text(x, y - h_data / 2 - 0.05, txt, linespacing=1.15,
                    ha="center", va="top", fontsize=fs,
                    color=GOLD if nid in plan else MUTE, alpha=alpha, zorder=6)

    # depth ruler down the left margin
    for d in range(int(-ys.min()) + 1):
        ax.text(xs.min() - 1.05, -d, f"d{d}", ha="left", va="center",
                fontsize=fs + 0.5, color="#5A6069", zorder=6)

    # ---- header -----------------------------------------------------------
    knobs, diag = _header_text(meta, nodes)
    fig.text(pad_x / fig_w, 1 - 0.30 / fig_h,
             "MCTS search tree  ·  {}".format(meta.get("regime", os.path.basename(trace_path))),
             ha="left", va="top", fontsize=13, color=INK, weight="bold")
    fig.text(pad_x / fig_w, 1 - 0.60 / fig_h, knobs,
             ha="left", va="top", fontsize=8.5, color=MUTE, family="monospace")
    fig.text(pad_x / fig_w, 1 - 0.85 / fig_h, diag,
             ha="left", va="top", fontsize=8.5, color=MUTE, family="monospace")
    fig.text(1 - pad_x / fig_w, 1 - 0.30 / fig_h,
             "gold = plan   ·   border = edge_val (red bad → green centred)   ·   "
             "edge width ∝ visits   ·   faded = never visited",
             ha="right", va="top", fontsize=8.5, color=MUTE)

    out_path = out_path or os.path.join(
        base, os.path.basename(trace_path).replace(".json", "") + "_tree.png")
    fig.savefig(out_path, facecolor=BG, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_path}  ({fig_w:.1f}x{fig_h:.1f} in @ {dpi} dpi = "
          f"{int(fig_w*dpi)}x{int(fig_h*dpi)} px, {len(nodes)} nodes, style={style})")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="mcts_trace_<regime>.json")
    ap.add_argument("-o", "--out", default=None, help="output image (.png/.pdf/.svg)")
    ap.add_argument("--style", default="auto", choices=["auto", "thumbs", "box", "dot"],
                    help="thumbs = decoded states; box/dot = compact, for huge trees")
    ap.add_argument("--cell-w", type=float, default=None,
                    help="inches per leaf column (canvas grows with the tree); "
                         "default depends on --style")
    ap.add_argument("--cell-h", type=float, default=None,
                    help="inches per depth row; default depends on --style")
    ap.add_argument("--thumb-frac", type=float, default=0.86,
                    help="node size as a fraction of one column")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--no-labels", action="store_true", help="hide the n/V captions")
    ap.add_argument("--no-dim", action="store_true", help="don't fade unvisited nodes")
    a = ap.parse_args()
    render(a.trace, a.out, style=a.style, cell_w=a.cell_w, cell_h=a.cell_h,
           thumb_frac=a.thumb_frac, labels=not a.no_labels, dpi=a.dpi,
           dim_unvisited=not a.no_dim)


if __name__ == "__main__":
    main()
