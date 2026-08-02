"""Tree-level diagnostic metrics for a finished MCTS planner.

``compute_tree_metrics(planner, descriptor)`` returns a flat ``dict`` of scalars
(one CSV row's worth) covering families A–E and the tree-internal part of F from
``../mcts_study_plan.md``. Baselines (g_random_peak, g_greedy_peak) and the root reward live
in ``baselines.py`` / ``run_tree.py`` because they need the model; everything here
is computed from the finished tree object (+ its ``trace`` for time-resolved
metrics) and a pluggable ``StateDescriptor`` for diversity.

Metric IDs (e.g. ``A3``, ``B4``) match the plan doc; full per-column definitions live in
``analysis/METRICS_GUIDE.md``. Everything degrades to ``nan`` rather than raising when a
tree is too small / degenerate to define a metric.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

NAN = float("nan")


# ---------------------------------------------------------------------------
# numeric helpers (numpy only)
# ---------------------------------------------------------------------------

def _std(x) -> float:
    """Population std of a 1-D array; NaN if fewer than 2 points."""
    x = np.asarray(x, float)
    return float(x.std()) if x.size >= 2 else NAN


def _norm_entropy(counts) -> float:
    """Shannon entropy of a count vector, normalised to [0,1] by log(k)."""
    c = np.asarray(counts, float)
    tot = c.sum()
    if tot <= 0 or c.size < 2:
        return NAN
    p = c / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(c.size))


def _gini(x) -> float:
    x = np.asarray(x, float)
    if x.size < 2 or x.sum() <= 0:
        return NAN
    xs = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * xs).sum()) / (n * xs.sum()) - (n + 1) / n)


def _spearman(a, b) -> float:
    """Spearman rank correlation = Pearson on the ranks; NaN if either side is
    constant or shorter than 2."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2:
        return NAN
    ra = a.argsort().argsort().astype(float)   # argsort-of-argsort = rank of each element
    rb = b.argsort().argsort().astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return NAN
    return float(np.corrcoef(ra, rb)[0, 1])


def _pairwise(X) -> tuple:
    """(mean, min) pairwise Euclidean distance over rows of X."""
    X = np.asarray(X, float)
    if X.shape[0] < 2:
        return NAN, NAN
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(X.shape[0], k=1)
    dv = d[iu]
    return float(dv.mean()), float(dv.min())


def _eff_count(X, eps: float) -> float:
    """Effective number of distinct rows = #clusters under **complete-linkage**
    agglomerative clustering, cut at diameter ``eps`` (every pair *inside* a cluster is
    within ``eps``). Order-independent, and unlike single-linkage it does not *chain*
    (points at 0, eps, 2*eps stay 2 clusters, not 1)."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    if n == 0:
        return NAN
    if n == 1:
        return 1.0
    # sklearn merges only *below* the threshold; bump it so distance == eps still merges
    # (matches the "within eps" / <= convention used by _duplicate_rate and the old code)
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=float(eps) * (1 + 1e-9),
                                     linkage="complete").fit_predict(X)
    return float(len(set(labels)))


def _duplicate_rate(X, eps: float) -> float:
    """Fraction of rows that have some *other* row within eps."""
    X = np.asarray(X, float)
    n = X.shape[0]
    if n < 2:
        return NAN
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float((d.min(1) <= eps).mean())


def _subtree_size(node) -> int:
    """Number of nodes in the subtree rooted at ``node`` (node itself included)."""
    return 1 + sum(_subtree_size(c) for c in node.children)


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------

def _structural(planner) -> Dict[str, float]:
    """Family A - tree shape / size. First-line collapse detection: a healthy MCTS tree is
    bushy and *asymmetric* (it deepens promising lines); a degenerate one is a shallow
    uniform bush or a single unsupported spike."""
    nodes = planner.all_nodes
    root = planner.root
    n = len(nodes)
    depths = np.array([nd.depth for nd in nodes], float)
    visits = np.array([nd.n_visit for nd in nodes], float)
    # NB: ``max_depth_realised``, NOT ``max_depth`` — the latter is the PlanConfig *knob* (the
    # cap), and run_tree merges the config row and the metrics row into one flat dict. Reusing
    # the name silently overwrote the knob with this measurement.
    depth_realised = float(depths.max()) if n else NAN
    # visit-weighted mean depth: where the search actually spent its budget (~1 = never looks ahead)
    mean_vdepth = float((visits * depths).sum() / visits.sum()) if visits.sum() > 0 else NAN

    expanded = [nd for nd in nodes if nd.children]
    # realized branching: mean # of *visited* children per expanded node (vs the nominal `branching`)
    realized_b = float(np.mean([sum(c.n_visit > 0 for c in nd.children) for nd in expanded])) if expanded else NAN
    geom_b = float(n ** (1.0 / depth_realised)) if depth_realised and depth_realised > 0 else NAN

    sizes = [_subtree_size(c) for c in root.children] if root else []   # subtree size under each root arm
    return {
        "n_nodes": float(n),
        "n_forward": float(planner.n_forward),                          # world-model calls (cost proxy)
        "n_expanded": float(len(expanded)),
        "expansion_efficiency": float((n - 1) / max(planner.n_forward, 1)),  # new structure bought per forward
        "max_depth_realised": depth_realised,   # deepest node reached (<= the max_depth knob)
        "mean_visited_depth": mean_vdepth,
        "eff_branching_realized": realized_b,
        "eff_branching_geom": geom_b,
        "root_width": float(len(root.children)) if root else NAN,
        # Gini of root-subtree sizes: 0 = uniform bush (no selectivity), high = mass on a few lines
        "subtree_size_gini": _gini(sizes),
    }


def _root_children_stats(root):
    """The root's "arms": (children, visit counts N_i, mean values Q_i, visited-mask).
    Unvisited children get value = NaN (their ``Node.value`` would otherwise be -inf)."""
    ch = root.children
    visits = np.array([c.n_visit for c in ch], float)
    vis_mask = visits > 0
    values = np.array([c.value if c.n_visit > 0 else NAN for c in ch], float)
    return ch, visits, values, vis_mask


def _allocation(planner) -> Dict[str, float]:
    """Family B - how the search allocated visits over the root arms (exploration vs
    exploitation): 'is the bandit spending its budget where the value is?'."""
    root = planner.root
    ch, visits, values, vm = _root_children_stats(root)
    k = len(ch)
    out = {
        "visit_entropy": _norm_entropy(visits),          # 1 = uniform (undecided), 0 = committed
        "commit_top1": float(visits.max() / visits.sum()) if visits.sum() > 0 else NAN,  # top-arm visit share
        "visit_value_corr": _spearman(visits[vm], values[vm]) if vm.sum() >= 2 else NAN,  # do visits track value?
    }
    # B4 exploit/explore ratio — the value-scale catch
    if vm.sum() >= 1 and root.n_visit > 0:
        c_ucb = planner.cfg.c_ucb
        explore = c_ucb * np.sqrt(np.log(max(root.n_visit, 1)) / visits[vm])
        out["exploit_explore_ratio"] = float(explore.mean() / (_std(values[vm]) + 1e-9))
        out["mean_explore_term"] = float(explore.mean())
    else:
        out["exploit_explore_ratio"] = NAN
        out["mean_explore_term"] = NAN
    # B5 recommendation agreement (visits vs value argmax)
    if vm.sum() >= 1:
        out["reco_agreement"] = float(int(np.nanargmax(np.where(vm, visits, -1))
                                          == np.nanargmax(np.where(vm, values, -np.inf))))
    else:
        out["reco_agreement"] = NAN
    # B6 selection-depth of exploration (final-tree proxy: deepest internal node
    # whose most-visited child != highest-value child)
    depth_expl = -1
    for nd in planner.all_nodes:
        if len(nd.children) >= 2:
            cv = np.array([c.n_visit for c in nd.children], float)
            qv = np.array([c.value if c.n_visit > 0 else -np.inf for c in nd.children], float)
            if cv.max() > 0 and int(cv.argmax()) != int(qv.argmax()):
                depth_expl = max(depth_expl, nd.depth)
    out["selection_depth_exploration"] = float(depth_expl)
    return out


def _value(planner) -> Dict[str, float]:
    """Family C - do the backed-up values *separate* the root arms? If not, no selection
    rule can plan (the value-side face of collapse)."""
    root = planner.root
    _, visits, values, vm = _root_children_stats(root)
    q = values[vm]                                   # mean values of the visited arms
    out = {"val_std": _std(q),                       # ->0 = arms indistinguishable
           "val_spread": float(q.max() - q.mean()) if q.size >= 1 else NAN}
    if q.size >= 2:
        qs = np.sort(q)[::-1]                         # values, descending
        # decision margin: best-minus-2nd-best, normalised by the spread (~0 = coin-flip)
        out["q_margin"] = float((qs[0] - qs[1]) / (q.std() + 1e-9))
    else:
        out["q_margin"] = NAN
    # C3 discrimination decay: mean sibling value-std at each interior depth (does separation vanish deeper?)
    by_depth: Dict[int, List[float]] = {}
    for nd in planner.all_nodes:
        vis = [c.value for c in nd.children if c.n_visit > 0]
        if len(vis) >= 2:
            by_depth.setdefault(nd.depth, []).append(float(np.std(vis)))
    for d in (0, 1, 2):                               # -> val_std_depth1/2/3 (NaN deep: few sibling sets)
        out[f"val_std_depth{d + 1}"] = float(np.mean(by_depth[d])) if d in by_depth else NAN
    return out


def _diversity(planner, descriptor) -> Dict[str, float]:
    """Family D - are the edges out of a node diverse (the stated failure mode)? Measured
    per expansion (a sibling set) in two spaces: ACTION space (edge_a) and, via the
    ``descriptor``, decoded task-OUTCOME space (T-pose). Reported both aggregated over the
    whole tree (``div_*_mean``) and for the root expansion alone (``root_*``)."""
    nodes = planner.all_nodes
    term = [nd for nd in nodes if nd.edge_z is not None]      # every non-root node was reached by one edge
    out = {"edge_val_std": _std([nd.edge_val for nd in term]),          # spread of terminal edge REWARDS
           "best_edge_val_tree": float(max((nd.edge_val for nd in term), default=NAN))}
    # featurise every node's terminal latent once (one batched decode+segment) -> id -> (phi, found)
    feat_by_id = {}
    if descriptor is not None and term:
        term_lat = torch.stack([nd.edge_z[-1] for nd in term], 0)      # (M, N_lat, D_lat)
        feats, found = descriptor(term_lat)
        feat_by_id = {nd.id: (feats[i], bool(found[i])) for i, nd in enumerate(term)}

    def expansion(children):
        """Diversity metrics for ONE sibling set (the B children of one expansion)."""
        A = np.stack([c.edge_a.reshape(-1).float().cpu().numpy() for c in children])   # (B, H*n_act)
        act_mean, act_min = _pairwise(A)                              # L1: action-space diversity
        res = {"action_div": act_mean, "action_dmin": act_min}
        if feat_by_id:
            fr = [feat_by_id[c.id] for c in children if c.id in feat_by_id]
            F = np.stack([f for f, ok in fr if ok]) if any(ok for _, ok in fr) else np.empty((0,))
            res["found_frac"] = float(np.mean([ok for _, ok in fr])) if fr else NAN   # fraction with a valid T
            if F.shape[0] >= 2:                                      # need >=2 valid outcomes to compare
                om, omin = _pairwise(F)                              # L2: outcome (T-pose) diversity
                neff = _eff_count(F, descriptor.dup_eps)             # effective # of distinct outcomes
                res.update(outcome_div=om, outcome_dmin=omin,
                           bci=1.0 - neff / len(children),           # branch-collapse index: 0=distinct, ->1=collapsed
                           duplicate_rate=_duplicate_rate(F, descriptor.dup_eps),
                           # outcome/action ratio: localises the collapse (model-insensitive vs sampler)
                           pose_action_ratio=(om / (act_mean + 1e-9)) if act_mean == act_mean else NAN)
            else:
                res.update(outcome_div=NAN, outcome_dmin=NAN, bci=NAN,
                           duplicate_rate=NAN, pose_action_ratio=NAN)
        return res

    expanded = [nd for nd in nodes if nd.children]
    per = [expansion(nd.children) for nd in expanded]
    keys = per[0].keys() if per else []
    for key in keys:                                                 # aggregate each metric over all expansions
        vals = [p[key] for p in per if p.get(key) == p.get(key)]     # `x == x` is False for NaN -> drops NaNs
        out[f"div_{key}_mean"] = float(np.mean(vals)) if vals else NAN
    # root expansion, reported separately (where the decision actually lives)
    if planner.root and planner.root.children:
        for key, val in expansion(planner.root.children).items():
            out[f"root_{key}"] = val
    return out


def _outcome_internal(planner, result) -> Dict[str, float]:
    """Family F (tree-internal part only). The reward-vs-baseline gains that actually grade
    the plan (``g_random_peak`` etc.) need the world model and live in ``baselines.py``; here we
    record only quantities readable straight off the finished tree / plan path."""
    out = {"best_node_value": float(planner._best_node().value) if planner.all_nodes else NAN}
    bp = (result or {}).get("best_path") or []       # nodes on the returned root->best path
    out["best_edge_val_on_plan"] = float(max((n.edge_val for n in bp), default=NAN))
    out["plan_len"] = float(len(bp))                 # number of edges in the returned plan
    return out


def _trajectory(planner) -> Dict[str, float]:
    """Time-resolved allocation metrics reconstructed from the planner trace
    (requires ``MCTS(trace=True)``; returns all-NaN otherwise).

    Each MCTS iteration ends in one ``backprop`` event whose ``stats`` snapshot
    ``[n_visit, value]`` for the nodes on that path. Replaying the events lets us follow,
    over iterations, the root-children visit vector and the root value -- i.e. how the
    search *converged*, which the final tree alone cannot show."""
    tr = getattr(planner, "trace", None)
    if not tr or not planner.root:
        return {"entropy_auc": NAN, "best_action_switches": NAN, "root_value_stability": NAN}
    root_id = planner.root.id
    child_ids = [c.id for c in planner.root.children]
    rv = {c: 0 for c in child_ids}          # running visit count per root child
    rq = {c: -np.inf for c in child_ids}    # running mean value per root child
    ent, argm, root_q = [], [], []
    for ev in tr.get("events", []):
        if ev.get("type") != "backprop":
            continue                        # one backprop event == one completed iteration
        stats = {int(i): v for i, v in ev["stats"].items()}   # id -> [n_visit, value] at that moment
        for c in child_ids:                 # a backprop touches only its path; unmentioned arms carry forward
            if c in stats:
                rv[c], rq[c] = stats[c][0], stats[c][1]
        if root_id in stats:
            root_q.append(stats[root_id][1])                   # root value trajectory
        ent.append(_norm_entropy([rv[c] for c in child_ids]))  # visit entropy at this iteration
        qv = np.array([rq[c] for c in child_ids])
        if np.isfinite(qv).any():
            argm.append(int(np.nanargmax(np.where(np.isfinite(qv), qv, -np.inf))))  # current best (argmax-value) arm
    # #times the recommended arm flipped over the search (high = unconverged)
    switches = float(sum(argm[i] != argm[i - 1] for i in range(1, len(argm)))) if len(argm) > 1 else NAN
    kk = min(5, len(root_q))
    return {
        "entropy_auc": float(np.nanmean(ent)) if ent else NAN,         # mean visit-entropy over iterations
        "best_action_switches": switches,
        "root_value_stability": float(np.std(root_q[-kk:])) if kk >= 2 else NAN,  # std of root value, last ~5 iters
    }


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def compute_tree_metrics(planner, descriptor=None, *, result: Optional[dict] = None) -> Dict[str, float]:
    """Flat dict of tree diagnostics. ``descriptor`` (a ``StateDescriptor``) enables
    the diversity family; pass ``None`` to skip it. ``result`` is ``planner.plan(...)``'s
    return dict (for the best-path outcome metrics)."""
    if not planner.all_nodes or planner.root is None:
        return {}
    m: Dict[str, float] = {}
    m.update(_structural(planner))                # A: tree shape
    m.update(_allocation(planner))                # B: visit allocation / selection
    m.update(_value(planner))                     # C: value separation
    m.update(_diversity(planner, descriptor))     # D: edge diversity (skipped if descriptor is None)
    m.update(_outcome_internal(planner, result))  # F: tree-internal outcome
    m.update(_trajectory(planner))                # B (time-resolved): needs trace=True
    return m
