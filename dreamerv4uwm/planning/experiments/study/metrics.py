"""Tree-level diagnostic metrics for a finished MCTS planner.

``compute_tree_metrics(planner, descriptor)`` returns a flat ``dict`` of scalars
(one CSV row's worth) covering families A–E and the tree-internal part of F from
``../mcts_study_plan.md``. Baselines (g_rand, g_policy) and the root reward live
in ``baselines.py`` / ``run_tree.py`` because they need the model; everything here
is computed from the finished tree object (+ its ``trace`` for time-resolved
metrics) and a pluggable ``StateDescriptor`` for diversity.

Metric IDs (e.g. ``A3``, ``B4``) match the plan doc. Everything degrades to
``nan`` rather than raising when a tree is too small/degenerate to define a metric.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

NAN = float("nan")


# ---------------------------------------------------------------------------
# numeric helpers (numpy only)
# ---------------------------------------------------------------------------

def _std(x) -> float:
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
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2:
        return NAN
    ra = a.argsort().argsort().astype(float)
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
    """Effective number of distinct rows: greedy eps-clustering cluster count."""
    X = np.asarray(X, float)
    if X.shape[0] == 0:
        return NAN
    centers = []
    for row in X:
        if not any(np.linalg.norm(row - c) <= eps for c in centers):
            centers.append(row)
    return float(len(centers))


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
    return 1 + sum(_subtree_size(c) for c in node.children)


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------

def _structural(planner) -> Dict[str, float]:
    nodes = planner.all_nodes
    root = planner.root
    n = len(nodes)
    depths = np.array([nd.depth for nd in nodes], float)
    visits = np.array([nd.n_visit for nd in nodes], float)
    max_depth = float(depths.max()) if n else NAN
    mean_vdepth = float((visits * depths).sum() / visits.sum()) if visits.sum() > 0 else NAN

    expanded = [nd for nd in nodes if nd.children]
    realized_b = float(np.mean([sum(c.n_visit > 0 for c in nd.children) for nd in expanded])) if expanded else NAN
    geom_b = float(n ** (1.0 / max_depth)) if max_depth and max_depth > 0 else NAN

    sizes = [_subtree_size(c) for c in root.children] if root else []
    return {
        "n_nodes": float(n),
        "n_forward": float(planner.n_forward),
        "n_expanded": float(len(expanded)),
        "expansion_efficiency": float((n - 1) / max(planner.n_forward, 1)),
        "max_depth": max_depth,
        "mean_visited_depth": mean_vdepth,
        "eff_branching_realized": realized_b,
        "eff_branching_geom": geom_b,
        "root_width": float(len(root.children)) if root else NAN,
        "subtree_size_gini": _gini(sizes),
    }


def _root_children_stats(root):
    ch = root.children
    visits = np.array([c.n_visit for c in ch], float)
    vis_mask = visits > 0
    values = np.array([c.value if c.n_visit > 0 else NAN for c in ch], float)
    return ch, visits, values, vis_mask


def _allocation(planner) -> Dict[str, float]:
    root = planner.root
    ch, visits, values, vm = _root_children_stats(root)
    k = len(ch)
    out = {
        "visit_entropy": _norm_entropy(visits),
        "commit_top1": float(visits.max() / visits.sum()) if visits.sum() > 0 else NAN,
        "visit_value_corr": _spearman(visits[vm], values[vm]) if vm.sum() >= 2 else NAN,
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
    root = planner.root
    _, visits, values, vm = _root_children_stats(root)
    q = values[vm]
    out = {"val_std": _std(q),
           "val_spread": float(q.max() - q.mean()) if q.size >= 1 else NAN}
    if q.size >= 2:
        qs = np.sort(q)[::-1]
        out["q_margin"] = float((qs[0] - qs[1]) / (q.std() + 1e-9))
    else:
        out["q_margin"] = NAN
    # C3 discrimination decay: mean sibling val-std at each interior depth
    by_depth: Dict[int, List[float]] = {}
    for nd in planner.all_nodes:
        vis = [c.value for c in nd.children if c.n_visit > 0]
        if len(vis) >= 2:
            by_depth.setdefault(nd.depth, []).append(float(np.std(vis)))
    for d in (0, 1, 2):
        out[f"val_std_depth{d + 1}"] = float(np.mean(by_depth[d])) if d in by_depth else NAN
    return out


def _diversity(planner, descriptor) -> Dict[str, float]:
    nodes = planner.all_nodes
    term = [nd for nd in nodes if nd.edge_z is not None]
    out = {"edge_val_std": _std([nd.edge_val for nd in term]),
           "best_edge_val_tree": float(max((nd.edge_val for nd in term), default=NAN))}
    feat_by_id = {}
    if descriptor is not None and term:
        term_lat = torch.stack([nd.edge_z[-1] for nd in term], 0)
        feats, found = descriptor(term_lat)
        feat_by_id = {nd.id: (feats[i], bool(found[i])) for i, nd in enumerate(term)}

    def expansion(children):
        A = np.stack([c.edge_a.reshape(-1).float().cpu().numpy() for c in children])
        act_mean, act_min = _pairwise(A)
        res = {"action_div": act_mean, "action_dmin": act_min}
        if feat_by_id:
            fr = [feat_by_id[c.id] for c in children if c.id in feat_by_id]
            F = np.stack([f for f, ok in fr if ok]) if any(ok for _, ok in fr) else np.empty((0,))
            res["found_frac"] = float(np.mean([ok for _, ok in fr])) if fr else NAN
            if F.shape[0] >= 2:
                om, omin = _pairwise(F)
                neff = _eff_count(F, descriptor.dup_eps)
                res.update(outcome_div=om, outcome_dmin=omin,
                           bci=1.0 - neff / len(children),
                           duplicate_rate=_duplicate_rate(F, descriptor.dup_eps),
                           pose_action_ratio=(om / (act_mean + 1e-9)) if act_mean == act_mean else NAN)
            else:
                res.update(outcome_div=NAN, outcome_dmin=NAN, bci=NAN,
                           duplicate_rate=NAN, pose_action_ratio=NAN)
        return res

    expanded = [nd for nd in nodes if nd.children]
    per = [expansion(nd.children) for nd in expanded]
    keys = per[0].keys() if per else []
    for key in keys:
        vals = [p[key] for p in per if p.get(key) == p.get(key)]  # drop nan
        out[f"div_{key}_mean"] = float(np.mean(vals)) if vals else NAN
    # root expansion, reported separately (where the decision lives)
    if planner.root and planner.root.children:
        for key, val in expansion(planner.root.children).items():
            out[f"root_{key}"] = val
    return out


def _outcome_internal(planner, result) -> Dict[str, float]:
    out = {"best_node_value": float(planner._best_node().value) if planner.all_nodes else NAN}
    bp = (result or {}).get("best_path") or []
    out["best_edge_val_on_plan"] = float(max((n.edge_val for n in bp), default=NAN))
    out["plan_len"] = float(len(bp))
    return out


def _trajectory(planner) -> Dict[str, float]:
    """Time-resolved allocation metrics from the trace (needs trace=True)."""
    tr = getattr(planner, "trace", None)
    if not tr or not planner.root:
        return {"entropy_auc": NAN, "best_action_switches": NAN, "root_value_stability": NAN}
    root_id = planner.root.id
    child_ids = [c.id for c in planner.root.children]
    rv = {c: 0 for c in child_ids}
    rq = {c: -np.inf for c in child_ids}
    ent, argm, root_q = [], [], []
    for ev in tr.get("events", []):
        if ev.get("type") != "backprop":
            continue
        stats = {int(i): v for i, v in ev["stats"].items()}
        for c in child_ids:
            if c in stats:
                rv[c], rq[c] = stats[c][0], stats[c][1]
        if root_id in stats:
            root_q.append(stats[root_id][1])
        ent.append(_norm_entropy([rv[c] for c in child_ids]))
        qv = np.array([rq[c] for c in child_ids])
        if np.isfinite(qv).any():
            argm.append(int(np.nanargmax(np.where(np.isfinite(qv), qv, -np.inf))))
    switches = float(sum(argm[i] != argm[i - 1] for i in range(1, len(argm)))) if len(argm) > 1 else NAN
    kk = min(5, len(root_q))
    return {
        "entropy_auc": float(np.nanmean(ent)) if ent else NAN,
        "best_action_switches": switches,
        "root_value_stability": float(np.std(root_q[-kk:])) if kk >= 2 else NAN,
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
    m.update(_structural(planner))
    m.update(_allocation(planner))
    m.update(_value(planner))
    m.update(_diversity(planner, descriptor))
    m.update(_outcome_internal(planner, result))
    m.update(_trajectory(planner))
    return m
