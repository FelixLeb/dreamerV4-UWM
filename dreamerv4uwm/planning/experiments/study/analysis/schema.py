"""Column taxonomy for the sweep CSVs: which columns are knobs (factors),
outcomes (dependent), or process metrics (candidate predictors), plus the mapping
of process metrics onto the L1–L5 causal chain (plan §1). Everything else keys off
these lists, so adding a metric only means listing it here.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

# identifiers / bookkeeping — never analysed as variables
META = ["config_id", "config_tag", "reward_kind", "descriptor_kind", "window_idx",
        "t0", "init_id", "plan_seed", "plan_secs", "dtype"]

# swept knobs (independent variables)
FACTORS = ["ctx_noise", "horizon", "sim_horizon", "branching", "action_temp",
           "action_prior", "action_noise", "action_noise_dist", "edge_mode",
           "c_ucb", "max_depth", "n_iterations", "sim_rollouts",
           "K_steps", "gamma", "max_ctx", "n_min", "ctx_noise_honest"]

# dependent variables (did planning help?). Both baselines share the plan's lookahead &
# construction (depth-deep, re-conditioned); g_random = vs ONE undirected rollout,
# g_greedy = vs best-of-N random shooting. Each comes in two flavours: the PEAK objective
# (best frame anywhere, MPC-style) and the LAST objective (the trajectory's final state).
OUTCOMES = ["g_random", "g_greedy", "delta_over_root", "tree_peak",
            "g_random_last", "g_greedy_last", "tree_last"]

# outcome intermediates that are not themselves predictors
_OUTCOME_AUX = ["root_reward", "random_peak", "greedy_peak", "best_node_value",
                "best_edge_val_on_plan", "best_edge_val_tree",
                "random_last", "greedy_last"]

# gain outcomes: planning "succeeds" when these are positive (it beat the baseline).
GAIN_OUTCOMES = ["g_random", "g_greedy", "delta_over_root", "g_random_last", "g_greedy_last"]
PEAK_THRESHOLD = 0.7   # tree_peak / tree_last >= this = reached a well-centred T (task success)


def add_success(df: pd.DataFrame) -> List[str]:
    """Add a ``success_<outcome>`` flag for each outcome present. Gains succeed when
    positive (planning helped over the matched baseline); ``tree_peak`` / ``tree_last``
    succeed at the task threshold (a well-centred T — passed through, resp. ended on).
    Mutates ``df`` in place; returns the columns added."""
    added = []
    for oc in GAIN_OUTCOMES:
        if oc in df.columns:
            df[f"success_{oc}"] = (df[oc] > 0).astype(int)
            added.append(f"success_{oc}")
    for oc in ("tree_peak", "tree_last"):
        if oc in df.columns:
            df[f"success_{oc}"] = (df[oc] >= PEAK_THRESHOLD).astype(int)
            added.append(f"success_{oc}")
    return added

# process metrics grouped by causal-chain link (plan §1). Only those present in the
# dataframe are used; missing ones (e.g. fidelity E-metrics from M6) are ignored.
LINKS = {
    "L1_sampler": ["div_action_div_mean", "div_action_dmin_mean", "root_action_div",
                   "root_action_dmin"],
    "L2_diversity": ["div_outcome_div_mean", "div_outcome_dmin_mean", "root_outcome_div",
                     "root_outcome_dmin", "div_bci_mean", "root_bci",
                     "div_duplicate_rate_mean", "root_duplicate_rate",
                     "div_pose_action_ratio_mean", "root_pose_action_ratio",
                     "div_found_frac_mean", "root_found_frac", "edge_val_std",
                     # fidelity (M6, optional)
                     "ood_rate", "reward_degenerate_rate", "h_star"],
    "L3_value": ["val_std", "val_spread", "q_margin",
                 "val_std_depth1", "val_std_depth2", "val_std_depth3"],
    "L4_selection": ["exploit_explore_ratio", "mean_explore_term", "visit_entropy",
                     "commit_top1", "visit_value_corr", "entropy_auc",
                     "reco_agreement", "selection_depth_exploration"],
    "L5_structure": ["subtree_size_gini", "mean_visited_depth", "eff_branching_realized",
                     "eff_branching_geom", "max_depth", "n_nodes", "n_expanded",
                     "expansion_efficiency", "best_action_switches",
                     "root_value_stability"],
}


def process_cols(df: pd.DataFrame) -> List[str]:
    """Numeric columns that are candidate predictors (metrics), i.e. not meta,
    factor, outcome, or outcome-aux."""
    excluded = set(META + FACTORS + OUTCOMES + _OUTCOME_AUX + ["n_forward"])
    out = []
    for c in df.columns:
        if c in excluded or c.startswith("success"):     # success_* are derived labels
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1:
            out.append(c)
    return out


def link_of(metric: str) -> str:
    for link, cols in LINKS.items():
        if metric in cols:
            return link
    return "other"


def present(cols: List[str], df: pd.DataFrame) -> List[str]:
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------------------------
# data dictionary: one line per column (single source of truth for the docs,
# the notebook, and anything else that needs to explain the schema)
# ---------------------------------------------------------------------------

DESCRIPTIONS = {
    # --- meta / bookkeeping ---
    "config_id": "index of the config in the sweep (0..n_configs-1)",
    "config_tag": "human tag of the config: 'base' or 'ofat.<knob>=<value>'",
    "reward_kind": "reward that scored this tree: center | straight (axis) | angle (upright)",
    "descriptor_kind": "diversity descriptor: pose (axis theta) | heading (up/down-resolved)",
    "window_idx": "dataset index of the demo window the initial context came from",
    "t0": "frame offset inside that window where the Tc-frame context starts",
    "init_id": "index of the curated initial state (decision point)",
    "plan_seed": "RNG seed of the planner for this tree",
    "plan_secs": "wall-clock seconds spent building this tree",
    "edge_mode": "edge sampler: 'imagine' (joint) | 'two_stage' (policy->WM whole horizon) | 'autoregressive' (step-by-step)",
    "dtype": "autocast dtype used for the rollouts (bfloat16)",
    # --- factors (the swept knobs) ---
    "horizon": "H - frames per EDGE rollout (how long each tree edge is)",
    "branching": "B - children created per expansion",
    "sim_horizon": "H_sim - frames per SIMULATION rollout (the value estimate)",
    "sim_rollouts": "M - parallel rollouts per simulation",
    "n_iterations": "search budget: tree-building iterations",
    "c_ucb": "UCT exploration constant",
    "gamma": "per-frame discount inside a simulation rollout",
    "n_min": "min visits for a node to be a valid returned solution",
    "max_depth": "cap on tree depth",
    "K_steps": "denoiser Euler integration steps per rollout",
    "ctx_noise": "noise mixed into the observation context (0=clean) - the diversity lever",
    "ctx_noise_honest": "tell the model the true context-noise level (vs. claim 'clean')",
    "action_temp": "std of the action noise prior (sampling temperature)",
    "action_prior": "action noise-prior distribution: normal (action_temp*N(0,I)) | uniform (std-matched U)",
    "action_noise": "(autoregressive edge only) magnitude of extra noise added to each policy action (0=none)",
    "action_noise_dist": "(autoregressive edge only) shape of that added noise: normal | uniform (std-matched)",
    "max_ctx": "max context frames kept per node",
    # --- outcomes ---
    "tree_peak": "best single-frame reward over the frames of the RETURNED PLAN",
    "root_reward": "reward of the start state (last context frame)",
    "random_peak": "best reward over ONE depth-deep re-conditioned rollout (plan lookahead, one sample per edge, no search)",
    "greedy_peak": "best reward over N depth-deep re-conditioned rollouts (random shooting, best-of-N)",
    "g_random": "tree_peak - random_peak: planning gain over a single undirected rollout at the plan's lookahead (horizon*max_depth, re-conditioned)",
    "g_greedy": "tree_peak - greedy_peak: search gain over best-of-N random shooting at the same lookahead",
    "delta_over_root": "tree_peak - root_reward: did the plan improve on the start state at all?",
    # --- terminal-objective ("last") counterparts: score where the trajectory ENDS, not
    # the best frame it passes through. Only ever compared like-with-like (last vs last).
    "tree_last": "reward of the plan's FINAL state (last frame of its last edge)",
    "random_last": "reward of the FINAL state of ONE depth-deep re-conditioned rollout (no search)",
    "greedy_last": "best FINAL-state reward over N depth-deep re-conditioned rollouts (random shooting)",
    "g_random_last": "tree_last - random_last: terminal-objective gain over a single undirected rollout",
    "g_greedy_last": "tree_last - greedy_last: terminal-objective gain over best-of-N random shooting",
    # derived success flags (added by dataset.build via schema.add_success):
    "success_g_random": "derived label: 1 if g_random > 0 (planning beat a single undirected rollout)",
    "success_g_greedy": "derived label: 1 if g_greedy > 0 (search beat best-of-N random shooting)",
    "success_delta_over_root": "derived label: 1 if delta_over_root > 0 (plan beat the start state)",
    "success_tree_peak": "derived label: 1 if tree_peak >= 0.7 (passed through a well-centred T)",
    "success_g_random_last": "derived label: 1 if g_random_last > 0 (terminal objective, vs one rollout)",
    "success_g_greedy_last": "derived label: 1 if g_greedy_last > 0 (terminal objective, vs best-of-N)",
    "success_tree_last": "derived label: 1 if tree_last >= 0.7 (ENDED on a well-centred T)",
    "best_node_value": "highest backed-up MEAN VALUE among nodes (cumulative-sum scale)",
    "best_edge_val_tree": "max terminal edge reward found anywhere in the tree",
    "best_edge_val_on_plan": "max terminal edge reward along the returned plan path",
    # --- structural metrics ---
    "n_nodes": "total nodes in the tree",
    "n_forward": "batched world-model calls used (compute-cost proxy)",
    "n_expanded": "nodes that were given children",
    "expansion_efficiency": "(n_nodes-1)/n_forward: how much compute bought new structure",
    "mean_visited_depth": "visit-weighted mean node depth; ~1 = shallow, never looks ahead",
    "eff_branching_realized": "mean #children with >=1 visit per expanded node (vs nominal B)",
    "eff_branching_geom": "n_nodes^(1/max_depth): geometric branching estimate",
    "root_width": "number of root children (= branching)",
    "subtree_size_gini": "Gini of root-children subtree sizes; 0 = uniform bush, high = selective",
    "plan_len": "number of edges in the returned plan",
    # --- allocation / selection metrics ---
    "visit_entropy": "normalised entropy of root visit shares; 1 = uniform (undecided), 0 = committed",
    "commit_top1": "visit share of the most-visited root child; high = decisive",
    "visit_value_corr": "Spearman(visits, values) over root children: is the bandit tracking value?",
    "exploit_explore_ratio": "UCB explore bonus / value spread; >>1 = noise-dominated, <<1 = greedy",
    "mean_explore_term": "mean magnitude of the UCB exploration bonus",
    "reco_agreement": "1 if the most-visited root child is also the highest-value one",
    "selection_depth_exploration": "deepest depth where most-visited != highest-value child (-1 = never)",
    "entropy_auc": "mean of the visit-entropy trajectory over iterations (did it ever concentrate?)",
    "best_action_switches": "#times the recommended root child changed during the search",
    "root_value_stability": "std of the root value over the last ~5 iterations; high = unconverged",
    # --- value metrics ---
    "val_std": "std of the root children's mean values; ->0 = children indistinguishable",
    "val_spread": "max - mean of the root children's values",
    "q_margin": "(best - 2nd best)/std of root values: decision confidence; ~0 = coin-flip",
    "val_std_depth1": "mean sibling value-std at interior depth 1 (discrimination at the root)",
    "val_std_depth2": "mean sibling value-std at interior depth 2 (does discrimination decay?)",
    "val_std_depth3": "mean sibling value-std at interior depth 3 (often NaN: few deep sibling sets)",
    # --- diversity metrics (div_*_mean = averaged over ALL expansions) ---
    "edge_val_std": "std of terminal edge REWARDS across the tree; ->0 = edges equally (un)rewarding",
    "div_action_div_mean": "mean pairwise distance between sibling ACTION sequences (sampler diversity)",
    "div_action_dmin_mean": "min pairwise action distance (near-duplicate action detector)",
    "div_outcome_div_mean": "mean pairwise distance between sibling decoded T-POSES (outcome diversity)",
    "div_outcome_dmin_mean": "min pairwise T-pose distance (near-duplicate outcome detector)",
    "div_bci_mean": "branch-collapse index 1 - N_eff/B; 0 = all edges distinct, ->1 = collapsed",
    "div_duplicate_rate_mean": "fraction of siblings having a near-duplicate outcome",
    "div_pose_action_ratio_mean": "outcome diversity / action diversity: WHICH stage collapsed",
    "div_found_frac_mean": "fraction of sibling states where a valid T was detected",
    # --- same diversity metrics, root expansion only (where the decision lives) ---
    "root_action_div": "action diversity at the ROOT expansion",
    "root_action_dmin": "min pairwise action distance at the root expansion",
    "root_outcome_div": "T-pose outcome diversity at the ROOT expansion",
    "root_outcome_dmin": "min pairwise T-pose distance at the root expansion",
    "root_bci": "branch-collapse index at the ROOT expansion (the headline collapse number)",
    "root_duplicate_rate": "fraction of root children having a near-duplicate outcome",
    "root_pose_action_ratio": "outcome/action diversity ratio at the root expansion",
    "root_found_frac": "fraction of root children where a valid T was detected",
}

_KIND_ORDER = {"meta": 0, "factor": 1, "outcome": 2, "outcome-aux": 3, "metric": 4}


def kind_of(col: str) -> str:
    """Which family a column belongs to: meta / factor / outcome / outcome-aux / metric."""
    if col in META:
        return "meta"
    if col in FACTORS:
        return "factor"
    if col in OUTCOMES:
        return "outcome"
    if col in _OUTCOME_AUX or col.startswith("success"):
        return "outcome-aux"
    return "metric"


def data_dictionary(df: pd.DataFrame, kind: Optional[str] = None) -> pd.DataFrame:
    """One row per column of ``df``: kind, causal link, dtype and a one-line
    description. Undocumented columns are flagged so nothing goes unexplained."""
    rows = []
    for c in df.columns:
        k = kind_of(c)
        rows.append({
            "column": c,
            "kind": k,
            "link": link_of(c) if k == "metric" else "",
            "dtype": str(df[c].dtype),
            "description": DESCRIPTIONS.get(c, "** UNDOCUMENTED **"),
        })
    out = pd.DataFrame(rows)
    out["_o"] = out["kind"].map(_KIND_ORDER)
    out = out.sort_values(["_o", "link", "column"]).drop(columns="_o").reset_index(drop=True)
    if kind is not None:
        out = out[out["kind"] == kind].reset_index(drop=True)
    return out
