"""Column taxonomy for the sweep CSVs: which columns are knobs (factors),
outcomes (dependent), or process metrics (candidate predictors), plus the mapping
of process metrics onto the L1–L5 causal chain (plan §1). Everything else keys off
these lists, so adding a metric only means listing it here.
"""
from __future__ import annotations

from typing import List

import pandas as pd

# identifiers / bookkeeping — never analysed as variables
META = ["config_id", "config_tag", "window_idx", "t0", "init_id", "plan_seed",
        "plan_secs", "dtype", "edge_mode"]

# swept knobs (independent variables)
FACTORS = ["ctx_noise", "horizon", "sim_horizon", "branching", "action_temp",
           "c_ucb", "max_depth", "n_iterations", "sim_rollouts", "K_steps",
           "gamma", "max_ctx", "n_min", "ctx_noise_honest"]

# dependent variables (did planning help?)
OUTCOMES = ["g_shootN", "g_1shot", "delta_over_root", "tree_peak"]

# outcome intermediates that are not themselves predictors
_OUTCOME_AUX = ["root_reward", "shootN_peak", "oneshot_peak", "best_node_value",
                "best_edge_val_on_plan", "best_edge_val_tree"]

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

# legacy outcome column names (pre-rename runs) -> current names
LEGACY_RENAME = {
    "g_policy": "g_1shot", "g_rand": "g_shootN",
    "policy_peak": "oneshot_peak", "rand_peak": "shootN_peak",
    "success_policy": "success_1shot",
}


def apply_legacy_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename old outcome columns to current names (idempotent) so CSVs written
    before the g_policy->g_1shot / g_rand->g_shootN rename still load."""
    ren = {k: v for k, v in LEGACY_RENAME.items() if k in df.columns and v not in df.columns}
    return df.rename(columns=ren) if ren else df


def process_cols(df: pd.DataFrame) -> List[str]:
    """Numeric columns that are candidate predictors (metrics), i.e. not meta,
    factor, outcome, or outcome-aux."""
    excluded = set(META + FACTORS + OUTCOMES + _OUTCOME_AUX
                   + ["success_1shot", "success_peak", "n_forward"])
    out = []
    for c in df.columns:
        if c in excluded:
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
