"""Planning on top of the UWM denoiser. See ``README.md`` in this directory.

Three core modules — the planner and nothing else:

* :mod:`rollout` — batched generative rollout primitives (``policy`` / ``transition`` /
  ``imagine`` / ``autoregressive``) sharing one set of diversity knobs. These are the
  planner's edge samplers, and also what the policy-diversity experiments pull on.
* :mod:`reward`  — the objective: what makes one imagined state better than another.
* :mod:`mcts`    — the search itself. Nodes are latent states, edges are short
  ``pi_prior`` rollouts.

And three around it:

* :mod:`world`       — setup: load the denoiser/tokenizer, build a ``decode_fn``, source
  initial contexts from real demos.
* :mod:`evaluate`    — readouts: what a plan achieved (``plan_peak`` / ``plan_last``) and
  the no-search controls it should be measured against (``compute_baselines``).
* :mod:`diagnostics` — **debugging only**: task-specific state descriptors for inspecting
  a finished search. Never used inside the planner.
"""
from . import mcts, rollout, reward
from .rollout import policy, transition, imagine, autoregressive
from .reward import (RewardFn, RewardModel, ZeroReward, GoalLatentReward,
                     CallableReward, TCenterReward, TCenterStraightReward,
                     TCenterAngleReward, DINOGoalReward, load_dino,
                     score_loc, score_t_centered,
                     score_t_centered_straight, score_t_centered_angle, annotate_t)
from .mcts import MCTS, PlanConfig, Node

__all__ = [
    "rollout", "reward", "mcts",
    "policy", "transition", "imagine", "autoregressive",
    "RewardFn", "RewardModel", "ZeroReward", "GoalLatentReward", "CallableReward",
    "TCenterReward", "TCenterStraightReward", "TCenterAngleReward",
    "DINOGoalReward", "load_dino",
    "score_loc", "score_t_centered", "score_t_centered_straight",
    "score_t_centered_angle", "annotate_t",
    "MCTS", "PlanConfig", "Node",
]

# `world`, `evaluate` and `diagnostics` are deliberately NOT imported here: `world` pulls
# in hydra + the dataset layer and `diagnostics` needs cv2, so importing the planner stays
# cheap. Import them explicitly, e.g. `from dreamerv4uwm.planning.world import ...`.
