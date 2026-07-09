"""Planning on top of the UWM denoiser.

Two layers:

* :mod:`rollout` — batched generative rollout primitives (``policy`` /
  ``transition`` / ``imagine``) with shared diversity knobs. These back both the
  policy-mode diversity experiments and the planner.
* :mod:`mcts` — a continuous-action Monte-Carlo tree search whose nodes are
  latent states and whose edges are short ``pi_prior`` rollouts, scored by a
  basic pluggable :mod:`reward`.
"""
from . import mcts, rollout, reward
from .rollout import policy, transition, imagine
from .reward import (RewardFn, RewardModel, ZeroReward, GoalLatentReward,
                     CallableReward, TCenterReward, score_loc, score_t_centered,
                     annotate_t)
from .mcts import MCTS, PlanConfig, Node

__all__ = [
    "rollout", "reward", "mcts",
    "policy", "transition", "imagine",
    "RewardFn", "RewardModel", "ZeroReward", "GoalLatentReward", "CallableReward",
    "TCenterReward", "score_loc", "score_t_centered", "annotate_t",
    "MCTS", "PlanConfig", "Node",
]