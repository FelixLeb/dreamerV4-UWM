"""Planning on top of the UWM denoiser.

Two layers:

* :mod:`rollout` — batched generative rollout primitives (``policy`` /
  ``transition`` / ``imagine``) with shared diversity knobs. These back both the
  policy-mode diversity experiments and the planner.
* :mod:`mcts` — a continuous-action Monte-Carlo tree search whose nodes are
  latent states and whose edges are short ``pi_prior`` rollouts, scored by a
  basic pluggable :mod:`reward`.
"""
from . import rollout, reward, mcts, easy_mcts
from .rollout import policy, transition, imagine
from .reward import (RewardFn, ZeroReward, GoalLatentReward, CallableReward,
                     TCenterReward, score_t_centered, annotate_t)
from .mcts import MCTS, PlanConfig, Node, Edge
from .easy_mcts import EasyMCTS, EasyPlanConfig, EasyNode

__all__ = [
    "rollout", "reward", "mcts", "easy_mcts",
    "policy", "transition", "imagine",
    "RewardFn", "ZeroReward", "GoalLatentReward", "CallableReward",
    "TCenterReward", "score_t_centered", "annotate_t", "score_loc",
    "MCTS", "PlanConfig", "Node", "Edge",
    "EasyMCTS", "EasyPlanConfig", "EasyNode",
]