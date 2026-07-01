"""Basic, pluggable reward functions for planning.

The current checkpoint has **no learned reward head** (``train_reward_model=
False``), and a reward model is *not* the focus of the project right now — so
these are deliberately simple, hand-specified scorers over tokenizer latents.
Swap in a learned head later by writing any object with the same call
signature.

Contract
--------
A reward function maps a batch of latent **states** to scalar rewards::

    reward_fn(z) -> r            z: (..., N_lat, D_lat)  ->  r: (...)

i.e. it reduces the two trailing latent dims and preserves all leading (batch /
time) dims. The planner calls it on imagined horizon states ``(B, H, N_lat,
D_lat)`` and gets per-frame rewards ``(B, H)``.
"""
from __future__ import annotations

from typing import Callable

import torch

RewardFn = Callable[[torch.Tensor], torch.Tensor]


class ZeroReward:
    """Reward ≡ 0. Use for pure world-model exploration / debugging the search."""

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return z.new_zeros(z.shape[:-2])


class GoalLatentReward:
    """Negative distance from each state's latent to a fixed **goal latent**.

    ``r(z) = -dist(z, goal) / scale``. With ``metric='l2'`` the distance is the
    mean-squared error over the ``(N_lat, D_lat)`` latent grid (so the scale is
    comparable across token counts); ``metric='cosine'`` uses ``1 - cos`` over
    the flattened latent. ``scale`` just rescales reward magnitude so the
    exploration constant in the search has a sane range.

    The goal is whatever clean latent you hand it — e.g. a target frame the
    robot should reach. Basic by design; replace with a learned critic later.
    """

    def __init__(self, goal_z: torch.Tensor, scale: float = 1.0, metric: str = "l2"):
        # goal_z: (N_lat, D_lat) or (1, 1, N_lat, D_lat) — squeezed to (N_lat, D_lat).
        g = goal_z.detach()
        while g.dim() > 2:
            g = g.squeeze(0)
        self.goal = g                      # (N_lat, D_lat)
        self.scale = float(scale)
        assert metric in ("l2", "cosine")
        self.metric = metric

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        goal = self.goal.to(device=z.device, dtype=z.dtype)
        if self.metric == "l2":
            dist = (z - goal).pow(2).mean(dim=(-1, -2))
        else:  # cosine over the flattened latent
            zf = z.flatten(-2)             # (..., N_lat*D_lat)
            gf = goal.flatten().expand_as(zf)
            dist = 1.0 - torch.cosine_similarity(zf, gf, dim=-1)
        return -dist / self.scale


class CallableReward:
    """Adapt a plain ``z -> r`` callable (e.g. a decoded-pixel scorer) to the
    reward contract, so arbitrary user functions drop into the planner."""

    def __init__(self, fn: RewardFn):
        self.fn = fn

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.fn(z)