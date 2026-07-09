"""Reward Models for Planning with MCTS

Contract
--------
A reward function maps a batch of latent **states** to scalar rewards::

    reward_fn(z) -> r            z: (..., N_lat, D_lat)  ->  r: (...)

i.e. it reduces the two trailing latent dims and preserves all leading (batch /
time) dims. The planner calls it on imagined horizon states ``(B, H, N_lat,
D_lat)`` and gets per-frame rewards ``(B, H)``.

Two families here (for now):

* **Latent** rewards (``GoalLatentReward``) — cheap, but latent L2 to a goal is a
  weak, uninformative signal (using DINO encoder would certainly be better).
* **Pixel / task** rewards (``TCenterReward``) — decode the latent to an image
  and score the *task* directly. For the real-robot pushT scene this segments
  the red **T** and rewards it being **centered** (orientation is intentionally
  ignored). Much more informative than latent L2; the price is a tokenizer
  ``decode`` per evaluated state.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
import cv2


# ===========================================================================
# Template Reward Model
# ===========================================================================

@runtime_checkable
class RewardModel(Protocol):
    """The contract a reward must satisfy to plug into the planner.

    A reward maps a batch of latent **states** to scalar rewards: it reduces the
    two trailing latent dims and preserves every leading (batch / time) dim::

        reward(z) -> r      z: (..., N_lat, D_lat)  ->  r: (...)

    Requirements on the output ``r``:
      * a **float** ``torch.Tensor`` of shape exactly ``z.shape[:-2]``;
      * on ``z.device`` (the search builds a discount tensor on ``r.device``
        and multiplies element-wise).

    The planner evaluates it on ``(B, 1, N, D)`` terminal states (expansion) and
    on ``(M, H, N, D)`` rollouts (simulation), so it must accept arbitrary
    leading dims. Reward *scale* is free — the search only compares and
    accumulates rewards — but keeping it roughly O(1) per frame keeps the UCB
    exploration constant ``c_ucb`` meaningful.

    This is a structural ``Protocol``: any object with a matching ``__call__``
    plugs in (a plain function via :class:`CallableReward`, an ``nn.Module``, a
    decode-then-score object like :class:`TCenterReward`, ...). It exists to
    document the contract and to allow ``isinstance(obj, RewardModel)`` checks.
    """

    def __call__(self, z: torch.Tensor) -> torch.Tensor: ...


# ===========================================================================
# Callable Reward
# ===========================================================================

RewardFn = Callable[[torch.Tensor], torch.Tensor]

class CallableReward:
    """Adapt a plain ``z -> r`` callable (e.g. a decoded-pixel scorer) to the
    reward contract, so arbitrary user functions drop into the planner."""

    def __init__(self, fn: RewardFn):
        self.fn = fn

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.fn(z)


# ===========================================================================
# Zero Reward
# ===========================================================================

class ZeroReward:
    """Reward is the null function. 
    Use for pure world-model exploration / debugging the search."""

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return z.new_zeros(z.shape[:-2])


# ===========================================================================
# Goal Latent Reward
# ===========================================================================

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


# ===========================================================================
# pushT pixel reward - red T "centered"
# ===========================================================================

def _red_mask(hsv, s_min, v_min, hue_lo, hue_hi):
    """Binary mask of saturated-red pixels (two hue bands around 0 / 180)."""
    lo1 = np.array([0, s_min, v_min], np.uint8);  hi1 = np.array([hue_lo, 255, 255], np.uint8)
    lo2 = np.array([hue_hi, s_min, v_min], np.uint8); hi2 = np.array([180, 255, 255], np.uint8)
    return cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

def score_loc(
        loc_xy: Tuple[float, float],  # normalized image coords
        *,
        center_xy: Tuple[float, float] = (0.5, 0.5),   # target, normalized image coords
        sigma: float = 0.25
) -> float:
    """Score how well a normalized image location is *centered* in the image.

    Returns a score in [0, 1], where 1 is at the target and 0 is far away.
    """
    d = float(np.hypot(loc_xy[0] - center_xy[0], loc_xy[1] - center_xy[1]))
    return float(np.exp(-0.5 * (d / sigma) ** 2))

def score_t_centered(
    rgb: np.ndarray,                       # (H, W, 3) uint8 RGB
    *,
    center_xy: Tuple[float, float] = (0.5, 0.5),   # target, normalized image coords
    sigma: float = 0.25,                   # center-Gaussian width (normalized)
    s_min: int = 90, v_min: int = 60,      # red HSV gates
    hue_lo: int = 12, hue_hi: int = 168,
    min_area_frac: float = 0.0015,         # reject specks / "T vanished"
    floor: float = 0.0,
    open_ksize: int = 3,
) -> Tuple[float, dict]:
    """Score how well the red **T** is *centered* in one frame.

    Segment the red T (largest red connected component), then score purely by how
    close its centroid is to ``center_xy``::

        score = exp(-(1/2) (d/sigma)²),   d = normalized distance from centroid to target

    (1 at the target, 0 far away). Orientation / uprightness is intentionally
    ignored — we only care that the T is centered.

    Returns ``(score, debug)``; ``score`` in ``[floor, 1]``, or ``floor`` if no T
    is found. ``debug`` holds the mask / centroid for :func:`annotate_t`.
    """
    if cv2 is None:
        raise ImportError("score_t_centered needs opencv-python (cv2).")
    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = _red_mask(hsv, s_min, v_min, hue_lo, hue_hi)
    if open_ksize:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((open_ksize, open_ksize), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    dbg = dict(found=False, score=floor, center=0.0, area=0)
    if n <= 1:
        return floor, dbg
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < min_area_frac * H * W:
        return floor, {**dbg, "area": area}

    cx, cy = float(cent[i][0]), float(cent[i][1])
    # d = float(np.hypot(cx / W - center_xy[0], cy / H - center_xy[1]))
    # center = float(np.exp(-0.5 * (d / sigma) ** 2))
    center = score_loc((cx / W, cy / H), center_xy=center_xy, sigma=sigma)

    comp = (lab == i).astype(np.uint8)
    score = center
    dbg = dict(found=True, score=float(score), center=center,
               area=area, centroid=(cx, cy), mask=comp)
    return float(score), dbg

def annotate_t(rgb: np.ndarray, debug: dict) -> np.ndarray:
    """Overlay the reward's view onto ``rgb`` (RGB uint8): image-center cross,
    T contour, centroid, and the scalar score. For eyeballing what the reward
    sees. Returns a new RGB uint8 image."""
    if cv2 is None:
        raise ImportError("annotate_t needs opencv-python (cv2).")
    out = np.ascontiguousarray(rgb.copy())
    H, W = out.shape[:2]
    cv2.drawMarker(out, (W // 2, H // 2), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
    if debug.get("found"):
        cx, cy = debug["centroid"]
        cnts, _ = cv2.findContours(debug["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 0), 1)
        cv2.circle(out, (int(cx), int(cy)), 3, (255, 255, 255), -1)
    cv2.putText(out, f"{debug.get('score', 0.0):.2f}", (4, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class TCenterReward:
    """Reward the red pushT **T** being *centered* (pixel space).

    Plugs into the planner's reward contract: ``__call__(z) -> r`` over latent
    states ``(..., N_lat, D_lat)``. It **decodes** each latent to an image and
    scores it with :func:`score_t_centered`. Pass either a ``decode_fn``
    (``lat (M,1,N,D) -> img (M,1,3,H,W) in [0,1]``) or a ``tokenizer`` (its
    ``decode`` is used under bf16 autocast). Extra keyword args are forwarded to
    :func:`score_t_centered` (``center_xy``, ``sigma``, ``s_min``, ``v_min``,
    ``min_area_frac``, ...).

    Note: one tokenizer ``decode`` per evaluated state — the dominant planning
    cost. Keep the tree modest, or evaluate fewer frames.
    """

    def __init__(self, decode_fn=None, tokenizer=None,
                 dtype: Optional[torch.dtype] = torch.bfloat16,
                 max_batch: int = 64, **score_kwargs):
        if cv2 is None:
            raise ImportError("TCenterReward needs opencv-python (cv2).")
        if decode_fn is None and tokenizer is None:
            raise ValueError("provide decode_fn or tokenizer")
        self._decode_fn = decode_fn
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.max_batch = int(max_batch)
        self.score_kwargs = score_kwargs

    def _decode(self, lat: torch.Tensor) -> torch.Tensor:      # (M,1,N,D) -> (M,1,3,H,W)
        if self._decode_fn is not None:
            return self._decode_fn(lat)
        dev = next(self.tokenizer.parameters()).device
        ctx = (torch.autocast(device_type="cuda", dtype=self.dtype)
               if (dev.type == "cuda" and self.dtype) else nullcontext())
        with torch.no_grad(), ctx:
            return self.tokenizer.decode(lat.to(dev)).float().clamp(0, 1)

    def _rgb_batch(self, lat: torch.Tensor) -> np.ndarray:     # (m,1,N,D) -> (m,H,W,3) uint8
        imgs = self._decode(lat)[:, 0]                          # (m,3,H,W)
        return (imgs.permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        Nl, Dl = z.shape[-2], z.shape[-1]
        flat = z.reshape(-1, Nl, Dl)
        scores = []
        for s in range(0, flat.shape[0], self.max_batch):
            rgb = self._rgb_batch(flat[s:s + self.max_batch][:, None])
            scores.extend(score_t_centered(rgb[k], **self.score_kwargs)[0]
                          for k in range(rgb.shape[0]))
        return torch.tensor(scores, device=z.device, dtype=torch.float32).reshape(z.shape[:-2])

    # convenience for notebooks: score + debug straight from a latent frame
    def score_latent(self, z_1frame: torch.Tensor) -> Tuple[float, dict, np.ndarray]:
        """z (N,D) or (1,1,N,D) -> (score, debug, rgb_uint8)."""
        z = z_1frame.reshape(1, z_1frame.shape[-2], z_1frame.shape[-1])
        rgb = self._rgb_batch(z[:, None])[0]
        score, dbg = score_t_centered(rgb, **self.score_kwargs)
        return score, dbg, rgb