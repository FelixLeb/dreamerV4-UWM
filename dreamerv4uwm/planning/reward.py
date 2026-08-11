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
from typing import Callable, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as Fnn

try:
    import cv2
except Exception:  # pragma: no cover - only the pixel/task rewards need it
    cv2 = None


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


def _t_axis_theta(mask: np.ndarray) -> float:
    """Principal-axis orientation of a binary mask, from central second moments.

    Returns radians in ``(-pi/2, pi/2]``. This is an **axis**, hence 180-degree
    ambiguous (an upright and an upside-down T give the same value) — matching
    ``TPoseDescriptor._theta`` and the fact that we score the T's axis, not which way
    its stem points.
    """
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    return 0.5 * float(np.arctan2(2.0 * m["mu11"], (m["mu20"] - m["mu02"]) + 1e-9))


def _axis_alignment(theta: float, target: float, sigma_theta: float) -> float:
    """Gaussian on the smallest angle between axis ``theta`` and ``target`` (both
    radians), measured **mod pi** so ``+90`` and ``-90`` count as the same axis."""
    d = (theta - target + np.pi / 2.0) % np.pi - np.pi / 2.0    # -> (-pi/2, pi/2]
    return float(np.exp(-0.5 * (d / sigma_theta) ** 2))


def score_t_centered_straight(
    rgb: np.ndarray,                       # (H, W, 3) uint8 RGB
    *,
    center_xy: Tuple[float, float] = (0.5, 0.5),   # target centroid, normalized
    sigma: float = 0.25,                   # center-Gaussian width (normalized)
    target_theta_deg: float = 90.0,        # upright T axis; +/-90 == vertical
    sigma_theta_deg: float = 20.0,         # orientation tolerance (std, degrees)
    w_center: float = 0.5,                 # weights for combine="weighted_sum"
    w_orient: float = 0.5,
    combine: str = "product",              # "product" (straight AND centered) | "weighted_sum"
    floor: float = 0.0,
    **seg_kwargs,                          # forwarded to score_t_centered (s_min, v_min, min_area_frac, ...)
) -> Tuple[float, dict]:
    """Score how well the red **T** is BOTH *centered* and *straight* (axis-aligned).

    Two sub-scores in ``[0, 1]``::

        center = exp(-1/2 (d / sigma)^2)                 # centroid distance to center_xy
        orient = exp(-1/2 (dtheta / sigma_theta)^2)      # axis distance to target_theta_deg

    ``center`` is exactly :func:`score_t_centered`; ``orient`` peaks when the T's
    principal axis matches ``target_theta_deg`` (measured mod 180 deg, so +90 and -90
    are the same upright axis — an upside-down T scores the SAME as an upright one; use
    :func:`score_t_centered_angle` if you need to tell them apart). They combine
    multiplicatively by default (reward high only when *both* hold);
    ``combine="weighted_sum"`` uses ``w_center``/``w_orient`` (auto-normalised) instead.

    Returns ``(score, debug)``; ``score`` in ``[floor, 1]``, or ``floor`` if no T is
    found. ``debug`` adds ``theta`` (rad), ``theta_deg``, and ``orient`` to what
    :func:`score_t_centered` returns, so :func:`annotate_t` still works.

    **Convention note.** ``target_theta_deg=90`` assumes the upright T's principal axis
    is vertical. If your decoded upright T reads a different angle, decode one
    known-good frame, print ``debug['theta_deg']``, and set ``target_theta_deg`` to it.
    """
    center, dbg = score_t_centered(rgb, center_xy=center_xy, sigma=sigma,
                                   floor=floor, **seg_kwargs)
    if not dbg.get("found"):
        return floor, {**dbg, "orient": 0.0, "theta": float("nan"),
                       "theta_deg": float("nan")}
    theta = _t_axis_theta(dbg["mask"])
    orient = _axis_alignment(theta, np.deg2rad(target_theta_deg), np.deg2rad(sigma_theta_deg))
    if combine == "weighted_sum":
        wsum = w_center + w_orient
        score = (w_center * center + w_orient * orient) / (wsum if wsum > 0 else 1.0)
    else:  # "product": straight AND centered
        score = center * orient
    score = max(float(floor), float(score))
    dbg = {**dbg, "center": float(center), "orient": float(orient),
           "theta": float(theta), "theta_deg": float(np.degrees(theta)),
           "score": score}
    return score, dbg


def _t_heading(mask: np.ndarray) -> float:
    """Full heading of a T-shaped mask in ``(-pi, pi]`` (radians), resolving the
    180-degree axis ambiguity that :func:`_t_axis_theta` cannot.

    The vector points **toward the crossbar** — the flat top of an upright T — using the
    T's defining asymmetry: split the shape at its centroid along the principal (stem)
    axis; the crossbar half is *wider* (larger spread perpendicular to the stem) than the
    stem half, so heading points to whichever half spreads more. Independent of the
    (arbitrary) eigenvector sign. Convention is image coordinates (y **down**), so an
    upright T (crossbar up) reads ``~ -90 deg``, an upside-down one ``~ +90 deg``.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return float("nan")
    x = xs.astype(np.float64) - xs.mean()
    y = ys.astype(np.float64) - ys.mean()
    _, evecs = np.linalg.eigh(np.cov(np.stack([x, y])))       # eigenvectors, ascending eigenvalues
    v = evecs[:, -1]                                          # major (stem) axis, unit
    perp = np.array([-v[1], v[0]])                            # crossbar-width axis
    p = x * v[0] + y * v[1]                                   # coord along the stem axis
    q = x * perp[0] + y * perp[1]                             # coord across it
    hi, lo = p > 0, p <= 0
    spread_hi = float(q[hi].std()) if hi.sum() > 1 else 0.0
    spread_lo = float(q[lo].std()) if lo.sum() > 1 else 0.0
    h = v if spread_hi >= spread_lo else -v                   # toward the wider (crossbar) half
    return float(np.arctan2(h[1], h[0]))


def _heading_alignment(phi: float, target: float, sigma: float) -> float:
    """Gaussian on the full-circle angle between heading ``phi`` and ``target`` (both
    radians, wrapped **mod 2*pi**), so an upside-down T (180 deg off) scores ~0."""
    d = (phi - target + np.pi) % (2.0 * np.pi) - np.pi        # -> (-pi, pi]
    return float(np.exp(-0.5 * (d / sigma) ** 2))


def score_t_centered_angle(
    rgb: np.ndarray,                       # (H, W, 3) uint8 RGB
    *,
    center_xy: Tuple[float, float] = (0.5, 0.5),   # target centroid, normalized
    sigma: float = 0.25,                   # center-Gaussian width (normalized)
    target_heading_deg: float = -90.0,     # upright T: crossbar points 'up' (image y-down)
    sigma_heading_deg: float = 25.0,       # heading tolerance (std, degrees)
    w_center: float = 0.5,                 # weights for combine="weighted_sum"
    w_orient: float = 0.5,
    combine: str = "product",              # "product" (upright AND centered) | "weighted_sum"
    floor: float = 0.0,
    **seg_kwargs,                          # forwarded to score_t_centered (s_min, v_min, min_area_frac, ...)
) -> Tuple[float, dict]:
    """Score how well the red **T** is *centered* and *upright* — like
    :func:`score_t_centered_straight`, but the orientation term uses the **full heading**
    (mod 360 deg), so an upside-down T is penalised instead of scoring the same as an
    upright one.

    Two sub-scores in ``[0, 1]``::

        center = exp(-1/2 (d / sigma)^2)                     # centroid distance to center_xy
        orient = exp(-1/2 (dphi / sigma_heading)^2)          # heading distance to target_heading_deg

    ``orient`` peaks when the T's heading (direction of its crossbar, from
    :func:`_t_heading`) matches ``target_heading_deg`` and falls to ~0 when the T is
    flipped 180 deg. They combine multiplicatively by default (reward high only when the T
    is *both* upright and centred); ``combine="weighted_sum"`` uses ``w_center``/
    ``w_orient`` (auto-normalised) instead.

    Returns ``(score, debug)``; ``score`` in ``[floor, 1]``, or ``floor`` if no T is
    found. ``debug`` adds ``heading`` (rad), ``heading_deg``, and ``orient``, so
    :func:`annotate_t` still works.

    **Convention note.** ``target_heading_deg=-90`` assumes an upright T's crossbar points
    up in image coordinates (y down). If your decoded upright T reads a different heading,
    decode one known-good frame, print ``debug['heading_deg']``, and set
    ``target_heading_deg`` to it.
    """
    center, dbg = score_t_centered(rgb, center_xy=center_xy, sigma=sigma,
                                   floor=floor, **seg_kwargs)
    if not dbg.get("found"):
        return floor, {**dbg, "orient": 0.0, "heading": float("nan"),
                       "heading_deg": float("nan")}
    phi = _t_heading(dbg["mask"])
    orient = _heading_alignment(phi, np.deg2rad(target_heading_deg), np.deg2rad(sigma_heading_deg))
    if combine == "weighted_sum":
        wsum = w_center + w_orient
        score = (w_center * center + w_orient * orient) / (wsum if wsum > 0 else 1.0)
    else:  # "product": upright AND centered
        score = center * orient
    score = max(float(floor), float(score))
    dbg = {**dbg, "center": float(center), "orient": float(orient),
           "heading": float(phi), "heading_deg": float(np.degrees(phi)),
           "score": score}
    return score, dbg


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


class TCenterStraightReward(TCenterReward):
    """Reward the red pushT **T** being BOTH *centered* and *straight* (axis-aligned).

    Extends :class:`TCenterReward` — identical decode plumbing — but scores each decoded
    frame with :func:`score_t_centered_straight`, which multiplies the centering score by
    an orientation score that peaks when the T's principal axis matches
    ``target_theta_deg`` (default 90 deg = vertical). Orientation is axis-only, so an
    upright and an upside-down T score the **same**; use :class:`TCenterAngleReward` when
    you need to reward upright specifically.

    The constructor is inherited: pass a ``decode_fn`` or ``tokenizer`` plus any of
    :func:`score_t_centered_straight`'s keyword args (``center_xy``, ``sigma``,
    ``target_theta_deg``, ``sigma_theta_deg``, ``w_center``, ``w_orient``, ``combine``,
    and the segmentation gates). One tokenizer ``decode`` per evaluated state, same cost
    as :class:`TCenterReward`.
    """

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        Nl, Dl = z.shape[-2], z.shape[-1]
        flat = z.reshape(-1, Nl, Dl)
        scores = []
        for s in range(0, flat.shape[0], self.max_batch):
            rgb = self._rgb_batch(flat[s:s + self.max_batch][:, None])
            scores.extend(score_t_centered_straight(rgb[k], **self.score_kwargs)[0]
                          for k in range(rgb.shape[0]))
        return torch.tensor(scores, device=z.device, dtype=torch.float32).reshape(z.shape[:-2])

    # convenience for notebooks: score + debug straight from a latent frame
    def score_latent(self, z_1frame: torch.Tensor) -> Tuple[float, dict, np.ndarray]:
        """z (N,D) or (1,1,N,D) -> (score, debug, rgb_uint8)."""
        z = z_1frame.reshape(1, z_1frame.shape[-2], z_1frame.shape[-1])
        rgb = self._rgb_batch(z[:, None])[0]
        score, dbg = score_t_centered_straight(rgb, **self.score_kwargs)
        return score, dbg, rgb


class TCenterAngleReward(TCenterReward):
    """Reward the red pushT **T** being *centered* and **upright** (not upside-down).

    Like :class:`TCenterStraightReward`, but the orientation term uses the T's **full
    heading** (:func:`score_t_centered_angle`), so a flipped (upside-down) T is penalised
    rather than scoring the same as an upright one. Reward is maximal only when the T is
    both centred and pointing the upright way (crossbar toward ``target_heading_deg``,
    default -90 deg = up in image coordinates).

    The constructor is inherited: pass a ``decode_fn`` or ``tokenizer`` plus any of
    :func:`score_t_centered_angle`'s keyword args (``center_xy``, ``sigma``,
    ``target_heading_deg``, ``sigma_heading_deg``, ``w_center``, ``w_orient``,
    ``combine``, and the segmentation gates). One tokenizer ``decode`` per evaluated
    state, same cost as :class:`TCenterReward`.
    """

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        Nl, Dl = z.shape[-2], z.shape[-1]
        flat = z.reshape(-1, Nl, Dl)
        scores = []
        for s in range(0, flat.shape[0], self.max_batch):
            rgb = self._rgb_batch(flat[s:s + self.max_batch][:, None])
            scores.extend(score_t_centered_angle(rgb[k], **self.score_kwargs)[0]
                          for k in range(rgb.shape[0]))
        return torch.tensor(scores, device=z.device, dtype=torch.float32).reshape(z.shape[:-2])

    # convenience for notebooks: score + debug straight from a latent frame
    def score_latent(self, z_1frame: torch.Tensor) -> Tuple[float, dict, np.ndarray]:
        """z (N,D) or (1,1,N,D) -> (score, debug, rgb_uint8)."""
        z = z_1frame.reshape(1, z_1frame.shape[-2], z_1frame.shape[-1])
        rgb = self._rgb_batch(z[:, None])[0]
        score, dbg = score_t_centered_angle(rgb, **self.score_kwargs)
        return score, dbg, rgb

# ===========================================================================
# task-agnostic: distance to a goal in a frozen DINO embedding space
# ===========================================================================

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dino(name: str = "dinov2_vits14", device=None, repo: str = "facebookresearch/dinov2"):
    """Load a frozen DINOv2 backbone from ``torch.hub`` (eval, no grads).

    ``dinov2_vit{s,b,l,g}14`` — ``s`` (384-d, ~21M params) is plenty here and the cheapest.
    Needs network access on first call (then cached in ``~/.cache/torch/hub``); on an offline
    cluster, load the model yourself and pass the module to :class:`DINOGoalReward` directly.
    """
    model = torch.hub.load(repo, name)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval().to(device) if device is not None else model.eval()


class DINOGoalReward:
    """Reward = closeness to a **goal image** in a frozen DINO embedding space.

    Task-agnostic counterpart of :class:`TCenterReward`: instead of a hand-designed pixel
    heuristic, score each state by how close its decoded frame is to a goal frame in the
    representation of a self-supervised encoder. Nothing here knows about PushT, so moving to
    a new environment is a change of *goal*, not of code.

    Pipeline per state: ``latent -> tokenizer.decode -> resize+ImageNet-normalise -> DINO
    -> distance to the goal embedding -> reward``.

    Parameters
    ----------
    goal_z / goal_rgb / goal_feat
        The goal, given exactly one of three ways: a clean **latent** ``(N_lat, D_lat)`` (the
        usual case — encode a real target frame), an **RGB** ``uint8 (H, W, 3)`` image, or a
        precomputed **embedding**. A latent goal is decoded and embedded once, at construction.
    decode_fn / tokenizer
        Same contract as :class:`TCenterReward` — either a ``lat (M,1,N,D) -> img (M,1,3,H,W)``
        callable or a tokenizer whose ``decode`` is used under autocast.
    model
        A hub name (loaded via :func:`load_dino`) or an already-constructed ``nn.Module``.
        Pass the module on offline machines.
    feature : {'patch', 'cls', 'patch_mean'}
        Which DINO output to compare. **``patch`` (default)** keeps the full patch grid, so the
        embedding is *spatially* sensitive — moving the object changes it, which is what a
        planning objective needs. ``cls`` is the compact global token (semantic, but trained to
        be crop-invariant, so it can be nearly blind to *where* the object is). ``patch_mean``
        pools the patches and likewise discards layout.
    metric : {'cosine', 'l2'}
        ``cosine`` = ``1 - cos`` on the flattened feature (in ``[0, 2]``, well-behaved in high
        dimension). ``l2`` = Euclidean distance divided by ``sqrt(dim)`` (an RMS per-coordinate
        distance, so the scale does not drift with ``feature`` or backbone width).
    mode : {'gauss', 'neg'}
        How distance becomes reward. ``gauss`` -> ``exp(-1/2 (d/sigma)^2)`` in ``[0, 1]``,
        matching the scale of the T rewards so ``c_ucb`` keeps its meaning — **recommended**.
        ``neg`` -> ``-d / scale``, unbounded below.

    Notes
    -----
    * ``sigma`` must be calibrated for the chosen ``feature``/``metric``: use
      :meth:`distances` on a batch of real latents and set ``sigma`` near the median (see
      :meth:`suggest_sigma`). A badly-scaled ``sigma`` gives a flat reward — precisely the
      ``val_std -> 0`` pathology the diagnostics look for.
    * Cost is one decode **plus** one DINO forward per evaluated state. Unlike the OpenCV
      T-segmentation this stays on the GPU, so despite the extra network it is usually not
      slower end-to-end.
    """

    def __init__(self, *, goal_z: Optional[torch.Tensor] = None,
                 goal_rgb: Optional[np.ndarray] = None,
                 goal_feat: Optional[torch.Tensor] = None,
                 decode_fn=None, tokenizer=None,
                 model: Union[str, torch.nn.Module] = "dinov2_vits14",
                 feature: str = "patch", metric: str = "cosine", mode: str = "gauss",
                 sigma: float = 0.25, scale: float = 1.0, resize: int = 224,
                 dtype: Optional[torch.dtype] = torch.bfloat16,
                 max_batch: int = 32, device=None):
        if decode_fn is None and tokenizer is None:
            raise ValueError("provide decode_fn or tokenizer")
        if sum(g is not None for g in (goal_z, goal_rgb, goal_feat)) != 1:
            raise ValueError("provide exactly one of goal_z / goal_rgb / goal_feat")
        if feature not in ("patch", "cls", "patch_mean"):
            raise ValueError(f"unknown feature={feature!r} (expected patch|cls|patch_mean)")
        if metric not in ("cosine", "l2"):
            raise ValueError(f"unknown metric={metric!r} (expected cosine|l2)")
        if mode not in ("gauss", "neg"):
            raise ValueError(f"unknown mode={mode!r} (expected gauss|neg)")

        self._decode_fn, self.tokenizer = decode_fn, tokenizer
        self.feature, self.metric, self.mode = feature, metric, mode
        self.sigma, self.scale = float(sigma), float(scale)
        self.resize, self.dtype, self.max_batch = int(resize), dtype, int(max_batch)

        if device is None:
            device = (next(tokenizer.parameters()).device if tokenizer is not None
                      else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(device)
        # DINOv2 patches are 14x14: the input side must be a multiple of 14 (224 = 16 patches).
        if self.resize % 14:
            raise ValueError(f"resize={self.resize} must be a multiple of 14 for DINOv2")
        self.model = (load_dino(model, device=self.device) if isinstance(model, str)
                      else model.eval().to(self.device))

        m = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        s = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = m, s

        if goal_feat is not None:
            self.goal_feat = goal_feat.to(self.device).reshape(1, -1).float()
        elif goal_rgb is not None:
            img = torch.from_numpy(np.ascontiguousarray(goal_rgb)).permute(2, 0, 1)[None]
            self.goal_feat = self._embed(img.float().to(self.device) / 255.0)
        else:
            g = goal_z.detach()
            while g.dim() > 2:                       # (1,1,N,D) / (1,N,D) -> (N,D)
                g = g.squeeze(0)
            self.goal_feat = self._embed(self._decode(g[None, None]))
        self.dim = int(self.goal_feat.shape[-1])

    # ---- plumbing ---------------------------------------------------------
    def _decode(self, lat: torch.Tensor) -> torch.Tensor:      # (m,1,N,D) -> (m,3,H,W) in [0,1]
        if self._decode_fn is not None:
            return self._decode_fn(lat)[:, 0]
        ctx = (torch.autocast(device_type="cuda", dtype=self.dtype)
               if (self.device.type == "cuda" and self.dtype) else nullcontext())
        with torch.no_grad(), ctx:
            return self.tokenizer.decode(lat.to(self.device)).float().clamp(0, 1)[:, 0]

    @torch.no_grad()
    def _embed(self, imgs: torch.Tensor) -> torch.Tensor:      # (m,3,H,W) in [0,1] -> (m, dim)
        x = imgs.to(self.device).float().clamp(0, 1)
        if x.shape[-1] != self.resize or x.shape[-2] != self.resize:
            x = Fnn.interpolate(x, (self.resize, self.resize), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        ctx = (torch.autocast(device_type="cuda", dtype=self.dtype)
               if (self.device.type == "cuda" and self.dtype) else nullcontext())
        with ctx:
            out = (self.model.forward_features(x) if hasattr(self.model, "forward_features")
                   else self.model(x))
        if isinstance(out, dict):                              # DINOv2 forward_features
            f = (out["x_norm_clstoken"] if self.feature == "cls"
                 else out["x_norm_patchtokens"])               # (m, n_patches, d)
        else:
            f = out
        f = f.float()
        if self.feature == "patch_mean" and f.dim() == 3:
            f = f.mean(1)
        return f.reshape(f.shape[0], -1)                       # flatten the patch grid if present

    def _dist(self, feat: torch.Tensor) -> torch.Tensor:       # (m, dim) -> (m,)
        g = self.goal_feat.expand_as(feat)
        if self.metric == "cosine":
            return 1.0 - Fnn.cosine_similarity(feat, g, dim=-1)          # in [0, 2]
        return (feat - g).norm(dim=-1) / (feat.shape[-1] ** 0.5)         # RMS per-coordinate

    # ---- the reward contract ----------------------------------------------
    @torch.no_grad()
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        d = self.distances(z)
        r = (torch.exp(-0.5 * (d / self.sigma) ** 2) if self.mode == "gauss"
             else -d / self.scale)
        return r.to(device=z.device, dtype=torch.float32).reshape(z.shape[:-2])

    @torch.no_grad()
    def distances(self, z: torch.Tensor) -> torch.Tensor:
        """Raw goal distances for states ``z (..., N_lat, D_lat)`` -> ``(prod(...),)``.
        Use this to calibrate ``sigma`` (see :meth:`suggest_sigma`)."""
        flat = z.reshape(-1, z.shape[-2], z.shape[-1])
        out = [self._dist(self._embed(self._decode(flat[s:s + self.max_batch][:, None])))
               for s in range(0, flat.shape[0], self.max_batch)]
        return torch.cat(out)

    @torch.no_grad()
    def suggest_sigma(self, z: torch.Tensor, quantile: float = 0.5) -> float:
        """A starting ``sigma``: the given quantile of goal distances over a batch of *real*
        latents. At ``sigma = median`` the reward spans roughly ``[0.6, 1]`` over the states
        you actually visit — enough spread for the values to separate."""
        return float(self.distances(z).quantile(quantile).item())

    # convenience for notebooks, mirroring the T rewards
    @torch.no_grad()
    def score_latent(self, z_1frame: torch.Tensor) -> Tuple[float, dict, np.ndarray]:
        """z (N,D) or (1,1,N,D) -> (score, debug, rgb_uint8)."""
        z = z_1frame.reshape(1, z_1frame.shape[-2], z_1frame.shape[-1])
        img = self._decode(z[:, None])
        d = float(self._dist(self._embed(img))[0].item())
        score = float(np.exp(-0.5 * (d / self.sigma) ** 2)) if self.mode == "gauss" else -d / self.scale
        rgb = (img[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        return score, dict(distance=d, score=score, feature=self.feature, metric=self.metric), rgb
