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

Two families here:

* **Latent** rewards (``GoalLatentReward``) — cheap, but latent L2 to a goal is a
  weak, uninformative signal (planning barely beats random with it).
* **Pixel / task** rewards (``TCenterReward``) — decode the latent to an image
  and score the *task* directly. For the real-robot pushT scene this segments
  the red **T** and rewards it being **centered and straight**. Much more
  informative; the price is a tokenizer ``decode`` per evaluated state.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional, Tuple

import numpy as np
import torch

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

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


# ===========================================================================
# pushT pixel reward — red T "centered and straight"
# ===========================================================================

def _red_mask(hsv, s_min, v_min, hue_lo, hue_hi):
    """Binary mask of saturated-red pixels (two hue bands around 0 / 180)."""
    lo1 = np.array([0, s_min, v_min], np.uint8);  hi1 = np.array([hue_lo, 255, 255], np.uint8)
    lo2 = np.array([hue_hi, s_min, v_min], np.uint8); hi2 = np.array([180, 255, 255], np.uint8)
    return cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)


def _lr_symmetry(comp: np.ndarray) -> float:
    """IoU of the mask with its reflection about its own vertical centroid line.

    A robust, shape-aware "is it vertically aligned?" measure. A T (or any shape)
    that is upright is left-right symmetric → IoU near 1; tilting breaks the
    symmetry → IoU drops. Unlike principal-axis / moment orientation, this does
    NOT rely on the shape being anisotropic — crucial for a T, whose second-order
    axes are near-degenerate (stem length ≈ crossbar width), so its moment angle
    is unstable and useless for orientation."""
    ys, xs = np.nonzero(comp)
    if len(xs) == 0:
        return 0.0
    cx = int(round(xs.mean())); W = comp.shape[1]
    nx = 2 * cx - xs
    ok = (nx >= 0) & (nx < W)
    refl = np.zeros_like(comp)
    refl[ys[ok], nx[ok]] = 1
    inter = int((comp & refl).sum())
    union = int((comp | refl).sum())
    return inter / max(union, 1)


def score_t_centered(
    rgb: np.ndarray,                       # (H, W, 3) uint8 RGB
    *,
    center_xy: Tuple[float, float] = (0.5, 0.5),   # target, normalized image coords
    sigma: float = 0.25,                   # center-Gaussian width (normalized)
    w_center: float = 0.6,
    w_orient: float = 0.4,
    s_min: int = 90, v_min: int = 60,      # red HSV gates
    hue_lo: int = 12, hue_hi: int = 168,
    min_area_frac: float = 0.0015,         # reject specks / "T vanished"
    orient_method: str = "vertical",       # "vertical" (robust) | "axis" | "match"
    target_angle_deg: Optional[float] = 90,  # for "match": canonical angle
    sym_lo: float = 0.35, sym_hi: float = 0.72,  # "vertical" LR-sym -> [0,1] gates
    floor: float = 0.0,
    open_ksize: int = 3,
) -> Tuple[float, dict]:
    """Score how well the red **T** is *centered and straight* in one frame.

    Segment the red T (largest red connected component), then combine:

    * **center** = ``exp(-½ (d/σ)²)`` with ``d`` the normalized distance from the
      T centroid to ``center_xy`` (1 at the target, →0 far away);
    * **orient** — how "straight" the T is, via ``orient_method``:
        - ``"vertical"`` (**default; recommended for a T**): left-right
          reflection symmetry (:func:`_lr_symmetry`) gated ``[sym_lo,sym_hi]→
          [0,1]``. Robust — a T's moment axis is near-degenerate and unstable,
          but its mirror symmetry is a reliable "upright?" signal.
        - ``"axis"``: ``|cos 2θ|`` from image moments (1 = axis-aligned, 0 at
          45°). **Unreliable for a near-isotropic T**; fine for elongated shapes.
        - ``"match"``: ``½(1+cos 2(θ-θ*))`` toward ``target_angle_deg``.

    Returns ``(score, debug)``; ``score = w_center·center + w_orient·orient`` in
    ``[floor, w_center+w_orient]``, or ``floor`` if no T is found. ``debug`` holds
    the mask / centroid / angle / lr_sym for :func:`annotate_t`.
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
    dbg = dict(found=False, score=floor, center=0.0, orient=0.0, area=0)
    if n <= 1:
        return floor, dbg
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < min_area_frac * H * W:
        return floor, {**dbg, "area": area}

    cx, cy = float(cent[i][0]), float(cent[i][1])
    d = float(np.hypot(cx / W - center_xy[0], cy / H - center_xy[1]))
    center = float(np.exp(-0.5 * (d / sigma) ** 2))

    comp = (lab == i).astype(np.uint8)
    mu = cv2.moments(comp, binaryImage=True)
    theta = 0.5 * np.arctan2(2 * mu["mu11"], (mu["mu20"] - mu["mu02"]) + 1e-9)
    lr = _lr_symmetry(comp)
    if orient_method == "vertical":
        orient = float(np.clip((lr - sym_lo) / (sym_hi - sym_lo + 1e-9), 0.0, 1.0))
    elif orient_method == "match" and target_angle_deg is not None:
        ta = np.radians(target_angle_deg)
        orient = float(0.5 * (1 + np.cos(2 * (theta - ta))))
    else:  # "axis"
        orient = float(abs(np.cos(2 * theta)))

    score = w_center * center + w_orient * orient
    dbg = dict(found=True, score=float(score), center=center, orient=orient,
               area=area, centroid=(cx, cy), angle_deg=float(np.degrees(theta)),
               lr_sym=float(lr), mask=comp)
    return float(score), dbg


def annotate_t(rgb: np.ndarray, debug: dict) -> np.ndarray:
    """Overlay the reward's view onto ``rgb`` (RGB uint8): image-center cross,
    T contour, centroid, principal axis, and the scalar score. For eyeballing
    what the reward sees. Returns a new RGB uint8 image."""
    if cv2 is None:
        raise ImportError("annotate_t needs opencv-python (cv2).")
    out = np.ascontiguousarray(rgb.copy())
    H, W = out.shape[:2]
    cv2.drawMarker(out, (W // 2, H // 2), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
    if debug.get("found"):
        cx, cy = debug["centroid"]
        th = np.radians(debug["angle_deg"])
        cnts, _ = cv2.findContours(debug["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 0), 1)
        L = 0.28 * max(H, W)
        p1 = (int(cx - L * np.cos(th)), int(cy - L * np.sin(th)))
        p2 = (int(cx + L * np.cos(th)), int(cy + L * np.sin(th)))
        cv2.line(out, p1, p2, (255, 0, 255), 1)
        cv2.circle(out, (int(cx), int(cy)), 3, (255, 255, 255), -1)
    cv2.putText(out, f"{debug.get('score', 0.0):.2f}", (4, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class TCenterReward:
    """Reward the red pushT **T** being *centered and straight* (pixel space).

    Plugs into the planner's reward contract: ``__call__(z) -> r`` over latent
    states ``(..., N_lat, D_lat)``. It **decodes** each latent to an image and
    scores it with :func:`score_t_centered`. Pass either a ``decode_fn``
    (``lat (M,1,N,D) -> img (M,1,3,H,W) in [0,1]``) or a ``tokenizer`` (its
    ``decode`` is used under bf16 autocast). Extra keyword args are forwarded to
    :func:`score_t_centered` (``center_xy``, ``sigma``, ``w_center``,
    ``w_orient``, ``target_angle_deg``, ...).

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