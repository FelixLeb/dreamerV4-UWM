"""State descriptors ``phi(state)`` for diversity metrics.

Diversity of the *raw* tokenised latent is not meaningful here (distance
concentration in ~8k dims + task-irrelevant nuisance variance). We therefore
measure diversity in a low-dimensional, task-grounded descriptor of the
**decoded** state. See ``../../mcts_planning_diagnostics.tex`` (Family D) and
``../mcts_study_plan.md`` (§2.D) for the rationale.

A ``StateDescriptor`` maps a batch of terminal latents ``(M, N_lat, D_lat)`` to
features ``(M, d)`` plus a ``found`` mask ``(M,)`` (False = descriptor undefined
for that state, e.g. no T visible). Everything downstream (``metrics.py``) is
written against this interface, so a new environment/reward is a descriptor swap.

Implementations:
* ``TPoseDescriptor``     — PushT / ``TCenterReward``: decode + segment the red T,
  return normalised ``(cx, cy[, theta, sqrt_area])`` with ``theta`` the second-moment
  *axis* (180-degree ambiguous). Reuses ``score_t_centered``.
* ``THeadingDescriptor``  — same, but orientation is the up/down-resolved **heading**
  encoded as ``(cos, sin)`` (circular, distinguishes upright from upside-down). Pair with
  ``TCenterAngleReward`` when diversity should care which way up the T is.
* ``RewardScalarDescriptor`` — env-agnostic fallback: the scalar reward itself as a
  1-D feature (weak: two different states with equal reward look identical).
* ``LatentMeanDescriptor`` — trivial, model-free; for unit-testing metrics only.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import cv2
except Exception:  # pragma: no cover - cv2 optional for non-pushT descriptors
    cv2 = None

from ...reward import score_t_centered, _t_heading, score_t_centered_angle


class StateDescriptor:
    """Protocol: ``__call__(term_lat) -> (features (M,d) float32, found (M,) bool)``."""

    dim: int = 0
    dup_eps: float = 0.05  # near-duplicate threshold in feature space (see metrics.duplicate_rate)

    def __call__(self, term_lat: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# PushT / TCenterReward : the T object pose
# ---------------------------------------------------------------------------

class TPoseDescriptor(StateDescriptor):
    """Decode each terminal latent, segment the red T, return its pose.

    ``phi = (cx, cy [, theta, sqrt_area])`` — centroid normalised to [0,1] by
    image size, orientation from the mask's second moments in [-1,1] (theta/pi),
    and ``sqrt(area)`` normalised by image side. ``found=False`` when no valid T
    is present (the reward floored); such rows are excluded from diversity.

    This is *the same* decode+segment the reward already performs, so if you run
    it on the tree's terminal latents you pay one extra batched decode over the
    ~n_nodes states (cheap). ``score_kwargs`` are forwarded to
    :func:`score_t_centered` and should match the reward's.
    """

    def __init__(self, decode_fn=None, tokenizer=None, *, use_theta: bool = True,
                 use_area: bool = True, dup_eps: float = 0.05,
                 dtype: Optional[torch.dtype] = torch.bfloat16,
                 max_batch: int = 64, **score_kwargs):
        if cv2 is None:
            raise ImportError("TPoseDescriptor needs opencv-python (cv2).")
        if decode_fn is None and tokenizer is None:
            raise ValueError("provide decode_fn or tokenizer")
        self._decode_fn = decode_fn
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.max_batch = int(max_batch)
        self.use_theta = use_theta
        self.use_area = use_area
        self.dup_eps = float(dup_eps)
        self.score_kwargs = score_kwargs
        self.dim = 2 + int(use_theta) + int(use_area)

    def _decode(self, lat: torch.Tensor) -> torch.Tensor:  # (M,1,N,D) -> (M,1,3,H,W)
        if self._decode_fn is not None:
            return self._decode_fn(lat)
        dev = next(self.tokenizer.parameters()).device
        ctx = (torch.autocast(device_type="cuda", dtype=self.dtype)
               if (dev.type == "cuda" and self.dtype) else nullcontext())
        with torch.no_grad(), ctx:
            return self.tokenizer.decode(lat.to(dev)).float().clamp(0, 1)

    def _rgb_batch(self, lat: torch.Tensor) -> np.ndarray:  # (m,1,N,D) -> (m,H,W,3) uint8
        imgs = self._decode(lat)[:, 0]
        return (imgs.permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    @staticmethod
    def _theta(mask: np.ndarray) -> float:
        """Orientation of a binary mask from central second moments, in [-pi/2, pi/2]."""
        m = cv2.moments(mask, binaryImage=True)
        mu20, mu02, mu11 = m["mu20"], m["mu02"], m["mu11"]
        return 0.5 * float(np.arctan2(2.0 * mu11, (mu20 - mu02) + 1e-9))

    @torch.no_grad()
    def __call__(self, term_lat: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        flat = term_lat.reshape(-1, term_lat.shape[-2], term_lat.shape[-1])
        feats, found = [], []
        for s in range(0, flat.shape[0], self.max_batch):
            rgb = self._rgb_batch(flat[s:s + self.max_batch][:, None])
            H, W = rgb.shape[1], rgb.shape[2]
            side = float(np.sqrt(H * W))
            for k in range(rgb.shape[0]):
                _, dbg = score_t_centered(rgb[k], **self.score_kwargs)
                if not dbg.get("found"):
                    row = [np.nan] * self.dim
                    found.append(False)
                else:
                    cx, cy = dbg["centroid"]
                    row = [cx / W, cy / H]
                    if self.use_theta:
                        row.append(self._theta(dbg["mask"]) / np.pi)
                    if self.use_area:
                        row.append(float(np.sqrt(dbg["area"])) / side)
                    found.append(True)
                feats.append(row)
        return np.asarray(feats, dtype=np.float32), np.asarray(found, dtype=bool)


class THeadingDescriptor(TPoseDescriptor):
    """Like :class:`TPoseDescriptor`, but with the T's **real orientation**.

    ``TPoseDescriptor`` uses the second-moment ``theta`` — an *axis* in
    ``[-pi/2, pi/2]``, so an upright and an upside-down T map to the **same** value and a
    180-degree flip contributes **zero** to the diversity distance. This descriptor
    instead uses the full **heading** ``phi`` from :func:`_t_heading` (the same
    up/down-resolved angle :class:`~...reward.TCenterAngleReward` scores), encoded as a
    circular pair::

        phi_feat = (cx, cy [, w*cos(phi), w*sin(phi)] [, sqrt_area])

    The ``(cos, sin)`` encoding is *circular* (no ``+pi/-pi`` wraparound artifact) and
    makes a flipped T genuinely far away (distance ``~2*w`` for a 180-degree flip).
    ``orient_weight`` (``w``, default 0.5) keeps the orientation pair on the same scale as
    a full-frame centroid move (a flip then costs ``~1``), so ``dup_eps`` stays meaningful.

    Use this when the diversity metrics should reflect *which way up* the T is (i.e. with
    ``reward.kind=angle``); use :class:`TPoseDescriptor` when only the axis matters.
    Decode/segment plumbing is inherited unchanged.
    """

    def __init__(self, decode_fn=None, tokenizer=None, *, use_theta: bool = True,
                 use_area: bool = True, dup_eps: float = 0.05, orient_weight: float = 0.5,
                 dtype: Optional[torch.dtype] = torch.bfloat16, max_batch: int = 64,
                 **score_kwargs):
        super().__init__(decode_fn=decode_fn, tokenizer=tokenizer, use_theta=use_theta,
                         use_area=use_area, dup_eps=dup_eps, dtype=dtype,
                         max_batch=max_batch, **score_kwargs)
        self.orient_weight = float(orient_weight)
        self.dim = 2 + 2 * int(use_theta) + int(use_area)   # orientation = (cos, sin)

    @torch.no_grad()
    def __call__(self, term_lat: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        flat = term_lat.reshape(-1, term_lat.shape[-2], term_lat.shape[-1])
        feats, found = [], []
        for s in range(0, flat.shape[0], self.max_batch):
            rgb = self._rgb_batch(flat[s:s + self.max_batch][:, None])
            H, W = rgb.shape[1], rgb.shape[2]
            side = float(np.sqrt(H * W))
            for k in range(rgb.shape[0]):
                _, dbg = score_t_centered_angle(rgb[k], **self.score_kwargs)
                phi = _t_heading(dbg["mask"]) if dbg.get("found") else float("nan")
                # undefined pose: no T, or a mask too degenerate for a stable heading
                if not dbg.get("found") or (self.use_theta and not np.isfinite(phi)):
                    feats.append([np.nan] * self.dim)
                    found.append(False)
                    continue
                cx, cy = dbg["centroid"]
                row = [cx / W, cy / H]
                if self.use_theta:
                    row += [self.orient_weight * float(np.cos(phi)),
                            self.orient_weight * float(np.sin(phi))]
                if self.use_area:
                    row.append(float(np.sqrt(dbg["area"])) / side)
                feats.append(row)
                found.append(True)
        return np.asarray(feats, dtype=np.float32), np.asarray(found, dtype=bool)


# ---------------------------------------------------------------------------
# env-agnostic fallbacks
# ---------------------------------------------------------------------------

class RewardScalarDescriptor(StateDescriptor):
    """Fallback: the scalar reward as a 1-D descriptor. Always available for any
    ``RewardFn``, but weak — two states with equal reward are indistinguishable
    (this is exactly the blind spot ``edge_val_std`` already has)."""

    def __init__(self, reward_fn, dup_eps: float = 0.02):
        self.reward_fn = reward_fn
        self.dim = 1
        self.dup_eps = float(dup_eps)

    @torch.no_grad()
    def __call__(self, term_lat: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        flat = term_lat.reshape(-1, term_lat.shape[-2], term_lat.shape[-1])
        r = self.reward_fn(flat).reshape(-1).float().cpu().numpy()
        feats = r[:, None].astype(np.float32)
        return feats, np.ones(feats.shape[0], dtype=bool)


class LatentMeanDescriptor(StateDescriptor):
    """Trivial model-free descriptor (mean + std of the latent grid). For
    unit-testing the metric code without a decoder; NOT a meaningful diversity
    space (see the module docstring)."""

    dim = 2
    dup_eps = 1e-3

    @torch.no_grad()
    def __call__(self, term_lat: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        flat = term_lat.reshape(term_lat.shape[0], -1).float().cpu().numpy()
        feats = np.stack([flat.mean(1), flat.std(1)], axis=1).astype(np.float32)
        return feats, np.ones(feats.shape[0], dtype=bool)
