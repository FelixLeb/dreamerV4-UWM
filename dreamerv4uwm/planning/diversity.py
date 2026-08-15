"""How different are the ``B`` edges sampled from one node? (Section 3 of ``edge-diversity.tex``)

Everything here answers one question: given ``B`` sibling edges drawn from a single node,
how many *distinct* successors did we actually get? Collapse (all siblings identical) makes
UCB blind, so this is the quantity every intervention in the research plan is judged on.

Three groups:

* **Aggregators** — :func:`vendi_score` (effective number of distinct items, the headline
  metric) and :func:`knn_entropy` (RE3-style particle estimator). Both take features and a
  metric; they are agnostic to *what* the features are.
* **Representations** — the features to aggregate over. Action space is free and needs no
  decode; :class:`RandomEncoder` is the RE3 random-CNN readout of the decoded frame.
* **Fidelity** — :func:`self_consistency`, so diversity is never reported alone. Diversity
  is trivially maximised by emitting garbage; the pair is the unit of measurement.

.. warning::

   **Fix the kernel bandwidth across conditions.** :func:`vendi_score` with
   ``bandwidth=None`` uses the median heuristic *on the set being scored*, which
   **self-normalises**: a fully collapsed set gets a tiny bandwidth and scores as diverse.
   That makes the metric nearly blind to exactly the failure it exists to detect, and is a
   likely explanation for an inconclusive Vendi result. Compute one bandwidth from a fixed
   reference sample with :func:`median_bandwidth` and pass it to every condition.

.. note::

   **Do not measure in the raw latent.** The tokenizer's latent is a set of *learned query
   tokens* (``learned_latent_tokens`` in ``models/tokenizer.py``, Perceiver/DETR style),
   concatenated to the patch tokens and attended over — token ``i`` carries no spatial
   meaning and the set has no grid structure. It is then squashed through ``tanh``, so
   coordinates saturate toward +/-1 and Euclidean distance is dominated by sign flips.
   Distance there has no reason to track behavioural similarity. For the same reason a
   random *linear* projection of the latent will not help: Johnson-Lindenstrauss preserves
   pairwise distances, so it preserves the problem. Use action space, or a nonlinear
   encoder on the **decoded frame** (:class:`RandomEncoder`).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# feature preparation
# ---------------------------------------------------------------------------

def flatten_edges(x: torch.Tensor) -> np.ndarray:
    """``(B, ...) -> (B, d)`` float64 numpy. Use on ``a_hor (B,H,n_act)`` for the free,
    decode-less action-space readout."""
    return x.detach().reshape(x.shape[0], -1).float().cpu().numpy().astype(np.float64)


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Full ``(B, B)`` Euclidean distance matrix."""
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.sqrt(np.maximum(d2, 0.0))


def median_bandwidth(X_ref: np.ndarray) -> float:
    """Median pairwise distance of a **reference** sample — the bandwidth to reuse across
    every condition being compared (see the module warning).

    A good reference: edges drawn from *different* contexts in the demo data, so that
    ``bandwidth`` means "as different as two unrelated real trajectories"."""
    D = pairwise_distances(X_ref)
    iu = np.triu_indices(len(X_ref), k=1)
    m = float(np.median(D[iu])) if iu[0].size else 0.0
    return m if m > 0 else 1.0


# ---------------------------------------------------------------------------
# pairwise statistics
# ---------------------------------------------------------------------------

def pairwise_stats(X: np.ndarray, *, dup_eps: float = 0.05,
                   ref_scale: Optional[float] = None) -> dict:
    """Spread statistics over ``B`` edge features ``X (B, d)``.

    * ``apd``      — average pairwise distance (mean spread);
    * ``min_asd``  — **minimum** pairwise distance. The sensitive one: it detects that *at
      least two* siblings are duplicates even when the rest are spread out, which is exactly
      the failure ``apd`` averages away;
    * ``max_pd``   — diameter of the sibling set;
    * ``dup_rate`` — fraction of pairs closer than ``dup_eps`` (in ``ref_scale`` units if
      given);
    * ``apd_norm`` — ``apd / ref_scale``; ``~1`` means siblings are as different as two
      unrelated real trajectories.
    """
    B = len(X)
    if B < 2:
        return dict(apd=float("nan"), min_asd=float("nan"), max_pd=float("nan"),
                    dup_rate=float("nan"), apd_norm=float("nan"))
    D = pairwise_distances(X)
    iu = np.triu_indices(B, k=1)
    pd = D[iu]
    scale = ref_scale if ref_scale else 1.0
    return dict(
        apd=float(pd.mean()),
        min_asd=float(pd.min()),
        max_pd=float(pd.max()),
        dup_rate=float((pd / scale < dup_eps).mean()),
        apd_norm=float(pd.mean() / scale),
    )


# ---------------------------------------------------------------------------
# aggregator 1 : the Vendi Score
# ---------------------------------------------------------------------------

def vendi_score(X: np.ndarray, *, q: float = 1.0, bandwidth: Optional[float] = None,
                return_eigenvalues: bool = False):
    """Effective number of distinct items among ``B`` edge features ``X (B, d)``.

    ``VS = exp(-sum_i lambda_i log lambda_i)`` over the eigenvalues of ``K/B``, where
    ``K_ij = exp(-||x_i - x_j||^2 / (2 h^2))`` is an RBF kernel with unit diagonal. Bounds
    are ``1 <= VS <= B``: ``1`` iff all edges are identical (total collapse), ``B`` iff they
    are mutually orthogonal under the kernel. ``VS = 1.2`` at ``branching=5`` means the
    expansion effectively produced one child.

    ``q`` selects the Renyi/Hill order: ``1`` (default) is Shannon, ``0`` counts non-zero
    eigenvalues (richness), ``inf`` gives ``1/lambda_max`` (sensitive to one mode swallowing
    everything). Log all three to tell "a few rare distinct edges" from "balanced spread".

    ``bandwidth`` **must be held fixed across conditions** — see the module warning.
    """
    B = len(X)
    if B < 2:
        return (float("nan"), np.array([])) if return_eigenvalues else float("nan")
    h = bandwidth if bandwidth else median_bandwidth(X)
    K = np.exp(-(pairwise_distances(X) ** 2) / (2.0 * h * h))     # unit diagonal by construction
    lam = np.linalg.eigvalsh(K / B)
    lam = np.clip(lam, 0.0, None)
    s = lam.sum()
    lam = lam / s if s > 0 else lam
    nz = lam[lam > 1e-12]

    if np.isinf(q):
        vs = float(1.0 / nz.max()) if nz.size else float("nan")
    elif abs(q - 1.0) < 1e-9:
        vs = float(np.exp(-(nz * np.log(nz)).sum()))
    elif q == 0:
        vs = float(nz.size)
    else:
        vs = float(np.exp(np.log((nz ** q).sum()) / (1.0 - q)))
    return (vs, lam) if return_eigenvalues else vs


def vendi_profile(X: np.ndarray, *, bandwidth: Optional[float] = None) -> dict:
    """``{'vs_0': richness, 'vs_1': Vendi, 'vs_inf': 1/lambda_max}`` in one call."""
    return {"vs_0": vendi_score(X, q=0.0, bandwidth=bandwidth),
            "vs_1": vendi_score(X, q=1.0, bandwidth=bandwidth),
            "vs_inf": vendi_score(X, q=np.inf, bandwidth=bandwidth)}


# ---------------------------------------------------------------------------
# aggregator 2 : RE3-style k-NN particle entropy
# ---------------------------------------------------------------------------

def knn_entropy(X: np.ndarray, *, k: int = 1, ref_scale: Optional[float] = None) -> float:
    """RE3's particle entropy estimate, ``mean_i log(||y_i - y_i^kNN||_2 + 1)``.

    From Seo et al., ICML 2021 (arXiv:2102.09430), which estimates state entropy with the
    Singh k-NN estimator in a fixed random-encoder space. Here the "particles" are the ``B``
    siblings of one expansion.

    **Read it as a relative score, not as entropy in nats.** k-NN entropy estimators are
    heavily biased at small ``N``, and ``B`` is typically 3--16. Only compare values at equal
    ``B``, and pool over many expansions before concluding anything. At ``k=1`` this is a
    smoothed cousin of ``pairwise_stats(...)['min_asd']``.

    ``ref_scale`` divides the distances first, which makes the ``+1`` offset meaningful
    across feature spaces of different scale — pass the same reference used for Vendi.
    """
    B = len(X)
    if B < 2:
        return float("nan")
    k = min(k, B - 1)
    D = pairwise_distances(X) / (ref_scale if ref_scale else 1.0)
    np.fill_diagonal(D, np.inf)
    knn = np.sort(D, axis=1)[:, k - 1]                 # distance to the k-th neighbour
    return float(np.log(knn + 1.0).mean())


# ---------------------------------------------------------------------------
# representation : RE3 random encoder (on DECODED frames)
# ---------------------------------------------------------------------------

class RandomEncoder(nn.Module):
    """Fixed, randomly-initialised conv encoder for RE3-style diversity on decoded frames.

    RE3's justification is that "the structure alone of deep convolutional networks is a
    powerful inductive bias" — locality and translation equivariance over a **spatial grid**.
    That is why this runs on the decoded image and not on the latent, which has no grid
    structure (see the module note).

    Cost is near-zero in the planner: the reward already decodes the very frames we want to
    measure (edge-terminal states in ``_expand``, every frame in ``_simulate``), so this
    piggybacks on a decode that has already happened.

    RE3 froze a *random* encoder to avoid non-stationarity while an agent trains. That
    concern does not apply to an offline diagnostic, so a pretrained encoder is a legitimate
    (and probably stronger) alternative — e.g. the DINOv2 already loaded by
    ``reward.DINOGoalReward``, using **patch** tokens rather than CLS, since CLS is trained
    crop-invariant and can be nearly blind to *where* the object is. Run both; if they agree,
    keep this one for speed.
    """

    def __init__(self, in_ch: int = 3, width: int = 32, out_dim: int = 64, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, stride=2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        for p in self.net.parameters():                # deterministic init from `seed`
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * (1.0 / max(p.shape[-1], 1) ** 0.5)
                        if p.dim() > 1 else torch.zeros(p.shape))
        self.proj = nn.Linear(width * 2 * 4, out_dim)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=g)
                                   / self.proj.weight.shape[1] ** 0.5)
            self.proj.bias.zero_()
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """``(B,3,H,W)`` in ``[0,1]`` -> ``(B, out_dim)``."""
        return self.proj(self.net(imgs))

    @torch.no_grad()
    def features(self, imgs: torch.Tensor) -> np.ndarray:
        """Same, returned as ``(B, out_dim)`` float64 numpy ready for the aggregators."""
        dev = next(self.parameters()).device
        return self(imgs.to(dev)).float().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# fidelity : is the diverse sample still a sample the model believes in?
# ---------------------------------------------------------------------------

@torch.no_grad()
def self_consistency(denoiser, ctx_z, ctx_a, z_hor, a_hor, *, K: int = 6,
                     dtype=torch.bfloat16, generator=None) -> float:
    """Relative residual between an edge's imagined states and the world model's own
    prediction for its actions: ``rho = ||z - transition(ctx, a)|| / ||z||``.

    ``imagine`` returns ``(z, a)`` that are supposed to be *jointly* consistent under the
    model. Re-running the world model on the sampled actions and finding a different state
    means the joint sample is internally incoherent — a direct signature of an off-manifold
    sample, and the cheapest fidelity guard available: one extra ``transition`` call, no
    decode and no reward.

    Report every diversity number paired with this. Any intervention can be pushed until the
    edges are diverse *and worthless*; only the pair distinguishes the two.
    """
    from .rollout import transition
    z_pred = transition(denoiser, ctx_z, ctx_a, a_hor, K=K, dtype=dtype, generator=generator)
    num = (z_hor - z_pred).flatten(1).norm(dim=1)
    den = z_hor.flatten(1).norm(dim=1).clamp_min(1e-8)
    return float((num / den).mean().item())


def action_range_violation(a_hor: torch.Tensor, lo: float, hi: float) -> float:
    """Fraction of sampled action components outside the demo-data range ``[lo, hi]``.

    The crudest fidelity guard, and the one that catches ``action_temp`` / additive
    ``action_noise`` emitting actions the robot could not execute."""
    a = a_hor.detach()
    return float(((a < lo) | (a > hi)).float().mean().item())


# ---------------------------------------------------------------------------
# one call for a whole sibling set
# ---------------------------------------------------------------------------

def edge_diversity(a_hor: torch.Tensor, *, bandwidth: Optional[float] = None,
                   ref_scale: Optional[float] = None, dup_eps: float = 0.05,
                   feats: Optional[np.ndarray] = None, prefix: str = "") -> dict:
    """Full diversity readout for one expansion's ``B`` siblings.

    Pass ``a_hor (B,H,n_act)`` for the free action-space readout, or supply ``feats (B,d)``
    from :class:`RandomEncoder` for the state-space one. Returns a flat dict, prefixed by
    ``prefix`` so action- and state-space readouts can be merged into one row::

        row = {**edge_diversity(a, prefix="act_"), **edge_diversity(a, feats=f, prefix="re3_")}

    Comparing the two spaces is the informative bit: actions diverse but states collapsed
    means the *world model* funnels, and no sampler fix will help.
    """
    X = feats if feats is not None else flatten_edges(a_hor)
    out = {**pairwise_stats(X, dup_eps=dup_eps, ref_scale=ref_scale),
           **vendi_profile(X, bandwidth=bandwidth),
           "knn_ent": knn_entropy(X, k=1, ref_scale=ref_scale),
           "B": len(X)}
    return {f"{prefix}{k}": v for k, v in out.items()}
