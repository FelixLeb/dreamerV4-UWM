"""Bayesian optimisation over the MCTS planner's hyper-parameters.

One BO *evaluation* = one config run over the whole curated init set (the sweep's unit of
comparison), scored by the mean of an outcome column. That is expensive and noisy, which is
exactly the regime BO is for: a Gaussian-process surrogate is fitted to every config
evaluated so far and an acquisition function decides where to spend the next batch.

Three things make this instance non-generic, and all three are handled here:

* **Known, heterogeneous noise.** The per-config standard error is measured (over the inits)
  rather than fitted, and passed to the GP per point. It is strongly heteroscedastic --- on
  the reference runs the replicate spread grows ~6x from ``ctx_noise=0`` to ``0.9``, i.e. the
  noise is largest exactly where the optimum lives. A constant-noise GP misfits there.
* **A hard compute budget.** Quality genuinely improves with compute, so without a constraint
  the optimiser just buys it. Candidates are filtered by :func:`plan_frames` *before* the
  acquisition is evaluated, so every proposal is iso-compute by construction.
* **A warm start.** Existing sweep shards are read straight in, so the first batch is already
  informed by ~90 evaluated configs instead of starting from a blank GP.

Typical use --- propose the next batch from the runs done so far::

    python -m dreamerv4uwm.planning.experiments.study.optimize \\
        --runs <sweep>/run_normal <sweep>/run_uniform \\
        --outcome tree_last --budget 25000 --q 8 --out rounds/round_001.yaml

then run that file through ``run_sweep.py`` (``configs.mode: file``), and call this again
with the new shard directory appended to ``--runs``.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# search space
# ---------------------------------------------------------------------------
# (name, kind, spec). kind: "float" | "int" -> spec is (low, high) inclusive
#                           "cat"           -> spec is the list of levels
# Anything a PlanConfig needs but that is NOT listed here is held at FIXED and written
# verbatim into every proposed config, so a round is reproducible from this file alone.

SPACE: List[Tuple[str, str, object]] = [
    ("ctx_noise",    "float", (0.0, 0.9)),
    ("horizon",      "int",   (6, 28)),
    ("sim_horizon",  "int",   (4, 24)),
    ("branching",    "int",   (3, 8)),
    ("sim_rollouts", "int",   (2, 5)),
    ("n_iterations", "int",   (16, 48)),
    ("max_depth",    "int",   (2, 5)),
    ("c_ucb",        "float", (0.1, 8.0)),
    ("action_temp",  "float", (0.5, 1.5)),
    ("action_prior", "cat",   ["normal", "uniform"]),
    ("state_prior",  "cat",   ["normal", "uniform"]),
]

FIXED: Dict[str, object] = {
    "edge_mode": "imagine",          # autoregressive is swept separately (2H cost per edge)
    "K_steps": 6,
    "max_ctx": 8,
    "gamma": 0.98,
    "n_min": 0,
    "ctx_noise_honest": True,
    "action_noise": 0.0,
    "action_noise_dist": "normal",
}

# Fraction of MCTS iterations that actually expand the tree. Each iteration always simulates
# but only expands when the selected leaf has been visited AND is below max_depth (see
# mcts._iteration) -- so the rate rises with the depth cap, and a single constant biases the
# budget. Measured over the reference runs (5760 trees) as
# rho = (n_forward - 1 - n_iterations) / n_iterations:
_EXPANSION_RATE = {2: 0.134, 3: 0.231, 5: 0.313}


def expansion_rate(max_depth) -> float:
    """Measured expansion rate for a depth cap, linearly interpolated between the observed
    points and clamped outside them. Using the pooled average instead (0.22) under-costs a
    ``max_depth=5`` tree by ~10% -- a systematic leak in exactly the direction the optimiser
    pushes, since deeper trees score better."""
    xs = sorted(_EXPANSION_RATE)
    return float(np.interp(float(max_depth), xs, [_EXPANSION_RATE[x] for x in xs]))


def plan_frames(cfg: Dict) -> float:
    """Denoiser frames consumed by one tree --- the compute budget the search is capped on.

    Each rollout primitive runs ``K_steps`` denoiser passes over a batch of ``B`` sequences of
    length ``max_ctx + H``, so frames (not calls) is what tracks cost. Counting *calls*
    instead --- the planner's own ``n_forward`` --- misses ``branching`` and ``sim_rollouts``
    entirely: on the reference runs two configs with identical ``n_forward`` differed by 1.8x
    in wall clock, which this formula recovers as 1.80x (r=0.87 against measured seconds,
    vs 0.76 for ``n_forward``).

    Computable from the config alone, so it can gate candidates before anything is run.
    """
    g = lambda k: float(cfg.get(k, FIXED.get(k)))
    k_steps, max_ctx, n_iter = g("K_steps"), g("max_ctx"), g("n_iterations")
    rho = expansion_rate(g("max_depth"))
    expand = (1.0 + rho * n_iter) * g("branching") * (max_ctx + g("horizon"))
    simulate = n_iter * g("sim_rollouts") * (max_ctx + g("sim_horizon"))
    return k_steps * (expand + simulate)


# ---------------------------------------------------------------------------
# encoding:  config dict <-> plain numeric vector for the GP
# ---------------------------------------------------------------------------

def encode(cfg: Dict) -> np.ndarray:
    """Config -> vector in [0,1]^d (each dimension scaled to its own range, categoricals
    to level indices), so one ARD length-scale per dimension is comparable across knobs."""
    v = []
    for name, kind, spec in SPACE:
        if kind == "cat":
            v.append(spec.index(str(cfg[name])) / max(len(spec) - 1, 1))
        else:
            lo, hi = spec
            v.append((float(cfg[name]) - lo) / (hi - lo))
    return np.asarray(v, float)


def decode(vec: Sequence[float]) -> Dict:
    """Vector in [0,1]^d -> config dict, with integers rounded and FIXED merged in."""
    cfg = dict(FIXED)
    for x, (name, kind, spec) in zip(vec, SPACE):
        if kind == "cat":
            cfg[name] = spec[int(round(float(x) * (len(spec) - 1)))]
        else:
            lo, hi = spec
            val = lo + float(x) * (hi - lo)
            cfg[name] = int(round(val)) if kind == "int" else round(float(val), 4)
    return cfg


def sample_candidates(n: int, rng) -> np.ndarray:
    return rng.random((n, len(SPACE)))


# ---------------------------------------------------------------------------
# warm start: sweep shards -> one observation per config
# ---------------------------------------------------------------------------

def load_observations(run_dirs: Sequence[str], outcome: str, min_inits: int = 20):
    """Read ``shard_*.csv`` from each run and reduce to one row per distinct config.

    Rows are grouped by the SPACE dimensions only, so the same config evaluated in several
    runs is pooled into a single, better-estimated observation (more inits -> smaller SEM).
    Returns ``(X, y, yvar, cfgs, n_inits)``; ``yvar`` is the squared standard error of the
    mean over inits, i.e. the observation noise handed to the GP.
    """
    groups = defaultdict(list)
    cfg_of = {}
    for d in run_dirs:
        files = sorted(glob.glob(str(Path(d) / "shard_*.csv")))
        if not files:
            raise FileNotFoundError(f"no shard_*.csv in {d}")
        for f in files:
            for r in csv.DictReader(open(f)):
                if outcome not in r or r[outcome] in ("", "nan"):
                    continue
                try:
                    key = tuple(str(r[name]) for name, _, _ in SPACE)
                    val = float(r[outcome])
                except (KeyError, ValueError):
                    continue
                if not math.isfinite(val):
                    continue
                groups[key].append(val)
                cfg_of.setdefault(key, {**FIXED, **{name: r[name] for name, _, _ in SPACE},
                                        **{k: r[k] for k in ("K_steps", "max_ctx") if k in r}})

    X, y, yvar, cfgs, ns = [], [], [], [], []
    for key, vals in groups.items():
        if len(vals) < min_inits:
            continue
        raw = cfg_of[key]
        cfg = decode(encode({**raw, **{name: _num(raw[name], kind)
                                       for name, kind, _ in SPACE}}))
        arr = np.asarray(vals, float)
        sem = arr.std(ddof=1) / math.sqrt(arr.size) if arr.size > 1 else float("nan")
        X.append(encode(cfg)); y.append(arr.mean()); yvar.append(sem ** 2)
        cfgs.append(cfg); ns.append(arr.size)
    return (np.asarray(X), np.asarray(y), np.asarray(yvar), cfgs, np.asarray(ns))


def _num(v, kind):
    return v if kind == "cat" else float(v)


# ---------------------------------------------------------------------------
# surrogate + acquisition
# ---------------------------------------------------------------------------

def fit_gp(X, y, yvar):
    """ARD Matern-5/2 GP with **known, per-point** observation noise.

    y is standardised by hand (and ``yvar`` with it) rather than via ``normalize_y``: sklearn
    adds ``alpha`` to the kernel matrix in the *model's* y-units, so with an internally
    normalised y the supplied variances would be interpreted on the wrong scale. Doing it
    explicitly keeps the noise and the signal in the same units.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    y_mean, y_std = float(y.mean()), float(y.std()) or 1.0
    # Upper length-scale bound is deliberately far above the [0,1] input range: ARD signals an
    # irrelevant dimension by sending its length-scale to infinity, and on this data several
    # knobs (action_temp, the two priors, sim_rollouts) genuinely are flat. A tight bound
    # would clip that and silently keep dead dimensions in the distance metric.
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e5), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=np.maximum(yvar, 1e-12) / y_std ** 2,
                                  normalize_y=False, n_restarts_optimizer=5, random_state=0)
    gp.fit(X, (y - y_mean) / y_std)
    return gp, y_mean, y_std


def predict(gp, y_mean, y_std, X):
    mu, sd = gp.predict(np.atleast_2d(X), return_std=True)
    return mu * y_std + y_mean, sd * y_std


def cross_validate(X, y, yvar, k=5, seed=0):
    """K-fold check that the surrogate is worth spending GPU hours on.

    Two numbers matter. **RMSE vs the noise floor**: the GP can never beat the observation
    noise (median sem), so ``rmse ~ noise`` means it has learned everything the data supports,
    while ``rmse >> noise`` means it is missing structure. **Coverage**: the fraction of
    held-out points inside +/-2 sigma of their prediction should be near 0.95 --- much lower
    means the GP is overconfident, and an overconfident GP proposes extrapolations it cannot
    back up (which is exactly the risk when the optimiser walks outside the swept grid).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    err, inside, sds = [], [], []
    for f in folds:
        tr = np.setdiff1d(idx, f)
        gp, ym, ys = fit_gp(X[tr], y[tr], yvar[tr])
        mu, sd = predict(gp, ym, ys, X[f])
        # the held-out value is itself noisy: the predictive interval must cover both
        tot = np.sqrt(sd ** 2 + yvar[f])
        err.append(mu - y[f]); inside.append(np.abs(mu - y[f]) <= 2 * tot); sds.append(sd)
    err = np.concatenate(err)
    return {"rmse": float(np.sqrt((err ** 2).mean())),
            "noise_floor": float(np.sqrt(np.median(yvar))),
            "coverage_2sigma": float(np.concatenate(inside).mean()),
            "mean_pred_sd": float(np.concatenate(sds).mean())}


def expected_improvement(mu, sd, best, xi=0.0):
    """E[max(0, f(x) - best)] for a maximisation problem (closed form under the GP posterior)."""
    from scipy.stats import norm
    sd = np.maximum(sd, 1e-12)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def propose(X, y, yvar, observed_cfgs, *, budget, q=8, n_candidates=40000, seed=0, xi=0.0):
    """Return ``q`` configs to evaluate next, all satisfying ``plan_frames <= budget``.

    The budget is enforced by discarding infeasible candidates *before* the acquisition is
    scored --- a hard constraint, not a penalty, so no proposal can ever trade compute for
    quality. Batches use the kriging-believer heuristic: after each pick the surrogate is
    refitted with that point's *predicted* value inserted, which keeps the batch from
    collapsing onto one spot without needing a second evaluation.
    """
    rng = np.random.default_rng(seed)
    Xw, yw, vw = X.copy(), y.copy(), yvar.copy()
    median_var = float(np.median(vw[np.isfinite(vw)])) if np.isfinite(vw).any() else 1e-4
    # observations that already satisfy the budget: the incumbent must be one of these, since
    # an over-budget config is not a solution to the constrained problem we are posing
    feasible_obs = np.array([plan_frames(c) <= budget for c in observed_cfgs])

    cand = sample_candidates(n_candidates, rng)
    frames = np.array([plan_frames(decode(c)) for c in cand])
    cand = cand[frames <= budget]
    if cand.size == 0:
        raise ValueError(f"no candidate fits budget={budget:g} plan_frames; "
                         f"cheapest sampled = {frames.min():.0f}")

    picks = []
    for _ in range(q):
        gp, y_mean, y_std = fit_gp(Xw, yw, vw)
        mu, sd = predict(gp, y_mean, y_std, cand)
        # Incumbent = best POSTERIOR MEAN among feasible evaluated points, not the best
        # observed value. With ~90 noisy observations (sem ~0.03) the running max is biased
        # upward by well over one sem, and using it as the EI reference makes the acquisition
        # chase a target that no config actually achieves. The posterior mean is denoised.
        mu_obs, _ = predict(gp, y_mean, y_std, X[feasible_obs])
        feasible_best = float(np.max(mu_obs)) if mu_obs.size else float(np.max(yw))
        ei = expected_improvement(mu, sd, feasible_best, xi=xi)
        i = int(np.argmax(ei))
        picks.append((decode(cand[i]), float(mu[i]), float(sd[i]), float(ei[i])))
        # kriging believer: pretend we observed the posterior mean there, then re-fit
        Xw = np.vstack([Xw, cand[i]]); yw = np.append(yw, mu[i]); vw = np.append(vw, median_var)
        cand = np.delete(cand, i, axis=0)
    return picks


# ---------------------------------------------------------------------------
# round file (consumed by run_sweep.py with configs.mode=file)
# ---------------------------------------------------------------------------

def write_round(picks, path, *, outcome, budget, round_id):
    from omegaconf import OmegaConf
    configs = []
    for i, (cfg, mu, sd, ei) in enumerate(picks):
        configs.append({"tag": f"bo.r{round_id}.{i}",
                        "predicted_mean": round(mu, 5), "predicted_std": round(sd, 5),
                        "expected_improvement": round(ei, 6),
                        "plan_frames": round(plan_frames(cfg), 1),
                        **{k: cfg[k] for k in cfg}})
    doc = {"meta": {"outcome": outcome, "budget_plan_frames": budget, "round": round_id},
           "configs": configs}
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(OmegaConf.to_yaml(OmegaConf.create(doc)))
    return p


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="dirs of shard_*.csv (warm start)")
    ap.add_argument("--outcome", default="tree_last")
    ap.add_argument("--budget", type=float, default=None,
                    help="max plan_frames per tree (default: median of the warm-start configs)")
    ap.add_argument("--q", type=int, default=8, help="configs to propose (match the array size)")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--out", default=None, help="round yaml to write")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xi", type=float, default=0.0,
                    help="EI exploration margin in outcome units; raise (e.g. one sem, ~0.03) "
                         "if a batch clusters too tightly in the known-good basin")
    ap.add_argument("--cv", type=int, default=5, help="k-fold surrogate check (0 to skip)")
    a = ap.parse_args(argv)

    X, y, yvar, cfgs, ns = load_observations(a.runs, a.outcome)
    frames = np.array([plan_frames(c) for c in cfgs])
    budget = a.budget if a.budget is not None else float(np.median(frames))

    print(f"warm start: {len(y)} configs, {int(ns.sum())} trees, outcome={a.outcome}")
    print(f"  {a.outcome}: best={y.max():.4f}  mean={y.mean():.4f}  spread(sd)={y.std():.4f}")
    print(f"  noise (sem): median={np.sqrt(np.median(yvar)):.4f}  "
          f"min={np.sqrt(yvar.min()):.4f}  max={np.sqrt(yvar.max()):.4f}")
    print(f"  plan_frames: min={frames.min():.0f}  median={np.median(frames):.0f}  max={frames.max():.0f}")
    print(f"budget = {budget:.0f} plan_frames  "
          f"({(frames <= budget).sum()}/{len(frames)} warm-start configs feasible)")

    if a.cv:
        cv = cross_validate(X, y, yvar, k=a.cv, seed=a.seed)
        ratio = cv["rmse"] / cv["noise_floor"]
        verdict = ("at the noise floor" if ratio < 1.3 else
                   "usable" if ratio < 2.0 else "POOR - the GP is missing structure")
        print(f"surrogate ({a.cv}-fold): rmse={cv['rmse']:.4f} vs noise floor "
              f"{cv['noise_floor']:.4f} ({ratio:.1f}x, {verdict})")
        print(f"  2-sigma coverage={cv['coverage_2sigma']:.2f} (want ~0.95; "
              f"{'ok' if cv['coverage_2sigma'] >= 0.85 else 'OVERCONFIDENT - distrust extrapolations'})")

    picks = propose(X, y, yvar, cfgs, budget=budget, q=a.q, seed=a.seed, xi=a.xi)
    print(f"\nproposed {len(picks)} configs:")
    hdr = ["ctx_noise", "horizon", "sim_horizon", "branching", "sim_rollouts",
           "n_iterations", "max_depth", "c_ucb", "action_temp", "action_prior", "state_prior"]
    print("  " + "".join(f"{h[:11]:>12s}" for h in hdr) + f"{'frames':>9s}{'pred':>8s}{'sd':>7s}")
    for cfg, mu, sd, ei in picks:
        print("  " + "".join(f"{str(cfg[h])[:11]:>12s}" for h in hdr)
              + f"{plan_frames(cfg):9.0f}{mu:8.3f}{sd:7.3f}")
    if a.out:
        print("\nwrote", write_round(picks, a.out, outcome=a.outcome, budget=budget, round_id=a.round))


if __name__ == "__main__":
    main()
