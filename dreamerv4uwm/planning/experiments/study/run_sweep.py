"""Sweep driver: expand a factor grid into a job list, build one tree per job,
write a CSV shard. Designed for SLURM array parallelism and crash-resume.

Config is an OmegaConf YAML (``config/sweep.yaml``); CLI takes ``key=value``
dotlist overrides plus ``--task-id/--num-tasks`` for array slicing. We deliberately
do NOT use ``@hydra.main`` here so the model's own Hydra compose
(``model.load_world_model``) doesn't collide with an outer Hydra context.

Job list = (config × initial-context), identical across all array tasks (built
from fixed seeds), then strided by ``task_id`` so each task gets a balanced mix.
Each task appends to ``shard_<task_id>.csv`` and skips ``(config_id, init_id)``
rows already present (resume).

Run:
    python -m dreamerv4uwm.planning.experiments.study.run_sweep \
        --config <path>/config/sweep.yaml --task-id 0 --num-tasks 8 output_dir=/scratch/mcts_sweep
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from ...mcts import PlanConfig
from ...reward import TCenterReward
from .model import load_world_model, make_decode_fn, model_dims
from .data import make_dataset, sample_initial_contexts, load_curated_contexts
from .descriptors import TPoseDescriptor
from .run_tree import run_one_tree

_PLAN_FIELDS = {f.name for f in dataclasses.fields(PlanConfig)}


# ---------------------------------------------------------------------------
# job list
# ---------------------------------------------------------------------------

def build_configs(base: dict, sweep: dict) -> List[Tuple[str, dict]]:
    """List of (tag, override_dict). OFAT perturbs one factor at a time from
    ``base``; the optional random tier draws each factor uniformly from its grid."""
    configs: List[Tuple[str, dict]] = [("base", {})]
    ofat = dict(sweep.get("ofat") or {})
    for factor, values in ofat.items():
        for v in values:
            if factor in base and _eq(v, base[factor]):
                continue
            configs.append((f"ofat.{factor}={v}", {factor: v}))
    rnd = dict(sweep.get("random") or {})
    n = int(rnd.get("n_samples", 0))
    if n > 0 and ofat:
        rng = np.random.default_rng(int(rnd.get("seed", 0)))
        for i in range(n):
            ov = {f: _coerce(rng.choice(np.asarray(vs))) for f, vs in ofat.items()}
            configs.append((f"rand.{i}", ov))
    return configs


def _eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return a == b


def _coerce(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v.item() if isinstance(v, np.generic) else v


def make_plan_cfg(base: dict, override: dict) -> PlanConfig:
    merged = {**base, **override}
    kw = {k: merged[k] for k in merged if k in _PLAN_FIELDS}
    return PlanConfig(**kw)


# ---------------------------------------------------------------------------
# csv shard (append + resume)
# ---------------------------------------------------------------------------

def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            done.add((int(r["config_id"]), int(r["init_id"])))
    return done


def _append_row(path: Path, row: dict, fieldnames):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def load_cfg(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to sweep.yaml")
    ap.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--limit", type=int, default=None, help="cap #jobs (local smoke)")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides key=value")
    args = ap.parse_args(argv)
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    return args, cfg


def main(argv=None):
    args, cfg = load_cfg(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] task {args.task_id}/{args.num_tasks} | device={device} | out={out_dir}", flush=True)

    denoiser, tokenizer, _ = load_world_model(
        dynamics_ckpt=cfg.model.dynamics_ckpt, tokenizer_ckpt=cfg.model.tokenizer_ckpt,
        config_name=cfg.model.get("config_name", "dynamics/pushT-large"),
        overrides=list(cfg.model.get("overrides", ["denoiser.horizon_aware=false"])),
        device=device)
    dims = model_dims(denoiser)
    decode = make_decode_fn(tokenizer, device)
    score_kw = dict(center_xy=tuple(cfg.reward.center_xy), sigma=float(cfg.reward.sigma))
    reward = TCenterReward(decode_fn=decode, **score_kw)
    descriptor = TPoseDescriptor(decode_fn=decode, dup_eps=float(cfg.get("descriptor", {}).get("dup_eps", 0.05)),
                                 **score_kw)

    inits_cfg = cfg.get("inits", None)
    mode = str(inits_cfg.mode) if (inits_cfg and "mode" in inits_cfg) else "random"
    if mode == "file":
        ipath = Path(str(inits_cfg.path))
        if not ipath.is_absolute():
            ipath = Path(args.config).resolve().parent / ipath
        spec = OmegaConf.to_container(OmegaConf.load(ipath), resolve=True)
        d = spec.get("dataset", {})  # structural params pin window_idx and MUST match curation;
        # the location (data_dir) is machine-specific -> take it from data.data_dir (CLI/SLURM-overridable),
        # falling back to the value pinned in the curated file. The start_reward fingerprint catches mismatch.
        data_dir = cfg.data.get("data_dir", None) or d.get("data_dir")
        print(f"[sweep] file-mode dataset: data_dir={data_dir} (structural params from {ipath.name})", flush=True)
        dataset = make_dataset(data_dir,
                               window_size=int(d.get("window_size", cfg.data.get("window_size", 64))),
                               stride=int(d.get("stride", 1)), split=str(d.get("split", "train")),
                               train_fraction=float(d.get("train_fraction", 0.9)),
                               split_seed=int(d.get("split_seed", 123)),
                               shuffle_windows=bool(d.get("shuffle_windows", False)))
        inits = load_curated_contexts(dataset, tokenizer, spec, device=device,
                                      n_actions=dims["n_actions"], reward_fn=reward)
        print(f"[sweep] loaded {len(inits)} curated inits from {ipath}", flush=True)
    else:
        dataset = make_dataset(cfg.data.data_dir, window_size=int(cfg.data.get("window_size", 64)))
        inits = sample_initial_contexts(dataset, tokenizer, n=int(cfg.n_inits), Tc=int(cfg.Tc),
                                        device=device, n_actions=dims["n_actions"], seed=int(cfg.seed))

    base = OmegaConf.to_container(cfg.base_plan, resolve=True)
    configs = build_configs(base, OmegaConf.to_container(cfg.sweep, resolve=True))
    jobs = [(ci, tag, ov, init) for ci, (tag, ov) in enumerate(configs) for init in inits]
    jobs = jobs[args.task_id::args.num_tasks]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[sweep] {len(configs)} configs x {len(inits)} inits -> {len(jobs)} jobs for this task", flush=True)

    shard = out_dir / f"shard_{args.task_id:04d}.csv"
    done = _done_keys(shard)
    fieldnames: List[str] = None
    n_done = 0
    for ci, tag, ov, init in jobs:
        if (ci, init["init_id"]) in done:
            continue
        plan_cfg = make_plan_cfg(base, ov)
        row = run_one_tree(
            denoiser=denoiser, reward_fn=reward, descriptor=descriptor, plan_cfg=plan_cfg,
            ctx_z=init["ctx_z"], ctx_a=init["ctx_a"], plan_seed=int(init["init_id"]),
            meta=dict(config_id=ci, config_tag=tag, window_idx=init["window_idx"],
                      t0=init["t0"], init_id=init["init_id"]),
            n_random=int(cfg.get("n_random", 16)))
        if fieldnames is None:
            fieldnames = (["config_id", "config_tag", "window_idx", "t0", "init_id"]
                          + [k for k in row if k not in
                             ("config_id", "config_tag", "window_idx", "t0", "init_id")])
        _append_row(shard, row, fieldnames)
        n_done += 1
        if n_done % 10 == 0 or n_done == 1:
            print(f"[sweep] {n_done}/{len(jobs)} done | last={tag} "
                  f"g_rand={row.get('g_rand'):+.3f} bci={row.get('root_bci')}", flush=True)

    print(f"[sweep] task {args.task_id} finished: wrote {n_done} rows to {shard}", flush=True)


if __name__ == "__main__":
    main()
