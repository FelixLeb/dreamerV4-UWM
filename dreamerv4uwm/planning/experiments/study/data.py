"""Initial-context sourcing for the sweep.

An "initialisation" is a decision point: a context window ``(ctx_z, ctx_a)`` of
``Tc`` frames encoded from a real PushT demo window at a random ``(window, t0)``.
Mirrors the notebook's ``get_window`` / ``make_state`` (cells 5, 7).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.nn.functional import interpolate


def make_dataset(data_dir: str, *, window_size: int = 64, stride: int = 1,
                 split: str = "train", train_fraction: float = 0.9,
                 split_seed: int = 123, shuffle_windows: bool = False):
    from dreamerv4uwm.datasets import ShardedHDF5Dataset
    return ShardedHDF5Dataset(data_dir=data_dir, window_size=window_size, stride=stride,
                              split=split, train_fraction=train_fraction,
                              split_seed=split_seed, shuffle_windows=shuffle_windows)


@torch.no_grad()
def encode_window(dataset, tokenizer, idx: int, device: torch.device, n_actions: int,
                  resolution=(256, 256), dtype: torch.dtype = torch.bfloat16):
    """Return ``(latents (1,T,N,D), actions (1,T,n_act))`` for demo window ``idx``."""
    batch = dataset[idx]
    imgs = interpolate(batch["image"], resolution).to(device)[None]        # (1,T,3,H,W)
    actions = batch["action"][:, :n_actions][None].to(device)              # (1,T,n_act)
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else None
    if ctx is not None:
        with ctx:
            latents = tokenizer.encode(imgs).float()
    else:
        latents = tokenizer.encode(imgs).float()
    return latents, actions


def sample_initial_contexts(dataset, tokenizer, *, n: int, Tc: int, device: torch.device,
                            n_actions: int, seed: int, min_t0: int = 0,
                            resolution=(256, 256)) -> List[dict]:
    """``n`` reproducible initial contexts. Each item:
    ``{ctx_z (1,Tc,N,D), ctx_a (1,Tc,n_act), window_idx, t0, init_id}``."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        idx = int(rng.integers(len(dataset)))
        latents, actions = encode_window(dataset, tokenizer, idx, device, n_actions, resolution)
        T = latents.shape[1]
        hi = max(min_t0 + 1, T - Tc)
        t0 = int(rng.integers(min_t0, hi))
        out.append(dict(
            ctx_z=latents[:, t0:t0 + Tc].clone(),
            ctx_a=actions[:, t0:t0 + Tc].clone(),
            window_idx=idx, t0=t0, init_id=i,
        ))
    return out


@torch.no_grad()
def load_curated_contexts(dataset, tokenizer, spec: dict, *, device: torch.device,
                          n_actions: int, reward_fn=None, fingerprint_tol: float = 0.05,
                          resolution=(256, 256)) -> List[dict]:
    """Load a hand-curated init set (see config/inits/*.yaml). ``spec`` is the parsed
    YAML: ``{Tc, inits:[{id, window_idx, t0, label, start_reward?}], ...}``.

    If ``reward_fn`` and a stored ``start_reward`` are present, the reward of the
    decision frame is recomputed and a drift warning is printed on mismatch — this
    is the tripwire against a silently-remapped dataset (window_idx only means
    something under a fixed dataset construction; see the yaml's ``dataset:`` block)."""
    Tc = int(spec["Tc"])
    out = []
    for e in spec["inits"]:
        widx, t0 = int(e["window_idx"]), int(e["t0"])
        latents, actions = encode_window(dataset, tokenizer, widx, device, n_actions, resolution)
        T = latents.shape[1]
        if t0 + Tc > T:
            raise ValueError(f"init id={e.get('id')}: t0+Tc={t0 + Tc} exceeds window len {T}")
        cz = latents[:, t0:t0 + Tc].clone()
        ca = actions[:, t0:t0 + Tc].clone()
        item = dict(ctx_z=cz, ctx_a=ca, window_idx=widx, t0=t0,
                    init_id=int(e.get("id", len(out))), label=e.get("label"))
        if reward_fn is not None and e.get("start_reward") is not None:
            r = float(reward_fn(cz[:, -1:]).reshape(-1)[0])
            if abs(r - float(e["start_reward"])) > fingerprint_tol:
                print(f"[warn] init {item['init_id']}: start_reward drift "
                      f"(stored {float(e['start_reward']):.3f} vs recomputed {r:.3f}) "
                      f"— dataset construction may have changed.", flush=True)
            item["start_reward"] = r
        out.append(item)
    return out
