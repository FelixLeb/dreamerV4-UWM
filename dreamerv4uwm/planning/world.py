"""Stand up a planning experiment: load the world model, get initial contexts.

Everything a notebook needs *before* it can plan, and nothing else. No metrics, no
search, no rewards — just the two ingredients :mod:`mcts` consumes:

* a **model** — the frozen denoiser + tokenizer, plus a ``decode_fn`` for looking at
  latents (:func:`load_world_model`, :func:`make_decode_fn`, :func:`model_dims`);
* an **initial context** — a window ``(ctx_z, ctx_a)`` of ``Tc`` real frames encoded from a
  PushT demo, i.e. the decision point the planner starts from
  (:func:`make_dataset`, :func:`encode_window`, :func:`sample_initial_contexts`,
  :func:`load_curated_contexts`).

A notebook should be able to start in three lines::

    denoiser, tokenizer, cfg = load_world_model(dynamics_ckpt=DYN, tokenizer_ckpt=TOK)
    decode = make_decode_fn(tokenizer, device)
    ctx = sample_initial_contexts(make_dataset(DATA), tokenizer, n=8, Tc=8, device=device,
                                  n_actions=cfg.denoiser.n_actions, seed=0)

Everything runs under bf16 autocast — fp32 OOMs at planning batch sizes.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.functional import interpolate


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

def load_world_model(
    *,
    dynamics_ckpt: str,
    tokenizer_ckpt: str,
    config_name: str = "dynamics/pushT-large",
    overrides: Sequence[str] = ("denoiser.horizon_aware=false",),
    config_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    max_num_forward_steps: int = 300,
) -> Tuple[object, object, object]:
    """Return ``(denoiser, tokenizer, cfg)`` on ``device``, eval mode.

    Hydra-composes ``config_name`` and points it at the two checkpoints. ``config_dir``
    defaults to the repo's ``scripts/config`` resolved from the installed package, so this
    works from any working directory (a relative ``../scripts/config`` only resolves when
    the notebook is run from ``notebooks/``).
    """
    import dreamerv4uwm
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from dreamerv4uwm.models.utils import load_tokenizer, load_denoiser

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config_dir is None:
        config_dir = str(Path(dreamerv4uwm.__file__).resolve().parent.parent / "scripts" / "config")

    GlobalHydra.instance().clear()  # safe to call load_world_model more than once
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    cfg.dynamics_ckpt = str(dynamics_ckpt)
    cfg.tokenizer_ckpt = str(tokenizer_ckpt)

    denoiser = load_denoiser(cfg, device, max_num_forward_steps=max_num_forward_steps).eval().to(device)
    tokenizer = load_tokenizer(cfg, device, max_num_forward_steps=max_num_forward_steps).eval().to(device)
    return denoiser, tokenizer, cfg


def make_decode_fn(tokenizer, device: torch.device, dtype: Optional[torch.dtype] = torch.bfloat16):
    """``lat (B,T,N,D) -> video (B,T,3,H,W) float[0,1]`` under bf16 autocast.

    This is the ``decode_fn=`` every pixel reward and descriptor takes."""
    @torch.no_grad()
    def decode(lat: torch.Tensor) -> torch.Tensor:
        ctx = (torch.autocast(device_type="cuda", dtype=dtype)
               if (device.type == "cuda" and dtype) else nullcontext())
        with ctx:
            v = tokenizer.decode(lat.to(device))
        return v.float().clamp(0, 1)
    return decode


def model_dims(denoiser) -> dict:
    """The four shape constants the rollout primitives read off the checkpoint."""
    d = denoiser.cfg.denoiser
    return dict(num_noise_levels=int(d.num_noise_levels),
                num_latent_tokens=int(d.num_latent_tokens),
                latent_dim=int(d.latent_dim),
                n_actions=int(d.n_actions))


# ---------------------------------------------------------------------------
# initial contexts
# ---------------------------------------------------------------------------

def make_dataset(data_dir: str, *, window_size: int = 64, stride: int = 1,
                 split: str = "train", train_fraction: float = 0.9,
                 split_seed: int = 123, shuffle_windows: bool = False):
    """The sharded-HDF5 PushT demo dataset the contexts are drawn from."""
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
    """``n`` reproducible initial contexts at random ``(window, t0)``. Each item:
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
    """Load a hand-curated init set. ``spec`` is the parsed
    YAML: ``{Tc, reward_kind?, inits:[{id, window_idx, t0, label, start_reward?,
    start_center_fingerprint?}], ...}``.

    The drift tripwire compares a recomputed reward on the decision frame against the
    stored **fingerprint** and warns on mismatch (window_idx only means something under a
    fixed dataset construction; see the spec's ``dataset:`` block). The fingerprint is the
    stored ``start_center_fingerprint`` — the kind-independent CENTER score — so ``reward_fn`` should
    be a center reward; this stays decode-robust regardless of which reward kind the
    experiment uses. Older files without ``start_center_fingerprint`` fall back to
    ``start_reward`` (which they stored as a center score). ``start_reward`` is carried
    through unchanged for reference."""
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
        if e.get("start_reward") is not None:            # carry the kind score through (reference)
            item["start_reward"] = float(e["start_reward"])
        fp_key = "start_center_fingerprint" if e.get("start_center_fingerprint") is not None else "start_reward"
        if reward_fn is not None and e.get(fp_key) is not None:
            r = float(reward_fn(cz[:, -1:]).reshape(-1)[0])
            if abs(r - float(e[fp_key])) > fingerprint_tol:
                print(f"[warn] init {item['init_id']}: {fp_key} drift "
                      f"(stored {float(e[fp_key]):.3f} vs recomputed {r:.3f}) "
                      f"— dataset construction may have changed.", flush=True)
            item["start_center_fingerprint"] = r
        out.append(item)
    return out
