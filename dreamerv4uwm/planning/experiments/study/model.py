"""Load the world model once per process, and build a decode fn.

Mirrors the notebook's setup (``planning-mcts-experiments.ipynb`` cells 3–5):
Hydra-compose ``dynamics/pushT-large`` (``horizon_aware=false``), point at the
checkpoints, load denoiser + tokenizer, run everything under bf16.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch


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
    """Return ``(denoiser, tokenizer, cfg)`` on ``device``, eval mode."""
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
    """``lat (B,T,N,D) -> video (B,T,3,H,W) float[0,1]`` under bf16 autocast."""
    @torch.no_grad()
    def decode(lat: torch.Tensor) -> torch.Tensor:
        ctx = (torch.autocast(device_type="cuda", dtype=dtype)
               if (device.type == "cuda" and dtype) else nullcontext())
        with ctx:
            v = tokenizer.decode(lat.to(device))
        return v.float().clamp(0, 1)
    return decode


def model_dims(denoiser) -> dict:
    d = denoiser.cfg.denoiser
    return dict(num_noise_levels=int(d.num_noise_levels),
                num_latent_tokens=int(d.num_latent_tokens),
                latent_dim=int(d.latent_dim),
                n_actions=int(d.n_actions))
