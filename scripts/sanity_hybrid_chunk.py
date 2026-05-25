"""Sanity test for the HybridChunkSampler call chain.

Smoke test only — does NOT validate output quality. Verifies:
  1. The new methods plumb through without shape/type errors.
  2. The cache append_partial mechanism behaves correctly under both
     normal append and roll-on-overflow.
  3. The autoregressive m=1, k=1 hybrid path runs without crashing.

Usage:
    python scripts/sanity_hybrid_chunk.py \
        --config-path scripts/config --config-name dynamics/pushT

(uses dummy random latents/actions — no dataset / tokenizer / checkpoint
needed. Construct a freshly-initialized denoiser from the config alone.)
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Allow running this script directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreamerv4uwm.models.dynamics import DenoiserWrapper      # noqa: E402
from dreamerv4uwm.inference import HybridChunkSampler         # noqa: E402


def _build_denoiser(cfg, device, dtype):
    denoiser = DenoiserWrapper(
        cfg, max_num_forward_steps=cfg.denoiser.max_sequence_length,
    ).to(device=device, dtype=dtype)
    denoiser.eval()
    for p in denoiser.parameters():
        p.requires_grad_(False)
    return denoiser


def _resolve_dataset_block(cfg):
    """Strip required dataset/tokenizer composition so we can load the bare
    config standalone (no Hydra overrides needed for the sanity)."""
    # Wipe optional but required-at-runtime path fields with placeholders.
    cfg.dataset = OmegaConf.create({})
    return cfg


def test_kvcache_append_partial(device):
    from dreamerv4uwm.models.blocks import KVCache
    cap = 8
    cache = KVCache(
        context_length=cap, batch_size=2, num_heads=4, head_dim=16,
        device=device, dtype=torch.float32,
    )
    # Append 5 fresh tokens.
    k1 = torch.randn(2, 4, 6, 16, device=device)
    v1 = torch.randn(2, 4, 6, 16, device=device)
    cache.append_partial(k1, v1, n=5)
    assert cache.curr_len == 5
    assert torch.allclose(cache.k_cache[:, :, :5], k1[:, :, :5])

    # Append partial 2 of a chunk of 4 (should fit, len becomes 7).
    k2 = torch.randn(2, 4, 4, 16, device=device)
    v2 = torch.randn(2, 4, 4, 16, device=device)
    cache.append_partial(k2, v2, n=2)
    assert cache.curr_len == 7
    assert torch.allclose(cache.k_cache[:, :, 5:7], k2[:, :, :2])

    # Append partial 3 (would overflow: 7+3=10 > 8). Should roll left by 2.
    k3 = torch.randn(2, 4, 3, 16, device=device)
    v3 = torch.randn(2, 4, 3, 16, device=device)
    cache.append_partial(k3, v3, n=3)
    assert cache.curr_len == 8, f"got {cache.curr_len}"
    # After rolling left by 2: original entries 2..7 are now at 0..5; new
    # 3 entries at 5..8.
    assert torch.allclose(cache.k_cache[:, :, 5:8], k3[:, :, :3])
    print(f"  ✓ KVCache.append_partial: normal + roll-on-overflow OK")


def test_hybrid_chunk_sampler(cfg, device, dtype, chunk_size, commit_per_chunk):
    denoiser = _build_denoiser(cfg, device, dtype)
    sampler = HybridChunkSampler(
        denoiser=denoiser,
        cfg=cfg,
        chunk_size=chunk_size,
        commit_per_chunk=commit_per_chunk,
        num_diffusion_steps=2,        # cheap K for sanity
        action_noise_std=1.0,
        clean_commit_pass=True,
        device=device,
        dtype=dtype,
    )

    B = 1
    sampler.init_cache(batch_size=B)

    # Seed with 4 clean frames.
    seed_T = 4
    seed_latents = torch.randn(
        B, seed_T, cfg.denoiser.num_latent_tokens, cfg.denoiser.latent_dim,
        device=device, dtype=dtype,
    )
    seed_actions = torch.randn(
        B, seed_T, cfg.denoiser.n_actions, device=device, dtype=dtype,
    )
    sampler.warm_up_cache(seed_latents, seed_actions)
    assert sampler._cache_len == seed_T

    # Generate 6 frames total.
    z_gen, a_gen = sampler.generate(n_frames=6)
    assert z_gen.shape == (B, 6, cfg.denoiser.num_latent_tokens, cfg.denoiser.latent_dim), \
        f"z_gen shape {z_gen.shape}"
    assert a_gen.shape == (B, 6, cfg.denoiser.n_actions), f"a_gen shape {a_gen.shape}"
    assert torch.isfinite(z_gen).all(), "non-finite values in z_gen"
    assert torch.isfinite(a_gen).all(), "non-finite values in a_gen"
    print(
        f"  ✓ HybridChunkSampler m={chunk_size}, k={commit_per_chunk}: "
        f"generated {z_gen.shape[1]} frames, cache_len={sampler._cache_len}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config-path", default="scripts/config",
        help="Hydra config root (relative to repo root).",
    )
    ap.add_argument(
        "--config-name", default="dynamics/pushT",
        help="Hydra config name (without .yaml).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_root = (repo_root / args.config_path).resolve()
    cfg_path = config_root / f"{args.config_name}.yaml"
    cfg = OmegaConf.load(cfg_path)
    # Pull in the tokenizer composition if it's a defaults-list ref.
    # For the sanity test we don't actually use the tokenizer; the denoiser
    # only needs cfg.denoiser to construct.

    cfg = _resolve_dataset_block(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Config: {args.config_name}")
    print(f"  num_latent_tokens={cfg.denoiser.num_latent_tokens}, "
          f"latent_dim={cfg.denoiser.latent_dim}, "
          f"n_actions={cfg.denoiser.n_actions}, "
          f"context_length={cfg.denoiser.context_length}")

    print("\n[1] KVCache.append_partial")
    test_kvcache_append_partial(device)

    print("\n[2] HybridChunkSampler — autoregressive equivalent (m=1, k=1)")
    test_hybrid_chunk_sampler(cfg, device, dtype, chunk_size=1, commit_per_chunk=1)

    print("\n[3] HybridChunkSampler — chunk-wise (m=4, k=4)")
    test_hybrid_chunk_sampler(cfg, device, dtype, chunk_size=4, commit_per_chunk=4)

    print("\n[4] HybridChunkSampler — MPC-style (m=4, k=1)")
    test_hybrid_chunk_sampler(cfg, device, dtype, chunk_size=4, commit_per_chunk=1)

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
