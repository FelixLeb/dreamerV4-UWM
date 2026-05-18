#!/usr/bin/env python3
"""
gen_fdpo_negatives.py

Generate (positive, negative) trajectory pairs for FDPO alignment training.

For each demo window of length `ctx_frames + horizon_frames`:
  - The positive trajectory ξ⁺ is the full demo window as recorded.
  - The negative trajectory ξ⁻ shares the same first `ctx_frames` (demo
    context), then continues for `horizon_frames` with a joint state-action
    rollout from the play-trained reference denoiser in policy mode, sampled
    via the progressive autoregressive rolling sampler.

The output is a sharded HDF5 dataset in a pair-centric schema (see
`PreferencePairDataset` in `dreamerv4uwm/datasets.py`):

  pos_images, pos_actions, neg_images, neg_actions, episode_lengths

The first `ctx_frames` of pos_images and neg_images are identical by
construction (clean demo context shared between ξ⁺ and ξ⁻); the trailing
`horizon_frames` are the demo's true continuation (positive) and the
reference's rollout (negative). The boundary is implicit at `ctx_frames`,
recorded in `metadata.json`.

Example:
    conda run -n dreamerv4 python scripts/preprocessing/gen_fdpo_negatives.py \\
        --src_data_dir /home/mim-server/datasets/pushT/sharded-demo \\
        --output_dir   /home/mim-server/datasets/pushT/fdpo-pairs-v0 \\
        --ckpt_dir     checkpoints/dynamics/pushT/final/joint-model/v1 \\
        --ckpt_file    432360.pt \\
        --tokenizer_ckpt checkpoints/tokenizer_ckpts/pushT.pt \\
        --stride 32 --max_pairs 4 --device cuda:0
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamerv4uwm.datasets import ShardedHDF5Dataset
from dreamerv4uwm.models.utils import load_denoiser, load_tokenizer
from dreamerv4uwm.sampling import get_noise_index


# ---------------------------------------------------------------------------
# Progressive autoregressive rolling sampler — copied from notebooks/planning.ipynb
# so this script is self-contained. Mirrors the pattern used in
# `synthetic_reward_model_datagen.py:hybrid_progressive_sampler`.
#
# Per-frame staircase noise schedule (Xie et al., 2024, arXiv:2410.08151)
# adapted to the UWM joint obs+action flowmatching denoiser. Each main-loop
# iteration:
#   1. All F window frames denoise by one quantum (dt = 1/F).
#   2. The frontmost frame reaches tau=1 (clean) and is emitted.
#   3. The window shifts left; a fresh pure-noise frame is appended at the back.
#   4. The clean prefix (length C) gains the emitted frame and drops its oldest.
# ---------------------------------------------------------------------------

@torch.no_grad()
def progressive_rolling_sampler(
    denoiser,
    init_latents,
    init_actions,
    total_pred_steps,
    window_size=16,
    context_size=8,
    context_cond_tau=0.9,
    action_noise_std=1.0,
):
    F, C = window_size, context_size
    device = init_latents.device
    dtype  = init_latents.dtype
    B, T_init, N_lat, D_lat = init_latents.shape
    n_act = init_actions.shape[-1]
    init_actions = init_actions.to(device=device, dtype=dtype)

    assert C <= T_init, f"context_size C={C} cannot exceed init context T_init={T_init}"
    N = denoiser.cfg.denoiser.num_noise_levels
    assert N % F == 0, "num_noise_levels must be divisible by window_size"

    stride       = N // F
    step_size    = 1.0 / F
    tau_cond_idx = get_noise_index(context_cond_tau, N)

    prefix_obs = init_latents[:, -C:].clone() if C > 0 else None
    prefix_act = init_actions[:, -C:].clone() if C > 0 else None

    z_obs   = torch.randn(B, 1, N_lat, D_lat, device=device, dtype=dtype)
    z_act   = action_noise_std * torch.randn(B, 1, n_act, device=device, dtype=dtype)
    tau_idx = torch.zeros((B, 1), dtype=torch.long, device=device)

    def _fresh_noise():
        return (
            torch.randn(B, 1, N_lat, D_lat, device=device, dtype=dtype),
            action_noise_std * torch.randn(B, 1, n_act, device=device, dtype=dtype),
            torch.zeros((B, 1), dtype=torch.long, device=device),
        )

    pred_obs_buf, pred_act_buf = [], []
    while len(pred_obs_buf) < total_pred_steps:
        cur_F = z_obs.shape[1]
        if C > 0:
            p_obs_in = (1.0 - context_cond_tau) * torch.randn_like(prefix_obs) + context_cond_tau * prefix_obs
            p_act_in = ((1.0 - context_cond_tau) * action_noise_std * torch.randn_like(prefix_act)
                        + context_cond_tau * prefix_act)
            obs_in = torch.cat([p_obs_in, z_obs], dim=1)
            act_in = torch.cat([p_act_in, z_act], dim=1)
            prefix_sigma = torch.full((B, C), tau_cond_idx, dtype=torch.long, device=device)
            sigma_idx = torch.cat([prefix_sigma, tau_idx], dim=1)
        else:
            obs_in, act_in, sigma_idx = z_obs, z_act, tau_idx

        step_idx_tensor = torch.zeros((B, C + cur_F), dtype=torch.long, device=device)

        # Current denoiser returns (z_hat, act_hat, pred_rewards-or-None).
        out = denoiser(
            noisy_act    = act_in,
            noisy_obs    = obs_in,
            obs_sigma_idx= sigma_idx,
            obs_step_idx = step_idx_tensor,
            act_sigma_idx= sigma_idx,
            act_step_idx = step_idx_tensor,
        )
        if isinstance(out, tuple) and len(out) == 3:
            z_hat, act_hat, _ = out
        else:
            z_hat, act_hat = out
        act_hat = act_hat.squeeze(-2)
        z_hat_w, act_hat_w = z_hat[:, C:], act_hat[:, C:]

        tau_cont = tau_idx.float() / float(N)
        denom    = (1.0 - tau_cont).clamp_min(1e-5)
        z_obs = z_obs + (z_hat_w   - z_obs) / denom[..., None, None] * step_size
        z_act = z_act + (act_hat_w - z_act) / denom[..., None]       * step_size
        tau_idx = tau_idx + stride

        if cur_F < F:
            new_o, new_a, new_t = _fresh_noise()
            z_obs   = torch.cat([z_obs,   new_o], dim=1)
            z_act   = torch.cat([z_act,   new_a], dim=1)
            tau_idx = torch.cat([tau_idx, new_t], dim=1)
        else:
            emit_obs, emit_act = z_obs[:, :1], z_act[:, :1]
            pred_obs_buf.append(emit_obs)
            pred_act_buf.append(emit_act)

            new_o, new_a, new_t = _fresh_noise()
            z_obs   = torch.cat([z_obs[:, 1:],   new_o], dim=1)
            z_act   = torch.cat([z_act[:, 1:],   new_a], dim=1)
            tau_idx = torch.cat([tau_idx[:, 1:], new_t], dim=1)

            if C > 0:
                prefix_obs = torch.cat([prefix_obs[:, 1:], emit_obs], dim=1)
                prefix_act = torch.cat([prefix_act[:, 1:], emit_act], dim=1)

    return torch.cat(pred_obs_buf, dim=1), torch.cat(pred_act_buf, dim=1)


def _strip_deprecated_keys(cfg):
    if 'latent_attends_action' in cfg.denoiser:
        del cfg.denoiser['latent_attends_action']


def _load_models(ckpt_dir: Path, ckpt_file: str, tokenizer_ckpt: str, device,
                 max_seq_len: int):
    cfg = OmegaConf.load(ckpt_dir / 'config.yaml')
    cfg.dynamics_ckpt = str(ckpt_dir / ckpt_file)
    cfg.tokenizer_ckpt = str(tokenizer_ckpt)
    _strip_deprecated_keys(cfg)
    # The trained tokenizer config has max_sequence_length=32; we need to
    # encode/decode up to `max_seq_len` frames at once. RoPE buffers are sized
    # off max_num_forward_steps, so override.
    denoiser  = load_denoiser(cfg,  device, max_num_forward_steps=max_seq_len).eval()
    tokenizer = load_tokenizer(cfg, device, max_num_forward_steps=max_seq_len).eval()
    for p in denoiser.parameters():
        p.requires_grad_(False)
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    return cfg, denoiser, tokenizer


def _to_uint8_thwc(imgs_chw_float):
    """(T, C, H, W) float in [0, 1] -> (T, H, W, C) uint8."""
    arr = imgs_chw_float.clamp(0, 1).permute(0, 2, 3, 1).to(torch.float32).cpu().numpy()
    return (arr * 255.0).astype(np.uint8)


@torch.no_grad()
def _generate_one_pair(
    demo_images_chw_float,   # (T, C, H, W) float in [0, 1], from ShardedHDF5Dataset
    demo_actions,            # (T, A_full) float — full action dim from HDF5
    denoiser,
    tokenizer,
    ctx_frames: int,
    horizon_frames: int,
    n_actions: int,
    window_size: int,
    device,
    autocast_dtype,
):
    """Build one (pos, neg) pair from a single demo window.

    Returns (pos_imgs_THWC_u8, pos_acts_TA_f32, neg_imgs_THWC_u8, neg_acts_TA_f32).
    """
    T = ctx_frames + horizon_frames
    assert demo_images_chw_float.shape[0] >= T

    imgs = demo_images_chw_float[:T].to(device=device, dtype=torch.float32).unsqueeze(0)
    acts = demo_actions[:T, :n_actions].to(device=device, dtype=torch.float32).unsqueeze(0)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                        enabled=(device.type == 'cuda')):
        ctx_latents = tokenizer.encode(imgs[:, :ctx_frames])
        ctx_actions = acts[:, :ctx_frames]

        # Progressive autoregressive rollout. The sampler's clean-prefix size is
        # always the full init context — for FDPO we want maximum conditioning
        # of the negative on the demo prefix.
        pred_latents, pred_actions = progressive_rolling_sampler(
            denoiser,
            init_latents=ctx_latents,
            init_actions=ctx_actions,
            total_pred_steps=horizon_frames,
            window_size=window_size,
            context_size=ctx_latents.shape[1],
        )
        pred_images = tokenizer.decode(pred_latents)

    # ξ⁺ = the full demo window as recorded
    pos_imgs_np = _to_uint8_thwc(imgs[0])
    pos_acts_np = acts[0].cpu().numpy().astype(np.float32)

    # ξ⁻ = clean demo prefix + ref-rolled continuation
    neg_imgs_chw = torch.cat([imgs[0, :ctx_frames], pred_images[0]], dim=0)
    neg_acts     = torch.cat([acts[0, :ctx_frames], pred_actions[0]], dim=0)
    neg_imgs_np  = _to_uint8_thwc(neg_imgs_chw)
    neg_acts_np  = neg_acts.cpu().numpy().astype(np.float32)

    return pos_imgs_np, pos_acts_np, neg_imgs_np, neg_acts_np


def _write_shard(shard_path: Path, pairs):
    """pairs: list of (pos_imgs, pos_acts, neg_imgs, neg_acts)."""
    n = len(pairs)
    T, H, W, C = pairs[0][0].shape
    n_act = pairs[0][1].shape[-1]

    pos_imgs_arr = np.stack([p[0] for p in pairs], axis=0)
    pos_acts_arr = np.stack([p[1] for p in pairs], axis=0)
    neg_imgs_arr = np.stack([p[2] for p in pairs], axis=0)
    neg_acts_arr = np.stack([p[3] for p in pairs], axis=0)
    ep_lengths   = np.full((n,), T, dtype=np.int32)

    comp = {'compression': 'gzip', 'compression_opts': 4}
    img_chunks = (1, min(T, 8), H, W, C)
    act_chunks = (1, min(T, 8), n_act)
    with h5py.File(shard_path, 'w') as f:
        f.create_dataset('pos_images',      data=pos_imgs_arr, chunks=img_chunks, **comp)
        f.create_dataset('pos_actions',     data=pos_acts_arr, chunks=act_chunks, **comp)
        f.create_dataset('neg_images',      data=neg_imgs_arr, chunks=img_chunks, **comp)
        f.create_dataset('neg_actions',     data=neg_acts_arr, chunks=act_chunks, **comp)
        f.create_dataset('episode_lengths', data=ep_lengths)
        f.attrs['num_pairs'] = n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # I/O
    p.add_argument('--src_data_dir',     type=str, required=True)
    p.add_argument('--output_dir',       type=str, required=True)
    p.add_argument('--ckpt_dir',         type=str, required=True,
                   help="Dir holding config.yaml + the .pt checkpoint.")
    p.add_argument('--ckpt_file',        type=str, required=True,
                   help="Filename of the .pt under ckpt_dir.")
    p.add_argument('--tokenizer_ckpt',   type=str, required=True)
    # Window split
    p.add_argument('--ctx_frames',       type=int, default=32)
    p.add_argument('--horizon_frames',   type=int, default=32)
    p.add_argument('--stride',           type=int, default=1,
                   help="Stride between demo windows. Controls dataset size. "
                        "Default 1 = densest (one pair per demo frame offset).")
    # Sampling (progressive autoregressive rolling sampler).
    p.add_argument('--window_size',  type=int, default=16,
                   help="F: per-frame staircase depth (= #denoising steps to clean one frame).")
    # Output
    p.add_argument('--pairs_per_shard',  type=int, default=64)
    # Limits
    p.add_argument('--max_pairs',        type=int, default=None,
                   help="Optional cap on total pairs (for smoke tests).")
    # Misc
    p.add_argument('--device',           type=str, default='cuda:0')
    p.add_argument('--seed',             type=int, default=0)
    # Source-dataset split
    p.add_argument('--src_split',        type=str, default='train', choices=['train', 'test'])
    p.add_argument('--src_train_fraction', type=float, default=1.0,
                   help="Default 1.0: use the whole demo set as source (demos are scarce).")
    p.add_argument('--src_split_seed',   type=int, default=123)

    args = p.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    autocast_dtype = torch.bfloat16

    # --- Models ---
    print(f"Loading models from {args.ckpt_dir}/{args.ckpt_file}")
    window = args.ctx_frames + args.horizon_frames
    cfg, denoiser, tokenizer = _load_models(
        Path(args.ckpt_dir), args.ckpt_file, args.tokenizer_ckpt, device,
        max_seq_len=window,
    )
    n_actions = cfg.denoiser.n_actions
    print(f"  denoiser n_actions={n_actions}, context_length={cfg.denoiser.context_length}, "
          f"max_seq={cfg.denoiser.max_sequence_length}")
    print(f"  sampler: progressive rolling | window_size F={args.window_size} | "
          f"clean prefix C=ctx_frames={args.ctx_frames}")

    # --- Demo dataset (source of windows) ---
    print(f"Opening demo dataset {args.src_data_dir} (window={window}, stride={args.stride})")
    src_dataset = ShardedHDF5Dataset(
        data_dir       = args.src_data_dir,
        window_size    = window,
        stride         = args.stride,
        split          = args.src_split,
        train_fraction = args.src_train_fraction,
        split_seed     = args.src_split_seed,
        shuffle_windows= False,
    )
    n_windows = len(src_dataset)
    if args.max_pairs is not None:
        n_windows = min(n_windows, args.max_pairs)
    print(f"  {len(src_dataset)} windows available; generating {n_windows} pairs")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Generation loop ---
    pbar = tqdm(total=n_windows, desc="Pairs")
    image_shape, action_shape = None, None
    shard_index = 0
    pairs_buffer = []

    for win_idx in range(n_windows):
        sample = src_dataset[win_idx]
        pair = _generate_one_pair(
            demo_images_chw_float = sample['image'],
            demo_actions          = sample['action'],
            denoiser              = denoiser,
            tokenizer             = tokenizer,
            ctx_frames            = args.ctx_frames,
            horizon_frames        = args.horizon_frames,
            n_actions             = n_actions,
            window_size           = args.window_size,
            device                = device,
            autocast_dtype        = autocast_dtype,
        )
        if image_shape is None:
            image_shape  = list(pair[0].shape[1:])  # pos_imgs (T, H, W, C) — drop T
            action_shape = list(pair[1].shape[1:])  # pos_acts (T, n_act) — drop T
        pairs_buffer.append(pair)

        if len(pairs_buffer) >= args.pairs_per_shard:
            _write_shard(output_path / f"shard_{shard_index:04d}.h5", pairs_buffer)
            shard_index += 1
            pairs_buffer = []
        pbar.update(1)
    if pairs_buffer:
        _write_shard(output_path / f"shard_{shard_index:04d}.h5", pairs_buffer)
        shard_index += 1
    pbar.close()

    # --- Metadata ---
    metadata = {
        'schema':           'fdpo_pairs_v1',
        'num_shards':       shard_index,
        'total_pairs':      n_windows,
        'image_shape':      image_shape,
        'action_shape':     action_shape,
        'ctx_frames':       args.ctx_frames,
        'horizon_frames':   args.horizon_frames,
        'src_data_dir':     args.src_data_dir,
        'ckpt_dir':         args.ckpt_dir,
        'ckpt_file':        args.ckpt_file,
        'tokenizer_ckpt':   args.tokenizer_ckpt,
        'sampler':          'progressive_rolling',
        'window_size':      args.window_size,
        'seed':             args.seed,
        'stride':           args.stride,
    }
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Wrote {shard_index} shards ({n_windows} pairs) -> {output_path}")


if __name__ == '__main__':
    main()
