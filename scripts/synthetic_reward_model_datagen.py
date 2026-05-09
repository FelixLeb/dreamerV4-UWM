#!/usr/bin/env python3
"""
synthetic_reward_model_datagen.py

Generate a sharded HDF5 dataset of hybrid demo→play snippets for training a
reward / discriminator model, in the same on-disk schema as the dataset
loaded by `dreamerv4uwm.datasets.ShardedHDF5Dataset`.

Each generated episode has fixed length `--episode_length` (default 96) and
carries a per-frame `is_demo` label. Internally we generate `episode_length
+ 1` frames and drop the first emission (which sits right against the clean
real-demo seed and is the only visually discontinuous frame), leaving
`episode_length` frames on the WM manifold.

Per episode:
  1. Pick a random demo trajectory from the source ShardedHDF5Dataset.
  2. Pick a random split point  s ∈ [1, episode_length], so demo_len = s,
     play_pred = (episode_length + 1) - s.
  3. Run `hybrid_progressive_sampler`:
       - First s emissions: action sigma = clean_idx, action *value* pinned
         to the real demo action; obs denoised through the staircase.
       - Remaining play_pred emissions: obs and action denoised jointly.
  4. Drop emission 0; keep `episode_length` frames + length-`episode_length`
     `is_demo` array.

Both halves go through the *same* obs pipeline so a downstream reward model
can't latch on to pipeline statistics — the only structural difference
between halves is which actions condition the dynamics.
"""

import argparse
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.nn.functional import interpolate
from tqdm import tqdm

from dreamerv4uwm.datasets import ShardedHDF5Dataset
from dreamerv4uwm.models.utils import load_denoiser, load_tokenizer
from dreamerv4uwm.sampling import get_noise_index


# ---------------------------------------------------------------------------
# Hybrid progressive sampler — copied from notebooks/planning.ipynb so this
# script is self-contained. See the notebook for design notes.
# ---------------------------------------------------------------------------

@torch.no_grad()
def hybrid_progressive_sampler(
    denoiser,
    init_latents,
    init_actions,
    demo_actions,
    play_pred_steps,
    window_size=4,
    context_size=28,
    context_cond_tau=0.9,
    action_noise_std=1.0,
):
    F, C = window_size, context_size
    device, dtype = init_latents.device, init_latents.dtype
    B, T_init, N_lat, D_lat = init_latents.shape
    n_act = init_actions.shape[-1]
    init_actions = init_actions.to(device=device, dtype=dtype)
    demo_actions = demo_actions.to(device=device, dtype=dtype)
    demo_len = demo_actions.shape[1]
    L = demo_len + play_pred_steps

    assert T_init >= 1, "init_latents must have at least 1 frame"
    N = denoiser.cfg.denoiser.num_noise_levels
    assert N % F == 0, "num_noise_levels must be divisible by window_size"

    stride       = N // F
    step_size    = 1.0 / F
    tau_cond_idx = get_noise_index(context_cond_tau, N)
    clean_idx    = N - 1

    # Prefix starts with whatever the caller supplied (up to C frames) and grows
    # toward C as emissions accumulate; once full it rolls.
    if C > 0:
        prefix_obs = init_latents[:, -min(C, T_init):].clone()
        prefix_act = init_actions[:, -min(C, T_init):].clone()
    else:
        prefix_obs = prefix_act = None

    is_demo_emit = [True] * demo_len + [False] * play_pred_steps
    next_emit = [0]

    def _make_slot():
        i = next_emit[0]
        is_demo = is_demo_emit[i]
        z_o = torch.randn(B, 1, N_lat, D_lat, device=device, dtype=dtype)
        if is_demo:
            z_a = demo_actions[:, i:i + 1].clone()
        else:
            z_a = action_noise_std * torch.randn(B, 1, n_act, device=device, dtype=dtype)
        t = torch.zeros((B, 1), dtype=torch.long, device=device)
        next_emit[0] += 1
        return z_o, z_a, t, is_demo

    z_obs, z_act, tau_idx, is_d = _make_slot()
    window_demo = [is_d]
    pred_obs_buf, pred_act_buf, label_buf = [], [], []

    while len(pred_obs_buf) < L:
        cur_F = z_obs.shape[1]
        cur_C = prefix_obs.shape[1] if prefix_obs is not None else 0
        demo_mask = torch.tensor(window_demo, dtype=torch.bool, device=device)

        if cur_C > 0:
            p_obs_in = (1.0 - context_cond_tau) * torch.randn_like(prefix_obs) + context_cond_tau * prefix_obs
            p_act_in = ((1.0 - context_cond_tau) * action_noise_std * torch.randn_like(prefix_act)
                        + context_cond_tau * prefix_act)
            obs_in = torch.cat([p_obs_in, z_obs], dim=1)
            act_in = torch.cat([p_act_in, z_act], dim=1)
            prefix_sigma = torch.full((B, cur_C), tau_cond_idx, dtype=torch.long, device=device)
            obs_sigma_idx = torch.cat([prefix_sigma, tau_idx], dim=1)
        else:
            obs_in, act_in, obs_sigma_idx = z_obs, z_act, tau_idx

        win_act_sigma = torch.where(
            demo_mask.unsqueeze(0).expand(B, -1),
            torch.full_like(tau_idx, clean_idx),
            tau_idx,
        )
        act_sigma_idx = torch.cat([prefix_sigma, win_act_sigma], dim=1) if cur_C > 0 else win_act_sigma

        step_idx_tensor = torch.zeros((B, cur_C + cur_F), dtype=torch.long, device=device)

        z_hat, act_hat = denoiser(
            noisy_act    = act_in,
            noisy_obs    = obs_in,
            obs_sigma_idx= obs_sigma_idx,
            obs_step_idx = step_idx_tensor,
            act_sigma_idx= act_sigma_idx,
            act_step_idx = step_idx_tensor,
        )
        act_hat = act_hat.squeeze(-2)
        z_hat_w, act_hat_w = z_hat[:, cur_C:], act_hat[:, cur_C:]

        tau_cont = tau_idx.float() / float(N)
        denom    = (1.0 - tau_cont).clamp_min(1e-5)
        z_obs = z_obs + (z_hat_w - z_obs) / denom[..., None, None] * step_size
        play_mask_f = (~demo_mask).to(dtype).view(1, cur_F, 1)
        z_act = z_act + (act_hat_w - z_act) / denom[..., None] * step_size * play_mask_f
        tau_idx = tau_idx + stride

        if int(tau_idx[0, 0].item()) >= N:
            emit_obs, emit_act = z_obs[:, :1], z_act[:, :1]
            pred_obs_buf.append(emit_obs)
            pred_act_buf.append(emit_act)
            label_buf.append(int(window_demo[0]))
            z_obs, z_act, tau_idx = z_obs[:, 1:], z_act[:, 1:], tau_idx[:, 1:]
            window_demo = window_demo[1:]
            if C > 0:
                # grow prefix toward C, then roll
                if prefix_obs.shape[1] < C:
                    prefix_obs = torch.cat([prefix_obs, emit_obs], dim=1)
                    prefix_act = torch.cat([prefix_act, emit_act], dim=1)
                else:
                    prefix_obs = torch.cat([prefix_obs[:, 1:], emit_obs], dim=1)
                    prefix_act = torch.cat([prefix_act[:, 1:], emit_act], dim=1)

        if z_obs.shape[1] < F and next_emit[0] < L:
            new_o, new_a, new_t, new_d = _make_slot()
            z_obs   = torch.cat([z_obs,   new_o], dim=1)
            z_act   = torch.cat([z_act,   new_a], dim=1)
            tau_idx = torch.cat([tau_idx, new_t], dim=1)
            window_demo.append(new_d)

    pred_obs = torch.cat(pred_obs_buf, dim=1)
    pred_act = torch.cat(pred_act_buf, dim=1)
    labels   = torch.tensor(label_buf, dtype=torch.long, device=device)
    return pred_obs, pred_act, labels


# ---------------------------------------------------------------------------
# Per-episode generation + shard writer
# ---------------------------------------------------------------------------

def generate_episode(
    src_dataset,
    denoiser,
    tokenizer,
    n_actions,
    episode_length,
    window_size,
    context_size,
    context_cond_tau,
    action_noise_std,
    resolution,
    rng,
    device,
    autocast_dtype,
):
    """Generate one synthetic episode. Returns (images_uint8, actions_f32, is_demo_u8)."""
    total_emit = episode_length + 1
    demo_len   = int(rng.integers(1, total_emit))   # ∈ [1, episode_length]
    play_pred  = total_emit - demo_len

    sample_idx = int(rng.integers(0, len(src_dataset)))
    sample = src_dataset[sample_idx]
    src_imgs = sample['image']                              # (T, C, H, W) float in [0,1]
    src_acts = sample['action'][:, :n_actions]              # (T, n_act)

    src_imgs = interpolate(src_imgs.to(device=device, dtype=torch.float32),
                           tuple(resolution)).unsqueeze(0)   # (1, T, C, H, W)
    src_acts = src_acts.unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, T, n_act)

    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
        with torch.no_grad():
            src_latents = tokenizer.encode(src_imgs)        # (1, T, N_lat, D_lat) bf16

            init_latents = src_latents[:, :1]                # 1 prefix frame
            init_actions = src_acts[:, :1]
            demo_acts_in = src_acts[:, 1:1 + demo_len]       # actions paired with demo emissions

            pred_obs, pred_act, labels = hybrid_progressive_sampler(
                denoiser,
                init_latents     = init_latents,
                init_actions     = init_actions,
                demo_actions     = demo_acts_in,
                play_pred_steps  = play_pred,
                window_size      = window_size,
                context_size     = context_size,
                context_cond_tau = context_cond_tau,
                action_noise_std = action_noise_std,
            )
            pred_imgs = tokenizer.decode(pred_obs)           # (1, total_emit, C, H, W)

    # Drop emission 0 — the seam frame against the raw clean prefix.
    imgs_t = pred_imgs[0, 1:].clamp(0, 1)                    # (T, C, H, W)
    acts_t = pred_act[0, 1:]                                 # (T, n_act)
    lbls_t = labels[1:]                                      # (T,)

    imgs_np = imgs_t.permute(0, 2, 3, 1).to(torch.float32).cpu().numpy()
    imgs_np = (imgs_np * 255.0).astype(np.uint8)
    acts_np = acts_t.to(torch.float32).cpu().numpy()
    lbls_np = lbls_t.to(torch.uint8).cpu().numpy()
    return imgs_np, acts_np, lbls_np


def write_shard(shard_path, episodes):
    """Write a list of (images_uint8, actions_f32, is_demo_u8) to one h5 shard."""
    num_eps = len(episodes)
    T, H, W, C = episodes[0][0].shape
    n_act = episodes[0][1].shape[-1]

    images_arr  = np.stack([ep[0] for ep in episodes], axis=0)   # (E, T, H, W, C)
    actions_arr = np.stack([ep[1] for ep in episodes], axis=0)   # (E, T, n_act) float32
    is_demo_arr = np.stack([ep[2] for ep in episodes], axis=0)   # (E, T) uint8
    ep_lengths  = np.full((num_eps,), T, dtype=np.int32)

    comp = {'compression': 'gzip', 'compression_opts': 4}
    chunk_t = min(T, 8)
    with h5py.File(shard_path, 'w') as f:
        f.create_dataset('images',          data=images_arr,
                         chunks=(1, chunk_t, H, W, C), **comp)
        f.create_dataset('actions',         data=actions_arr,
                         chunks=(1, chunk_t, n_act), **comp)
        f.create_dataset('actions_rel',     data=actions_arr,
                         chunks=(1, chunk_t, n_act), **comp)
        f.create_dataset('is_demo',         data=is_demo_arr,
                         chunks=(1, chunk_t), dtype='uint8', **comp)
        f.create_dataset('episode_lengths', data=ep_lengths)
        f.attrs['num_episodes'] = num_eps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # I/O
    parser.add_argument('--src_data_dir',      type=str, required=True,
                        help="Source demo dataset (sharded HDF5).")
    parser.add_argument('--output_dir',        type=str, required=True)
    parser.add_argument('--config_path',       type=str,
                        default=str(Path(__file__).resolve().parent / 'config'),
                        help="Hydra config dir (must be absolute).")
    parser.add_argument('--config_name',       type=str, default='dynamics/pushT.yaml')
    parser.add_argument('--dynamics_ckpt',     type=str, required=True)
    parser.add_argument('--tokenizer_ckpt',    type=str, required=True)

    # Output schema
    parser.add_argument('--num_shards',         type=int, default=1)
    parser.add_argument('--episodes_per_shard', type=int, default=100)
    parser.add_argument('--episode_length',     type=int, default=96,
                        help="Stored frames per episode; one extra is generated and dropped.")
    parser.add_argument('--first_shard_index',  type=int, default=0,
                        help="Starting index for shard filenames (e.g. for resumed runs).")

    # Sampler knobs
    parser.add_argument('--window_size',       type=int, default=4,
                        help="F: progressive sampler window (= #steps to fully clean a frame).")
    parser.add_argument('--context_size',      type=int, default=28,
                        help="C: clean overlap-conditioning prefix length. Default C+F=32 = WM context cap.")
    parser.add_argument('--context_cond_tau',  type=float, default=0.9)
    parser.add_argument('--action_noise_std',  type=float, default=1.0)

    # Source-dataset reader
    parser.add_argument('--src_split',          type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--src_train_fraction', type=float, default=0.9)
    parser.add_argument('--src_split_seed',     type=int, default=123)
    parser.add_argument('--src_window_size',    type=int, default=None,
                        help="Source snippet length. If unset, uses episode_length + 1 "
                             "(1 prefix + up to episode_length demo actions).")

    # Misc
    parser.add_argument('--resolution',        type=int, nargs=2, default=(256, 256))
    parser.add_argument('--device',            type=str, default='cuda:0')
    parser.add_argument('--seed',              type=int, default=0)
    parser.add_argument('--max_num_forward_steps', type=int, default=300,
                        help="Passed to load_denoiser/load_tokenizer.")

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    autocast_dtype = torch.bfloat16

    # --- Hydra config ---
    cfg_dir = Path(args.config_path).resolve()
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=args.config_name)
    cfg.dynamics_ckpt = args.dynamics_ckpt
    cfg.tokenizer_ckpt = args.tokenizer_ckpt
    n_actions = cfg.denoiser.n_actions

    # --- Models ---
    print(f"Loading denoiser  from {args.dynamics_ckpt}")
    denoiser  = load_denoiser(cfg, device, max_num_forward_steps=args.max_num_forward_steps).eval().to(device)
    print(f"Loading tokenizer from {args.tokenizer_ckpt}")
    tokenizer = load_tokenizer(cfg, device, max_num_forward_steps=args.max_num_forward_steps).eval().to(device)

    # --- Source dataset ---
    src_window_size = args.src_window_size or (args.episode_length + 1)
    print(f"Loading source dataset {args.src_data_dir} "
          f"(split={args.src_split}, window={src_window_size})")
    src_dataset = ShardedHDF5Dataset(
        data_dir       = args.src_data_dir,
        window_size    = src_window_size,
        stride         = 1,
        split          = args.src_split,
        train_fraction = args.src_train_fraction,
        split_seed     = args.src_split_seed,
        shuffle_windows= True,
    )
    if len(src_dataset) == 0:
        raise RuntimeError(f"Source dataset is empty for window_size={src_window_size}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --- Generation loop ---
    total_eps = args.num_shards * args.episodes_per_shard
    pbar = tqdm(total=total_eps, desc="Episodes")

    image_shape, action_shape = None, None
    for shard_i in range(args.num_shards):
        shard_path = output_path / f'shard_{args.first_shard_index + shard_i:04d}.h5'
        eps = []
        for _ in range(args.episodes_per_shard):
            imgs, acts, is_demo = generate_episode(
                src_dataset,
                denoiser,
                tokenizer,
                n_actions        = n_actions,
                episode_length   = args.episode_length,
                window_size      = args.window_size,
                context_size     = args.context_size,
                context_cond_tau = args.context_cond_tau,
                action_noise_std = args.action_noise_std,
                resolution       = args.resolution,
                rng              = rng,
                device           = device,
                autocast_dtype   = autocast_dtype,
            )
            eps.append((imgs, acts, is_demo))
            if image_shape is None:
                image_shape  = list(imgs.shape[1:])
                action_shape = list(acts.shape[1:])
            pbar.update(1)
        write_shard(shard_path, eps)
    pbar.close()

    # --- Metadata (mirror the source where possible) ---
    src_meta = {}
    src_meta_path = Path(args.src_data_dir) / 'metadata.json'
    if src_meta_path.exists():
        with open(src_meta_path) as f:
            src_meta = json.load(f)

    metadata = {
        'num_shards':       args.num_shards,
        'total_episodes':   total_eps,
        'image_shape':      image_shape,
        'action_shape':     action_shape,
        'relative_actions': True,
        'delta_direction':  src_meta.get('delta_direction', 'future'),
        'dt_source':        src_meta.get('dt_source', 'nominal'),
        # Provenance
        'synthetic':        True,
        'src_data_dir':     args.src_data_dir,
        'src_split':        args.src_split,
        'episode_length':   args.episode_length,
        'window_size':      args.window_size,
        'context_size':     args.context_size,
        'dynamics_ckpt':    args.dynamics_ckpt,
        'tokenizer_ckpt':   args.tokenizer_ckpt,
        'seed':             args.seed,
    }
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Wrote {args.num_shards} shards x {args.episodes_per_shard} episodes -> {output_path}")


if __name__ == '__main__':
    main()
