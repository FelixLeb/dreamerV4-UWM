#!/usr/bin/env python3
"""
Convert Hugging Face / LeRobot datasets to sharded HDF5.
Features:
- Action deltas paired with the image at t. Direction selectable via
  --delta_direction:
    * future: a_{t+k} - a_t  — standard convention for world models and
      action-chunking policies.
    * past: a_t - a_{t-k}  — legacy convention.
- Delta normalization selectable via --dt_source:
    * nominal: divide by 1/target_fps. Stable; matches the inference rate.
    * timestamp: divide by the recorded inter-frame dt. Most physically
      accurate; reflects real recording jitter.
    * none: store raw deltas, no division. Let downstream normalization handle
      scale (Diffusion-Policy style).
- Resizes images & Adjusts FPS
- Sharded HDF5 output optimized for HPC dataloading
"""

import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

import json
import numpy as np
import h5py
import torch
from pathlib import Path
from tqdm import tqdm
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Subset
import cv2

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: 'lerobot' library not found.")


def write_episode(episode_data, ep_length, output_path):
    with h5py.File(output_path, 'w') as f:
        comp_args = {'compression': 'gzip', 'compression_opts': 4}
        for key in episode_data:
            data = episode_data[key]
            if len(data) > 0:
                data = np.stack(data)
                assert data.shape[0] == ep_length, \
                    f'{key} field in the episode data should have the same number of frames as episode length'

                # Optimized chunking for random access during DL training (e.g., 8-frame sequences)
                chunk_time = min(data.shape[0], 8)
                ds = f.create_dataset(
                    key,
                    shape=(1, *data.shape),
                    dtype=data.dtype,
                    chunks=(1, chunk_time, *data.shape[1:]),
                    **comp_args
                )
                ds[0] = data

        # For compatibility with SOAR
        f.create_dataset('episode_lengths', data=np.array([ep_length], dtype=np.int32))
        f.attrs['num_episodes'] = 1


def _process_sample(sample, args):
    """Decode one DataLoader sample (batch=1) into the per-frame fields we store.

    Returns a dict with the resized HWC uint8 image, the action tensor (no batch
    dim), and any optional state/effort/is_demo fields present in the sample.
    """
    img_tensor = sample[args.image_key].squeeze(0)  # C, H, W
    if args.rectangular:
        C, H, W = img_tensor.shape
        max_side = max(H, W)
        img_padded = torch.zeros((C, max_side, max_side), dtype=img_tensor.dtype)
        r, c = (max_side - H) // 2, (max_side - W) // 2
        img_padded[:, r:r+H, c:c+W] = img_tensor
        img_to_resize = img_padded
    else:
        img_to_resize = img_tensor

    if args.width and args.height:
        resized_img = interpolate(
            img_to_resize.unsqueeze(0).float(),
            size=(args.height, args.width),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    else:
        resized_img = img_to_resize

    resized_img = resized_img.numpy().transpose(1, 2, 0).copy()
    if np.issubdtype(resized_img.dtype, np.floating):
        resized_img = np.clip(resized_img * 255.0, 0, 255)
    resized_img = resized_img.astype(np.uint8)

    if args.visualize:
        cv2.imshow('vis', cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    out = {
        'image': resized_img,
        'action': sample[args.action_key].squeeze(0)[:3], # Fix this hardcoded slicing for 3-DoF actions; ideally should be configurable or inferred from data
    }
    if args.state_key and args.state_key in sample:
        out['state'] = sample[args.state_key].squeeze(0).numpy().copy()
    if args.effort_key and args.effort_key in sample:
        out['effort'] = sample[args.effort_key].squeeze(0).numpy().copy()
    if 'is_demo' in sample:
        out['is_demo'] = sample['is_demo'].squeeze(0).numpy()
    else:
        out['is_demo'] = np.array([int(args.is_demo)])
    if 'timestamp' in sample:
        out['timestamp'] = float(sample['timestamp'].squeeze().item())
    return out


def _append_frame(current_episode, frame, action_rel_np, args):
    """Append one stored frame and its delta to the per-episode buffers."""
    action_np = frame['action'].numpy()
    current_episode['images'].append(frame['image'])
    current_episode['actions_rel'].append(action_rel_np)
    current_episode['actions_abs'].append(action_np)
    current_episode['actions'].append(action_rel_np if args.relative_actions else action_np)
    if 'state' in frame:
        current_episode['states'].append(frame['state'])
    if 'effort' in frame:
        current_episode['efforts'].append(frame['effort'])
    current_episode['is_demo'].append(frame['is_demo'])


def _resolve_dt(args, earlier, later, nominal_dt):
    """Return the divisor to apply to the raw action delta for one step."""
    if args.dt_source == 'none':
        return 1.0
    if args.dt_source == 'timestamp':
        if 'timestamp' not in earlier or 'timestamp' not in later:
            raise KeyError(
                "--dt_source=timestamp requested but the 'timestamp' field is missing "
                "from the dataset samples."
            )
        dt = later['timestamp'] - earlier['timestamp']
        if dt <= 0:
            raise ValueError(
                f"Non-positive dt={dt:.6g}s computed from timestamps "
                f"({earlier['timestamp']:.6g} → {later['timestamp']:.6g}). "
                "Check for clock anomalies or duplicate frames."
            )
        return dt
    return nominal_dt


def convert_dataset(args):
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.repo_id}")
    dataset_kwargs = {'repo_id': args.repo_id, 'video_backend': 'pyav', 'tolerance_s': 0.05}
    if args.local_dir:
        dataset_kwargs['root'] = args.local_dir
    dataset = LeRobotDataset(**dataset_kwargs)

    source_fps = dataset.fps
    target_fps = args.target_fps if args.target_fps else source_fps
    assert target_fps > 0, "Target FPS must be greater than 0"
    assert source_fps >= target_fps, "Target FPS must be less than or equal to Source FPS"
    assert source_fps % target_fps == 0, "Source FPS must be divisible by Target FPS"
    ds_ratio = max(1, int(source_fps // target_fps))

    print(f"FPS: {source_fps} -> {target_fps} (Subsample: {ds_ratio})")

    # Build a flat index list for the DataLoader. For each episode we load
    # N+1 consecutive subsampled frames; we store N of them paired with a delta.
    # In `future` mode the trailing frame is consumed only for the last delta;
    # in `past` mode the leading frame is consumed only to seed the first delta.
    all_frame_indices = []
    episode_meta = []

    ep_number = args.first_episode_number
    for ep in dataset.meta.episodes:
        s = ep['dataset_from_index'] + args.truncate_length
        e = ep['dataset_to_index'] - args.truncate_length
        if s >= e:
            continue

        raw_indices = list(range(s, e, ds_ratio))
        if len(raw_indices) < 2:
            continue

        num_stored = len(raw_indices) - 1
        shard_path = output_path / f'shard_{ep_number:04d}.h5'
        episode_meta.append((shard_path, num_stored))
        all_frame_indices.extend(raw_indices)
        ep_number += 1

    if not all_frame_indices:
        print("Warning: No valid frames found. Skipping metadata.")
        return

    # Single DataLoader over the whole dataset — workers are reused across episodes.
    loader = DataLoader(
        Subset(dataset, all_frame_indices),
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    meta_image_shape = None
    meta_action_shape = None
    actual_episodes = 0
    frame_iter = iter(loader)

    nominal_dt = 1. / target_fps
    for shard_path, num_frames in tqdm(episode_meta, desc="Episodes"):
        current_episode = {
            'images': [], 'actions': [], 'actions_rel': [],
            'actions_abs': [], 'efforts': [], 'states': [], 'is_demo': []
        }

        if args.delta_direction == 'future':
            # Store cur paired with delta = nxt.action - cur.action.
            # The trailing frame is consumed only for the last delta.
            cur = _process_sample(next(frame_iter), args)
            for _ in range(num_frames):
                nxt = _process_sample(next(frame_iter), args)
                step_dt = _resolve_dt(args, cur, nxt, nominal_dt)
                action_rel = (nxt['action'] - cur['action']) / step_dt
                _append_frame(current_episode, cur, action_rel.numpy(), args)
                if meta_image_shape is None:
                    meta_image_shape = cur['image'].shape
                    meta_action_shape = tuple(cur['action'].shape)
                cur = nxt
        else:  # past
            # Store cur paired with delta = cur.action - prev.action.
            # The leading frame is consumed only to seed the first delta.
            prev = _process_sample(next(frame_iter), args)
            for _ in range(num_frames):
                cur = _process_sample(next(frame_iter), args)
                step_dt = _resolve_dt(args, prev, cur, nominal_dt)
                action_rel = (cur['action'] - prev['action']) / step_dt
                _append_frame(current_episode, cur, action_rel.numpy(), args)
                if meta_image_shape is None:
                    meta_image_shape = cur['image'].shape
                    meta_action_shape = tuple(cur['action'].shape)
                prev = cur

        write_episode(current_episode, num_frames, shard_path)
        actual_episodes += 1

    # Save Metadata
    if actual_episodes == 0 or meta_image_shape is None or meta_action_shape is None:
        print("Warning: No valid episodes were written. Skipping metadata.")
        return

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump({
            'num_shards': actual_episodes,
            'total_episodes': actual_episodes,
            'image_shape': meta_image_shape,
            'action_shape': meta_action_shape,
            'relative_actions': args.relative_actions,
            'delta_direction': args.delta_direction,
            'dt_source': args.dt_source,
        }, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--local_dir', type=str, default=None)
    parser.add_argument('--first_episode_number', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader worker count for parallel frame loading")

    # Image Args
    parser.add_argument('--target_fps', type=int, default=10)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--truncate_length', type=int, default=0)
    parser.add_argument('--rectangular', action='store_true', default=False, help="Pad images to square")
    parser.add_argument('--visualize', action='store_true', default=False, help="Visualize the images as they are processed")

    # Action/State Args
    parser.add_argument('--relative_actions', action=argparse.BooleanOptionalAction, default=True, help="Store action deltas (a/k_t differences scaled by dt_target) in the 'actions' field instead of absolute actions. The delta convention is set by --delta_direction.")
    parser.add_argument('--delta_direction', type=str, default='future', choices=['future', 'past'],
                        help="future: delta = a_{t+k} - a_t paired with image at t (recommended for world models / policies). "
                             "past (default): delta = a_t - a_{t-k} paired with image at t (legacy convention).")
    parser.add_argument('--dt_source', type=str, default='timestamp', choices=['nominal', 'timestamp', 'none'],
                        help="nominal (default): divide delta by 1/target_fps. "
                             "timestamp: divide by recorded inter-frame dt (per-step, captures jitter). "
                             "none: store raw deltas with no division.")
    parser.add_argument('--image_key', type=str, default='observation.images.camera_0')
    parser.add_argument('--state_key', type=str, default='observation.state')
    parser.add_argument('--effort_key', type=str, default='observation.effort')
    parser.add_argument('--action_key', type=str, default='action')
    parser.add_argument('--is_demo', action='store_true', help="Flag episode as demo")

    args = parser.parse_args()

    if LEROBOT_AVAILABLE:
        convert_dataset(args)
    else:
        print("Please install 'lerobot'")

if __name__ == "__main__":
    main()
