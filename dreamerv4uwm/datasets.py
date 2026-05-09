import h5py
import torch
import torch.nn.functional as F
import random
import json
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class ShardedHDF5Dataset(Dataset):
    """
    Dataset for sharded HDF5 files optimized for multi-node training.
    Each worker preferentially reads from local shards when available.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int,
        stride: int = 1,
        split: str = "train",          # "train" or "test"
        train_fraction: float = 0.9,   # fraction of episodes in train
        split_seed: int = 42,          # seed for reproducible split
        shuffle_windows: bool = True, 
        ep_count_offset: int = 0,      # offset to add to episode indices (for multi-dataset merging)   
        absolute_actions=False,                 # whether to convert actions to absolute coordinates (deprecated
    ):
        self.shuffle_windows = shuffle_windows
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.split = split
        self.train_fraction = train_fraction
        self.split_seed = split_seed
        self.absolute_actions = absolute_actions

        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        self.num_shards = self.metadata['num_shards']
        self.shard_files = [
            self.data_dir / f"shard_{i+ep_count_offset:04d}.h5"
            for i in range(self.num_shards)
        ]

        # Probe the first shard once for an optional reward field. Datasets
        # without dense reward simply don't return the key — `compute_uwm_loss`
        # raises if `train_reward_model=True` but no rewards are passed in.
        with h5py.File(self.shard_files[0], 'r') as f:
            self.has_rewards = 'is_demo' in f

        # Build window index across all shards
        self.windows = []
        self.episode_lengths = []  # Store all episode lengths for analysis

        for shard_idx, shard_file in enumerate(self.shard_files):
            with h5py.File(shard_file, 'r') as f:
                num_episodes = f.attrs['num_episodes']
                try:
                    lengths = f['episode_lengths'][:]
                except:
                    lengths = np.array([f['episode_lengths'][()]])


                # Store episode lengths for statistics
                self.episode_lengths.extend(lengths.tolist())

                for ep_idx, ep_length in enumerate(lengths):
                    for start in range(0, ep_length - window_size + 1, stride):
                        self.windows.append((shard_idx, ep_idx, start))

        # Collect all (shard_idx, ep_idx) pairs
        all_episodes = sorted({(shard_idx, ep_idx) for shard_idx, ep_idx, _ in self.windows})

        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(all_episodes))

        num_train_eps = int(self.train_fraction * len(all_episodes))
        train_eps = {all_episodes[i] for i in perm[:num_train_eps]}
        test_eps  = {all_episodes[i] for i in perm[num_train_eps:]}

        self.split_info = {
            "train_episodes": sorted(list(train_eps)),
            "test_episodes": sorted(list(test_eps)),
        }

        if self.split == "train":
            keep = train_eps
        elif self.split == "test":
            keep = test_eps
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Filter windows based on chosen split
        self.windows = [w for w in self.windows if (w[0], w[1]) in keep]
        print(f"{self.split.capitalize()} split: {len(self.windows)} windows "
              f"from {len(keep)} episodes")
        if self.shuffle_windows:
            random.shuffle(self.windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        shard_idx, ep_idx, start = self.windows[idx]
        end = start + self.window_size

        shard_file = self.shard_files[shard_idx]

        # Open HDF5 file (each worker maintains its own handle)
        with h5py.File(shard_file, 'r') as f:
            images = f['images'][ep_idx, start:end]
            try:
                if self.absolute_actions:
                    actions = f['actions_abs'][ep_idx, start:end]
                else:
                    actions = f['actions_rel'][ep_idx, start:end]
            except:
                actions = f['actions'][ep_idx, start:end]
            rewards = f['is_demo'][ep_idx, start:end] if self.has_rewards else None

        # Convert to PyTorch
        images = torch.from_numpy(images).float() / 255.0
        images = images.permute(0, 3, 1, 2)
        actions = torch.from_numpy(actions)

        out = {'image': images, 'action': actions}
        if rewards is not None:
            out['is_demo'] = torch.from_numpy(rewards).float()
        return out


    def get_episode_length_statistics(self):
        """
        Calculate comprehensive statistics about episode lengths.
        
        Returns:
            dict with statistics about episode lengths
        """
        lengths = np.array(self.episode_lengths)
        
        stats = {
            'total_episodes': len(lengths),
            'total_timesteps': int(np.sum(lengths)),
            'min_length': int(np.min(lengths)),
            'max_length': int(np.max(lengths)),
            'mean_length': float(np.mean(lengths)),
            'median_length': float(np.median(lengths)),
            'std_length': float(np.std(lengths)),
            'percentile_25': float(np.percentile(lengths, 25)),
            'percentile_75': float(np.percentile(lengths, 75)),
            'percentile_90': float(np.percentile(lengths, 90)),
            'percentile_95': float(np.percentile(lengths, 95)),
            'percentile_99': float(np.percentile(lengths, 99)),
        }
        
        return stats

class G1Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        window_size: int,
        stride: int = 1,
        shuffle_windows: bool = True,
        shard_glob: str = "*.h5",
        img_key='images',
        action_key=None,
        state_key=None,
        reward_key=None,
        is_done_key=None,
        bgr_to_rgb: bool = False,
        overlapping = True,
        pushT_backward_compatibility = False, # A temporary hack to support the old pushT dataset which had a non-standard layout and naming convention. Ignored if the dataset doesn't match that layout.
        image_size=None,                       # Optional (H, W) to resize images to via F.interpolate
        resize_mode: str = "bilinear",         # Interpolation mode used when image_size is set
    ):
        self.pushT_backward_compatibility = pushT_backward_compatibility
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.shuffle_windows = shuffle_windows
        self.img_key = img_key
        self.action_key = action_key
        self.state_key = state_key
        self.reward_key = reward_key
        self.is_done_key = is_done_key
        self.bgr_to_rgb = bgr_to_rgb
        self.image_size = tuple(image_size) if image_size is not None else None
        self.resize_mode = resize_mode
        self.keys = [k for k in (img_key, action_key, state_key, reward_key, is_done_key) if k is not None]
        if img_key is not None:
            len_key = img_key
        elif action_key is not None:
            len_key = action_key
        
        # Auto-discover shards (sorted by filename for determinism)
        self.shard_files = sorted(self.data_dir.glob(shard_glob))
        if not self.shard_files:
            raise FileNotFoundError(
                f"No shards matched {shard_glob!r} under {self.data_dir}"
            )
        self.num_shards = len(self.shard_files)
        self.episodes = []
        for shard_idx, shard_file in enumerate(self.shard_files):
            with h5py.File(shard_file, 'r') as f:
                if img_key is not None:
                    assert img_key in f.keys(), f"Shard {shard_file} is missing required key {img_key}. Available keys: {list(f.keys())}"
                if action_key is not None:
                    assert action_key in f.keys(), f"Shard {shard_file} is missing required key {action_key}. Available keys: {list(f.keys())}"
                if state_key is not None:
                    assert state_key in f.keys(), f"Shard {shard_file} is missing required key {state_key}. Available keys: {list(f.keys())}"
                if reward_key is not None:
                    assert reward_key in f.keys(), f"Shard {shard_file} is missing required key {reward_key}. Available keys: {list(f.keys())}"
                if pushT_backward_compatibility:
                    ep_len = f[len_key].shape[1] 
                else:
                    ep_len = f[len_key].shape[0]   
                self.episodes.append((shard_idx, ep_len))
        self.windows = []  
        for shard_idx, length in self.episodes:
            if overlapping:
                starts = torch.arange(0, length - window_size + 1, 1)
                ends = starts + window_size
                idx = torch.ones(len(starts), dtype=torch.long) * shard_idx
                w = torch.stack((idx, starts, ends), dim=1)
                self.windows.append(w)
            else:
                starts = torch.arange(0, length - window_size + 1, stride)
                ends = starts + window_size
                idx = torch.ones(len(starts), dtype=torch.long) * shard_idx
                w = torch.stack((idx, starts, ends), dim=1)
                self.windows.append(w)
        self.windows = torch.vstack(self.windows).tolist() 
        if self.shuffle_windows:
            random.shuffle(self.windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        shard_idx, start, end= self.windows[idx]
        out = {}
        with h5py.File(self.shard_files[shard_idx], 'r') as f:
            for k in self.keys:
                if self.pushT_backward_compatibility:
                    arr = f[k][0, start:end]
                else:
                    arr = f[k][start:end]
                if k==self.img_key:
                    if self.bgr_to_rgb and arr.ndim >= 1 and arr.shape[-1] == 3:
                        arr = arr[..., ::-1]
                    t = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
                    if t.ndim == 4:  # (T, H, W, C) → (T, C, H, W)
                        t = t.permute(0, 3, 1, 2).contiguous()
                    if self.image_size is not None:
                        align = None if self.resize_mode in ("nearest", "area") else False
                        if align is None:
                            t = F.interpolate(t, size=self.image_size, mode=self.resize_mode)
                        else:
                            t = F.interpolate(t, size=self.image_size, mode=self.resize_mode, align_corners=align)
                else:
                    t = torch.from_numpy(np.ascontiguousarray(arr))
                if self.pushT_backward_compatibility and k == self.action_key:
                    out[k] = t[:, :2]
                else:
                    out[k] = t
        return out

def create_distributed_dataloader(
    data_dir: str,
    window_size: int,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 4,
    stride: int = 1,
    seed: int = 42,
    split: str = "train",
    train_fraction: float = 0.9,
    split_seed: int = 42,
    shuffle: bool = True,
    drop_last: bool = True,
    absolute_actions: bool = False,
):
    """
    Create DataLoader with DistributedSampler for sharded HDF5 dataset.
    """
    # Create the dataset with a fixed split
    dataset = ShardedHDF5Dataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        split=split,
        train_fraction=train_fraction,
        split_seed=split_seed,
        absolute_actions=absolute_actions,
    )

    # Create DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataloader, sampler, dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze ShardedHDF5Dataset and print statistics"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/scratch/ja5009/soar_data_sharded/',
        help='Path to sharded HDF5 directory'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=96,
        help='Window size for sliding windows'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Stride for sliding windows'
    )
    parser.add_argument(
        '--show_histogram',
        action='store_true',
        help='Show histogram of episode lengths (requires matplotlib)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("ShardedHDF5Dataset Analysis")
    print("="*70)
    print(f"\nLoading dataset from: {args.data_dir}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}\n")
    
    # Create dataset
    dataset = ShardedHDF5Dataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
    )
    
    # Get statistics
    stats = dataset.get_episode_length_statistics()
    
    print("\n" + "="*70)
    print("Episode Length Statistics")
    print("="*70)
    print(f"Total Episodes:           {stats['total_episodes']:,}")
    print(f"Total Timesteps:          {stats['total_timesteps']:,}")
    print(f"Total Windows:            {len(dataset):,}")
    print(f"\nLength Statistics:")
    print(f"  Min:                    {stats['min_length']:.0f} steps")
    print(f"  Max:                    {stats['max_length']:.0f} steps")
    print(f"  Mean:                   {stats['mean_length']:.2f} steps")
    print(f"  Median:                 {stats['median_length']:.2f} steps")
    print(f"  Std Dev:                {stats['std_length']:.2f} steps")
    print(f"\nPercentiles:")
    print(f"  25th percentile:        {stats['percentile_25']:.0f} steps")
    print(f"  75th percentile:        {stats['percentile_75']:.0f} steps")
    print(f"  90th percentile:        {stats['percentile_90']:.0f} steps")
    print(f"  95th percentile:        {stats['percentile_95']:.0f} steps")
    print(f"  99th percentile:        {stats['percentile_99']:.0f} steps")
    
    # Calculate storage efficiency
    avg_windows_per_episode = len(dataset) / stats['total_episodes']
    print(f"\nDataset Efficiency:")
    print(f"  Avg windows per episode: {avg_windows_per_episode:.2f}")
    print(f"  Window coverage:         {avg_windows_per_episode * args.stride / stats['mean_length'] * 100:.1f}%")
    
    # Shard information
    print(f"\nShard Information:")
    print(f"  Number of shards:        {dataset.num_shards}")
    print(f"  Avg episodes per shard:  {stats['total_episodes'] / dataset.num_shards:.1f}")
    
    # Calculate approximate memory usage per batch
    if 'image_shape' in dataset.metadata:
        img_shape = dataset.metadata['image_shape']
        bytes_per_window = (
            args.window_size * img_shape[0] * img_shape[1] * img_shape[2] * 4  # float32
        )
        print(f"\nMemory Usage (per window):")
        print(f"  Image shape:             {img_shape}")
        print(f"  Bytes per window:        {bytes_per_window / (1024**2):.2f} MB")
        print(f"  Batch of 5 windows:      {5 * bytes_per_window / (1024**2):.2f} MB")
    
    print("\n" + "="*70)
    
    # Optional: Show histogram
    if args.show_histogram:
        try:
            import matplotlib.pyplot as plt
            
            lengths = np.array(dataset.episode_lengths)
            
            plt.figure(figsize=(12, 6))
            
            # Histogram
            plt.subplot(1, 2, 1)
            plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
            plt.axvline(stats['mean_length'], color='r', linestyle='--', 
                       label=f"Mean: {stats['mean_length']:.1f}")
            plt.axvline(stats['median_length'], color='g', linestyle='--', 
                       label=f"Median: {stats['median_length']:.1f}")
            plt.xlabel('Episode Length (timesteps)')
            plt.ylabel('Frequency')
            plt.title('Episode Length Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Box plot
            plt.subplot(1, 2, 2)
            plt.boxplot(lengths, vert=True)
            plt.ylabel('Episode Length (timesteps)')
            plt.title('Episode Length Box Plot')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            output_file = Path(args.data_dir) / 'episode_length_analysis.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nHistogram saved to: {output_file}")
            
            plt.show()
            
        except ImportError:
            print("\nWarning: matplotlib not installed. Cannot show histogram.")
            print("Install with: pip install matplotlib")
    
    # Test loading a sample
    print("\nTesting data loading...")
    try:
        sample = dataset[0]
        print(f"  Sample shapes:")
        print(f"    Images: {sample['image'].shape}")
        print(f"    Actions: {sample['action'].shape}")
        print(f"  Sample dtypes:")
        print(f"    Images: {sample['image'].dtype}")
        print(f"    Actions: {sample['action'].dtype}")
        print(f"  Image value range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print("  ✓ Data loading successful!")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
    
    print("\n" + "="*70)
