import h5py
import torch
import random
import json
from functools import partial
from torch.utils.data import DataLoader, Sampler
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

        # Build window index across all shards
        self.windows = []
        self.episode_lengths = []  # Store all episode lengths for analysis
        self.has_rewards = False

        for shard_idx, shard_file in enumerate(self.shard_files):
            with h5py.File(shard_file, 'r') as f:
                num_episodes = f.attrs['num_episodes']
                if shard_idx == 0:
                    self.has_rewards = 'rewards' in f
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
            rewards = f['rewards'][ep_idx, start:end] if self.has_rewards else None

        # Convert to PyTorch
        images = torch.from_numpy(images).float() / 255.0
        images = images.permute(0, 3, 1, 2)
        actions = torch.from_numpy(actions)

        out = {'image': images, 'action': actions}
        if rewards is not None:
            out['reward'] = torch.from_numpy(rewards).float()
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


class G1ChunkDataset(Dataset):
    """
    Dataset over the G1 world-model chunked HDF5 layout
    (`/scratch/rk4342/datasets/G1/wm/`).

    Layout differs from `ShardedHDF5Dataset`: each `chunk_XXXX.h5` is a flat
    per-frame store (no episode dim, no `episode_lengths`) with `images
    (T,256,256,3) uint8 BGR`, `commands (T,22) float32`, `dones (T,) float32`,
    and `source_rrd (T,) vlen utf-8`. Episode boundaries are inferred per-chunk
    from `dones==1` (last frame of a source `.rrd`) and `source_rrd`
    transitions. We do not stitch episodes across chunk seams — windows that
    would cross a chunk boundary are dropped (the README notes this loses at
    most ~228 * window_size frames, negligible vs. 593k total).

    Returns `{'image': (T,3,H,W) float in [0,1] RGB, 'action': (T, 22)}`.
    BGR→RGB flip happens here. Action is the full 22-dim `commands` vector;
    downstream loss code crops to `cfg.denoiser.n_actions` if needed.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int,
        stride: int = 1,
        split: str = "train",
        train_fraction: float = 0.9,
        split_seed: int = 42,
        shuffle_windows: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.split = split
        self.shuffle_windows = shuffle_windows

        self.chunk_paths = sorted(self.data_dir.glob("chunk_*.h5"))
        if not self.chunk_paths:
            raise FileNotFoundError(
                f"No chunk_*.h5 files found in {self.data_dir}"
            )

        # Build a flat episode list across all chunks. Each episode is a
        # contiguous run of frames within a single chunk that shares the
        # same source_rrd and ends at either dones==1 or the last frame
        # before a source_rrd transition.
        all_episodes = []  # list of (chunk_idx, start_t, end_t_exclusive)
        for chunk_idx, p in enumerate(self.chunk_paths):
            with h5py.File(p, "r") as f:
                T = f["dones"].shape[0]
                if T == 0:
                    continue
                dones = f["dones"][:]
                source_rrd = f["source_rrd"][:]

            ep_start = 0
            for t in range(T):
                is_done = dones[t] > 0.5
                next_changes_rrd = (t + 1 < T) and (
                    source_rrd[t + 1] != source_rrd[t]
                )
                if is_done or next_changes_rrd or t == T - 1:
                    all_episodes.append((chunk_idx, ep_start, t + 1))
                    ep_start = t + 1

        # Reproducible per-episode train/test split.
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(len(all_episodes))
        num_train = int(train_fraction * len(all_episodes))
        if split == "train":
            keep_idx = set(perm[:num_train].tolist())
        elif split == "test":
            keep_idx = set(perm[num_train:].tolist())
        else:
            raise ValueError(f"Unknown split: {split}")

        # Build window list across kept episodes.
        self.windows = []  # list of (chunk_idx, t0)
        self.episode_lengths = []
        kept_ep_count = 0
        for ep_i, (chunk_idx, e_start, e_end) in enumerate(all_episodes):
            ep_len = e_end - e_start
            self.episode_lengths.append(ep_len)
            if ep_i not in keep_idx:
                continue
            kept_ep_count += 1
            for t0 in range(e_start, e_end - window_size + 1, stride):
                self.windows.append((chunk_idx, t0))

        if self.shuffle_windows:
            random.shuffle(self.windows)

        self.has_rewards = False
        print(
            f"G1[{split}]: {len(self.windows)} windows from {kept_ep_count} "
            f"episodes (of {len(all_episodes)} total) across "
            f"{len(self.chunk_paths)} chunks"
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        chunk_idx, t0 = self.windows[idx]
        path = self.chunk_paths[chunk_idx]
        end = t0 + self.window_size

        with h5py.File(
            path, "r", rdcc_nbytes=128 * 1024 * 1024, rdcc_nslots=int(1e6)
        ) as f:
            images = f["images"][t0:end]            # (T, H, W, 3) uint8 BGR
            commands = f["commands"][t0:end]        # (T, 22) float32
            proprio_np = f["proprio"][t0:end] if "proprio" in f else None  # (T, 50) float32

        # BGR -> RGB; .copy() to drop the negative-stride view torch can't take.
        images = images[..., ::-1].copy()
        images = torch.from_numpy(images).float().div_(255.0)
        images = images.permute(0, 3, 1, 2)
        actions = torch.from_numpy(commands.copy())

        out = {"image": images, "action": actions}
        if proprio_np is not None:
            out["proprio"] = torch.from_numpy(proprio_np.copy())
        return out

    def get_episode_length_statistics(self):
        lengths = np.array(self.episode_lengths)
        return {
            "total_episodes": int(len(lengths)),
            "total_timesteps": int(np.sum(lengths)),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "mean_length": float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
            "std_length": float(np.std(lengths)),
        }


class MixedDemoPlayDataset(Dataset):
    """50/50 mix of demo + play windows with synthesized per-frame rewards.

    Holds two `ShardedHDF5Dataset` instances (one demo, one play). Each
    `__getitem__` flips a fair coin and draws a window from the chosen
    subset; the returned dict is augmented with a synthesized `'reward'`
    tensor of shape `(window_size,)`:

        - play  → `play_reward` everywhere (default 0.0).
        - demo  → `demo_reward` everywhere (default 1.0), except the last
                  `goal_frames` frames of the source episode → `goal_reward`
                  (default 20.0).

    Goal frames are computed in *absolute episode frame* coords using
    `episode_lengths`, so the mask is correct even if a shard pads multiple
    episodes to a common max length.

    Demo-window sampling has a `terminal_bias` knob (default 0.5): with that
    probability, the demo branch picks a window whose `start` equals the
    terminal start (`ep_length - window_size`), guaranteeing the goal frames
    are present in the window. Without this bias, in long demo episodes the
    `+goal_reward` signal is heavily diluted (e.g. `window_size=64`, demo
    length 150 → only ~13% of random window starts overlap the last 5
    frames). The remaining `1 - terminal_bias` of demo draws are uniformly
    random over the demo's windows, which preserves coverage of mid-episode
    states and prevents the head from collapsing onto pure positional cues.

    This class is a stop-gap label-free signal for fine-tuning the reward MTP
    head before real per-frame rewards are written into the shards. The
    play=0 / demo=1 prior actively biases the head against high-reward
    predictions on play-distribution states, so treat early reward-loss
    curves as "the pipeline works", not "the head learned reward."
    """

    def __init__(
        self,
        demo_data_dir: str,
        play_data_dir: str,
        window_size: int,
        stride: int = 1,
        split: str = "train",
        train_fraction: float = 0.9,
        split_seed: int = 42,
        shuffle_windows: bool = True,
        absolute_actions: bool = False,
        goal_reward: float = 20.0,
        demo_reward: float = 1.0,
        play_reward: float = 0.0,
        goal_frames: int = 5,
        terminal_bias: float = 0.5,
    ):
        assert 0.0 <= terminal_bias <= 1.0, (
            f"terminal_bias must be in [0, 1], got {terminal_bias}"
        )
        common = dict(
            window_size=window_size,
            stride=stride,
            split=split,
            train_fraction=train_fraction,
            split_seed=split_seed,
            shuffle_windows=shuffle_windows,
            absolute_actions=absolute_actions,
        )
        self.demo = ShardedHDF5Dataset(data_dir=demo_data_dir, **common)
        self.play = ShardedHDF5Dataset(data_dir=play_data_dir, **common)
        self.window_size = int(window_size)
        self.goal_reward = float(goal_reward)
        self.demo_reward = float(demo_reward)
        self.play_reward = float(play_reward)
        self.goal_frames = int(goal_frames)
        self.terminal_bias = float(terminal_bias)

        # (shard_idx, ep_idx) -> ep_length lookup for the demo dataset, used
        # to identify goal frames in absolute-frame coords.
        self._demo_ep_length: dict = {}
        for shard_idx, shard_file in enumerate(self.demo.shard_files):
            with h5py.File(shard_file, 'r') as f:
                try:
                    lengths = f['episode_lengths'][:]
                except Exception:
                    lengths = np.array([f['episode_lengths'][()]])
            for ep_idx, L in enumerate(lengths):
                self._demo_ep_length[(int(shard_idx), int(ep_idx))] = int(L)

        # Pre-index the "terminal" windows of the demo dataset — those whose
        # `start == ep_length - window_size`, i.e. the window covers exactly
        # the last `window_size` frames and is guaranteed to contain the
        # goal-frame slice. Used by the terminal-bias sampling branch.
        self._demo_terminal_indices = [
            i
            for i, (shard_idx, ep_idx, start) in enumerate(self.demo.windows)
            if int(start)
            == self._demo_ep_length[(int(shard_idx), int(ep_idx))] - self.window_size
        ]

        # Always emits 'reward' — the training script keys off batch contents
        # to decide whether to call the reward loss.
        self.has_rewards = True

    def __len__(self):
        return len(self.demo) + len(self.play)

    def __getitem__(self, idx):
        # 50/50 random per-call mixing — idx is consumed by the sampler /
        # DataLoader iteration count but the actual window choice is
        # randomized within the worker.
        if random.random() < 0.5:
            # Demo branch: with probability `terminal_bias`, pick a window
            # that's guaranteed to contain the goal frames; otherwise sample
            # uniformly. Falls back to uniform if there are no terminal
            # windows in the kept split (shouldn't happen in practice).
            if (
                self._demo_terminal_indices
                and random.random() < self.terminal_bias
            ):
                inner_idx = random.choice(self._demo_terminal_indices)
            else:
                inner_idx = random.randrange(len(self.demo))
            sample = self.demo[inner_idx]
            shard_idx, ep_idx, start = self.demo.windows[inner_idx]
            ep_length = self._demo_ep_length[(int(shard_idx), int(ep_idx))]
            t = torch.arange(self.window_size, dtype=torch.long)
            abs_frame = t + int(start)
            is_goal = abs_frame >= (ep_length - self.goal_frames)
            reward = torch.full(
                (self.window_size,), self.demo_reward, dtype=torch.float32,
            )
            reward[is_goal] = self.goal_reward
        else:
            inner_idx = random.randrange(len(self.play))
            sample = self.play[inner_idx]
            reward = torch.full(
                (self.window_size,), self.play_reward, dtype=torch.float32,
            )
        sample['reward'] = reward
        return sample


class MultiShardedHDF5Dataset(Dataset):
    """Concat of N `ShardedHDF5Dataset` subsets, one per `data_dirs[i]`.

    Each subset performs its own per-dataset train/test split with the shared
    `split_seed`, then we expose a flat `(subset_idx, local_idx)` window index
    over the kept windows. `sample_weights[i]` gives the per-window probability
    used by `DistributedWeightedSampler` for equal-weight-per-dataset training
    (each dataset contributes a total mass of `1/N` regardless of its size).

    Asserts that all subsets share the same `action_shape` (read from each
    subset's `metadata.json`) so a single denoiser `n_actions` is valid across
    the union.
    """

    def __init__(
        self,
        data_dirs,
        window_size: int,
        stride: int = 1,
        split: str = "train",
        train_fraction: float = 0.9,
        split_seed: int = 42,
        shuffle_windows: bool = False,
        absolute_actions: bool = False,
    ):
        if not data_dirs:
            raise ValueError("MultiShardedHDF5Dataset requires a non-empty data_dirs list")

        self.data_dirs = [Path(p) for p in data_dirs]
        self.window_size = int(window_size)

        # Read each subset's metadata first to assert action-shape uniformity.
        action_shapes = []
        for d in self.data_dirs:
            with open(d / "metadata.json", "r") as f:
                m = json.load(f)
            action_shapes.append(tuple(m.get("action_shape", ())))
        if len(set(action_shapes)) > 1:
            raise ValueError(
                f"MultiShardedHDF5Dataset: heterogeneous action_shape across data_dirs: "
                f"{dict(zip([str(d) for d in self.data_dirs], action_shapes))}"
            )

        # Build subsets. Each does its own split with the shared seed.
        # shuffle_windows is forced off inside subsets — the outer sampler
        # controls ordering for distributed training.
        self.subsets = [
            ShardedHDF5Dataset(
                data_dir=str(d),
                window_size=window_size,
                stride=stride,
                split=split,
                train_fraction=train_fraction,
                split_seed=split_seed,
                shuffle_windows=False,
                absolute_actions=absolute_actions,
            )
            for d in self.data_dirs
        ]

        # Flat (subset_idx, local_idx) index + per-window weights such that
        # total mass per subset == 1/N.
        N = len(self.subsets)
        self.flat_index = []
        weights = []
        for s_idx, sub in enumerate(self.subsets):
            n_i = len(sub)
            if n_i == 0:
                continue
            w_i = 1.0 / (N * n_i)
            self.flat_index.extend((s_idx, j) for j in range(n_i))
            weights.extend([w_i] * n_i)
        self.sample_weights = np.asarray(weights, dtype=np.float64)

        # Plain concat fallback flag (always emit 'reward' iff every subset has it).
        self.has_rewards = all(sub.has_rewards for sub in self.subsets)

        if shuffle_windows:
            order = np.random.default_rng(split_seed).permutation(len(self.flat_index))
            self.flat_index = [self.flat_index[i] for i in order]
            self.sample_weights = self.sample_weights[order]

        # Summary
        per_subset = [(str(d), len(s)) for d, s in zip(self.data_dirs, self.subsets)]
        print(
            f"MultiShardedHDF5Dataset[{split}]: {len(self.flat_index)} windows across "
            f"{N} subsets — " + ", ".join(f"{Path(d).name}:{n}" for d, n in per_subset)
        )

    def __len__(self):
        return len(self.flat_index)

    def __getitem__(self, idx):
        s_idx, local_idx = self.flat_index[idx]
        return self.subsets[s_idx][local_idx]

    def get_episode_length_statistics(self):
        merged = []
        for sub in self.subsets:
            merged.extend(sub.episode_lengths)
        lengths = np.array(merged)
        return {
            "total_episodes": int(len(lengths)),
            "total_timesteps": int(np.sum(lengths)),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "mean_length": float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
            "std_length": float(np.std(lengths)),
        }


class DistributedWeightedSampler(Sampler):
    """Distributed sampler that draws indices proportional to `weights`.

    A single deterministic `torch.multinomial` draw (seeded by `seed + epoch`)
    produces `num_samples` indices, identical across ranks; each rank then
    takes its disjoint slice of length `num_samples // num_replicas`. Call
    `set_epoch(e)` between epochs to reshuffle.

    Standard PyTorch `WeightedRandomSampler` does not interoperate with
    `DistributedSampler` (it samples without rank-awareness), hence this small
    custom variant. Used by `kind: multi_sharded_hdf5` for equal-weight-per-
    dataset sampling under DDP.
    """

    def __init__(
        self,
        weights,
        num_replicas: int,
        rank: int,
        num_samples: int = None,
        seed: int = 0,
        replacement: bool = True,
        drop_last: bool = True,
    ):
        if num_replicas <= 0 or rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"DistributedWeightedSampler: bad rank/world_size: rank={rank}, "
                f"num_replicas={num_replicas}"
            )
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        total = int(num_samples) if num_samples is not None else int(self.weights.numel())
        if self.drop_last:
            self.num_samples_total = (total // self.num_replicas) * self.num_replicas
        else:
            # Round up so all ranks have equal length; we'll wrap padding at __iter__.
            self.num_samples_total = (
                (total + self.num_replicas - 1) // self.num_replicas
            ) * self.num_replicas
        self.num_samples_per_rank = self.num_samples_total // self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Single global draw, identical across ranks (same seed, same generator
        # state). Each rank slices its disjoint chunk.
        indices = torch.multinomial(
            self.weights, self.num_samples_total, replacement=self.replacement, generator=g
        ).tolist()
        start = self.rank * self.num_samples_per_rank
        end = start + self.num_samples_per_rank
        return iter(indices[start:end])

    def __len__(self):
        return self.num_samples_per_rank

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


# class PushTDataset(Dataset):
#     """
#     Fast dataset for a single HDF5 sequence.
#     Each HDF5 file = one trajectory (len_traj, C, H, W).
#     """

#     def __init__(
#         self,
#         hd5_file_path: str,
#         traj_len: int,
#         non_overlapping: bool = False,
#         load_to_ram: bool = True,
#         nu: int = 2,
#         rate: int = 1,
#     ):
#         """
#         Args:
#             hd5_file_path: path to HDF5 file for a single sequence.
#             traj_len: number of time steps in a chunk.
#             non_overlapping: if True, segments do not overlap.
#             load_to_ram: if True, load entire sequence into RAM as torch tensors.
#             nu: number of action/state dims to keep.
#             rate: temporal downsampling factor.
#         """
#         super().__init__()

#         self.hd5_file_path = hd5_file_path
#         self.traj_len = traj_len
#         self.non_overlapping = non_overlapping
#         self.load_to_ram = load_to_ram
#         self.nu = nu
#         self.rate = rate

#         # We only open once here to inspect shape & optionally load-to-RAM.
#         with h5py.File(self.hd5_file_path, "r") as f:
#             cam1 = f["cam1"]
#             if 'dreamer-tokens' in f:
#                 self.tokens = f['dreamer-tokens']
#                 T1, self.C, self.H, self.W = cam1.shape
#                 T2 = self.tokens.shape[0]
#                 self.len_traj = min(T1, T2)

#             else:
#                 self.tokens = None
#                 self.len_traj, self.C, self.H, self.W = cam1.shape

#             if self.traj_len * self.rate > self.len_traj:
#                 raise ValueError(
#                     f"traj_len * rate = {self.traj_len} * {self.rate} "
#                     f"> len_traj = {self.len_traj}"
#                 )

#             self.has_states = "states" in f.keys()

#             if self.load_to_ram:
#                 # Load everything once and convert to tensors
#                 self.cam1 = torch.from_numpy(cam1[:]).float()          # (T, C, H, W)
#                 if self.tokens is not None:
#                     self.tokens = torch.from_numpy(self.tokens[:]).float()          # (T, C, H, W)
#                 self.actions = torch.from_numpy(f["actions"][:]).float()  # (T, A)
#                 if self.has_states:
#                     self.states = torch.from_numpy(f["states"][:]).float()  # (T, S)
#                 else:
#                     self.states = None

#         # For on-disk mode, we'll open lazily per worker in __getitem__.
#         if not self.load_to_ram:
#             self.cam1 = None
#             self.tokens = None
#             self.actions = None
#             self.states = None
#             self._h5 = None  # will be opened lazily

#         # Compute how many segments and their start indices
#         self._compute_starts()

#     def _compute_starts(self):
#         # Max possible number of overlapping segments
#         max_overlap_len = self.len_traj - self.traj_len * self.rate + 1

#         if self.non_overlapping:
#             # Each segment spans traj_len * rate time steps
#             n_segments = self.len_traj // (self.traj_len * self.rate)
#             # Starts: 0, traj_len*rate, 2*traj_len*rate, ...
#             starts = torch.arange(
#                 0,
#                 n_segments * self.traj_len * self.rate,
#                 self.traj_len * self.rate,
#                 dtype=torch.long,
#             )
#         else:
#             # Fully overlapping sliding window
#             n_segments = max_overlap_len
#             starts = torch.arange(0, n_segments, dtype=torch.long)

#         self.num_segments = int(n_segments)
#         self.starts = starts  # (num_segments,)

#     def __len__(self):
#         return self.num_segments

#     def _lazy_open_h5(self):
#         """
#         Open HDF5 file lazily per worker process.
#         """
#         if self._h5 is None:
#             self._h5 = h5py.File(self.hd5_file_path, "r")
#         return self._h5

#     def _get_from_disk(self, sl):
#         """
#         Fetch a slice from disk and convert to tensors.
#         """
#         f = self._lazy_open_h5()
#         cam1 = f["cam1"][sl]        # (traj_len, C, H, W)
#         actions = f["actions"][sl]  # (traj_len, A)
#         tokens = f["dreamer-tokens"][sl] if "dreamer-tokens" in f else None

#         imgs = torch.from_numpy(cam1).float()
#         actions = torch.from_numpy(actions).float()

#         if "states" in f.keys():
#             states_np = f["states"][sl]
#             states = torch.from_numpy(states_np).float()
#         else:
#             states = torch.zeros_like(actions)

#         return imgs, actions, states, tokens

#     def _get_from_ram(self, sl):
#         """
#         Fetch a slice from preloaded tensors in RAM.
#         """
#         imgs = self.cam1[sl]        # (traj_len, C, H, W)
#         actions = self.actions[sl]  # (traj_len, A)
#         if self.tokens is not None:
#             tokens = self.tokens[sl]
#         else:
#             tokens = None
#         if self.states is not None:
#             states = self.states[sl]
#         else:
#             states = torch.zeros_like(actions)
#         return imgs, actions, states, tokens

#     def __getitem__(self, idx: int):
#         if idx < 0 or idx >= self.num_segments:
#             raise IndexError(f"Index {idx} out of range 0..{self.num_segments - 1}")

#         start = int(self.starts[idx].item())
#         end = start + self.traj_len * self.rate
#         sl = slice(start, end, self.rate)  # stride by rate

#         # Read data either from RAM or from disk
#         if self.load_to_ram:
#             imgs, actions, states, tokens = self._get_from_ram(sl)
#         else:
#             imgs, actions, states, tokens = self._get_from_disk(sl)

#         # Crop state/action dims to nu
#         Dout = {
#             "observation.image": imgs,                                   # (T, C, H, W)
#             "observation.state": states[:, : self.nu],                   # (T, nu)
#             "action": actions[:, : self.nu],                             # (T, nu)
#         }
#         if tokens is not None:
#             Dout["observation.tokens"] = tokens                         # (T, C, H, W)

#         return Dout
    
#     def close(self):
#         if self.load_to_ram:
#             return
#         if self._h5 is not None:
#             try:
#                 self._h5.close()
#             except:
#                 pass
#             self._h5 = None

# # To be depricated
# class SingleViewSequenceDataset(Dataset):
#     """
#     Fast dataset for a single HDF5 sequence.
#     Each HDF5 file = one trajectory (len_traj, C, H, W).
#     """

#     def __init__(
#         self,
#         hd5_file_path: str,
#         traj_len: int,
#         non_overlapping: bool = False,
#         load_to_ram: bool = True,
#         nu: int = 2,
#         rate: int = 1,
#     ):
#         """
#         Args:
#             hd5_file_path: path to HDF5 file for a single sequence.
#             traj_len: number of time steps in a chunk.
#             non_overlapping: if True, segments do not overlap.
#             load_to_ram: if True, load entire sequence into RAM as torch tensors.
#             nu: number of action/state dims to keep.
#             rate: temporal downsampling factor.
#         """
#         super().__init__()

#         self.hd5_file_path = hd5_file_path
#         self.traj_len = traj_len
#         self.non_overlapping = non_overlapping
#         self.load_to_ram = load_to_ram
#         self.nu = nu
#         self.rate = rate

#         # We only open once here to inspect shape & optionally load-to-RAM.
#         with h5py.File(self.hd5_file_path, "r") as f:
#             cam1 = f["cam1"]
#             if 'dreamer-tokens' in f:
#                 self.tokens = f['dreamer-tokens']
#                 T1, self.C, self.H, self.W = cam1.shape
#                 T2 = self.tokens.shape[0]
#                 self.len_traj = min(T1, T2)

#             else:
#                 self.tokens = None
#                 self.len_traj, self.C, self.H, self.W = cam1.shape

#             if self.traj_len * self.rate > self.len_traj:
#                 raise ValueError(
#                     f"traj_len * rate = {self.traj_len} * {self.rate} "
#                     f"> len_traj = {self.len_traj}"
#                 )

#             self.has_states = "states" in f.keys()

#             if self.load_to_ram:
#                 # Load everything once and convert to tensors
#                 self.cam1 = torch.from_numpy(cam1[:]).float()          # (T, C, H, W)
#                 if self.tokens is not None:
#                     self.tokens = torch.from_numpy(self.tokens[:]).float()          # (T, C, H, W)
#                 self.actions = torch.from_numpy(f["actions"][:]).float()  # (T, A)
#                 if self.has_states:
#                     self.states = torch.from_numpy(f["states"][:]).float()  # (T, S)
#                 else:
#                     self.states = None

#         # For on-disk mode, we'll open lazily per worker in __getitem__.
#         if not self.load_to_ram:
#             self.cam1 = None
#             self.tokens = None
#             self.actions = None
#             self.states = None
#             self._h5 = None  # will be opened lazily

#         # Compute how many segments and their start indices
#         self._compute_starts()

#     def _compute_starts(self):
#         # Max possible number of overlapping segments
#         max_overlap_len = self.len_traj - self.traj_len * self.rate + 1

#         if self.non_overlapping:
#             # Each segment spans traj_len * rate time steps
#             n_segments = self.len_traj // (self.traj_len * self.rate)
#             # Starts: 0, traj_len*rate, 2*traj_len*rate, ...
#             starts = torch.arange(
#                 0,
#                 n_segments * self.traj_len * self.rate,
#                 self.traj_len * self.rate,
#                 dtype=torch.long,
#             )
#         else:
#             # Fully overlapping sliding window
#             n_segments = max_overlap_len
#             starts = torch.arange(0, n_segments, dtype=torch.long)

#         self.num_segments = int(n_segments)
#         self.starts = starts  # (num_segments,)

#     def __len__(self):
#         return self.num_segments

#     def _lazy_open_h5(self):
#         """
#         Open HDF5 file lazily per worker process.
#         """
#         if self._h5 is None:
#             self._h5 = h5py.File(self.hd5_file_path, "r")
#         return self._h5

#     def _get_from_disk(self, sl):
#         """
#         Fetch a slice from disk and convert to tensors.
#         """
#         f = self._lazy_open_h5()
#         cam1 = f["cam1"][sl]        # (traj_len, C, H, W)
#         actions = f["actions"][sl]  # (traj_len, A)
#         tokens = f["dreamer-tokens"][sl] if "dreamer-tokens" in f else None

#         imgs = torch.from_numpy(cam1).float()
#         actions = torch.from_numpy(actions).float()

#         if "states" in f.keys():
#             states_np = f["states"][sl]
#             states = torch.from_numpy(states_np).float()
#         else:
#             states = torch.zeros_like(actions)

#         return imgs, actions, states, tokens

#     def _get_from_ram(self, sl):
#         """
#         Fetch a slice from preloaded tensors in RAM.
#         """
#         imgs = self.cam1[sl]        # (traj_len, C, H, W)
#         actions = self.actions[sl]  # (traj_len, A)
#         if self.tokens is not None:
#             tokens = self.tokens[sl]
#         else:
#             tokens = None
#         if self.states is not None:
#             states = self.states[sl]
#         else:
#             states = torch.zeros_like(actions)
#         return imgs, actions, states, tokens

#     def __getitem__(self, idx: int):
#         if idx < 0 or idx >= self.num_segments:
#             raise IndexError(f"Index {idx} out of range 0..{self.num_segments - 1}")

#         start = int(self.starts[idx].item())
#         end = start + self.traj_len * self.rate
#         sl = slice(start, end, self.rate)  # stride by rate

#         # Read data either from RAM or from disk
#         if self.load_to_ram:
#             imgs, actions, states, tokens = self._get_from_ram(sl)
#         else:
#             imgs, actions, states, tokens = self._get_from_disk(sl)

#         # Crop state/action dims to nu
#         Dout = {
#             "observation.image": imgs,                                   # (T, C, H, W)
#             "observation.state": states[:, : self.nu],                   # (T, nu)
#             "action": actions[:, : self.nu],                             # (T, nu)
#         }
#         if tokens is not None:
#             Dout["observation.tokens"] = tokens                         # (T, C, H, W)

#         return Dout
    
#     def close(self):
#         if self.load_to_ram:
#             return
#         if self._h5 is not None:
#             try:
#                 self._h5.close()
#             except:
#                 pass
#             self._h5 = None
    
# class ShardedHDF5Dataset(Dataset):
#     """
#     Dataset for sharded HDF5 files optimized for multi-node training.
#     Each worker preferentially reads from local shards when available.
#     """

#     def __init__(
#         self,
#         data_dir: str,
#         window_size: int,
#         stride: int = 1,
#         split: str = "train",          # "train" or "test"
#         train_fraction: float = 0.9,   # fraction of episodes in train
#         split_seed: int = 42,          # seed for reproducible split
#         shuffle_windows: bool = True, 
#         static_prob: float = 0.0,      # probability of applying no-motion augmentation   
#     ):
#         self.shuffle_windows = shuffle_windows
#         self.data_dir = Path(data_dir)
#         self.window_size = window_size
#         self.stride = stride
#         self.split = split
#         self.train_fraction = train_fraction
#         self.split_seed = split_seed
#         self.static_prob = static_prob
#         # Load metadata
#         with open(self.data_dir / 'metadata.json', 'r') as f:
#             self.metadata = json.load(f)

#         self.num_shards = self.metadata['num_shards']
#         self.shard_files = [
#             self.data_dir / f"shard_{i:04d}.h5"
#             for i in range(self.num_shards)
#         ]

#         # Build window index across all shards
#         self.windows = []
#         self.episode_lengths = []  # Store all episode lengths for analysis

#         for shard_idx, shard_file in enumerate(self.shard_files):
#             with h5py.File(shard_file, 'r') as f:
#                 num_episodes = f.attrs['num_episodes']
#                 lengths = f['episode_lengths'][:]

#                 # Store episode lengths for statistics
#                 self.episode_lengths.extend(lengths.tolist())

#                 for ep_idx, ep_length in enumerate(lengths):
#                     for start in range(0, ep_length - window_size + 1, stride):
#                         self.windows.append((shard_idx, ep_idx, start))

#         # Collect all (shard_idx, ep_idx) pairs
#         all_episodes = sorted({(shard_idx, ep_idx) for shard_idx, ep_idx, _ in self.windows})

#         rng = np.random.default_rng(self.split_seed)
#         perm = rng.permutation(len(all_episodes))

#         num_train_eps = int(self.train_fraction * len(all_episodes))
#         train_eps = {all_episodes[i] for i in perm[:num_train_eps]}
#         test_eps  = {all_episodes[i] for i in perm[num_train_eps:]}

#         self.split_info = {
#             "train_episodes": sorted(list(train_eps)),
#             "test_episodes": sorted(list(test_eps)),
#         }

#         if self.split == "train":
#             keep = train_eps
#         elif self.split == "test":
#             keep = test_eps
#         else:
#             raise ValueError(f"Unknown split: {self.split}")

#         # Filter windows based on chosen split
#         self.windows = [w for w in self.windows if (w[0], w[1]) in keep]
#         print(f"{self.split.capitalize()} split: {len(self.windows)} windows "
#               f"from {len(keep)} episodes")
#         if self.shuffle_windows:
#             random.shuffle(self.windows)

#     def __len__(self):
#         return len(self.windows)

#     def __getitem__(self, idx):
#         shard_idx, ep_idx, start = self.windows[idx]
#         end = start + self.window_size

#         shard_file = self.shard_files[shard_idx]

#         # Open HDF5 file (each worker maintains its own handle)
#         with h5py.File(shard_file, 'r') as f:
#             images = f['images'][ep_idx, start:end]
#             actions = f['actions'][ep_idx, start:end]

#         # Convert to PyTorch
#         images = torch.from_numpy(images).float() / 255.0
#         images = images.permute(0, 3, 1, 2)
#         actions = torch.from_numpy(actions)

#         if torch.rand(1).item() < self.static_prob:  # chance to apply no-motion augmentation
#             images[:, ...] = images[0, ...]  # Broadcast first frame for no motion augmentation
#             actions = torch.zeros_like(actions)  # Zero out actions for no motion augmentation
        
#         return {'image': images, 'action': actions}


#     def get_episode_length_statistics(self):
#         """
#         Calculate comprehensive statistics about episode lengths.
        
#         Returns:
#             dict with statistics about episode lengths
#         """
#         lengths = np.array(self.episode_lengths)
        
#         stats = {
#             'total_episodes': len(lengths),
#             'total_timesteps': int(np.sum(lengths)),
#             'min_length': int(np.min(lengths)),
#             'max_length': int(np.max(lengths)),
#             'mean_length': float(np.mean(lengths)),
#             'median_length': float(np.median(lengths)),
#             'std_length': float(np.std(lengths)),
#             'percentile_25': float(np.percentile(lengths, 25)),
#             'percentile_75': float(np.percentile(lengths, 75)),
#             'percentile_90': float(np.percentile(lengths, 90)),
#             'percentile_95': float(np.percentile(lengths, 95)),
#             'percentile_99': float(np.percentile(lengths, 99)),
#         }
        
#         return stats




# class HDF5SequenceDataset(Dataset):
#     """
#     Dataset that creates sliding windows from a directory of independent .h5 files.
#     It automatically parses 'dones' to ensure windows do not cross episode boundaries.
#     """

#     def __init__(
#         self,
#         data_dir: str,
#         window_size: int,
#         stride: int = 1,
#     ):
#         """
#         Args:
#             data_dir: Directory containing .h5 files.
#             window_size: Sequence length (batch time dimension).
#             stride: Step size between windows.
#         """
#         self.data_dir = Path(data_dir)
#         self.window_size = window_size
#         self.stride = stride
        
#         # Find all H5 files
#         self.h5_files = sorted([
#             f for f in self.data_dir.glob("*.h5") 
#             if "shard" not in f.name # Exclude shard files if mixed
#         ])
        
#         if not self.h5_files:
#             raise ValueError(f"No .h5 files found in {data_dir}")

#         # Indexing structure: list of (file_index, start_frame_index)
#         self.windows = []
        
#         print(f"Scanning {len(self.h5_files)} files for valid episodes...")
        
#         total_frames = 0
#         total_episodes = 0
        
#         for file_idx, file_path in enumerate(self.h5_files):
#             try:
#                 with h5py.File(file_path, 'r') as f:
#                     if 'dones' not in f or 'images' not in f:
#                         print(f"Skipping {file_path.name}: missing datasets")
#                         continue
                        
#                     n_frames = len(f['images'])
#                     dones = f['dones'][:]
                    
#                     # Identify Episode Boundaries
#                     # An episode ends where done=1.
#                     # We need start and end indices for every continuous segment.
                    
#                     # Indices where done == 1
#                     done_indices = np.where(dones > 0.5)[0]
                    
#                     # Episode Starts: [0] + [idx+1 for idx in done_indices if idx+1 < n_frames]
#                     # Episode Ends:   [idx+1 for idx in done_indices] + [n_frames] (if last frame isn't done)
                    
#                     # Simpler logic: Iterate through done indices to carve out chunks
#                     start_idx = 0
                    
#                     # Add a synthetic done at the very end to close the last loop
#                     all_boundaries = list(done_indices)
#                     if len(all_boundaries) == 0 or all_boundaries[-1] != n_frames - 1:
#                         all_boundaries.append(n_frames - 1)
                        
#                     for end_idx in all_boundaries:
#                         # The episode is valid from start_idx to end_idx (inclusive)
#                         # Length = end_idx - start_idx + 1
                        
#                         # Generate windows for this segment
#                         # Valid starts: range(segment_start, segment_end - window_size + 2, stride)
#                         # Example: Ep len 50, window 50. range(0, 0+1) -> [0]. Window 0:50.
                        
#                         # Note: HDF5 slicing [start:end] is exclusive at end, so we use end_idx + 1
#                         segment_len = (end_idx - start_idx) + 1
                        
#                         if segment_len >= self.window_size:
#                             num_windows_in_ep = (segment_len - self.window_size) // self.stride + 1
                            
#                             for k in range(num_windows_in_ep):
#                                 global_start = start_idx + (k * self.stride)
#                                 self.windows.append((file_idx, global_start))
                            
#                             total_episodes += 1
                        
#                         # Next episode starts after this done
#                         start_idx = end_idx + 1
                        
#                     total_frames += n_frames
                    
#             except Exception as e:
#                 print(f"Error reading {file_path.name}: {e}")

#         print(f"Found {len(self.windows)} windows across {total_episodes} valid episodes.")
#         print(f"Total raw frames processed: {total_frames}")

#     def __len__(self):
#         return len(self.windows)

#     def __getitem__(self, idx):
#         file_idx, start_frame = self.windows[idx]
#         file_path = self.h5_files[file_idx]
#         end_frame = start_frame + self.window_size

#         with h5py.File(file_path, 'r') as f:
#             images = f['images'][start_frame:end_frame]  # [T, H, W, 3] (BGR)
#             commands = f['commands'][start_frame:end_frame]
#             dones = f['dones'][start_frame:end_frame]

#             # --- Handle optional is_demo field ---
#             if 'is_demo' in f:
#                 is_demo = f['is_demo'][start_frame:end_frame]
#             else:
#                 # Create zeros matching the sequence length
#                 is_demo = np.zeros((self.window_size, 1), dtype=np.float32)

#         # --- Convert BGR to RGB ---
#         # Numpy array slicing is the fastest way to do this
#         images = images[..., ::-1].copy()

#         # Convert to Torch
#         images = torch.from_numpy(images).float() / 255.0
#         images = images.permute(0, 3, 1, 2)  # [T, C, H, W]
        
#         commands = torch.from_numpy(commands).float()
#         dones = torch.from_numpy(dones).float()

#         # Ensure is_demo is a tensor
#         if isinstance(is_demo, np.ndarray):
#             is_demo = torch.from_numpy(is_demo).float()

#         return {
#             'image': images,
#             'action': commands,
#             'done': dones,
#             'is_demo': is_demo
#         }

def create_distributed_dataloader(
    data_dir: str = None,
    window_size: int = None,
    batch_size: int = None,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    stride: int = 1,
    seed: int = 42,
    split: str = "train",
    train_fraction: float = 0.9,
    split_seed: int = 42,
    shuffle: bool = True,
    drop_last: bool = True,
    absolute_actions: bool = False,
    kind: str = "sharded_hdf5",
    data_dirs=None,
):
    """
    Create DataLoader with DistributedSampler.

    `kind` selects the dataset layout:
      - 'sharded_hdf5' (default): legacy per-episode sharded layout
        (`metadata.json` + `shard_XXXX.h5` with `images (N,T,H,W,C)`). Reads
        from `data_dir`.
      - 'g1_chunked': flat per-frame G1 chunks (`chunk_XXXX.h5`); episode
        boundaries inferred from `dones` and `source_rrd`. `absolute_actions`
        is ignored in this branch. Reads from `data_dir`.
      - 'multi_sharded_hdf5': union of N `sharded_hdf5` datasets, one per
        `data_dirs[i]`. Each subset does its own per-dataset train/test split
        with the shared `split_seed`. Sampling is equal-weight-per-dataset via
        `DistributedWeightedSampler` (the `shuffle` flag is ignored — order is
        always randomized; `seed` controls the multinomial draw).
    """
    if kind == "sharded_hdf5":
        dataset = ShardedHDF5Dataset(
            data_dir=data_dir,
            window_size=window_size,
            stride=stride,
            split=split,
            train_fraction=train_fraction,
            split_seed=split_seed,
            absolute_actions=absolute_actions,
        )
    elif kind == "g1_chunked":
        dataset = G1ChunkDataset(
            data_dir=data_dir,
            window_size=window_size,
            stride=stride,
            split=split,
            train_fraction=train_fraction,
            split_seed=split_seed,
        )
    elif kind == "multi_sharded_hdf5":
        if not data_dirs:
            raise ValueError(
                "kind='multi_sharded_hdf5' requires `data_dirs` (list of paths)"
            )
        dataset = MultiShardedHDF5Dataset(
            data_dirs=data_dirs,
            window_size=window_size,
            stride=stride,
            split=split,
            train_fraction=train_fraction,
            split_seed=split_seed,
            absolute_actions=absolute_actions,
        )
    else:
        raise ValueError(
            f"Unknown dataset kind: {kind!r} "
            f"(expected 'sharded_hdf5', 'g1_chunked', or 'multi_sharded_hdf5')"
        )

    if kind == "multi_sharded_hdf5":
        sampler = DistributedWeightedSampler(
            weights=dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            num_samples=len(dataset),
            seed=seed,
            replacement=True,
            drop_last=drop_last,
        )
    else:
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


def create_distributed_demo_play_dataloader(
    demo_data_dir: str,
    play_data_dir: str,
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
    goal_reward: float = 20.0,
    demo_reward: float = 1.0,
    play_reward: float = 0.0,
    goal_frames: int = 5,
    terminal_bias: float = 0.5,
):
    """DataLoader over a 50/50 mix of demo + play windows with synthesized rewards.

    See `MixedDemoPlayDataset` for the reward synthesis convention. Mirrors
    `create_distributed_dataloader` in shape — DistributedSampler + DataLoader
    with the same defaults — so it can be a drop-in swap in training scripts.
    """
    dataset = MixedDemoPlayDataset(
        demo_data_dir=demo_data_dir,
        play_data_dir=play_data_dir,
        window_size=window_size,
        stride=stride,
        split=split,
        train_fraction=train_fraction,
        split_seed=split_seed,
        absolute_actions=absolute_actions,
        goal_reward=goal_reward,
        demo_reward=demo_reward,
        play_reward=play_reward,
        goal_frames=goal_frames,
        terminal_bias=terminal_bias,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
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
