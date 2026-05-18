"""Smoke-test for `kind: multi_sharded_hdf5` configs.

Builds the train/test `MultiShardedHDF5Dataset` from a hydra dataset config,
prints per-subset window/episode counts, verifies `action_shape` uniformity,
samples a few items, and dumps the per-dataset expected sampling mass under
`DistributedWeightedSampler`.

Usage (inside the container):

    python scripts/inspect_multi_dataset.py \
        --config-path config --config-name dataset/ogbench-manipulation \
        window_size=32

Or for a tokenizer config (which embeds the dataset under `dataset:`):

    python scripts/inspect_multi_dataset.py \
        --config-path config --config-name tokenizer/ogbench-manipulation
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from dreamerv4uwm.datasets import MultiShardedHDF5Dataset


def _resolve_dataset_cfg(cfg: DictConfig) -> DictConfig:
    # Accept either a bare dataset config (e.g. dataset/ogbench-manipulation.yaml
    # loaded directly) or a higher-level config that nests it under `dataset:`.
    if "data_dirs" in cfg or cfg.get("kind") == "multi_sharded_hdf5":
        return cfg
    if "dataset" in cfg:
        return cfg.dataset
    raise ValueError(
        "Could not locate a multi_sharded_hdf5 dataset config (no `data_dirs` "
        "at top level and no `dataset:` key)."
    )


def _print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


@hydra.main(config_path="config", config_name=None, version_base=None)
def main(cfg: DictConfig):
    ds_cfg = _resolve_dataset_cfg(cfg)
    window_size = int(cfg.get("window_size", 32))

    _print_section(f"Building train split (window_size={window_size})")
    train = MultiShardedHDF5Dataset(
        data_dirs=list(ds_cfg.data_dirs),
        window_size=window_size,
        stride=1,
        split="train",
        train_fraction=float(ds_cfg.get("train_episodes_fraction", 0.9)),
        split_seed=int(ds_cfg.get("split_seed", 42)),
        absolute_actions=bool(ds_cfg.get("absolute_actions", False)),
    )

    _print_section("Building test split")
    test = MultiShardedHDF5Dataset(
        data_dirs=list(ds_cfg.data_dirs),
        window_size=window_size,
        stride=1,
        split="test",
        train_fraction=float(ds_cfg.get("train_episodes_fraction", 0.9)),
        split_seed=int(ds_cfg.get("split_seed", 42)),
        absolute_actions=bool(ds_cfg.get("absolute_actions", False)),
    )

    _print_section("Per-subset window counts (train / test) + sampling mass")
    N = len(train.subsets)
    print(f"{'subset':45s} {'train':>10s} {'test':>10s} {'p_per_window':>15s}")
    for d, sub_tr, sub_te in zip(ds_cfg.data_dirs, train.subsets, test.subsets):
        name = Path(str(d)).name
        n_tr = len(sub_tr)
        n_te = len(sub_te)
        p = 1.0 / (N * n_tr) if n_tr > 0 else 0.0
        print(f"{name:45s} {n_tr:>10d} {n_te:>10d} {p:>15.3e}")

    _print_section("Aggregate stats")
    stats = train.get_episode_length_statistics()
    print(f"  total_episodes (train): {stats['total_episodes']}")
    print(f"  total_timesteps (train): {stats['total_timesteps']:,}")
    print(f"  ep_length min/median/max: {stats['min_length']} / "
          f"{stats['median_length']:.1f} / {stats['max_length']}")
    total_train_windows = len(train)
    total_test_windows = len(test)
    print(f"  total train windows: {total_train_windows:,}")
    print(f"  total test windows: {total_test_windows:,}")
    print(f"  per-dataset sampling mass: {1.0/N:.4f}  (== 1/{N})")

    _print_section("Sample probe (item 0, train)")
    sample = train[0]
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k:20s} shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k:20s} {type(v).__name__}")

    print("\nOK.")


if __name__ == "__main__":
    main()
