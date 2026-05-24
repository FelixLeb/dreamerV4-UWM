#!/usr/bin/env python3
"""
eval_fdpo_offline.py

Offline eval of an FDPO-aligned denoiser against its play-trained reference,
producing the three scalars from `ideas/wam-dpo.tex` §5:

  - **demo  policy** : aligned vs. ref ℓ_θ on held-out demo positives, in
    policy mode. Primary signal — aligned should drop.
  - **play  policy** : same metric on held-out play windows. Retention check
    on the "joint generation" regime — aligned should be ≈ ref.
  - **play  wm**     : same metric in WM mode (clean context + clean actions,
    noisy horizon state). Mode-1 regression check — aligned should be ≈ ref.

The script reuses `compute_per_sample_uwm_loss` and `UWMForwardProcess.apply_diff`
so that ref and aligned see exactly the same `(τ, ε)` samples per window — the
same variance-reduction trick used during FDPO training.

Run:

    conda run -n dreamerv4 python scripts/eval_fdpo_offline.py \\
        --ref_ckpt_dir  checkpoints/dynamics/pushT/final/joint-model/v1 \\
        --ref_ckpt_file 432360.pt \\
        --tokenizer_ckpt checkpoints/tokenizer_ckpts/pushT.pt \\
        --pairs_dir /home/mim-server/datasets/pushT/fdpo-pairs-v0 \\
        --play_dir  /home/mim-server/datasets/pushT/sharded \\
        --aligned_ckpt checkpoints/align/<run>/final_merged.pt

The `--aligned_ckpt` is optional; if omitted, only the ref columns are filled
(useful for establishing a baseline before any FDPO training has finished).
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torch.utils.data import DataLoader

from dreamerv4uwm.datasets import PreferencePairDataset, ShardedHDF5Dataset
from dreamerv4uwm.loss import UWMForwardProcess, compute_per_sample_uwm_loss
from dreamerv4uwm.models.utils import load_denoiser, load_tokenizer


def _strip_deprecated_keys(cfg):
    if 'latent_attends_action' in cfg.denoiser:
        del cfg.denoiser['latent_attends_action']


def _build_aligned_from_adapter(cfg, adapter_path: Path, train_cfg_path: Path,
                                device, max_seq):
    """Load base denoiser + wrap with LoRA matching the training config + load adapter."""
    train_cfg = OmegaConf.load(train_cfg_path)
    print(f"Loading aligned denoiser via adapter:")
    print(f"  base    : {cfg.dynamics_ckpt}")
    print(f"  adapter : {adapter_path}")
    print(f"  LoRA cfg: r={train_cfg.lora.r} alpha={train_cfg.lora.lora_alpha} "
          f"targets={train_cfg.lora.target_modules}")

    base = load_denoiser(cfg, device, max_num_forward_steps=max_seq)

    lora = train_cfg.lora
    target = lora.target_modules
    if (isinstance(target, (list, tuple)) or
            (hasattr(target, "__iter__") and not isinstance(target, str))):
        target = list(target)
    modules_to_save = lora.get("modules_to_save", None)
    if modules_to_save is not None:
        modules_to_save = list(modules_to_save)
    lora_cfg = LoraConfig(
        r=int(lora.r),
        lora_alpha=int(lora.lora_alpha),
        lora_dropout=float(lora.lora_dropout),
        bias=str(lora.bias),
        target_modules=target,
        modules_to_save=modules_to_save,
    )
    aligned = get_peft_model(base, lora_cfg)

    ckpt = torch.load(adapter_path, map_location=device, weights_only=False)
    if "adapter" not in ckpt:
        raise KeyError(
            f"{adapter_path} is missing the 'adapter' key. Expected an FDPO "
            f"LoRA checkpoint as saved by `save_lora_checkpoint`."
        )
    set_peft_model_state_dict(aligned, ckpt["adapter"])
    aligned.eval()
    return aligned


def _build(args):
    """Build cfg, tokenizer, ref, aligned (or None), diffuser."""
    if args.aligned_ckpt is not None and args.aligned_adapter is not None:
        raise SystemExit(
            "Pass at most one of --aligned_ckpt (merged) and --aligned_adapter "
            "(LoRA adapter)."
        )

    ref_dir = Path(args.ref_ckpt_dir)
    cfg = OmegaConf.load(ref_dir / 'config.yaml')
    cfg.dynamics_ckpt = str(ref_dir / args.ref_ckpt_file)
    cfg.tokenizer_ckpt = args.tokenizer_ckpt
    _strip_deprecated_keys(cfg)
    max_seq = cfg.denoiser.max_sequence_length
    device = torch.device(args.device)

    tokenizer = load_tokenizer(cfg, device, max_num_forward_steps=max_seq).eval()
    ref = load_denoiser(cfg, device, max_num_forward_steps=max_seq).eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    for p in ref.parameters():
        p.requires_grad_(False)

    aligned = None
    if args.aligned_ckpt is not None:
        # Drop-in: merged FDPO checkpoint shares the base architecture.
        cfg_aligned = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_aligned.dynamics_ckpt = args.aligned_ckpt
        print(f"Loading aligned denoiser from merged checkpoint {args.aligned_ckpt}")
        aligned = load_denoiser(cfg_aligned, device, max_num_forward_steps=max_seq).eval()
    elif args.aligned_adapter is not None:
        adapter_path = Path(args.aligned_adapter)
        train_cfg_path = (Path(args.aligned_train_cfg)
                          if args.aligned_train_cfg
                          else adapter_path.parent / "config.yaml")
        if not train_cfg_path.exists():
            raise FileNotFoundError(
                f"Need training config at {train_cfg_path} (override with "
                f"--aligned_train_cfg)."
            )
        cfg_aligned = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        aligned = _build_aligned_from_adapter(
            cfg_aligned, adapter_path, train_cfg_path, device, max_seq,
        )
    else:
        print("No aligned checkpoint provided; reporting ref-only baseline.")

    if aligned is not None:
        for p in aligned.parameters():
            p.requires_grad_(False)

    diffuser = UWMForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        mode_weights={'policy': 1.0, 'wm': 1.0},
        horizon_aware=bool(cfg.denoiser.get("horizon_aware", False)),
        device=device,
    )
    return cfg, tokenizer, ref, aligned, diffuser, device


@torch.no_grad()
def _eval_one_pass(dataset, batch_size, ref, aligned, tokenizer, diffuser,
                   device, force_mode, n_actions, n_tau_samples,
                   key_image, key_action, max_batches=None):
    """Walk a dataset, average per-sample ℓ_θ over windows × τ-samples.

    Ref and aligned see the same `(τ, ε)` per τ-sample iteration so the
    comparison is on-paired.

    Returns: dict with `ref`, `aligned` (None if aligned is None), `n_windows`.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True, drop_last=False)
    ref_sum = 0.0
    aligned_sum = 0.0
    n_windows = 0

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        images = batch[key_image].to(device, non_blocking=True).to(torch.bfloat16)
        actions_full = batch[key_action].to(device, non_blocking=True).to(torch.bfloat16)
        actions = actions_full[:, :, :n_actions].unsqueeze(-2)
        B, T = images.shape[0], images.shape[1]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            z_clean = tokenizer.encode(images).detach()

        for _ in range(n_tau_samples):
            obs_diff, act_diff, ctx_len, mode_out, is_horizon = diffuser.sample_step_noise(
                B, T, force_mode=force_mode,
            )
            z0 = torch.randn_like(z_clean)
            a0 = diffuser.action_noise_std * torch.randn_like(actions)
            info = diffuser.apply_diff(
                z_clean, actions, obs_diff, act_diff, ctx_len, mode_out,
                z0=z0, a0=a0, is_horizon=is_horizon,
            )
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                ref_per = compute_per_sample_uwm_loss(info, ref, device=device)
                ref_sum += ref_per['total_flow_loss'].float().sum().item()
                if aligned is not None:
                    aligned_per = compute_per_sample_uwm_loss(info, aligned, device=device)
                    aligned_sum += aligned_per['total_flow_loss'].float().sum().item()
        n_windows += B

    denom = max(n_windows * n_tau_samples, 1)
    return {
        'ref': ref_sum / denom,
        'aligned': (aligned_sum / denom) if aligned is not None else None,
        'n_windows': n_windows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Models
    p.add_argument('--ref_ckpt_dir',   required=True,
                   help="Dir holding config.yaml + the original .pt.")
    p.add_argument('--ref_ckpt_file',  required=True,
                   help="Filename of the play-trained .pt under ref_ckpt_dir.")
    p.add_argument('--tokenizer_ckpt', required=True)
    p.add_argument('--aligned_ckpt',   default=None,
                   help="Path to a merged FDPO checkpoint (e.g. `final_merged.pt`). "
                        "Mutually exclusive with --aligned_adapter.")
    p.add_argument('--aligned_adapter', default=None,
                   help="Path to a LoRA adapter checkpoint (e.g. `adapter_NNN.pt`). "
                        "The matching LoRA config is read from "
                        "<adapter_dir>/config.yaml unless --aligned_train_cfg is set.")
    p.add_argument('--aligned_train_cfg', default=None,
                   help="Override path to the FDPO training run's config.yaml. "
                        "Only used with --aligned_adapter.")
    # Datasets
    p.add_argument('--pairs_dir', required=True,
                   help="Output of gen_fdpo_negatives.py. The 'test' split of "
                        "this dataset is the held-out demo set for metric 1.")
    p.add_argument('--play_dir',  required=True,
                   help="Raw play dataset (sharded HDF5). Held-out windows are "
                        "used for metrics 2 and 3.")
    # Splits (must match what training used for the pairs!)
    p.add_argument('--pairs_train_fraction', type=float, default=0.9)
    p.add_argument('--pairs_split_seed',     type=int,   default=42)
    p.add_argument('--play_train_fraction',  type=float, default=0.9)
    p.add_argument('--play_split_seed',      type=int,   default=42)
    p.add_argument('--play_stride',          type=int,   default=64,
                   help="Stride for windowing the play dataset. Default 64 = "
                        "non-overlapping 64-frame windows.")
    # Eval knobs
    p.add_argument('--batch_size',   type=int, default=4)
    p.add_argument('--n_tau_samples', type=int, default=4,
                   help="Number of (τ, ε) samples averaged per window.")
    p.add_argument('--max_batches',  type=int, default=None,
                   help="Optional cap per metric, for fast sanity runs.")
    # Misc
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed',   type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg, tokenizer, ref, aligned, diffuser, device = _build(args)
    n_actions = cfg.denoiser.n_actions
    max_seq = cfg.denoiser.max_sequence_length

    # --- Held-out datasets ---
    demo_pairs = PreferencePairDataset(
        data_dir=args.pairs_dir,
        split='test',
        train_fraction=args.pairs_train_fraction,
        split_seed=args.pairs_split_seed,
        shuffle_pairs=False,
    )
    play_test = ShardedHDF5Dataset(
        data_dir=args.play_dir,
        window_size=max_seq,
        stride=args.play_stride,
        split='test',
        train_fraction=args.play_train_fraction,
        split_seed=args.play_split_seed,
        shuffle_windows=False,
    )
    print(f"Held-out demo pairs : {len(demo_pairs)}")
    print(f"Held-out play windows: {len(play_test)}  "
          f"(stride={args.play_stride}, T={max_seq})")
    print(f"Eval: n_tau_samples={args.n_tau_samples}, "
          f"batch_size={args.batch_size}, max_batches={args.max_batches}")

    metrics = [
        # (label, dataset, force_mode, key_image, key_action)
        ('demo policy', demo_pairs, 'policy', 'pos_image', 'pos_action'),
        ('play policy', play_test,  'policy', 'image',     'action'),
        ('play wm    ', play_test,  'wm',     'image',     'action'),
    ]
    results = []
    for label, ds, mode, key_img, key_act in metrics:
        print(f"\n[{label}]  iterating...")
        out = _eval_one_pass(
            dataset=ds, batch_size=args.batch_size,
            ref=ref, aligned=aligned, tokenizer=tokenizer, diffuser=diffuser,
            device=device, force_mode=mode, n_actions=n_actions,
            n_tau_samples=args.n_tau_samples, key_image=key_img,
            key_action=key_act, max_batches=args.max_batches,
        )
        results.append((label, out))
        msg = (f"  windows={out['n_windows']}  ref={out['ref']:.4f}")
        if out['aligned'] is not None:
            delta = out['aligned'] - out['ref']
            msg += (f"  aligned={out['aligned']:.4f}  "
                    f"Δ={delta:+.4f}  ({'better' if delta < 0 else 'worse'})")
        print(msg)

    # --- Summary table ---
    print('\n' + '=' * 72)
    print(f'{"metric":<14}{"windows":>10}{"ref ℓ_θ":>14}{"aligned ℓ_θ":>16}{"Δ":>14}')
    print('-' * 72)
    for label, r in results:
        n   = r['n_windows']
        ref_str = f"{r['ref']:>14.4f}"
        if r['aligned'] is None:
            a_str, d_str = f"{'(n/a)':>16}", f"{'-':>14}"
        else:
            delta = r['aligned'] - r['ref']
            a_str = f"{r['aligned']:>16.4f}"
            d_str = f"{delta:>+14.4f}"
        print(f'{label:<14}{n:>10d}{ref_str}{a_str}{d_str}')
    print('=' * 72)

    if aligned is None:
        print("\nNote: --aligned_ckpt was not provided. Run again after FDPO "
              "training completes, passing `--aligned_ckpt <run>/final_merged.pt`, "
              "to fill the aligned and Δ columns. Decision rule (v0):")
    else:
        print("\nDecision rule (v0):")
    print("  - `demo policy` Δ should be materially NEGATIVE → FDPO is working")
    print("  - `play policy` and `play wm` Δ should be within run-to-run noise of zero")
    print("    → no retention / WM-mode regression. If either blows up, the LoRA")
    print("    rank or β was too aggressive.")


if __name__ == '__main__':
    main()
