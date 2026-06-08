"""AdaLN-Zero class-conditioning alignment on top of the block-causal / unified
pipeline (`loss_new`).

This is the unified-loss sibling of `scripts/train_align_cond.py`. It forks
`scripts/train_dynamics_uwm_new.py` (the block-causal / unified dynamics trainer)
and adds provenance conditioning:

- The denoiser is conditioned on a discrete **3-class** label `null` / `play` /
  `demo` (indices 0/1/2; null = 0 = unconditioned) via the zero-init AdaLN-Zero
  retrofit (`denoiser.cond_adaln`). At day 0 (and whenever `cond_class` is None)
  the network is bit-identical to the pretrained model.

- The class is sampled with **equal probability** (1/3 each), independently of the
  diffusion **mode** (`unified` / `video_pretraining` / `action_pretraining` /
  `block_causal_sanity`) and the length **branch** (`long` / `short` / `image`),
  all of which are retained from the dynamics trainer.

- **Class → dataset routing**:
    play → play data only · demo → demo data only ·
    null → both (per window: play w.p. `null_play_fraction`, else demo)
  via two independent dataloaders (`play_data_dir` / `demo_data_dir`).

- Backbone freeze is a flag `train.freeze_backbone` (default True, ControlNet
  style); `lora.enabled` is independent (LoRA implies a frozen base, with the
  conditioning params kept trainable via `modules_to_save`).

- Base weights load via a `strict=False` load with an explicit missing/unexpected
  key report; conditioning-disabled / freeze-matched-nothing both hard-assert.

Run length is driven by `train.num_training_steps` (total optimizer steps) with a
cosine schedule over the same horizon — the loaders cycle as needed.
"""

import math
import os
import time

import hydra
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from dreamerv4uwm.datasets import create_distributed_dataloader
from dreamerv4uwm.loss_new import (
    ActionPretrainingForwardProcess,
    BlockCausalSanityForwardProcess,
    ImageForwardProcess,
    RMSLossScaler,
    UnifiedForwardProcess,
    VideoPretrainingForwardProcess,
    compute_action_pretraining_loss,
    compute_block_causal_sanity_loss,
    compute_image_loss,
    compute_unified_uwm_loss,
    compute_video_pretraining_loss,
)
from dreamerv4uwm.models.dynamics import DenoiserWrapper
from dreamerv4uwm.models.utils import load_tokenizer
from dreamerv4uwm.utils.distributed import (
    cleanup_distributed,
    load_ddp_checkpoint,
    save_ddp_checkpoint,
    setup_distributed,
    unwrap_model,
)


# ---------------------------------------------------------------------------
# Conditioning classes / sources (shared with train_align_cond.py)
# ---------------------------------------------------------------------------
CLASSES = ['null', 'play', 'demo']     # index 0/1/2 — null = unconditioned
SOURCES = ['play', 'demo']
CLASS_SOURCE_RULE = {
    'null':  'mixed',  # both play and demo (null_play_fraction by default 0.5)
    'play':  'play',
    'demo':  'demo',
}
COND_PARAM_KEYS = ('adaln', 'class_embedder')
BRANCHES = ('long', 'image', 'short')
MODES = ('unified', 'video_pretraining', 'action_pretraining', 'block_causal_sanity')


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-8):
    peak_lr = optimizer.defaults["lr"]
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / peak_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Dataloaders (play + demo)
# ---------------------------------------------------------------------------

def build_dataloaders(cfg, rank, world_size):
    """Two loaders (play, demo). Each delivers max_sequence_length frames per
    sample at loader_bs = max(short_bs, long_bs) so any branch can be served."""
    long_bs = int(cfg.train.get("long_seq_batch_per_gpu", cfg.train.batch_per_gpu))
    short_bs = int(cfg.train.batch_per_gpu)
    loader_bs = max(short_bs, long_bs)
    window = int(cfg.denoiser.max_sequence_length)
    kind = str(cfg.dataset.get("kind", "sharded_hdf5"))

    play_loader, play_sampler, _ = create_distributed_dataloader(
        data_dir=cfg.dataset.play_data_dir,
        window_size=window, batch_size=loader_bs,
        rank=rank, world_size=world_size,
        num_workers=cfg.train.num_workers,
        stride=1, seed=cfg.seed, split="train",
        train_fraction=cfg.dataset.train_episodes_fraction,
        split_seed=cfg.dataset.split_seed,
        shuffle=True, drop_last=True,
        absolute_actions=cfg.train.absolute_actions, kind=kind,
    )
    demo_loader, demo_sampler, _ = create_distributed_dataloader(
        data_dir=cfg.dataset.demo_data_dir,
        window_size=window, batch_size=loader_bs,
        rank=rank, world_size=world_size,
        num_workers=max(1, cfg.train.num_workers // 2),
        stride=1, seed=cfg.seed + 7919, split="train",
        train_fraction=cfg.dataset.train_episodes_fraction,
        split_seed=cfg.dataset.split_seed,
        shuffle=True, drop_last=True,
        absolute_actions=cfg.train.absolute_actions, kind=kind,
    )
    return play_loader, play_sampler, demo_loader, demo_sampler


def _cycling_iter(loader, sampler=None, epoch_ref=None, base_offset=0):
    """Infinite iterator over a DataLoader; advances DistributedSampler epoch on
    each restart so shuffles keep varying across passes."""
    pass_idx = 0
    while True:
        if sampler is not None:
            sampler.set_epoch((epoch_ref[0] if epoch_ref else 0) * 10_000
                              + base_offset + pass_idx)
        for b in loader:
            yield b
        pass_idx += 1


# ---------------------------------------------------------------------------
# Optional LoRA wrap + checkpoint report (shared logic with train_align_cond.py)
# ---------------------------------------------------------------------------

def build_lora_config(cfg: DictConfig, extra_modules_to_save=None):
    from peft import LoraConfig
    lora = cfg.lora
    target = lora.target_modules
    if (isinstance(target, (list, tuple)) or
            (hasattr(target, '__iter__') and not isinstance(target, str))):
        target = list(target)
    modules_to_save = lora.get('modules_to_save', None)
    modules_to_save = list(modules_to_save) if modules_to_save is not None else []
    if extra_modules_to_save:
        for m in extra_modules_to_save:
            if m not in modules_to_save:
                modules_to_save.append(m)
    return LoraConfig(
        r=int(lora.r), lora_alpha=int(lora.lora_alpha),
        lora_dropout=float(lora.lora_dropout), bias=str(lora.bias),
        target_modules=target, modules_to_save=modules_to_save or None,
    )


def _strip_deprecated_keys(cfg):
    if 'latent_attends_action' in cfg.denoiser:
        del cfg.denoiser['latent_attends_action']


def _load_base_weights_with_report(denoiser, ckpt_path, model_key, rank):
    """strict=False load with classified missing/unexpected-key reporting so
    silent mismatches surface loudly on rank 0."""
    state = torch.load(ckpt_path, map_location='cpu')
    sd = state[model_key]
    incompat = denoiser.load_state_dict(sd, strict=False)

    def _is_cond(name):
        return any(k in name for k in COND_PARAM_KEYS)

    cond_missing = [k for k in incompat.missing_keys if _is_cond(k)]
    other_missing = [k for k in incompat.missing_keys if not _is_cond(k)]
    unexpected = list(incompat.unexpected_keys)
    if rank == 0:
        if cond_missing:
            print(f"[cond] {len(cond_missing)} conditioning params not in checkpoint "
                  f"→ zero-initialized (expected when conditioning a base checkpoint).")
        if other_missing:
            print("=" * 70)
            print(f"WARNING: {len(other_missing)} NON-conditioning params missing from "
                  f"the checkpoint — likely a wrong/incompatible base checkpoint:")
            for k in other_missing[:20]:
                print(f"    missing: {k}")
            if len(other_missing) > 20:
                print(f"    ... and {len(other_missing) - 20} more")
            print("=" * 70)
        if unexpected:
            print("=" * 70)
            print(f"WARNING: {len(unexpected)} checkpoint params UNEXPECTED "
                  f"(in ckpt, absent in model) — architecture drift:")
            for k in unexpected[:20]:
                print(f"    unexpected: {k}")
            if len(unexpected) > 20:
                print(f"    ... and {len(unexpected) - 20} more")
            print("=" * 70)
    return denoiser


# ---------------------------------------------------------------------------
# Model + forward-process construction
# ---------------------------------------------------------------------------

def build_models(cfg, device, local_rank, rank):
    _strip_deprecated_keys(cfg)

    # --- Conditioning must be enabled (guardrail). ---
    assert bool(cfg.denoiser.get('cond_adaln', False)), (
        "train_align_cond_new.py requires denoiser.cond_adaln=True, otherwise it "
        "would silently run as plain non-conditioned training. Set "
        "denoiser.cond_adaln=true (and denoiser.num_cond_classes>=3)."
    )
    n_classes = int(cfg.denoiser.get('num_cond_classes', 0))
    assert n_classes >= len(CLASSES), (
        f"denoiser.num_cond_classes ({n_classes}) must be >= {len(CLASSES)}."
    )

    tokenizer = load_tokenizer(
        cfg, device=device, max_num_forward_steps=cfg.denoiser.max_sequence_length
    )

    denoiser = DenoiserWrapper(cfg, max_num_forward_steps=cfg.denoiser.max_sequence_length)
    if cfg.dynamics_ckpt:
        if rank == 0:
            print(f"Loading dynamics weights (strict=False) from: {cfg.dynamics_ckpt}")
        denoiser = _load_base_weights_with_report(
            denoiser, cfg.dynamics_ckpt, model_key="model", rank=rank,
        )
    elif rank == 0:
        print("No dynamics_ckpt — denoiser (incl. conditioning) is randomly initialized.")

    # --- Forward processes (identical knobs to train_dynamics_uwm_new.py) ---
    unified_cfg = cfg.train.get("unified", {}) or {}
    diffuser = UnifiedForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        action_noise_std=float(unified_cfg.get("action_noise_std", 1.0)),
        theta_id_prob=float(unified_cfg.get("theta_id_prob", 0.15)),
        theta_policy_prob=float(unified_cfg.get("theta_policy_prob", 0.15)),
        theta_wm_prob=float(unified_cfg.get("theta_wm_prob", 0.15)),
        theta_continuum_prob=float(unified_cfg.get("theta_continuum_prob", 0.55)),
        profile_step_prob=float(unified_cfg.get("profile_step_prob", 0.5)),
        profile_progressive_prob=float(unified_cfg.get("profile_progressive_prob", 0.3)),
        profile_constant_prob=float(unified_cfg.get("profile_constant_prob", 0.2)),
        profile_diffusion_forcing_prob=float(unified_cfg.get("profile_diffusion_forcing_prob", 0.0)),
        profile_reverse_step_prob=float(unified_cfg.get("profile_reverse_step_prob", 0.0)),
        r_beta_alpha=float(unified_cfg.get("r_beta_alpha", 1.0)),
        r_beta_beta=float(unified_cfg.get("r_beta_beta", 1.0)),
        diffusion_forcing_bidir_prob=float(unified_cfg.get("diffusion_forcing_bidir_prob", 0.5)),
        device=device,
    )
    image_diffuser = ImageForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        action_noise_std=float(unified_cfg.get("action_noise_std", 1.0)),
        device=device,
    )
    video_pretraining_diffuser = VideoPretrainingForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        action_noise_std=float(unified_cfg.get("action_noise_std", 1.0)),
        bidir_prob=float(unified_cfg.get("pretraining_bidir_prob", 0.5)),
        device=device,
    )
    action_pretraining_diffuser = ActionPretrainingForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        action_noise_std=float(unified_cfg.get("action_noise_std", 1.0)),
        bidir_prob=float(unified_cfg.get("pretraining_bidir_prob", 0.5)),
        device=device,
    )
    bc_cfg = unified_cfg.get("block_causal", {}) or {}
    block_causal_sanity_diffuser = BlockCausalSanityForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        action_noise_std=float(unified_cfg.get("action_noise_std", 1.0)),
        block_sizes=list(bc_cfg.get("block_sizes", [1, 2, 4, 8, 16])),
        ctx_dropout_prob=float(bc_cfg.get("ctx_dropout_prob", 0.1)),
        ctx_tau_idx_min=bc_cfg.get("ctx_tau_idx_min", None),
        ctx_tau_idx_max=bc_cfg.get("ctx_tau_idx_max", None),
        couple_modality_tau=bool(bc_cfg.get("couple_modality_tau", False)),
        device=device,
    )

    tokenizer = tokenizer.to(device).eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    denoiser = denoiser.to(device)

    lora_enabled = bool(cfg.lora.get("enabled", False))
    freeze_backbone = bool(cfg.train.get("freeze_backbone", True))
    if lora_enabled:
        from peft import get_peft_model
        lora_cfg = build_lora_config(cfg, extra_modules_to_save=list(COND_PARAM_KEYS))
        denoiser = get_peft_model(denoiser, lora_cfg)
        if rank == 0:
            if not freeze_backbone:
                print("NOTE: lora.enabled=True forces a frozen base; "
                      "train.freeze_backbone is effectively ignored.")
            denoiser.print_trainable_parameters()
    elif freeze_backbone:
        n_cond = 0
        for name, p in denoiser.named_parameters():
            is_cond = any(k in name for k in COND_PARAM_KEYS)
            p.requires_grad_(is_cond)
            n_cond += int(is_cond)
        assert n_cond > 0, (
            f"freeze_backbone=True but no params matched {COND_PARAM_KEYS} — "
            f"nothing would train. Check naming / cond_adaln."
        )
        if rank == 0:
            print(f"freeze_backbone=True → training {n_cond} conditioning params only.")
    elif rank == 0:
        print("freeze_backbone=False → full fine-tune (backbone + conditioning).")

    if cfg.train.use_compile:
        if lora_enabled:
            if rank == 0:
                print("WARNING: use_compile=True with LoRA — skipping compile.")
        else:
            denoiser = torch.compile(denoiser, mode="max-autotune-no-cudagraphs", fullgraph=True)
            tokenizer = torch.compile(tokenizer, mode="max-autotune-no-cudagraphs", fullgraph=False)

    denoiser = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=False)

    return (
        tokenizer, denoiser, lora_enabled,
        diffuser, image_diffuser,
        video_pretraining_diffuser, action_pretraining_diffuser,
        block_causal_sanity_diffuser,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(cfg, rank, log_dir, wandb_run_id):
    if rank != 0:
        return None, None
    if log_dir is None:
        log_dir = cfg.output_dir
        os.makedirs(log_dir, exist_ok=True)
    else:
        print(f"Reusing log directory: {log_dir}")
    tb_log_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    if cfg.wandb.enable:
        if wandb_run_id is not None:
            wandb.init(project=cfg.wandb.project, id=wandb_run_id, resume="allow",
                       config=OmegaConf.to_container(cfg, resolve=True),
                       sync_tensorboard=True, dir=log_dir)
        else:
            wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name,
                       config=OmegaConf.to_container(cfg, resolve=True),
                       sync_tensorboard=True, dir=log_dir)
        wandb_run_id = wandb.run.id
    return SummaryWriter(log_dir=tb_log_dir), wandb_run_id


# ---------------------------------------------------------------------------
# LoRA save helpers
# ---------------------------------------------------------------------------

def save_lora_adapter(ckpt_path, epoch, global_update, model, optim, scheduler,
                      rank, wandb_run_id=None, log_dir=None):
    if rank != 0:
        return
    from peft import get_peft_model_state_dict
    peft_model = unwrap_model(model)
    adapter_state = get_peft_model_state_dict(peft_model)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({'epoch': epoch, 'global_update': global_update, 'adapter': adapter_state,
                'optim': optim.state_dict(), 'scheduler': scheduler.state_dict(),
                'wandb_run_id': wandb_run_id, 'log_dir': log_dir}, ckpt_path)
    print(f"[rank0] Saved LoRA adapter checkpoint to {ckpt_path}")


def save_lora_merged(model, ckpt_path, rank):
    if rank != 0:
        return
    peft_model = unwrap_model(model)
    base = peft_model.merge_and_unload()
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({'model': base.state_dict()}, ckpt_path)
    print(f"[rank0] Saved merged full checkpoint to {ckpt_path}")


# ---------------------------------------------------------------------------
# Per-window (class, source, branch, mode, crop) sampling
# ---------------------------------------------------------------------------

def sample_window(cfg, rank, device, generator,
                  effective_long_prob, image_prob,
                  mp_unified, mp_video, mp_action,
                  has_states, has_actions,
                  max_seq, short_seq_len):
    """Sample window-level routing on rank 0; broadcast as a length-5 long vector
    [class_idx, src_idx, branch_code, mode_code, crop_start]."""
    if rank == 0:
        class_idx = int(torch.randint(0, len(CLASSES), (1,), generator=generator).item())
        rule = CLASS_SOURCE_RULE[CLASSES[class_idx]]
        if rule == 'mixed':
            src_idx = 0 if (torch.rand(1, generator=generator).item()
                            < float(cfg.train.get('null_play_fraction', 0.5))) else 1
        else:
            src_idx = SOURCES.index(rule)

        r = torch.rand(1, generator=generator).item()
        if r < effective_long_prob:
            branch_code = 0  # long
        elif r < effective_long_prob + image_prob:
            branch_code = 1  # image
        else:
            branch_code = 2  # short

        if not has_actions:
            mode_code = 1     # video_pretraining
        elif not has_states:
            mode_code = 2     # action_pretraining
        else:
            rm = torch.rand(1, generator=generator).item()
            if rm < mp_unified:
                mode_code = 0
            elif rm < mp_unified + mp_video:
                mode_code = 1
            elif rm < mp_unified + mp_video + mp_action:
                mode_code = 2
            else:
                mode_code = 3

        if branch_code == 2 and max_seq > short_seq_len:
            crop_start = int(torch.randint(0, max_seq - short_seq_len + 1, (1,),
                                           generator=generator).item())
        else:
            crop_start = 0
        codes = torch.tensor([class_idx, src_idx, branch_code, mode_code, crop_start],
                             device=device, dtype=torch.long)
    else:
        codes = torch.zeros(5, device=device, dtype=torch.long)
    dist.broadcast(codes, src=0)
    c = [int(v) for v in codes]
    return c[0], SOURCES[c[1]], BRANCHES[c[2]], MODES[c[3]], c[4]


# ---------------------------------------------------------------------------
# Training loop (single pass of total_micro_steps)
# ---------------------------------------------------------------------------

def train_loop(
    play_iter, demo_iter, total_micro_steps,
    tokenizer, denoiser,
    diffuser, image_diffuser, video_pretraining_diffuser,
    action_pretraining_diffuser, block_causal_sanity_diffuser,
    optim, scheduler, tb_writer, cfg, rank, device,
    global_update, log_dir, wandb_run_id, loss_scaler,
    trainable_params, lora_enabled, rng,
):
    denoiser.train()

    short_bs = int(cfg.train.batch_per_gpu)
    long_bs = int(cfg.train.get("long_seq_batch_per_gpu", short_bs))
    image_bs = int(cfg.train.get("image_batch_per_gpu", short_bs))
    loader_bs = max(short_bs, long_bs)
    short_seq_len = int(cfg.train.get("short_sequence_length", cfg.denoiser.context_length))
    max_seq = int(cfg.denoiser.max_sequence_length)
    long_seq_prob = float(cfg.train.get("long_seq_prob", 0.0))
    image_prob = float(cfg.train.get("image_prob", 0.0))
    assert short_seq_len <= max_seq
    effective_long_prob = long_seq_prob if max_seq > short_seq_len else 0.0
    assert 0.0 <= image_prob <= 1.0
    assert effective_long_prob + image_prob <= 1.0 + 1e-6
    max_image_rows = loader_bs * max_seq
    assert image_bs <= max_image_rows

    unified_cfg = cfg.train.get("unified", {}) or {}
    causal_eps = float(unified_cfg.get("causal_eps", 1e-3))
    ramp_beta = float(unified_cfg.get("ramp_beta", 1.0))
    mode_probs_cfg = unified_cfg.get("mode_probs", {}) or {}
    _mp_u = float(mode_probs_cfg.get("unified", 1.0 / 3.0))
    _mp_v = float(mode_probs_cfg.get("video_pretraining", 1.0 / 3.0))
    _mp_a = float(mode_probs_cfg.get("action_pretraining", 1.0 / 3.0))
    _mp_b = float(mode_probs_cfg.get("block_causal_sanity", 0.0))
    _mp_tot = _mp_u + _mp_v + _mp_a + _mp_b
    assert _mp_tot > 0, "at least one of train.unified.mode_probs.* must be > 0"
    mp_unified, mp_video, mp_action = _mp_u / _mp_tot, _mp_v / _mp_tot, _mp_a / _mp_tot

    dataset_cfg = cfg.get("dataset", {}) or {}
    has_states = bool(dataset_cfg.get("has_states", True))
    has_actions = bool(dataset_cfg.get("has_actions", True))
    assert has_states or has_actions

    train_reward = bool(cfg.denoiser.get("train_reward_model", False))
    reward_weight = float(cfg.train.get("reward_weight", 1.0))

    epoch_start = time.perf_counter()
    epoch_loss_sum = 0.0
    num_updates = 0

    accum_obs_flow = accum_act_flow = 0.0
    accum_obs_flow_scaled = accum_act_flow_scaled = 0.0
    accum_reward = accum_total = accum_total_raw = 0.0
    accum_class = 'null'
    accum_source = 'play'
    accum_branch = 'short'
    accum_mode = 'unified'
    crop_start = 0
    class_count = {c: 0 for c in CLASSES}

    for step_idx in range(total_micro_steps):
        micro_idx = step_idx % cfg.train.accum_grad_steps
        is_last_micro = micro_idx == cfg.train.accum_grad_steps - 1

        if micro_idx == 0:
            optim.zero_grad(set_to_none=True)
            class_idx, accum_source, accum_branch, accum_mode, crop_start = sample_window(
                cfg, rank, device, rng,
                effective_long_prob, image_prob,
                mp_unified, mp_video, mp_action,
                has_states, has_actions, max_seq, short_seq_len,
            )
            accum_class = CLASSES[class_idx]
            class_count[accum_class] += 1

        batch = next(play_iter if accum_source == 'play' else demo_iter)
        class_idx = CLASSES.index(accum_class)

        images = batch["image"].to(device, non_blocking=True).to(torch.bfloat16)
        actions = (batch["action"].to(device, non_blocking=True)
                   .to(torch.bfloat16)[:, :, :cfg.denoiser.n_actions].unsqueeze(-2))
        rewards = batch.get("reward", None)
        if rewards is not None:
            rewards = rewards.to(device, non_blocking=True).float()
            if rewards.dim() == 3 and rewards.shape[-1] == 1:
                rewards = rewards.squeeze(-1)

        # --- Branch slicing ---
        if accum_branch == "short":
            images = images[:short_bs, crop_start:crop_start + short_seq_len]
            actions = actions[:short_bs, crop_start:crop_start + short_seq_len]
            if rewards is not None:
                rewards = rewards[:short_bs, crop_start:crop_start + short_seq_len]
        elif accum_branch == "long":
            images = images[:long_bs]
            actions = actions[:long_bs]
            if rewards is not None:
                rewards = rewards[:long_bs]
        elif accum_branch == "image":
            B_in, T_in = images.shape[:2]
            images = images.reshape(B_in * T_in, 1, *images.shape[2:])
            actions = actions.reshape(B_in * T_in, 1, *actions.shape[2:])
            if rewards is not None:
                rewards = rewards.reshape(B_in * T_in, 1)
            perm = torch.randperm(B_in * T_in, device=images.device)[:image_bs]
            images = images[perm]
            actions = actions[perm]
            if rewards is not None:
                rewards = rewards[perm]

        cond_class = torch.full((images.shape[0],), class_idx, device=device, dtype=torch.long)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_clean = tokenizer.encode(images).detach().clone()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if accum_branch == "image":
                info = image_diffuser(z_clean, actions)
                losses = compute_image_loss(
                    info, denoiser, device=device,
                    rewards=rewards if train_reward else None,
                    scaler=loss_scaler, cond_class=cond_class,
                )
            elif accum_mode == "video_pretraining":
                info = video_pretraining_diffuser(z_clean, actions)
                losses = compute_video_pretraining_loss(
                    info, denoiser, device=device,
                    rewards=rewards if train_reward else None,
                    scaler=loss_scaler, cond_class=cond_class,
                )
            elif accum_mode == "action_pretraining":
                info = action_pretraining_diffuser(z_clean, actions)
                losses = compute_action_pretraining_loss(
                    info, denoiser, device=device,
                    rewards=rewards if train_reward else None,
                    scaler=loss_scaler, cond_class=cond_class,
                )
            elif accum_mode == "block_causal_sanity":
                info = block_causal_sanity_diffuser(z_clean, actions)
                losses = compute_block_causal_sanity_loss(
                    info, denoiser, device=device,
                    rewards=rewards if train_reward else None,
                    scaler=None, cond_class=cond_class,
                )
            else:  # unified
                info = diffuser(z_clean, actions)
                losses = compute_unified_uwm_loss(
                    info, denoiser, device=device,
                    causal_eps=causal_eps, ramp_beta=ramp_beta,
                    rewards=rewards if train_reward else None,
                    scaler=loss_scaler, cond_class=cond_class,
                )
            obs_flow_loss = losses["obs_flow_loss"]
            act_flow_loss = losses["act_flow_loss"]
            reward_loss = losses["reward_loss"]
            total_loss = obs_flow_loss + act_flow_loss
            if reward_loss is not None:
                total_loss = total_loss + reward_weight * reward_loss
            loss_micro = total_loss / cfg.train.accum_grad_steps

        loss_micro.backward()

        accum_obs_flow += losses["obs_flow_loss_raw"].item()
        accum_act_flow += losses["act_flow_loss_raw"].item()
        accum_total_raw += (losses["obs_flow_loss_raw"]
                            + losses["act_flow_loss_raw"]).item() / cfg.train.accum_grad_steps
        accum_obs_flow_scaled += obs_flow_loss.detach().item()
        accum_act_flow_scaled += act_flow_loss.detach().item()
        if reward_loss is not None:
            accum_reward += reward_loss.item()
        accum_total += loss_micro.item()

        if is_last_micro:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(cfg.train.get("clip_grad_norm", 1.0)))
            optim.step()
            scheduler.step()
            global_update += 1
            total_tensor = torch.tensor([accum_total], device=device)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.AVG)
            sync_loss = total_tensor.item()
            epoch_loss_sum += sync_loss
            num_updates += 1

            if rank == 0:
                lr = scheduler.get_last_lr()[0]
                tb_writer.add_scalar("train/total_loss", accum_total_raw, global_update)
                tb_writer.add_scalar("train/obs_flow_loss", accum_obs_flow, global_update)
                tb_writer.add_scalar("train/act_flow_loss", accum_act_flow, global_update)
                tb_writer.add_scalar("train/total_loss_scaled", sync_loss, global_update)
                tb_writer.add_scalar("train/obs_flow_loss_scaled", accum_obs_flow_scaled, global_update)
                tb_writer.add_scalar("train/act_flow_loss_scaled", accum_act_flow_scaled, global_update)
                if train_reward:
                    tb_writer.add_scalar("train/reward_loss", accum_reward, global_update)
                tb_writer.add_scalar("train/lr", lr, global_update)
                # Per-(class × branch[/mode]) namespace.
                if accum_branch == "image":
                    ns = f"train/{accum_class}/image"
                else:
                    ns = f"train/{accum_class}/{accum_branch}_{accum_mode}"
                tb_writer.add_scalar(f"{ns}/total_loss", accum_total_raw, global_update)
                tb_writer.add_scalar(f"{ns}/obs_flow_loss", accum_obs_flow, global_update)
                tb_writer.add_scalar(f"{ns}/act_flow_loss", accum_act_flow, global_update)
                # Mix traces.
                for c in CLASSES:
                    tb_writer.add_scalar(f"mix/class_{c}", class_count[c], global_update)
                tb_writer.add_scalar("train/class_code", CLASSES.index(accum_class), global_update)
                tb_writer.add_scalar("train/branch_code",
                                     {"long": 0, "image": 1, "short": 2}[accum_branch], global_update)
                tb_writer.add_scalar("train/mode_code",
                                     -1 if accum_branch == "image" else MODES.index(accum_mode),
                                     global_update)

                if global_update % cfg.print_every == 0:
                    tag = "image" if accum_branch == "image" else f"{accum_branch}/{accum_mode}"
                    print(f"  [step {global_update}]  [{accum_class}|{tag}]"
                          f"  loss: {accum_total_raw:.4f}  obs: {accum_obs_flow:.4f}"
                          f"  act: {accum_act_flow:.4f}  lr: {lr:.2e}")

                if global_update % cfg.save_every == 0:
                    if lora_enabled:
                        save_lora_adapter(
                            ckpt_path=os.path.join(log_dir, f"adapter_{global_update}.pt"),
                            epoch=0, global_update=global_update, model=denoiser,
                            optim=optim, scheduler=scheduler, rank=rank,
                            wandb_run_id=wandb_run_id, log_dir=log_dir)
                    else:
                        save_ddp_checkpoint(
                            ckpt_path=os.path.join(log_dir, f"{global_update}.pt"),
                            epoch=0, global_update=global_update, model=denoiser,
                            optim=optim, scheduler=scheduler, rank=rank,
                            wandb_run_id=wandb_run_id, log_dir=log_dir)

            accum_obs_flow = accum_act_flow = 0.0
            accum_obs_flow_scaled = accum_act_flow_scaled = 0.0
            accum_reward = accum_total = accum_total_raw = 0.0
            class_count = {c: 0 for c in CLASSES}

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = epoch_loss_sum / num_updates if num_updates > 0 else 0.0
    if rank == 0:
        print(f"\n{'='*60}\nTraining loop done. Avg loss: {avg_loss:.6f}  "
              f"Time: {epoch_time:.2f}s\n{'='*60}\n")
    return global_update, avg_loss, epoch_time


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@record
@hydra.main(config_path="config", config_name="align/pushT-cond-new", version_base=None)
def main(cfg: DictConfig):
    torch.backends.cuda.matmul.allow_tf32 = cfg.train.enable_fast_matmul

    rank, local_rank, world_size, device = setup_distributed()
    torch.manual_seed(cfg.seed + rank)

    if rank == 0:
        print(f"Distributed training: {world_size} GPU(s)")
        eff = cfg.train.batch_per_gpu * world_size * cfg.train.accum_grad_steps
        print(f"Effective global batch size: {eff}")

    # --- Data ---
    play_loader, play_sampler, demo_loader, demo_sampler = build_dataloaders(cfg, rank, world_size)
    if rank == 0:
        print(f"Play loader: {len(play_loader)} steps/pass (dir={cfg.dataset.play_data_dir})")
        print(f"Demo loader: {len(demo_loader)} steps/pass (dir={cfg.dataset.demo_data_dir})")
        print(f"Classes (equal prob): {CLASSES}   null_play_fraction="
              f"{cfg.train.get('null_play_fraction', 0.5)}")
        print(f"Class → source rule: {CLASS_SOURCE_RULE}")

    # --- Models + forward processes ---
    if rank == 0:
        print("Building models...")
    (tokenizer, denoiser, lora_enabled,
     diffuser, image_diffuser, video_pretraining_diffuser,
     action_pretraining_diffuser, block_causal_sanity_diffuser) = build_models(
        cfg, device, local_rank, rank)
    trainable_params = [p for p in denoiser.parameters() if p.requires_grad]
    if rank == 0:
        n_tr = sum(p.numel() for p in trainable_params)
        n_tot = sum(p.numel() for p in denoiser.parameters())
        freeze_backbone = bool(cfg.train.get("freeze_backbone", True))
        tag = " [LoRA]" if lora_enabled else (" [frozen backbone]" if freeze_backbone else " [full FT]")
        print(f"Trainable: {n_tr:,} / {n_tot:,} ({100 * n_tr / n_tot:.3f}%)" + tag)

    # --- Optimizer + scheduler ---
    optim = torch.optim.AdamW(
        trainable_params, lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)))
    total_grad_steps = int(cfg.train.num_training_steps)
    total_micro_steps = total_grad_steps * cfg.train.accum_grad_steps
    warmup_cfg = cfg.train.get("warmup_steps", None)
    warmup_steps = int(warmup_cfg) if warmup_cfg is not None else int(0.05 * total_grad_steps)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_grad_steps)
    if rank == 0:
        print(f"Schedule: total grad steps={total_grad_steps}, "
              f"micro-steps={total_micro_steps}, warmup={warmup_steps}")

    # --- RMS loss scaler (persists across the run; same semantics as dynamics) ---
    _unified_cfg = cfg.train.get("unified", {}) or {}
    _rms_decay = float(_unified_cfg.get("rms_scale_decay", 0.99))
    _rms_enabled = bool(_unified_cfg.get("rms_scale_loss", True))
    loss_scaler = RMSLossScaler(decay=_rms_decay) if _rms_enabled else None
    if rank == 0:
        print(f"RMS loss scaler: {'enabled' if _rms_enabled else 'disabled'}")

    # --- Resume (model+optim) or fresh ---
    wandb_run_id = cfg.wandb.run_name
    log_dir = None
    global_update = 0
    if cfg.reload_checkpoint is not None:
        if rank == 0:
            print(f"Resuming training state from: {cfg.reload_checkpoint}")
        _, global_update, _cum, wandb_run_id, log_dir = load_ddp_checkpoint(
            ckpt_path=cfg.reload_checkpoint, model=denoiser,
            optim=optim, scheduler=scheduler, rank=rank)
    elif rank == 0:
        print("Fresh optimizer/scheduler (weights from dynamics_ckpt + zero-init conditioning).")

    # --- Logging ---
    tb_writer, wandb_run_id_new = setup_logging(cfg, rank, log_dir, wandb_run_id)
    if rank == 0:
        log_dir = cfg.output_dir if log_dir is None else log_dir
        wandb_run_id = wandb_run_id_new
    obj_list = [log_dir, wandb_run_id]
    dist.broadcast_object_list(obj_list, src=0)
    log_dir, wandb_run_id = obj_list

    dist.barrier()
    if rank == 0:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    dist.barrier()

    # --- Cycling iterators ---
    epoch_ref = [0]
    play_iter = _cycling_iter(play_loader, sampler=play_sampler, epoch_ref=epoch_ref)
    demo_iter = _cycling_iter(demo_loader, sampler=demo_sampler, epoch_ref=epoch_ref,
                              base_offset=104729)
    rng = torch.Generator(device='cpu').manual_seed(cfg.seed + 31337)

    if rank == 0:
        print(f"Starting training: {total_grad_steps} grad steps "
              f"({total_micro_steps} micro-steps, accum={cfg.train.accum_grad_steps}).")

    global_update, avg_loss, _ = train_loop(
        play_iter=play_iter, demo_iter=demo_iter, total_micro_steps=total_micro_steps,
        tokenizer=tokenizer, denoiser=denoiser,
        diffuser=diffuser, image_diffuser=image_diffuser,
        video_pretraining_diffuser=video_pretraining_diffuser,
        action_pretraining_diffuser=action_pretraining_diffuser,
        block_causal_sanity_diffuser=block_causal_sanity_diffuser,
        optim=optim, scheduler=scheduler, tb_writer=tb_writer, cfg=cfg, rank=rank,
        device=device, global_update=global_update, log_dir=log_dir,
        wandb_run_id=wandb_run_id, loss_scaler=loss_scaler,
        trainable_params=trainable_params, lora_enabled=lora_enabled, rng=rng,
    )

    if rank == 0:
        if lora_enabled:
            save_lora_adapter(
                ckpt_path=os.path.join(log_dir, f"adapter_{global_update}.pt"),
                epoch=0, global_update=global_update, model=denoiser,
                optim=optim, scheduler=scheduler, rank=rank,
                wandb_run_id=wandb_run_id, log_dir=log_dir)
        else:
            save_ddp_checkpoint(
                ckpt_path=os.path.join(log_dir, f"{global_update}.pt"),
                epoch=0, global_update=global_update, model=denoiser,
                optim=optim, scheduler=scheduler, rank=rank,
                wandb_run_id=wandb_run_id, log_dir=log_dir)
    dist.barrier()

    if lora_enabled and bool(cfg.lora.get("save_merged_final", True)):
        save_lora_merged(denoiser, os.path.join(log_dir, "final_merged.pt"), rank)
    dist.barrier()

    if rank == 0:
        cur = torch.cuda.memory_allocated(device) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\n{'='*60}\nTraining Complete!  Avg loss: {avg_loss:.6f}")
        print(f"  GPU memory: {cur:.2f} GB current / {peak:.2f} GB peak\n{'='*60}")
        tb_writer.close()
        if cfg.wandb.enable:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
