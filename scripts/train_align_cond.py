"""AdaLN-Zero class-conditioning post-training for dynamics alignment.

Successor to `scripts/train_align_mix.py`. The mode→dataset *routing* idea did
not align the world model as hoped; this script replaces it with **provenance
conditioning** via an AdaLN-Zero retrofit on the denoiser (see
`dreamerv4uwm/models/blocks.py` / `dynamics.py`, `denoiser.cond_adaln`):

1. **3 discrete classes** `null` / `play` / `demo` (indices 0/1/2; null = 0 =
   unconditioned). The class is fed to the denoiser as `cond_class` and modulates
   every transformer layer's RMSNorms through a zero-init AdaLN projection — so at
   day 0 (and whenever `cond_class` is None) the network is bit-identical to the
   pretrained model.

2. **Equal-probability class sampling** (1/3 each), independent of the diffusion
   `mode` (which is still drawn from `cfg.train.mode_weights`). Every mode
   (`wm`/`policy`/`forcing`/…) runs within every class.

3. **Class → dataset routing**:
     - `play` → play data only
     - `demo` → demo data only
     - `null` → both (per step: play with prob `null_play_fraction`, else demo)

4. **Backbone freeze is a flag** `cfg.train.freeze_backbone` (default True,
   ControlNet-style: only the class embedder + per-layer AdaLN projections train,
   the pretrained backbone is frozen). `cfg.lora.enabled` is an independent flag;
   when on, PEFT freezes the base and the conditioning params are kept trainable
   via `modules_to_save`.

5. **Init-from-pretrained, fresh optimizer**: `cfg.dynamics_ckpt` supplies base
   weights via a `strict=False` load (the base ckpt lacks the conditioning keys,
   which are zero-initialized). Leave `cfg.reload_checkpoint=null` to start the
   optimizer/scheduler from scratch.

Guardrails throughout fail loud (rank-0 warnings / asserts) rather than silently
degrading to the old non-conditioned pipeline — see `build_models`,
`sample_mode_and_class`, and the conditioning assertion in `main`.
"""

import math
import os
import random
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
from dreamerv4uwm.loss import UWMForwardProcess, compute_uwm_loss
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
# Modes (diffusion task) and conditioning classes / sources
# ---------------------------------------------------------------------------
#
# Stable indexing for cross-rank broadcasting.
MODES = ['wm', 'forcing', 'policy', 'id', 'video']
CLASSES = ['null', 'play', 'demo']     # index 0/1/2 — null = unconditioned
SOURCES = ['play', 'demo']
# Class → dataset routing. 'null' is resolved per-step via null_play_fraction.
CLASS_SOURCE_RULE = {
    'null':  'mixed',  # both play and demo (null_play_fraction by default 0.5)
    'play':  'play',   # play data only
    'demo':  'demo',   # demo data only
}
# Substrings identifying the conditioning params (everything else = backbone).
COND_PARAM_KEYS = ('adaln', 'class_embedder')


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-8):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / optimizer.defaults["lr"], cosine_decay)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Dataloaders (identical to train_align_mix.py)
# ---------------------------------------------------------------------------

def build_dataloaders(cfg, rank, world_size):
    """Build (play_loader, play_sampler, demo_loader, demo_sampler)."""
    window = int(cfg.denoiser.max_sequence_length)
    play_bs = int(cfg.train.batch_per_gpu)
    demo_bs = int(cfg.train.get('demo_batch_per_gpu', cfg.train.batch_per_gpu))
    kind = str(cfg.dataset.get('kind', 'sharded_hdf5'))

    play_loader, play_sampler, _ = create_distributed_dataloader(
        data_dir=cfg.dataset.play_data_dir,
        window_size=window,
        batch_size=play_bs,
        rank=rank, world_size=world_size,
        num_workers=cfg.train.num_workers,
        stride=1, seed=cfg.seed,
        split='train',
        train_fraction=cfg.dataset.train_episodes_fraction,
        split_seed=cfg.dataset.split_seed,
        shuffle=True, drop_last=True,
        absolute_actions=cfg.train.absolute_actions,
        kind=kind,
    )
    demo_loader, demo_sampler, _ = create_distributed_dataloader(
        data_dir=cfg.dataset.demo_data_dir,
        window_size=window,
        batch_size=demo_bs,
        rank=rank, world_size=world_size,
        num_workers=max(1, cfg.train.num_workers // 2),
        stride=1, seed=cfg.seed + 7919,   # decorrelate from play
        split='train',
        train_fraction=cfg.dataset.train_episodes_fraction,
        split_seed=cfg.dataset.split_seed,
        shuffle=True, drop_last=True,
        absolute_actions=cfg.train.absolute_actions,
        kind=kind,
    )
    return play_loader, play_sampler, demo_loader, demo_sampler


def _cycling_iter(loader, sampler=None, epoch_ref=None, base_offset=0):
    """Infinite iterator over a DataLoader; advances DistributedSampler epoch on
    each restart so we don't see identical shuffles across passes."""
    pass_idx = 0
    while True:
        if sampler is not None:
            sampler.set_epoch((epoch_ref[0] if epoch_ref else 0) * 10_000
                              + base_offset + pass_idx)
        for b in loader:
            yield b
        pass_idx += 1


# ---------------------------------------------------------------------------
# Mode + class + source sampling
# ---------------------------------------------------------------------------

def sample_mode_and_class(cfg, rank, device, generator=None):
    """Sample (mode, class, source) jointly on rank 0; broadcast to all ranks.

    - mode  ~ cfg.train.mode_weights (categorical).
    - class ~ uniform over {null, play, demo} (equal probability).
    - source: play→play, demo→demo, null→ play w.p. null_play_fraction else demo.

    Returns: (mode_str, class_str, source_str).
    """
    if rank == 0:
        weights = torch.tensor(
            [float(cfg.train.mode_weights.get(m, 0.0)) for m in MODES],
            device='cpu', dtype=torch.float64,
        )
        assert weights.sum() > 0, "At least one mode must have positive weight"
        mode_idx = int(torch.multinomial(weights, 1, generator=generator).item())
        # Equal-probability class draw.
        class_idx = int(torch.randint(0, len(CLASSES), (1,), generator=generator).item())
        cls = CLASSES[class_idx]
        rule = CLASS_SOURCE_RULE[cls]
        if rule == 'mixed':
            src_idx = 0 if (torch.rand(1, generator=generator).item()
                            < float(cfg.train.get('null_play_fraction', 0.5))) else 1
        else:
            src_idx = SOURCES.index(rule)
        codes = torch.tensor([mode_idx, class_idx, src_idx], device=device, dtype=torch.long)
    else:
        codes = torch.zeros(3, device=device, dtype=torch.long)
    dist.broadcast(codes, src=0)
    return (MODES[int(codes[0])], CLASSES[int(codes[1])], SOURCES[int(codes[2])])


# ---------------------------------------------------------------------------
# Optional LoRA wrap
# ---------------------------------------------------------------------------

def build_lora_config(cfg: DictConfig, extra_modules_to_save=None):
    """Construct a `LoraConfig` from cfg.lora. `extra_modules_to_save` ensures the
    conditioning params stay trainable when PEFT freezes the base."""
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
        r=int(lora.r),
        lora_alpha=int(lora.lora_alpha),
        lora_dropout=float(lora.lora_dropout),
        bias=str(lora.bias),
        target_modules=target,
        modules_to_save=modules_to_save or None,
    )


def _strip_deprecated_keys(cfg):
    if 'latent_attends_action' in cfg.denoiser:
        del cfg.denoiser['latent_attends_action']


# ---------------------------------------------------------------------------
# Checkpoint load with an explicit missing/unexpected-key report (guardrail)
# ---------------------------------------------------------------------------

def _load_base_weights_with_report(denoiser, ckpt_path, model_key, rank):
    """Load base weights with strict=False and classify the resulting
    missing/unexpected keys so silent mismatches surface loudly on rank 0.

    Expected on a first conditioning run: the conditioning params
    (adaln / class_embedder) are absent from a base checkpoint → zero-initialized.
    Anything else missing, or any unexpected key, is a real mismatch (e.g. wrong
    base checkpoint or arch drift) and is WARNED about.
    """
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
            print(f"WARNING: {len(other_missing)} NON-conditioning params are missing "
                  f"from the checkpoint — this is NOT expected and likely means a "
                  f"wrong/incompatible base checkpoint:")
            for k in other_missing[:20]:
                print(f"    missing: {k}")
            if len(other_missing) > 20:
                print(f"    ... and {len(other_missing) - 20} more")
            print("=" * 70)
        if unexpected:
            print("=" * 70)
            print(f"WARNING: {len(unexpected)} checkpoint params are UNEXPECTED "
                  f"(present in ckpt, absent in model) — architecture drift:")
            for k in unexpected[:20]:
                print(f"    unexpected: {k}")
            if len(unexpected) > 20:
                print(f"    ... and {len(unexpected) - 20} more")
            print("=" * 70)
    return denoiser


def build_models(cfg, device, local_rank, rank):
    _strip_deprecated_keys(cfg)

    # --- Conditioning must be enabled for this script (guardrail). ---
    assert bool(cfg.denoiser.get('cond_adaln', False)), (
        "train_align_cond.py requires denoiser.cond_adaln=True, otherwise it would "
        "silently run as plain non-conditioned mixed training. Set "
        "denoiser.cond_adaln=true (and denoiser.num_cond_classes>=3) in the config."
    )
    n_classes = int(cfg.denoiser.get('num_cond_classes', 0))
    assert n_classes >= len(CLASSES), (
        f"denoiser.num_cond_classes ({n_classes}) must be >= {len(CLASSES)} "
        f"for classes {CLASSES}."
    )

    tokenizer = load_tokenizer(
        cfg, device=device, max_num_forward_steps=cfg.denoiser.max_sequence_length
    )
    tokenizer = tokenizer.to(device).eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    # Build the (conditioned) denoiser and load base weights with strict=False.
    denoiser = DenoiserWrapper(
        cfg, max_num_forward_steps=cfg.denoiser.max_sequence_length,
    )
    if cfg.dynamics_ckpt:
        if rank == 0:
            print(f"Loading dynamics weights (strict=False) from: {cfg.dynamics_ckpt}")
        denoiser = _load_base_weights_with_report(
            denoiser, cfg.dynamics_ckpt, model_key='model', rank=rank,
        )
    elif rank == 0:
        print("No dynamics_ckpt — denoiser (incl. conditioning) is randomly initialized.")
    denoiser = denoiser.to(device)

    lora_enabled = bool(cfg.lora.get('enabled', False))
    freeze_backbone = bool(cfg.train.get('freeze_backbone', True))

    if lora_enabled:
        # PEFT freezes the base; keep conditioning params trainable via
        # modules_to_save. freeze_backbone is implied (base linears are frozen).
        from peft import get_peft_model
        lora_cfg = build_lora_config(cfg, extra_modules_to_save=list(COND_PARAM_KEYS))
        denoiser = get_peft_model(denoiser, lora_cfg)
        if rank == 0:
            if not freeze_backbone:
                print("NOTE: lora.enabled=True forces a frozen base; "
                      "train.freeze_backbone is effectively ignored.")
            denoiser.print_trainable_parameters()
    elif freeze_backbone:
        # ControlNet-style: train only the conditioning params.
        n_cond = 0
        for name, p in denoiser.named_parameters():
            is_cond = any(k in name for k in COND_PARAM_KEYS)
            p.requires_grad_(is_cond)
            n_cond += int(is_cond)
        assert n_cond > 0, (
            "freeze_backbone=True but no params matched "
            f"{COND_PARAM_KEYS} — nothing would train. Check naming / cond_adaln."
        )
        if rank == 0:
            print(f"freeze_backbone=True → training {n_cond} conditioning params only.")
    else:
        if rank == 0:
            print("freeze_backbone=False → full fine-tune (backbone + conditioning).")

    if cfg.train.use_compile:
        if lora_enabled:
            if rank == 0:
                print("WARNING: use_compile=True with LoRA — skipping compile.")
        else:
            denoiser = torch.compile(denoiser, mode='max-autotune-no-cudagraphs', fullgraph=True)
            tokenizer = torch.compile(tokenizer, mode='max-autotune-no-cudagraphs', fullgraph=False)

    denoiser = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=False)

    fc_noise = cfg.train.get('forcing_context_noise', {})
    diffuser = UWMForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        mode_weights=OmegaConf.to_container(cfg.train.mode_weights, resolve=True),
        forcing_context_noise_bias=float(fc_noise.get('bias', 0.0)),
        forcing_context_noise_alpha=float(fc_noise.get('alpha', 0.5)),
        forcing_context_noise_beta=float(fc_noise.get('beta', 2.0)),
        forcing_mask_actions=bool(cfg.train.get('forcing_mask_actions', False)),
        horizon_aware=bool(cfg.denoiser.get('horizon_aware', False)),
        device=device,
    )
    return tokenizer, denoiser, diffuser, lora_enabled


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(cfg, rank, log_dir, wandb_run_id):
    if rank != 0:
        return None, None

    if log_dir is None:
        log_dir = cfg.output_dir
        os.makedirs(log_dir, exist_ok=True)
    else:
        print(f"Reusing log directory: {log_dir}")

    tb_log_dir = os.path.join(log_dir, 'tensorboard')
    os.makedirs(tb_log_dir, exist_ok=True)

    if cfg.wandb.enable:
        if wandb_run_id is not None:
            wandb.init(
                project=cfg.wandb.project, id=wandb_run_id, resume='allow',
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True, dir=log_dir,
            )
        else:
            wandb.init(
                project=cfg.wandb.project, name=cfg.wandb.run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True, dir=log_dir,
            )
        wandb_run_id = wandb.run.id

    return SummaryWriter(log_dir=tb_log_dir), wandb_run_id


# ---------------------------------------------------------------------------
# LoRA save helpers (used only when cfg.lora.enabled)
# ---------------------------------------------------------------------------

def save_lora_adapter(ckpt_path, epoch, global_update, model, optim, scheduler,
                     rank, wandb_run_id=None, log_dir=None):
    if rank != 0:
        return
    from peft import get_peft_model_state_dict
    peft_model = unwrap_model(model)
    adapter_state = get_peft_model_state_dict(peft_model)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'global_update': global_update,
        'adapter': adapter_state,
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'wandb_run_id': wandb_run_id,
        'log_dir': log_dir,
    }, ckpt_path)
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
# One-epoch loop
# ---------------------------------------------------------------------------

def train_epoch(
    epoch, play_iter, demo_iter, steps_in_epoch,
    tokenizer, denoiser, diffuser,
    optim, scheduler, tb_writer,
    cfg, rank, device, global_update, log_dir, wandb_run_id,
    trainable_params, lora_enabled, rng,
):
    """Run `steps_in_epoch` micro-steps; each samples a (mode, class, source) on
    rank 0, pulls the next batch from the corresponding loader, and conditions
    the denoiser on the sampled class."""
    denoiser.train()
    epoch_start = time.perf_counter()
    epoch_loss_sum = 0.0
    num_updates = 0

    accum_total = 0.0
    accum_obs = 0.0
    accum_act = 0.0
    mode_count = {m: 0 for m in MODES}
    class_count = {c: 0 for c in CLASSES}
    src_count = {s: 0 for s in SOURCES}
    # Per-(mode, class) loss accumulators — sparse, only the tuples sampled in
    # the current accumulation window get an emission.
    mc_total: dict = {}
    mc_obs:   dict = {}
    mc_act:   dict = {}
    mc_n:     dict = {}

    n_actions = cfg.denoiser.n_actions

    for step_idx in range(steps_in_epoch):
        micro_idx = step_idx % cfg.train.accum_grad_steps
        is_last_micro = micro_idx == cfg.train.accum_grad_steps - 1

        if micro_idx == 0:
            optim.zero_grad(set_to_none=True)

        mode, cls, source = sample_mode_and_class(cfg, rank, device, generator=rng)
        mode_count[mode] += 1
        class_count[cls] += 1
        src_count[source] += 1
        class_idx = CLASSES.index(cls)

        batch = next(play_iter if source == 'play' else demo_iter)

        images = batch['image'].to(device, non_blocking=True).to(torch.bfloat16)
        actions = (batch['action'].to(device, non_blocking=True)
                   .to(torch.bfloat16)[:, :, :n_actions].unsqueeze(-2))
        B = images.shape[0]
        cond_class = torch.full((B,), class_idx, device=device, dtype=torch.long)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                z_clean = tokenizer.encode(images).detach()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            info = diffuser(z_clean, actions, force_mode=mode)
            losses = compute_uwm_loss(
                info, denoiser, device=device,
                loss_weighting=str(cfg.train.get('loss_weighting', 'ramp')),
                cond_class=cond_class,
            )
            obs = losses['obs_flow_loss']
            act = losses['act_flow_loss']
            total = obs + act
            loss_micro = total / cfg.train.accum_grad_steps

        loss_micro.backward()

        obs_mean = obs.mean().item()
        act_mean = act.mean().item()
        accum_total += loss_micro.item()
        accum_obs   += obs_mean
        accum_act   += act_mean
        mc_key = (mode, cls)
        mc_total[mc_key] = mc_total.get(mc_key, 0.0) + obs_mean + act_mean
        mc_obs[mc_key]   = mc_obs.get(mc_key,   0.0) + obs_mean
        mc_act[mc_key]   = mc_act.get(mc_key,   0.0) + act_mean
        mc_n[mc_key]     = mc_n.get(mc_key,     0)   + 1

        if is_last_micro:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(cfg.train.clip_grad_norm))
            optim.step()
            scheduler.step()
            global_update += 1

            total_tensor = torch.tensor([accum_total], device=device)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.AVG)
            sync_loss = total_tensor.item()
            epoch_loss_sum += sync_loss
            num_updates += 1

            if rank == 0:
                inv = 1.0 / cfg.train.accum_grad_steps
                lr = scheduler.get_last_lr()[0]
                tb_writer.add_scalar('train/total_loss', sync_loss,       global_update)
                tb_writer.add_scalar('train/obs_flow',   accum_obs * inv, global_update)
                tb_writer.add_scalar('train/act_flow',   accum_act * inv, global_update)
                tb_writer.add_scalar('train/lr',         lr,              global_update)
                # Per-window mix (categorical traces; sum to accum_grad_steps).
                for m in MODES:
                    tb_writer.add_scalar(f'mix/mode_{m}',    mode_count[m],  global_update)
                for c in CLASSES:
                    tb_writer.add_scalar(f'mix/class_{c}',   class_count[c], global_update)
                for s in SOURCES:
                    tb_writer.add_scalar(f'mix/source_{s}',  src_count[s],   global_update)
                # Per-(mode, class) loss curves — sparse in the global-update axis.
                for key, n in mc_n.items():
                    if n == 0:
                        continue
                    m_, c_ = key
                    tag = f"{m_}_{c_}"
                    tb_writer.add_scalar(f"by_mc/{tag}/total", mc_total[key] / n, global_update)
                    tb_writer.add_scalar(f"by_mc/{tag}/obs",   mc_obs[key]   / n, global_update)
                    tb_writer.add_scalar(f"by_mc/{tag}/act",   mc_act[key]   / n, global_update)

                if global_update % cfg.print_every == 0:
                    mix_str = ' '.join(f'{c}={class_count[c]}' for c in CLASSES)
                    print(
                        f"  [step {global_update}]"
                        f"  loss: {sync_loss:.4f}"
                        f"  obs: {accum_obs * inv:.4f}"
                        f"  act: {accum_act * inv:.4f}"
                        f"  lr: {lr:.2e}"
                        f"  cls: {mix_str}"
                    )

                if global_update % cfg.save_every == 0:
                    if lora_enabled:
                        save_lora_adapter(
                            ckpt_path=os.path.join(log_dir, f'adapter_{global_update}.pt'),
                            epoch=epoch, global_update=global_update,
                            model=denoiser, optim=optim, scheduler=scheduler,
                            rank=rank, wandb_run_id=wandb_run_id, log_dir=log_dir,
                        )
                    else:
                        save_ddp_checkpoint(
                            ckpt_path=os.path.join(log_dir, f'{global_update}.pt'),
                            epoch=epoch, global_update=global_update,
                            model=denoiser, optim=optim, scheduler=scheduler,
                            rank=rank, wandb_run_id=wandb_run_id, log_dir=log_dir,
                        )

            accum_total = 0.0
            accum_obs   = 0.0
            accum_act   = 0.0
            mode_count  = {m: 0 for m in MODES}
            class_count = {c: 0 for c in CLASSES}
            src_count   = {s: 0 for s in SOURCES}
            mc_total.clear(); mc_obs.clear(); mc_act.clear(); mc_n.clear()

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = epoch_loss_sum / num_updates if num_updates > 0 else 0.0

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Avg loss:   {avg_loss:.6f}")
        print(f"  Epoch time: {epoch_time:.2f}s")
        print(f"{'='*60}\n")

    return global_update, avg_loss, epoch_time


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@record
@hydra.main(config_path='config', config_name='align/pushT-cond', version_base=None)
def main(cfg: DictConfig):
    torch.backends.cuda.matmul.allow_tf32 = cfg.train.enable_fast_matmul

    rank, local_rank, world_size, device = setup_distributed()
    torch.manual_seed(cfg.seed + rank)

    if rank == 0:
        print(f"Distributed training: {world_size} GPU(s)")
        eff = (cfg.train.batch_per_gpu * world_size * cfg.train.accum_grad_steps)
        print(f"Effective global batch size: {eff} (per-mode; demo-side uses demo_batch_per_gpu)")

    # --- Data ---
    play_loader, play_sampler, demo_loader, demo_sampler = build_dataloaders(
        cfg, rank, world_size,
    )
    if rank == 0:
        print(f"Play loader: {len(play_loader)} steps/pass   "
              f"(batch={cfg.train.batch_per_gpu}, dir={cfg.dataset.play_data_dir})")
        print(f"Demo loader: {len(demo_loader)} steps/pass   "
              f"(batch={cfg.train.get('demo_batch_per_gpu', cfg.train.batch_per_gpu)}, "
              f"dir={cfg.dataset.demo_data_dir})")
        print(f"Mode weights: {OmegaConf.to_container(cfg.train.mode_weights)}")
        print(f"Classes (equal prob): {CLASSES}")
        print(f"  null_play_fraction = {cfg.train.get('null_play_fraction', 0.5)}")
        print(f"Class → source rule: {CLASS_SOURCE_RULE}")

    # --- Models ---
    if rank == 0:
        print("Building models...")
    tokenizer, denoiser, diffuser, lora_enabled = build_models(
        cfg, device, local_rank, rank,
    )
    trainable_params = [p for p in denoiser.parameters() if p.requires_grad]
    if rank == 0:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in denoiser.parameters())
        freeze_backbone = bool(cfg.train.get('freeze_backbone', True))
        tag = " [LoRA]" if lora_enabled else (" [frozen backbone]" if freeze_backbone else " [full FT]")
        print(f"Trainable: {n_trainable:,} / {n_total:,} "
              f"({100 * n_trainable / n_total:.3f}%)" + tag)

    # --- Optimizer + scheduler ---
    optim = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.get('weight_decay', 0.0)),
    )
    total_grad_steps = int(cfg.train.num_training_steps)
    total_micro_steps = total_grad_steps * cfg.train.accum_grad_steps
    warmup_cfg = cfg.train.get('warmup_steps', None)
    if warmup_cfg is not None:
        warmup_steps = int(warmup_cfg)
    else:
        warmup_steps = int(0.05 * total_grad_steps)
    if rank == 0:
        print(f"Schedule: total grad steps={total_grad_steps}, "
              f"micro-steps={total_micro_steps}, warmup={warmup_steps}")
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_grad_steps)

    # --- Resume (model only; optimizer/scheduler start fresh by design) ---
    wandb_run_id = cfg.wandb.run_name
    log_dir = None
    global_update = 0
    if cfg.reload_checkpoint is not None:
        if rank == 0:
            print(f"Resuming training state from: {cfg.reload_checkpoint}")
        _, global_update, _cumulative_samples, wandb_run_id, log_dir = load_ddp_checkpoint(
            ckpt_path=cfg.reload_checkpoint,
            model=denoiser, optim=optim, scheduler=scheduler, rank=rank,
        )
    elif rank == 0:
        print("Starting from a fresh optimizer/scheduler "
              "(model weights initialized from cfg.dynamics_ckpt + zero-init conditioning).")

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
        OmegaConf.save(cfg, os.path.join(log_dir, 'config.yaml'))
    dist.barrier()

    # --- Cycling iterators ---
    epoch_ref = [0]
    play_iter = _cycling_iter(play_loader, sampler=play_sampler, epoch_ref=epoch_ref)
    demo_iter = _cycling_iter(demo_loader, sampler=demo_sampler, epoch_ref=epoch_ref,
                              base_offset=104729)  # decorrelate restart epochs

    # --- Per-rank RNG for mode/class/source sampling (rank 0 actually uses it). ---
    rng = torch.Generator(device='cpu').manual_seed(cfg.seed + 31337)

    if rank == 0:
        print(f"Starting training: {total_grad_steps} grad steps "
              f"({total_micro_steps} micro-steps, accum={cfg.train.accum_grad_steps}).")

    epoch_losses, epoch_times = [], []
    global_update, avg_loss, epoch_time = train_epoch(
        epoch=0,
        play_iter=play_iter, demo_iter=demo_iter,
        steps_in_epoch=total_micro_steps,
        tokenizer=tokenizer, denoiser=denoiser, diffuser=diffuser,
        optim=optim, scheduler=scheduler, tb_writer=tb_writer,
        cfg=cfg, rank=rank, device=device,
        global_update=global_update, log_dir=log_dir, wandb_run_id=wandb_run_id,
        trainable_params=trainable_params, lora_enabled=lora_enabled, rng=rng,
    )
    epoch_losses.append(avg_loss)
    epoch_times.append(epoch_time)

    if rank == 0:
        if lora_enabled:
            save_lora_adapter(
                ckpt_path=os.path.join(log_dir, f'adapter_{global_update}.pt'),
                epoch=0, global_update=global_update,
                model=denoiser, optim=optim, scheduler=scheduler,
                rank=rank, wandb_run_id=wandb_run_id, log_dir=log_dir,
            )
        else:
            save_ddp_checkpoint(
                ckpt_path=os.path.join(log_dir, f'{global_update}.pt'),
                epoch=0, global_update=global_update,
                model=denoiser, optim=optim, scheduler=scheduler,
                rank=rank, wandb_run_id=wandb_run_id, log_dir=log_dir,
            )
    dist.barrier()

    # --- Final merged save when using LoRA ---
    if lora_enabled and bool(cfg.lora.get('save_merged_final', True)):
        merged_path = os.path.join(log_dir, 'final_merged.pt')
        save_lora_merged(denoiser, merged_path, rank)
    dist.barrier()

    if rank == 0:
        cur = torch.cuda.memory_allocated(device) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\n{'='*60}")
        print("Training Complete!")
        if epoch_losses:
            print(f"  Avg loss:      {sum(epoch_losses) / len(epoch_losses):.6f}")
            print(f"  Avg epoch:     {sum(epoch_times) / len(epoch_times):.2f}s")
        print(f"  GPU memory:    {cur:.2f} GB current / {peak:.2f} GB peak")
        print(f"{'='*60}")
        tb_writer.close()
        if cfg.wandb.enable:
            wandb.finish()

    cleanup_distributed()


if __name__ == '__main__':
    main()
