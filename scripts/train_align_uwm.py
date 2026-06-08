"""Multi-dataset continued-training (alignment) on the unified-loss pipeline.

Goal: align UWM task/action dynamics with a `demo` dataset while preserving
broad world-model knowledge from a `play` dataset, with a routing scheme
designed to embed attractors that pull play-side behavior toward demo
regions of state space (few-shot imitation hypothesis).

Per-path operational structure (each path forces a specific corner of the
(θ, marginal) noise-square design):

    Path             | Forward process                       | θ / structure
    -----------------|---------------------------------------|--------------
    wm               | UnifiedForwardProcess(force_theta=π/2)| WM ray (state noisy, action clean)
    id               | UnifiedForwardProcess(force_theta=0)  | ID ray (action noisy, state clean)
    video            | VideoPretrainingForwardProcess        | right edge (state marginal, per-frame DF)
    action_sampler   | ActionPretrainingForwardProcess       | top edge  (action marginal, sequence-uniform)

Per-path source routing (default = "consistent" anchoring; no marginal mismatch):

    Path             | Default source           | Override knob
    -----------------|--------------------------|---------------------------
    wm               | mixed (50% play / 50% demo) | train.align.play_fractions.wm           (default 0.5)
    id               | demo only                | train.align.play_fractions.id           (default 0.0)
    video            | demo only                | train.align.play_fractions.video        (default 0.0)
    action_sampler   | demo only                | train.align.play_fractions.action_sampler (default 0.0;
                                                                                       bump >0 to ablate
                                                                                       the marginal-vs-
                                                                                       conditional inconsistency)

Under these defaults:
  • p(s)        is anchored to demo (video)
  • p(a|s)      is anchored to demo (id)
  • p(a)        is anchored to demo (action_sampler)
  • p(s|a)      stays broad — Play + Demo (wm)
That keeps the conditional & marginal action/state distributions internally
consistent while letting WM-direction dynamics knowledge stay general.

Derived from `train_align_mix.py` (the legacy mode-routed alignment script);
all FDPO machinery has been dropped. Optional LoRA is preserved.

Run length: a single knob `train.num_training_steps` drives the cosine
schedule and the outer loop. Model weights load from `cfg.dynamics_ckpt`;
optimizer and scheduler start fresh.

One script, two configs:
  scripts/config/align/pushT-mix-uwm.yaml
  scripts/config/align/g1-mix-uwm.yaml
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
    RMSLossScaler,
    UnifiedForwardProcess,
    VideoPretrainingForwardProcess,
    compute_action_pretraining_loss,
    compute_unified_uwm_loss,
    compute_video_pretraining_loss,
)
from dreamerv4uwm.models.dynamics import DenoiserWrapper
from dreamerv4uwm.models.utils import load_denoiser, load_tokenizer
from dreamerv4uwm.utils.distributed import (
    cleanup_distributed,
    load_ddp_checkpoint,
    save_ddp_checkpoint,
    setup_distributed,
    unwrap_model,
)


# ---------------------------------------------------------------------------
# Path enum + routing
# ---------------------------------------------------------------------------
# Stable indexing for cross-rank broadcasting of (path, source) decisions.
ALIGN_PATHS = ['wm', 'id', 'video', 'action_sampler']
SOURCES = ['play', 'demo']

# θ values forced on the unified diffuser for wm/id paths. The video and
# action_sampler paths route to their own dedicated forward processes (the
# marginal edge classes), where θ doesn't apply.
PATH_FORCED_THETA = {
    'wm': math.pi / 2,
    'id': 0.0,
}

# Per-path P(play) defaults. The configs override these via
# `train.align.play_fractions.{path}`. Keep these in sync with the docstring.
PATH_PLAY_FRACTION_DEFAULTS = {
    'wm':             0.5,
    'id':             0.0,
    'video':          0.0,
    'action_sampler': 0.0,
}


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
# Dataloaders
# ---------------------------------------------------------------------------

def build_dataloaders(cfg, rank, world_size):
    """Build (play_loader, play_sampler, demo_loader, demo_sampler).

    Same window size and dataset class for both; differs only in `data_dir`,
    `batch_size`, and `seed` (decorrelated so the two samplers don't shuffle
    in lockstep).
    """
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
        stride=1, seed=cfg.seed + 7919,
        split='train',
        train_fraction=cfg.dataset.train_episodes_fraction,
        split_seed=cfg.dataset.split_seed,
        shuffle=True, drop_last=True,
        absolute_actions=cfg.train.absolute_actions,
        kind=kind,
    )
    return play_loader, play_sampler, demo_loader, demo_sampler


def _cycling_iter(loader, sampler=None, epoch_ref=None, base_offset=0):
    """Infinite iterator over a DataLoader. Advances DistributedSampler epoch
    on each restart so shuffles vary across passes."""
    pass_idx = 0
    while True:
        if sampler is not None:
            sampler.set_epoch((epoch_ref[0] if epoch_ref else 0) * 10_000
                              + base_offset + pass_idx)
        for b in loader:
            yield b
        pass_idx += 1


# ---------------------------------------------------------------------------
# (Path, source) sampling
# ---------------------------------------------------------------------------

def _resolve_play_fractions(cfg):
    """Build a {path: play_fraction} dict from cfg with sensible defaults."""
    align_cfg = cfg.train.get('align', {}) or {}
    pf_cfg = align_cfg.get('play_fractions', {}) or {}
    return {
        p: float(pf_cfg.get(p, PATH_PLAY_FRACTION_DEFAULTS[p]))
        for p in ALIGN_PATHS
    }


def _resolve_path_weights(cfg):
    """Build a (len(ALIGN_PATHS),) torch tensor of path-mixture weights."""
    align_cfg = cfg.train.get('align', {}) or {}
    pw_cfg = align_cfg.get('path_weights', {}) or {}
    return torch.tensor(
        [float(pw_cfg.get(p, 0.0)) for p in ALIGN_PATHS],
        dtype=torch.float64,
    )


def sample_path_and_source(path_weights_cpu, play_fractions, rank, device,
                           generator=None):
    """Sample (path, source) jointly on rank 0; broadcast to all ranks.

    `path_weights_cpu`: precomputed (4,) CPU tensor; saves rebuilding it
    every micro-step. `play_fractions`: dict {path: P(play)}.
    """
    if rank == 0:
        path_idx = int(torch.multinomial(path_weights_cpu, 1, generator=generator).item())
        path = ALIGN_PATHS[path_idx]
        play_frac = play_fractions[path]
        src_idx = (
            0 if torch.rand(1, generator=generator).item() < play_frac else 1
        )
        codes = torch.tensor([path_idx, src_idx], device=device, dtype=torch.long)
    else:
        codes = torch.zeros(2, device=device, dtype=torch.long)
    dist.broadcast(codes, src=0)
    return ALIGN_PATHS[int(codes[0].item())], SOURCES[int(codes[1].item())]


# ---------------------------------------------------------------------------
# Optional LoRA wrap
# ---------------------------------------------------------------------------

def build_lora_config(cfg: DictConfig):
    from peft import LoraConfig
    lora = cfg.lora
    target = lora.target_modules
    if (isinstance(target, (list, tuple)) or
            (hasattr(target, '__iter__') and not isinstance(target, str))):
        target = list(target)
    modules_to_save = lora.get('modules_to_save', None)
    if modules_to_save is not None:
        modules_to_save = list(modules_to_save)
    return LoraConfig(
        r=int(lora.r),
        lora_alpha=int(lora.lora_alpha),
        lora_dropout=float(lora.lora_dropout),
        bias=str(lora.bias),
        target_modules=target,
        modules_to_save=modules_to_save,
    )


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
# Build models
# ---------------------------------------------------------------------------

def _strip_deprecated_keys(cfg):
    if 'latent_attends_action' in cfg.denoiser:
        del cfg.denoiser['latent_attends_action']


def build_models(cfg, device, local_rank, rank):
    _strip_deprecated_keys(cfg)

    tokenizer = load_tokenizer(
        cfg, device=device, max_num_forward_steps=cfg.denoiser.max_sequence_length
    )
    tokenizer = tokenizer.to(device).eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    if cfg.dynamics_ckpt:
        if rank == 0:
            print(f"Loading dynamics weights from: {cfg.dynamics_ckpt}")
        denoiser = load_denoiser(
            cfg, device=device, model_key='model',
            max_num_forward_steps=cfg.denoiser.max_sequence_length,
        )
    else:
        denoiser = DenoiserWrapper(
            cfg, max_num_forward_steps=cfg.denoiser.max_sequence_length,
        )
    denoiser = denoiser.to(device)

    lora_enabled = bool(cfg.lora.get('enabled', False))
    if lora_enabled:
        from peft import get_peft_model
        lora_cfg = build_lora_config(cfg)
        denoiser = get_peft_model(denoiser, lora_cfg)
        if rank == 0:
            denoiser.print_trainable_parameters()

    if cfg.train.use_compile:
        if lora_enabled:
            if rank == 0:
                print("WARNING: use_compile=True with LoRA — skipping compile.")
        else:
            denoiser = torch.compile(denoiser, mode='max-autotune-no-cudagraphs', fullgraph=True)
            tokenizer = torch.compile(tokenizer, mode='max-autotune-no-cudagraphs', fullgraph=False)

    denoiser = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=False)

    # --- three diffusers: unified (for wm/id) + two marginals (for video/action_sampler) ---
    unified_cfg = cfg.train.get('unified', {}) or {}
    unified_diffuser = UnifiedForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        action_noise_std=float(unified_cfg.get('action_noise_std', 1.0)),
        # θ-mixture knobs are NOT used in alignment (we force θ per path),
        # but the constructor still asserts they sum positive. Pass through
        # whatever the config has so unified-loss debugging is unaffected.
        theta_id_prob=float(unified_cfg.get('theta_id_prob', 0.15)),
        theta_policy_prob=float(unified_cfg.get('theta_policy_prob', 0.15)),
        theta_wm_prob=float(unified_cfg.get('theta_wm_prob', 0.15)),
        theta_continuum_prob=float(unified_cfg.get('theta_continuum_prob', 0.55)),
        profile_step_prob=float(unified_cfg.get('profile_step_prob', 0.5)),
        profile_progressive_prob=float(unified_cfg.get('profile_progressive_prob', 0.3)),
        profile_constant_prob=float(unified_cfg.get('profile_constant_prob', 0.2)),
        profile_diffusion_forcing_prob=float(unified_cfg.get('profile_diffusion_forcing_prob', 0.0)),
        profile_reverse_step_prob=float(unified_cfg.get('profile_reverse_step_prob', 0.0)),
        r_beta_alpha=float(unified_cfg.get('r_beta_alpha', 1.0)),
        r_beta_beta=float(unified_cfg.get('r_beta_beta', 1.0)),
        diffusion_forcing_bidir_prob=float(unified_cfg.get('diffusion_forcing_bidir_prob', 0.5)),
        device=device,
    )
    video_diffuser = VideoPretrainingForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        action_noise_std=float(unified_cfg.get('action_noise_std', 1.0)),
        bidir_prob=float(unified_cfg.get('pretraining_bidir_prob', 0.5)),
        device=device,
    )
    action_diffuser = ActionPretrainingForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        action_noise_std=float(unified_cfg.get('action_noise_std', 1.0)),
        bidir_prob=float(unified_cfg.get('pretraining_bidir_prob', 0.5)),
        device=device,
    )

    return (
        tokenizer, denoiser,
        unified_diffuser, video_diffuser, action_diffuser,
        lora_enabled,
    )


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
# Single training loop
# ---------------------------------------------------------------------------

def train_epoch(
    epoch, play_iter, demo_iter, steps_in_epoch,
    tokenizer, denoiser,
    unified_diffuser, video_diffuser, action_diffuser,
    optim, scheduler, tb_writer,
    cfg, rank, device, global_update, log_dir, wandb_run_id,
    trainable_params, lora_enabled, rng,
    path_weights_cpu, play_fractions,
    causal_eps, ramp_beta,
    loss_scaler,
):
    denoiser.train()
    epoch_start = time.perf_counter()
    epoch_loss_sum = 0.0
    num_updates = 0

    accum_total = 0.0
    accum_obs = 0.0
    accum_act = 0.0
    path_count = {p: 0 for p in ALIGN_PATHS}
    src_count = {s: 0 for s in SOURCES}
    # Per-(path, source) loss accumulators — sparse W&B traces for diagnostics.
    ps_total: dict[tuple[str, str], float] = {}
    ps_obs:   dict[tuple[str, str], float] = {}
    ps_act:   dict[tuple[str, str], float] = {}
    ps_n:     dict[tuple[str, str], int]   = {}

    n_actions = cfg.denoiser.n_actions
    train_reward = bool(cfg.denoiser.get('train_reward_model', False))
    reward_weight = float(cfg.train.get('reward_weight', 1.0))

    for step_idx in range(steps_in_epoch):
        micro_idx = step_idx % cfg.train.accum_grad_steps
        is_last_micro = micro_idx == cfg.train.accum_grad_steps - 1

        if micro_idx == 0:
            optim.zero_grad(set_to_none=True)

        path, source = sample_path_and_source(
            path_weights_cpu, play_fractions, rank, device, generator=rng,
        )
        path_count[path] += 1
        src_count[source] += 1

        batch = next(play_iter if source == 'play' else demo_iter)

        images = batch['image'].to(device, non_blocking=True).to(torch.bfloat16)
        actions = (batch['action'].to(device, non_blocking=True)
                   .to(torch.bfloat16)[:, :, :n_actions].unsqueeze(-2))
        rewards = batch.get('reward', None)
        if rewards is not None:
            rewards = rewards.to(device, non_blocking=True).float()
            if rewards.dim() == 3 and rewards.shape[-1] == 1:
                rewards = rewards.squeeze(-1)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                z_clean = tokenizer.encode(images).detach()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if path in PATH_FORCED_THETA:
                # wm or id: unified diffuser with forced θ.
                info = unified_diffuser(
                    z_clean, actions, force_theta=PATH_FORCED_THETA[path],
                )
                losses = compute_unified_uwm_loss(
                    info, denoiser, device=device,
                    causal_eps=causal_eps,
                    ramp_beta=ramp_beta,
                    rewards=rewards if train_reward else None,
                    scaler=loss_scaler,
                )
            elif path == 'video':
                info = video_diffuser(z_clean, actions)
                losses = compute_video_pretraining_loss(
                    info, denoiser, device=device,
                    rewards=rewards if train_reward else None,
                    scaler=loss_scaler,
                )
            elif path == 'action_sampler':
                info = action_diffuser(z_clean, actions)
                losses = compute_action_pretraining_loss(
                    info, denoiser, device=device,
                    rewards=rewards if train_reward else None,
                    scaler=loss_scaler,
                )
            else:
                raise RuntimeError(f"unknown alignment path: {path!r}")

            obs = losses['obs_flow_loss']
            act = losses['act_flow_loss']
            reward_loss = losses.get('reward_loss', None)
            total = obs + act
            if reward_loss is not None:
                total = total + reward_weight * reward_loss
            loss_micro = total / cfg.train.accum_grad_steps

        loss_micro.backward()

        # Progress signal: prefer raw (pre-scaler) losses when the scaler is
        # active. Scaled losses normalize to unit RMS in steady state and
        # therefore can't be read as a training-progress curve.
        obs_for_log = losses.get('obs_flow_loss_raw', obs)
        act_for_log = losses.get('act_flow_loss_raw', act)
        obs_mean = obs_for_log.mean().item()
        act_mean = act_for_log.mean().item()
        accum_total += loss_micro.item()
        accum_obs   += obs_mean
        accum_act   += act_mean
        key = (path, source)
        ps_total[key] = ps_total.get(key, 0.0) + obs_mean + act_mean
        ps_obs[key]   = ps_obs.get(key,   0.0) + obs_mean
        ps_act[key]   = ps_act.get(key,   0.0) + act_mean
        ps_n[key]     = ps_n.get(key,     0)   + 1

        if is_last_micro:
            torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=float(cfg.train.clip_grad_norm),
            )
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
                # Per-accum-window mixture trace.
                for p in ALIGN_PATHS:
                    tb_writer.add_scalar(f'mix/path_{p}',   path_count[p], global_update)
                for s in SOURCES:
                    tb_writer.add_scalar(f'mix/source_{s}', src_count[s],  global_update)
                # Per-(path, source) loss curves — sparse emissions.
                for k, n in ps_n.items():
                    if n == 0:
                        continue
                    p_, s_ = k
                    tag = f"{p_}_{s_}"
                    tb_writer.add_scalar(f"by_ps/{tag}/total", ps_total[k] / n, global_update)
                    tb_writer.add_scalar(f"by_ps/{tag}/obs",   ps_obs[k]   / n, global_update)
                    tb_writer.add_scalar(f"by_ps/{tag}/act",   ps_act[k]   / n, global_update)

                if global_update % cfg.print_every == 0:
                    mix_str = ' '.join(
                        f'{p}={path_count[p]}' for p in ALIGN_PATHS if path_count[p] > 0
                    )
                    src_str = ' '.join(f'{s}={src_count[s]}' for s in SOURCES)
                    print(
                        f"  [step {global_update}]"
                        f"  loss: {sync_loss:.4f}"
                        f"  obs: {accum_obs * inv:.4f}"
                        f"  act: {accum_act * inv:.4f}"
                        f"  lr: {lr:.2e}"
                        f"  path: {mix_str}"
                        f"  src: {src_str}"
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
            path_count  = {p: 0 for p in ALIGN_PATHS}
            src_count   = {s: 0 for s in SOURCES}
            ps_total.clear(); ps_obs.clear(); ps_act.clear(); ps_n.clear()

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
@hydra.main(config_path='config', config_name='align/pushT-mix-uwm', version_base=None)
def main(cfg: DictConfig):
    torch.backends.cuda.matmul.allow_tf32 = cfg.train.enable_fast_matmul

    rank, local_rank, world_size, device = setup_distributed()
    torch.manual_seed(cfg.seed + rank)

    if rank == 0:
        print(f"Distributed alignment training: {world_size} GPU(s)")
        eff = (cfg.train.batch_per_gpu * world_size * cfg.train.accum_grad_steps)
        print(f"Effective global batch size: {eff} (per-step; demo side uses demo_batch_per_gpu)")

    # --- Data ---
    play_loader, play_sampler, demo_loader, demo_sampler = build_dataloaders(
        cfg, rank, world_size,
    )

    path_weights_cpu = _resolve_path_weights(cfg)
    play_fractions = _resolve_play_fractions(cfg)
    assert path_weights_cpu.sum() > 0, (
        "At least one alignment path must have positive weight in "
        "train.align.path_weights"
    )

    if rank == 0:
        print(f"Play loader: {len(play_loader)} steps/pass "
              f"(batch={cfg.train.batch_per_gpu}, dir={cfg.dataset.play_data_dir})")
        print(f"Demo loader: {len(demo_loader)} steps/pass "
              f"(batch={cfg.train.get('demo_batch_per_gpu', cfg.train.batch_per_gpu)}, "
              f"dir={cfg.dataset.demo_data_dir})")
        # Pretty-print path weights as the routing being trained.
        norm = path_weights_cpu / path_weights_cpu.sum()
        print("Alignment paths:")
        for i, p in enumerate(ALIGN_PATHS):
            print(f"  {p:18s}  weight={float(norm[i].item()):.3f}  "
                  f"P(play)={play_fractions[p]:.3f}")

    # --- Models ---
    if rank == 0:
        print("Building models...")
    (tokenizer, denoiser,
     unified_diffuser, video_diffuser, action_diffuser,
     lora_enabled) = build_models(cfg, device, local_rank, rank)
    trainable_params = [p for p in denoiser.parameters() if p.requires_grad]
    if rank == 0:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in denoiser.parameters())
        print(f"Trainable: {n_trainable:,} / {n_total:,} "
              f"({100 * n_trainable / n_total:.3f}%)"
              + (" [LoRA]" if lora_enabled else " [full FT]"))

    # --- Unified-loss runtime knobs (re-read here to support live overrides) ---
    unified_cfg = cfg.train.get('unified', {}) or {}
    causal_eps = float(unified_cfg.get('causal_eps', 1e-3))
    ramp_beta = float(unified_cfg.get('ramp_beta', 1.0))

    # --- Loss scaler (RMS-normalizes obs vs act loss magnitudes) ---
    # Mirrors the pattern in train_dynamics_uwm_new.py. State isn't persisted
    # with the checkpoint — first ~100 steps post-resume run at slightly
    # mis-scaled magnitudes while the EMA reconverges (usually negligible).
    rms_enabled = bool(unified_cfg.get('rms_scale_loss', True))
    rms_decay = float(unified_cfg.get('rms_scale_decay', 0.99))
    loss_scaler = RMSLossScaler(decay=rms_decay) if rms_enabled else None
    if rank == 0:
        print(f"RMS loss scaler: {'enabled' if rms_enabled else 'disabled'}"
              + (f" (decay={rms_decay})" if rms_enabled else ""))

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

    # --- Optional resume (model only by design) ---
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
              "(model weights initialized from cfg.dynamics_ckpt).")

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
                              base_offset=104729)

    # --- Per-rank RNG for path/source sampling (rank 0 actually uses it) ---
    rng = torch.Generator(device='cpu').manual_seed(cfg.seed + 31337)

    if rank == 0:
        print(f"Starting training: {total_grad_steps} grad steps "
              f"({total_micro_steps} micro-steps, accum={cfg.train.accum_grad_steps}).")

    epoch_losses, epoch_times = [], []
    global_update, avg_loss, epoch_time = train_epoch(
        epoch=0,
        play_iter=play_iter, demo_iter=demo_iter,
        steps_in_epoch=total_micro_steps,
        tokenizer=tokenizer, denoiser=denoiser,
        unified_diffuser=unified_diffuser,
        video_diffuser=video_diffuser,
        action_diffuser=action_diffuser,
        optim=optim, scheduler=scheduler, tb_writer=tb_writer,
        cfg=cfg, rank=rank, device=device,
        global_update=global_update, log_dir=log_dir, wandb_run_id=wandb_run_id,
        trainable_params=trainable_params, lora_enabled=lora_enabled, rng=rng,
        path_weights_cpu=path_weights_cpu, play_fractions=play_fractions,
        causal_eps=causal_eps, ramp_beta=ramp_beta,
        loss_scaler=loss_scaler,
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

    if lora_enabled and bool(cfg.lora.get('save_merged_final', True)):
        merged_path = os.path.join(log_dir, 'final_merged.pt')
        save_lora_merged(denoiser, merged_path, rank)
    dist.barrier()

    if rank == 0:
        cur = torch.cuda.memory_allocated(device) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\n{'='*60}")
        print("Alignment Training Complete!")
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
