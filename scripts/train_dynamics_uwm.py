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
from dreamerv4uwm.loss import UWMForwardProcess, compute_uwm_loss
from dreamerv4uwm.models.dynamics import DenoiserWrapper
from dreamerv4uwm.models.utils import load_denoiser, load_tokenizer
from dreamerv4uwm.utils.distributed import (
    cleanup_distributed,
    load_ddp_checkpoint,
    save_ddp_checkpoint,
    setup_distributed,
)



# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-8):
    """Linear warmup + cosine decay keyed on LambdaLR's internal step counter.

    `warmup_steps` and `total_steps` are optimizer-step counts. The counter
    advances by one each `scheduler.step()` (called after `optim.step()`).
    """
    peak_lr = optimizer.defaults["lr"]
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / peak_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def build_dataloader(cfg, rank, world_size):
    # Loader always delivers max_sequence_length frames per sample. Each
    # branch consumes a different shape:
    #   long  → slices to [:long_bs] from the loader's tensor
    #   short → slices to [:short_bs] and crops T to short_sequence_length
    #   image → flattens (loader_bs × max_seq) and slices to image_bs
    # The loader's batch size is therefore max(short_bs, long_bs) so either
    # branch can be served from the same physical batch. When short_bs > long_bs
    # this slightly inflates I/O on long-branch micro-batches (extra rows are
    # discarded) but lets you train cheap-T short batches at a larger row count
    # than the long branch would fit.
    long_bs = int(cfg.train.get("long_seq_batch_per_gpu", cfg.train.batch_per_gpu))
    short_bs = int(cfg.train.batch_per_gpu)
    loader_bs = max(short_bs, long_bs)
    loader, sampler, _ = create_distributed_dataloader(
        data_dir=cfg.dataset.get("data_dir", None),
        data_dirs=cfg.dataset.get("data_dirs", None),
        window_size=cfg.denoiser.max_sequence_length,
        batch_size=loader_bs,
        rank=rank,
        world_size=world_size,
        num_workers=cfg.train.num_workers,
        stride=1,
        seed=cfg.seed,
        split="train",
        train_fraction=cfg.dataset.train_episodes_fraction,
        split_seed=cfg.dataset.split_seed,
        shuffle=True,
        drop_last=True,
        absolute_actions=cfg.train.absolute_actions,
        kind=cfg.dataset.get("kind", "sharded_hdf5"),
    )
    return loader, sampler


def build_models(cfg, device, local_rank):
    tokenizer = load_tokenizer(
        cfg, device=device, max_num_forward_steps=cfg.denoiser.max_sequence_length
    )

    if cfg.dynamics_ckpt:
        print(f"Loading dynamics from: {cfg.dynamics_ckpt}")
        denoiser = load_denoiser(
            cfg, device=device, model_key="model",
            max_num_forward_steps=cfg.denoiser.max_sequence_length,
        )
    else:
        denoiser = DenoiserWrapper(cfg, max_num_forward_steps=cfg.denoiser.max_sequence_length)

    fc_noise = cfg.train.get("forcing_context_noise", {})
    diffuser = UWMForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        mode_weights=OmegaConf.to_container(cfg.train.mode_weights, resolve=True),
        forcing_context_noise_bias=float(fc_noise.get("bias", 0.0)),
        forcing_context_noise_alpha=float(fc_noise.get("alpha", 0.5)),
        forcing_context_noise_beta=float(fc_noise.get("beta", 2.0)),
        forcing_mask_actions=bool(cfg.train.get("forcing_mask_actions", False)),
        device=device,
    )

    tokenizer = tokenizer.to(device)
    denoiser = denoiser.to(device)

    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    if cfg.train.use_compile:
        denoiser = torch.compile(denoiser, mode="max-autotune-no-cudagraphs", fullgraph=True)
        tokenizer = torch.compile(tokenizer, mode="max-autotune-no-cudagraphs", fullgraph=False)

    denoiser = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=False)

    return tokenizer, denoiser, diffuser


def setup_logging(cfg, rank, log_dir, wandb_run_id):
    """Initialize TensorBoard and (optionally) W&B on rank 0. Returns (tb_writer, wandb_run_id)."""
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
            wandb.init(
                project=cfg.wandb.project,
                id=wandb_run_id,
                resume="allow",
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True,
                dir=log_dir,
            )
        else:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True,
                dir=log_dir,
            )
        wandb_run_id = wandb.run.id

    return SummaryWriter(log_dir=tb_log_dir), wandb_run_id


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    epoch,
    train_loader,
    train_sampler,
    tokenizer,
    denoiser,
    diffuser,
    optim,
    scheduler,
    tb_writer,
    cfg,
    rank,
    device,
    global_update,
    log_dir,
    wandb_run_id,
):
    denoiser.train()
    train_sampler.set_epoch(epoch)
    world_size = dist.get_world_size()

    # --- Batch-length branch selection (Dreamer-V4 paper) ---
    # Loader delivers (B_long, T_long) where T_long = max_sequence_length.
    # Per accumulation window, sample a 3-way categorical branch:
    #   long  → consume (B_long, T_long) as-is (exercises windowed-temporal
    #           attention; requires max_seq > short_seq_len to do anything useful).
    #   image → reshape (B_long, T_long, …) → (B_long*T_long, 1, …), then
    #           trim to image_batch_per_gpu. Each frame becomes an
    #           independent length-1 sequence; temporal attention is a no-op.
    #           force_mode='video' in the forward process pins actions to
    #           pure noise and applies state loss across the (T=1) sequence —
    #           unconditional single-frame generation for start-frame cold
    #           start.
    #   short → random-crop along T to short_sequence_length (defaults to
    #           context_length), take batch_per_gpu samples.
    # Branch is fixed across all micro-batches in one optimizer step so the
    # gradient is coherent and metrics are unambiguous to attribute.
    short_bs = int(cfg.train.batch_per_gpu)
    long_bs = int(cfg.train.get("long_seq_batch_per_gpu", short_bs))
    image_bs = int(cfg.train.get("image_batch_per_gpu", short_bs))
    # Loader fetches max(short_bs, long_bs) rows; each branch slices its share.
    loader_bs = max(short_bs, long_bs)
    # Short-branch crop length. Decoupled from `denoiser.context_length` (which
    # only sets the temporal-attention window) so the short-branch T can be
    # tuned independently. Lives under `train:` because it's a data-pipeline
    # knob, not a model knob; falls back to context_length for backward compat.
    short_seq_len = int(cfg.train.get("short_sequence_length", cfg.denoiser.context_length))
    max_seq = int(cfg.denoiser.max_sequence_length)
    long_seq_prob = float(cfg.train.get("long_seq_prob", 0.0))
    image_prob = float(cfg.train.get("image_prob", 0.0))
    assert short_seq_len <= max_seq, (
        f"short_sequence_length ({short_seq_len}) must be <= max_sequence_length ({max_seq})"
    )
    # Long branch only fires when it can actually produce T > short_seq_len.
    effective_long_prob = long_seq_prob if max_seq > short_seq_len else 0.0
    assert 0.0 <= image_prob <= 1.0
    assert effective_long_prob + image_prob <= 1.0 + 1e-6, (
        f"effective_long_prob ({effective_long_prob}) + image_prob "
        f"({image_prob}) must be <= 1"
    )
    # Image branch trims from the reshaped pool of loader_bs * max_seq frames.
    max_image_rows = loader_bs * max_seq
    assert image_bs <= max_image_rows, (
        f"image_batch_per_gpu ({image_bs}) exceeds available rows "
        f"(max(batch_per_gpu, long_seq_batch_per_gpu) * max_sequence_length "
        f"= {max_image_rows})"
    )

    epoch_start = time.perf_counter()
    epoch_loss_sum = 0.0
    num_updates = 0
    step_times = []
    data_times = []

    accum_obs_flow = 0.0
    accum_act_flow = 0.0
    accum_reward = 0.0
    accum_total = 0.0
    accum_branch = "short"  # set on the first micro of each window
    train_reward = bool(cfg.denoiser.get("train_reward_model", False))
    reward_weight = float(cfg.train.get("reward_weight", 1.0))

    data_start = time.perf_counter()

    for step_idx, batch in enumerate(train_loader):
        micro_idx = step_idx % cfg.train.accum_grad_steps
        is_last_micro = micro_idx == cfg.train.accum_grad_steps - 1

        data_times.append(time.perf_counter() - data_start)

        # --- Prepare batch ---
        images = batch["image"].to(device, non_blocking=True)  # (B, T, C, H, W)
        actions = batch["action"].to(device, non_blocking=True)  # (B, T, action_dim)
        images = images.to(torch.bfloat16)
        # Slice to configured action dims and add token dimension: (B, T, A) -> (B, T, 1, A)
        actions = actions.to(torch.bfloat16)[:, :, :cfg.denoiser.n_actions].unsqueeze(-2)
        # Reward field is dataset-optional. When present we keep fp32 — symlog
        # CE underflows in bf16 and we cast logits in compute_reward_mtp_loss
        # back to fp32 anyway. Squeeze a trailing singleton if shards were
        # written as (..., 1).
        rewards = batch.get("reward", None)
        if rewards is not None:
            rewards = rewards.to(device, non_blocking=True).float()
            if rewards.dim() == 3 and rewards.shape[-1] == 1:
                rewards = rewards.squeeze(-1)

        # --- Branch decision for this accumulation window ---
        # Sampled on rank 0 and broadcast so every rank takes the same path
        # (otherwise DDP all-reduce would mix different effective batch sizes).
        # Categorical over {long=0, image=1, short=2}. Mutually exclusive so
        # each gradient step is attributable to exactly one recipe.
        if micro_idx == 0:
            if rank == 0:
                r = torch.rand(1).item()
                if r < effective_long_prob:
                    code = 0
                elif r < effective_long_prob + image_prob:
                    code = 1
                else:
                    code = 2
                branch_code_t = torch.tensor(
                    [code], device=device, dtype=torch.long,
                )
            else:
                branch_code_t = torch.zeros(1, device=device, dtype=torch.long)
            dist.broadcast(branch_code_t, src=0)
            accum_branch = ("long", "image", "short")[int(branch_code_t.item())]

            # Random crop start — short branch only.
            if accum_branch == "short" and max_seq > short_seq_len:
                if rank == 0:
                    start_t = torch.randint(
                        0, max_seq - short_seq_len + 1, (1,),
                        device=device, dtype=torch.long,
                    )
                else:
                    start_t = torch.zeros(1, device=device, dtype=torch.long)
                dist.broadcast(start_t, src=0)
                crop_start = int(start_t.item())
            else:
                crop_start = 0

        # --- Apply branch slicing ---
        if accum_branch == "short":
            images = images[:short_bs, crop_start:crop_start + short_seq_len]
            actions = actions[:short_bs, crop_start:crop_start + short_seq_len]
            if rewards is not None:
                rewards = rewards[:short_bs, crop_start:crop_start + short_seq_len]
        elif accum_branch == "long":
            # Loader fetches loader_bs = max(short_bs, long_bs); explicitly trim
            # to long_bs here so the long branch sees its configured batch size
            # regardless of which other knob is larger.
            images = images[:long_bs]
            actions = actions[:long_bs]
            if rewards is not None:
                rewards = rewards[:long_bs]
        elif accum_branch == "image":
            # Flatten time into batch — (B_long, T_long, …) → (B_long*T_long,
            # 1, …) — so every frame is an independent length-1 sequence.
            # Temporal attention has nothing to attend to; the model learns
            # the marginal image distribution. force_mode='video' below pins
            # actions to pure noise and applies state loss across the T=1
            # sequence, matching the paper's "30% videos as separate images"
            # recipe for unconditioned start-frame generation.
            B_in, T_in = images.shape[:2]
            images = images.reshape(B_in * T_in, 1, *images.shape[2:])
            actions = actions.reshape(B_in * T_in, 1, *actions.shape[2:])
            if rewards is not None:
                rewards = rewards.reshape(B_in * T_in, 1)
            # Shuffle the flattened (window × time) axis before slicing so we
            # sample uniformly across the loaded window. Without this, taking
            # `[:image_bs]` always picks the earliest frames and over-samples
            # episode-start state. Each rank shuffles independently — rank-local
            # data tensors don't need to align.
            perm = torch.randperm(B_in * T_in, device=images.device)[:image_bs]
            images = images[perm]
            actions = actions[perm]
            if rewards is not None:
                rewards = rewards[perm]

        torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        if micro_idx == 0:
            optim.zero_grad(set_to_none=True)

        # --- Forward pass ---
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_clean = tokenizer.encode(images).detach().clone()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            diffused_info = diffuser(
                z_clean, actions,
                force_mode="video" if accum_branch == "image" else None,
            )
            losses = compute_uwm_loss(
                diffused_info, denoiser, device=device,
                loss_weighting=str(cfg.train.get("loss_weighting", "ramp")),
                rewards=rewards if train_reward else None,
            )
            obs_flow_loss = losses["obs_flow_loss"]
            act_flow_loss = losses["act_flow_loss"]
            reward_loss = losses["reward_loss"]
            total_loss = obs_flow_loss + act_flow_loss
            if reward_loss is not None:
                total_loss = total_loss + reward_weight * reward_loss
            loss_micro = total_loss / cfg.train.accum_grad_steps

        loss_micro.backward()

        accum_obs_flow += obs_flow_loss.mean().item()
        accum_act_flow += act_flow_loss.mean().item()
        if reward_loss is not None:
            accum_reward += reward_loss.item()
        accum_total += loss_micro.item()

        # --- Optimizer step at end of accumulation window ---
        if is_last_micro:
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optim.step()
            global_update += 1
            scheduler.step()
            total_tensor = torch.tensor([accum_total], device=device)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.AVG)
            sync_loss = total_tensor.item()
            epoch_loss_sum += sync_loss
            num_updates += 1

            if rank == 0:
                lr = scheduler.get_last_lr()[0]
                # Combined curves (both branches) for an at-a-glance view.
                tb_writer.add_scalar("train/total_loss", sync_loss, global_update)
                tb_writer.add_scalar("train/obs_flow_loss", accum_obs_flow, global_update)
                tb_writer.add_scalar("train/act_flow_loss", accum_act_flow, global_update)
                if train_reward:
                    tb_writer.add_scalar("train/reward_loss", accum_reward, global_update)
                tb_writer.add_scalar("train/lr", lr, global_update)
                tb_writer.add_scalar("train/global_update", global_update, global_update)
                # Branch-namespaced curves. Sparse by design — only one of
                # {short, long, image} updates per step. Losses across
                # branches are not magnitude-comparable (different effective
                # receptive field, different T-dependent schedules, image
                # mode zeros the action loss).
                ns = f"train/{accum_branch}"
                tb_writer.add_scalar(f"{ns}/total_loss", sync_loss, global_update)
                tb_writer.add_scalar(f"{ns}/obs_flow_loss", accum_obs_flow, global_update)
                tb_writer.add_scalar(f"{ns}/act_flow_loss", accum_act_flow, global_update)
                if train_reward:
                    tb_writer.add_scalar(f"{ns}/reward_loss", accum_reward, global_update)
                # Categorical branch trace: 0=long, 1=image, 2=short.
                _branch_code = {"long": 0, "image": 1, "short": 2}[accum_branch]
                tb_writer.add_scalar("train/branch_code", _branch_code, global_update)

                if global_update % cfg.print_every == 0:
                    print(
                        f"  [step {global_update}]"
                        f"  [{accum_branch}]"
                        f"  loss: {sync_loss:.4f}"
                        f"  obs: {accum_obs_flow:.4f}"
                        f"  act: {accum_act_flow:.4f}"
                        f"  lr: {lr:.2e}"
                    )

                if global_update % cfg.save_every == 0:
                    print(f"[Checkpoint] Saving at global_update={global_update}")
                    save_ddp_checkpoint(
                        ckpt_path=os.path.join(log_dir, f"{global_update}.pt"),
                        epoch=epoch,
                        global_update=global_update,
                        model=denoiser,
                        optim=optim,
                        scheduler=scheduler,
                        rank=rank,
                        wandb_run_id=wandb_run_id,
                        log_dir=log_dir,
                    )

            accum_obs_flow = 0.0
            accum_act_flow = 0.0
            accum_reward = 0.0
            accum_total = 0.0

        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_start)
        data_start = time.perf_counter()

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = epoch_loss_sum / num_updates if num_updates > 0 else 0.0
    # Loader I/O throughput — uses long_bs * max_seq because the loader
    # always pulls long-shaped tensors (short steps subsample/crop, but the
    # bytes were still moved). This is a data-pipeline metric, not a
    # train-effective-frames metric.
    long_bs = int(cfg.train.get("long_seq_batch_per_gpu", cfg.train.batch_per_gpu))
    total_frames = long_bs * cfg.denoiser.max_sequence_length * len(train_loader)
    epoch_fps = total_frames / epoch_time

    if rank == 0:
        avg_step = sum(step_times) / len(step_times)
        avg_data = sum(data_times) / len(data_times)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss:        {avg_loss:.6f}")
        print(f"  Epoch Time:        {epoch_time:.2f}s")
        print(f"  Throughput:        {epoch_fps:.2f} FPS")
        print(f"  Avg Step Time:     {avg_step:.3f}s")
        print(f"  Avg Data Time:     {avg_data:.3f}s")
        print(f"{'='*60}\n")

    return global_update, avg_loss, epoch_time, epoch_fps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@record
@hydra.main(config_path="config", config_name="dynamics/pushT", version_base=None)
def main(cfg: DictConfig):
    torch.backends.cuda.matmul.allow_tf32 = cfg.train.enable_fast_matmul

    rank, local_rank, world_size, device = setup_distributed()
    torch.manual_seed(cfg.seed + rank)

    if rank == 0:
        print(f"Distributed training: {world_size} GPU(s)")
        print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
        print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
        effective_batch = cfg.train.batch_per_gpu * world_size * cfg.train.accum_grad_steps
        print(f"Effective global batch size: {effective_batch}")

    # --- Data ---
    train_loader, train_sampler = build_dataloader(cfg, rank, world_size)

    # --- Models ---
    if rank == 0:
        print("Building models...")
    tokenizer, denoiser, diffuser = build_models(cfg, device, local_rank)
    if rank == 0:
        n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        print(f"Denoiser learnable parameters: {n_params:,}")

    # --- Optimizer & scheduler ---
    optim = torch.optim.AdamW(
        denoiser.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    warmup_steps = int(cfg.train.warmup_samples)
    total_steps = int(cfg.train.total_samples)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    # --- Checkpoint resume ---
    wandb_run_id = cfg.wandb.run_name
    log_dir = None
    start_epoch = 0
    global_update = 0

    if cfg.reload_checkpoint is not None:
        if rank == 0:
            print(f"Resuming from checkpoint: {cfg.reload_checkpoint}")
        start_epoch, global_update, _cumulative_samples, wandb_run_id, log_dir = load_ddp_checkpoint(
            ckpt_path=cfg.reload_checkpoint,
            model=denoiser,
            optim=optim,
            scheduler=scheduler,
            rank=rank,
        )
    elif rank == 0:
        print("Starting from scratch.")

    # --- Logging (rank 0 only, then broadcast) ---
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

    # --- Training loop ---
    epoch_losses, epoch_times, epoch_fps_vals = [], [], []

    if rank == 0:
        print("Starting training...")

    for epoch in range(start_epoch, cfg.train.num_epochs):
        global_update, avg_loss, epoch_time, epoch_fps = train_epoch(
            epoch=epoch,
            train_loader=train_loader,
            train_sampler=train_sampler,
            tokenizer=tokenizer,
            denoiser=denoiser,
            diffuser=diffuser,
            optim=optim,
            scheduler=scheduler,
            tb_writer=tb_writer,
            cfg=cfg,
            rank=rank,
            device=device,
            global_update=global_update,
            log_dir=log_dir,
            wandb_run_id=wandb_run_id,
        )
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        epoch_fps_vals.append(epoch_fps)

        if rank == 0:
            save_ddp_checkpoint(
                ckpt_path=os.path.join(log_dir, f"{global_update}.pt"),
                epoch=epoch,
                global_update=global_update,
                model=denoiser,
                optim=optim,
                scheduler=scheduler,
                rank=rank,
                wandb_run_id=wandb_run_id,
                log_dir=log_dir,
            )
        dist.barrier()

    # --- Final summary ---
    if rank == 0:
        cur_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"  Avg loss:      {sum(epoch_losses) / len(epoch_losses):.6f}")
        print(f"  Avg epoch time:{sum(epoch_times) / len(epoch_times):.2f}s")
        print(f"  Avg FPS:       {sum(epoch_fps_vals) / len(epoch_fps_vals):.2f}")
        print(f"  GPU memory:    {cur_alloc:.2f} GB current / {peak_alloc:.2f} GB peak")
        print(f"{'='*60}")

        tb_writer.close()
        if cfg.wandb.enable:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
