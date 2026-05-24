"""
LoRA finetuning of the dynamics denoiser.

Mirrors `train_dynamics_uwm.py` but:
  - REQUIRES `cfg.dynamics_ckpt` (LoRA finetunes a base model).
  - Wraps `DenoiserWrapper` with PEFT LoRA before DDP.
  - Only LoRA matrices receive gradients; base weights stay frozen.
  - Saves a small adapter-only `.pt` per `save_every`, plus an optional final
    merged full state_dict that is drop-in compatible with `load_denoiser`.

Resume: pass `cfg.reload_checkpoint=<path-to-adapter-ckpt>.pt`.
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
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from dreamerv4uwm.datasets import create_distributed_dataloader
from dreamerv4uwm.loss import UWMForwardProcess, compute_uwm_loss
from dreamerv4uwm.models.utils import load_denoiser, load_tokenizer
from dreamerv4uwm.utils.distributed import (
    cleanup_distributed,
    setup_distributed,
    unwrap_model,
)


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
# LoRA helpers
# ---------------------------------------------------------------------------

def build_lora_config(cfg: DictConfig) -> LoraConfig:
    lora = cfg.lora
    target = lora.target_modules
    if isinstance(target, (list, tuple)) or hasattr(target, "__iter__") and not isinstance(target, str):
        target = list(target)
    modules_to_save = lora.get("modules_to_save", None)
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


def save_lora_checkpoint(
    ckpt_path: str,
    epoch: int,
    global_update: int,
    model,            # DDP-wrapped PeftModel
    optim: torch.optim.Optimizer,
    scheduler,
    rank: int,
    wandb_run_id: str = None,
    log_dir: str = None,
):
    """Save adapter-only state dict + trainer state. Tiny file (~MB)."""
    if rank != 0:
        return
    peft_model = unwrap_model(model)  # PeftModel
    adapter_state = get_peft_model_state_dict(peft_model)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "global_update": global_update,
        "adapter": adapter_state,
        "optim": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "wandb_run_id": wandb_run_id,
        "log_dir": log_dir,
    }
    torch.save(ckpt, ckpt_path)
    print(f"[rank0] Saved LoRA adapter checkpoint to {ckpt_path}")


def load_lora_checkpoint(
    ckpt_path: str,
    model,            # DDP-wrapped PeftModel
    optim: torch.optim.Optimizer,
    scheduler,
    rank: int,
):
    if not os.path.isfile(ckpt_path):
        if rank == 0:
            print(f"No checkpoint found at {ckpt_path}, starting LoRA from scratch.")
        return 0, 0, None, None

    if rank == 0:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(f"[rank0] Loaded LoRA adapter checkpoint from {ckpt_path}")
    else:
        ckpt = None

    obj_list = [ckpt]
    dist.broadcast_object_list(obj_list, src=0)
    ckpt = obj_list[0]

    start_epoch = ckpt.get("epoch", 0)
    global_update = ckpt.get("global_update", 0)
    wandb_run_id = ckpt.get("wandb_run_id", None)
    log_dir = ckpt.get("log_dir", None)

    set_peft_model_state_dict(unwrap_model(model), ckpt["adapter"])
    optim.load_state_dict(ckpt["optim"])
    scheduler.load_state_dict(ckpt["scheduler"])

    if rank == 0:
        print(f"Resuming LoRA from epoch {start_epoch + 1}, global_update {global_update}")
    return start_epoch, global_update, wandb_run_id, log_dir


def save_merged_final(model, ckpt_path: str, rank: int):
    """Merge LoRA into base, save as a standard `{model: state_dict}` .pt for load_denoiser."""
    if rank != 0:
        return
    peft_model = unwrap_model(model)
    base = peft_model.merge_and_unload()  # returns the underlying DenoiserWrapper
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": base.state_dict()}, ckpt_path)
    print(f"[rank0] Saved merged full checkpoint to {ckpt_path}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_dataloader(cfg, rank, world_size):
    loader, sampler, _ = create_distributed_dataloader(
        data_dir=cfg.dataset.get("data_dir", None),
        data_dirs=cfg.dataset.get("data_dirs", None),
        window_size=cfg.denoiser.max_sequence_length,
        batch_size=cfg.train.batch_per_gpu,
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


def build_models(cfg, device, local_rank, rank):
    assert cfg.dynamics_ckpt, (
        "LoRA finetuning requires `cfg.dynamics_ckpt` to point at a pretrained denoiser."
    )

    tokenizer = load_tokenizer(
        cfg, device=device, max_num_forward_steps=cfg.denoiser.max_sequence_length
    ).to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    if rank == 0:
        print(f"Loading pretrained denoiser from: {cfg.dynamics_ckpt}")
    denoiser = load_denoiser(
        cfg, device=device, model_key="model",
        max_num_forward_steps=cfg.denoiser.max_sequence_length,
    ).to(device)

    # --- LoRA wrap (must happen BEFORE DDP) ---
    lora_cfg = build_lora_config(cfg)
    denoiser = get_peft_model(denoiser, lora_cfg)
    if rank == 0:
        denoiser.print_trainable_parameters()

    # All LoRA params are touched on every forward (loss sums over all modes & both
    # heads; even mode-zeroed losses keep the autograd path), so DDP doesn't need
    # find_unused_parameters=True.
    denoiser = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=False)

    diffuser = UWMForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        mode_weights=OmegaConf.to_container(cfg.train.mode_weights, resolve=True),
        horizon_aware=bool(cfg.denoiser.get("horizon_aware", False)),
        device=device,
    )
    return tokenizer, denoiser, diffuser


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
    epoch, train_loader, train_sampler,
    tokenizer, denoiser, diffuser,
    optim, scheduler, tb_writer,
    cfg, rank, device, global_update, log_dir, wandb_run_id,
    trainable_params,
):
    denoiser.train()
    train_sampler.set_epoch(epoch)

    epoch_start = time.perf_counter()
    epoch_loss_sum = 0.0
    num_updates = 0
    step_times, data_times = [], []

    accum_obs_flow = 0.0
    accum_act_flow = 0.0
    accum_total = 0.0

    data_start = time.perf_counter()

    for step_idx, batch in enumerate(train_loader):
        micro_idx = step_idx % cfg.train.accum_grad_steps
        is_last_micro = micro_idx == cfg.train.accum_grad_steps - 1

        data_times.append(time.perf_counter() - data_start)

        images = batch["image"].to(device, non_blocking=True).to(torch.bfloat16)
        actions = batch["action"].to(device, non_blocking=True).to(torch.bfloat16)
        actions = actions[:, :, :cfg.denoiser.n_actions].unsqueeze(-2)

        torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        if micro_idx == 0:
            optim.zero_grad(set_to_none=True)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_clean = tokenizer.encode(images).detach().clone()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            diffused_info = diffuser(z_clean, actions)
            obs_flow_loss, act_flow_loss = compute_uwm_loss(
                diffused_info, denoiser, device=device
            )
            loss_micro = (obs_flow_loss + act_flow_loss) / cfg.train.accum_grad_steps

        loss_micro.backward()

        accum_obs_flow += obs_flow_loss.mean().item()
        accum_act_flow += act_flow_loss.mean().item()
        accum_total += loss_micro.item()

        if is_last_micro:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.train.clip_grad_norm)
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
                tb_writer.add_scalar("train/total_loss", sync_loss, global_update)
                tb_writer.add_scalar("train/obs_flow_loss", accum_obs_flow, global_update)
                tb_writer.add_scalar("train/act_flow_loss", accum_act_flow, global_update)
                tb_writer.add_scalar("train/lr", lr, global_update)

                if global_update % cfg.print_every == 0:
                    print(
                        f"  [step {global_update}]"
                        f"  loss: {sync_loss:.4f}"
                        f"  obs: {accum_obs_flow:.4f}"
                        f"  act: {accum_act_flow:.4f}"
                        f"  lr: {lr:.2e}"
                    )

                if global_update % cfg.save_every == 0:
                    save_lora_checkpoint(
                        ckpt_path=os.path.join(log_dir, f"adapter_{global_update}.pt"),
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
            accum_total = 0.0

        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_start)
        data_start = time.perf_counter()

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = epoch_loss_sum / num_updates if num_updates > 0 else 0.0
    total_frames = (
        cfg.train.batch_per_gpu * cfg.denoiser.max_sequence_length * len(train_loader)
    )
    epoch_fps = total_frames / epoch_time

    if rank == 0:
        avg_step = sum(step_times) / max(len(step_times), 1)
        avg_data = sum(data_times) / max(len(data_times), 1)
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
@hydra.main(config_path="config", config_name="dynamics/lewm-cubes-lora", version_base=None)
def main(cfg: DictConfig):
    torch.backends.cuda.matmul.allow_tf32 = cfg.train.enable_fast_matmul

    rank, local_rank, world_size, device = setup_distributed()
    torch.manual_seed(cfg.seed + rank)

    if rank == 0:
        print(f"Distributed LoRA finetuning: {world_size} GPU(s)")
        effective_batch = cfg.train.batch_per_gpu * world_size * cfg.train.accum_grad_steps
        print(f"Effective global batch size: {effective_batch}")

    train_loader, train_sampler = build_dataloader(cfg, rank, world_size)

    if rank == 0:
        print("Building models + LoRA wrap...")
    tokenizer, denoiser, diffuser = build_models(cfg, device, local_rank, rank)

    trainable_params = [p for p in denoiser.parameters() if p.requires_grad]
    if rank == 0:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in denoiser.parameters())
        print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.3f}%)")

    optim = torch.optim.AdamW(
        trainable_params,
        lr=cfg.train.lr,
        weight_decay=cfg.train.get("weight_decay", 0.0),
    )
    steps_per_epoch = len(train_loader)
    total_steps = cfg.train.num_epochs * steps_per_epoch // cfg.train.accum_grad_steps
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    wandb_run_id = cfg.wandb.run_name
    log_dir = None
    start_epoch = 0
    global_update = 0

    if cfg.reload_checkpoint is not None:
        if rank == 0:
            print(f"Resuming from LoRA checkpoint: {cfg.reload_checkpoint}")
        start_epoch, global_update, wandb_run_id, log_dir = load_lora_checkpoint(
            ckpt_path=cfg.reload_checkpoint,
            model=denoiser,
            optim=optim,
            scheduler=scheduler,
            rank=rank,
        )
    elif rank == 0:
        print("Starting LoRA from scratch.")

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

    epoch_losses, epoch_times, epoch_fps_vals = [], [], []
    if rank == 0:
        print("Starting LoRA training...")

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
            trainable_params=trainable_params,
        )
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        epoch_fps_vals.append(epoch_fps)

        if rank == 0:
            save_lora_checkpoint(
                ckpt_path=os.path.join(log_dir, f"adapter_{global_update}.pt"),
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

    # --- Final merged save ---
    if cfg.lora.get("save_merged_final", True):
        merged_path = os.path.join(log_dir, "final_merged.pt")
        save_merged_final(denoiser, merged_path, rank)
    dist.barrier()

    if rank == 0:
        cur_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\n{'='*60}")
        print("LoRA Training Complete!")
        print(f"  Avg loss:      {sum(epoch_losses) / max(len(epoch_losses), 1):.6f}")
        print(f"  Avg epoch time:{sum(epoch_times) / max(len(epoch_times), 1):.2f}s")
        print(f"  Avg FPS:       {sum(epoch_fps_vals) / max(len(epoch_fps_vals), 1):.2f}")
        print(f"  GPU memory:    {cur_alloc:.2f} GB current / {peak_alloc:.2f} GB peak")
        print(f"{'='*60}")

        tb_writer.close()
        if cfg.wandb.enable:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
