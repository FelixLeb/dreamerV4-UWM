"""
LoRA finetuning of the tokenizer (encoder + decoder).

Mirrors `train_tokenizer.py` but:
  - REQUIRES `cfg.tokenizer_ckpt` (LoRA finetunes a base model).
  - Wraps `TokenizerWrapper` with PEFT LoRA before DDP.
  - Uses **DDP** instead of FSDP — with base weights frozen, FSDP's sharding
    overhead is not worth the (small) memory savings.
  - Saves a small adapter-only `.pt` per `save_every`, plus an optional final
    merged full state_dict that is drop-in compatible with `load_tokenizer`.

Resume: pass `cfg.reload_checkpoint=<path-to-adapter-ckpt>.pt`.
"""

import math
import os
import time
from pathlib import Path

import hydra
import lpips
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from dreamerv4uwm.datasets import create_distributed_dataloader
from dreamerv4uwm.models.utils import load_tokenizer
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
# RMS loss scaler (same as full-finetune script)
# ---------------------------------------------------------------------------

class RMSLossScaler:
    def __init__(self, decay: float = 0.99, eps: float = 1e-8):
        self.decay = decay
        self.eps = eps
        self.ema_sq = {}

    def __call__(self, name: str, value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean_sq = value.detach().pow(2).mean()
            if dist.is_initialized():
                dist.all_reduce(mean_sq, op=dist.ReduceOp.AVG)
            if name not in self.ema_sq:
                self.ema_sq[name] = mean_sq
            else:
                self.ema_sq[name] = (
                    self.decay * self.ema_sq[name] + (1.0 - self.decay) * mean_sq
                )
            rms = (self.ema_sq[name] + self.eps).sqrt()
        return value / rms


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
    model,
    optim: torch.optim.Optimizer,
    scheduler,
    rms_norm: RMSLossScaler,
    rank: int,
    wandb_run_id: str = None,
    log_dir: str = None,
):
    if rank != 0:
        return
    peft_model = unwrap_model(model)
    adapter_state = get_peft_model_state_dict(peft_model)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "global_update": global_update,
        "adapter": adapter_state,
        "optim": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rms_norm": rms_norm.ema_sq,
        "wandb_run_id": wandb_run_id,
        "log_dir": log_dir,
    }
    torch.save(ckpt, ckpt_path)
    print(f"[rank0] Saved LoRA adapter checkpoint to {ckpt_path}")


def load_lora_checkpoint(
    ckpt_path: str,
    model,
    optim: torch.optim.Optimizer,
    scheduler,
    rms_norm: RMSLossScaler,
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
    rms_norm.ema_sq = ckpt.get("rms_norm", {})

    if rank == 0:
        print(f"Resuming LoRA from epoch {start_epoch + 1}, global_update {global_update}")
    return start_epoch, global_update, wandb_run_id, log_dir


def save_merged_final(model, ckpt_path: str, rank: int):
    if rank != 0:
        return
    peft_model = unwrap_model(model)
    base = peft_model.merge_and_unload()  # returns the underlying TokenizerWrapper
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": base.state_dict()}, ckpt_path)
    print(f"[rank0] Saved merged full checkpoint to {ckpt_path}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_dataloaders(cfg, rank, world_size):
    train_loader, train_sampler, _ = create_distributed_dataloader(
        data_dir=cfg.dataset.data_dir,
        window_size=cfg.tokenizer.max_sequence_length,
        batch_size=cfg.train.batch_per_gpu,
        rank=rank,
        world_size=world_size,
        num_workers=cfg.train.num_workers,
        stride=1,
        seed=cfg.seed,
        split="train",
        train_fraction=0.9,
        split_seed=cfg.dataset.split_seed,
        shuffle=True,
        drop_last=True,
    )
    test_loader, test_sampler, _ = create_distributed_dataloader(
        data_dir=cfg.dataset.data_dir,
        window_size=cfg.tokenizer.max_sequence_length,
        batch_size=cfg.train.batch_per_gpu,
        rank=rank,
        world_size=world_size,
        num_workers=cfg.train.num_workers,
        stride=1,
        seed=cfg.seed,
        split="test",
        train_fraction=0.9,
        split_seed=cfg.dataset.split_seed,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, train_sampler, test_loader, test_sampler


def build_lpips(rank, device):
    if rank == 0:
        lpips_model = lpips.LPIPS(net="vgg").to(device=device, dtype=torch.bfloat16)
    else:
        lpips_model = lpips.LPIPS(net="vgg", pretrained=False).to(device=device, dtype=torch.bfloat16)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    sd = lpips_model.state_dict() if rank == 0 else None
    obj_list = [sd]
    dist.broadcast_object_list(obj_list, src=0)
    if rank != 0:
        lpips_model.load_state_dict(obj_list[0])
    return lpips_model


def build_tokenizer(cfg, device, local_rank, rank):
    assert cfg.tokenizer_ckpt, (
        "LoRA finetuning requires `cfg.tokenizer_ckpt` to point at a pretrained tokenizer."
    )
    if rank == 0:
        print(f"Loading pretrained tokenizer from: {cfg.tokenizer_ckpt}")
    tokenizer = load_tokenizer(
        cfg, device=device, max_num_forward_steps=cfg.tokenizer.max_sequence_length
    ).to(device)

    lora_cfg = build_lora_config(cfg)
    tokenizer = get_peft_model(tokenizer, lora_cfg)
    if rank == 0:
        tokenizer.print_trainable_parameters()

    # Reconstruction touches every linear in encoder+decoder, so no unused params.
    tokenizer = DDP(tokenizer, device_ids=[local_rank], find_unused_parameters=False)
    return tokenizer


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
# Losses
# ---------------------------------------------------------------------------

def compute_recon_losses(x_hat, images, lpips_model):
    mse_loss = nn.functional.mse_loss(x_hat, images)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        images_lpips = (images * 2.0) - 1.0
        x_hat_lpips = (x_hat * 2.0) - 1.0
        B, T, C, H, W = images_lpips.shape
        images_lpips_flat = images_lpips.view(B * T, C, H, W)
        x_hat_lpips_flat = x_hat_lpips.view(B * T, C, H, W)
        lpips_loss = lpips_model(x_hat_lpips_flat, images_lpips_flat).mean()
    return mse_loss, lpips_loss


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def train_epoch(
    epoch, train_loader, train_sampler,
    tokenizer, lpips_model, rms_norm,
    optim, scheduler, tb_writer,
    cfg, rank, world_size, device, global_update, log_dir, wandb_run_id,
    trainable_params,
):
    tokenizer.train()
    train_sampler.set_epoch(epoch)

    epoch_start = time.perf_counter()
    epoch_loss_sum = 0.0
    num_updates = 0
    step_times, data_times = [], []

    accum_mse = 0.0
    accum_lpips = 0.0
    accum_raw_loss = 0.0
    accum_norm = 0.0

    steps_per_epoch = len(train_loader)
    data_start = time.perf_counter()

    for step_idx, batch in enumerate(train_loader):
        micro_idx = step_idx % cfg.train.grad_accum_steps
        is_last_micro = micro_idx == cfg.train.grad_accum_steps - 1

        data_times.append(time.perf_counter() - data_start)

        images = batch["image"].to(device, non_blocking=True).to(torch.bfloat16)

        torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        if micro_idx == 0:
            optim.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_hat = tokenizer(images)

        mse_loss, lpips_loss = compute_recon_losses(x_hat, images, lpips_model)
        raw_loss = mse_loss + cfg.train.lpips_weight * lpips_loss

        accum_mse += mse_loss.detach().item()
        accum_lpips += lpips_loss.detach().item()
        accum_raw_loss += raw_loss.detach().item()

        mse_norm = rms_norm("mse", mse_loss)
        lpips_norm = rms_norm("lpips", lpips_loss)
        loss_micro = (mse_norm + cfg.train.lpips_weight * lpips_norm) / cfg.train.grad_accum_steps
        accum_norm += loss_micro.detach().item()

        if not is_last_micro:
            with tokenizer.no_sync():
                loss_micro.backward()
        else:
            loss_micro.backward()

        if is_last_micro:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.train.clip_grad_norm)
            optim.step()
            scheduler.step()
            global_update += 1

            stats = torch.tensor(
                [accum_norm, accum_mse, accum_lpips, accum_raw_loss], device=device
            )
            dist.all_reduce(stats, op=dist.ReduceOp.AVG)
            sync_norm, sync_mse, sync_lpips, sync_raw = stats.tolist()
            epoch_loss_sum += sync_norm
            num_updates += 1

            mse_mean = sync_mse / cfg.train.grad_accum_steps
            lpips_mean = sync_lpips / cfg.train.grad_accum_steps
            raw_mean = sync_raw / cfg.train.grad_accum_steps

            if rank == 0 and num_updates % cfg.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                tb_writer.add_scalar("train/mse_mean", mse_mean, global_update)
                tb_writer.add_scalar("train/lpips_mean", lpips_mean, global_update)
                tb_writer.add_scalar("train/raw_loss_mean", raw_mean, global_update)
                tb_writer.add_scalar("train/normalized_loss_mean", sync_norm, global_update)
                tb_writer.add_scalar("train/lr", lr, global_update)

            if global_update % cfg.save_every == 0:
                save_lora_checkpoint(
                    ckpt_path=os.path.join(log_dir, f"adapter_{global_update}.pt"),
                    epoch=epoch,
                    global_update=global_update,
                    model=tokenizer,
                    optim=optim,
                    scheduler=scheduler,
                    rms_norm=rms_norm,
                    rank=rank,
                    wandb_run_id=wandb_run_id,
                    log_dir=log_dir,
                )

            if rank == 0 and global_update % cfg.print_every == 0:
                avg_step_time = sum(step_times[-cfg.print_every:]) / max(
                    len(step_times[-cfg.print_every:]), 1
                )
                avg_data_time = sum(data_times[-cfg.print_every:]) / max(
                    len(data_times[-cfg.print_every:]), 1
                )
                frames_per_step = (
                    cfg.train.batch_per_gpu * cfg.tokenizer.max_sequence_length * world_size
                )
                step_fps = frames_per_step / avg_step_time if avg_step_time > 0 else float("inf")
                data_fps = frames_per_step / avg_data_time if avg_data_time > 0 else float("inf")
                print(
                    f"Epoch {epoch + 1}/{cfg.train.num_epochs} | "
                    f"Step {step_idx + 1}/{steps_per_epoch} | "
                    f"Loss: {sync_norm:.6f} | "
                    f"MSE: {mse_mean:.6f} | LPIPS: {lpips_mean:.6f} | "
                    f"Data: {avg_data_time:.3f}s ({data_fps:.1f} fps) | "
                    f"Compute: {avg_step_time:.3f}s ({step_fps:.1f} fps)"
                )

            accum_mse = 0.0
            accum_lpips = 0.0
            accum_raw_loss = 0.0
            accum_norm = 0.0

        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_start)
        data_start = time.perf_counter()

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = epoch_loss_sum / num_updates if num_updates > 0 else 0.0
    total_frames = (
        cfg.train.batch_per_gpu * cfg.tokenizer.context_length * world_size * steps_per_epoch
    )
    epoch_fps = total_frames / epoch_time

    if rank == 0:
        avg_step = sum(step_times) / max(len(step_times), 1)
        avg_data = sum(data_times) / max(len(data_times), 1)
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1} Train Summary:")
        print(f"  Train Loss:    {avg_loss:.6f}")
        print(f"  Epoch Time:    {epoch_time:.2f}s")
        print(f"  Total Frames:  {total_frames:,}")
        print(f"  Throughput:    {epoch_fps:.2f} FPS")
        print(f"  Avg Step Time: {avg_step:.3f}s")
        print(f"  Avg Data Time: {avg_data:.3f}s")
        print(f"{'=' * 60}\n")

    return global_update, avg_loss, epoch_time, epoch_fps


def validate(epoch, test_loader, test_sampler, tokenizer, lpips_model, cfg, rank, device, tb_writer):
    tokenizer.eval()
    test_sampler.set_epoch(epoch)

    val_loss_sum = 0.0
    val_mse_sum = 0.0
    val_lpips_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True).to(torch.bfloat16)
            x_hat = tokenizer(images)
            mse_loss, lpips_loss = compute_recon_losses(x_hat, images, lpips_model)
            val_loss = mse_loss + cfg.train.lpips_weight * lpips_loss

            val_loss_sum += val_loss.item()
            val_mse_sum += mse_loss.item()
            val_lpips_sum += lpips_loss.item()
            val_count += 1

    stats = torch.tensor(
        [val_loss_sum, val_mse_sum, val_lpips_sum, float(val_count)], device=device
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_count = max(int(stats[3].item()), 1)
    avg_val_loss = stats[0].item() / total_count
    avg_val_mse = stats[1].item() / total_count
    avg_val_lpips = stats[2].item() / total_count

    if rank == 0:
        tb_writer.add_scalar("val/raw_loss", avg_val_loss, epoch + 1)
        tb_writer.add_scalar("val/mse", avg_val_mse, epoch + 1)
        tb_writer.add_scalar("val/lpips", avg_val_lpips, epoch + 1)

    return avg_val_loss, avg_val_mse, avg_val_lpips


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@record
@hydra.main(config_path="config", config_name="tokenizer/pushT-lora", version_base=None)
def main(cfg: DictConfig):
    torch.backends.cuda.matmul.allow_tf32 = cfg.train.enable_fast_matmul

    rank, local_rank, world_size, device = setup_distributed()
    torch.manual_seed(cfg.seed + rank)

    if rank == 0:
        print(f"Distributed LoRA tokenizer finetuning: {world_size} GPU(s)")
        effective_batch = cfg.train.batch_per_gpu * world_size * cfg.train.grad_accum_steps
        print(f"Effective global batch size: {effective_batch}")

    train_loader, train_sampler, test_loader, test_sampler = build_dataloaders(
        cfg, rank, world_size
    )

    if rank == 0:
        print("Building LPIPS and tokenizer + LoRA wrap...")
    lpips_model = build_lpips(rank, device)
    tokenizer = build_tokenizer(cfg, device, local_rank, rank)

    trainable_params = [p for p in tokenizer.parameters() if p.requires_grad]
    if rank == 0:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in tokenizer.parameters())
        print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.3f}%)")

    optim = torch.optim.AdamW(
        trainable_params,
        lr=cfg.train.lr,
        weight_decay=cfg.train.get("weight_decay", 0.0),
    )
    steps_per_epoch = len(train_loader)
    total_steps = cfg.train.num_epochs * steps_per_epoch // cfg.train.grad_accum_steps
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    rms_norm = RMSLossScaler(decay=0.99, eps=1e-8)

    wandb_run_id = cfg.wandb.run_name
    log_dir = None
    start_epoch = 0
    global_update = 0

    if cfg.reload_checkpoint is not None:
        if rank == 0:
            print(f"Resuming from LoRA checkpoint: {cfg.reload_checkpoint}")
        start_epoch, global_update, wandb_run_id, log_dir = load_lora_checkpoint(
            ckpt_path=cfg.reload_checkpoint,
            model=tokenizer,
            optim=optim,
            scheduler=scheduler,
            rms_norm=rms_norm,
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
        print("Starting LoRA tokenizer training...")

    for epoch in range(start_epoch, cfg.train.num_epochs):
        global_update, avg_loss, epoch_time, epoch_fps = train_epoch(
            epoch=epoch,
            train_loader=train_loader,
            train_sampler=train_sampler,
            tokenizer=tokenizer,
            lpips_model=lpips_model,
            rms_norm=rms_norm,
            optim=optim,
            scheduler=scheduler,
            tb_writer=tb_writer,
            cfg=cfg,
            rank=rank,
            world_size=world_size,
            device=device,
            global_update=global_update,
            log_dir=log_dir,
            wandb_run_id=wandb_run_id,
            trainable_params=trainable_params,
        )
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        epoch_fps_vals.append(epoch_fps)

        optim.zero_grad(set_to_none=True)
        avg_val_loss, avg_val_mse, avg_val_lpips = validate(
            epoch=epoch,
            test_loader=test_loader,
            test_sampler=test_sampler,
            tokenizer=tokenizer,
            lpips_model=lpips_model,
            cfg=cfg,
            rank=rank,
            device=device,
            tb_writer=tb_writer,
        )

        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{cfg.train.num_epochs} Val Summary:")
            print(f"  Val Loss:  {avg_val_loss:.6f}")
            print(f"  Val MSE:   {avg_val_mse:.6f}")
            print(f"  Val LPIPS: {avg_val_lpips:.6f}")
            print(f"{'=' * 60}\n")

        save_lora_checkpoint(
            ckpt_path=os.path.join(log_dir, f"adapter_{global_update}.pt"),
            epoch=epoch,
            global_update=global_update,
            model=tokenizer,
            optim=optim,
            scheduler=scheduler,
            rms_norm=rms_norm,
            rank=rank,
            wandb_run_id=wandb_run_id,
            log_dir=log_dir,
        )
        dist.barrier()

    # --- Final merged save ---
    if cfg.lora.get("save_merged_final", True):
        merged_path = os.path.join(log_dir, "final_merged.pt")
        save_merged_final(tokenizer, merged_path, rank)
    dist.barrier()

    if rank == 0:
        cur_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\n{'=' * 60}")
        print("LoRA Tokenizer Training Complete!")
        print(f"  Avg loss:       {sum(epoch_losses) / max(len(epoch_losses), 1):.6f}")
        print(f"  Avg epoch time: {sum(epoch_times) / max(len(epoch_times), 1):.2f}s")
        print(f"  Avg FPS:        {sum(epoch_fps_vals) / max(len(epoch_fps_vals), 1):.2f}")
        print(f"  GPU memory:     {cur_alloc:.2f} GB current / {peak_alloc:.2f} GB peak")
        print(f"{'=' * 60}")

        tb_writer.close()
        if cfg.wandb.enable:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
