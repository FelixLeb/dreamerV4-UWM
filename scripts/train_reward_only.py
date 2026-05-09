"""Reward-head fine-tuning script (Option 2 from the design discussion).

Loads a pretrained dynamics checkpoint, *freezes everything except* the
`agent_token` parameter and the `reward_head` submodule, and trains those
to predict bucketized symlog rewards via the MTP head. The frozen world
model is structurally insulated by the agent-token isolation mask, so this
training cannot regress modes 1 / 2 by construction.

Noise schedule: simpler than `train_dynamics_uwm.py`. Per batch element we
flip a Bernoulli(`reward_clean_prob`); on heads we pin τ=clean for both
state & action, on tails we sample τ uniformly per-frame from the full
diffusion grid. Default 70/30 — primarily clean (the head's main inference
use case) with some exposure to noisier conditioning so the head doesn't
brittle-fail when run on intermediate diffusion samples.

Single training branch — no long/short/image branch logic. The dataset
loader is requested at `window_size = context_length` so each sample is
already the right size; no random crop needed.
"""

import math
import os
import time

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from dreamerv4uwm.datasets import (
    create_distributed_dataloader,
    create_distributed_demo_play_dataloader,
)
from dreamerv4uwm.loss import compute_reward_mtp_loss
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
# Forward process — clean / uniform-noisy mixture
# ---------------------------------------------------------------------------

class RewardOnlyForwardProcess(nn.Module):
    """Two-mode noise schedule for reward-head fine-tuning.

    Per batch element draws Bernoulli(`clean_prob`):
      - clean (prob `clean_prob`):    obs τ, act τ pinned to ~1 across all T.
      - noisy (prob 1 - `clean_prob`): per-frame independent uniform draw on
                                        the diffusion grid for both streams.
    """

    def __init__(self, max_diff_steps: int, clean_prob: float, action_noise_std: float, device):
        super().__init__()
        assert 0.0 <= clean_prob <= 1.0
        self.max_diff_steps = int(max_diff_steps)
        self.clean_prob = float(clean_prob)
        self.action_noise_std = float(action_noise_std)
        self.device = device

    def forward(self, z_clean: torch.Tensor, a_clean: torch.Tensor):
        B, T, _, _ = z_clean.shape
        device = z_clean.device

        is_clean = (torch.rand(B, device=device) < self.clean_prob).unsqueeze(-1)  # (B, 1)
        clean_idx = torch.full(
            (B, T), self.max_diff_steps - 1, device=device, dtype=torch.long,
        )
        rand_obs_idx = torch.randint(0, self.max_diff_steps, (B, T), device=device)
        rand_act_idx = torch.randint(0, self.max_diff_steps, (B, T), device=device)
        obs_tau_idx = torch.where(is_clean, clean_idx, rand_obs_idx)
        act_tau_idx = torch.where(is_clean, clean_idx, rand_act_idx)

        obs_tau = obs_tau_idx.float() / self.max_diff_steps
        act_tau = act_tau_idx.float() / self.max_diff_steps

        z0 = torch.randn_like(z_clean)
        a0 = self.action_noise_std * torch.randn_like(a_clean)
        obs_tau_b = obs_tau.unsqueeze(-1).unsqueeze(-1)
        act_tau_b = act_tau.unsqueeze(-1).unsqueeze(-1)
        z_tau = (1.0 - obs_tau_b) * z0 + obs_tau_b * z_clean
        a_tau = (1.0 - act_tau_b) * a0 + act_tau_b * a_clean

        return {
            "x_tau": z_tau,
            "obs_tau_idx": obs_tau_idx,
            "a_tau": a_tau,
            "act_tau_idx": act_tau_idx,
            "frac_clean": is_clean.float().mean().detach(),
        }


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def build_dataloader(cfg, rank, world_size):
    # Reward-only training uses fixed-length windows == context_length, so no
    # crop is needed downstream. This differs from train_dynamics_uwm.py which
    # pulls long windows and slices per branch.
    common = dict(
        window_size=cfg.denoiser.context_length,
        batch_size=int(cfg.train.batch_per_gpu),
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
    )
    # When `dataset.use_demo_play_mix=true`, swap to the synthetic-reward
    # loader: 50/50 demo vs. play, reward = play_reward (default 0) on play,
    # demo_reward (default 1) on demo with the trailing `goal_frames` of each
    # demo episode bumped to `goal_reward` (default 20). Stop-gap until real
    # per-frame rewards are written into the shards.
    if bool(cfg.dataset.get("use_demo_play_mix", False)):
        loader, sampler, _ = create_distributed_demo_play_dataloader(
            demo_data_dir=cfg.dataset.demo_data_dir,
            play_data_dir=cfg.dataset.play_data_dir,
            goal_reward=float(cfg.dataset.get("goal_reward", 20.0)),
            demo_reward=float(cfg.dataset.get("demo_reward", 1.0)),
            play_reward=float(cfg.dataset.get("play_reward", 0.0)),
            goal_frames=int(cfg.dataset.get("goal_frames", 5)),
            terminal_bias=float(cfg.dataset.get("terminal_bias", 0.5)),
            **common,
        )
    else:
        loader, sampler, _ = create_distributed_dataloader(
            data_dir=cfg.dataset.data_dir, **common,
        )
    return loader, sampler


def freeze_world_model(denoiser: nn.Module):
    """Set requires_grad=False everywhere except agent_token + reward_head.

    Returns (n_trainable, n_frozen) parameter counts.
    """
    n_trainable = 0
    n_frozen = 0
    found_agent = False
    found_reward = False
    for name, p in denoiser.named_parameters():
        is_reward_param = (
            name.endswith("agent_token")
            or ".reward_head." in name
            or name.endswith("reward_head.weight")  # defensive
        )
        p.requires_grad_(is_reward_param)
        if is_reward_param:
            n_trainable += p.numel()
            if name.endswith("agent_token"):
                found_agent = True
            if ".reward_head." in name:
                found_reward = True
        else:
            n_frozen += p.numel()
    assert found_agent, (
        "freeze_world_model: did not find an `agent_token` parameter — was the "
        "denoiser built with denoiser.train_reward_model=True?"
    )
    assert found_reward, (
        "freeze_world_model: did not find any `reward_head.*` parameters — was "
        "the denoiser built with denoiser.train_reward_model=True?"
    )
    return n_trainable, n_frozen


def build_models(cfg, device, local_rank):
    assert cfg.denoiser.get("train_reward_model", False), (
        "train_reward_only requires denoiser.train_reward_model=true"
    )
    assert cfg.dynamics_ckpt is not None, (
        "train_reward_only requires a pretrained dynamics checkpoint via "
        "dynamics_ckpt — there is nothing to fine-tune from otherwise."
    )

    tokenizer = load_tokenizer(
        cfg, device=device, max_num_forward_steps=cfg.denoiser.max_sequence_length,
    )

    # strict=False so the agent_token + reward_head random init is preserved;
    # everything else is overwritten from the pretrained dynamics checkpoint.
    print(f"Loading dynamics weights (strict=False) from: {cfg.dynamics_ckpt}")
    denoiser = load_denoiser(
        cfg, device=device, model_key="model",
        max_num_forward_steps=cfg.denoiser.max_sequence_length,
        strict=False,
    )

    n_trainable, n_frozen = freeze_world_model(denoiser)

    diffuser = RewardOnlyForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        clean_prob=float(cfg.train.get("reward_clean_prob", 0.7)),
        action_noise_std=float(cfg.train.get("action_noise_std", 1.0)),
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

    # agent_token + reward_head are always exercised in forward when
    # train_reward_model=True, so DDP doesn't see any unused trainable params.
    denoiser = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=False)

    return tokenizer, denoiser, diffuser, n_trainable, n_frozen


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
                project=cfg.wandb.project, id=wandb_run_id, resume="allow",
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
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    epoch, train_loader, train_sampler, tokenizer, denoiser, diffuser,
    optim, scheduler, tb_writer, cfg, rank, device, global_update,
    log_dir, wandb_run_id,
):
    denoiser.train()
    train_sampler.set_epoch(epoch)

    epoch_start = time.perf_counter()
    epoch_loss_sum = 0.0
    num_updates = 0
    step_times = []
    data_times = []

    accum_reward = 0.0
    accum_frac_clean = 0.0
    data_start = time.perf_counter()

    for step_idx, batch in enumerate(train_loader):
        micro_idx = step_idx % cfg.train.accum_grad_steps
        is_last_micro = micro_idx == cfg.train.accum_grad_steps - 1
        data_times.append(time.perf_counter() - data_start)

        if "reward" not in batch:
            raise RuntimeError(
                "train_reward_only: batch is missing the 'reward' field. The "
                "shards at dataset.data_dir do not contain a 'rewards' "
                "dataset — generate / re-export them before running this "
                "script."
            )

        images = batch["image"].to(device, non_blocking=True).to(torch.bfloat16)
        actions = batch["action"].to(device, non_blocking=True).to(torch.bfloat16)
        actions = actions[:, :, :cfg.denoiser.n_actions].unsqueeze(-2)  # (B,T,1,A)
        # Rewards stay fp32 — symlog + CE underflows in bf16 (gate c).
        rewards = batch["reward"].to(device, non_blocking=True).float()
        if rewards.dim() == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)

        torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        if micro_idx == 0:
            optim.zero_grad(set_to_none=True)

        # --- Forward ---
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_clean = tokenizer.encode(images).detach().clone()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            diffused = diffuser(z_clean, actions)
            B, T = z_clean.shape[:2]
            step_idx_t = torch.zeros((B, T), dtype=torch.long, device=device)
            _, _, pred_rewards = denoiser(
                noisy_act=diffused["a_tau"].squeeze(-2),
                noisy_obs=diffused["x_tau"],
                obs_sigma_idx=diffused["obs_tau_idx"],
                obs_step_idx=step_idx_t,
                act_sigma_idx=diffused["act_tau_idx"],
                act_step_idx=step_idx_t,
            )
            assert pred_rewards is not None, (
                "denoiser returned pred_rewards=None — train_reward_model is "
                "off in the model. Check that cfg.denoiser.train_reward_model "
                "is True."
            )
            # compute_reward_mtp_loss casts logits to fp32 internally.
            reward_loss = compute_reward_mtp_loss(pred_rewards, rewards)
            loss_micro = reward_loss / cfg.train.accum_grad_steps

        loss_micro.backward()

        accum_reward += reward_loss.item()
        accum_frac_clean += float(diffused["frac_clean"].item())

        if is_last_micro:
            torch.nn.utils.clip_grad_norm_(
                [p for p in denoiser.parameters() if p.requires_grad], max_norm=1.0,
            )
            optim.step()
            scheduler.step()
            global_update += 1

            # All-reduce reward across ranks for logging.
            t = torch.tensor([accum_reward / cfg.train.accum_grad_steps], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            sync_reward = t.item()
            epoch_loss_sum += sync_reward
            num_updates += 1

            if rank == 0:
                lr = scheduler.get_last_lr()[0]
                tb_writer.add_scalar("train/reward_loss", sync_reward, global_update)
                tb_writer.add_scalar(
                    "train/frac_clean",
                    accum_frac_clean / cfg.train.accum_grad_steps,
                    global_update,
                )
                tb_writer.add_scalar("train/lr", lr, global_update)

                if global_update % cfg.print_every == 0:
                    print(
                        f"  [step {global_update}]  reward: {sync_reward:.4f}"
                        f"  frac_clean: {accum_frac_clean / cfg.train.accum_grad_steps:.2f}"
                        f"  lr: {lr:.2e}"
                    )

                if global_update % cfg.save_every == 0:
                    print(f"[Checkpoint] Saving at global_update={global_update}")
                    save_ddp_checkpoint(
                        ckpt_path=os.path.join(log_dir, f"{global_update}.pt"),
                        epoch=epoch, global_update=global_update,
                        model=denoiser, optim=optim, scheduler=scheduler,
                        rank=rank, wandb_run_id=wandb_run_id, log_dir=log_dir,
                    )

            accum_reward = 0.0
            accum_frac_clean = 0.0

        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_start)
        data_start = time.perf_counter()

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = epoch_loss_sum / num_updates if num_updates > 0 else 0.0
    total_frames = int(cfg.train.batch_per_gpu) * cfg.denoiser.context_length * len(train_loader)
    epoch_fps = total_frames / epoch_time

    if rank == 0:
        avg_step = sum(step_times) / len(step_times) if step_times else 0.0
        avg_data = sum(data_times) / len(data_times) if data_times else 0.0
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Reward Loss:     {avg_loss:.6f}")
        print(f"  Epoch Time:      {epoch_time:.2f}s")
        print(f"  Throughput:      {epoch_fps:.2f} FPS")
        print(f"  Avg Step Time:   {avg_step:.3f}s")
        print(f"  Avg Data Time:   {avg_data:.3f}s")
        print(f"{'='*60}\n")

    return global_update, avg_loss, epoch_time, epoch_fps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@record
@hydra.main(config_path="config", config_name="dynamics/pushT-reward-only", version_base=None)
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

    train_loader, train_sampler = build_dataloader(cfg, rank, world_size)

    if rank == 0:
        print("Building models...")
    tokenizer, denoiser, diffuser, n_trainable, n_frozen = build_models(
        cfg, device, local_rank,
    )
    if rank == 0:
        print(
            f"Trainable parameters: {n_trainable:,}  |  "
            f"Frozen parameters: {n_frozen:,}  |  "
            f"Trainable fraction: {n_trainable / (n_trainable + n_frozen):.4%}"
        )

    trainable_params = [p for p in denoiser.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
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
            print(f"Resuming from checkpoint: {cfg.reload_checkpoint}")
        start_epoch, global_update, wandb_run_id, log_dir = load_ddp_checkpoint(
            ckpt_path=cfg.reload_checkpoint, model=denoiser,
            optim=optim, scheduler=scheduler, rank=rank,
        )
    elif rank == 0:
        print("Starting reward-head training from pretrained WM checkpoint.")

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
        print("Starting training...")

    for epoch in range(start_epoch, cfg.train.num_epochs):
        global_update, avg_loss, epoch_time, epoch_fps = train_epoch(
            epoch=epoch, train_loader=train_loader, train_sampler=train_sampler,
            tokenizer=tokenizer, denoiser=denoiser, diffuser=diffuser,
            optim=optim, scheduler=scheduler, tb_writer=tb_writer, cfg=cfg,
            rank=rank, device=device, global_update=global_update,
            log_dir=log_dir, wandb_run_id=wandb_run_id,
        )
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        epoch_fps_vals.append(epoch_fps)

        if rank == 0:
            save_ddp_checkpoint(
                ckpt_path=os.path.join(log_dir, f"{global_update}.pt"),
                epoch=epoch, global_update=global_update,
                model=denoiser, optim=optim, scheduler=scheduler,
                rank=rank, wandb_run_id=wandb_run_id, log_dir=log_dir,
            )
        dist.barrier()

    if rank == 0:
        cur_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\n{'='*60}")
        print("Reward-head training complete.")
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
