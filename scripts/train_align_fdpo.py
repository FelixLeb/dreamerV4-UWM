"""
FDPO alignment of a play-trained dynamics denoiser.

For each batch of (positive, negative) trajectory pairs from PreferencePairDataset:
  1. Tokenize both image streams (frozen tokenizer).
  2. Sample one (τ, ε) schedule via UWMForwardProcess (force_mode='policy') and
     apply it identically to pos and neg via `apply_diff` — this is the
     Diffusion-DPO variance-reduction trick.
  3. Run the LoRA-wrapped aligned denoiser on both with gradients.
  4. Run the frozen reference denoiser on both with no_grad.
  5. Combine per-sample flow-matching losses into compute_flow_dpo_loss + a
     small anchor term λ · ℓ_θ(ξ⁺).mean().
  6. Backprop only through LoRA parameters.

Both ref and aligned are loaded from the same `cfg.dynamics_ckpt`. LoRA save/load
helpers mirror `train_dynamics_uwm_lora.py`. Resume: `cfg.reload_checkpoint=<adapter>.pt`.
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
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from dreamerv4uwm.datasets import create_distributed_pair_dataloader
from dreamerv4uwm.loss import (
    UWMForwardProcess,
    compute_per_sample_uwm_loss,
    compute_flow_dpo_loss,
)
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
# LoRA helpers (mirror train_dynamics_uwm_lora.py)
# ---------------------------------------------------------------------------

def build_lora_config(cfg: DictConfig) -> LoraConfig:
    lora = cfg.lora
    target = lora.target_modules
    if (isinstance(target, (list, tuple)) or
            (hasattr(target, "__iter__") and not isinstance(target, str))):
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
        "wandb_run_id": wandb_run_id,
        "log_dir": log_dir,
    }
    torch.save(ckpt, ckpt_path)
    print(f"[rank0] Saved LoRA adapter checkpoint to {ckpt_path}")


def load_lora_checkpoint(ckpt_path: str, model, optim, scheduler, rank: int):
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
    if rank != 0:
        return
    peft_model = unwrap_model(model)
    base = peft_model.merge_and_unload()
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": base.state_dict()}, ckpt_path)
    print(f"[rank0] Saved merged full checkpoint to {ckpt_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_deprecated_keys(cfg):
    """Remove keys saved in older training configs that the current denoiser
    dataclass no longer accepts (e.g. latent_attends_action)."""
    if 'latent_attends_action' in cfg.denoiser:
        del cfg.denoiser['latent_attends_action']


def build_models(cfg, device, local_rank, rank):
    """Build the frozen reference and the LoRA-wrapped aligned denoiser.

    Both share weights at init (loaded from `cfg.dynamics_ckpt`). The
    reference stays bf16-eval-no_grad; the aligned copy is wrapped with PEFT
    LoRA and DDP.
    """
    assert cfg.dynamics_ckpt, (
        "FDPO alignment requires `cfg.dynamics_ckpt` to point at a pretrained denoiser."
    )
    _strip_deprecated_keys(cfg)

    tokenizer = load_tokenizer(
        cfg, device=device, max_num_forward_steps=cfg.denoiser.max_sequence_length
    ).to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    if rank == 0:
        print(f"Loading frozen reference denoiser from: {cfg.dynamics_ckpt}")
    ref_denoiser = load_denoiser(
        cfg, device=device, model_key="model",
        max_num_forward_steps=cfg.denoiser.max_sequence_length,
    ).to(device)
    ref_denoiser.eval()
    for p in ref_denoiser.parameters():
        p.requires_grad_(False)

    if rank == 0:
        print(f"Loading aligned denoiser (LoRA-wrapped) from: {cfg.dynamics_ckpt}")
    aligned_denoiser = load_denoiser(
        cfg, device=device, model_key="model",
        max_num_forward_steps=cfg.denoiser.max_sequence_length,
    ).to(device)

    lora_cfg = build_lora_config(cfg)
    aligned_denoiser = get_peft_model(aligned_denoiser, lora_cfg)
    if rank == 0:
        aligned_denoiser.print_trainable_parameters()

    aligned_denoiser = DDP(
        aligned_denoiser, device_ids=[local_rank], find_unused_parameters=False
    )

    diffuser = UWMForwardProcess(
        max_diff_steps=cfg.denoiser.num_noise_levels,
        mode_weights={cfg.fdpo.force_mode: 1.0},  # we always force a single mode
        device=device,
    )
    return tokenizer, ref_denoiser, aligned_denoiser, diffuser


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
# FDPO step
# ---------------------------------------------------------------------------

def _encode(tokenizer, images_bt_chw):
    """Tokenize a (B, T, C, H, W) image batch under autocast and detach."""
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return tokenizer.encode(images_bt_chw).detach()


_DPO_STREAM_KEY = {
    "obs":   "obs_flow_loss",
    "act":   "act_flow_loss",
    "both":  "total_flow_loss",
}


def fdpo_step(
    pos_image, pos_action, neg_image, neg_action,
    tokenizer, ref_denoiser, aligned_denoiser, diffuser,
    cfg, device,
    play_image=None, play_action=None,
):
    """Compute the FDPO + anchor loss for one (pos, neg) [+ optional play] batch.

    The DPO contrast is computed over the stream(s) chosen by
    `cfg.fdpo.dpo_streams` ∈ {'obs', 'act', 'both'} (default 'both'). Restricting
    to 'obs' closes the trivial action-noise discrimination loophole — pos
    actions are human-recorded smooth, neg actions are sampler-rolled noisy, so
    a 'both' contrast can saturate on smoothness instead of expert behavior.

    If `play_image`/`play_action` are provided, a play-data WM-mode anchor term
    is added: `λ_wm · ℓ_θ(play, force_mode=wm_anchor_force_mode)`. This is the
    "preserve world-model competence on the raw play distribution" anchor.

    Returns a dict; `loss` has grad attached.
    """
    n_act = cfg.denoiser.n_actions

    pos_z = _encode(tokenizer, pos_image)
    neg_z = _encode(tokenizer, neg_image)
    pos_a = pos_action[:, :, :n_act].unsqueeze(-2)
    neg_a = neg_action[:, :, :n_act].unsqueeze(-2)

    B, T = pos_z.shape[:2]

    # Shared (τ, ε) across pos/neg per pair.
    obs_diff, act_diff, ctx_len, mode = diffuser.sample_step_noise(
        B, T, force_mode=cfg.fdpo.force_mode,
    )
    z0 = torch.randn_like(pos_z)
    a0 = diffuser.action_noise_std * torch.randn_like(pos_a)
    pos_info = diffuser.apply_diff(pos_z, pos_a, obs_diff, act_diff, ctx_len, mode, z0=z0, a0=a0)
    neg_info = diffuser.apply_diff(neg_z, neg_a, obs_diff, act_diff, ctx_len, mode, z0=z0, a0=a0)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        per_aligned_pos = compute_per_sample_uwm_loss(pos_info, aligned_denoiser, device=device)
        per_aligned_neg = compute_per_sample_uwm_loss(neg_info, aligned_denoiser, device=device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            per_ref_pos = compute_per_sample_uwm_loss(pos_info, ref_denoiser, device=device)
            per_ref_neg = compute_per_sample_uwm_loss(neg_info, ref_denoiser, device=device)

    streams = str(cfg.fdpo.get("dpo_streams", "both"))
    if streams not in _DPO_STREAM_KEY:
        raise ValueError(f"cfg.fdpo.dpo_streams must be one of {list(_DPO_STREAM_KEY)}, "
                         f"got {streams!r}")
    key = _DPO_STREAM_KEY[streams]
    dpo = compute_flow_dpo_loss(
        theta_loss_pos=per_aligned_pos[key],
        theta_loss_neg=per_aligned_neg[key],
        ref_loss_pos=per_ref_pos[key],
        ref_loss_neg=per_ref_neg[key],
        beta=float(cfg.fdpo.beta),
    )

    # Demo BC anchor — still on total_flow_loss to keep the metric comparable
    # across stream choices.
    anchor = per_aligned_pos["total_flow_loss"].mean()
    total = dpo["loss"] + float(cfg.fdpo.anchor_weight) * anchor

    # Play-WM anchor: keeps world-model competence on the raw play distribution
    # intact, avoiding the bias of anchoring against ref-induced negatives.
    play_wm_anchor = torch.zeros((), device=device)
    if play_image is not None and play_action is not None:
        play_z = _encode(tokenizer, play_image)
        play_a = play_action[:, :, :n_act].unsqueeze(-2)
        Bp, Tp = play_z.shape[:2]
        wm_mode = str(cfg.fdpo.get("wm_anchor_force_mode", "wm"))
        p_obs_diff, p_act_diff, p_ctx_len, p_mode = diffuser.sample_step_noise(
            Bp, Tp, force_mode=wm_mode,
        )
        p_z0 = torch.randn_like(play_z)
        p_a0 = diffuser.action_noise_std * torch.randn_like(play_a)
        play_info = diffuser.apply_diff(
            play_z, play_a, p_obs_diff, p_act_diff, p_ctx_len, p_mode,
            z0=p_z0, a0=p_a0,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            per_play = compute_per_sample_uwm_loss(play_info, aligned_denoiser, device=device)
        play_wm_anchor = per_play["total_flow_loss"].mean()
        total = total + float(cfg.fdpo.get("wm_anchor_weight", 0.0)) * play_wm_anchor

    return {
        "loss":           total,
        "dpo_loss":       dpo["loss"].detach(),
        "anchor_loss":    anchor.detach(),
        "play_wm_anchor": play_wm_anchor.detach(),
        "margin_mean":    dpo["margin"].mean(),
        "reward_pos":     dpo["reward_pos"].mean(),
        "reward_neg":     dpo["reward_neg"].mean(),
        "accuracy":       dpo["accuracy"],
        "aligned_pos_loss": per_aligned_pos["total_flow_loss"].mean().detach(),
        "aligned_neg_loss": per_aligned_neg["total_flow_loss"].mean().detach(),
        "ref_pos_loss":     per_ref_pos["total_flow_loss"].mean().detach(),
        "ref_neg_loss":     per_ref_neg["total_flow_loss"].mean().detach(),
    }


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------

def _cycling_iter(loader, sampler=None, epoch_ref=None):
    """Yield batches from `loader` forever, restarting the iterator each pass.
    If a DistributedSampler is passed, its `set_epoch` is advanced on restart.
    """
    pass_idx = 0
    while True:
        if sampler is not None:
            sampler.set_epoch((epoch_ref[0] if epoch_ref else 0) * 10_000 + pass_idx)
        for b in loader:
            yield b
        pass_idx += 1


def train_epoch(
    epoch, train_loader, train_sampler,
    tokenizer, ref_denoiser, aligned_denoiser, diffuser,
    optim, scheduler, tb_writer,
    cfg, rank, device, global_update, log_dir, wandb_run_id,
    trainable_params,
    play_iter=None,
):
    aligned_denoiser.train()
    train_sampler.set_epoch(epoch)

    epoch_start = time.perf_counter()
    epoch_loss_sum = 0.0
    num_updates = 0

    accum_total = 0.0
    accum_dpo = 0.0
    accum_anchor = 0.0
    accum_play_wm = 0.0
    accum_margin = 0.0
    accum_acc = 0.0

    for step_idx, batch in enumerate(train_loader):
        micro_idx = step_idx % cfg.train.accum_grad_steps
        is_last_micro = micro_idx == cfg.train.accum_grad_steps - 1

        if micro_idx == 0:
            optim.zero_grad(set_to_none=True)

        pos_image  = batch["pos_image"].to(device,  non_blocking=True).to(torch.bfloat16)
        neg_image  = batch["neg_image"].to(device,  non_blocking=True).to(torch.bfloat16)
        pos_action = batch["pos_action"].to(device, non_blocking=True).to(torch.bfloat16)
        neg_action = batch["neg_action"].to(device, non_blocking=True).to(torch.bfloat16)

        play_image = play_action = None
        if play_iter is not None:
            play_batch = next(play_iter)
            play_image  = play_batch["image"].to(device,  non_blocking=True).to(torch.bfloat16)
            play_action = play_batch["action"].to(device, non_blocking=True).to(torch.bfloat16)

        out = fdpo_step(
            pos_image, pos_action, neg_image, neg_action,
            tokenizer, ref_denoiser, aligned_denoiser, diffuser,
            cfg, device,
            play_image=play_image, play_action=play_action,
        )
        loss_micro = out["loss"] / cfg.train.accum_grad_steps
        loss_micro.backward()

        accum_total  += out["loss"].item()
        accum_dpo    += out["dpo_loss"].item()
        accum_anchor += out["anchor_loss"].item()
        accum_play_wm += out["play_wm_anchor"].item()
        accum_margin += out["margin_mean"].item()
        accum_acc    += out["accuracy"].item()

        if is_last_micro:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.train.clip_grad_norm)
            optim.step()
            scheduler.step()
            global_update += 1

            # Average loss across ranks for logging.
            total_tensor = torch.tensor([accum_total], device=device)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.AVG)
            sync_loss = total_tensor.item()
            epoch_loss_sum += sync_loss
            num_updates += 1

            if rank == 0:
                # Accumulators are sums across `accum_grad_steps` micro-batches.
                # Divide for display so logs read as per-micro-batch averages.
                inv = 1.0 / cfg.train.accum_grad_steps
                lr = scheduler.get_last_lr()[0]
                tb_writer.add_scalar("train/total_loss",  sync_loss     * inv, global_update)
                tb_writer.add_scalar("train/dpo_loss",    accum_dpo     * inv, global_update)
                tb_writer.add_scalar("train/anchor_loss", accum_anchor  * inv, global_update)
                tb_writer.add_scalar("train/play_wm_anchor", accum_play_wm * inv, global_update)
                tb_writer.add_scalar("train/margin_mean", accum_margin  * inv, global_update)
                tb_writer.add_scalar("train/accuracy",    accum_acc     * inv, global_update)
                tb_writer.add_scalar("train/lr",          lr,                  global_update)
                tb_writer.add_scalar("train/aligned_pos", out["aligned_pos_loss"].item(), global_update)
                tb_writer.add_scalar("train/aligned_neg", out["aligned_neg_loss"].item(), global_update)
                tb_writer.add_scalar("train/ref_pos",     out["ref_pos_loss"].item(),     global_update)
                tb_writer.add_scalar("train/ref_neg",     out["ref_neg_loss"].item(),     global_update)

                if global_update % cfg.print_every == 0:
                    print(
                        f"  [step {global_update}]"
                        f"  loss: {sync_loss   * inv:.4f}"
                        f"  dpo: {accum_dpo    * inv:.4f}"
                        f"  anch: {accum_anchor * inv:.4f}"
                        f"  pwm: {accum_play_wm * inv:.4f}"
                        f"  margin: {accum_margin * inv:+.3f}"
                        f"  acc: {accum_acc   * inv:.2f}"
                        f"  lr: {lr:.2e}"
                    )

                if global_update % cfg.save_every == 0:
                    save_lora_checkpoint(
                        ckpt_path=os.path.join(log_dir, f"adapter_{global_update}.pt"),
                        epoch=epoch, global_update=global_update,
                        model=aligned_denoiser, optim=optim, scheduler=scheduler,
                        rank=rank, wandb_run_id=wandb_run_id, log_dir=log_dir,
                    )

            accum_total = accum_dpo = accum_anchor = accum_play_wm = accum_margin = accum_acc = 0.0

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = epoch_loss_sum / num_updates if num_updates > 0 else 0.0

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss:        {avg_loss:.6f}")
        print(f"  Epoch Time:        {epoch_time:.2f}s")
        print(f"{'='*60}\n")

    return global_update, avg_loss, epoch_time


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@record
@hydra.main(config_path="config", config_name="align/pushT-fdpo", version_base=None)
def main(cfg: DictConfig):
    torch.backends.cuda.matmul.allow_tf32 = cfg.train.enable_fast_matmul

    rank, local_rank, world_size, device = setup_distributed()
    torch.manual_seed(cfg.seed + rank)

    if rank == 0:
        print(f"Distributed FDPO alignment: {world_size} GPU(s)")
        effective_batch = cfg.train.batch_per_gpu * world_size * cfg.train.accum_grad_steps
        print(f"Effective global batch size: {effective_batch} (pairs)")

    # --- Dataloader ---
    train_loader, train_sampler, train_dataset = create_distributed_pair_dataloader(
        data_dir=cfg.fdpo.pairs_dir,
        batch_size=cfg.train.batch_per_gpu,
        rank=rank, world_size=world_size,
        num_workers=cfg.train.num_workers,
        seed=cfg.seed,
        split="train",
        train_fraction=cfg.fdpo.train_fraction,
        split_seed=cfg.fdpo.split_seed,
        shuffle=True, drop_last=True,
    )
    if rank == 0:
        print(f"Train pairs: {len(train_dataset)}  |  steps/epoch: {len(train_loader)}")
        print(f"FDPO: β={cfg.fdpo.beta}  anchor_weight={cfg.fdpo.anchor_weight}  "
              f"force_mode={cfg.fdpo.force_mode}  "
              f"dpo_streams={cfg.fdpo.get('dpo_streams', 'both')}")

    # --- Optional play-data WM anchor loader ---
    # When cfg.fdpo.play_data_dir is set, a second sampler runs over the raw
    # play dataset and feeds a WM-mode anchor term each step (preserves the
    # model's world-model competence on the broad play distribution).
    play_loader = None
    play_sampler = None
    play_dataset = None
    if cfg.fdpo.get("play_data_dir", None):
        from dreamerv4uwm.datasets import create_distributed_dataloader
        # Window length: must match the pair window so the WM loss is comparable.
        # Pair-dataset metadata records ctx + horizon → just use that sum.
        pair_meta_path = Path(cfg.fdpo.pairs_dir) / "metadata.json"
        if pair_meta_path.exists():
            import json as _json
            _pm = _json.loads(pair_meta_path.read_text())
            play_window = int(_pm.get("ctx_frames", 32)) + int(_pm.get("horizon_frames", 32))
        else:
            play_window = cfg.denoiser.max_sequence_length
        play_bs = int(cfg.fdpo.get("play_batch_per_gpu", cfg.train.batch_per_gpu))
        play_loader, play_sampler, play_dataset = create_distributed_dataloader(
            data_dir=cfg.fdpo.play_data_dir,
            window_size=play_window,
            batch_size=play_bs,
            rank=rank, world_size=world_size,
            num_workers=max(1, cfg.train.num_workers // 2),
            stride=1, seed=cfg.seed + 7919,  # decorrelate from pair sampler
            split="train",
            train_fraction=cfg.fdpo.get("play_train_fraction", 0.9),
            split_seed=cfg.fdpo.get("play_split_seed", 42),
            shuffle=True, drop_last=True,
            absolute_actions=False,
        )
        if rank == 0:
            print(f"Play-WM anchor: src={cfg.fdpo.play_data_dir}  "
                  f"window={play_window}  batch_per_gpu={play_bs}  "
                  f"wm_anchor_weight={cfg.fdpo.get('wm_anchor_weight', 0.0)}  "
                  f"wm_anchor_force_mode={cfg.fdpo.get('wm_anchor_force_mode', 'wm')}  "
                  f"windows={len(play_dataset)}")

    # --- Models ---
    if rank == 0:
        print("Building models (frozen ref + LoRA aligned)...")
    tokenizer, ref_denoiser, aligned_denoiser, diffuser = build_models(
        cfg, device, local_rank, rank,
    )

    trainable_params = [p for p in aligned_denoiser.parameters() if p.requires_grad]
    if rank == 0:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in aligned_denoiser.parameters())
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

    # --- Resume ---
    wandb_run_id = cfg.wandb.run_name
    log_dir = None
    start_epoch = 0
    global_update = 0

    if cfg.reload_checkpoint is not None:
        if rank == 0:
            print(f"Resuming from FDPO LoRA checkpoint: {cfg.reload_checkpoint}")
        start_epoch, global_update, wandb_run_id, log_dir = load_lora_checkpoint(
            ckpt_path=cfg.reload_checkpoint,
            model=aligned_denoiser, optim=optim, scheduler=scheduler, rank=rank,
        )
    elif rank == 0:
        print("Starting FDPO LoRA from scratch.")

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

    # --- Training ---
    epoch_losses, epoch_times = [], []
    if rank == 0:
        print("Starting FDPO alignment...")

    # Build a cycling iterator over the play loader (advances independently of
    # the main pair loader; restarts when exhausted with a new sampler epoch).
    epoch_ref = [0]
    play_iter = (_cycling_iter(play_loader, sampler=play_sampler, epoch_ref=epoch_ref)
                 if play_loader is not None else None)

    for epoch in range(start_epoch, cfg.train.num_epochs):
        epoch_ref[0] = epoch
        global_update, avg_loss, epoch_time = train_epoch(
            epoch=epoch,
            train_loader=train_loader, train_sampler=train_sampler,
            tokenizer=tokenizer, ref_denoiser=ref_denoiser,
            aligned_denoiser=aligned_denoiser, diffuser=diffuser,
            optim=optim, scheduler=scheduler, tb_writer=tb_writer,
            cfg=cfg, rank=rank, device=device,
            global_update=global_update, log_dir=log_dir, wandb_run_id=wandb_run_id,
            trainable_params=trainable_params,
            play_iter=play_iter,
        )
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)

        if rank == 0:
            save_lora_checkpoint(
                ckpt_path=os.path.join(log_dir, f"adapter_{global_update}.pt"),
                epoch=epoch, global_update=global_update,
                model=aligned_denoiser, optim=optim, scheduler=scheduler,
                rank=rank, wandb_run_id=wandb_run_id, log_dir=log_dir,
            )
        dist.barrier()

    # --- Final merged save ---
    if cfg.lora.get("save_merged_final", True):
        merged_path = os.path.join(log_dir, "final_merged.pt")
        save_merged_final(aligned_denoiser, merged_path, rank)
    dist.barrier()

    if rank == 0:
        cur_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\n{'='*60}")
        print("FDPO Alignment Complete!")
        print(f"  Avg loss:      {sum(epoch_losses) / max(len(epoch_losses), 1):.6f}")
        print(f"  Avg epoch:     {sum(epoch_times) / max(len(epoch_times), 1):.2f}s")
        print(f"  GPU memory:    {cur_alloc:.2f} GB current / {peak_alloc:.2f} GB peak")
        print(f"{'='*60}")

        tb_writer.close()
        if cfg.wandb.enable:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
