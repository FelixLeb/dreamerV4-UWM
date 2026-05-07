# Reward head + agent token — implementation plan

This is a hand-off doc for adding a Dreamer-style reward / multi-token-prediction (MTP) head to the UWM denoiser. It is self-contained: a fresh Claude session reading only this file (plus the repo) should know **what** to build, **why**, **where to put it**, and **how to verify** it doesn't regress existing behavior.

Status when this doc was written: **design locked, no code written.** A previous session prototyped `SymlogTwoHotHead` and `RewardMTPHead` and a partial `datasets.py` edit; those changes were reverted. Treat this doc as the authoritative starting point.

---

## 1. What this project is

`dreamerV4-UWM` extends Dreamer-V4 with a second independent flow-matching channel for actions. The single denoiser exposes **independent per-frame noise levels for state and action (τ_state, τ_action)**, so inference-time choices select between three roles:

1. **World model** — clean context + clean actions → rollout noisy states.
2. **Policy** — clean context → joint generation of (state, action).
3. **Random action proposer** — *noisy* context → joint (state, action), high entropy, used by downstream planners.

This is research code. The CLAUDE.md ground rules apply:
- Propose, don't refactor.
- New experiments are cloned slurm scripts.
- Always answer "what does this change do to modes 1 and 2?" before claiming success.

## 2. Goal of this work item

Add a **reward / value-prediction capability** to the denoiser **without disturbing the joint state-action denoising objective**. The reference implementation in `/scratch/rk4342/projects/tmp/dreamer-v4/{models.py, go2_train_dreamer_dynamics_ddp.py}` shows a clean way to do this:

- Append a single learned **agent token** to each frame.
- Make the agent token **read-only**: it can attend to all "world" tokens (latents, registers, controls, action), but no world token can attend to it.
- Read out per-timestep reward predictions from that agent token via an **MTP head** that predicts L=8 future-step rewards from each timestep, classified into 255 symlog buckets (two-hot CE).

Because the agent token is structurally isolated, reward gradients cannot corrupt dynamics learning. A pretrained dynamics-only checkpoint can be loaded with `strict=False` and finetuned with this head turned on.

## 3. Reference files (read these first)

- `/scratch/rk4342/projects/tmp/dreamer-v4/models.py`
  - `SymlogTwoHotHead` — lines 10–60.
  - `RewardMTPHead` — lines 62–86.
  - Denoiser with `train_reward_model` flag, `agent_token`, spatial mask, forward path — around lines 1000–1190. Pay attention to where the mask is applied (spatial layers only) and how outputs are sliced (`world_x = x[:, :, :-1, :]`, `agent_x = x[:, :, -1, :]`).
- `/scratch/rk4342/projects/tmp/dreamer-v4/go2_train_dreamer_dynamics_ddp.py`
  - The training-side glue: how rewards are unfolded into MTP targets, how validity masks handle episode ends, how the reward CE is reduced and added to the dynamics loss, how `RMSLossScaler` is used per-loss-term.

Don't blindly copy the reference — its denoiser has a different token layout (no IC/AC controls, no separate state/action τ). The patterns to lift are the **head classes**, the **mask polarity**, and the **MTP loss math**.

## 4. Current code: where things live (verify before editing)

All paths are relative to repo root.

- `dreamerv4uwm/models/dynamics.py`
  - `build_spatial_attention_mask(...)` at **line 11**. Today it covers groups `[Z, Reg, IC, AC, A]` — needs to optionally accept an Agent group.
  - `DreamerV4DenoiserCfg` dataclass at **line 110** — extend with two fields.
  - `DreamerV4Denoiser.__init__` at **line 139** — needs the conditional `agent_token` parameter and `reward_head` module, plus a stored `agent_spatial_mask` buffer.
  - `dynamics_spatial_mask` is **built at line 192** and registered at line 200 — but at **line 252** the spatial layer pass uses `spatial_mask=None`. This means the existing mask is built-but-unused, so flipping it on globally would silently change baselines. The reward-head mask must therefore be gated on a flag.
  - `forward` runs lines 202–259. It concatenates `[obs_tokens, reg_tokens, obs_diff_control, act_diff_control, act_tokens]` along the modality axis at line 244. The agent token would be appended after `act_tokens`.
  - **Skip `forward_step`** (line 261). It references `self.diffusion_embedder` (line 273), which no longer exists on the new dual-stream model — it's already broken. Don't touch it as part of this work item.

- `dreamerv4uwm/loss.py`
  - `RMSLossScaler` at **line 38** — already supports per-name running RMS. Reuse it; just add a new key for reward.
  - `compute_uwm_loss(info, denoiser, device, loss_weighting)` at **line 249** — currently returns `(obs_flow_loss, act_flow_loss)`. Needs to grow a `rewards=None` kwarg and return a dict so we don't break call sites or fight Python tuple-arity changes.

- `dreamerv4uwm/datasets.py`
  - `ShardedHDF5Dataset.__getitem__` at **line 102**. Today returns `{'image': images, 'action': actions}`. Needs to optionally read a reward field. Detect once per dataset (e.g., probe shard 0 in `__init__`).
  - There's a commented-out `is_demo` prototype at **lines 760–786**. Useful only as a reminder that an earlier draft considered this.

- `scripts/train_dynamics_uwm.py`
  - Loss is called at **line 319**: `obs_flow_loss, act_flow_loss = compute_uwm_loss(...)`.
  - Sum into `loss_micro` at lines 323–325. Accumulators at 329–331.
  - The accumulation loop at lines 285–301 has `short` / `image` / `long` branches. Reward training only makes sense on the `long` branch (the others crop or reshape time). Either skip reward loss on non-long branches, or just zero it out when `'reward'` isn't in the batch.

- `scripts/train_dynamics_uwm_with_shortcut.py` — same shape as the non-shortcut script; mirror the changes there. The shortcut path still uses the same denoiser, so reward outputs flow through it the same way.

- `scripts/config/dynamics/{pushT,pushT-large,lewm-pushT,lewm-cubes,finger}.yaml`
  - Each has a `denoiser:` block. Add `train_reward_model: false` and `mtp_length: 8` defaults so existing configs keep their current behavior.

## 5. Locked design decisions — do **not** re-litigate

1. **Read-only agent token, polarity is `mask[:-1, -1] = -inf`.** Agent reads world; world cannot read agent. (Don't flip the polarity "for symmetry" — that defeats the isolation.)
2. **Spatial-only masking.** Apply the agent mask only on spatial layers. Temporal layers attend along time per-token-position, so they don't need it. This matches the reference.
3. **Conditional mask gate.** The mask is applied **only when `train_reward_model=True`**. With the flag off, baselines are bit-equivalent to current main (`spatial_mask=None`). This is non-negotiable — it's how we keep modes 1 and 2 from silently regressing.
4. **MTP head architecture.**
   - L = 8 future-reward predictions per timestep (i.e. predicts `r_t, r_{t+1}, ..., r_{t+7}` from each frame).
   - Shared MLP trunk: `Linear(d_model, hidden=512) → SiLU`.
   - L independent `SymlogTwoHotHead`s on top of the trunk. They do **not** share output weights.
   - Each head: 255 buckets, range [-20, 20] in symlog space.
5. **Two-hot edge case.** When the target lands exactly on a bucket index, naive `low_w = high_w = 0`. Force `low_w = 1.0, high_w = 0.0` whenever `low == high` (also covers the post-clamp out-of-range case). Easy to miss when porting.
6. **Loss scaling.** Reuse `RMSLossScaler` in `dreamerv4uwm/loss.py:38`. The reward term gets its own key (e.g. `"reward"`) so its running RMS is tracked separately from `obs_flow` / `act_flow`.
7. **Always-on when enabled.** The reward objective is **not** gated by the `policy/wm/id/video/forcing` mode mix. Because the agent token is structurally isolated, the same reward target is valid in any mode. (One nuance: in `video` and `forcing` modes the actions/state going in are noisy — the agent token still observes that noisy world and is asked to predict the clean reward. That's by design; it teaches reward inference under ambiguity.)
8. **Default off.** New config knobs default to `train_reward_model: false`. No baseline run changes shape unless someone explicitly opts in.

## 6. Implementation plan, file by file

### 6.1 `dreamerv4uwm/models/dynamics.py`

#### a) Add `SymlogTwoHotHead` and `RewardMTPHead` near the top of the file.

Put them above `build_spatial_attention_mask`. Use these implementations verbatim (they include the edge-case fix and have been reviewed):

```python
class SymlogTwoHotHead(nn.Module):
    """Symlog-space two-hot classification head over a fixed bucket grid.

    Predicts a scalar via classification: logits over `num_buckets` evenly
    spaced in symlog space across [min_val, max_val]. `get_targets` returns
    the (low_idx, low_weight, high_idx, high_weight) tuple for the two-hot
    CE target.
    """

    def __init__(self, input_dim: int, num_buckets: int = 255,
                 min_val: float = -20.0, max_val: float = 20.0):
        super().__init__()
        self.num_buckets = num_buckets
        self.min_val = min_val
        self.max_val = max_val
        self.linear = nn.Linear(input_dim, num_buckets)
        buckets = torch.linspace(min_val, max_val, num_buckets)
        self.register_buffer("buckets", buckets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @staticmethod
    def to_symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    @staticmethod
    def from_symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

    def get_targets(self, rewards: torch.Tensor):
        """Build two-hot targets from raw rewards.

        Out-of-range rewards are clamped to the edge buckets — the head
        cannot represent values beyond [min_val, max_val] in symlog space.

        Edge case: when the target lands exactly on a bucket index, naive
        `high - low` weights collapse to 0/0. Force all mass onto `low` in
        that case (also covers the post-clamp case where low == high).

        Returns:
            (low, low_w, high, high_w) — same shape as `rewards`.
        """
        y = self.to_symlog(rewards)
        width = (self.max_val - self.min_val) / (self.num_buckets - 1)
        indices = (y - self.min_val) / width
        indices = indices.clamp(0, self.num_buckets - 1)

        low = indices.floor().long()
        high = indices.ceil().long()
        low_weight = high.float() - indices
        high_weight = indices - low.float()

        mask = (low == high)
        low_weight[mask] = 1.0
        high_weight[mask] = 0.0

        return low, low_weight, high, high_weight


class RewardMTPHead(nn.Module):
    """L parallel reward predictions per timestep, shared trunk, separate output heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 mtp_length: int = 8, num_buckets: int = 255):
        super().__init__()
        self.mtp_length = mtp_length
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([
            SymlogTwoHotHead(hidden_dim, num_buckets) for _ in range(mtp_length)
        ])

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """h_t: (B, T, input_dim) → logits (B, T, L, num_buckets)."""
        x = self.hidden(h_t)
        outputs = [head(x) for head in self.heads]
        return torch.stack(outputs, dim=2)
```

#### b) Extend `build_spatial_attention_mask` to take an optional Agent group.

Today's signature:

```python
def build_spatial_attention_mask(n_latent, n_register, n_image_control,
                                 n_action_control, n_action,
                                 latent_attends_action=False) -> torch.Tensor
```

Add `n_agent: int = 0` as a final positional/keyword arg. When `n_agent > 0`:

- Extend `n_total` by `n_agent`.
- Append an `agent_start, agent_end` slice after the action group.
- Allow agent rows to attend to **everything** (`Z, Reg, IC, AC, A, Agent`).
- Block the agent **column** for all non-agent rows. The cleanest expression: build the mask exactly as today (without the agent group), then for each row `r` in `[0, n_total - n_agent)` set `mask_bool[r, agent_start:agent_end] = False`. (Or equivalently in the float mask, set those cells to `-inf`.)
- Don't change behavior for `n_agent=0` callers — same mask as today.

Verify the bool→float conversion at the bottom of the function still produces `-inf` where blocked. (Today's code uses `mask_float[mask_bool] = 0.0` and presumably `-inf` elsewhere — confirm by reading lines 83–end of the function before editing.)

#### c) Extend `DreamerV4DenoiserCfg`

Add two optional fields with off-by-default values:

```python
train_reward_model: bool = False
mtp_length: int = 8
```

Place them at the bottom of the dataclass alongside other behavior flags.

#### d) Wire the agent token into `DreamerV4Denoiser.__init__`

After `self.action_input_proj` is built, add:

```python
if cfg.train_reward_model:
    self.agent_token = nn.Parameter(torch.randn(1, 1, 1, cfg.model_dim) * 0.02)
    self.reward_head = RewardMTPHead(input_dim=cfg.model_dim,
                                     mtp_length=cfg.mtp_length)
else:
    self.register_parameter('agent_token', None)
    self.reward_head = None
```

(Shape `(1, 1, 1, D)` is intentional — broadcasts cleanly to `(B, T, 1, D)` at forward time. Match the reference's init scale of `0.02`.)

When building the spatial mask (line 192), pass `n_agent = 1 if cfg.train_reward_model else 0`. **Update `self.num_modality_tokens`** (line 153) to include the agent token in the same conditional. The transformer block's `modality_dim_max_seq_len` depends on it, so getting this right is what makes the new token actually fit through the layers.

Also bump the `num_modality_tokens` count consistently — search the whole file for any other site that hard-codes the modality count.

#### e) Update `forward`

Right before the transformer loop (around line 244–250):

1. After concatenating the existing `[obs_tokens, reg_tokens, obs_diff_control, act_diff_control, act_tokens]`, append the agent token if `cfg.train_reward_model`:

   ```python
   if self.cfg.train_reward_model:
       agent_part = self.agent_token.expand(B, T, -1, -1)  # (B, T, 1, D)
       x = torch.cat([x, agent_part], dim=-2)
   ```

2. Inside the `for layer in self.layers` loop, branch by layer type:
   - On **spatial** layers, when `cfg.train_reward_model=True`, pass the new mask. When the flag is off, keep `spatial_mask=None` exactly as today.
   - On **temporal** layers, always `spatial_mask=None`.

   The current implementation just calls `layer(x, spatial_mask=None)`. The cleanest gate is to read the `LayerType` from `self.layer_types[i]`. Verify how `EfficientTransformerBlock` exposes its layer kind — if it's stored on `layer.layer_type` you can branch on that; otherwise iterate over `enumerate(zip(self.layer_types, self.layers))`.

3. After the transformer, slice outputs:
   - When `cfg.train_reward_model`:
     ```python
     world_x = x[:, :, :-1, :]                      # drop agent column
     agent_x = x[:, :, -1, :]                       # (B, T, D)
     obs_output = self.obs_projector(world_x[:, :, :self.cfg.num_latent_tokens, :])
     act_output = self.action_projector(world_x[:, :, -self.cfg.num_action_tokens:, :])
     pred_rewards = self.reward_head(agent_x)       # (B, T, L, num_buckets)
     return obs_output, act_output, pred_rewards
     ```
   - Otherwise return `(obs_output, act_output)` as today (or `(obs_output, act_output, None)` — see the next paragraph).

   **Return-shape decision:** to avoid touching every call site, prefer making the third return value optional via `None`. So always return a 3-tuple `(obs_output, act_output, pred_rewards_or_None)` and update *every* current unpack site. There aren't many — `compute_uwm_loss` and any tests/scripts.

   Search the codebase for `denoiser(` calls and `model(` calls to confirm the inventory of unpack sites.

#### f) Skip `forward_step`

It's already broken (references `self.diffusion_embedder`, which doesn't exist after the dual-stream split). Don't add reward support to it as part of this work item. Add a one-line comment noting this.

### 6.2 `dreamerv4uwm/loss.py`

#### a) Extend `compute_uwm_loss` signature

Add `rewards: torch.Tensor | None = None` after `loss_weighting`. Document that:
- Shape `(B, T)` if present.
- Required when the denoiser was built with `train_reward_model=True`. Raise a clear `RuntimeError` if `rewards is None` but `denoiser.cfg.train_reward_model is True`.

#### b) Compute the reward MTP loss

After the existing `obs_flow_loss / act_flow_loss` block, when `denoiser.cfg.train_reward_model` is true:

1. **Unpack the third denoiser return** (`pred_rewards`, shape `(B, T, L, K)` where K = num_buckets, L = mtp_length).
2. **Build per-(t, l) targets** by rolling the reward window:
   ```python
   L = denoiser.cfg.mtp_length
   B, T = rewards.shape
   # We want target_{t, l} = rewards[t + l] for l in [0, L)
   # Pad with zeros at the right so out-of-episode positions are handled by the validity mask.
   padded = F.pad(rewards, (0, L - 1))                 # (B, T + L - 1)
   targets = padded.unfold(dimension=1, size=L, step=1)[:, :T]  # (B, T, L)
   ```
3. **Validity mask:** any `(t, l)` with `t + l >= T` falls off the window — mask it out.
   ```python
   t_idx = torch.arange(T, device=rewards.device).view(1, T, 1)
   l_idx = torch.arange(L, device=rewards.device).view(1, 1, L)
   valid = (t_idx + l_idx) < T   # (1, T, L) bool
   ```
4. **Two-hot CE.** Get bucket indices and weights from any one of the heads' `get_targets` (they share the bucket grid):
   ```python
   low, low_w, high, high_w = denoiser.reward_head.heads[0].get_targets(targets)
   logp = F.log_softmax(pred_rewards.float(), dim=-1)   # cast to fp32 — bf16 underflows
   nll = -(low_w * logp.gather(-1, low.unsqueeze(-1)).squeeze(-1)
         + high_w * logp.gather(-1, high.unsqueeze(-1)).squeeze(-1))
   reward_loss = (nll * valid).sum() / valid.sum().clamp_min(1)
   ```
5. **Scaling:** wrap the existing flow losses + this new term with `RMSLossScaler` so they coexist without one swamping the others. Or, simpler: return them unscaled from `compute_uwm_loss` and let the training script apply scaling after summing — easier to read in TB.

#### c) Return shape

Change return to a dict for forward-compatibility:

```python
return {
    "obs_flow_loss": obs_flow_loss,
    "act_flow_loss": act_flow_loss,
    "reward_loss": reward_loss if denoiser.cfg.train_reward_model else None,
}
```

Update both call sites in `scripts/train_dynamics_uwm.py:319` and `scripts/train_dynamics_uwm_with_shortcut.py` to unpack the dict.

Also check `compute_bootstrap_uwm_loss` (line 574 in loss.py): it likely calls `compute_uwm_loss` internally or duplicates its math. If it uses the denoiser directly, mirror the same return contract.

### 6.3 `dreamerv4uwm/datasets.py`

In `ShardedHDF5Dataset.__init__`, probe shard 0 once for the reward field name:

```python
with h5py.File(self.shard_files[0], 'r') as f:
    self.has_rewards = 'rewards' in f
```

In `__getitem__` (line 102), conditionally read the reward window inside the `with h5py.File(...)` block:

```python
rewards = f['rewards'][ep_idx, start:end] if self.has_rewards else None
```

Append to the return dict:

```python
out = {'image': images, 'action': actions}
if rewards is not None:
    out['reward'] = torch.from_numpy(rewards).float()
return out
```

**Pre-check before any production run.** The exact reward field name and shape varies across datasets. Inspect a shard from each dataset before relying on `'rewards'`:

```bash
python -c "import h5py; f=h5py.File('<path>/shard_0000.h5','r'); print(list(f.keys()), {k: f[k].shape for k in f.keys()})"
```

Datasets to check: `pushT`, `lewm-cubes`, `lewm-pushT`, `finger`, `soar` (whichever are live in this branch). The reference uses `is_demo` as a binary surrogate for reward in environments without a dense reward — be ready for that to be the field name instead.

### 6.4 `scripts/train_dynamics_uwm.py` and `scripts/train_dynamics_uwm_with_shortcut.py`

In the per-micro-batch forward block (around line 314–325 of `train_dynamics_uwm.py`):

1. Pull `rewards = batch.get('reward', None)` — note the dataset uses key `'reward'` (singular) per §6.3.
2. Pass it into the loss call: `compute_uwm_loss(diffused_info, denoiser, ..., rewards=rewards)`.
3. Unpack the dict return.
4. Sum into `loss_micro`:
   ```python
   total = losses["obs_flow_loss"] + losses["act_flow_loss"]
   if losses["reward_loss"] is not None:
       total = total + reward_weight * losses["reward_loss"]
   loss_micro = total / cfg.train.accum_grad_steps
   ```
5. Add `accum_reward_loss` accumulator (mirror `accum_obs_flow / accum_act_flow` at lines 329–331).
6. Log it on tensorboard: `tb_writer.add_scalar("train/reward_loss", ..., global_update)`.

Mirror all of the above in `scripts/train_dynamics_uwm_with_shortcut.py`. The shortcut script has the same structure; the bootstrap loss path doesn't change anything about reward — reward is computed once per forward, like the dynamics flow loss.

Add a config-driven `reward_weight` (e.g. `train.reward_weight: 1.0`) so it's tunable without code changes.

**Branch-mix nuance:** the training loop has `short` / `image` / `long` accumulation branches (lines 285–301). Reward MTP needs a real time axis to roll the window over, so it's only meaningful on `long` (and arguably `short` if `ctx_len >= L+1`). The simplest correct behavior is: only compute reward loss when `'reward'` is in the batch (datasets without rewards just don't return the key) **and** when `T >= L`. In the `image` branch where `T=1`, skip. Document this in the loss function with an early `return reward_loss = 0` when `T < L`.

### 6.5 Configs

For each `scripts/config/dynamics/*.yaml`, add to the `denoiser:` block:

```yaml
denoiser:
  ...existing fields...
  train_reward_model: false
  mtp_length: 8
```

And to the `train:` block:

```yaml
train:
  ...existing fields...
  reward_weight: 1.0    # ignored when denoiser.train_reward_model=false
```

Defaults stay off — backward-compatible by construction.

## 7. Validation plan (run *all* of these before claiming success)

These map onto CLAUDE.md's explicit "what does this do to modes 1 and 2?" requirement.

### 7.1 Bit-equivalence check (with flag off)

Goal: prove that `train_reward_model=False` produces a denoiser numerically identical to current main.

```python
import torch
from dreamerv4uwm.models.dynamics import DreamerV4Denoiser, DreamerV4DenoiserCfg

torch.manual_seed(0)
cfg = DreamerV4DenoiserCfg(... train_reward_model=False ...)
m = DreamerV4Denoiser(cfg).eval()
# random inputs
out_new = m(noisy_act, noisy_obs, ...)

# Switch to a checkout of main without these changes, build the same cfg
# (with no `train_reward_model` field needed), run the same inputs.
# out_main and out_new must match exactly (allclose with atol=0).
```

If they don't match, the conditional gating in §6.1.d/e is wrong. Likely culprit: `num_modality_tokens` was incremented unconditionally, which changes the transformer's positional setup.

### 7.2 Mode-1/2 regression run

Goal: prove that turning the flag on (with reward weight=0) doesn't measurably hurt world-model or policy losses.

- Clone the closest existing slurm — pick a `pushT-uwm-policy-wm` baseline from `hpc/slurms/dynamics/pushT/no-shortcut/`.
- Override:
  - `denoiser.train_reward_model=true`
  - `denoiser.mtp_length=8`
  - `train.reward_weight=0.0`
- Run for the same step count as the baseline.
- Compare `train/obs_flow_loss` and `train/act_flow_loss` curves to the baseline. They must track within run-to-run noise. Eyeball a few thousand updates, not the whole run.

If they diverge, the agent token is bleeding gradients into the world stream — most likely the spatial mask polarity or its application is wrong. Re-check §5.1 and §6.1.b/e.

### 7.3 Reward-on smoke test

Goal: confirm reward loss decreases on a dataset that has rewards.

- Clone the same slurm, set `train.reward_weight=1.0`, find a dataset with a `'rewards'` HDF5 field (verify with the one-liner in §6.3).
- Watch `train/reward_loss` for the first ~500 steps. It should drop from `~ln(255) ≈ 5.54` (uniform-bucket prior) toward something noticeably smaller. If it sits at the prior, the bucket math or two-hot weighting is wrong.

### 7.4 Hard-error check

Goal: don't silently train a zero-target reward head.

- Ensure that with `train_reward_model=True` and a dataset whose shards do **not** contain a reward field, training raises a clear error early (before the first optimizer step). The check belongs in `compute_uwm_loss` (raise `RuntimeError` when `rewards is None` but the flag is on).

## 8. Pitfalls (anticipate these — they cost hours each)

- **dtype underflow.** Two-hot CE on bf16 logits with one-bucket-wide targets underflows fast. Cast `pred_rewards.float()` before `log_softmax`. Don't autocast the reward branch.
- **Mask polarity.** Test it with an explicit unit check: build the mask, manually verify `mask[i, -1] == -inf` for `i < n_total - 1` and `mask[-1, j] == 0` for all `j`.
- **Mask dtype.** SDPA kernels (memory-efficient attention) need the mask cast to the activation dtype. The reference does `mask.to(dtype=x.dtype)` at every spatial layer call. Mirror this.
- **DDP unused-parameter error.** When `train_reward_model=False`, `agent_token` is `None` and `reward_head` is `None` — no issue. When it's `True` but `reward_weight=0.0`, the reward branch produces grads that get scaled to zero; depending on how the training script implements zero-weight, you may need `find_unused_parameters=True` or to skip the reward forward entirely when `reward_weight==0`. Prefer the latter — cheaper and avoids the DDP gotcha.
- **`compile` + new params.** If `torch.compile` is on (it is for dynamics — see `cfg.train.use_compile=true`), recompile triggers can fire when the parameter set changes. The `_orig_mod` unwrap in `load_ddp_checkpoint` (see CLAUDE.md gotchas) only restores model weights — it doesn't recreate optimizer state for new params. A reward-head finetune from a dynamics-only checkpoint will start the reward head from random init, which is fine, but the optimizer's per-param state for the new params starts fresh too.
- **`num_modality_tokens` mismatch.** If you forget to bump it, you'll see a crash inside the transformer block on the first forward — `modality_dim_max_seq_len` is the sequence-length setting for spatial attention. Conditional gating means it should be bumped iff `train_reward_model=True`.
- **Episode boundaries inside a window.** The unfold-and-mask approach in §6.2.b assumes rewards are dense and intra-episode. Sharded HDF5 windows are sampled within a single episode (see `windows` index in `datasets.py`), so that's fine — but if you ever start sampling cross-episode windows, the validity mask needs to also zero out positions that fall past the episode's end inside the window. Out of scope for this work item.

## 9. Suggested implementation order

1. Heads only (§6.1.a). Run a tiny standalone script that instantiates `RewardMTPHead`, runs `forward` on a `(2, 4, 768)` tensor, calls `get_targets` on a `(2, 4, 8)` random reward tensor. Verify shapes and that `low_w + high_w == 1` everywhere.
2. Mask extension (§6.1.b) plus a unit-style check at the bottom of the file under `if __name__ == '__main__':` — verify `n_agent=0` matches today exactly, and `n_agent=1` blocks the right column.
3. Denoiser plumbing (§6.1.c–e). Run the bit-equivalence test (§7.1) immediately.
4. Datasets + loss + train-script wiring (§6.2–6.4). Unit-test the loss math with a hand-computed example.
5. Configs (§6.5). Smoke-run §7.2 and §7.3.

## 10. What's *not* in scope (don't get pulled in)

- Inference-time reward sampling / planning. The head is trained here; a planner/sampler that uses it is a separate work item.
- `forward_step` rewrite. It's already broken on the dual-stream branch.
- New experiment design for "what reward signal to use." That's a research conversation with Roohollah, not an implementation choice.
- Refactoring the existing modality-token bookkeeping in `DreamerV4Denoiser`. The current code paths work; new code threads through them, doesn't reshape them.

When in doubt, propose the change and ask before writing it.
