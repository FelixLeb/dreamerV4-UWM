# Plan: AdaLN-Zero 3-class conditioning for dynamics alignment

## Context

The mode→dataset *routing* baseline (`scripts/train_align_mix.py`) did not align the
world model as hoped. New approach: instead of routing modes to datasets, **condition the
denoiser on a discrete provenance class** (`null` / `play` / `demo`) via an **AdaLN-Zero /
ControlNet-zero retrofit**, so that:

- **Day 0 is bit-identical to the pretrained model.** The retrofit zero-inits the
  per-layer modulation, so `norm(x)·(1+γ_c)+β_c = norm(x)` at init. Crucially we do **not**
  zero a residual gate (that would erase the pretrained attn/FFN); the pretrained residual
  branches are left untouched.
- **The base model stays loadable/usable.** Old code loading the new checkpoint with
  `strict=False` ignores the adaln keys and recovers the base net. With `freeze_backbone`
  the `null`/unconditioned path drifts only through its own zero-init adaln row.

Training then teaches the model what each provenance means: `play`→play data only,
`demo`→demo data only, `null`→both (50/50/step), each class sampled with prob 1/3. All
existing modes (`wm`/`policy`/`forcing`/…) still run within every class.

## Decisions (confirmed with user)

- Injection: **per-layer AdaLN-Zero** on *all* transformer layers (both RMSNorms).
- Scope: wire conditioning through **all 3 forward paths** (`forward`, `forward_step`,
  `forward_chunk_step`); **no CFG sampler** yet.
- `null` source mixing: **50/50 per step** (config knob `null_play_fraction`, default 0.5).
- Script: **new** `scripts/train_align_cond.py` (fork of `train_align_mix.py`); new configs.
- Freeze vs full-FT: a **script/config flag** `train.freeze_backbone` (default `true`,
  ControlNet-style), independent of the existing `lora.enabled` flag.

## Changes

### 1. `dreamerv4uwm/models/dynamics.py` — config + class embedder

- Add to `DreamerV4DenoiserCfg` (both default-off → existing ckpts/configs unaffected,
  `strict=True` still loads non-cond models):
  - `cond_adaln: bool = False`
  - `num_cond_classes: int = 3`
- In `DreamerV4Denoiser.__init__`, when `cfg.cond_adaln`: register
  `self.class_embedder = DiscreteEmbedder(cfg.num_cond_classes, cfg.model_dim)` (reuse the
  existing `DiscreteEmbedder`, `dynamics.py:188`; normal-init is fine — zero-init lives in
  the per-layer projection). Else `self.class_embedder = None`.
- Pass `cond_adaln=cfg.cond_adaln` into each `EfficientTransformerBlock`.
- In all three forwards (`forward`, `forward_step`, `forward_chunk_step`) add
  `cond_class: Optional[torch.Tensor] = None` (shape `(B,)` long). Compute
  `cond_emb = self.class_embedder(cond_class)` (→ `(B, D)`) when enabled & provided, else
  `None`, and thread it into every `layer(...)` / `layer.forward_step/forward_chunk_step`
  call. When `None`, behavior is byte-identical to today.

### 2. `dreamerv4uwm/models/blocks.py` — per-layer modulation

- `EfficientTransformerLayer.__init__`: add `cond_adaln: bool = False`. When set, register
  `self.adaln = nn.Linear(model_dim, 4*model_dim)` with **weight and bias zero-init**
  (γ1,β1 for `norm1`, γ2,β2 for `norm2`). Else `self.adaln = None`.
- Add a small helper `_modulate(h, shift, scale)` → `h*(1+scale[:,None,None,:])+shift[...]`.
- In `forward`, `forward_chunk_step` (and the temporal call inside the block's
  `forward_step`, which routes through `layer.forward`): accept `cond_emb=None`; when
  `self.adaln is not None and cond_emb is not None`, compute
  `s1,sc1,s2,sc2 = self.adaln(F.silu(cond_emb)).chunk(4, -1)` once, then modulate the
  `norm1` output (pre-attn) and the `norm2` output (pre-ffn). **No residual gate** — the
  `x = x + dropout(...)` residuals stay exactly as pretrained.
- `EfficientTransformerBlock`: build inner layers with `cond_adaln=cond_adaln`; thread
  `cond_emb` through `forward`, `forward_step`, `forward_chunk_step` to each layer.

Param cost note: ~`4·D·D` per layer; for pushT (`D=768`, 24 layers) ≈ **57M** extra
trainable params (zero-init). Acceptable for fine-tuning; flagged for awareness.

### 3. `dreamerv4uwm/models/dynamics.py` — `DenoiserWrapper`

- Add `cond_class=None` to `forward`/`forward_step`/`forward_chunk_step` and pass through.
  (All existing callers — samplers in `sampling*.py`/`inference/`, other train scripts —
  keep working unchanged because the arg defaults to `None`.)

### 4. `dreamerv4uwm/loss.py` — thread condition into the loss

- `compute_uwm_loss(...)`: add `cond_class: Optional[torch.Tensor] = None`, forward it into
  the `denoiser(...)` call (`loss.py:423`). Default `None` → `train_align_mix.py` and any
  other caller are unaffected.

### 5. `scripts/train_align_cond.py` — new training script (fork of `train_align_mix.py`)

Reuse the fork wholesale; replace the routing logic:

- Drop `MODE_SOURCE_RULE`. Keep mode sampling from `cfg.train.mode_weights`.
- New `CLASSES = ['null', 'play', 'demo']` (index 0/1/2; **null = 0** = unconditioned).
- New `sample_mode_and_class(cfg, rank, device, rng)`: on rank 0 sample `mode` from weights,
  sample `class` uniformly over the 3 (equal prob), derive `source`:
  `play→play`, `demo→demo`, `null→ play if rand()<null_play_fraction else demo`.
  Broadcast `(mode_idx, class_idx, src_idx)` to all ranks (same pattern as
  `sample_mode_and_source`, `train_align_mix.py:162`).
- In the step loop: pull batch from `play_iter`/`demo_iter` by `source`, build
  `cond_class = torch.full((B,), class_idx, device=device, dtype=torch.long)`, and call
  `compute_uwm_loss(info, denoiser, ..., cond_class=cond_class)`.
- `build_models`: pass `strict=False` to `load_denoiser` (base ckpt lacks adaln keys);
  rely on `cfg.denoiser.cond_adaln=true` to register the modulation. After build (pre-DDP),
  if `cfg.train.freeze_backbone`: set `requires_grad=True` only for params whose name
  contains `adaln` or `class_embedder`, `False` elsewhere; `trainable_params` becomes that
  set. LoRA stays an independent flag (when both on, add `adaln`/`class_embedder` to PEFT
  `modules_to_save` so they remain trainable alongside LoRA). DDP
  `find_unused_parameters=False` is safe (every adaln + the embedder Parameter receives grad
  each step).
- Logging: swap the mode/source mix traces for **class mix** + keep mode mix, and emit
  per-`(mode, class)` loss curves (same sparse style as the existing `by_ms/*` block).

### 6. Configs — `scripts/config/align/pushT-cond.yaml` (new, fork of `pushT-mix.yaml`)

- `denoiser.cond_adaln: true`, `denoiser.num_cond_classes: 3`.
- `train.freeze_backbone: true`, `train.null_play_fraction: 0.5`.
- Keep `dynamics_ckpt` (pretrained), `mode_weights`, dataset play/demo dirs. Drop the now
  unused `wmid_play_fraction` (harmless if left).
- (A `g1-cond.yaml` can follow the same recipe on request.)

### 7. Guardrails — fail loud, never silently fall back

Every place the new feature could silently degenerate to base/unconditioned behavior emits
a **rank-0 warning (or hard assert)**. No silent retractions to the old pipeline.

- **Checkpoint missing conditioning params** (config says `cond_adaln=true`, ckpt lacks the
  keys): do the load explicitly so the `IncompatibleKeys` is inspectable — either capture
  `incompat = model.load_state_dict(sd, strict=False)` in `build_models`, or extend
  `load_denoiser` to return the keys. Then:
  - `missing_keys` that are **all** `adaln`/`class_embedder` → INFO: "N conditioning params
    not in checkpoint → zero-initialized (expected when conditioning a base checkpoint)."
  - `missing_keys` containing **non-conditioning** names → loud WARNING (real mismatch).
  - any `unexpected_keys` (ckpt has params the model lacks) → loud WARNING (architecture
    drift, e.g. wrong base ckpt).
- **Conditioning enabled but never exercised**: if `cfg.denoiser.cond_adaln` is true but
  `cond_class` is `None` at the loss/forward call → assert in the training step (the
  modulation would be a silent no-op).
- **Script expects conditioning but config disabled it**: in `train_align_cond.py`, assert
  `cfg.denoiser.get('cond_adaln', False)` is true — otherwise the cond script would silently
  run as plain mixed training. Loud message naming the flag to set.
- **Freeze flag selected nothing**: if `freeze_backbone=true` but zero params match
  `adaln`/`class_embedder` (name drift) → assert (would otherwise train nothing, or with a
  logic slip, everything). Always print the resolved trainable param count + names summary.
- **num_cond_classes mismatch on resume**: if a resumed `class_embedder` row count differs
  from `cfg.denoiser.num_cond_classes` → loud WARNING/assert.
- Reuse this same pattern for any other base-pipeline fallbacks touched here (e.g. cond
  disabled paths): surface them, don't swallow them.

### 8. Store the plan in the repo

Copy this plan to the project root as `dreamerV4-UWM/plan_adaln_cond_conditioning.md`
(alongside the existing untracked `arch.md`) as part of implementation, so the design lives
with the code. (Untracked unless you choose to commit it.)

## Backward-compatibility guarantees

- `cond_adaln=false` default ⇒ no new modules registered ⇒ existing checkpoints load with
  `strict=True`, and `forward*` paths are byte-identical (cond_emb=None branch).
- With `cond_adaln=true`, day-0 output == pretrained (zero-init γ/β, ungated residuals).
- New checkpoints remain loadable by old code via `strict=False` (adaln keys ignored).

## Verification

1. **Init bit-equivalence (offline, 1 GPU in container — see `run-python-in-container`
   memory):** build the denoiser with `cond_adaln=true`, load base ckpt `strict=False`; run
   one batch through `forward(...)` with `cond_class=None` vs `cond_class=zeros` vs a base
   model with `cond_adaln=false` — assert all three outputs are equal (allclose, bf16 tol).
2. **Forward-path parity:** confirm `forward_step`/`forward_chunk_step` accept `cond_class`
   and, at zero-init, match `forward` on a short rollout (existing sampler smoke path).
3. **Smoke train:** launch `train_align_cond.py` on pushT for ~20 steps (tiny
   `num_training_steps`), `freeze_backbone=true`: check it runs under DDP+compile, that only
   adaln+class_embedder appear in `trainable_params` (the printed trainable % is small), and
   that per-class loss curves populate for all 3 classes.
4. **Full run:** standard SLURM launch; compare `null`-class world-model competence (should
   track base) vs `demo`-class task behavior in the experiment log (`docs/experiments.md`).
