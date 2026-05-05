# UWM experiment log

Running research log for the unified world model. One section per experiment; newest at top within each part. Source of truth for "what has been tried and what happened."

Log format per run: **Hypothesis → Setup → Result → Takeaway.** Keep it terse. If a hypothesis is falsified or superseded, label it rather than deleting — the chain of reasoning is valuable.

---

## Open questions / active hypotheses

- **H1 (active).** Can training with the `video` mode (unconditioned video generation) fix the `action_sampler` regression observed in the best `policy+wm` recipe? Core idea: generating consistent rollouts from a fully-noisy context is the extreme case of the random-action-proposer regime; learning it should transfer.
- **H1a (active).** Within video-mode training, does **progressive noising** (per-frame linear τ schedule, frame 0 cleanest) beat uniform τ across the sequence?
- **H1b (active).** Within video-mode training, does **uniform loss weighting** beat `ramp` weighting for the noisy-context regime? `ramp` (`0.9τ+0.1`) systematically down-weights the noisy end, which is exactly the regime we care about.
- **H2 (open).** Does shortcut bootstrap help any of the three target modes, or does it mostly compress inference-time cost? Note: `ShortcutUWMForwardProcess` currently has no `action_sampler` mode, so testing H2 against mode 3 requires a code change.
- **H3 (open).** Does `latent_attends_action=true` (Z-tokens seeing AC/A tokens within the intra-frame mask) help joint generation modes at the cost of clean separation? The `pushT-uwm-all.slurm` turns this on; isolated ablation TBD.
- **H4 (open, code ready 2026-04-23).** Following the Dreamer-V4 paper's note on alternating short/long batches: does mixing in occasional long-T (T > `context_length`) batches during training prevent the transformer from overfitting to "frame 0 is always episode start" and improve length generalization at inference? The plumbing for this is now in `train_dynamics_uwm.py` — controlled by `train.long_seq_prob` and `train.long_seq_batch_per_gpu`, with the loader always delivering `max_sequence_length` frames. The existing windowed-temporal-attention path in `blocks.py:472-487` engages automatically when `T > context_length`. Defaults preserve old behavior. Untested as of this entry — see "Pending tests" memory.
- **H5 (open, code ready 2026-04-24).** Per the Dreamer-V4 paper's "treat 30% of videos as separate images" note: does adding a third training branch that flattens `(B, T, …) → (B·T, 1, …)` and trains each frame as an independent length-1 sequence teach the model the marginal image distribution p(x) — i.e., generate plausible *start frames* from pure noise? A causal temporal transformer never sees a true "frame 0 from nothing" during normal training; the image branch fills that gap. Implemented by reusing `video` mode under the hood (action τ=0, state loss across T=1) via a new `force_mode` kwarg on `UWMForwardProcess.forward()`. Controlled by `train.image_prob` / `train.image_batch_per_gpu`; defaults preserve old behavior. Untested — see "Pending tests" memory for the smoke sequence.

---

## Runs

### pushT-uwm-progressive-forcing-action-video-wm-image (queued 2026-04-24)
**Hypothesis.** Layering H5 (image-mode, 25%) and H4 (long-batch, 15%) on top of the current best recipe (progressive-forcing-action-video-wm) rescues mode 3 without regressing modes 1/2.
- Image mode → marginal p(x) for unconditional start-frame generation.
- Long branch → length generalization beyond `context_length`; breaks the "frame 0 = episode start" overfit.
- Base recipe stays strongest on policy + world-model; mode 3 is the target.

**Setup.**
- `train_dynamics_uwm.py`, config `dynamics/pushT`.
- `mode_weights.forcing=1, mode_weights.wm=2` (rest 0); `forcing_mask_actions=false`; `loss_weighting=uniform`; `latent_attends_action=true`.
- Branch mix: `image_prob=0.25, long_seq_prob=0.15` → short=0.60.
- `max_sequence_length=128, context_length=64` (long branch needs `max_seq > ctx_len` to fire).
- `image_batch_per_gpu=64` (conservative; image steps are compute-light since T=1 + no temporal attention).
- Fresh run (`reload_checkpoint=null`).
- SLURM: `hpc/slurms/dynamics/pushT/no-shortcut/pushT-uwm-progressive-forcing-action-video-wm-image.slurm`, 4×H200, 48h.

**Result.** Queued — image-mode and short/long branches are untested end-to-end. Local smoke tests (see `memory/project_pending_tests.md`) run first.

**Takeaway.** Regression checks on submit: (a) mode 1 (world-model loss) vs. PFAV-WM baseline; (b) mode 3 (action-sampler quality from noisy context); (c) unconditional start-frame generation quality (decode single-frame samples from pure noise, eyeball plausibility); (d) length-generalization eval at T > 64.

**Code notes.** First run that exercises the `image` branch (`force_mode='video'` at T=1) and the long branch simultaneously. Three `torch.compile` warmups expected (short `(4,64)`, long `(4,128)`, image `(64,1)`).

---

### pushT-uwm-policy-wm (baseline for modes 1 & 2)
**Hypothesis.** Training with only `policy` and `wm` modes produces a model that is simultaneously a good world model and a good policy.

**Setup.**
- `train_dynamics_uwm.py`, config `dynamics/pushT`.
- `mode_weights.policy=1`, `mode_weights.wm=1`, all others `0`.
- PushT dataset; 110M denoiser.

**Result.** Works well as world model and as policy. As **action sampler** (mode 3), quality degrades when conditioned on noisy context — the core problem motivating H1.

**Takeaway.** Current best recipe, but insufficient alone; additional training signal is needed for mode 3.

---

### pushT-uwm-all (kitchen-sink baseline)
**Hypothesis.** Turning on *all* modes simultaneously gives the model the widest competence.

**Setup.**
- `train_dynamics_uwm.py`, config `dynamics/pushT`.
- `mode_weights.{policy,video,forcing,id,wm}=1`, `latent_attends_action=true`.
- SLURM: `hpc/slurms/dynamics/pushT/no-shortcut/pushT-uwm-all.slurm`, 4×H200, 48h.

**Result.** TODO — link in metrics / W&B run when characterized.

**Takeaway.** Reference recipe for "no mode left out." Needed to isolate whether individual modes hurt or help.

---

### pushT-uwm-video (H1 primary)
**Hypothesis.** Adding `video` mode teaches generation-from-noise, which should transfer to `action_sampler` quality from noisy contexts.

**Setup.**
- `train_dynamics_uwm.py`, `mode_weights.video` dominant (see `pushT-uwm-video.slurm`).
- Flow matching, no shortcut.

**Result (partial).**
- With `ramp` weighting + no progressive noising: consistent but *low-entropy* videos from partially noisy context. As context noise grows, **temporal consistency and object permanence break down**.
- Follow-up runs sweep progressive noising and uniform weighting to attack this failure.

**Takeaway.** Problem localized to high-context-noise regime; two mechanisms under test (progressive noising, uniform weighting) as specific remedies.

---

### pushT-uwm-progressive-forcing-{video, action-video}
**Hypothesis.** Progressive per-frame τ schedule (frame 0 cleanest, frame T-1 noisiest, causal grounding) stabilizes generation from heavily-noisy contexts compared to single-τ-across-sequence.

**Setup.** `forcing` mode active with `forcing_context_noise.bias > 0`, optionally `forcing_mask_actions=true` for video-only progressive forcing.

**Result.** TODO.

**Takeaway.** TODO.

---

### pushT-uwm-progressive-forcing-action-video-wm (queued 2026-04-17)
**Hypothesis.** Adding `wm` mode on top of progressive-forcing-action-video explicitly teaches state-conditional-on-action generation (clean ctx + clean actions → predict state), which the pure-forcing mix never isolates. At 2:1 `wm:forcing`, the joint-generation signal from forcing still dominates the noisy-context regime (H1 target) while `wm` anchors the clean-context world-model behavior.

**Setup.**
- `train_dynamics_uwm.py`, config `dynamics/pushT`.
- `mode_weights.forcing=1`, `mode_weights.wm=2`, all others `0`.
- `forcing_mask_actions=false` (action channel participates in progressive forcing).
- `loss_weighting=uniform`, `latent_attends_action=true`.
- SLURM: `hpc/slurms/dynamics/pushT/no-shortcut/pushT-uwm-progressive-forcing-action-video-wm.slurm`, 4×H200, 48h.
- Run alongside uniform-weighting `video` and `forcing` baselines currently training.

**Result.** Queued — awaiting GPU allocation.

**Takeaway.** Regression check when it lands: (a) mode 1 (world model) — should improve vs. pure-forcing baseline given explicit `wm` training; (b) mode 3 (action sampler) — compare to the pure-forcing-action-video sibling to isolate whether the `wm` addition hurts the noisy-context generation the forcing schedule was installed to teach. Mode 2 (policy) remains untrained in this mix.

---

### pushT-shortcut-all / pushT-shortcut-wm-only / pushT-shortcut-wm-policy (shortcut variants)
**Hypothesis.** Shortcut bootstrap reduces inference-time cost and may improve sample quality via the EMA-teacher target.

**Setup.** `train_dynamics_uwm_with_shortcut.py`, `dynamics/pushT`. Shortcut modes: `{policy, video, wm, id, forcing}` (no `action_sampler`). Flow bias ≈ 0.75.

**Result (partial).** `pushT-shortcut-all` "not acting as a world model after 130k steps" per `ongoing-runs.md`. Other variants TBD.

**Takeaway.** Kitchen-sink shortcut training appears to hurt mode 1. Needs narrower mode mix to converge cleanly. Also: mode 3 test requires extending `ShortcutUWMForwardProcess` to add `action_sampler`.

---

### Sanity runs (Test0 / Test1, early block-causal UWM)
Migrated from `hpc/training-notes.md`.

- **Test0** — non-causal, full diffusion forcing on images, no diffusion forcing on actions, shortcut learning (same `d`, different obs/act noise), initialized from pretrained model, short-sequence only. Sanity only.
- **Test1** — non-causal, trained from scratch, same noise scheme as Test0, short-long sequence with truncated temporal attention, random fixed-horizon denoising conditioned on random context length `c` and `L-c` prediction horizon. Sanity only.

---

## Conventions

- **τ = 1 clean, τ = 0 noisy.** Opposite of the usual diffusion convention. `ramp` weighting `0.9τ+0.1` therefore down-weights the *noisy* end — be careful not to propagate the opposite assumption when reasoning about weighting schemes.
- **Action shape.** Dataset provides `(B, T, A)`; training scripts slice to `cfg.denoiser.n_actions` and `unsqueeze(-2)` to `(B, T, 1, n_actions)` — `num_action_tokens` must stay `1` unless this is revisited.
- **bf16 autocast** for both the frozen tokenizer and the trainable denoiser. Tokenizer is always in `no_grad`.
- **Slurm → hydra.** Every experiment is one slurm script → one `train_dynamics_uwm{,_with_shortcut}.py` invocation → hydra overrides. When adding a new experiment, clone the nearest `.slurm` and rename; don't invent a runner.
