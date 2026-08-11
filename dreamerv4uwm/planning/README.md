# Planning with the Unified World Model

Planning on top of the pushT UWM denoiser. The world model is used as a **simulator**:
nodes are **latent states** (an observation/action history the denoiser conditions on),
edges are **short rollouts** sampled from a generative prior policy `π_prior`, and a
pluggable reward scores states. Everything stays in tokenizer-latent space — we decode
only to score and to look — and runs under **bf16** autocast (fp32 OOMs at planning batch
sizes and is ~3.5× slower, with reconstruction error within noise of fp32).

The checkpoint has **no learned reward head**. Rewards here are either hand-designed pixel
heuristics on the decoded frame or a distance in a frozen DINOv2 embedding. Swap in a
learned critic later.

---

## Layout

Three core modules — the planner and nothing else:

| file | what it is |
|---|---|
| [`rollout.py`](rollout.py) | The **edge samplers**. Four batched primitives over one flow-matching idea, sharing one set of diversity knobs. |
| [`reward.py`](reward.py) | The **objective** — what makes one imagined state better than another. |
| [`mcts.py`](mcts.py) | The **search**. WorldPlanner-style UCT over latent states. |

Three modules around it:

| file | what it is |
|---|---|
| [`world.py`](world.py) | **Setup**: load denoiser + tokenizer, build a `decode_fn`, source initial contexts from real demos. |
| [`evaluate.py`](evaluate.py) | **Readouts**: what a plan achieved (`plan_peak` / `plan_last`) and the no-search controls to measure it against (`compute_baselines`). |
| [`diagnostics.py`](diagnostics.py) | **Debugging only**: task-specific state descriptors for inspecting a finished search. Never used inside the planner. |

Plus [`notebooks/`](notebooks/) (one experiment per notebook) and
[`visualization/`](visualization/) (trace → manimgl video).

Importing `dreamerv4uwm.planning` pulls in only the three core modules — `world`
(hydra + datasets) and `diagnostics` (cv2) are imported explicitly, so the planner stays
cheap to import.

---

## The three layers

### 1. Edge samplers — `rollout.py`

All four re-attach an optionally-noised clean context, then Euler-integrate the horizon
from its noise prior to clean. They differ only in *which* modality is integrated:

| primitive | horizon state | horizon action | cost/edge | what it is |
|---|---|---|---|---|
| `policy` | held at noise | integrated | 1 call | `p(a_{t:t+H} \| o_{≤t})` — the policy marginal |
| `transition` | integrated | given, clean | 1 call | the world model, `s, a → s'` |
| `imagine` | integrated | integrated | 1 call | joint `(a, o')` — **the default edge generator** |
| `autoregressive` | per-frame | per-frame | 2·H calls | `policy`→`transition` one frame at a time, re-conditioning on the realized state |

Each "call" is `K_steps` denoiser forwards. `imagine` is a policy and its own forward model
in a single object; `autoregressive` is the causally honest version (every action sees the
true previous state) at 2·H× the cost.

**Diversity knobs** (shared by all four — this is why they live in one file):

- `ctx_noise ∈ [0,1]` — noise mixed into the observation context. `0` = clean.
- `ctx_noise_honest` — tell the denoiser the context's *true* noise level (a principled
  widening of the posterior) or lie and announce it clean (mismatched / OOD contrast).
- `action_temp` — std of the action noise prior. Flow matching transports `N(0,I)` to
  `p(a|o)`, so scaling the prior is temperature-like. `≠ 1` is mildly OOD.
- `action_prior` / `state_prior` — `"normal"` or std-matched `"uniform"`. Note the latent is
  ~8k-dimensional, where a Gaussian and a std-matched uniform have near-identical norm
  concentration: expect the state-side knob to be weak, and read a null result as such.
- `action_noise` / `action_noise_dist` — (`autoregressive` only) extra noise **added** on
  top of the policy's own stochasticity.
- `K` — Euler steps.

> **Convention.** `n` is *noise level* (0 = clean, 1 = pure noise); `tau = 1 - n` is
> *cleanness*, the index the embeddings expect. Args take an `n`, integration runs in `tau`.

### 2. Objective — `reward.py`

Contract: `reward(z) -> r`, mapping `(..., N_lat, D_lat) -> (...)`. Reduce the two trailing
latent dims, preserve every leading batch/time dim, return float on `z.device`. Scale is
free, but keeping it ~O(1) per frame keeps `c_ucb` meaningful.

| reward | signal | cost |
|---|---|---|
| `TCenterReward` | red T's centroid near the target | 1 decode/state |
| `TCenterStraightReward` | + principal **axis** upright (180°-ambiguous: an upside-down T scores the same) | 1 decode/state |
| `TCenterAngleReward` | + full **heading** (penalises upside-down) | 1 decode/state |
| `DINOGoalReward` | distance to a goal frame in frozen DINOv2 space — task-agnostic | 1 decode + 1 ViT/state |
| `GoalLatentReward` | negative latent L2/cosine to a goal latent | free, but a weak signal |
| `ZeroReward` | nothing — for debugging the search itself | free |

`DINOGoalReward` needs `sigma` calibrated to the chosen `feature`/`metric`; use
`suggest_sigma()` on a batch of real latents. An uncalibrated `sigma` gives a flat reward.

**Decoding is the dominant planning cost.** One iteration decodes
`sim_rollouts × sim_horizon` states, plus `branching` more on the iterations that expand.
`budget()["n_reward_evals"]` reports the exact total.

### 3. Search — `mcts.py`

Textbook UCT, four steps per iteration:

1. **Selection** — descend from the root by max UCB1 until a childless leaf.
2. **Expansion** — if that leaf has been simulated once, sample `branching` edges from
   `π_prior` in one batched call; each edge's final state becomes a child.
3. **Simulation** — run `sim_rollouts` × `sim_horizon` frames of `π_prior` and reduce them
   to one scalar `R` via `value_backup`.
4. **Backpropagation** — add `R` to `V_total` and bump `n_visit` along the path.

The plan is the path from the root to the highest **average**-value node with
`n_visit > n_min`. `plan()` returns `a_seq`/`z_seq` (first edge), `plan_a`/`plan_z` (full
plan), `best_node`, `root`, `trace`, and `budget`.

---

## Three things that will bite you when designing a sweep

These are properties of the mechanism, not bugs. Each one is a confound if you sweep
around it without noticing.

### `value_backup` silently changes meaning with the reward family

The default `"best_prefix"` is WorldPlanner's `max over rollouts and prefixes`. But
`cumsum` of a **nonnegative** sequence is nondecreasing, so the best prefix is always the
last one — for every `TCenter*Reward` (range `[floor, 1]`) and `DINOGoalReward(mode='gauss')`,
`"best_prefix"` is *exactly* `"sum"`. The prefix logic only becomes live for rewards that go
negative (`GoalLatentReward`, `DINOGoalReward(mode='neg')`).

So the backup rule differs between reward families unless you pin it. Options:

| `value_backup` | `R` | over rollouts |
|---|---|---|
| `"best_prefix"` | `max_{k,T} Σ_{t≤T} γᵗ r_t^k` | max (default, WorldPlanner) |
| `"sum"` | `max_k Σ_t γᵗ r_t^k` | max |
| `"mean"` | `mean_k Σ_t γᵗ r_t^k` | mean |
| `"terminal"` | `mean_k r_{H_sim-1}^k` | mean |

### `sim_rollouts` and `sim_horizon` are not neutral compute knobs

Under the two max-based backups, `R` is a maximum over `M_sim` samples of a sum over
`H_sim` frames. A max over more samples is mechanically larger; a longer sum of nonnegative
rewards is mechanically larger. Both grow whether or not the policy improved, and they grow
unevenly across nodes. To sweep either as a *compute* knob, use `"mean"` or `"terminal"`.

### `n_iterations` is not a fair budget across `edge_mode`

One edge costs 1 primitive call under `imagine`, 2 under `two_stage`, and `2·horizon` under
`autoregressive` — each call being `K_steps` denoiser forwards. At `horizon=3` that is a 6×
gap; at `horizon=28`, 56×. Equalise `budget()["n_denoiser_calls"]` (or `plan_secs`), never
`n_iterations`:

```python
out = planner.plan(ctx_z, ctx_a)
out["budget"]   # n_forward, n_denoiser_calls, n_reward_evals, n_nodes, plan_secs
```

---

## A minimal experiment

```python
import torch
from dreamerv4uwm.planning import MCTS, PlanConfig, TCenterReward
from dreamerv4uwm.planning.world import (load_world_model, make_decode_fn,
                                         make_dataset, sample_initial_contexts)
from dreamerv4uwm.planning.evaluate import plan_peak, plan_last, compute_baselines

device = torch.device("cuda:0")

denoiser, tokenizer, cfg = load_world_model(dynamics_ckpt=DYN_CKPT, tokenizer_ckpt=TOK_CKPT)
decode = make_decode_fn(tokenizer, device)
reward = TCenterReward(decode_fn=decode, center_xy=(0.5, 0.5), sigma=0.25)

ctx = sample_initial_contexts(make_dataset(DATA_DIR), tokenizer, n=8, Tc=8,
                              device=device, n_actions=cfg.denoiser.n_actions, seed=0)[0]

plan_cfg = PlanConfig(horizon=3, branching=3, n_iterations=48,
                      ctx_noise=0.5, value_backup="best_prefix")
out = MCTS(denoiser, reward, plan_cfg, seed=0).plan(ctx["ctx_z"], ctx["ctx_a"])

peak, last = plan_peak(reward, out), plan_last(reward, out)
gains = compute_baselines(denoiser, reward, ctx["ctx_z"], ctx["ctx_a"], plan_cfg,
                          tree_peak=peak, tree_last=last)
print(peak, last, gains["g_greedy_peak"], out["budget"])
```

### Did search actually help?

There is **no closed-loop simulator**, so "did planning help?" is answered *relative* to
cheaper in-model controls, not against an environment. `compute_baselines` builds both
controls exactly the way the tree builds a plan — `max_depth` edges of `horizon` frames,
re-conditioning and truncating to `max_ctx` after each edge — so they share the plan's
lookahead and only the *search* differs:

- **random** — one depth-deep rollout, no selection (the undirected control);
- **greedy** — best of `n_random` depth-deep rollouts (random shooting).

Each is read out under two objectives: `*_peak` (best frame anywhere, MPC-style) and
`*_last` (the final state — "where did you end up"). A plan that shoves the T across the
centre and out again scores well on `peak` and badly on `last`. **Compare like with like**:
`g_*_peak` off `plan_peak`, `g_*_last` off `plan_last`, never crossed.

> **Caveat.** The baselines always build edges with `imagine`. Under
> `edge_mode="two_stage"`/`"autoregressive"` the gain then also reflects the edge-sampler
> choice, not search alone.

---

## Gotchas

- **The context is a sliding window, by design.** `_advance_ctx` appends each edge's frames
  and keeps only the last `max_ctx`: a node's "state" is the *recent* observation/action
  history the denoiser conditions on, not the whole trajectory. So the start frames drop out
  as the plan deepens, and deep nodes are conditioned entirely on imagined frames — that is
  the intent, not a leak. Size `max_ctx` to the window the denoiser conditions well on. The
  one thing to watch: the **root** is truncated the same way (`ctx_z[:, -max_ctx:]`), so
  handing `plan()` a context longer than `max_ctx` silently discards real frames you
  provided — keep `Tc ≤ max_ctx`.
- **`edge_val` is diagnostic only.** It costs `branching` decodes per expansion and is never
  read by the search — only by the trace and the visualization.
- **Actions are never clipped.** `action_temp > 1` and `action_noise` can emit actions
  outside the real action range, which `autoregressive` then feeds back as *clean* context.
  With no executor this never errors, so some "diversity" may be unexecutable.
- **Open loop only.** `plan()` is single-shot; there is no receding-horizon `act()` because
  there is nothing to execute against.
- **Determinism** is per-planner: `MCTS(seed=...)` owns one `torch.Generator`. Same seed +
  same context + same config ⇒ same tree.

---

## Visualization

[`visualization/make_mcts_trace_pushT.py`](visualization/make_mcts_trace_pushT.py) builds
the **branchable vs collapsed** contrast — the same search, same budget, two knobs apart:

```
branchable :  ctx_noise = 0.7 , edge horizon H = 28
collapsed  :  ctx_noise = 0.0 , edge horizon H =  6
```

With observation noise and long edges, sibling edges reach *distinguishable* states, UCB
commits, and the tree plans deep. With zero noise and short edges every sibling collapses
onto the same world-model trajectory, UCB has nothing to choose between, and visits spread
uniformly with flat values — broad, shallow, no real search.

It writes `mcts_trace_<regime>.json` plus decoded node/edge frames. Render from
`visualization/` with the installed **manimgl** (not community manim):

```bash
MCTS_TRACE=mcts_trace_branchable.json \
    xvfb-run -a manimgl viz_mcts_pushT_manim.py MCTSTreeScene -w --hd --file_name mcts_branchable
```

Note this script predates the `value_backup` knob, so it runs the default `"best_prefix"`.

---

## Reference

Khorrambakht, Ortiz-Haro et al., *WorldPlanner* (2025).

pushT dims: `n_actions=2`, `num_latent_tokens=256`, `latent_dim=32`, `num_noise_levels=128`.
Config `dynamics/pushT-large` with override `denoiser.horizon_aware=false`.
