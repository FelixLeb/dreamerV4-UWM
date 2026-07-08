"""Generate replayable EasyMCTS traces for the **branchable vs collapsed** pushT
contrast (notebook ``planning-mcts-experiments_v3_clean.ipynb``, Part IV).

Both regimes run the *same* WorldPlanner UCT search, from the *same* pushT start
context, with the *same* branching / iteration budget. Only two knobs differ:

    branchable :  ctx_noise = 0.7 , edge horizon H = 28
    collapsed  :  ctx_noise = 0.0 , edge horizon H =  6

With observation noise + long edges the prior policy ``pi_prior`` produces
sibling edges that reach **distinguishable** states — the tree branches, UCB
commits to the good branch and plans several steps deep. With zero noise + short
edges every sibling collapses back onto the *same* world-model trajectory —
UCB has nothing to choose between and round-robins blindly (broad & shallow,
flat values).

For every tree node we decode its world-model **state** to a pushT image; for
every edge we decode its short **policy/world-model rollout** to a filmstrip.
We also replay the event stream to attach a *running* metrics snapshot (visit
entropy, value spread, edge-reward std, commit fraction) to each backprop, so
the manim dashboard can update live. The scene ``viz_mcts_pushT_manim_v2.py``
replays each trace.

Output (next to this file):
    mcts_trace_<regime>.json                 # topology + events + metrics + layout
    frames_<regime>/node_<id>.png            # decoded terminal state of each node
    frames_<regime>/edge_<id>.png            # decoded rollout filmstrip per edge
    frames_<regime>/{start,best}.png

Run (conda env with the model deps):  python make_mcts_trace_pushT_v2.py
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate
from hydra import initialize_config_dir, compose

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dreamerv4uwm.models.utils import load_tokenizer, load_denoiser
from dreamerv4uwm.datasets import ShardedHDF5Dataset
from dreamerv4uwm.planning.easy_mcts import EasyMCTS, EasyPlanConfig
from dreamerv4uwm.planning.reward import TCenterReward, score_t_centered

# --- checkpoints / data (local runtime; see local-runtime-bf16 memory) -------
CFG_DIR = os.path.join(_REPO, "scripts", "config")
DYN_CKPT = "/home/mim-server/projects/rooholla/dreamerV4-UWM/checkpoints/blockcausal/pushT-post-train/97500.pt"
TOK_CKPT = "/home/mim-server/projects/rooholla/dreamerV4-UWM/checkpoints/tokenizer/pushT.pt"
DATA = "/home/mim-server/datasets/pushT/h5/play"

# --- notebook start context: window 1234, decision at frame t0+T_ctx ---------
WINDOW_SEED, T0, T_CTX = 1234, 50, 8
SCORE_KW = dict(center_xy=(0.5, 0.5), sigma=0.25)
THUMB = 132          # node thumbnail px
STRIP_FRAMES = 8     # frames kept in an edge filmstrip
STRIP_H = 96         # filmstrip row height px

# shared search budget; only the two knobs below differ between regimes
BASE = dict(branching=5, max_depth=3, sim_rollouts=3, action_temp=1.0,
            n_iterations=12, K_steps=6, gamma=0.98, c_ucb=0.5, n_min=0, max_ctx=24)
REGIMES = {
    "branchable": dict(horizon=28, sim_horizon=28, ctx_noise=0.7),
    "collapsed":  dict(horizon=6,  sim_horizon=6,  ctx_noise=0.0),
}
VERDICT = {
    "branchable": "Sibling edges reach DIFFERENT states -> UCB can choose. "
                  "The tree branches, commits to the best child, and plans deep.",
    "collapsed":  "Sibling edges COLLAPSE to the same state -> UCB is blind. "
                  "Visits spread uniformly, values are flat: no real search.",
}

device = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------
def _to_u8(chw):
    return (chw.permute(1, 2, 0).float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def _save(img_u8, path, size=None):
    im = Image.fromarray(img_u8)
    if size:
        im = im.resize(size, Image.BILINEAR)
    im.save(path)


def _filmstrip(strip_thw3_or_tchw):
    """(H,3,h,w) float lat

    -> single (STRIP_H, W, 3) uint8 row, evenly subsampled to STRIP_FRAMES."""
    T = strip_thw3_or_tchw.shape[0]
    idx = np.linspace(0, T - 1, min(STRIP_FRAMES, T), dtype=int)
    frames = [_to_u8(strip_thw3_or_tchw[t]) for t in idx]
    sep = np.full((frames[0].shape[0], 3, 3), 255, np.uint8)
    row = frames[0]
    for fr in frames[1:]:
        row = np.concatenate([row, sep, fr], axis=1)
    w = int(row.shape[1] * STRIP_H / row.shape[0])
    return row, (w, STRIP_H)


# ---------------------------------------------------------------------------
# tree layout: x by in-order leaf position, y by -depth
# ---------------------------------------------------------------------------
def layout_tree(nodes):
    children = {nid: [] for nid in nodes}
    for nid, meta in nodes.items():
        if meta["parent"] is not None:
            children[meta["parent"]].append(nid)
    # keep children in id order (creation order) so siblings read left->right
    for nid in children:
        children[nid].sort()
    cursor = [0.0]

    def rec(nid):
        kids = children[nid]
        if not kids:
            x = cursor[0]
            cursor[0] += 1.0
        else:
            for k in kids:
                rec(k)
            x = sum(nodes[k]["x"] for k in kids) / len(kids)
        nodes[nid]["x"] = x
        nodes[nid]["y"] = -float(nodes[nid]["depth"])

    root_id = next(nid for nid, m in nodes.items() if m["parent"] is None)
    rec(root_id)
    return root_id


# ---------------------------------------------------------------------------
# replay events -> running metrics on every backprop
# ---------------------------------------------------------------------------
def _norm_entropy(counts):
    tot = counts.sum()
    if tot <= 0 or len(counts) <= 1:
        return 0.0
    p = counts / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(len(counts)))


def annotate_running_metrics(trace, nodes, root_id):
    """Walk the event stream; after each backprop attach a metrics snapshot
    computed from the tree state *so far* (root-child visit entropy / value
    spread / commit, and edge-reward std over all edges created so far)."""
    root_children = [nid for nid, m in nodes.items() if m["parent"] == root_id]
    n_visit = {nid: 0 for nid in nodes}
    v_total = {nid: 0.0 for nid in nodes}
    created = set()
    sim_count = 0
    for e in trace["events"]:
        if e["type"] == "expand":
            created.add(e["node"])
            created.update(e["children"])
        elif e["type"] == "simulate":
            sim_count += 1
        elif e["type"] == "backprop":
            R = e["R"]
            for nid in e["path"]:
                n_visit[nid] += 1
                v_total[nid] += R
            rc_visits = np.array([n_visit[c] for c in root_children], float)
            vis_vals = [v_total[c] / n_visit[c] for c in root_children if n_visit[c] > 0]
            val_spread = (float(max(vis_vals) - np.mean(vis_vals)) if vis_vals else 0.0)
            # commit: share of root visits on the currently-best root child
            if vis_vals:
                best_c = max((c for c in root_children if n_visit[c] > 0),
                             key=lambda c: v_total[c] / n_visit[c])
                commit = float(n_visit[best_c] / max(rc_visits.sum(), 1))
            else:
                commit = 0.0
            edge_vals = np.array([nodes[c]["edge_val"] for c in created
                                  if nodes[c]["parent"] is not None])
            e["metrics"] = dict(
                iter=sim_count,
                n_nodes=len(created),
                visit_entropy=round(_norm_entropy(rc_visits), 3),
                val_spread=round(val_spread, 3),
                edge_val_std=round(float(edge_vals.std()) if edge_vals.size else 0.0, 3),
                commit=round(commit, 3),
            )


# ---------------------------------------------------------------------------
def build_regime(name, denoiser, decode, reward, cz, ca, n_act):
    knobs = {**REGIMES[name]}
    cfg_plan = EasyPlanConfig(**{**BASE, **knobs})
    print(f"\n=== {name} ===  cfg: ctx_noise={cfg_plan.ctx_noise} H={cfg_plan.horizon} "
          f"branching={cfg_plan.branching} n_iter={cfg_plan.n_iterations}")
    planner = EasyMCTS(denoiser, reward, cfg_plan, seed=0, trace=True)
    t = time.time()
    out = planner.plan(cz, ca, verbose=True)
    trace = out["trace"]
    nodes = trace["nodes"]
    print(f"  planned in {time.time()-t:.0f}s | nodes={len(planner.all_nodes)} "
          f"| forwards={planner.n_forward}")

    frame_dir = os.path.join(_HERE, f"frames_{name}")
    os.makedirs(frame_dir, exist_ok=True)

    # start reference (last context frame) + its reward
    start_state = cz[0, -1]
    start_rgb = _to_u8(decode(start_state[None, None])[0, 0])
    _save(start_rgb, os.path.join(frame_dir, "start.png"), (THUMB, THUMB))
    start_reward = float(score_t_centered(start_rgb, **SCORE_KW)[0])

    # decode every node state + edge rollout; stash per-node scalars
    for n in planner.all_nodes:
        state = n.edge_z[-1] if n.edge_z is not None else start_state
        _save(_to_u8(decode(state[None, None])[0, 0]),
              os.path.join(frame_dir, f"node_{n.id}.png"), (THUMB, THUMB))
        meta = nodes[n.id]
        meta["img"] = f"frames_{name}/node_{n.id}.png"
        meta["n_visit"] = n.n_visit
        meta["value"] = None if n.n_visit == 0 else round(n.value, 3)
        meta["edge_val"] = round(float(n.edge_val), 4)   # terminal reward in [0,1]
        if n.edge_z is not None:
            strip = decode(n.edge_z[None])[0]            # (H,3,h,w)
            row, sz = _filmstrip(strip)
            _save(row, os.path.join(frame_dir, f"edge_{n.id}.png"), sz)
            meta["strip"] = f"frames_{name}/edge_{n.id}.png"

    root_id = layout_tree(nodes)
    annotate_running_metrics(trace, nodes, root_id)

    # best-found node reference
    best = out["best_node"]
    best_rgb = _to_u8(decode((best.edge_z[-1])[None, None])[0, 0])
    _save(best_rgb, os.path.join(frame_dir, "best.png"), (THUMB, THUMB))
    best_reward = float(score_t_centered(best_rgb, **SCORE_KW)[0])

    # final whole-tree metrics
    ch = planner.root.children
    rc_visits = np.array([c.n_visit for c in ch], float)
    vis_vals = [c.value for c in ch if c.n_visit > 0]
    edge_vals = np.array([nd.edge_val for nd in planner.all_nodes if nd.parent is not None])
    depths = [nd.depth for nd in planner.all_nodes]
    plan_path = [nd.id for nd in planner._path_to(best)]

    trace["meta"] = dict(
        regime=name, verdict=VERDICT[name],
        root=root_id, best_node=best.id, plan_path=plan_path,
        n_nodes=len(planner.all_nodes), n_forward=planner.n_forward,
        max_reached_depth=max(depths),
        start_img=f"frames_{name}/start.png", start_reward=round(start_reward, 3),
        best_img=f"frames_{name}/best.png", best_reward=round(best_reward, 3),
        # knobs shown in the header
        ctx_noise=cfg_plan.ctx_noise, horizon=cfg_plan.horizon,
        branching=cfg_plan.branching, sim_horizon=cfg_plan.sim_horizon,
        sim_rollouts=cfg_plan.sim_rollouts, c_ucb=cfg_plan.c_ucb,
        gamma=cfg_plan.gamma, max_depth=cfg_plan.max_depth,
        n_iterations=cfg_plan.n_iterations,
        # final diagnostic scalars
        visit_entropy=round(_norm_entropy(rc_visits), 3),
        val_spread=round(float(max(vis_vals) - np.mean(vis_vals)) if vis_vals else 0.0, 3),
        edge_val_std=round(float(edge_vals.std()), 3),
        edge_val_min=round(float(edge_vals.min()), 3),
        edge_val_max=round(float(edge_vals.max()), 3),
    )
    trace["nodes"] = {str(k): v for k, v in nodes.items()}

    out_json = os.path.join(_HERE, f"mcts_trace_{name}.json")
    with open(out_json, "w") as f:
        json.dump(trace, f, indent=1)
    m = trace["meta"]
    print(f"  wrote {os.path.basename(out_json)}  best={best.id} plan={plan_path}")
    print(f"  visit_entropy={m['visit_entropy']:.2f}  val_spread={m['val_spread']:.2f}  "
          f"edge_val_std={m['edge_val_std']:.3f}  "
          f"reward {m['start_reward']:.2f}->{m['best_reward']:.2f}")
    return out_json


def main():
    torch.manual_seed(0)
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(config_name="dynamics/pushT-large",
                      overrides=["denoiser.horizon_aware=false"])
    cfg.dynamics_ckpt, cfg.tokenizer_ckpt = DYN_CKPT, TOK_CKPT
    denoiser = load_denoiser(cfg, device, max_num_forward_steps=300).eval().cuda()
    tokenizer = load_tokenizer(cfg, device, max_num_forward_steps=300).eval().cuda()

    @torch.no_grad()
    def decode(lat):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return tokenizer.decode(lat.to(device)).float().clamp(0, 1)

    ds = ShardedHDF5Dataset(data_dir=DATA, window_size=64, stride=1, split="train",
                            train_fraction=0.9, split_seed=123, shuffle_windows=False)
    b = ds[WINDOW_SEED]
    imgs = interpolate(b["image"], (256, 256)).to(device)[None]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        latents = tokenizer.encode(imgs).float()
    cz = latents[:, T0:T0 + T_CTX].clone()
    ca = b["action"][:, :cfg.denoiser.n_actions][None, T0:T0 + T_CTX].to(device).clone()
    reward = TCenterReward(decode_fn=decode, **SCORE_KW)

    for name in REGIMES:
        build_regime(name, denoiser, decode, reward, cz, ca, cfg.denoiser.n_actions)
    print("\nall traces written.")


if __name__ == "__main__":
    main()
