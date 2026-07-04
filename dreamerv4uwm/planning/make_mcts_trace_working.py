"""Generate a replayable EasyMCTS trace for the **working pushT plan**.

Companion to ``make_mcts_trace_pushT.py`` (the corner-start demo, where the world
model can barely move the T so planning ~ random). This variant starts from a
genuinely controllable non-optimal state — ``dataset[12000]`` frame 14, the T on
its side and off-center — where a small EasyMCTS search finds a plan that pushes
the T **upright and centered** (reward climbs from ~0.13). We keep the tree small
and legible (short branching / depth) so the manim scene reads cleanly, while the
edges are long enough for the T to actually move.

Output (next to this file):
    mcts_trace_working.json            # topology + events + image paths + layout
    working_frames/node_<id>.png       # decoded terminal state of each node
    working_frames/edge_<id>.png       # decoded H-frame rollout filmstrip per edge
    working_frames/{start,goal}.png

Run (conda env with the model deps):  python make_mcts_trace_working.py
Render:  MCTS_TRACE=mcts_trace_working.json \
             xvfb-run -a manimgl viz_mcts_pushT_manim.py MCTSPushTScene -w --low_quality
"""
import json
import os
import sys

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

CFG_DIR = os.path.join(_REPO, "scripts", "config")
DYN_CKPT = "/home/mim-server/projects/rooholla/dreamerV4-UWM/checkpoints/blockcausal/pushT-post-train/97500.pt"
TOK_CKPT = "/home/mim-server/projects/rooholla/dreamerV4-UWM/checkpoints/tokenizer/pushT.pt"
DATA = "/home/mim-server/datasets/pushT/h5/play"

OUT_JSON = os.path.join(_HERE, "mcts_trace_working.json")
FRAME_DIR = os.path.join(_HERE, "working_frames")
FRAME_REL = "working_frames"
THUMB = 132   # node thumbnail px

# --- the working start + a small, legible search ---------------------------
START_IDX, START_FRAME, T_ctx = 12000, 14, 8
SCORE_KW = dict(center_xy=(0.5, 0.5), sigma=0.22, w_center=0.6, w_orient=0.4,
                orient_method="vertical")
# Kept shallow (max_depth=2, branching=3 -> <=13 nodes) so the tree renders
# legibly, but with long edges + the winning diversity knobs so a 2-step plan
# still moves the T a long way toward centered+vertical. The search is small and
# stochastic, so we try a few seeds and keep the tree whose plan reaches the
# highest reward (SEEDS below).
PLAN = EasyPlanConfig(
    horizon=14, branching=3, sim_horizon=10, sim_rollouts=2,
    n_iterations=14, max_depth=2, c_ucb=0.7, gamma=0.98, n_min=0,
    K_steps=6, ctx_noise=0.7, action_temp=2.5, max_ctx=24)
SEEDS = [0, 1, 2, 3, 4]

device = torch.device("cuda:0")


def _to_u8(chw):
    return (chw.permute(1, 2, 0).float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def _save(img_u8, path, size=None):
    im = Image.fromarray(img_u8)
    if size:
        im = im.resize(size, Image.BILINEAR)
    im.save(path)


def main():
    os.makedirs(FRAME_DIR, exist_ok=True)
    torch.manual_seed(0)
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(config_name="dynamics/pushT-large",
                      overrides=["denoiser.horizon_aware=false"])
    cfg.dynamics_ckpt, cfg.tokenizer_ckpt = DYN_CKPT, TOK_CKPT
    denoiser = load_denoiser(cfg, device, max_num_forward_steps=300).eval().cuda()
    tokenizer = load_tokenizer(cfg, device, max_num_forward_steps=300).eval().cuda()

    @torch.no_grad()
    def decode(lat):  # (B,T,N,D) -> (B,T,3,H,W) float[0,1]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return tokenizer.decode(lat.to(device)).float().clamp(0, 1)

    def state_reward(state_lat):  # (N,D) -> scalar TCenterReward
        rgb = _to_u8(decode(state_lat[None, None])[0, 0])
        return float(score_t_centered(rgb, **SCORE_KW)[0])

    # --- controllable non-optimal start: T on its side, off-center -----------
    ds = ShardedHDF5Dataset(data_dir=DATA, window_size=64, stride=1,
                            split="train", train_fraction=0.9, split_seed=123)
    b = ds[START_IDX]
    sl = slice(START_FRAME - T_ctx, START_FRAME)
    # NOTE: the tokenizer encoder is temporally causal, so a frame's latent depends
    # on the frames before it. Encode the full clip up to the start and slice out the
    # context window (matches find_plan.py) so we reproduce the R~0.13 start exactly —
    # encoding only the 8-frame window would give a different (easier) state.
    imgs = interpolate(b["image"][:START_FRAME], (256, 256)).to(device)[None]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        lat_full = tokenizer.encode(imgs).float()
    plan_cz = lat_full[:, sl].contiguous()
    plan_ca = b["action"][sl, :cfg.denoiser.n_actions][None].to(device)

    reward = TCenterReward(decode_fn=decode, **SCORE_KW)
    r_start = state_reward(plan_cz[0, -1])

    # small stochastic search: try a few seeds, keep the tree whose plan path
    # reaches the highest terminal reward (peak along the recommended plan).
    planner = out = None
    best_metric = -1e9
    for seed in SEEDS:
        pl = EasyMCTS(denoiser, reward, PLAN, seed=seed, trace=True)
        o = pl.plan(plan_cz, plan_ca)
        path = pl._path_to(o["best_node"])
        peak = max((state_reward(n.edge_z[-1]) for n in path if n.edge_z is not None),
                   default=r_start)
        print(f"  seed={seed}  nodes={len(pl.all_nodes)}  plan_peak={peak:.3f}")
        if peak > best_metric:
            best_metric, planner, out = peak, pl, o
    trace = out["trace"]
    nodes = trace["nodes"]

    # --- decode node states + edge rollouts ----------------------------------
    _save(_to_u8(decode(plan_cz[:, -1:])[0, 0]), os.path.join(FRAME_DIR, "start.png"), (THUMB, THUMB))
    for n in planner.all_nodes:
        state = n.edge_z[-1] if n.edge_z is not None else plan_cz[0, -1]
        _save(_to_u8(decode(state[None, None])[0, 0]),
              os.path.join(FRAME_DIR, f"node_{n.id}.png"), (THUMB, THUMB))
        nodes[n.id]["img"] = f"{FRAME_REL}/node_{n.id}.png"
        nodes[n.id]["n_visit"] = n.n_visit
        nodes[n.id]["value"] = None if n.n_visit == 0 else round(n.value, 3)
        if n.edge_z is not None:
            strip = decode(n.edge_z[None])[0]                      # (H,3,H,W)
            frames = [_to_u8(strip[t]) for t in range(strip.shape[0])]
            sep = np.full((frames[0].shape[0], 3, 3), 255, np.uint8)
            row = frames[0]
            for fr in frames[1:]:
                row = np.concatenate([row, sep, fr], axis=1)
            h = 96
            w = int(row.shape[1] * h / row.shape[0])
            _save(row, os.path.join(FRAME_DIR, f"edge_{n.id}.png"), (w, h))
            nodes[n.id]["strip"] = f"{FRAME_REL}/edge_{n.id}.png"

    # --- tree layout (x by in-order leaves, y by depth) ----------------------
    children = {nid: [] for nid in nodes}
    for nid, meta in nodes.items():
        if meta["parent"] is not None:
            children[meta["parent"]].append(nid)
    xc = [0.0]

    def layout(nid):
        kids = children[nid]
        if not kids:
            x = xc[0]; xc[0] += 1.0
        else:
            for k in kids:
                layout(k)
            x = sum(nodes[k]["x"] for k in kids) / len(kids)
        nodes[nid]["x"] = x
        nodes[nid]["y"] = -float(nodes[nid]["depth"])

    root_id = next(nid for nid, m in nodes.items() if m["parent"] is None)
    layout(root_id)

    # "best found" panel = the highest-reward state along the recommended plan
    path = planner._path_to(out["best_node"])
    goal_node = max((n for n in path if n.edge_z is not None),
                    key=lambda n: state_reward(n.edge_z[-1]), default=out["best_node"])
    r_best = state_reward(goal_node.edge_z[-1]) if goal_node.edge_z is not None else r_start
    best_img = f"{FRAME_REL}/node_{goal_node.id}.png"

    trace["meta"] = dict(
        root=root_id, best_node=out["best_node"].id,
        plan_path=[n.id for n in path],
        n_nodes=len(planner.all_nodes), branching=PLAN.branching,
        sim_rollouts=PLAN.sim_rollouts, sim_horizon=PLAN.sim_horizon, c_ucb=PLAN.c_ucb,
        goal_img=best_img, goal_label="best found",
        start_img=f"{FRAME_REL}/start.png", start_idx=START_IDX,
        r_start=round(r_start, 3), r_best=round(r_best, 3))
    trace["nodes"] = {str(k): v for k, v in nodes.items()}
    with open(OUT_JSON, "w") as f:
        json.dump(trace, f, indent=1)
    print(f"wrote {OUT_JSON}")
    print(f"  nodes={len(planner.all_nodes)}  events={len(trace['events'])}  "
          f"best={out['best_node'].id}  plan={trace['meta']['plan_path']}")
    print(f"  reward: start={r_start:.3f} -> best node={r_best:.3f}  (gain {r_best - r_start:+.3f})")
    print(f"  frames -> {FRAME_DIR}")


if __name__ == "__main__":
    main()