"""Generate a replayable EasyMCTS trace on the **real pushT experiment**.

Same setup as ``notebooks/diverse-tests.ipynb`` Part IV (post-trained pushT UWM,
goal = a future frame), but with a small, legible search so the tree renders
cleanly. For every tree node we decode its world-model **state** to a pushT
image, and for every edge we decode the short **policy/world-model rollout**
to a filmstrip. The manim scene ``viz_mcts_pushT_manim.py`` replays it.

Output (next to this file):
    mcts_trace_pushT.json          # topology + events + image paths + layout
    pushT_frames/node_<id>.png     # decoded terminal state of each node
    pushT_frames/edge_<id>.png     # decoded H-frame rollout filmstrip per edge
    pushT_frames/{start,goal}.png

Run (conda env with the model deps):  python make_mcts_trace_pushT.py
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

OUT_JSON = os.path.join(_HERE, "mcts_trace_pushT.json")
FRAME_DIR = os.path.join(_HERE, "pushT_frames")
THUMB = 132   # node thumbnail px

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

    # --- non-optimal pushT start: T off-center (bottom-left), from window 24700 ---
    ds = ShardedHDF5Dataset(data_dir=DATA, window_size=64, stride=1,
                            split="train", train_fraction=0.9, split_seed=123)
    T_ctx, START_IDX = 8, 24700
    b = ds[START_IDX]
    sl = slice(24, 24 + T_ctx)
    imgs = interpolate(b["image"][sl], (256, 256)).to(device)[None]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        plan_cz = tokenizer.encode(imgs).float()
    plan_ca = b["action"][sl, :cfg.denoiser.n_actions][None].to(device)

    # task reward: red T centered + vertical (robust LR-symmetry orientation)
    reward = TCenterReward(decode_fn=decode, center_xy=(0.5, 0.5), sigma=0.22,
                           orient_method="vertical")

    # small, legible search (longer edges than the mock so the T can move)
    cfg_plan = EasyPlanConfig(
        horizon=6, branching=3, sim_horizon=6, sim_rollouts=2,
        n_iterations=10, max_depth=2, c_ucb=0.5, gamma=0.98, n_min=0,
        K_steps=6, ctx_noise=0.6, action_temp=2.0, max_ctx=16)
    planner = EasyMCTS(denoiser, reward, cfg_plan, seed=0, trace=True)
    out = planner.plan(plan_cz, plan_ca, verbose=True)
    trace = out["trace"]
    nodes = trace["nodes"]

    # --- decode node states + edge rollouts ----------------------------------
    _save(_to_u8(decode(plan_cz[:, -1:])[0, 0]), os.path.join(FRAME_DIR, "start.png"), (THUMB, THUMB))

    for n in planner.all_nodes:
        # node state image (terminal frame of its edge; root = last context frame)
        state = n.edge_z[-1] if n.edge_z is not None else plan_cz[0, -1]
        _save(_to_u8(decode(state[None, None])[0, 0]),
              os.path.join(FRAME_DIR, f"node_{n.id}.png"), (THUMB, THUMB))
        nodes[n.id]["img"] = f"pushT_frames/node_{n.id}.png"
        nodes[n.id]["n_visit"] = n.n_visit
        nodes[n.id]["value"] = None if n.n_visit == 0 else round(n.value, 3)
        # edge rollout filmstrip (H frames)
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
            nodes[n.id]["strip"] = f"pushT_frames/edge_{n.id}.png"

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

    # "target" panel = the highest-reward node the search found (its state image)
    best_r = max((n for n in planner.all_nodes if n.edge_z is not None),
                 key=lambda n: nodes[n.id].get("value") or -1e9, default=None)
    best_img = f"pushT_frames/node_{best_r.id}.png" if best_r else "pushT_frames/start.png"

    trace["meta"] = dict(
        root=root_id, best_node=out["best_node"].id,
        plan_path=[n.id for n in planner._path_to(out["best_node"])],
        n_nodes=len(planner.all_nodes), branching=cfg_plan.branching,
        sim_rollouts=cfg_plan.sim_rollouts, sim_horizon=cfg_plan.sim_horizon,
        c_ucb=cfg_plan.c_ucb, goal_img=best_img, goal_label="best found",
        start_img="pushT_frames/start.png", start_idx=START_IDX)
    trace["nodes"] = {str(k): v for k, v in nodes.items()}
    with open(OUT_JSON, "w") as f:
        json.dump(trace, f, indent=1)
    print(f"wrote {OUT_JSON}")
    print(f"  nodes={len(planner.all_nodes)}  events={len(trace['events'])}  "
          f"best={out['best_node'].id}  plan={trace['meta']['plan_path']}  "
          f"frames -> {FRAME_DIR}")


if __name__ == "__main__":
    main()