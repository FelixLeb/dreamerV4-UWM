"""EasyMCTS - re-implementation of the WorldPlanner MCTS planner.

Reference: Khorrambakht, Ortiz-Haro et al., *WorldPlanner* (2025).

Nodes are states; edges are short rollouts from the prior policy ``pi_prior`` 
derived from UWT formulation;  the world model provides the transition. 
The four textbook steps per iteration:

1. **Selection** - from the root, descend by picking the child with the largest
   UCB1 score::

       UCB1(child) = V_total(child) / n(child)
                     + c * sqrt( ln n(parent) / n(child) )

   until a leaf (an un-expanded node) is reached.
2. **Expansion** - if that leaf has already been simulated once
   (``n_visit > 0``), sample a *fixed* ``branching`` number of short action
   rollouts from ``pi_prior`` and roll them out through the world model, in
   parallel; the last state of each becomes a child. Descend to the first new
   child.
3. **Simulation** - from the current node, run ``sim_rollouts`` (``M_sim``)
   rollouts of the prior policy for ``sim_horizon`` (``H_sim``) steps and score
   them with the *best cumulative reward over any rollout and any prefix*::

       R = max_{k in rollouts, T in 0..H_sim}  sum_{t=0}^{T} gamma^t r(s_t^k)

4. **Backpropagation** - add ``R`` to ``V_total`` and increment ``n_visit`` for
   every node on the path from the simulated node back to the root.

After ``n_iterations`` the plan is the path from the root to the node with the
highest **average** value among nodes visited more than ``n_min`` times.

The planner can emit a structured **trace** (``trace=True``) that records every
selection / expansion / simulation / backpropagation event, consumed by the
manimgl visualization (``viz_mcts_manim.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

from . import rollout as R
from .reward import RewardFn, ZeroReward

EdgeSampler = Callable[[torch.Tensor, torch.Tensor, int, int], Tuple[torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class EasyPlanConfig:
    # --- expansion (edges) ---
    horizon: int = 3              # H: frames per expansion rollout (edge length)
    branching: int = 3            # B: children created per expansion (parallel rollouts)
    edge_mode: str = "imagine"    # "imagine" (joint) or "two_stage" (policy -> world model)
    # --- simulation ---
    sim_horizon: int = 8          # H_sim: frames per simulation rollout
    sim_rollouts: int = 4         # M_sim: parallel simulation rollouts
    # --- search ---
    n_iterations: int = 48        # tree-building iterations
    c_ucb: float = 1.0            # UCT exploration constant
    gamma: float = 1.0            # per-frame discount (1.0 == undiscounted)
    n_min: int = 1                # min visits for a node to be a valid solution
    max_depth: int = 4            # cap on tree depth
    # --- rollout / bookkeeping ---
    K_steps: int = 6              # number of denoiser steps per rollout 
    ctx_noise: float = 0.5        # diversity so pi_prior branches 
    ctx_noise_honest: bool = True # False -> lie to the denoiser about the context noise
    action_temp: float = 1.0      # temperature for sampling actions from pi_prior
    max_ctx: int = 16             # max context length (frames) to keep in the tree
    dtype: Optional[torch.dtype] = torch.bfloat16


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------

@dataclass
class EasyNode:
    ctx_z: torch.Tensor           # (1, Tc, N_lat, D_lat) conditioning context
    ctx_a: torch.Tensor           # (1, Tc, n_act)
    depth: int                    # depth in the tree (root=0)
    id: int                       # unique node id (for tracing / debugging)
    parent: Optional["EasyNode"] = None     # parent node (None for root)
    edge_a: Optional[torch.Tensor] = None   # (H, n_act) rollout that reached this node
    edge_z: Optional[torch.Tensor] = None   # (H, N_lat, D_lat)
    edge_val: float = 0.0                   # terminal reward of the reaching edge (diagnostic)
    n_visit: int = 0              # number of times this node has been visited
    V_total: float = 0.0          # total value accumulated from simulations
    children: List["EasyNode"] = field(default_factory=list)  # child nodes
    expanded: bool = False        # True if children have been created

    @property
    def value(self) -> float:
        return self.V_total / self.n_visit if self.n_visit > 0 else float("-inf")

    # aliases so a node used as a plan "edge" matches the mcts.Edge interface
    # (best_path elements expose .z_seq / .a_seq in both planners)
    @property
    def z_seq(self) -> Optional[torch.Tensor]:
        return self.edge_z

    @property
    def a_seq(self) -> Optional[torch.Tensor]:
        return self.edge_a


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

class EasyMCTS:
    """WorldPlanner-style planner. See module docstring."""

    def __init__(
        self,
        denoiser,
        reward_fn: Optional[RewardFn] = None,
        cfg: Optional[EasyPlanConfig] = None,
        edge_sampler: Optional[EdgeSampler] = None,
        seed: int = 0,
        trace: bool = False,
    ):
        self.denoiser = denoiser
        self.reward_fn = reward_fn if reward_fn is not None else ZeroReward()
        self.cfg = cfg if cfg is not None else EasyPlanConfig()
        self.device = (next(denoiser.parameters()).device
                       if denoiser is not None else torch.device("cpu"))
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self._edge_sampler = edge_sampler  # None -> use imagine / two_stage per cfg
        self.root: Optional[EasyNode] = None
        self.all_nodes: List[EasyNode] = []
        self.n_forward = 0
        self._next_id = 0
        self.tracing = bool(trace)
        self.trace: dict = {"nodes": {}, "events": []} if trace else None

    # ---- node bookkeeping ---------------------------------------------------
    def _new_node(self, ctx_z, ctx_a, depth, parent=None,
                  edge_a=None, edge_z=None, edge_val=0.0) -> EasyNode:
        node = EasyNode(ctx_z=ctx_z, ctx_a=ctx_a, depth=depth, id=self._next_id,
                        parent=parent, edge_a=edge_a, edge_z=edge_z, edge_val=edge_val)
        self._next_id += 1
        self.all_nodes.append(node)
        if self.tracing:
            self.trace["nodes"][node.id] = {
                "id": node.id,
                "parent": parent.id if parent is not None else None,
                "depth": depth,
                "edge_val": float(edge_val),
                "action0": float(edge_a[0, 0]) if edge_a is not None else None,
            }
        return node

    def _log(self, **event):
        if self.tracing:
            self.trace["events"].append(event)

    # ---- rollout / edge sampling -------------------------------------------
    def _sample_edges(self, node: EasyNode, B: int, H: int):
        """Return (z_seq (B,H,N,D), a_seq (B,H,n_act)) from pi_prior + world model."""
        c = self.cfg
        if self._edge_sampler is not None:
            out = self._edge_sampler(node.ctx_z, node.ctx_a, H, B)
        elif c.edge_mode == "two_stage":
            a = R.policy(self.denoiser, node.ctx_z, node.ctx_a, H, B=B, K=c.K_steps,
                         ctx_noise=c.ctx_noise, ctx_noise_honest=c.ctx_noise_honest,
                         action_temp=c.action_temp, dtype=c.dtype, generator=self.gen)
            z = R.transition(self.denoiser, node.ctx_z, node.ctx_a, a, K=c.K_steps,
                             ctx_noise=c.ctx_noise, ctx_noise_honest=c.ctx_noise_honest,
                             dtype=c.dtype, generator=self.gen)
            out = (z, a)
            self.n_forward += 1
        else:  # joint imagination (default)
            out = R.imagine(self.denoiser, node.ctx_z, node.ctx_a, H, B=B, K=c.K_steps,
                            ctx_noise=c.ctx_noise, ctx_noise_honest=c.ctx_noise_honest,
                            action_temp=c.action_temp, dtype=c.dtype, generator=self.gen)
        self.n_forward += 1
        return out

    def _advance_ctx(self, ctx_z, ctx_a, z_seq, a_seq):
        cz = torch.cat([ctx_z, z_seq[None]], dim=1)
        ca = torch.cat([ctx_a, a_seq[None]], dim=1)
        m = self.cfg.max_ctx
        return cz[:, -m:].contiguous(), ca[:, -m:].contiguous()

    # ---- 1. selection -------------------------------------------------------
    def _ucb1(self, child: EasyNode, parent: EasyNode) -> float:
        if child.n_visit == 0:
            return float("inf")                          # visit every child once first
        exploit = child.V_total / child.n_visit
        explore = self.cfg.c_ucb * math.sqrt(
            math.log(max(parent.n_visit, 1)) / child.n_visit)
        return exploit + explore

    def _select(self) -> List[EasyNode]:
        """Descend root -> leaf by max UCB1; return the path (inclusive)."""
        path = [self.root]
        node = self.root
        while node.children:
            scores = [self._ucb1(ch, node) for ch in node.children]
            node = node.children[int(_argmax(scores))]
            path.append(node)
        if self.tracing:
            self._log(type="select", path=[n.id for n in path])
        return path

    # ---- 2. expansion -------------------------------------------------------
    def _expand(self, node: EasyNode):
        c = self.cfg
        z, a = self._sample_edges(node, c.branching, c.horizon)      # (B,H,..)
        term_r = self.reward_fn(z[:, -1:]).squeeze(-1)               # (B,) terminal reward
        for b in range(z.shape[0]):
            cz, ca = self._advance_ctx(node.ctx_z, node.ctx_a, z[b], a[b])
            child = self._new_node(cz, ca, node.depth + 1, parent=node,
                                   edge_a=a[b], edge_z=z[b], edge_val=float(term_r[b]))
            node.children.append(child)
        node.expanded = True
        if self.tracing:
            self._log(type="expand", node=node.id,
                      children=[ch.id for ch in node.children])

    # ---- 3. simulation ---------------------------------------------
    def _simulate(self, node: EasyNode) -> float:
        c = self.cfg
        z, _ = self._sample_edges(node, c.sim_rollouts, c.sim_horizon)   # (M,Hsim,N,D)
        r = self.reward_fn(z)                                            # (M, Hsim)
        Hs = r.shape[1]
        disc = c.gamma ** torch.arange(Hs, device=r.device, dtype=r.dtype)
        cum = (r * disc).cumsum(dim=1)                                   # prefix sums
        R_score = float(cum.max().item())                               # max over rollouts & prefixes
        if self.tracing:
            best = int(cum.max(dim=1).values.argmax().item())
            self._log(type="simulate", node=node.id, R=R_score,
                      curve=[float(x) for x in cum[best].tolist()])
        return R_score

    # ---- 4. backpropagation -------------------------------------------------
    def _backprop(self, path: List[EasyNode], R_score: float):
        for u in path:
            u.n_visit += 1
            u.V_total += R_score
        if self.tracing:
            self._log(type="backprop", path=[u.id for u in path], R=R_score,
                      stats={u.id: [u.n_visit, u.value] for u in path})

    # ---- one iteration ------------------------------------------------------
    def _iteration(self):
        path = self._select()                       # 1. selection -> leaf
        node = path[-1]
        if node.n_visit > 0 and node.depth < self.cfg.max_depth:
            self._expand(node)                       # 2. expansion
            node = node.children[0]                  #    descend to first new child
            path.append(node)
        R_score = self._simulate(node)               # 3. simulation
        self._backprop(path, R_score)                # 4. backpropagation

    # ---- public API ---------------------------------------------------------
    @torch.no_grad()
    def plan(self, ctx_z: torch.Tensor, ctx_a: torch.Tensor, verbose: bool = False):
        """Run WorldPlanner MCTS from an initial context; return plan + tree.

        Returns a dict with:
            ``a_seq``      — action rollout of the best root edge (H, n_act).
            ``plan_a`` / ``plan_z`` — full action / state plan root -> best node.
            ``best_node``  — the chosen node (max avg value, n_visit > n_min).
            ``root``       — the root EasyNode.
            ``trace``      — the event trace (if ``trace=True``), else None.
            ``n_forward``  — batched denoiser calls used.
        """
        m = self.cfg.max_ctx
        self.root = self._new_node(ctx_z[:, -m:].clone(), ctx_a[:, -m:].clone(), depth=0)
        self._expand(self.root)
        if self.tracing:
            self._log(type="root", node=self.root.id)
        for i in range(self.cfg.n_iterations):
            self._iteration()
            if verbose and (i + 1) % max(1, self.cfg.n_iterations // 5) == 0:
                print(f"  iter {i+1:3d}/{self.cfg.n_iterations}  "
                      f"nodes={len(self.all_nodes)}  forwards={self.n_forward}")

        best = self._best_node()
        plan_nodes = self._path_to(best)
        plan_a = torch.stack([n.edge_a for n in plan_nodes], 0) if plan_nodes else None
        plan_z = torch.stack([n.edge_z for n in plan_nodes], 0) if plan_nodes else None
        first = plan_nodes[0] if plan_nodes else max(self.root.children, key=lambda n: n.value)
        return dict(
            # --- shared contract with mcts.MCTS.plan() (drop-in compatible) ---
            a_seq=first.edge_a, z_seq=first.edge_z, best_path=plan_nodes,
            root=self.root, n_forward=self.n_forward,
            root_child_stats=[(c.n_visit, c.value) for c in self.root.children],
            # --- EasyMCTS extras ---
            plan_a=plan_a, plan_z=plan_z, best_node=best, trace=self.trace)

    def _best_node(self) -> EasyNode:
        """Node with the highest average value among those visited > n_min times."""
        cand = [n for n in self.all_nodes if n.n_visit > self.cfg.n_min and n.parent is not None]
        if not cand:
            cand = [n for n in self.all_nodes if n.n_visit > 0 and n.parent is not None]
        return max(cand, key=lambda n: n.value)

    def _path_to(self, node: EasyNode) -> List[EasyNode]:
        """List of nodes from the first edge under the root down to ``node``."""
        chain = []
        while node is not None and node.parent is not None:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))


def _argmax(xs) -> int:
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i, best_v = i, v
    return best_i