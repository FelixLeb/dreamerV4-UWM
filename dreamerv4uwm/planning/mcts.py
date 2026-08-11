"""MCTS - re-implementation of the WorldPlanner MCTS planner.

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
   rollouts of the prior policy for ``sim_horizon`` (``H_sim``) steps and reduce
   them to one scalar ``R`` with ``cfg.value_backup``. The default is
   WorldPlanner's *best cumulative reward over any rollout and any prefix*::

       R = max_{k in rollouts, T in 0..H_sim}  sum_{t=0}^{T} gamma^t r(s_t^k)

   See :class:`PlanConfig` for the other backups and for two properties of this
   one that matter when designing a sweep: it is **a no-op for nonnegative
   rewards** (it degenerates to the full-horizon sum), and being a max it
   **inflates with** ``sim_rollouts`` / ``sim_horizon``.

4. **Backpropagation** - add ``R`` to ``V_total`` and increment ``n_visit`` for
   every node on the path from the simulated node back to the root.

After ``n_iterations`` the plan is the path from the root to the node with the
highest **average** value among nodes visited more than ``n_min`` times.

The planner can emit a structured **trace** (``trace=True``) that records every
selection / expansion / simulation / backpropagation event, consumed by the
manimgl visualization (``visualization/viz_mcts_pushT_manim.py``).
"""
from __future__ import annotations

import math
import time
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
class PlanConfig:
    """Search knobs. One field needs more than a line of comment:

    ``value_backup`` — how the ``(M_sim, H_sim)`` simulation rewards collapse into the
    single scalar ``R`` that gets backed up the path. Write ``G_k = sum_t gamma^t r_t^k``
    for rollout ``k``'s full discounted return:

    ==============  ================================================  ==================
    value_backup    R                                                 over rollouts
    ==============  ================================================  ==================
    "best_prefix"   ``max_{k,T} sum_{t<=T} gamma^t r_t^k``            max  (WorldPlanner)
    "sum"           ``max_k G_k``                                     max
    "mean"          ``mean_k G_k``                                    mean
    "terminal"      ``mean_k r_{H_sim-1}^k``                          mean
    ==============  ================================================  ==================

    Two things to know before sweeping anything that touches this:

    * **"best_prefix" is a no-op whenever the reward is nonnegative.** ``cumsum`` of a
      nonnegative sequence is nondecreasing, so the best prefix is always the last one and
      ``"best_prefix" == "sum"``. That covers every ``TCenter*Reward`` (range ``[floor, 1]``)
      and ``DINOGoalReward(mode='gauss')``. The prefix logic only becomes live for rewards
      that can go negative — ``GoalLatentReward``, ``DINOGoalReward(mode='neg')``. So the
      backup rule silently changes meaning with the reward family; pin it explicitly when
      comparing families.
    * **The max-based backups inflate with ``sim_rollouts`` and ``sim_horizon``.** A max
      over more samples is mechanically larger, and a longer sum of nonnegative rewards is
      mechanically larger — whether or not the policy improved. Sweeping either knob under
      ``"best_prefix"`` / ``"sum"`` measures the estimator as much as the planner; use
      ``"mean"`` or ``"terminal"`` to sweep them as neutral compute knobs.
    """
    # --- expansion (edges) ---
    horizon: int = 3              # H: frames per expansion rollout (edge length)
    branching: int = 3            # B: children created per expansion (parallel rollouts)
    edge_mode: str = "imagine"    # "imagine" (joint) | "two_stage" (policy->WM, whole horizon) | "autoregressive" (step-by-step)
    # --- simulation ---
    sim_horizon: int = 8          # H_sim: frames per simulation rollout
    sim_rollouts: int = 4         # M_sim: parallel simulation rollouts
    value_backup: str = "best_prefix"   # how M_sim x H_sim rewards become one backed-up R
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
    action_prior: str = "normal"  # action noise prior: "normal" (action_temp*N(0,I)) | "uniform" (std-matched)
    state_prior: str = "normal"   # obs noise prior (unit-scale, no state_temp): "normal" | "uniform" (std-matched)
    action_noise: float = 0.0     # (edge_mode=autoregressive) magnitude of extra noise ADDED to each policy action
    action_noise_dist: str = "normal"  # (edge_mode=autoregressive) shape of that added noise: "normal" | "uniform"
    max_ctx: int = 8             # sliding conditioning window: frames kept per node (also caps the root)
    dtype: Optional[torch.dtype] = torch.bfloat16


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------

@dataclass
class Node:
    ctx_z: torch.Tensor           # (1, Tc, N_lat, D_lat) conditioning context
    ctx_a: torch.Tensor           # (1, Tc, n_act)
    depth: int                    # depth in the tree (root=0)
    id: int                       # unique node id (for tracing / debugging)
    parent: Optional["Node"] = None     # parent node (None for root)
    edge_a: Optional[torch.Tensor] = None   # (H, n_act) rollout that reached this node
    edge_z: Optional[torch.Tensor] = None   # (H, N_lat, D_lat)
    edge_val: float = 0.0                   # terminal reward of the reaching edge (diagnostic)
    n_visit: int = 0              # number of times this node has been visited
    V_total: float = 0.0          # total value accumulated from simulations
    children: List["Node"] = field(default_factory=list)  # child nodes
    expanded: bool = False        # True if children have been created

    @property
    def value(self) -> float:
        return self.V_total / self.n_visit if self.n_visit > 0 else float("-inf")

    # aliases so a node used as a plan "edge" reads like a rollout: every
    # element of the returned best_path exposes .z_seq / .a_seq
    @property
    def z_seq(self) -> Optional[torch.Tensor]:
        return self.edge_z

    @property
    def a_seq(self) -> Optional[torch.Tensor]:
        return self.edge_a


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

class MCTS:
    """WorldPlanner-style planner. See module docstring."""

    def __init__(
        self,
        denoiser,
        reward_fn: Optional[RewardFn] = None,
        cfg: Optional[PlanConfig] = None,
        edge_sampler: Optional[EdgeSampler] = None,
        seed: int = 0,
        trace: bool = False,
    ):
        self.denoiser = denoiser
        self.reward_fn = reward_fn if reward_fn is not None else ZeroReward()
        self.cfg = cfg if cfg is not None else PlanConfig()
        self.device = (next(denoiser.parameters()).device
                       if denoiser is not None else torch.device("cpu"))
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self._edge_sampler = edge_sampler  # None -> use imagine / two_stage per cfg
        self.root: Optional[Node] = None
        self.all_nodes: List[Node] = []
        # --- compute budget (see .budget()) ---
        self.n_forward = 0          # batched primitive rollout calls (policy/transition/imagine)
        self.n_denoiser_calls = 0   # denoiser forwards: each primitive call is K_steps of them
        self.n_reward_evals = 0     # states scored by reward_fn -- one decode each, the bottleneck
        self.plan_secs = 0.0        # wall-clock of the last plan()
        self._next_id = 0
        self.tracing = bool(trace)
        self.trace: dict = {"nodes": {}, "events": []} if trace else None

    # ---- node bookkeeping ---------------------------------------------------
    def _new_node(self, ctx_z, ctx_a, depth, parent=None,
                  edge_a=None, edge_z=None, edge_val=0.0) -> Node:
        node = Node(ctx_z=ctx_z, ctx_a=ctx_a, depth=depth, id=self._next_id,
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

    # ---- compute accounting -------------------------------------------------
    def _reward(self, z: torch.Tensor) -> torch.Tensor:
        """``reward_fn`` with a counter. Every scored state costs one tokenizer decode,
        which dominates planning cost, so this is the budget term that matters most."""
        self.n_reward_evals += int(z.shape[:-2].numel())
        return self.reward_fn(z)

    def budget(self) -> dict:
        """What this search actually cost — the numbers to equalise when comparing configs.

        ``n_forward`` alone is NOT comparable across ``edge_mode``: one edge costs 1
        primitive call under ``imagine``, 2 under ``two_stage`` and ``2*horizon`` under
        ``autoregressive``, and each primitive call is ``K_steps`` denoiser forwards. At
        ``horizon=3`` that is already a 6x gap, at ``horizon=28`` it is 56x — so comparing
        edge modes at equal ``n_iterations`` compares them at wildly unequal compute. Use
        ``n_denoiser_calls`` (or ``plan_secs``) as the budget axis instead.
        """
        return dict(n_forward=self.n_forward,
                    n_denoiser_calls=self.n_denoiser_calls,
                    n_reward_evals=self.n_reward_evals,
                    n_nodes=len(self.all_nodes),
                    plan_secs=self.plan_secs)

    def _count(self, n_primitive_calls: int):
        self.n_forward += n_primitive_calls
        self.n_denoiser_calls += n_primitive_calls * int(self.cfg.K_steps)

    # ---- rollout / edge sampling -------------------------------------------
    def _sample_edges(self, node: Node, B: int, H: int):
        """Return (z_seq (B,H,N,D), a_seq (B,H,n_act)) from pi_prior + world model."""
        c = self.cfg
        if self._edge_sampler is not None:
            out = self._edge_sampler(node.ctx_z, node.ctx_a, H, B)
            self._count(1)          # unknowable for a custom sampler; assume one K_steps pass
        elif c.edge_mode == "two_stage":
            a = R.policy(self.denoiser, node.ctx_z, node.ctx_a, H, B=B, K=c.K_steps,
                         ctx_noise=c.ctx_noise, ctx_noise_honest=c.ctx_noise_honest,
                         action_temp=c.action_temp, action_prior=c.action_prior,
                         state_prior=c.state_prior, dtype=c.dtype, generator=self.gen)
            z = R.transition(self.denoiser, node.ctx_z, node.ctx_a, a, K=c.K_steps,
                             ctx_noise=c.ctx_noise, ctx_noise_honest=c.ctx_noise_honest,
                             state_prior=c.state_prior, dtype=c.dtype, generator=self.gen)
            out = (z, a)
            self._count(2)                                       # policy + transition
        elif c.edge_mode == "autoregressive":
            out = R.autoregressive(self.denoiser, node.ctx_z, node.ctx_a, H, B=B, K=c.K_steps,
                                   ctx_noise=c.ctx_noise, ctx_noise_honest=c.ctx_noise_honest,
                                   action_temp=c.action_temp, action_prior=c.action_prior,
                                   state_prior=c.state_prior,
                                   action_noise=c.action_noise, action_noise_dist=c.action_noise_dist,
                                   dtype=c.dtype, generator=self.gen)
            self._count(2 * H)                                   # H*(policy + transition)
        else:  # joint imagination (default)
            out = R.imagine(self.denoiser, node.ctx_z, node.ctx_a, H, B=B, K=c.K_steps,
                            ctx_noise=c.ctx_noise, ctx_noise_honest=c.ctx_noise_honest,
                            action_temp=c.action_temp, action_prior=c.action_prior,
                            state_prior=c.state_prior, dtype=c.dtype, generator=self.gen)
            self._count(1)
        return out

    def _advance_ctx(self, ctx_z, ctx_a, z_seq, a_seq):
        """Apply an edge: append its frames, then slide the conditioning window forward.

        A node's state is the **last ``max_ctx`` frames**, not the trajectory that reached
        it — the denoiser conditions on a bounded recent history, so the start frames
        deliberately fall out of the window as the plan deepens.
        """
        cz = torch.cat([ctx_z, z_seq[None]], dim=1)
        ca = torch.cat([ctx_a, a_seq[None]], dim=1)
        m = self.cfg.max_ctx
        return cz[:, -m:].contiguous(), ca[:, -m:].contiguous()

    # ---- 1. selection -------------------------------------------------------
    def _ucb1(self, child: Node, parent: Node) -> float:
        if child.n_visit == 0:
            return float("inf")                          # visit every child once first
        exploit = child.V_total / child.n_visit
        explore = self.cfg.c_ucb * math.sqrt(
            math.log(max(parent.n_visit, 1)) / child.n_visit)
        return exploit + explore

    def _select(self) -> List[Node]:
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
    def _expand(self, node: Node):
        c = self.cfg
        z, a = self._sample_edges(node, c.branching, c.horizon)      # (B,H,..)
        term_r = self._reward(z[:, -1:]).squeeze(-1)                 # (B,) terminal reward
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
    def _simulate(self, node: Node) -> float:
        c = self.cfg
        z, _ = self._sample_edges(node, c.sim_rollouts, c.sim_horizon)   # (M,Hsim,N,D)
        r = self._reward(z)                                              # (M, Hsim)
        Hs = r.shape[1]
        disc = c.gamma ** torch.arange(Hs, device=r.device, dtype=r.dtype)
        cum = (r * disc).cumsum(dim=1)                                   # prefix sums, (M,Hsim)
        R_score = _backup(c.value_backup, r, cum)
        if self.tracing:
            best = int(cum.max(dim=1).values.argmax().item())
            self._log(type="simulate", node=node.id, R=R_score,
                      curve=[float(x) for x in cum[best].tolist()])
        return R_score

    # ---- 4. backpropagation -------------------------------------------------
    def _backprop(self, path: List[Node], R_score: float):
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
            ``root``       — the root Node.
            ``trace``      — the event trace (if ``trace=True``), else None.
            ``n_forward``  — batched primitive rollout calls used.
            ``budget``     — what the search cost; see :meth:`budget`. Equalise
                             ``n_denoiser_calls``, not ``n_iterations``, when comparing
                             configs that differ in ``edge_mode`` or ``K_steps``.
        """
        t0 = time.time()
        m = self.cfg.max_ctx
        self.root = self._new_node(ctx_z[:, -m:].clone(), ctx_a[:, -m:].clone(), depth=0)
        self._expand(self.root)
        if self.tracing:
            self._log(type="root", node=self.root.id)
        for i in range(self.cfg.n_iterations):
            self._iteration()
            if verbose and (i + 1) % max(1, self.cfg.n_iterations // 5) == 0:
                print(f"  iter {i+1:3d}/{self.cfg.n_iterations}  "
                      f"nodes={len(self.all_nodes)}  denoiser_calls={self.n_denoiser_calls}")
        self.plan_secs = time.time() - t0

        best = self._best_node()
        plan_nodes = self._path_to(best)
        plan_a = torch.stack([n.edge_a for n in plan_nodes], 0) if plan_nodes else None
        plan_z = torch.stack([n.edge_z for n in plan_nodes], 0) if plan_nodes else None
        first = plan_nodes[0] if plan_nodes else max(self.root.children, key=lambda n: n.value)
        return dict(
            # --- first edge to execute + tree summary ---
            a_seq=first.edge_a, z_seq=first.edge_z, best_path=plan_nodes,
            root=self.root, n_forward=self.n_forward,
            root_child_stats=[(c.n_visit, c.value) for c in self.root.children],
            # --- the full plan + trace ---
            plan_a=plan_a, plan_z=plan_z, best_node=best, trace=self.trace,
            # --- what it cost ---
            budget=self.budget())

    def _best_node(self) -> Node:
        """Node with the highest average value among those visited > n_min times."""
        cand = [n for n in self.all_nodes if n.n_visit > self.cfg.n_min and n.parent is not None]
        if not cand:
            cand = [n for n in self.all_nodes if n.n_visit > 0 and n.parent is not None]
        return max(cand, key=lambda n: n.value)

    def _path_to(self, node: Node) -> List[Node]:
        """List of nodes from the first edge under the root down to ``node``."""
        chain = []
        while node is not None and node.parent is not None:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))


def _backup(kind: str, r: torch.Tensor, cum: torch.Tensor) -> float:
    """Reduce simulation rewards ``r (M, H_sim)`` and their discounted prefix sums
    ``cum (M, H_sim)`` to the scalar backed up the path. See :class:`PlanConfig`."""
    if kind == "best_prefix":
        return float(cum.max().item())          # max over rollouts AND prefixes
    if kind == "sum":
        return float(cum[:, -1].max().item())   # max over rollouts of the full return
    if kind == "mean":
        return float(cum[:, -1].mean().item())  # average full return
    if kind == "terminal":
        return float(r[:, -1].mean().item())    # average reward of the final state
    raise ValueError(f"unknown value_backup={kind!r} "
                     "(expected 'best_prefix' | 'sum' | 'mean' | 'terminal')")


def _argmax(xs) -> int:
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i, best_v = i, v
    return best_i