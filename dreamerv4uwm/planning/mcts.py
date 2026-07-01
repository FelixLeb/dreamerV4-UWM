"""Monte-Carlo Tree Search over latent world-model states.

Nodes are **states** (a latent observation/action history the denoiser
conditions on). Edges are **short rollouts** sampled from a generative prior
``pi_prior`` — by default :func:`rollout.imagine`, which jointly denoises an
action sequence *and* the states it induces. Applying an edge advances the
context window to a child node; a basic latent reward scores states.

Because the action space is continuous and edges are *sampled* (not enumerable),
this is a continuous / generative MCTS:

* **Progressive widening** — a node visited ``n`` times is allowed
  ``ceil(pw_c * n^pw_alpha)`` children. New children are revealed one at a time
  from a fixed candidate **pool** that was drawn in a single batched
  ``pi_prior`` call when the node was first expanded (amortizes the denoiser
  forward over the whole pool).
* **UCB selection** with AlphaZero-style ``sqrt(N_parent)/(1+N_child)``
  exploration and MuZero-style min/max value normalization (rewards live on an
  arbitrary negative-distance scale, so Q must be normalized for the exploration
  constant to behave).
* **Leaf evaluation** — a newly created node is valued by its own state reward,
  optionally bootstrapped by a few short imagined simulation rollouts.

The whole search is deterministic given a seed and runs entirely in latent
space; decode only for visualization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

from . import rollout as R
from .reward import RewardFn, ZeroReward


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class PlanConfig:
    # --- edge / rollout ---
    horizon: int = 4              # H: frames per edge (length of a short rollout)
    pool_size: int = 8            # P: candidate edges drawn per node expansion
    K_steps: int = 8              # denoiser Euler steps per rollout
    # diversity knobs forwarded to the default `imagine` edge sampler
    ctx_noise: float = 0.0
    ctx_noise_honest: bool = True
    action_temp: float = 1.0
    # --- search ---
    n_simulations: int = 64       # number of root-to-leaf traversals
    max_depth: int = 4            # max tree depth (edges from root)
    gamma: float = 0.99           # per-frame discount
    c_ucb: float = 1.25           # exploration constant
    pw_c: float = 1.0             # progressive-widening coefficient
    pw_alpha: float = 0.5         # progressive-widening exponent
    # --- leaf evaluation (simulation) ---
    value_rollouts: int = 0       # # of imagined sim rollouts to bootstrap leaf value (0 = state reward only)
    value_sim_depth: int = 1      # edges per simulation rollout
    # --- bookkeeping ---
    max_ctx: int = 16             # frames of history kept per node context (truncates the window)
    dtype: Optional[torch.dtype] = torch.bfloat16


# A pluggable edge sampler:  (ctx_z, ctx_a, H, B) -> (z_seq (B,H,N,D), a_seq (B,H,n_act))
EdgeSampler = Callable[[torch.Tensor, torch.Tensor, int, int], Tuple[torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------------
# value normalization
# ---------------------------------------------------------------------------

class _MinMax:
    """Running min/max for normalizing Q into [0, 1] (MuZero trick)."""

    def __init__(self):
        self.lo = float("inf")
        self.hi = -float("inf")

    def update(self, v: float):
        self.lo = min(self.lo, v)
        self.hi = max(self.hi, v)

    def norm(self, v: float) -> float:
        if self.hi > self.lo:
            return (v - self.lo) / (self.hi - self.lo)
        return v


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    a_seq: torch.Tensor           # (H, n_act) action rollout
    z_seq: torch.Tensor           # (H, N_lat, D_lat) state rollout
    g_edge: float                 # discounted reward accumulated along this edge
    child: "Node"
    N: int = 0                    # visit count
    W: float = 0.0                # total backed-up return

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


@dataclass
class Node:
    ctx_z: torch.Tensor           # (1, Tc, N_lat, D_lat) conditioning context
    ctx_a: torch.Tensor           # (1, Tc, n_act)
    depth: int
    N: int = 0                    # visit count
    V_leaf: float = 0.0           # initial leaf value estimate
    children: List[Edge] = field(default_factory=list)
    # candidate pool (drawn lazily on first expansion)
    _pool_z: Optional[torch.Tensor] = None   # (P, H, N_lat, D_lat)
    _pool_a: Optional[torch.Tensor] = None   # (P, H, n_act)
    _pool_g: Optional[torch.Tensor] = None   # (P,) edge returns
    _pool_used: int = 0

    @property
    def is_terminal(self) -> bool:
        return False  # depth cap is handled by the search, not the node


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

class MCTS:
    """Generative MCTS planner. See module docstring."""

    def __init__(
        self,
        denoiser,
        reward_fn: Optional[RewardFn] = None,
        cfg: Optional[PlanConfig] = None,
        edge_sampler: Optional[EdgeSampler] = None,
        seed: int = 0,
    ):
        self.denoiser = denoiser
        self.reward_fn = reward_fn if reward_fn is not None else ZeroReward()
        self.cfg = cfg if cfg is not None else PlanConfig()
        self.device = next(denoiser.parameters()).device
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.minmax = _MinMax()
        self._edge_sampler = edge_sampler or self._default_edge_sampler
        self.root: Optional[Node] = None
        self.n_forward = 0  # diagnostic: number of batched edge-sampler calls

    # --- default edge sampler: joint imagination with the config's knobs -----
    def _default_edge_sampler(self, ctx_z, ctx_a, H, B):
        return R.imagine(
            self.denoiser, ctx_z, ctx_a, H, B=B, K=self.cfg.K_steps,
            ctx_noise=self.cfg.ctx_noise, ctx_noise_honest=self.cfg.ctx_noise_honest,
            action_temp=self.cfg.action_temp, dtype=self.cfg.dtype, generator=self.gen)

    # --- reward of a state rollout -> discounted edge return -----------------
    def _edge_return(self, z_seq: torch.Tensor) -> torch.Tensor:
        """z_seq: (B, H, N_lat, D_lat) -> g: (B,) discounted reward along edge."""
        r = self.reward_fn(z_seq)                       # (B, H)
        H = r.shape[1]
        disc = (self.cfg.gamma ** torch.arange(H, device=r.device, dtype=r.dtype))
        return (r * disc).sum(dim=1)                    # (B,)

    # --- grow the conditioning window by one edge ----------------------------
    def _advance_ctx(self, ctx_z: torch.Tensor, ctx_a: torch.Tensor,
                     z_seq: torch.Tensor, a_seq: torch.Tensor):
        """Return (ctx_z, ctx_a) after appending one edge (z_seq (H,..), a_seq
        (H,..)) and truncating to the last ``max_ctx`` frames."""
        cz = torch.cat([ctx_z, z_seq[None]], dim=1)
        ca = torch.cat([ctx_a, a_seq[None]], dim=1)
        m = self.cfg.max_ctx
        return cz[:, -m:].contiguous(), ca[:, -m:].contiguous()

    # --- expand a node's candidate pool (one batched edge-sampler call) ------
    def _ensure_pool(self, node: Node):
        if node._pool_z is not None:
            return
        P, H = self.cfg.pool_size, self.cfg.horizon
        z_seq, a_seq = self._edge_sampler(node.ctx_z, node.ctx_a, H, P)  # (P,H,..)
        self.n_forward += 1
        node._pool_z = z_seq
        node._pool_a = a_seq
        node._pool_g = self._edge_return(z_seq)         # (P,)

    # --- progressive-widening reveal of the next pool edge -------------------
    def _maybe_widen(self, node: Node) -> Optional[Edge]:
        allowed = int(self.cfg.pw_c * (node.N + 1) ** self.cfg.pw_alpha)
        allowed = max(1, min(self.cfg.pool_size, allowed))
        if len(node.children) >= allowed or node._pool_used >= self.cfg.pool_size:
            return None
        i = node._pool_used
        node._pool_used += 1
        z_seq, a_seq = node._pool_z[i], node._pool_a[i]
        g = float(node._pool_g[i])
        cz, ca = self._advance_ctx(node.ctx_z, node.ctx_a, z_seq, a_seq)
        child = Node(ctx_z=cz, ctx_a=ca, depth=node.depth + 1)
        child.V_leaf = self._leaf_value(child)
        edge = Edge(a_seq=a_seq, z_seq=z_seq, g_edge=g, child=child)
        node.children.append(edge)
        return edge

    # --- leaf value: bootstrap of the return AFTER the leaf ------------------
    def _leaf_value(self, node: Node) -> float:
        """Bootstrap value of a fresh leaf = estimate of the discounted return
        *after* the leaf state (the leaf's own reward is already counted in its
        incoming edge's ``g_edge``, so we don't re-add it). Zero when no
        simulation is requested; otherwise the mean discounted return of a few
        short imagined rollouts continuing from the leaf (batched per hop)."""
        if self.cfg.value_rollouts <= 0:
            return 0.0
        B = self.cfg.value_rollouts
        cz = node.ctx_z.expand(B, -1, -1, -1).contiguous()
        ca = node.ctx_a.expand(B, -1, -1).contiguous()
        g_total = torch.zeros(B, device=self.device)
        disc = 1.0
        for _ in range(self.cfg.value_sim_depth):
            z_seq, a_seq = self._edge_sampler(cz, ca, self.cfg.horizon, B)
            self.n_forward += 1
            g_total = g_total + disc * self._edge_return(z_seq)      # (B,)
            disc *= self.cfg.gamma ** self.cfg.horizon
            cz = torch.cat([cz, z_seq], dim=1)[:, -self.cfg.max_ctx:].contiguous()
            ca = torch.cat([ca, a_seq], dim=1)[:, -self.cfg.max_ctx:].contiguous()
        return float(g_total.mean().item())

    # --- UCB child selection -------------------------------------------------
    def _select(self, node: Node) -> Edge:
        best, best_score = None, -float("inf")
        sqrtN = (node.N + 1) ** 0.5
        for e in node.children:
            q = self.minmax.norm(e.Q) if e.N > 0 else 0.0
            u = self.cfg.c_ucb * sqrtN / (1 + e.N)
            score = q + u
            if score > best_score:
                best, best_score = e, score
        return best

    # --- one simulation: descend, expand, backup -----------------------------
    def _simulate(self):
        path: List[Tuple[Node, Edge]] = []
        node = self.root
        while True:
            self._ensure_pool(node)
            edge = self._maybe_widen(node)            # try to reveal a fresh child
            if edge is None:                          # widening saturated -> select
                edge = self._select(node)
                path.append((node, edge))
                node = edge.child
                if node.depth >= self.cfg.max_depth:
                    break
                continue
            path.append((node, edge))                 # new edge -> stop & evaluate
            break

        # backup: G starts at the leaf value of the deepest child reached.
        leaf = path[-1][1].child
        G = leaf.V_leaf
        gamma_H = self.cfg.gamma ** self.cfg.horizon
        for node, edge in reversed(path):
            G = edge.g_edge + gamma_H * G
            edge.N += 1
            edge.W += G
            node.N += 1
            self.minmax.update(edge.Q)

    # --- public API ----------------------------------------------------------
    @torch.no_grad()
    def plan(self, ctx_z: torch.Tensor, ctx_a: torch.Tensor, verbose: bool = False):
        """Run the search from an initial context and return the recommended
        plan plus diagnostics.

        Args:
            ctx_z: (1, Tc, N_lat, D_lat) clean latent observation history.
            ctx_a: (1, Tc, n_act) clean action history.

        Returns a dict with:
            ``a_seq``  — best first-edge action rollout (H, n_act) to execute.
            ``z_seq``  — its predicted state rollout (H, N_lat, D_lat).
            ``root``   — the root Node (full tree, for inspection / viz).
            ``best_path`` — list of Edges along the most-visited path.
            ``n_forward`` — batched denoiser edge-sampler calls used.
        """
        m = self.cfg.max_ctx
        self.root = Node(ctx_z=ctx_z[:, -m:].clone(), ctx_a=ctx_a[:, -m:].clone(), depth=0)
        self.minmax = _MinMax()
        self.n_forward = 0
        for i in range(self.cfg.n_simulations):
            self._simulate()
            if verbose and (i + 1) % max(1, self.cfg.n_simulations // 5) == 0:
                nc = len(self.root.children)
                print(f"  sim {i+1:3d}/{self.cfg.n_simulations}  root children={nc}  "
                      f"forwards={self.n_forward}")
        if not self.root.children:
            raise RuntimeError("search produced no children (check n_simulations / pool_size)")
        best = max(self.root.children, key=lambda e: e.N)
        best_path = self._greedy_path()
        return dict(a_seq=best.a_seq, z_seq=best.z_seq, root=self.root,
                    best_path=best_path, n_forward=self.n_forward,
                    root_child_stats=[(e.N, e.Q) for e in self.root.children])

    def _greedy_path(self) -> List[Edge]:
        """Most-visited path from the root (the planned trajectory)."""
        path, node = [], self.root
        while node.children:
            e = max(node.children, key=lambda e: e.N)
            path.append(e)
            node = e.child
        return path