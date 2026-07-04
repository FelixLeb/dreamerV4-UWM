"""manimgl visualization of EasyMCTS on the **real pushT experiment**.

Same four-step animation as ``viz_mcts_manim.py`` but the tree is built from an
actual pushT UWM run: every node shows the decoded world-model **state**, and
during simulation the node's short **policy/world-model rollout** is shown as a
decoded filmstrip. The goal frame is pinned top-right for reference.

Colors:  Selection (yellow) / Expansion (green) / Simulation (blue) / Backprop (orange).
Node labels: visit count ``n`` and mean value ``Q = V_total/n`` (the UCB1 term).

Run
---
    python make_mcts_trace_pushT.py                          # writes trace + pushT_frames/*.png
    manimgl viz_mcts_pushT_manim.py MCTSPushTScene -w        # render to videos/*.mp4
    #  headless:  xvfb-run -a manimgl viz_mcts_pushT_manim.py MCTSPushTScene -w
"""
import json
import os

import numpy as np
from manimlib import *
from manimlib.mobject.types.image_mobject import ImageMobject

HERE = os.path.dirname(os.path.abspath(__file__))
# trace file is overridable so the same scene renders either the corner-start demo
# (default) or the working-plan tree:  MCTS_TRACE=mcts_trace_working.json manimgl ...
TRACE = os.environ.get("MCTS_TRACE") or os.path.join(HERE, "mcts_trace_pushT.json")
if not os.path.isabs(TRACE):
    TRACE = os.path.join(HERE, TRACE)

PHASE = {
    "select":   ("Selection",       YELLOW),
    "expand":   ("Expansion",       GREEN_B),
    "simulate": ("Simulation",      BLUE_B),
    "backprop": ("Backpropagation", ORANGE),
}
NODE_H = 0.92           # node thumbnail height (manim units)
HSCALE, VSCALE, TOP = 2.15, 2.35, 2.55


def _abs(rel):
    return os.path.join(HERE, rel)


class MCTSPushTScene(Scene):
    def construct(self):
        trace = json.load(open(TRACE))
        self.nodes = {int(k): v for k, v in trace["nodes"].items()}
        self.events = trace["events"]
        self.meta = trace["meta"]
        self.root_id = self.meta["root"]
        self.x_root = self.nodes[self.root_id]["x"]

        self.node_mobs = {}     # id -> dict(img, border, stat)
        self.edge_mobs = {}     # child_id -> Line
        self.sim_panel = None

        self._intro()
        self.play(FadeIn(self._make_node(self.root_id, root=True)), run_time=0.6)

        for e in self.events:
            t = e["type"]
            if t == "root":
                continue
            getattr(self, "_do_" + t)(e)

        self._highlight_plan()
        self.wait(2)

    # ------------------------------------------------------------- layout
    def _pos(self, nid):
        v = self.nodes[nid]
        return np.array([(v["x"] - self.x_root) * HSCALE, TOP + v["y"] * VSCALE, 0.0])

    def _make_node(self, nid, root=False):
        p = self._pos(nid)
        img = ImageMobject(_abs(self.nodes[nid]["img"])).set_height(NODE_H).move_to(p)
        border = Square(side_length=NODE_H).set_stroke(GREY_B, 2).set_fill(opacity=0).move_to(p)
        n = self.nodes[nid].get("n_visit", 0)
        val = self.nodes[nid].get("value")
        label = "start" if root else ("n=0" if not n else self._stat_txt(n, val))
        stat = Text(label, font_size=15, color=GREY_B).next_to(border, DOWN, buff=0.05)
        self.node_mobs[nid] = dict(img=img, border=border, stat=stat)
        return Group(img, border, stat)

    def _make_edge(self, child_id):
        parent = self.nodes[child_id]["parent"]
        line = Line(self._pos(parent), self._pos(child_id), stroke_width=3, color=GREY_B)
        line.set_length(max(0.05, line.get_length() - NODE_H))
        line.move_to((self._pos(parent) + self._pos(child_id)) / 2)
        self.edge_mobs[child_id] = line
        return line

    def _stat_txt(self, n, val):
        return "n=%d" % n if val is None else "n=%d  Q=%.1f" % (n, val)

    def _set_stat(self, nid, n, val):
        new = Text(self._stat_txt(n, val), font_size=15, color=WHITE).next_to(
            self.node_mobs[nid]["border"], DOWN, buff=0.05)
        self.node_mobs[nid]["stat"].become(new)

    # ------------------------------------------------------------- chrome
    def _intro(self):
        title = Text("EasyMCTS on pushT  (WorldPlanner UCT)", font_size=30).to_edge(UP)
        self.play(Write(title), run_time=0.9)
        self.title = title
        self.banner = Text("", font_size=24).to_corner(UL)
        self.add(self.banner)
        # target panel top-right (best state the search found, or a goal frame)
        goal = ImageMobject(_abs(self.meta["goal_img"])).set_height(1.15).to_corner(UR)
        gborder = Square(side_length=1.15).set_stroke(GOLD, 3).move_to(goal)
        glab = Text(self.meta.get("goal_label", "goal"), font_size=18, color=GOLD).next_to(goal, DOWN, buff=0.06)
        self.play(FadeIn(Group(goal, gborder, glab)), run_time=0.5)
        self._legend()

    def _legend(self):
        items = Group()
        for key in ["select", "expand", "simulate", "backprop"]:
            name, col = PHASE[key]
            dot = Dot(color=col, radius=0.07)
            lab = Text(name, font_size=15, color=col).next_to(dot, RIGHT, buff=0.1)
            items.add(Group(dot, lab))
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_corner(DL)
        self.play(FadeIn(items), run_time=0.4)

    def _set_phase(self, key):
        name, col = PHASE[key]
        self.play(Transform(self.banner, Text(name, font_size=24, color=col).to_corner(UL)),
                  run_time=0.25)

    # ------------------------------------------------------------- events
    def _do_expand(self, e):
        self._set_phase("expand")
        anims = []
        for k in e["children"]:
            anims += [ShowCreation(self._make_edge(k)), FadeIn(self._make_node(k))]
        self.play(*anims, run_time=0.9)

    def _do_select(self, e):
        self._set_phase("select")
        path = e["path"]
        borders = VGroup(*[self.node_mobs[nid]["border"] for nid in path])
        edges = VGroup(*[self.edge_mobs[nid] for nid in path[1:] if nid in self.edge_mobs])
        self.play(borders.animate.set_stroke(YELLOW, 4), run_time=0.25)
        if len(edges) > 0:
            self.play(ShowPassingFlash(edges.copy().set_color(YELLOW).set_stroke(width=6),
                                       time_width=0.6), run_time=0.6)
        self.play(borders.animate.set_stroke(GREY_B, 2), run_time=0.2)

    def _do_simulate(self, e):
        self._set_phase("simulate")
        nid = e["node"]
        rlabel = Text("R = %.1f" % e["R"], font_size=22, color=BLUE_B).next_to(
            self.node_mobs[nid]["border"], RIGHT, buff=0.12)
        shows = [FadeIn(rlabel, shift=0.1 * RIGHT),
                 Indicate(self.node_mobs[nid]["border"], color=BLUE_B)]
        # show this node's policy/world-model rollout as a decoded filmstrip
        panel = None
        strip_rel = self.nodes[nid].get("strip")
        if strip_rel:
            strip = ImageMobject(_abs(strip_rel)).set_width(7.0).to_edge(DOWN, buff=0.35)
            plab = Text("policy → world-model rollout", font_size=16, color=BLUE_B).next_to(
                strip, UP, buff=0.08)
            panel = Group(strip, plab)
            shows.append(FadeIn(panel))
        self.play(*shows, run_time=0.6)
        self.wait(0.3)
        outs = [FadeOut(rlabel)]
        if panel is not None:
            outs.append(FadeOut(panel))
        self.play(*outs, run_time=0.3)

    def _do_backprop(self, e):
        self._set_phase("backprop")
        path = e["path"]
        for nid in reversed(path[1:]):
            if nid in self.edge_mobs:
                self.play(ShowPassingFlash(
                    self.edge_mobs[nid].copy().set_color(ORANGE).set_stroke(width=6),
                    time_width=0.7), run_time=0.28)
        for nid in path:
            n, val = e["stats"][str(nid)]
            self._set_stat(nid, n, val)
        self.play(*[Indicate(self.node_mobs[nid]["border"], color=ORANGE, scale_factor=1.08)
                    for nid in path], run_time=0.35)

    def _highlight_plan(self):
        self.play(Transform(self.banner, Text("Plan", font_size=24, color=GOLD).to_corner(UL)),
                  run_time=0.25)
        plan = self.meta["plan_path"]
        gold_edges = VGroup(*[self.edge_mobs[nid] for nid in plan if nid in self.edge_mobs])
        borders = VGroup(*[self.node_mobs[nid]["border"] for nid in plan])
        self.play(gold_edges.animate.set_color(GOLD).set_stroke(width=6),
                  borders.animate.set_stroke(GOLD, 4), run_time=0.8)
        tag = Text("recommended plan (root → goal-ward)", font_size=20, color=GOLD).to_edge(DOWN)
        self.play(FadeIn(tag, shift=0.15 * UP), run_time=0.6)