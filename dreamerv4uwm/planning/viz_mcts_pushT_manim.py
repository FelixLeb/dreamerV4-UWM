"""Simple, teachable manimgl visualization of MCTS on pushT.

Replays a trace from ``make_mcts_trace_pushT_v2.py`` and shows *only* what you
need to understand the algorithm:

  * the search tree growing — every node is a decoded world-model **state**,
    its border tinted by how good that state is (red = bad, green = the red T
    is centered);
  * the **four steps** of one MCTS round, named in plain English and taught
    slowly for the first two rounds, then fast-forwarded;
  * the recommended **plan** at the end.

No dashboards, gauges, or reward curves — the branchable/collapsed contrast is
read straight off the tree: branchable children look *different* and turn green;
collapsed children look *identical* and stay amber.

Render (pick a trace via MCTS_TRACE), headless:
    MCTS_TRACE=mcts_trace_branchable.json \
        xvfb-run -a manimgl viz_mcts_pushT_manim_v2.py MCTSTreeScene -w --hd --file_name mcts_branchable
    MCTS_TRACE=mcts_trace_collapsed.json  \
        xvfb-run -a manimgl viz_mcts_pushT_manim_v2.py MCTSTreeScene -w --hd --file_name mcts_collapsed
"""
import json
import os

import numpy as np
from colour import Color
from manimlib import *
from manimlib.mobject.types.image_mobject import ImageMobject

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.environ.get("MCTS_TRACE") or os.path.join(HERE, "mcts_trace_branchable.json")
if not os.path.isabs(TRACE):
    TRACE = os.path.join(HERE, TRACE)

INK = "#ECECEC"
MUTE = "#9AA0A6"
GOLD_ = "#F4C430"

# the four steps: (colour, ①-badge, name, one-line plain-English description)
STEPS = {
    "select":   ("#F2C14E", "1", "SELECT",   "follow the most promising branch down to a leaf"),
    "expand":   ("#5FBB6E", "2", "EXPAND",   "imagine a few next moves — each opens a new branch"),
    "simulate": ("#4EA8DE", "3", "SIMULATE", "roll the world-model forward and score the result"),
    "backprop": ("#E8853B", "4", "BACK-UP",  "add the score to V_total and n, back up the path"),
}
TEACH_ROUNDS = 2                      # rounds shown slowly with full narration

# tree occupies most of the frame now that the chrome is gone
TREE_L, TREE_R = -5.5, 5.2
TOP_ROW, VPITCH = 1.55, 1.4
BANNER_Y = 2.55                      # step banner sits in a clear band below the title


def reward_color(v):
    v = float(np.clip(v, 0.0, 1.0))
    lo, mid, hi = Color("#D64550"), Color("#E8C547"), Color("#3FB47F")
    return interpolate_color(lo, mid, v / 0.5) if v < 0.5 else interpolate_color(mid, hi, (v - 0.5) / 0.5)


def _abs(rel):
    return os.path.join(HERE, rel)


class MCTSTreeScene(Scene):
    def construct(self):
        tr = json.load(open(TRACE))
        self.nodes = {int(k): v for k, v in tr["nodes"].items()}
        self.events = tr["events"]
        self.meta = tr["meta"]
        self.root_id = self.meta["root"]

        xs = [n["x"] for n in self.nodes.values()]
        self.xmin, self.xmax = min(xs), max(xs)
        self.node_h = float(np.clip(0.9 * (TREE_R - TREE_L) / max(self.xmax - self.xmin, 1.0),
                                    0.42, 0.86))
        self.node_h = min(self.node_h, 0.72 * VPITCH)
        self.lab_fs = 14 if self.node_h > 0.55 else 12

        self.node_mobs, self.edge_mobs = {}, {}
        self.round = 0                    # 1-based round counter
        self.fast = False

        self._intro()
        self.play(FadeIn(self._make_node(self.root_id, root=True)), run_time=0.6)

        for e in self.events:
            getattr(self, "_do_" + e["type"], lambda _e: None)(e)

        self._outro()

    # -------------------------------------------------------------- chrome
    def _intro(self):
        title = Text("Monte-Carlo Tree Search", font_size=30, color=INK).to_corner(UL, buff=0.35)
        sub = Text("planning with a world model  ·  pushT", font_size=16, color=MUTE)
        sub.next_to(title, DOWN, buff=0.1).align_to(title, LEFT)
        self.add(title, sub)

        # what "good" means: the start frame + the goal, top-right
        start = ImageMobject(_abs(self.meta["start_img"])).set_height(1.0)
        sb = Square(side_length=1.0).set_stroke(MUTE, 2.5).move_to(start)
        scap = Text("start", font_size=15, color=MUTE).next_to(sb, DOWN, buff=0.06)
        self.start_card = Group(start, sb, scap).to_corner(UR, buff=0.35)
        goal = Text("goal: push the red T to the centre", font_size=17, color=INK)
        goal.next_to(self.start_card, LEFT, buff=0.5).shift(UP * 0.05)
        self.add(self.start_card, goal)

        # colour key for the node borders (one small line)
        gk = Square(side_length=0.2).set_stroke(reward_color(1.0), 4).move_to([-6.1, -3.55, 0])
        gt = Text("good", font_size=14, color=INK).next_to(gk, RIGHT, buff=0.08)
        rk = Square(side_length=0.2).set_stroke(reward_color(0.0), 4).next_to(gt, RIGHT, buff=0.35)
        rt = Text("bad", font_size=14, color=INK).next_to(rk, RIGHT, buff=0.08)
        note = Text("border = how good the state is", font_size=13, color=MUTE)
        key = Group(gk, gt, rk, rt)
        note.next_to(key, UP, buff=0.1).align_to(key, LEFT)
        self.add(key, note)

        # invisible placeholder so _set_banner can always cross-fade
        self.banner = Dot(radius=0.01).set_opacity(0.0).move_to([0, BANNER_Y, 0])
        self.add(self.banner)

        card = Text("Build a tree of possible futures, then keep the best path.",
                    font_size=24, color=INK).move_to([0, 0.2, 0])
        sub2 = Text("Each round repeats four steps:", font_size=19, color=MUTE).next_to(card, DOWN, buff=0.3)
        steps = VGroup(*[
            self._chip(s) for s in ["select", "expand", "simulate", "backprop"]
        ]).arrange(RIGHT, buff=0.3).next_to(sub2, DOWN, buff=0.35)
        self.play(FadeIn(card), run_time=0.7)
        self.play(FadeIn(sub2), LaggedStartMap(FadeIn, steps, lag_ratio=0.3), run_time=1.4)
        self.wait(1.4)
        self.play(FadeOut(card), FadeOut(sub2), FadeOut(steps), run_time=0.5)

    def _chip(self, key):
        col, num, name, _ = STEPS[key]
        badge = Text(num, font_size=17, color=BLACK)
        circ = Circle(radius=0.18).set_fill(col, 1).set_stroke(width=0).move_to(badge)
        lab = Text(name, font_size=17, color=col).next_to(circ, RIGHT, buff=0.12)
        return VGroup(circ, badge, lab)

    def _banner_mob(self, key):
        col, num, name, desc = STEPS[key]
        badge = Text(num, font_size=22, color=BLACK)
        circ = Circle(radius=0.24).set_fill(col, 1).set_stroke(width=0).move_to(badge)
        node = Group(circ, badge)                       # number sits on the disc
        head = Text(name, font_size=26, color=col)
        body = Text(desc, font_size=19, color=INK)
        return Group(node, head, body).arrange(RIGHT, buff=0.2).move_to([0, BANNER_Y, 0])

    def _set_banner(self, key):
        new = self._banner_mob(key)
        self.play(FadeOut(self.banner, run_time=0.12))
        self.banner = new
        self.play(FadeIn(self.banner, run_time=0.15))

    def _fast_banner(self):
        head = Text("SEARCHING", font_size=24, color=INK)
        body = Text("the four steps repeat…", font_size=19, color=MUTE)
        g = Group(head, body).arrange(RIGHT, buff=0.3).move_to([0, BANNER_Y, 0])
        self.play(FadeOut(self.banner, run_time=0.12))
        self.banner = g
        self.play(FadeIn(self.banner, run_time=0.15))

    # -------------------------------------------------------------- geometry
    def _pos(self, nid):
        v = self.nodes[nid]
        sx = TREE_L + (v["x"] - self.xmin) / max(self.xmax - self.xmin, 1e-6) * (TREE_R - TREE_L)
        return np.array([sx, TOP_ROW + v["y"] * VPITCH, 0.0])

    def _make_node(self, nid, root=False):
        p = self._pos(nid)
        meta = self.nodes[nid]
        img = ImageMobject(_abs(meta["img"])).set_height(self.node_h).move_to(p)
        col = "#C9CDD2" if root else reward_color(meta["edge_val"])
        border = Square(side_length=self.node_h).set_stroke(col, 3.5).set_fill(opacity=0).move_to(p)
        stat = self._stat_mob(0, 0.0).next_to(border, DOWN, buff=0.05)
        self.node_mobs[nid] = dict(img=img, border=border, base=col, stat=stat)
        parts = [img, border, stat]
        if root:
            parts.append(Text("start", font_size=16, color=INK).next_to(border, UP, buff=0.08))
        return Group(*parts)

    def _stat_mob(self, n, v_total):
        """Two-line node label: visit count n over accumulated value V_total."""
        l1 = Text("n=%d" % n, font_size=self.lab_fs, color=MUTE)
        l2 = Text("V=%.1f" % v_total, font_size=self.lab_fs, color=MUTE)
        return VGroup(l1, l2).arrange(DOWN, buff=0.02)

    def _set_stat(self, nid, n, v_total):
        new = self._stat_mob(n, v_total).next_to(self.node_mobs[nid]["border"], DOWN, buff=0.05)
        self.node_mobs[nid]["stat"].become(new)

    def _make_edge(self, child_id):
        parent = self.nodes[child_id]["parent"]
        p0, p1 = self._pos(parent), self._pos(child_id)
        line = Line(p0, p1, stroke_width=2.5, color="#565C63")
        line.set_length(max(0.05, line.get_length() - self.node_h))
        line.move_to((p0 + p1) / 2)
        self.edge_mobs[child_id] = line
        return line

    def _restore(self, ids):
        self.play(*[self.node_mobs[i]["border"].animate.set_stroke(self.node_mobs[i]["base"], 3.5)
                    for i in ids], run_time=0.15)

    # -------------------------------------------------------------- events
    def _do_root(self, e):
        pass

    def _do_expand(self, e):
        is_root = (e["node"] == self.root_id and self.round == 0)
        if is_root:
            self._set_banner("expand")
        elif not self.fast:
            self._set_banner("expand")
        edges = [ShowCreation(self._make_edge(k)) for k in e["children"]]
        nodes = [FadeIn(self._make_node(k)) for k in e["children"]]
        rt_e, rt_n = (0.4, 0.7) if not self.fast else (0.22, 0.34)
        self.play(*edges, run_time=rt_e)
        self.play(*nodes, run_time=rt_n)
        if is_root:
            # linger on the first branching so the idea lands
            bs = VGroup(*[self.node_mobs[k]["border"] for k in e["children"]])
            self.play(bs.animate.set_stroke(width=6), run_time=0.3)
            self.play(bs.animate.set_stroke(width=3.5), run_time=0.4)
            self.wait(0.4)

    def _do_select(self, e):
        self.round += 1
        self.fast = self.round > TEACH_ROUNDS
        if self.fast and self.round == TEACH_ROUNDS + 1:
            self._fast_banner()
        elif not self.fast:
            self._set_banner("select")
        path = e["path"]
        col = STEPS["select"][0]
        borders = VGroup(*[self.node_mobs[nid]["border"] for nid in path])
        edges = VGroup(*[self.edge_mobs[nid] for nid in path[1:] if nid in self.edge_mobs])
        self.play(borders.animate.set_stroke(col, 6), run_time=0.2 if not self.fast else 0.12)
        if len(edges) > 0:
            self.play(ShowPassingFlash(edges.copy().set_color(col).set_stroke(width=7),
                                       time_width=0.6),
                      run_time=0.5 if not self.fast else 0.28)
        self._restore(path)

    def _score_tag(self, R, col):
        t = Text("score = %.1f" % R, font_size=15, color=col)
        bg = SurroundingRectangle(t, buff=0.07).set_fill("#20242A", 0.92).set_stroke(col, 1.5)
        return Group(bg, t)

    def _do_simulate(self, e):
        nid = e["node"]
        col = STEPS["simulate"][0]
        border = self.node_mobs[nid]["border"]
        tag = self._score_tag(e["R"], col).next_to(border, UP, buff=0.1)
        if not self.fast:
            self._set_banner("simulate")
            self.play(Indicate(border, color=col, scale_factor=1.15),
                      FadeIn(tag, shift=0.1 * UP), run_time=0.55)
            self.wait(0.35)
            self.play(FadeOut(tag), run_time=0.2)
        else:
            self.play(Indicate(border, color=col, scale_factor=1.12),
                      FadeIn(tag), run_time=0.24)
            self.play(FadeOut(tag), run_time=0.12)

    def _do_backprop(self, e):
        path = e["path"]
        col = STEPS["backprop"][0]
        if not self.fast:
            self._set_banner("backprop")
            for nid in reversed(path[1:]):
                if nid in self.edge_mobs:
                    self.play(ShowPassingFlash(
                        self.edge_mobs[nid].copy().set_color(col).set_stroke(width=7),
                        time_width=0.75), run_time=0.26)
            self._update_stats(e)
            self.play(*[Indicate(self.node_mobs[nid]["border"], color=col, scale_factor=1.06)
                        for nid in path], run_time=0.35)
        else:
            flashes = VGroup(*[self.edge_mobs[nid] for nid in path[1:] if nid in self.edge_mobs])
            if len(flashes):
                self.play(ShowPassingFlash(flashes.copy().set_color(col).set_stroke(width=6),
                                           time_width=0.7), run_time=0.22)
            self._update_stats(e)

    def _update_stats(self, e):
        # trace stores [n_visit, mean value]; V_total = n * mean
        for nid in e["path"]:
            n, mean = e["stats"][str(nid)]
            self._set_stat(nid, n, n * mean)

    # -------------------------------------------------------------- outro
    def _outro(self):
        # banner -> "PLAN"
        g = Text("PLAN — the best path the search found", font_size=24, color=GOLD_).move_to([0, BANNER_Y, 0])
        self.play(FadeOut(self.banner, run_time=0.15))
        self.banner = g
        self.play(FadeIn(self.banner, run_time=0.2))

        plan = self.meta["plan_path"]
        gold_edges = VGroup(*[self.edge_mobs[nid] for nid in plan if nid in self.edge_mobs])
        borders = VGroup(*[self.node_mobs[nid]["border"] for nid in plan])
        root_b = self.node_mobs[self.root_id]["border"]
        anims = [borders.animate.set_stroke(GOLD_, 6), root_b.animate.set_stroke(GOLD_, 6)]
        if len(gold_edges):
            anims.insert(0, gold_edges.animate.set_color(GOLD_).set_stroke(width=7))
        # fade the off-plan nodes (image, border, n/V label) so the plan stands out
        off = [i for i in self.node_mobs if i not in plan and i != self.root_id]
        dim = [self.node_mobs[i]["img"].animate.set_opacity(0.3) for i in off]
        dim += [self.node_mobs[i]["border"].animate.set_stroke(opacity=0.3) for i in off]
        dim += [self.node_mobs[i]["stat"].animate.set_opacity(0.3) for i in off]
        self.play(*anims, *dim, run_time=0.9)

        # before -> after
        best = ImageMobject(_abs(self.meta["best_img"])).set_height(1.0)
        bb = Square(side_length=1.0).set_stroke(GOLD_, 3).move_to(best)
        bcap = Text("result", font_size=15, color=GOLD_).next_to(bb, DOWN, buff=0.06)
        after = Group(best, bb, bcap).next_to(self.start_card, DOWN, buff=0.5)
        arrow = Arrow(self.start_card.get_bottom() + DOWN * 0.04, after.get_top() + UP * 0.04,
                      buff=0.05, color=MUTE, stroke_width=3)
        s, b = self.meta["start_reward"], self.meta["best_reward"]
        verb = "the T reaches the centre" if b > s + 0.15 else "the T barely moves — no real plan"
        line = Text("reward  %.2f  →  %.2f     (%s)" % (s, b, verb),
                    font_size=18, color=INK).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(after), FadeIn(arrow), run_time=0.6)
        self.play(FadeIn(line, shift=0.1 * UP), run_time=0.6)
        self.wait(2.5)
