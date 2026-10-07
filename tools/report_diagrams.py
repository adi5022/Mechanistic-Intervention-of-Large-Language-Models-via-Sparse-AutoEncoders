"""
Diagrams for the project report: system architecture, hybrid-sweep flow chart, gradient-descent editor flow, CounterFact data
pipeline and the two application wireframes. They contain no data. Text is wrapped and shrunk to fit its box by construction
(Canvas.fit), and a warning is printed if a box cannot hold its text even at the smallest allowed font.

Used by tools/make_report_figures.py.
"""
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE, ORANGE, GREEN, GRAY, DARK = "#2563eb", "#ea580c", "#16a34a", "#9ca3af", "#1f2937"
LBLUE, LORANGE, LGREEN, LGRAY, LYELLOW = "#dbeafe", "#ffedd5", "#dcfce7", "#f3f4f6", "#fefce8"
EDGE = "#6b7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300})


class Canvas:
    def __init__(self, w, h, figw):
        self.w, self.h, self.figw = w, h, figw
        self.fig = plt.figure(figsize=(figw, figw * h / w))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self.unit = figw * 72.0 / w                      # points per data unit
        self.warnings = []

    # ---- text fitting
    def _wrap(self, text, width_pt, fs, bold):
        cw = 0.60 * (1.08 if bold else 1.0) * fs          # conservative average character width (DejaVu Sans), points
        maxc = max(3, int(width_pt / cw))
        lines = []
        for para in text.split("\n"):
            lines += textwrap.wrap(para, maxc, break_long_words=False, break_on_hyphens=False) or [""]
        return lines, maxc

    def fit(self, text, w, h, fs, bold=False, minfs=6.0, pad=6.0):
        f = fs
        while True:
            lines, maxc = self._wrap(text, w * self.unit - 2 * pad, f, bold)
            too_wide = max(len(l) for l in lines) > maxc
            need = len(lines) * f * 1.28 + pad
            if (need <= h * self.unit and not too_wide) or f <= minfs:
                break
            f = round(f - 0.2, 2)
        if need > h * self.unit or too_wide:
            self.warnings.append(f"text does not fit ({f}pt): {text[:50]!r}")
        return "\n".join(lines), f

    # ---- primitives
    def box(self, x, y, w, h, text="", fc=LGRAY, ec=EDGE, fs=8.0, bold=False, lw=1.0, ls="-", title=None, tfs=None, color=DARK,
            align="center"):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12", fc=fc, ec=ec, lw=lw, ls=ls))
        top_used = 0.0
        if title:
            tfs = tfs or fs + 0.4
            tt, tf = self.fit(title, w, h * 0.45, tfs, bold=True)
            n = tt.count("\n") + 1
            th = n * tf * 1.28 / self.unit
            self.ax.text(x + w / 2, y + h - 0.06 - th / 2 - 0.02, tt, ha="center", va="center", fontsize=tf, fontweight="bold", color=color, linespacing=1.28)
            top_used = th + 0.1
        if text:
            bh = h - top_used
            tt, tf = self.fit(text, w, bh, fs, bold=bold)
            if align == "left":
                self.ax.text(x + 0.12, y + bh / 2, tt, ha="left", va="center", fontsize=tf, fontweight="bold" if bold else "normal", color=color, linespacing=1.28)
            else:
                self.ax.text(x + w / 2, y + bh / 2, tt, ha="center", va="center", fontsize=tf, fontweight="bold" if bold else "normal", color=color, linespacing=1.28)

    def arrow(self, x1, y1, x2, y2, color="#374151", lw=1.3, ls="-", rad=0.0):
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=lw, color=color, ls=ls,
                                          connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

    def poly(self, pts, color="#374151", lw=1.3, ls="-"):
        """Polyline whose last segment carries the arrow head."""
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt")
        self.arrow(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color, lw=lw, ls=ls)

    def tag(self, x, y, text, fs=7.6, color="#374151", ha="center"):
        self.ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center", style="italic")

    def frame(self, x, y, w, h, label, color):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.15", fc=color, ec="none", alpha=0.55))
        self.ax.text(x + 0.25, y + h - 0.3, label, fontsize=8.2, fontweight="bold", color="#374151", va="center")

    def save(self, out, name):
        path = os.path.join(out, name + ".png")
        self.fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(self.fig)
        print("wrote", path, ("   WARNINGS: " + "; ".join(self.warnings)) if self.warnings else "")


# ------------------------------------------------------------------------------------------------------------------------
def fig_arch(out):
    W, H = 14.4, 11.0
    c = Canvas(W, H, 7.2)
    # frames
    c.frame(0.2, 8.95, W - 0.4, 1.85, "PRESENTATION TIER", LBLUE)
    c.frame(0.2, 4.05, W - 0.4, 4.4, "INTERVENTION ENGINE TIER", LGREEN)
    c.frame(0.2, 0.15, W - 0.4, 3.45, "MODEL AND DATA TIER", LORANGE)
    # presentation tier
    c.ax.text(W / 2 + 1.2, 10.46, "experiment_app.py (Streamlit), seven tabs; command-line entry points run_batch.py and tools/", fontsize=7.8, ha="center", va="center", color="#374151")
    tabs = ["Hybrid mute and boost", "Prototype lab", "Batch: last vs all tokens", "Repair and SAE limit", "Session history and benchmarks",
            "Monosemanticity analysis", "Sequential vs batched proof"]
    tw, gap = 1.88, 0.11
    for i, t in enumerate(tabs):
        c.box(0.35 + i * (tw + gap), 9.1, tw, 0.95, t, fc="white", ec=BLUE, fs=7.2)
    # engine tier, row A
    eng = [("hooks.py", "delta-patch hooks: scale map, joint, per-row, additive"),
           ("editing.py", "clean context, candidate screening, causal ranking, safety filters"),
           ("batched_eval.py", "all candidates in one batched forward call"),
           ("hybrid_runner.py", "Hybrid sweep: cumulative steps, best-so-far, pool refill"),
           ("gradient_editing.py", "per-feature gradient descent, additive edit, generation")]
    ew, eg = 2.66, 0.1
    for i, (t, d) in enumerate(eng):
        c.box(0.35 + i * (ew + eg + 0.0), 5.95, ew, 1.85, d, fc="white", ec=GREEN, fs=7.2, title=t, tfs=7.4)
    # row B (dashed)
    studies = [("batch_runner, batch_analysis, paper_pack", "resumable, shardable batch studies; exact tests; result packs"),
               ("repair_diagnostics", "persistence trace, held edit, SAE-free bound"),
               ("strength_cache, strength_models, feature_models", "per-prompt cache, strength tables, learned-strength models")]
    sw_, sg = 4.38, 0.35
    for i, (t, d) in enumerate(studies):
        c.box(0.35 + i * (sw_ + sg), 4.2, sw_, 1.45, d, fc="white", ec=GREEN, ls=(0, (4, 2)), fs=7.0, title=t, tfs=7.2)
    # model and data tier
    md = [("GPT-2 Small", "TransformerLens HookedTransformer, 12 blocks, d = 768"),
          ("SAE gpt2-small-res-jb", "SAELens, 24,576 latents per layer at hook_resid_pre"),
          ("Fact data", "ROME known facts (45); CounterFact hard set (16,360 prompts) and split"),
          ("Cache and tables", "1,450 per-prompt files and 1,450 strength tables"),
          ("External services", "Neuronpedia lookups; Groq explanations (optional)")]
    for i, (t, d) in enumerate(md):
        c.box(0.35 + i * (ew + eg), 1.5, ew, 1.6, d, fc="white", ec=ORANGE, fs=7.2, title=t, tfs=7.4)
    c.box(0.35, 0.3, W - 0.7, 0.9, "Device layer: CUDA, Apple MPS or CPU, chosen by sae_utils.get_default_device() and device_utils.py; synchronisation and cache calls are device aware",
          fc="#fff7ed", ec=ORANGE, fs=7.4)
    # tier-to-tier arrows
    for (y1, y2) in [(8.95, 8.45), (4.05, 3.6)]:
        c.arrow(6.7, y1, 6.7, y2)
        c.arrow(7.7, y2, 7.7, y1, color="#9ca3af")
        c.tag(5.9, (y1 + y2) / 2, "calls", ha="right", fs=7.2)
        c.tag(8.5, (y1 + y2) / 2, "results", ha="left", fs=7.2, color="#6b7280")
    c.save(out, "fig_arch_v2")


def fig_algo(out):
    W, H = 12.6, 14.6
    c = Canvas(W, H, 6.2)
    x0, bw = 1.6, 6.9
    cx = x0 + bw / 2
    rx, rw = 8.95, 3.45

    def row(y, h, text, fc="white", ec=EDGE, w=bw, x=x0):
        c.box(x, y, w, h, text, fc=fc, ec=ec, fs=7.8)

    row(13.4, 0.8, "Prompt and target token", fc=LGRAY)
    row(12.0, 1.0, "Clean pass: rank and probability of the target; blocker = the top-1 token; residual stream cached")
    row(10.6, 1.0, "Candidate screening: top N active features by activation (last token, or any prompt position except BOS)")
    row(9.0, 1.3, "Causal ranking, one batched forward call. Mute list: features whose removal lowers the blocker most. Boost list: features whose removal lowers the target most", fc=LBLUE, ec=BLUE)
    row(7.1, 1.5, "Safety filter, each feature tested alone: reject it if the target gets worse (rank, or probability by more than 1e-6). A feature safe on both sides is kept on one side only", fc=LBLUE, ec=BLUE)
    row(5.5, 1.3, "Cumulative sweep, step k: the first k mutes and first k boosts in ONE joint delta-patch pass; record rank and probability; keep the best step", fc=LGREEN, ec=GREEN)
    c.box(cx - 1.8, 4.3, 3.6, 0.8, "Target at rank 1?", fc=LYELLOW, ec="#ca8a04", fs=8)
    c.box(cx - 1.8, 3.0, 3.6, 0.8, "Features left in the pools?", fc=LYELLOW, ec="#ca8a04", fs=8)
    row(1.1, 1.3, "Pool refill: keep every applied feature; one fresh pass gives the steered baseline and a new blocker; re-rank and re-filter the remaining active features (rejected ones are retried)", fc=LORANGE, ec=ORANGE)
    # right-hand boxes
    c.box(rx, 12.0, rw, 1.0, "Target already rank 1: stop, nothing to edit", fc=LGREEN, ec=GREEN, fs=7.4)
    c.box(rx, 4.2, rw, 1.0, "Stop: report the best step (not the last step)", fc=LGREEN, ec=GREEN, fs=7.4)
    c.box(rx, 0.95, rw, 1.6, "Stop when no active feature is left, no safe candidate remains, or the step limit is reached", fc=LGRAY, ec=EDGE, fs=7.4)
    # straight arrows down the main column
    ys = [(13.4, 13.0), (12.0, 11.6), (10.6, 10.3), (9.0, 8.6), (7.1, 6.8), (5.5, 5.1), (4.3, 3.8), (3.0, 2.4)]
    for y1, y2 in ys:
        c.arrow(cx, y1, cx, y2)
    c.tag(cx + 0.2, 4.05, "no", ha="left", color="#b45309")
    c.tag(cx + 0.2, 2.7, "no", ha="left", color="#b45309")
    # side arrows
    c.arrow(x0 + bw, 12.5, rx, 12.5, color=GREEN)
    c.arrow(cx + 1.8, 4.7, rx, 4.7, color=GREEN)
    c.tag((cx + 1.8 + rx) / 2, 4.95, "yes", color=GREEN)
    c.arrow(x0 + bw, 1.75, rx, 1.75, color="#6b7280")
    # loop: pools left -> next sweep step (lane x = 0.95)
    c.poly([(cx - 1.8, 3.4), (0.95, 3.4), (0.95, 6.15), (x0, 6.15)], color="#374151")
    c.tag(0.75, 4.6, "yes: step k + 1", fs=7.0)
    c.ax.texts[-1].set_rotation(90)
    # loop: refill -> next round (lane x = 0.4)
    c.poly([(x0, 1.75), (0.4, 1.75), (0.4, 9.65), (x0, 9.65)], color=ORANGE, ls="--")
    c.tag(0.2, 6.0, "next round (re-rank, re-filter)", fs=7.0, color=ORANGE)
    c.ax.texts[-1].set_rotation(90)
    c.save(out, "fig_algo_v2")


def fig_gdflow(out):
    W, H = 14.2, 7.9
    c = Canvas(W, H, 7.2)
    bw, gap = 2.55, 0.3
    xs = [0.1 + i * (bw + gap) for i in range(5)]
    top = [("1. Clean pass", "Residual stream of every position at layer 8; blocker = the top-1 token", LGRAY, EDGE),
           ("2. Candidates", "Top N = 200 features by maximum activation over the prompt positions (BOS excluded); activations A", LGRAY, EDGE),
           ("3. Free numbers", "One z_k per candidate, starting at no edit. Multiplier m_k = 1 + a_k, a_k = -1 + 3 sigmoid(z_k): 0 to 3 times", LBLUE, BLUE),
           ("4. Edit", "At every position: residual + sum_k a_k A(p,k) W_dec[k] (delta patch; reconstruction error untouched)", LBLUE, BLUE),
           ("5. Forward", "Blocks 8 to 11 run from the edited state; logits at the last position", LGREEN, GREEN)]
    for x, (t, d, fc, ec) in zip(xs, top):
        c.box(x, 5.45, bw, 2.1, d, fc=fc, ec=ec, fs=7.2, title=t, tfs=7.8)
    for x in xs[:-1]:
        c.arrow(x + bw, 6.5, x + bw + gap, 6.5)
    c.box(7.5, 2.95, 6.6, 1.95, "Rank term max(0, 0.3 - (target logit - best other logit)), plus 3.0 x KL(clean || edited) over the other tokens (only once the target leads), plus 0.005 x sum of |a_k| w_k (w_k = activity share of the feature)",
          fc=LORANGE, ec=ORANGE, fs=7.2, title="6. Loss", tfs=7.8)
    c.box(0.1, 2.95, 6.9, 1.95, "Adam on z only (learning rate 0.1, 100 steps). The gradient flows through blocks 8 to 11; no GPT-2 weight is changed or even given a gradient.",
          fc="white", ec=ORANGE, fs=7.2, title="7. Update", tfs=7.8)
    c.arrow(xs[4] + bw / 2, 5.45, 11.0, 4.9)
    c.arrow(7.5, 3.9, 7.0, 3.9, color=ORANGE)
    c.arrow(6.2, 4.9, 6.2, 5.45, color=ORANGE, ls="--")
    c.tag(6.45, 5.18, "new z", fs=7.0, color=ORANGE, ha="left")
    c.box(0.1, 0.2, 7.0, 1.95, "Keep the best iterate: a step where the target is rank 1 with the lowest loss (otherwise the lowest loss). Re-check it through the real hook in a full GPT-2 pass; report rank, probability, KL and edit size.",
          fc=LGREEN, ec=GREEN, fs=7.2, title="8. Result", tfs=7.8)
    c.arrow(3.55, 2.95, 3.55, 2.15, color=GREEN)
    c.box(7.5, 0.2, 6.6, 1.95, "Amounts c_k >= 0 added at the last position to features that are silent there, chosen by the gradient of the target margin, starting at exactly 0, capped by the loudest real feature; extra loss 0.005 x sum c_k / cap.",
          fc="white", ec=EDGE, ls=(0, (4, 2)), fs=7.2, title="Optional additive extension (exploratory)", tfs=7.6)
    c.save(out, "fig_gd_flow")


def fig_pipeline(out):
    W, H = 14.2, 8.4
    c = Canvas(W, H, 7.0)
    bw = 4.3
    xs = [0.1, 4.95, 9.8]
    c.box(xs[0], 6.35, bw, 1.8, "21,919 records: a prompt template, a subject, the true answer, 34 relations", fc=LGRAY, ec=EDGE, fs=7.4, title="CounterFact", tfs=8)
    c.box(xs[1], 6.35, bw, 1.8, "Keep records whose true answer is one GPT-2 token (21,913 usable). One clean pass per prompt, in batches of 64, gives the rank of the answer", fc=LBLUE, ec=BLUE, fs=7.2, title="Rank the answer", tfs=8)
    c.box(xs[2], 6.35, bw, 1.8, "1,817 already rank 1 and 3,736 above rank 1000 are dropped. Kept: 16,360 prompts at rank 2 to 1000 (15,594 subjects)", fc=LBLUE, ec=BLUE, fs=7.2, title="Hard set", tfs=8)
    c.arrow(xs[0] + bw, 7.25, xs[1], 7.25)
    c.arrow(xs[1] + bw, 7.25, xs[2], 7.25)
    c.box(0.1, 4.55, 14.0, 1.2, "A subject appears in one set only. Five whole relations are held out of training and validation: religion, sport, instrument, manufacturer, twinned city. Training order is balanced over relation and starting rank.",
          fc=LBLUE, ec=BLUE, fs=7.4, title="Safe split", tfs=8)
    c.arrow(xs[2] + bw / 2, 6.35, xs[2] + bw / 2, 5.75)
    sw_, sg = 3.2, 0.4
    sx = [0.1 + i * (sw_ + sg) for i in range(4)]
    sets = [("Training pool", "10,292 prompts, the first 1,000 used"), ("Validation", "150 prompts"), ("Test, seen relations", "150 prompts, new subjects"), ("Test, unseen relations", "150 prompts, held-out relations")]
    for x, (t, d) in zip(sx, sets):
        c.box(x, 2.7, sw_, 1.35, d, fc=LGREEN, ec=GREEN, fs=7.4, title=t, tfs=7.8)
        c.arrow(x + sw_ / 2, 4.55, x + sw_ / 2, 4.05)
        c.arrow(x + sw_ / 2, 2.7, x + sw_ / 2, 2.25, color=ORANGE)
    c.box(0.1, 1.3, 14.0, 0.95, "Per-prompt cache and strength tables for the 1,450 prompts used (1,000 + 150 + 150 + 150)", fc=LORANGE, ec=ORANGE, fs=7.6)
    c.box(0.1, 0.1, 14.0, 0.85, "Studies: headroom (all 1,450); learned strengths (train 1,000, test 300); gradient descent against the sweep (the 300 test prompts)", fc="white", ec=EDGE, fs=7.4)
    c.arrow(7.1, 1.3, 7.1, 0.95, color=ORANGE)
    c.save(out, "fig_pipeline")


def fig_ui(out):
    W, H = 12.0, 8.4
    c = Canvas(W, H, 6.4)
    c.box(0.1, 0.1, 2.75, 8.2, "", fc=LGRAY, ec=GRAY)
    c.ax.text(1.475, 7.95, "Sidebar", fontsize=8, fontweight="bold", ha="center", va="center")
    for i, t in enumerate(["Compute device: CUDA, MPS or CPU", "Select model layer (0 to 11)", "Enable AI explanations (Groq)"]):
        c.box(0.25, 6.65 - i * 1.15, 2.45, 0.9, t, fc="white", ec=GRAY, fs=7.2)
    c.box(3.0, 7.45, 8.9, 0.85, "Seven tabs: Hybrid mute and boost | Monosemanticity | Session history | Sequential vs batched | Batch | Repair and SAE limit | Prototype lab",
          fc=LBLUE, ec=BLUE, fs=7.2)
    c.box(3.0, 6.45, 8.9, 0.75, "Editing method:   ( ) fixed mute / boost sweep     ( ) gradient descent (one multiplier per feature)", fc="white", ec=GRAY, fs=7.4)
    c.box(3.0, 3.4, 4.35, 2.8, "Prompt\nTarget completion\nTop N candidate features (maximum)\nCandidate source: ( ) all prompt positions  ( ) last token only", fc="white", ec=GRAY, fs=7.4)
    c.box(7.55, 3.4, 4.35, 2.8, "Sweep: mute and boost strength, batch sizes, cumulative sweep, pool refill, maximum rounds, safety filter, GPU-batched evaluation.\nGradient descent: steps, learning rate, rank-1 margin, side-effect weight, edit-size weight", fc="white", ec=GRAY, fs=7.0)
    c.box(3.0, 2.45, 8.9, 0.7, "[ Run ]   progress bar and live tracker: round, step, rank, best rank, applied features", fc=LGREEN, ec=GREEN, fs=7.4)
    c.box(3.0, 0.1, 8.9, 2.1, "Results: best result found, rank-progression chart, run summary, per-round pools and rejections with Neuronpedia links, applied-feature ledger, stopwatch (model time and wall-clock time), optional AI explanation. The run is saved to the session history.",
          fc="white", ec=GRAY, fs=7.2)
    c.save(out, "fig_ui_hybrid")

    c = Canvas(W, H, 6.4)
    c.box(0.1, 7.5, 11.8, 0.8, "Prototype lab tab", fc=LBLUE, ec=BLUE, fs=8.4, bold=True)
    c.box(0.1, 4.55, 5.8, 2.7, "Prompt, target, Top N, candidate source\n[ ] Also allow adding to silent features (off by default, with a warning)\n[ ] Always penalise side effects\nTokens to generate after the prompt\n[x] Keep the edit on while generating", fc="white", ec=GRAY, fs=7.2)
    c.box(6.1, 4.55, 5.8, 2.7, "Gradient descent settings: steps, learning rate, rank-1 margin, side-effect weight, edit-size weight.\nAdditive settings (when on): silent candidates M, cap, sparsity weight, additive learning rate", fc="white", ec=GRAY, fs=7.2)
    c.box(0.1, 3.7, 11.8, 0.65, "[ Run ]   baseline top-1, target probability and rank; progress bar", fc=LGREEN, ec=GREEN, fs=7.4)
    c.box(0.1, 2.7, 11.8, 0.8, "Result, checked through the real hook: target rank, probability, new top-1, KL; muted, boosted and switched-on counts; edit size", fc="white", ec=GRAY, fs=7.2)
    c.box(0.1, 1.5, 5.8, 1.0, "Baseline against edited text: greedy, continues past the target, target highlighted", fc=LORANGE, ec=ORANGE, fs=7.2)
    c.box(6.1, 1.5, 5.8, 1.0, "Follow-up prompts: the same edit on other prompts (KL, top word with and without the edit)", fc=LORANGE, ec=ORANGE, fs=7.2)
    c.box(0.1, 0.1, 11.8, 1.2, "Rank-progression chart; switched-on silent features and per-feature multipliers with Neuronpedia links (shown last, because the lookups are slow); session-history record", fc="white", ec=GRAY, fs=7.2)
    c.save(out, "fig_ui_proto")
