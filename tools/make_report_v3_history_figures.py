"""
Figures for the history chapter of the report (docs/report_v2, rewritten 2026-10-11).

The early experiments (research journal entries 2, 3, 5, 8, 12, 13, 15) were run on one prompt or on 22 facts and were never
stored as one data pack; their numbers are the ones printed in those journal entries, copied below with the entry named next to
each block. The gradient-descent bars are read from docs/Research_Journal/images/e29_figure_data.json (the committed data of
entry 29). Later transfer figures are drawn by tools/make_report_v2_figures.py from the committed pack.

Writes docs/report_v2/figures/h_stages.png, h_early.png, h_speed.png, h_additive.png, h_gd_bars.png, h_gd_flow.png.
Run: .venv\\Scripts\\python.exe tools/make_report_v3_history_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report_v2" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, LBLUE, ORANGE, GREEN, GRAY, INK = "#2563eb", "#93c5fd", "#ea580c", "#16a34a", "#9ca3af", "#1f2937"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------------------------------------ evolution timeline
def stages():
    fig, ax = plt.subplots(figsize=(6.4, 6.1))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(-0.7, 12.45)
    ax.axis("off")
    # (title, result, colour, journal entries)
    items = [
        ("1  First prototype", "Eiffel Tower fixed;\n22 of 45 facts testable", GRAY, "J1"),
        ("2  Single ablation", "helpers 1, competitor 3\nof 22 facts", ORANGE, "J2-3"),
        ("3  Muting several", "one prompt: rank 8 to 6;\nsupport is spread out", ORANGE, "J4-8"),
        ("4  Mute and boost", "rank 8 to 1 on the\nprobe prompt", BLUE, "J5"),
        ("5  Layers and speed", "layers 6 to 9 best;\n186 to 6 model calls", BLUE, "J9-15"),
        ("6  Reliable sweep", "all positions + refill:\n10 of 13 vs 4 of 13", BLUE, "J16-21"),
        ("7  Limits of the edit", "no repair seen; a free\nedit reaches more", ORANGE, "J22"),
        ("8  Scale: CounterFact", "16,360 prompts;\nheadroom 80% / 41%", BLUE, "J23-24"),
        ("9  Learned strengths", "better fixed pair;\nno gain per prompt", ORANGE, "J25-27"),
        ("10  Gradient descent", "271 / 161 / 94 of 300;\n5 to 6.5 times faster", GREEN, "J28-29"),
        ("11  Additive edit", "20 of 20 random words\nat rank 1: weak test", ORANGE, "J30"),
        ("12  ROME reference", "official code runs; our edit\nscored with ROME's PS, NS", BLUE, "ROME log"),
        ("13  Applied: transfer", "carry an edit from\nSmall to Medium", GREEN, "J31-37"),
    ]
    w, h = 3.7, 1.95
    for n, (title, res, col, j) in enumerate(items):
        r, i = divmod(n, 3)
        x = 0.1 + i * 3.95
        y = 12.45 - (r + 1) * 2.45 + 0.2
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=col, lw=1.6))
        ax.text(x + 0.14, y + h - 0.2, title, fontsize=7.8, fontweight="bold", va="top", color=INK)
        ax.text(x + 0.14, y + h - 0.8, res, fontsize=7.2, va="top", color=INK, linespacing=1.3)
        ax.text(x + w - 0.12, y + 0.1, j, fontsize=6.4, ha="right", va="bottom", color=col)
        if i < 2 and n < len(items) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, y + h / 2), (x + 3.95 - 0.02, y + h / 2), arrowstyle="-|>", mutation_scale=7, color=INK, lw=0.9))
    for k, (col, lab) in enumerate([(GRAY, "start"), (ORANGE, "limiting result"), (BLUE, "adopted"), (GREEN, "current line")]):
        x0 = 0.1 + k * 3.0
        ax.add_patch(FancyBboxPatch((x0, -0.5), 0.3, 0.24, boxstyle="round,pad=0.01", fc="white", ec=col, lw=1.6))
        ax.text(x0 + 0.42, -0.38, lab, va="center", fontsize=6.8, color=INK)
    save(fig, "h_stages.png")


# ------------------------------------------------------------------------------------------------ early results
def early():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw={"width_ratios": [1, 2.1]})
    # Entries 2 and 3: 22 suppressed ROME facts
    names, vals = ["mute the\ntarget's helpers", "mute the\ncompetitor's drivers"], [1, 3]
    bars = a.bar(names, vals, color=[ORANGE, BLUE], width=0.55)
    for bar, v in zip(bars, vals):
        a.text(bar.get_x() + bar.get_width() / 2, v + 0.12, f"{v} of 22", ha="center", fontsize=8)
    a.set_ylim(0, 5)
    a.set_ylabel("facts brought to rank 1")
    a.set_title("(a) Single-feature ablation", fontsize=8.5, loc="left")
    a.tick_params(axis="x", labelsize=6.8)
    # Entries 4, 6, 8, 5: probe prompt "The location of MIT is in" -> Cambridge, clean rank 8
    steps = ["clean", "weighted\nmuting", "pooled\nmuting", "top-3\nmuting", "M3\nB1", "M3\nB2", "M3\nB3", "M3\nB4"]
    ranks = [8, 8, 8, 6, 4, 3, 1, 2]
    x = list(range(len(steps)))
    b.plot(x, ranks, color=BLUE, lw=1.6, marker="o", ms=5)
    b.plot([x[6]], [ranks[6]], marker="o", ms=8, color=GREEN)
    b.axhline(1, color=GREEN, ls="--", lw=0.8)
    for xi, r in zip(x, ranks):
        b.text(xi, r - 0.55, str(r), ha="center", fontsize=7.5, color=INK)
    b.set_ylim(9, 0)
    b.set_yticks([1, 2, 4, 6, 8])
    b.set_xticks(x)
    b.set_xticklabels(steps, fontsize=6.6)
    b.set_ylabel("rank of ' Cambridge'")
    b.set_title("(b) One probe prompt (M = muted, B = boosted features)", fontsize=8.5, loc="left")
    b.annotate("a fourth boost\nhelps a competitor", xy=(7, 2), xytext=(5.55, 6.4), fontsize=6.6, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    fig.tight_layout()
    save(fig, "h_early.png")


# ------------------------------------------------------------------------------------------------ speed
def speed():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.4, 2.3), gridspec_kw={"width_ratios": [1.0, 1.0, 0.8]})
    # Entries 13, 14: model calls per prompt and layer
    labels, vals = ["reference", "redundant\npasses\nremoved", "candidates\nbatched"], [186, 126, 6]
    bars = a.bar(labels, vals, color=[GRAY, LBLUE, BLUE], width=0.6)
    for bar, v in zip(bars, vals):
        a.text(bar.get_x() + bar.get_width() / 2, v + 4, str(v), ha="center", fontsize=8)
    a.set_ylim(0, 215)
    a.set_title("(a) Model calls per\nprompt and layer", fontsize=8, loc="left")
    a.tick_params(axis="x", labelsize=6.3)
    # Entry 15: sequential vs batched workflow, three runs
    seq, bat = [12.16, 11.55, 11.75], [0.90, 0.71, 0.58]
    for s, t in zip(seq, bat):
        b.plot([0, 1], [s, t], color=GRAY, lw=1, marker="o", ms=4)
    b.plot([0] * 3, seq, "o", color=ORANGE, ms=5)
    b.plot([1] * 3, bat, "o", color=BLUE, ms=5)
    b.set_xlim(-0.4, 1.4)
    b.set_xticks([0, 1])
    b.set_xticklabels(["one candidate\nat a time", "batched"], fontsize=6.6)
    b.set_ylabel("seconds")
    b.set_title("(b) Whole Hybrid run,\nthree repeats", fontsize=8, loc="left")
    # Entry 12: CPU vs GPU
    bars = c.bar(["CPU", "GPU"], [574.6, 150.1], color=[GRAY, BLUE], width=0.55)
    for bar, v in zip(bars, [574.6, 150.1]):
        c.text(bar.get_x() + bar.get_width() / 2, v + 12, f"{v:.0f} s", ha="center", fontsize=7.5)
    c.set_ylim(0, 680)
    c.set_title("(c) 3 prompts, 4 layers", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "h_speed.png")


# ------------------------------------------------------------------------------------------------ additive schematic
def additive():
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.25), sharey=True)
    base = [5.0, 3.0, 0.0, 1.5, 0.0, 0.0]        # activations of six features at one position (f3, f5, f6 are silent)
    mult = [1.5, 0.0, 1.0, 1.0, 1.0, 1.0]        # multipliers m_k (1 leaves a feature alone)
    after_m = [b * m for b, m in zip(base, mult)]
    c = [0, 0, 3.2, 0, 2.1, 0.0]                 # added amounts, silent features only
    after_a = [bm + ci for bm, ci in zip(after_m, c)]
    xs = list(range(6))
    for ax, vals, title in [(axs[0], base, "(a) Feature volumes\nbefore the edit"),
                            (axs[1], after_m, "(b) Multiplier edit:\nvolume x m"),
                            (axs[2], after_a, "(c) Plus additive edit:\nvolume x m + c")]:
        ax.bar(xs, vals, color=[BLUE if v > 0 else "white" for v in vals], edgecolor=BLUE, width=0.6, lw=1)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"f{k+1}" for k in xs], fontsize=7)
        ax.set_ylim(0, 9)
    axs[1].text(2, 0.2, "0 x m = 0", ha="center", va="bottom", rotation=90, fontsize=6.4, color=ORANGE)
    axs[1].text(4, 0.2, "0 x m = 0", ha="center", va="bottom", rotation=90, fontsize=6.4, color=ORANGE)
    axs[1].text(1, 0.2, "m = 0", ha="center", va="bottom", rotation=90, fontsize=6.4, color=INK)
    axs[2].bar([2, 4], [after_a[2], after_a[4]], color=ORANGE, width=0.6)
    axs[2].text(2, after_a[2] + 0.25, "0 + c", ha="center", fontsize=6.6, color=ORANGE)
    axs[2].text(4, after_a[4] + 0.25, "0 + c", ha="center", fontsize=6.6, color=ORANGE)
    axs[0].set_ylabel("activation")
    fig.tight_layout()
    save(fig, "h_additive.png")


# ------------------------------------------------------------------------------------------------ gradient descent vs sweeps (committed data of entry 29)
def gd_bars():
    path = ROOT / "docs" / "Research_Journal" / "images" / "e29_figure_data.json"
    data = json.load(open(path, encoding="utf8"))["fig1_rank1"]
    bands = ["2-5", "6-20", "21-100", "101-1000", "All 300"]
    names = [("sweep_ref", "sweep, original settings (0.4 to 1.5 times)", GRAY),
             ("sweep_wide", "sweep, same 0 to 3 range", ORANGE),
             ("gd", "gradient descent (0 to 3 times)", GREEN)]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    w = 0.26
    for j, (key, lab, col) in enumerate(names):
        xs = [i + (j - 1) * w for i in range(len(bands))]
        vals = [100.0 * data[b][key] / data[b]["n"] for b in bands]
        ax.bar(xs, vals, width=w * 0.92, color=col, label=lab)
        for x, b in zip(xs, bands):
            ax.text(x, 100.0 * data[b][key] / data[b]["n"] + 1.5, str(data[b][key]), ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f"{b}\n(n = {data[b]['n']})" for b in bands], fontsize=8)
    ax.set_ylim(0, 142)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{t}%" for t in [0, 25, 50, 75, 100]], fontsize=8)
    ax.set_ylabel("prompts brought to rank 1", fontsize=8.5)
    ax.set_xlabel("starting rank of the true answer", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7.6, frameon=False, ncol=1)
    fig.tight_layout()
    save(fig, "h_gd_bars.png")


# ------------------------------------------------------------------------------------------------ gradient descent flow
def gd_flow():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.set_xlim(0, 12.9)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    bw, bh = 4.0, 2.4
    X = [0.1, 4.45, 8.8]
    row1 = [
        (X[0], 4.7, "1  Clean pass", "residual stream of every\nposition at layer 8;\nblocker = top-1 token", GRAY),
        (X[1], 4.7, "2  Candidates", "200 features with the largest\nactivation over the prompt\npositions (start token excluded)", GRAY),
        (X[2], 4.7, "3  Free numbers", "one number z per feature;\nmultiplier m = 1 + a, with\na = -1 + 3 sigmoid(z);\nstart: no edit", BLUE),
    ]
    row2 = [
        (X[2], 1.9, "4  Edit and forward", "x' = x + sum_k a_k A_k W_k at\nevery position; run blocks 8\nto 11; read the last logits", BLUE),
        (X[1], 1.9, "5  Loss", "rank margin 0.3 + 3 x KL on the\nother tokens (once the target\nleads) + 0.005 x edit size", ORANGE),
        (X[0], 1.9, "6  Update, 100 steps", "Adam on z only; no GPT-2 or SAE\nweight is changed or given a\ngradient; then back to step 3", ORANGE),
    ]
    for x, y, t, body, col in row1 + row2:
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec=col, lw=1.5))
        ax.text(x + 0.15, y + bh - 0.2, t, fontsize=7.8, fontweight="bold", va="top", color=INK)
        ax.text(x + 0.15, y + bh - 0.78, body, fontsize=6.2, va="top", color=INK, linespacing=1.35)
    arrow = dict(arrowstyle="-|>", mutation_scale=8, color=INK, lw=1)
    ax.add_patch(FancyArrowPatch((X[0] + bw + 0.02, 5.9), (X[1] - 0.02, 5.9), **arrow))
    ax.add_patch(FancyArrowPatch((X[1] + bw + 0.02, 5.9), (X[2] - 0.02, 5.9), **arrow))
    ax.add_patch(FancyArrowPatch((X[2] + bw / 2, 4.68), (X[2] + bw / 2, 4.32), **arrow))
    ax.add_patch(FancyArrowPatch((X[2] - 0.02, 3.1), (X[1] + bw + 0.02, 3.1), **arrow))
    ax.add_patch(FancyArrowPatch((X[1] - 0.02, 3.1), (X[0] + bw + 0.02, 3.1), **arrow))
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 12.7, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec=GREEN, lw=1.5))
    ax.text(0.25, 1.25, "7  Result", fontsize=7.8, fontweight="bold", va="top", color=INK)
    ax.text(0.25, 0.75, "keep the best step (lowest loss among those where the target is first); re-check it through\nthe real hook in a full GPT-2 pass; report rank, probability, KL and edit size", fontsize=6.4, va="top", color=INK, linespacing=1.35)
    ax.add_patch(FancyArrowPatch((X[0] + bw / 2, 1.88), (X[0] + bw / 2, 1.42), **arrow))
    save(fig, "h_gd_flow.png")


if __name__ == "__main__":
    stages()
    early()
    speed()
    additive()
    gd_bars()
    gd_flow()
