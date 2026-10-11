"""
Figures added to the report (docs/report_v2) on 2026-10-12: the ROME reference chapter and the evolution-chapter figures that had data but no picture.

Every figure is drawn from data that is committed in the repository; no model is run:

    data/comparison/trace_summary.json                       ROME causal tracing over 97 known facts (teammate's branch, 2026-10-07)
    benchmark_results/rome_vs_scalpel/layer_sweep_dev.csv    ROME layer sweep on 100 dev records
    docs/Research_Journal/packs/cross_model_transfer/d_analysis_linear.json   our edit scored with ROME's own metrics (PS, NS)
    benchmark_results/candidate_source_study/*               candidate-source study (journal 20)
    docs/Research_Journal/packs/repair_diagnostics/*         repair and SAE-limit diagnostics (journal 22)
    docs/Research_Journal/packs/sweep_vs_gradient/*, gradient_descent/*, images/e29_figure_data.json   journals 28 and 29

Run:  .venv\\Scripts\\python.exe tools/make_report_v4_figures.py
Writes docs/report_v2/figures/v4_*.png (drawn here) and copies finished figures of the first report (docs/report/figures) that the rewritten report had left out.
"""
import csv
import json
import shutil
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report_v2" / "figures"
OLD = ROOT / "docs" / "report" / "figures"
PACK = ROOT / "docs" / "Research_Journal" / "packs"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, RED, YELLOW, GREY, DGREY, GREEN, ORANGE = "#2a78d6", "#d03a2f", "#eda100", "#b9b8b2", "#6f6e69", "#0a7a5a", "#eb6834"
W = 6.2
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED, "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "font.family": "DejaVu Sans", "axes.titlesize": 9,
                     "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, title, body="", fc="#eef4fd", ec=BLUE, tsize=8.5, bsize=7.3, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.9", linewidth=1.1, edgecolor=ec, facecolor=fc, linestyle="--" if dashed else "-"))
    if body:
        nl = title.count("\n") + 1
        ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=tsize, fontweight="bold", color=INK, linespacing=1.15)
        ax.text(x + w / 2, y + h - 1.3 - 2.3 * nl - 1.0, body, ha="center", va="top", fontsize=bsize, color=INK2, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=INK)


def arrow(ax, x0, y0, x1, y1, color=DGREY, dashed=False, text="", tdx=0, tdy=1.6, tsize=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, linewidth=1.3, color=color, linestyle="--" if dashed else "-"))
    if text:
        ax.text((x0 + x1) / 2 + tdx, (y0 + y1) / 2 + tdy, text, ha="center", va="center", fontsize=tsize, color=color)


# ===================================================================================================================== ROME
def rome_compare():
    """Two ways of changing what GPT-2 says: ROME rewrites a weight matrix; FeatureScalpel changes the activations for one forward pass."""
    fig, ax = plt.subplots(figsize=(W, 4.9))
    ax.set_xlim(0, 100); ax.set_ylim(-1, 66); ax.axis("off"); ax.grid(False)
    ax.text(25, 64, "ROME (Meng et al., 2022)", ha="center", fontsize=9.5, fontweight="bold", color=RED)
    ax.text(75, 64, "FeatureScalpel (this project)", ha="center", fontsize=9.5, fontweight="bold", color=BLUE)
    left = [("1. Locate", "Find the key vector k* that the\nlayer-3 MLP receives at the\nlast subject token"),
            ("2. Find the new value", "Gradient descent finds v*, the\nvector that makes the model\nsay the new answer"),
            ("3. Rewrite the weights", "One closed-form rank-one update\nof W; the covariance C of text\nlimits the side effects"),
            ("4. Result", "Permanent: every later prompt\nuses the edited weights")]
    right = [("1. Locate", "Rank the layer-8 SAE features by\nwhat removing each one does\nto the answer"),
             ("2. Choose multipliers", "A sweep or gradient descent\npicks one multiplier per\nfeature for this prompt"),
             ("3. Add the change", "The change is added to the\nresidual stream in one forward\npass; weights stay frozen"),
             ("4. Result", "Transient: the edit is discarded\nafter the prompt")]
    ys = [48, 32.6, 17.2, 1.8]
    h = 12.6
    for (t, bd), y in zip(left, ys):
        box(ax, 1, y, 47, h, t, bd, fc="#fdeeed", ec=RED, bsize=7.4)
    for (t, bd), y in zip(right, ys):
        box(ax, 52, y, 47, h, t, bd, fc="#eef4fd", ec=BLUE, bsize=7.4)
    for y in ys[:-1]:
        arrow(ax, 24.5, y - 0.1, 24.5, y - 2.7, RED)
        arrow(ax, 75.5, y - 0.1, 75.5, y - 2.7, BLUE)
    save(fig, "v4_rome_compare.png")


def rome_in_project():
    """How the ROME work is connected to the rest of the project: what exists, what is used, what is still open."""
    fig, ax = plt.subplots(figsize=(W, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off"); ax.grid(False)
    ax.text(1, 64.5, "Reference material and what we built on it", fontsize=8.6, color=INK2, fontweight="bold")
    bw, bh, gap = 29.3, 22, 4.4
    xs = [1, 1 + bw + gap, 1 + 2 * (bw + gap)]
    box(ax, xs[0], 38, bw, bh, "CounterFact\n(ROME paper)", "21,919 records: prompt, true\nanswer, counterfactual\nanswer, reworded and\nneighbour prompts", fc="#f4f4f2", ec=DGREY, bsize=6.5, tsize=7.8)
    box(ax, xs[1], 38, bw, bh, "ROME's three scores", "ES: new beats true answer\nPS: same on reworded prompts\nNS: true answer wins nearby\nS: harmonic mean of the three", fc="#fff6e0", ec=YELLOW, bsize=6.5, tsize=7.8)
    box(ax, xs[2], 38, bw, bh, "Our benchmark\n(done)", "Reworded and neighbour\nscores are ROME's PS and NS;\nthe 100 ROME dev records\nstay out of every test set", fc="#e7f5ef", ec=GREEN, bsize=6.5, tsize=7.8)
    arrow(ax, xs[0] + bw + 0.2, 49, xs[1] - 0.4, 49, DGREY)
    arrow(ax, xs[1] + bw + 0.2, 49, xs[2] - 0.4, 49, DGREY)
    ax.text(1, 33, "ROME's own pipeline and the planned comparison", fontsize=8.6, color=INK2, fontweight="bold")
    box(ax, xs[0], 8, bw, bh, "ROME on GPT-2 Small\n(done, Steps 0 to 5)", "Official code, causal\ntracing, layer sweep,\nedit layer 3, about 4 s\nper edit on a CPU", fc="#e7f5ef", ec=GREEN, bsize=6.5, tsize=7.8)
    box(ax, xs[1], 8, bw, bh, "Adapter for our editors\n(not done, Step 6)", "One frozen record list and\none scoring script that\nreads ROME's edit and ours", fc="#f4f4f2", ec=MUTED, bsize=6.5, tsize=7.8, dashed=True)
    box(ax, xs[2], 8, bw, bh, "Head-to-head table\n(not done, Steps 7, 8)", "ES, PS, NS, fluency and\ncost for both methods on\nthe same records", fc="#f4f4f2", ec=MUTED, bsize=6.5, tsize=7.8, dashed=True)
    arrow(ax, xs[0] + bw + 0.2, 19, xs[1] - 0.4, 19, MUTED, dashed=True)
    arrow(ax, xs[1] + bw + 0.2, 19, xs[2] - 0.4, 19, MUTED, dashed=True)
    ax.text(50, 2.2, "Solid outline: finished and stored in the repository.   Dashed outline: planned, not run.", ha="center", fontsize=7.4, color=INK2)
    save(fig, "v4_rome_in_project.png")


def rome_trace():
    d = json.load(open(ROOT / "data" / "comparison" / "trace_summary.json"))
    roles = ["first subject token", "middle subject token", "last subject token", "later tokens", "last token"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5))
    spec = [("resid", "Residual stream", "Purples"), ("mlp", "MLP output\n(3-layer window)", "Greens"), ("attn", "Attention output\n(3-layer window)", "Reds")]
    for ax, (key, title, cmap) in zip(axes, spec):
        a = np.array(d["effects"][key])
        im = ax.imshow(a, aspect="auto", cmap=cmap, vmin=0 if key != "mlp" else None)
        ax.set_title(title, fontsize=8.3)
        ax.set_xticks(range(0, 12, 2)); ax.set_xlabel("layer")
        ax.set_yticks(range(5)); ax.set_yticklabels(roles if key == "resid" else [""] * 5, fontsize=7)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04); cb.ax.tick_params(labelsize=6.5)
    fig.text(0.5, -0.04, "Average gain in the probability of the true answer when the clean state is restored at that place (97 facts)", ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "v4_rome_trace.png")


def rome_sites():
    d = json.load(open(ROOT / "data" / "comparison" / "trace_summary.json"))
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=False)
    cols = {"resid": BLUE, "mlp": GREEN, "attn": RED}
    names = {"resid": "residual stream", "mlp": "MLP", "attn": "attention"}
    for ax, row, title in ((axes[0], 2, "(a) last subject token\n(where ROME writes its edit)"), (axes[1], 4, "(b) last token of the prompt")):
        for k in ("resid", "mlp", "attn"):
            ax.plot(range(12), d["effects"][k][row], marker="o", ms=3.2, lw=1.4, color=cols[k], label=names[k])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=8.3, loc="left"); ax.set_xlabel("layer"); ax.set_xticks(range(0, 12))
    axes[0].set_ylabel("average gain in p(answer)")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "v4_rome_sites.png")


def rome_sweep():
    rows = list(csv.DictReader(open(ROOT / "benchmark_results" / "rome_vs_scalpel" / "layer_sweep_dev.csv")))
    labels = ["no edit" if r["layer"] == "none" else f"layer {r['layer']}" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={"width_ratios": [1.8, 1]})
    ax = axes[0]
    x = np.arange(len(rows)); bw = 0.26
    for j, (key, nm, c) in enumerate((("ES", "efficacy (ES)", BLUE), ("PS", "paraphrase (PS)", YELLOW), ("NS", "neighbourhood (NS)", GREEN))):
        vals = [float(r[key]) for r in rows]
        ax.bar(x + (j - 1) * bw, vals, bw, color=c, label=nm)
        for xi, v in zip(x + (j - 1) * bw, vals):
            ax.text(xi, v + 0.012, f"{100 * v:.0f}", ha="center", fontsize=5.8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.2); ax.set_ylim(0, 1.18); ax.set_ylabel("share of records")
    ax.set_title("(a) ROME's three scores by edit layer", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.9, handlelength=1.2)
    ax.grid(axis="x", visible=False)
    ax2 = axes[1]
    ls = [r for r in rows if r["layer"] != "none"]
    ax2.plot([int(r["layer"]) for r in ls], [float(r["S"]) for r in ls], marker="o", color=INK, lw=1.5, label="overall score S")
    ax2.plot([int(r["layer"]) for r in ls], [float(r["NS"]) for r in ls], marker="s", color=GREEN, lw=1.5, label="neighbourhood NS")
    ax2.axhline(float(rows[0]["NS"]), color=GREEN, lw=0.9, ls="--"); ax2.text(9.2, float(rows[0]["NS"]) + 0.012, "NS, no edit", fontsize=6.5, color=GREEN, ha="right")
    ax2.set_xticks([1, 3, 5, 7, 9]); ax2.set_xlabel("edit layer"); ax2.set_ylim(0.45, 0.95)
    ax2.set_title("(b) the cost of the edit", loc="left", fontsize=8.5)
    ax2.annotate("chosen: layer 3\n(S = 0.851)", xy=(3, 0.851), xytext=(4.1, 0.91), fontsize=6.8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax2.legend(frameon=False, fontsize=6.8, loc="lower left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "v4_rome_sweep.png")


def rome_vs_ours():
    rows = list(csv.DictReader(open(ROOT / "benchmark_results" / "rome_vs_scalpel" / "layer_sweep_dev.csv")))
    none = rows[0]; l3 = [r for r in rows if r["layer"] == "3"][0]
    ours = json.load(open(PACK / "cross_model_transfer" / "d_analysis_linear.json"))["s2m_L8_L16"]["small_ok"]["stage2"]
    data = {"PS": (float(none["PS"]), float(l3["PS"]), ours["small unchanged"]["PS"], ours["small with the edit"]["PS"]),
            "NS": (float(none["NS"]), float(l3["NS"]), ours["small unchanged"]["NS"], ours["small with the edit"]["NS"])}
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True)
    for ax, (key, title) in zip(axes, (("PS", "(a) reworded prompts: new answer beats true"), ("NS", "(b) neighbour prompts: true answer still wins"))):
        v = data[key]
        xs = [0, 1, 2.6, 3.6]
        cols = [GREY, RED, GREY, BLUE]
        ax.bar(xs, [100 * a for a in v], width=0.8, color=cols)
        for xi, a in zip(xs, v):
            ax.text(xi, 100 * a + 1.5, f"{100 * a:.0f}%", ha="center", fontsize=8, color=INK)
        ax.set_xticks(xs); ax.set_xticklabels(["no\nedit", "ROME\n(layer 3)", "no\nedit", "FeatureScalpel\n(layer 8)"], fontsize=7.3)
        ax.set_ylim(0, 112); ax.set_title(title, loc="left", fontsize=8.3); ax.grid(axis="x", visible=False)
        ax.text(0.5, -34, "ROME's 100 dev records", ha="center", fontsize=6.8, color=INK2)
        ax.text(3.1, -34, "our 88 test records", ha="center", fontsize=6.8, color=INK2)
    axes[0].set_ylabel("share of prompts (%)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "v4_rome_vs_ours.png")


# ===================================================================================================================== EVOLUTION
def hybrid_sophia():
    rows = list(csv.DictReader(open(ROOT / "benchmark_results" / "candidate_source_study" / "sophia_run_progression.csv")))
    step = [int(r["step"]) for r in rows]; rank = [int(r["target_rank"]) for r in rows]; prob = [float(r["target_prob_pct"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5))
    axes[0].step(step, rank, where="post", color=BLUE, lw=1.8)
    axes[0].axhline(1, color=MUTED, lw=0.9, ls="--"); axes[0].set_ylim(0.6, 6.6); axes[0].invert_yaxis()
    axes[0].set_xlabel("sweep step (features muted + boosted = 2 x step)"); axes[0].set_ylabel("rank of the target (woman)"); axes[0].set_yticks([1, 2, 3, 4, 5, 6])
    axes[0].set_title("(a) rank of the target", loc="left", fontsize=8.5)
    axes[0].annotate("rank 1 only at\nstep 34 (68 features)", xy=(34, 1), xytext=(15.5, 1.9), fontsize=7, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[1].plot(step, prob, color=GREEN, lw=1.8)
    axes[1].set_xlabel("sweep step"); axes[1].set_ylabel("probability of the target (%)"); axes[1].set_title("(b) probability of the target", loc="left", fontsize=8.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "v4_hybrid_sophia.png")


def cand_pools():
    rows = list(csv.DictReader(open(ROOT / "benchmark_results" / "candidate_source_study" / "batch_paired_summary.csv")))
    rows = [r for r in rows if r["all_round0_candidates"] and r["winner"].startswith(("all", "tie")) or (r["all_round0_candidates"] and r["winner"].startswith("n/a (multi"))]
    rows = [r for r in rows if r["all_round0_candidates"]]
    rows.sort(key=lambda r: int(r["all_round0_candidates"]))
    fig, ax = plt.subplots(figsize=(W, 3.5))
    for i, r in enumerate(rows):
        a, l = int(r["all_round0_candidates"]), int(r["last_round0_candidates"])
        ax.plot([l, a], [i, i], color=GREY, lw=1.2, zorder=1)
        ax.scatter([l], [i], s=34, color=ORANGE if r["last_success"] == "False" else GREEN, zorder=3, marker="o")
        ax.scatter([a], [i], s=34, color=ORANGE if r["all_success"] == "False" else GREEN, zorder=3, marker="s")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["prompt"][:34] + ("..." if len(r["prompt"]) > 34 else "") for r in rows], fontsize=6.6)
    ax.set_xscale("log"); ax.set_xlabel("features available for the first round (log scale)")
    ax.scatter([], [], marker="o", color=GREY, s=30, label="last token only"); ax.scatter([], [], marker="s", color=GREY, s=30, label="all prompt positions")
    ax.scatter([], [], marker="o", color=GREEN, s=30, label="reached rank 1"); ax.scatter([], [], marker="o", color=ORANGE, s=30, label="did not")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, columnspacing=1.2, handletextpad=0.3)
    ax.set_xticks([40, 60, 100, 200, 300, 500]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    save(fig, "v4_cand_pools.png")


def repair():
    d = json.load(open(PACK / "repair_diagnostics" / "merged_all_layers.json"))
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
    cols = [BLUE, ORANGE, GREEN, RED]
    names = ["Colosseum -> Rome", "door -> cat", "doctor -> she", "MIT -> Cambridge"]
    ax = axes[0]
    for pr, c, nm in zip(d["prompts"], cols, names):
        Ls = [l["layer"] for l in pr["layers"]]
        ax.plot(Ls, [l["sae"]["rank_after"] for l in pr["layers"]], marker="o", ms=4, lw=1.6, color=c, label=nm)
        ax.scatter(Ls, [l["optimal"]["matched"]["last"]["rank"] for l in pr["layers"]], marker="x", s=24, color=c, zorder=4)
    ax.set_yscale("log"); ax.invert_yaxis(); ax.set_xlabel("layer of the edit"); ax.set_ylabel("rank of the target after the edit (log; 1 = best)")
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_title("(a) SAE edit (dots) vs free residual edit (crosses)", loc="left", fontsize=8.2)
    ax.legend(frameon=False, fontsize=6.6, loc="center left", bbox_to_anchor=(0.0, 0.52))
    ax2 = axes[1]
    for pr, c, nm in zip(d["prompts"], cols, names):
        L8 = [l for l in pr["layers"] if l["layer"] == 8][0]
        rows = [r for r in L8["trace"]["rows"] if r.get("downstream") and r.get("persistence") is not None]
        ax2.plot([r["layer"] for r in rows], [r["persistence"] for r in rows], marker="o", ms=3, lw=1.4, color=c)
    ax2.axhline(1, color=MUTED, lw=0.8, ls="--"); ax2.set_xlabel("block after the edit"); ax2.set_ylabel("push size / injected size")
    ax2.set_title("(b) is the push undone downstream?", loc="left", fontsize=8.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "v4_repair.png")


def _runs():
    return [json.loads(l) for l in open(PACK / "sweep_vs_gradient" / "runs.jsonl", encoding="utf8")]


def gd_side():
    runs = _runs()
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.8), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    both = [r for r in runs if r["sweep_wide"]["rank"] == 1 and r["gd"]["rank"] == 1]
    x = np.array([r["sweep_wide"]["kl"] for r in both]); y = np.array([r["gd"]["kl"] for r in both])
    ax.scatter(np.clip(x, 1e-4, None), np.clip(y, 1e-4, None), s=9, color=GREEN, alpha=0.75, linewidths=0)
    lim = [1e-4, 3]; ax.plot(lim, lim, color=MUTED, lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("side effect of the sweep (KL, nats)"); ax.set_ylabel("side effect of gradient descent (KL)")
    ax.set_title(f"(a) {len(both)} prompts both solved", loc="left", fontsize=8.5)
    ax.text(2.0e-4, 1.6, "gradient descent\ndisturbs less", fontsize=7, color=INK2, va="top")
    ax.text(2.2, 2.0e-4, "equal", fontsize=6.5, color=MUTED, ha="right")
    ax2 = axes[1]
    arms = [("sweep_ref", "Sweep 0.6 / 0.5", BLUE), ("sweep_wide", "Sweep 1.0 / 2.0", ORANGE), ("gd", "Gradient descent", GREEN)]
    rng = np.random.default_rng(0)
    for i, (k, nm, c) in enumerate(arms):
        t = np.array([r[k]["time_s"] for r in runs])
        ax2.scatter(t, i + rng.uniform(-0.28, 0.28, len(t)), s=5, color=c, alpha=0.6, linewidths=0)
        ax2.plot([np.median(t)] * 2, [i - 0.36, i + 0.36], color=INK, lw=2)
        ax2.text(np.median(t) * 1.15, i + 0.42, f"median {np.median(t):.1f} s", fontsize=7, color=INK)
    ax2.set_yticks(range(3)); ax2.set_yticklabels([a[1] for a in arms], fontsize=7.5); ax2.invert_yaxis(); ax2.set_xscale("log")
    ax2.set_xlabel("model time per prompt (s, log scale)"); ax2.set_title("(b) time per prompt (300 prompts)", loc="left", fontsize=8.5); ax2.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=1.2)
    save(fig, "v4_gd_side.png")


def gd_tradeoff():
    d = json.load(open(ROOT / "docs" / "Research_Journal" / "images" / "e29_figure_data.json"))["fig3_success_vs_kl"]
    fig, ax = plt.subplots(figsize=(W, 3.0))
    spec = [("sweep_ref", "Sweep 0.6 / 0.5", BLUE), ("sweep_wide", "Sweep 1.0 / 2.0", ORANGE), ("gd", "Gradient descent", GREEN)]
    for k, nm, c in spec:
        bands = list(d[k].keys())
        xs = [d[k][b]["median_kl"] for b in bands]; ys = [d[k][b]["rank1_pct"] for b in bands]
        ax.plot(xs, ys, color=c, lw=1.5, marker="o", ms=5, label=nm)
        for b, xv, yv in zip(bands, xs, ys):
            ax.annotate(b, (xv, yv), textcoords="offset points", xytext=(4, 4), fontsize=6.3, color=INK2)
    ax.set_xlabel("median side effect on the same prompt (KL, nats; lower is better)"); ax.set_ylabel("prompts reaching rank 1 (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    save(fig, "v4_gd_tradeoff.png")


def gd_trajectories():
    s = json.load(open(PACK / "gradient_descent" / "session_20261003_015115.json"))
    want = {2: "The sky is -> black", 10: "Frankie Lee Sims died at -> Dallas", 11: "Kaka plays the sport -> soccer", 1: "My nephew is a -> rapist", 6: "The sky is -> powder", 12: "Hello, my name is -> Aryan"}
    by_id = {x["run_id"]: x for x in s}
    fig, axes = plt.subplots(2, 3, figsize=(W, 3.7), sharex=True)
    for ax, (rid, ttl) in zip(axes.ravel(), want.items()):
        x = by_id.get(rid)
        if x is None:
            ax.axis("off"); continue
        rp = x["rank_progression"]
        key = [k for k in rp[0].keys() if "Rank" in k][0]
        ranks = [e[key] for e in rp]
        ax.plot(range(len(ranks)), ranks, color=GREEN, lw=1.6); ax.set_yscale("log"); ax.invert_yaxis()
        ax.axhline(1, color=MUTED, lw=0.8, ls="--")
        ax.set_title(ttl, fontsize=6.8, loc="left"); ax.text(0.97, 0.06, f"{ranks[0]:,} to {ranks[-1]:,}", transform=ax.transAxes, ha="right", fontsize=6.8, color=INK)
        ax.tick_params(labelsize=6.5)
    for ax in axes[1]:
        ax.set_xlabel("gradient step", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("rank (log)", fontsize=7)
    fig.tight_layout(h_pad=0.6, w_pad=0.5)
    save(fig, "v4_gd_trajectories.png")


def copies():
    shutil.copy(ROOT / "docs" / "rome_baseline" / "tracing" / "eiffel.png", OUT / "v4_rome_eiffel.png")
    print("copied v4_rome_eiffel.png")
    for src, dst in [("fig_elig_v2.png", "v4_elig.png"), ("fig_layer12.png", "v4_layer12.png"), ("fig_filter.png", "v4_filter.png"), ("fig_pipeline.png", "v4_pipeline.png"),
                     ("fig_reworded.png", "v4_reworded.png"), ("fig_genquality.png", "v4_genquality.png"),
                     ("fig_arch_v2.png", "v4_arch.png"), ("fig_algo_v2.png", "v4_algo.png"), ("fig_ui_hybrid.png", "v4_ui_hybrid.png"), ("fig_ui_proto.png", "v4_ui_proto.png")]:
        shutil.copy(OLD / src, OUT / dst)
        print("copied", dst)


if __name__ == "__main__":
    rome_compare(); rome_in_project(); rome_trace(); rome_sites(); rome_sweep(); rome_vs_ours()
    hybrid_sophia(); cand_pools(); repair(); gd_side(); gd_tradeoff(); gd_trajectories(); copies()
