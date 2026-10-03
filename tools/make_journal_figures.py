"""
Figures for Research Journal Entries 28 and 29, drawn ONLY from data committed in the repository:

    docs/Research_Journal/packs/sweep_vs_gradient/runs.jsonl            (Entry 29: 300 prompts x 3 arms)
    docs/Research_Journal/packs/gradient_descent/session_20261003_015115.json   (Entry 28: hand-trial runs)

Output: docs/Research_Journal/images/e29_fig*.{png,svg}, e28_fig1_*.{png,svg}, and e29_figure_data.json holding every
number that is plotted (so each figure can be checked against the data).

    .venv\\Scripts\\python.exe tools/make_journal_figures.py

Colour: entity colours are fixed (sweep 0.6/0.5 = blue, sweep 1.0/2.0 = orange, gradient descent = aqua; the first three
slots of the validated reference palette, validated all-pairs for colour-vision deficiency). The aqua is below 3:1 contrast
on the light surface, so every aqua mark carries a visible value label (relief rule).
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "docs", "Research_Journal", "packs", "sweep_vs_gradient", "runs.jsonl")
SESSION = os.path.join(ROOT, "docs", "Research_Journal", "packs", "gradient_descent", "session_20261003_015115.json")
OUT = os.path.join(ROOT, "docs", "Research_Journal", "images")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
COL = {"sweep_ref": "#2a78d6", "sweep_wide": "#eb6834", "gd": "#1baf7a"}
LABEL = {"sweep_ref": "Sweep, mute 0.6 / boost 0.5 (reach 0.4x to 1.5x)",
         "sweep_wide": "Sweep, mute 1.0 / boost 2.0 (reach 0x to 3x)",
         "gd": "Gradient descent (reach 0x to 3x)"}
SHORT = {"sweep_ref": "Sweep 0.6 / 0.5", "sweep_wide": "Sweep 1.0 / 2.0", "gd": "Gradient descent"}
ARMS = ["sweep_ref", "sweep_wide", "gd"]
BANDS = ["2-5", "6-20", "21-100", "101-1000"]
SRC = "GPT-2 small, layer 8, 300 held-out CounterFact prompts (hard set); data: packs/sweep_vs_gradient/runs.jsonl"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "font.family": "DejaVu Sans",
})

rows = [json.loads(l) for l in open(RUNS, encoding="utf-8") if l.strip()]
assert len(rows) == 300 and not any(r.get("error") for r in rows)
DATA = {"n_prompts": len(rows)}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def footer(fig, text, y=-0.02):
    fig.text(0.01, y, text, fontsize=8.5, color=MUTED, ha="left", va="top")


def legend_handles(arms=ARMS):
    return [Line2D([0], [0], marker="s", linestyle="", markersize=9, color=COL[a], label=LABEL[a]) for a in arms]


# ------------------------------------------------------------------------------------------- Figure 1: rank 1 by band
def fig1():
    groups = BANDS + ["All 300"]
    vals = {a: [] for a in ARMS}
    cnts = {a: [] for a in ARMS}
    ns = []
    for g in groups:
        sub = rows if g == "All 300" else [r for r in rows if r["band"] == g]
        ns.append(len(sub))
        for a in ARMS:
            k = sum(1 for r in sub if r[a]["rank"] == 1)
            cnts[a].append(k)
            vals[a].append(100 * k / len(sub))
    DATA["fig1_rank1"] = {g: {"n": n, **{a: cnts[a][i] for a in ARMS}} for i, (g, n) in enumerate(zip(groups, ns))}
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    w = 0.26
    x = np.arange(len(groups))
    for j, a in enumerate(ARMS):
        pos = x + (j - 1) * (w + 0.02)
        ax.bar(pos, vals[a], width=w, color=COL[a], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for xi, v, c, n in zip(pos, vals[a], cnts[a], ns):
            ax.text(xi, v + 1.2, f"{c}", ha="center", va="bottom", fontsize=9.4, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n = {n})" if g != "All 300" else f"All prompts\n(n = {n})" for g, n in zip(groups, ns)])
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.set_ylabel("Prompts where the target reached rank 1")
    ax.set_xlabel("Starting rank of the true answer on the unedited model")
    ax.grid(axis="x", visible=False)
    ax.set_title("Reached rank 1, by starting-rank band")
    ax.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=1, fontsize=9.5)
    footer(fig, SRC + ". Labels: number of prompts solved (n of each group is under the axis).", y=-0.27)
    save(fig, "e29_fig1_rank1_by_band")


# ------------------------------------------------------------------------------------------- Figure 2: KL paired
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    DATA["fig2_kl_pairs"] = {}
    allv = [r[a]["kl"] for r in rows for a in ARMS]
    lo, hi = min(allv) * 0.8, max(allv) * 1.25
    for ax, other in zip(axes, ["sweep_ref", "sweep_wide"]):
        both = [r for r in rows if r[other]["rank"] == 1 and r["gd"]["rank"] == 1]
        xs = np.array([r[other]["kl"] for r in both])
        ys = np.array([r["gd"]["kl"] for r in both])
        lower = int((ys < xs).sum())
        DATA["fig2_kl_pairs"][other] = {"n_both_rank1": len(both), "gd_lower": lower, "median_other": float(np.median(xs)), "median_gd": float(np.median(ys))}
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.2, linestyle="--", zorder=2)
        ax.scatter(xs, ys, s=34, color=COL["gd"], edgecolor=SURFACE, linewidth=0.9, alpha=0.9, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Same-prompt KL, {SHORT[other]} (nats)")
        ax.set_title(f"vs {SHORT[other]}")
        ax.text(0.04, 0.96, f"{len(both)} prompts solved by both\nGradient descent lower on {lower} of {len(both)}\n"
                            f"median KL: {np.median(ys):.2f} vs {np.median(xs):.2f}",
                transform=ax.transAxes, va="top", fontsize=9.6, color=INK)
        ax.text(0.97, 0.04, "equal KL", transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED, rotation=0)
    axes[0].set_ylabel("Same-prompt KL, gradient descent (nats)")
    fig.suptitle("Side effects on the same prompt, prompts both methods solved (below the dashed line = gradient descent disturbs less)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", y=1.0)
    footer(fig, SRC + ". KL(clean || edited) over all tokens except the target and the original top-1, last position. Log axes.", y=0.02)
    save(fig, "e29_fig2_kl_paired")


# ------------------------------------------------------------------------------------------- Figure 3: success vs KL
def fig3():
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    DATA["fig3_success_vs_kl"] = {}
    for a in ARMS:
        xs, ys = [], []
        for b in BANDS:
            sub = [r for r in rows if r["band"] == b]
            xs.append(st.median([r[a]["kl"] for r in sub]))
            ys.append(100 * sum(1 for r in sub if r[a]["rank"] == 1) / len(sub))
        DATA["fig3_success_vs_kl"][a] = {b: {"median_kl": round(x, 4), "rank1_pct": round(y, 2)} for b, x, y in zip(BANDS, xs, ys)}
        ax.plot(xs, ys, color=COL[a], linewidth=2, zorder=3)
        ax.scatter(xs, ys, s=70, color=COL[a], edgecolor=SURFACE, linewidth=1.5, zorder=4)
        for b, x, y in zip(BANDS, xs, ys):
            ax.annotate(b, (x, y), textcoords="offset points", xytext=(7, -3 if a != "gd" else 6), fontsize=8.6, color=INK2)
        ax.annotate(SHORT[a], (xs[-1], ys[-1]), textcoords="offset points", xytext=(10, 12 if a == "gd" else -14), fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel("Median same-prompt KL within the band (nats; lower = fewer side effects)")
    ax.set_ylabel("Prompts reaching rank 1 within the band (%)")
    ax.set_ylim(-3, 108)
    ax.set_xlim(left=0)
    ax.set_title("Success against side effects (each marker = one starting-rank band)")
    ax.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=9.5)
    footer(fig, SRC + ". Band labels give the starting rank; up and to the left is better.", y=-0.27)
    save(fig, "e29_fig3_success_vs_kl")


# ------------------------------------------------------------------------------------------- Figure 4: time
def fig4():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    rng = np.random.default_rng(3)
    DATA["fig4_time_s"] = {}
    for i, a in enumerate(ARMS):
        t = np.array([r[a]["time_s"] for r in rows])
        DATA["fig4_time_s"][a] = {"mean": float(t.mean()), "median": float(np.median(t)), "min": float(t.min()), "max": float(t.max())}
        y = i + rng.uniform(-0.17, 0.17, len(t))
        ax.scatter(t, y, s=14, color=COL[a], alpha=0.55, edgecolor="none", zorder=3)
        ax.plot([np.median(t)] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=2.4, zorder=4)
        ax.text(max(t) * 1.12, i, f"median {np.median(t):.1f} s\nmean {t.mean():.1f} s", va="center", fontsize=9.4, color=INK)
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([SHORT[a] for a in ARMS])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(2, 600)
    ax.set_xlabel("Model-compute time per prompt (seconds, log scale)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Time per prompt (one dot = one prompt; black bar = median)")
    footer(fig, SRC + ". GTX 1660 Ti; the sweep time includes candidate ranking and safety filtering.", y=-0.04)
    save(fig, "e29_fig4_time")


# ------------------------------------------------------------------------------------------- Entry 28 figure: rank by step
def fig_e28():
    runs = {r["run_id"]: r for r in json.load(open(SESSION, encoding="utf-8"))}
    panels = [(2, "The sky is -> black"), (10, "Frankie Lee Sims died at -> Dallas"), (11, "Kaka professionally plays the sport -> soccer"),
              (1, "My nephew is a -> rapist"), (6, "The sky is -> powder"), (12, "Hello, my name is -> Aryan")]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7))
    DATA["e28_rank_by_step"] = {}
    for ax, (rid, title) in zip(axes.flat, panels):
        r = runs[rid]
        steps = [p["Step"] for p in r["rank_progression"]]
        ranks = [p["Target Rank"] for p in r["rank_progression"]]
        DATA["e28_rank_by_step"][title] = {"run_id": rid, "start_rank": ranks[0], "end_rank": ranks[-1], "steps": steps[-1],
                                           "kl": r["gradient_descent"]["kl_nats"], "settings": r["gradient_descent"]["settings"]}
        ax.axhline(1, color=MUTED, linestyle="--", linewidth=1.1, zorder=2)
        if rid == 6:                                                       # the one prompt where hand-run sweeps exist in the file
            for sid, lab in ((5, "sweep 0.3 / 0.5"), (9, "sweep 0.6 / 0.8")):
                s = runs[sid]
                ax.plot([p["Step"] for p in s["rank_progression"]], [p["Target Rank"] for p in s["rank_progression"]], color=MUTED,
                        linewidth=1.4, linestyle=(0, (4, 2)) if sid == 5 else "-", zorder=3)
                ax.text(s["rank_progression"][-1]["Step"] + 2, s["rank_progression"][-1]["Target Rank"], f"{lab}: end #{s['rank_progression'][-1]['Target Rank']}",
                        fontsize=8, color=INK2, va="center")
                DATA["e28_rank_by_step"][title + f" / {lab}"] = {"run_id": sid, "end_rank": s["rank_progression"][-1]["Target Rank"]}
        ax.plot(steps, ranks, color=COL["gd"], linewidth=2.2, zorder=4)
        ax.scatter([steps[0], steps[-1]], [ranks[0], ranks[-1]], s=45, color=COL["gd"], edgecolor=SURFACE, linewidth=1.3, zorder=5)
        ax.text(steps[0] + 1.5, ranks[0] * 1.3, f"start #{ranks[0]}", fontsize=8.8, color=INK, va="top",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2), zorder=6)
        ax.text(steps[-1] * 0.98, ranks[-1] * 1.35 if ranks[-1] > 1 else 1.55, f"end #{ranks[-1]}", fontsize=8.8, color=INK, ha="right", va="top",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2), zorder=6)
        ax.set_yscale("log")
        ax.set_ylim(top=max(ranks) * 1.7, bottom=0.7)
        ax.invert_yaxis()
        ax.set_title(f"{title}  (KL {r['gradient_descent']['kl_nats']:.2f})", fontsize=9.6, loc="left")
        ax.set_xlabel("gradient step", fontsize=9)
    axes[0, 0].set_ylabel("target rank (log, lower = better)")
    axes[1, 0].set_ylabel("target rank (log, lower = better)")
    fig.suptitle("Gradient descent, hand trials: target rank at every step (green; dashed line = rank 1)", x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=1.0)
    footer(fig, "GPT-2 small, layer 8, Top N 200; data: packs/gradient_descent/session_20261003_015115.json (hand-picked prompts, exploratory). "
                "Grey lines in the 'powder' panel are the two hand-run sweeps recorded in the same file.", y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save(fig, "e28_fig1_rank_by_step")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig_e28()
    json.dump(DATA, open(os.path.join(OUT, "e29_figure_data.json"), "w"), indent=1)
    print(json.dumps(DATA["fig1_rank1"], indent=0)[:1200])
    print(DATA["fig2_kl_pairs"])
