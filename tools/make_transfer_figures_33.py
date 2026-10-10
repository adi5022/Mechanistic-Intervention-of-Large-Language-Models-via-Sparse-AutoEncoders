"""
Figures for Research Journal Entry 33 (cross-model transfer, part 3: several layers, the ceiling test), drawn only from the committed data pack
docs/Research_Journal/packs/cross_model_transfer/ (no model is run):

    .venv\\Scripts\\python.exe tools/make_transfer_figures_33.py

e33_fig1_layers.*    step D5: top-1 rate and median rank of the target in medium for each injection configuration (one layer, several layers split, several full)
e33_fig2_ceiling.*   step D6: the edit tuned through the translator against the original edit and the controls: top-1, rewordings, neighbours

Static light-surface images, same palette and conventions as e32 (colour follows the entity; every bar is direct-labelled with its number).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "docs", "Research_Journal", "packs", "cross_model_transfer")
IMG = os.path.join(ROOT, "docs", "Research_Journal", "images")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, PURPLE, MAGENTA, YELLOW, GREY, DGREY, AQUA = "#2a78d6", "#d03a2f", "#e87ba4", "#eda100", "#b9b8b2", "#6f6e69", "#1baf7a"

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.family": "DejaVu Sans"})


def load(name):
    return json.load(open(os.path.join(PACK, name), encoding="utf8"))


def save(fig, name):
    os.makedirs(IMG, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(IMG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig1():
    d = load("d5_configs.json")
    cf, base = d["configs"], d["unchanged_medium"]
    groups = [("One layer of medium", ["L4", "L8", "L12", "L16", "L20"], BLUE),
              ("Several layers, dose split (same total push)", [k for k in cf if k.endswith("split")], BLUE),
              ("Several layers, full dose at each (bigger push; exploratory)", [k for k in cf if k.endswith("full")], GREY)]
    rows = []
    for title, keys, color in groups:
        rows.append(("header", title, None))
        rows += [("row", k, color) for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.4), sharey=True)
    for ax, key, title in ((axes[0], "test_top1", "Target is medium's top answer (%)"), (axes[1], "test_median_rank", "Median rank of the target (lower = better)")):
        for i, (kind, k, color) in enumerate(rows):
            if kind == "header":
                continue
            v = cf[k][key]
            ax.barh(i, 100 * v if key == "test_top1" else v, color=color, height=0.62)
            lab = f"{100 * v:.0f}%" if key == "test_top1" else f"{v:.0f}"
            ax.text((100 * v if key == "test_top1" else v) * (1.0 if key == "test_top1" else 1.0) + (0.6 if key == "test_top1" else 1.5), i, lab, va="center", fontsize=9, color=INK, zorder=5, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
            if key == "test_top1":
                ctrl = 100 * max(cf[k]["random_top1_same_dose"], cf[k]["wrong_recipe_top1_same_dose"])
                ax.scatter([ctrl], [i], color=INK2, marker="|", s=110, linewidths=1.8, zorder=3)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(0, 30)
    axes[1].set_xlim(0, 160)
    axes[1].axvline(base["median_rank"], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    axes[1].text(base["median_rank"] - 2, len(rows) - 0.6, f"unchanged medium: {base['median_rank']:.0f}", ha="right", va="bottom", fontsize=8.5, color=INK2)
    axes[0].axvline(15, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    axes[0].text(15.4, len(rows) - 0.6, "old Gate 3 line (15%)", ha="left", va="bottom", fontsize=8.5, color=INK2)
    labels = []
    for kind, k, _ in rows:
        labels.append(k if kind == "row" else k)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([("   " + k + (f"  (dose {cf[k]['dose']:g})" if kind == "row" else "")) if kind == "row" else k for kind, k, _ in rows], fontsize=9)
    for lbl, (kind, k, _) in zip(axes[0].get_yticklabels(), rows):
        if kind == "header":
            lbl.set_fontweight("bold"); lbl.set_color(INK)
    axes[0].invert_yaxis()
    fig.suptitle("Adding the translated edit at several layers of medium at once does not beat the best single layer", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.text(0.01, -0.01, f"{d['n_test_records']} test records whose edit reached rank 1 in small; dose chosen per configuration on dev. Tick marks (left panel) = the stronger of the two controls "
             "(random vector, another record's edit) at the same dose: 0–2% everywhere.\nUnchanged medium: 0% top-1. Gate 4 compared the best single layer on dev (16) with the best split set on dev (12+16): 8% against 7%.",
             fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "e33_fig1_layers")


def fig2():
    d = load("d6_b_aware.json")["summary"]["ok"]
    s2, t1, n = d["stage2"], d["top1"], d["n"]
    rows = [("medium unchanged", 0.0, s2["medium unchanged"], GREY),
            ("original edit (tuned in small)", t1["small_tuned"], s2["small_tuned"], BLUE),
            ("edit tuned through the translator", t1["b_aware"], s2["b_aware"], PURPLE),
            ("same, toward a random word\n(judged on that word)", t1["random_word_b"], s2["random_word_b"], MAGENTA),
            ("same, another record's recipe", t1["wrong_b"], s2["wrong_b"], YELLOW),
            ("small, with the edit itself", 1.0, s2["small with the edit"], DGREY)]
    panels = [("Target is the top answer\non the tuned prompt (%)", lambda r: r[1]), ("Rewordings: new beats true (%)\nhigher = more of the new answer", lambda r: r[2]["PS"]),
              ("Neighbours: true still beats new (%)\nhigher = fewer leaks", lambda r: r[2]["NS"])]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=True)
    for ax, (title, fn) in zip(axes, panels):
        for i, r in enumerate(rows):
            v = 100 * fn(r)
            ax.barh(i, v, color=r[3], height=0.62)
            ax.text(v + 1.2, i, f"{v:.0f}%", va="center", fontsize=9.5, color=INK, zorder=5, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
        ax.set_xlim(0, 112); ax.set_title(title, loc="left", fontsize=9.8, color=INK); ax.grid(axis="y", visible=False); ax.set_xlabel("% of records / prompts")
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([r[0] for r in rows], fontsize=9)
    axes[0].invert_yaxis()
    axes[1].axvline(100 * s2["small unchanged"]["PS"], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1)
    axes[2].axvline(100 * s2["small unchanged"]["NS"], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1)
    fig.suptitle("Ceiling test: an edit tuned through the translator against medium's output reaches the top answer on 95% of records, and still lifts rewordings far above the random-word edit",
                 x=0.01, ha="left", fontsize=10.8, color=INK, wrap=True)
    fig.text(0.01, -0.035, f"Dashed line = small without the edit. {n} test records whose original edit reached rank 1 in small; medium layer 16, dose 2.0 fixed beforehand. Top-1 of the random-word edit is for its own word (that word starts at median rank 174). "
             "The tuned edit's neighbour loss (8 points) is\nabout the same as the edit's own loss inside small (9 points). This tunes against medium's output, so it is a ceiling and not a transfer.", fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    save(fig, "e33_fig2_ceiling")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2"]
    for w in which:
        {"1": fig1, "2": fig2}[w]()
