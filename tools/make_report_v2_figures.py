"""
Figures for the updated project report (docs/report_v2/), drawn at the report's text width (6.2 in) from the committed data pack only; no model is run:

    .venv\\Scripts\\python.exe tools/make_report_v2_figures.py

Writes docs/report_v2/figures/r_*.png. Three figures of the first report (fig_pipeline, gd_rank1_by_band, fig_additive) are copied unchanged from docs/report/figures/.
Same palette as the journal figures (colour follows the entity); titles are left to the LaTeX captions.
"""
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "docs", "Research_Journal", "packs", "cross_model_transfer")
OLD = os.path.join(ROOT, "docs", "report", "figures")
OUT = os.path.join(ROOT, "docs", "report_v2", "figures")

SURFACE, INK, INK2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, RED, MAGENTA, YELLOW, GREY, DGREY, GREEN = "#2a78d6", "#d03a2f", "#e87ba4", "#eda100", "#b9b8b2", "#6f6e69", "#0a7a5a"
W = 6.2

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": MUTED, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "font.family": "DejaVu Sans", "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})


def load(name):
    return json.load(open(os.path.join(PACK, name), encoding="utf8"))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", name)


def label_bar(ax, y, v, fmt="{:.0f}%", dx=1.2):
    ax.text(v + dx, y, fmt.format(v), va="center", fontsize=7.5, color=INK, zorder=5, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6))


# ------------------------------------------------------------------------------------------------------------------------------------------------
def fig_overview():
    fig, ax = plt.subplots(figsize=(W, 2.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, body, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", linewidth=1.1, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=8, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.8, body, ha="center", va="top", fontsize=6.8, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1, color, text, ty):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, linewidth=1.3, color=color))
        ax.text((x0 + x1) / 2, y0 + ty, text, ha="center", va="center", fontsize=6.5, color=color)

    box(0.5, 12, 27, 30, "GPT-2 small", "12 layers, width 768\nSAE at layer 8\n(24,576 features)\n\nEdit: 200 multipliers\nchange h_S by Delta_S", "#eaf2fc", BLUE)
    box(36.5, 12, 27, 30, "Translator", "linear map W,\n768 to 1024\n(fitted on ordinary text,\nor trained on edits)\n\nno SAE on medium", "#f3f3f1", DGREY)
    box(72.5, 12, 27, 30, "GPT-2 medium", "24 layers, width 1024\nno weight change\n\nExport: add dose x\n(Delta_S W) at layer 16\nImport: read h_M\nat layer 16", "#e6f4ee", GREEN)
    arrow(28.2, 31, 35.8, 31, RED, "Export", 2.6)
    arrow(64.2, 31, 71.8, 31, RED, "", 0)
    arrow(71.8, 21, 64.2, 21, BLUE, "Import", -2.6)
    arrow(35.8, 21, 28.2, 21, BLUE, "", 0)
    ax.text(50, 6.3, "Export: the change an edit makes in small is translated and added to medium while it reads the same sentence.", ha="center", fontsize=6.9, color=INK2)
    ax.text(50, 2.2, "Import: medium's state is translated and mixed into small's layer 8. No weights change in either direction.", ha="center", fontsize=6.9, color=INK2)
    save(fig, "r_overview")


def fig_swap():
    g = {}
    for fn in ("b1_swap_primary.json",):
        g.update(load(fn)["gates"])
    rows = [("Export  (small 8 to medium 16)", g["s2m_L8_L16"]), ("Import  (medium 16 to small 8)", g["m2s_L16_L8"])]
    fig, ax = plt.subplots(figsize=(W, 1.9))
    for i, (name, d) in enumerate(rows):
        v = 100 * d["translated_hit"]
        c = 100 * max(d["controls_best_alpha_hit"].values())
        ax.barh(i, v, color=BLUE if i == 0 else GREEN, height=0.5)
        ax.text(v + 1.2, i, f"{v:.0f}%   (best control: {c:.1f}%)", va="center", fontsize=7.5, color=INK, zorder=5)
        ax.scatter([c], [i], color=INK2, marker="|", s=110, linewidths=1.8, zorder=4)
    ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 80)
    ax.set_xlabel("test cases where the swapped capital becomes the receiver's top answer (%)"); ax.grid(axis="y", visible=False)
    save(fig, "r_swap")


def fig_export_doses():
    d = load("d_analysis_linear.json")["s2m_L8_L16"]["small_ok"]
    cu = d["curves"]
    doses = cu["doses"]
    spec = [("word_push", "word push (uses the answer)", RED, "s"), ("translated", "translated edit", BLUE, "o"), ("randtarget", "edit toward a random word", MAGENTA, "^"),
            ("wrong_recipe", "another record's edit", YELLOW, "D"), ("random", "random vector", GREY, "v")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7))
    for arm, lab, col, mk in spec:
        axes[0].plot(doses, cu[arm]["median_rank"], color=col, marker=mk, markersize=3.6, linewidth=1.4, markeredgecolor=SURFACE, markeredgewidth=0.6, label=lab, zorder=3 if arm == "translated" else 2)
        axes[1].plot(doses, [100 * v for v in cu[arm]["top1"]], color=col, marker=mk, markersize=3.6, linewidth=1.4, markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3 if arm == "translated" else 2)
    axes[0].set_yscale("log")
    axes[0].axhline(cu["baseline_median_rank"], color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9, label=f"unchanged medium (median rank {cu['baseline_median_rank']:.0f})")
    axes[0].set_ylabel("median rank of the target\n(lower is better)"); axes[1].set_ylabel("target is medium's top answer (%)")
    axes[0].set_title("(a) rank of the target", loc="left"); axes[1].set_title("(b) top-1 rate", loc="left")
    for ax in axes:
        ax.set_xlabel("dose"); ax.set_xticks(doses); ax.set_xticklabels([f"{x:g}" for x in doses], fontsize=6.8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "r_export_doses")


def fig_export_side():
    d = load("d_analysis_linear.json")["s2m_L8_L16"]["small_ok"]
    s2, ic = d["stage2"], d["in_context"]
    rows = [("medium unchanged", s2["medium unchanged"], GREY), ("translated edit", s2["translated"], BLUE), ("random vector", s2["random"], GREY), ("another record's edit", s2["wrong_recipe"], YELLOW),
            ("word push", s2["word_push"], RED), ("fact in medium's prompt", {"PS": ic["PS"], "NS": ic["NS"]}, DGREY), ("small with the edit", s2["small with the edit"], DGREY)]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
    for ax, col, title in ((axes[0], "PS", "(a) rewordings: new beats true"), (axes[1], "NS", "(b) neighbours: true beats new")):
        for i, (nm, v, c) in enumerate(rows):
            ax.barh(i, 100 * v[col], color=c, height=0.62)
            label_bar(ax, i, 100 * v[col])
        ax.set_xlim(0, 112); ax.set_title(title, loc="left", fontsize=7.8); ax.grid(axis="y", visible=False); ax.set_xlabel("% of prompts")
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([r[0] for r in rows]); axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "r_export_side")


def fig_ladder():
    d6 = load("d6_b_aware.json")["summary"]["ok"]
    d7 = load("d7_output_matching.json")["summary"]["ok"]
    d5 = load("d5_multilayer.json")["gate4"]
    mlp = load("d_analysis_mlp.json")["s2m_L8_L16"]["small_ok"]["tuned"]["translated"]["top1"]
    d10 = load("d10_rank1_term.json")["summary"]["0.0"]["top1"]
    d13 = load("d13_training.json")["summary"]["1.0"]["top1"]
    rows = [("1. linear map fitted on text (baseline)", d6["top1"]["small_tuned"], BLUE), ("2. neural map fitted on text", mlp, BLUE), ("3. several injection layers", d5["best_set_top1"], BLUE),
            ("4. linear map trained on 289 edits", d7["top1"]["om_real"], GREEN), ("5. + rank-1 loss term (3 seeds)", d10, GREEN), ("6. + 1,221 more edits, 759 new words", d13, GREEN),
            ("7. dials re-tuned per sentence (ceiling)", d6["top1"]["b_aware"], RED)]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for i, (nm, v, c) in enumerate(rows):
        ax.barh(i, 100 * v, color=c, height=0.62)
        label_bar(ax, i, 100 * v)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 108)
    ax.axvline(50, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9); ax.text(50.8, -0.62, "bar for a working export (50%)", fontsize=6.8, color=INK2, va="bottom")
    ax.set_xlabel("target is medium's top answer on the 88 test records (%)"); ax.grid(axis="y", visible=False)
    save(fig, "r_ladder")


def fig_training_gap():
    d = load("d13_training.json")
    runs = d["runs"]
    pick = lambda r: [x for x in r["rows"] if x["dose"] == r["dose"]][0]
    order = [(0.0, "real edits\nonly"), (0.25, "+25%\nsynthetic"), (0.5, "+50%\nsynthetic"), (1.0, "+100%\nsynthetic")]
    xs = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for series, color, lab in (("train", BLUE, "its own training prompts"), ("test", GREEN, "unseen test records")):
        ys = []
        for j, (f, _) in enumerate(order):
            rs = [r for r in runs if r["fraction"] == f]
            v = [100 * r["train_top1_real"] for r in rs] if series == "train" else [100 * pick(r)["real"]["prim"][0] for r in rs]
            ys.append(sum(v) / len(v))
            ax.scatter([j] * len(v), v, color=color, alpha=0.3, s=12, zorder=3)
        ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
        for x, y in zip(xs, ys):
            ax.text(x - 0.12, y + (2.0 if series == "train" else -3.2), f"{y:.0f}%", ha="center", fontsize=7.5, color=INK, zorder=6)
        ax.text(xs[-1] + 0.22, ys[-1] + (0.8 if series == "train" else -0.8), lab, fontsize=7.2, color=BLUE if series == "train" else GREEN, va="center", ha="left")
    ax.set_xticks(xs); ax.set_xticklabels([o[1] for o in order]); ax.set_xlim(-0.3, 4.1); ax.set_ylim(0, 40); ax.grid(axis="x", visible=False)
    ax.set_ylabel("target is medium's top answer (%)")
    save(fig, "r_training_gap")


def fig_push():
    s2 = load("s2_d6_rewordings_test.json")["PRIMARY (original edit reached rank 1 in small)"]
    s3 = load("s3_random_word_baseline.json")["d7_map.pt"]["PRIMARY (original edit reached rank 1 in small)"]
    d7 = s3["arms"]
    groups = [("D6 (dials re-tuned per sentence)", 100 * s2["mean_real_lift"], 100 * s2["mean_rand_lift"], s2["p_wilcoxon_one_sided"]),
              ("D7 (map trained on edits)", 100 * d7["real target"]["ps_lift"], 100 * d7["random word (own)"]["ps_lift"], s3["p"])]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    for i, (nm, a, b, p) in enumerate(groups):
        for j, (v, c, lab) in enumerate(((a, BLUE, "real counterfactual target"), (b, MAGENTA, "random word"))):
            y = i + (j - 0.5) * 0.36
            ax.barh(y, v, color=c, height=0.32, label=lab if i == 0 else None)
            label_bar(ax, y, v, "{:.1f}", dx=0.5)
        ax.text(31.5, i, f"difference not significant\n(p = {p:.2g})", fontsize=6.8, color=INK2, va="center", ha="left")
    ax.set_yticks([0, 1]); ax.set_yticklabels([g[0] for g in groups]); ax.invert_yaxis(); ax.set_xlim(0, 48); ax.grid(axis="y", visible=False)
    ax.set_xlabel("gain on reworded prompts over the unchanged level of the same word (percentage points)")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "r_push")


def fig_sae():
    s1 = load("s1_basis_control.json")["PRIMARY (original edit reached rank 1 in small)"]
    d6 = load("d6_b_aware.json")["summary"]["ok"]
    rows = [("target top-1\n(real word)", d6["top1"]["b_aware"], s1["rb_real"]["top1"]), ("target top-1\n(random word)", d6["top1"]["random_word_b"], s1["rb_rand"]["top1"]),
            ("rewordings:\nnew beats true", d6["stage2"]["b_aware"]["PS"], s1["rb_real"]["PS"]), ("neighbours:\ntrue beats new", d6["stage2"]["b_aware"]["NS"], s1["rb_real"]["NS"])]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for i, (nm, a, b) in enumerate(rows):
        for j, (v, c, lab) in enumerate(((a, RED, "small's SAE directions"), (b, BLUE, "200 random directions"))):
            x = i + (j - 0.5) * 0.36
            ax.bar(x, 100 * v, width=0.32, color=c, label=lab if i == 0 else None)
            ax.text(x, 100 * v + 1.5, f"{100 * v:.0f}%", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(0, 112); ax.grid(axis="x", visible=False)
    ax.set_ylabel("% of the 88 test records / prompts"); ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.08))
    save(fig, "r_sae")


def fig_import():
    e = load("e5_softer_import.json")
    als = [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]
    B = lambda a, k: e["table"][f"P_{a}_None"]["B"][k]
    sB, mB = e["small_B"], e["medium_B"]
    xs = [0] + als
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7))
    ax = axes[0]
    for key, color, lab, y0 in (("survive_both", BLUE, "both knew: still right", 1.0), ("survive_small_only", RED, "only small knew: still right", 1.0), ("rescue", GREEN, "small wrong: now right", 0.0)):
        ys = [100 * y0] + [100 * B(a, key) for a in als]
        ax.plot(xs, ys, color=color, linewidth=1.6, marker="o", markersize=3.4, markeredgecolor=SURFACE, markeredgewidth=0.6, label=lab, zorder=4)
    ax.axvline(0.5, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.set_ylim(0, 108); ax.set_xlim(-0.03, 1.03); ax.set_xlabel("share of medium's translated state in the mix"); ax.set_ylabel("% of the facts in each group")
    ax.set_title("(a) what the mix keeps and fixes", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    ax = axes[1]
    ys = [100 * sB["acc"]] + [100 * B(a, "acc") for a in als]
    ax.plot(xs, ys, color=GREEN, linewidth=1.8, marker="o", markersize=3.8, markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=4)
    ax.axhline(100 * sB["acc"], color=DGREY, linestyle=(0, (4, 3)), linewidth=0.9); ax.axhline(100 * mB["acc"], color=GREY, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.text(0.02, 100 * sB["acc"] - 0.1, "small alone 8.5%", ha="left", va="top", fontsize=6.8, color=INK2)
    ax.text(0.0, 100 * mB["acc"] + 0.07, "medium alone 10.2%", ha="left", va="bottom", fontsize=6.8, color=INK2)
    ax.text(0.5, ys[xs.index(0.5)] + 0.2, f"{ys[xs.index(0.5)]:.1f}%", ha="center", fontsize=7.2)
    ax.text(0.97, ys[-1] - 0.12, f"{ys[-1]:.1f}%", ha="right", va="top", fontsize=7.2)
    ax.set_ylim(6.8, 10.8); ax.set_xlim(-0.03, 1.03); ax.set_xlabel("share of medium's translated state in the mix"); ax.set_ylabel("% of the test facts answered correctly")
    ax.set_title("(b) net accuracy (half B)", loc="left")
    ax.set_yticks([7, 8, 9, 10])
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.27, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "r_import")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in ("fig_pipeline.png", "gd_rank1_by_band.png", "fig_additive.png"):
        shutil.copy(os.path.join(OLD, f), os.path.join(OUT, f))
        print("copied", f)
    for fn in (fig_overview, fig_swap, fig_export_doses, fig_export_side, fig_ladder, fig_training_gap, fig_push, fig_sae, fig_import):
        fn()
