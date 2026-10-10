"""
Figures for Research Journal Entry 33 (cross-model transfer, part 3: several layers, the ceiling test), drawn only from the committed data pack
docs/Research_Journal/packs/cross_model_transfer/ (no model is run):

    .venv\\Scripts\\python.exe tools/make_transfer_figures_33.py

e33_fig1_layers.*    step D5: top-1 rate and median rank of the target in medium for each injection configuration (one layer, several layers split, several full)
e33_fig2_ceiling.*   step D6: the edit tuned through the translator against the original edit and the controls: top-1, rewordings, neighbours
e33_fig3_trained_map.*  step D7: the translator trained on edits (output matching) against the ridge map, the controls and the D6 ceiling
e33_fig4_dose_curves.*  step D9b: dev and test top-1 against the dose for maps trained 15 and 40 epochs (3 seeds each)
e33_fig5_datasize.*     step D8: top-1 and rank gain against the amount of training data
e33_fig6_training_curve.*  step D9: training loss and dev curves over 40 epochs (parsed from the saved terminal log)
e33_fig7_ladder.*       summary: top-1 of every export variant on the same 88 test records, with the 15% and 50% lines

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
BLUE, PURPLE, MAGENTA, YELLOW, GREY, DGREY, AQUA = "#2a78d6", "#d03a2f", "#e87ba4", "#eda100", "#b9b8b2", "#6f6e69", "#0a7a5a"

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


def fig3():
    d7 = load("d7_output_matching.json")["summary"]["ok"]
    d6 = load("d6_b_aware.json")["summary"]["ok"]
    s2, t1, n = d7["stage2"], d7["top1"], d7["n"]
    rows = [("medium unchanged", 0.0, s2["medium unchanged"], GREY),
            ("ridge map, original edit (D3)", t1["ridge_real"], s2["ridge_real"], BLUE),
            ("map trained on edits (D7)", t1["om_real"], s2["om_real"], AQUA),
            ("trained map, toward a random word\n(judged on that word)", t1["om_rand"], s2["om_rand"], MAGENTA),
            ("trained map, another record's recipe", t1["om_wrong"], s2["om_wrong"], YELLOW),
            ("D6 ceiling: tuned through the translator", d6["top1"]["b_aware"], d6["stage2"]["b_aware"], PURPLE),
            ("small, with the edit itself", 1.0, s2["small with the edit"], DGREY)]
    panels = [("Target is the top answer\non the tuned prompt (%)", lambda r: r[1]), ("Rewordings: new beats true (%)\nhigher = more of the new answer", lambda r: r[2]["PS"]),
              ("Neighbours: true still beats new (%)\nhigher = fewer leaks", lambda r: r[2]["NS"])]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=True)
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
    fig.suptitle("A translator trained on edits roughly doubles top-1 over the ridge map (8% to 18%) with nothing tuned on medium, but stays far below the 50% bar and the 95% ceiling",
                 x=0.01, ha="left", fontsize=10.8, color=INK, wrap=True)
    fig.text(0.01, -0.035, f"Dashed line = small without the edit. {n} test records whose original edit reached rank 1 in small; trained map small 8 to medium 16, dose 2.0 and epoch chosen on 50 dev records; "
             "trained on 289 edits whose target words and subjects are not those of any dev or test record.\nThe D6 row tunes against medium's output for every sentence (a ceiling, not a transfer). Median rank of the target: unchanged 142, ridge 14, trained map 7.",
             fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    save(fig, "e33_fig3_trained_map")


def fig4():
    d = load("d9b_dose_table.json")
    doses = [r["dose"] for r in d["runs"][0]["rows"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharex=True)
    for ax, which, title in ((axes[0], "dev", "Dev records (28): what the dose rule saw"), (axes[1], "test", "Test records (88): how the dose really did")):
        for h, color, name in ((15, BLUE, "trained 15 epochs"), (40, AQUA, "trained 40 epochs")):
            runs = [r for r in d["runs"] if r["horizon"] == h]
            ys_all = [[100 * row[which][0] for row in r["rows"]] for r in runs]
            for ys in ys_all:
                ax.plot(doses, ys, color=color, alpha=0.35, linewidth=1.2, marker="o", markersize=3.5)
            mean = [sum(col) / len(col) for col in zip(*ys_all)]
            ax.plot(doses, mean, color=color, linewidth=2.4, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label=name + " (mean of 3 seeds)")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("dose (how much of the translated edit is added)")
        ax.set_xticks(doses)
        ax.set_ylabel("target is medium's top answer (%)")
    axes[0].annotate("with 40 epochs, 3 dev records (11%) put\ndose 1.0 first, so the rule picked it", xy=(1.0, 11), xytext=(1.35, 14.5), fontsize=8.5, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=MUTED))
    axes[1].annotate("dose 1.0 gives 2% on test;\ndose 2.0 gives about 20% for both lengths", xy=(1.0, 2), xytext=(0.55, 12), fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    axes[0].set_ylim(0, 26)
    axes[1].set_ylim(0, 26)
    axes[1].legend(frameon=False, fontsize=8.8, loc="upper left")
    fig.suptitle("Longer training did not help, and it did not hurt: the 17-point drop in the first test came from the dose the 28 dev records picked", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, -0.03, "Thin lines are single seeds. Test top-1 of the ridge map at dose 2.0 on the same records: 8%. Top-1 on dev moves in steps of 3.6 points (one record).", fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "e33_fig4_dose_curves")


def fig5():
    d = load("d8_datasize.json")
    fr = sorted(d["runs"], key=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, key, ylabel, title in ((axes[0], "top1", "target is medium's top answer (%)", "Top-1 against the amount of training data"),
                                   (axes[1], "gain", "mean log-rank gain of the target (higher = better)", "Rank gain against the amount of training data")):
        xs_mean, ys_mean = [], []
        for f in fr:
            runs = d["runs"][f]
            n = runs[0]["n_records"]
            ys = [(100 * r["primary"]["top1"]) if key == "top1" else r["primary"]["gain"] for r in runs]
            ax.scatter([n] * len(ys), ys, color=AQUA, alpha=0.5, s=28, zorder=3)
            xs_mean.append(n)
            ys_mean.append(sum(ys) / len(ys))
        ax.plot(xs_mean, ys_mean, color=AQUA, linewidth=2.4, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4, label="map trained on edits (mean of 3 seeds)")
        ref = d["reference_ridge_primary"]
        ref_y = 100 * ref["top1"] if key == "top1" else ref["gain"]
        ax.axhline(ref_y, color=BLUE, linestyle=(0, (4, 3)), linewidth=1.3, label="ridge map, same edits")
        if key == "top1":
            ax.axhline(50, color=MUTED, linestyle=(0, (1, 2)), linewidth=1.3)
            ax.text(xs_mean[0], 51.5, "Gate 6 line (50%)", fontsize=8.5, color=INK2)
            ax.set_ylim(0, 58)
        ax.set_xticks(xs_mean)
        ax.set_xlabel("training edits (records)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[0].legend(frameon=False, fontsize=8.8, loc="center right")
    fig.suptitle("More training edits help a little but unevenly: flat from 72 to 144 records, then a jump at 289", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, -0.03, "88 primary test records, 3 seeds per fraction (a different random subset per seed below 100%; the 100% seeds share all their data and differ only in batch order, so their spread understates the noise).",
             fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "e33_fig5_datasize")


def fig6():
    import re
    text = open(os.path.join(PACK, "logs", "d9_longer_training.log"), encoding="utf8").read()
    rows = [(int(a), float(b), float(c), float(e), int(f)) for a, b, c, e, f in re.findall(r"epoch\s+(\d+): train KL ([\d.]+), dev top-1\s+(\d+)%, gain ([\d.]+), median rank (\d+)", text)]
    ep, kl, gain, rank = [r[0] for r in rows], [r[1] for r in rows], [r[3] for r in rows], [r[4] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax, ys, color, title, ylabel in ((axes[0], kl, AQUA, "Training loss keeps falling", "training KL (lower = fits the training edits better)"),
                                         (axes[1], gain, AQUA, "Dev gain flattens", "dev mean log-rank gain at dose 1"),
                                         (axes[2], rank, AQUA, "Dev median rank flattens", "dev median rank of the target (lower = better)")):
        ax.plot(ep, ys, color=color, linewidth=2.2)
        ax.axvline(15, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel, fontsize=8.8)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[0].text(15.6, max(kl) * 0.93, "earlier runs\nstopped here (15)", fontsize=8.5, color=INK2, va="top")
    fig.suptitle("Longer training fits the 289 training edits ever better but barely moves the dev records", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, -0.03, "Seed 0, dev set of 28 records, ridge-map dose 1.0 (the dose used during training); read from the D9 terminal log. Test results at every dose are in Figure 4.", fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "e33_fig6_training_curve")


def fig7():
    d6 = load("d6_b_aware.json")["summary"]["ok"]
    d7 = load("d7_output_matching.json")["summary"]["ok"]
    d5 = load("d5_multilayer.json")["gate4"]
    d9b = load("d9b_dose_table.json")["runs"]
    mean15 = [r["picked_test"]["B (best dev gain)"]["top1"] for r in d9b if r["horizon"] == 15]
    rows = [("medium unchanged", 0.0, GREY),
            ("D3: original edit through the ridge map", d6["top1"]["small_tuned"], BLUE),
            ("D5: best layer set chosen on dev (12+16)", d5["best_set_top1"], BLUE),
            ("D7: map trained on 289 edits", d7["top1"]["om_real"], AQUA),
            ("D8/D9b: same, 3 seeds, dose 2.0 (mean)", sum(mean15) / len(mean15), AQUA),
            ("D6 ceiling: edit tuned through the translator", d6["top1"]["b_aware"], PURPLE),
            ("small, with the edit itself", 1.0, DGREY)]
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    for i, (name, v, color) in enumerate(rows):
        ax.barh(i, 100 * v, color=color, height=0.62)
        ax.text(100 * v + 1.2, i, f"{100 * v:.0f}%", va="center", fontsize=9.5, color=INK, zorder=5, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.invert_yaxis()
    for x, label in ((15, "old Gate 3 line (15%)"), (50, "bar for a working export (50%)")):
        ax.axvline(x, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.text(x + 0.8, -0.62, label, fontsize=8.5, color=INK2, va="bottom")
    ax.set_xlim(0, 112)
    ax.set_xlabel("target is medium's top answer on the tuned prompt (%), 88 test records whose edit reached rank 1 in small")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Where the export stands: 8% to about 20% without looking at medium, 95% when the edit is tuned against medium's output", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, -0.02, "D6 uses medium's output for every sentence (a ceiling, not a transfer). D8/D9b row: map trained 15 epochs, dose 2.0, three training seeds.", fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save(fig, "e33_fig7_ladder")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7}[w]()
