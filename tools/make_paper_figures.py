"""
Regenerate the data-driven figures used in research/paper/draft_v1/main.tex from files that are in the repository.

    python tools/make_paper_figures.py

Inputs (all tracked in the repo):
  benchmark_results/layer_benchmark_20260909_summary.csv                      (tools/summarize_layer_benchmark.py)
  benchmark_results/layer_benchmark_20260909_143704_240449.json               (single MIT prompt, layers 2/5/8/10)
  benchmark_results/candidate_source_study/batch_paired_summary.csv           (Journal Entry 20)
  docs/Research_Journal/packs/entry21_pilot_*/tables/arms_all.csv             (safety-filter pilot pack)
  docs/Research_Journal/packs/entry21_full_last_*/tables/arms_last.csv        (safety-filter full last-token pack)
Outputs: research/paper/draft_v1/figures/*.pdf and *.png
"""
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research", "paper", "draft_v1", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42})
BLUE, ORANGE, GRAY, GREEN = "#2563eb", "#ea580c", "#9ca3af", "#16a34a"


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_layers():
    rows = read(os.path.join(ROOT, "benchmark_results", "layer_benchmark_20260909_summary.csv"))
    layers = [int(r["layer"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax[0].bar(layers, [float(r["mean_rank_gain"]) for r in rows], color=BLUE)
    ax[0].set_xlabel("SAE layer (hook_resid_pre)")
    ax[0].set_ylabel("mean rank gain")
    ax[0].set_xticks(layers)
    ax[0].set_title("(a) mean rank gain", fontsize=8)
    w = 0.4
    ax[1].bar([l - w / 2 for l in layers], [int(r["improved_rank"]) for r in rows], width=w, color=GRAY, label="rank improved")
    ax[1].bar([l + w / 2 for l in layers], [int(r["reached_rank1"]) for r in rows], width=w, color=GREEN, label="reached rank 1")
    ax[1].set_xlabel("SAE layer (hook_resid_pre)")
    ax[1].set_ylabel("prompts (of 12)")
    ax[1].set_xticks(layers)
    ax[1].set_title("(b) prompts improved / at rank 1", fontsize=8)
    ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_layer12")


def fig_mit_layers():
    d = json.load(open(os.path.join(ROOT, "benchmark_results", "layer_benchmark_20260909_143704_240449.json"), encoding="utf-8"))
    rows = [l for l in d["prompts"][0]["layers"] if "error" not in l]
    layers = [int(l["layer"]) for l in rows]
    gain = [int(l["clean_rank"]) - int(l["final_rank"]) for l in rows]
    dp = [100 * (float(l["final_probability"]) - float(l["clean_probability"])) for l in rows]
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.9))
    xs = range(len(layers))
    ax[0].bar(xs, gain, color=GREEN)
    ax[0].set_xticks(list(xs))
    ax[0].set_xticklabels([str(x) for x in layers])
    ax[0].set_xlabel("layer")
    ax[0].set_ylabel("rank places gained")
    for x, g in zip(xs, gain):
        ax[0].text(x, g + 0.1, f"+{g}", ha="center", fontsize=6.5)
    ax[1].bar(xs, dp, color=BLUE)
    ax[1].set_xticks(list(xs))
    ax[1].set_xticklabels([str(x) for x in layers])
    ax[1].set_xlabel("layer")
    ax[1].set_ylabel("probability gain (pp)")
    for x, g in zip(xs, dp):
        ax[1].text(x, g + 0.05, f"{g:.2f}", ha="center", fontsize=6)
    fig.tight_layout()
    save(fig, "fig_mit_layers")


def fig_candidate_source():
    rows = [r for r in read(os.path.join(ROOT, "benchmark_results", "candidate_source_study", "batch_paired_summary.csv"))
            if r["single_token"] == "True" and int(r["baseline_rank"]) > 1]
    short = {"This is Sophia, she is a": "Sophia", "Seiyu Group's headquarters are in": "Seiyu HQ", "The Eiffel Tower is in the city of": "Eiffel Tower",
             "Marie Curie won the Nobel Prize in": "Marie Curie", "The Colosseum is located in": "Colosseum", "Barack Obama was born in": "Obama",
             "She opened the door and saw a": "door / saw a", "The programming language created by Guido van Rossum is": "Guido van Rossum",
             "The doctor said that": "doctor said that", "Mount Everest is located in": "Mount Everest", "The CEO of Tesla is": "CEO of Tesla",
             "Albert Einstein was born in": "Einstein", "The largest planet in our solar system is": "largest planet"}
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    n = len(rows)
    for i, r in enumerate(rows):
        y = n - i
        base, a, l = int(r["baseline_rank"]), int(r["all_final_rank"]), int(r["last_final_rank"])
        ax.plot([a, l], [y, y], color="#d1d5db", lw=1, zorder=1)
        ax.scatter([base], [y], marker="|", s=40, color=GRAY, zorder=2, label="clean baseline" if i == 0 else None)
        ax.scatter([a], [y], s=16, color=BLUE, zorder=3, label="all positions" if i == 0 else None)
        ax.scatter([l], [y], s=16, color=ORANGE, zorder=3, label="last token" if i == 0 else None)
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([short.get(r["prompt"], r["prompt"][:24]) for r in rows], fontsize=6.5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0.8, 500)
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax.set_xlabel("target rank (lower is better)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_candidate_source")


def fig_filter():
    pilot = read(glob.glob(os.path.join(ROOT, "docs", "Research_Journal", "packs", "entry21_pilot_*", "tables", "arms_all.csv"))[0])
    full = read(glob.glob(os.path.join(ROOT, "docs", "Research_Journal", "packs", "entry21_full_last_*", "tables", "arms_last.csv"))[0])
    labels = {"strict": "strict", "off": "no filter", "tol5_after": "tolerance 5%", "graded5_after": "graded 5%\n(after)", "graded5_inter": "graded 5%\n(interleaved)"}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for a, rows, title in ((ax[0], pilot, "(a) all prompt positions"), (ax[1], full, "(b) last token only")):
        n = int(rows[0]["Prompts"])
        vals = [float(r["Success rate %"]) for r in rows]
        cols = [GRAY if r["Arm"] == "strict" else BLUE for r in rows]
        bars = a.bar(range(len(rows)), vals, color=cols)
        for b, r in zip(bars, rows):
            a.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{r['Reached rank #1']}/{r['Prompts']}", ha="center", fontsize=7)
        a.set_xticks(range(len(rows)))
        a.set_xticklabels([labels[r["Arm"]] for r in rows], fontsize=6.5)
        a.set_ylim(0, 100)
        a.set_title(f"{title}, n = {n}", fontsize=8)
    ax[0].set_ylabel("prompts reaching rank 1 (%)")
    fig.tight_layout()
    save(fig, "fig_filter")


if __name__ == "__main__":
    fig_layers()
    fig_mit_layers()
    fig_candidate_source()
    fig_filter()
    print("figures written to", OUT, sorted(os.listdir(OUT)))
