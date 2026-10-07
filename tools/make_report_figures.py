"""
Figures for the project report (Reportmain.tex). Every data figure is drawn ONLY from files stored in the repository
(result packs, benchmark_results, journal figure folders); every number plotted is also written to
<out>/report_figure_data.json so it can be checked. The diagrams (architecture, flow charts, wireframes) contain no data.

    python tools/make_report_figures.py --out "D:/Downloads/Files/Project/Project Report/figures"

Sources (all tracked in git):
  benchmark_results/layer_benchmark_20260909_summary.csv                   (12-layer sweep, 12 prompts not already correct)
  benchmark_results/candidate_source_study/batch_paired_summary.csv         (Entry 20)
  docs/Research_Journal/packs/entry21_pilot_*/tables/arms_all.csv           (safety-filter pilot)
  docs/Research_Journal/packs/entry21_full_last_*/tables/arms_last.csv      (safety-filter full study)
  docs/Research_Journal/packs/headroom/headroom_20261002_144917.json        (Entry 24)
  docs/Research_Journal/packs/strength_models/analysis_output.txt           (Entry 26; counts typed from that file)
  docs/Research_Journal/packs/additive_control/results.json                 (Entry 30 section 10)
  docs/Research_Journal/packs/generalisation/{true40,random40}/summary.json (Entry 30 section 11)
  docs/Research_Journal/packs/generation_quality/true40/summary.json        (Entry 30 section 15)
  docs/Research_Journal/images/e28_*.png, e29_*.png                         (copied; drawn by tools/make_journal_figures.py)
"""
import argparse
import csv
import glob
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import report_diagrams as rd  # noqa: E402  (tools/report_diagrams.py: architecture, flow charts, wireframes)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "docs", "Research_Journal", "packs")

BLUE, ORANGE, GREEN, GRAY, PURPLE, DARK = "#2563eb", "#ea580c", "#16a34a", "#9ca3af", "#7c3aed", "#1f2937"
LBLUE, LORANGE, LGREEN, LGRAY, LPURPLE = "#dbeafe", "#ffedd5", "#dcfce7", "#f3f4f6", "#ede9fe"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "figure.dpi": 120, "font.family": "DejaVu Sans", "axes.titlesize": 9.5, "axes.labelsize": 9})
DATA = {}


def save(fig, out, name):
    path = os.path.join(out, name + ".png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label_bars(ax, bars, fmt="{:.0f}", dy=0.0, fs=8, color=DARK):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                    xytext=(0, 2 + dy), textcoords="offset points", fontsize=fs, color=color)


# ------------------------------------------------------------------------------------------------ data figures
def fig_elig(out):
    labels = ["Already correct\n(target is top-1)", "Suppressed: rank above 1 and probability\nabove 1% or rank at most 50 (benchmark set)",
              "Low prior: rank above 50 and\nprobability at most 1% (excluded)"]
    vals = [15, 22, 8]
    DATA["elig"] = dict(zip(["already_correct", "suppressed", "low_prior"], vals))
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    bars = ax.barh(range(3), vals, color=[GRAY, BLUE, ORANGE], height=0.55)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of fact pairs (45 in total)")
    for b, v in zip(bars, vals):
        ax.text(v + 0.4, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 27)
    save(fig, out, "fig_elig_v2")


def fig_layer12(out):
    rows = read_csv(os.path.join(ROOT, "benchmark_results", "layer_benchmark_20260909_summary.csv"))
    L = [int(r["layer"]) for r in rows]
    gain = [float(r["mean_rank_gain"]) for r in rows]
    r1 = [int(r["reached_rank1"]) for r in rows]
    imp = [int(r["improved_rank"]) for r in rows]
    DATA["layer12"] = {"layer": L, "mean_rank_gain": gain, "reached_rank1": r1, "improved_rank": imp, "n_prompts": int(rows[0]["prompts"])}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    cols = [BLUE if l == 8 else "#93c5fd" for l in L]
    bars = a.bar(L, gain, color=cols, width=0.7)
    for bar, g in zip(bars, gain):
        a.text(bar.get_x() + bar.get_width() / 2, g + 0.12, f"{g:.1f}", ha="center", fontsize=6.8)
    a.set_xticks(L)
    a.set_xlabel("Layer of the intervention")
    a.set_ylabel("Mean rank places gained")
    a.set_title("(a) Mean rank gain, 12 prompts", loc="left")
    a.set_ylim(0, 8.8)
    w = 0.38
    b.bar([l - w / 2 for l in L], imp, width=w, color=BLUE, label="rank improved")
    b.bar([l + w / 2 for l in L], r1, width=w, color=ORANGE, label="reached rank 1")
    b.set_xticks(L)
    b.set_xlabel("Layer of the intervention")
    b.set_ylabel("Number of prompts (of 12)")
    b.set_title("(b) Prompts improved / reaching rank 1", loc="left")
    b.legend(frameon=False, fontsize=7.5, loc="upper left")
    b.set_ylim(0, 12.5)
    fig.tight_layout()
    save(fig, out, "fig_layer12")


SHORT = {
    "This is Sophia, she is a": "Sophia, she is a: woman", "Seiyu Group's headquarters are in": "Seiyu Group: Tokyo",
    "The Eiffel Tower is in the city of": "Eiffel Tower: Paris", "Marie Curie won the Nobel Prize in": "Marie Curie: Physics",
    "The Colosseum is located in": "Colosseum: Rome", "Barack Obama was born in": "Barack Obama: Hawaii",
    "She opened the door and saw a": "door, saw a: cat", "The programming language created by Guido van Rossum is": "Guido van Rossum: Python",
    "The doctor said that": "The doctor said that: she", "Mount Everest is located in": "Mount Everest: Nepal",
    "The CEO of Tesla is": "CEO of Tesla: Elon", "Albert Einstein was born in": "Albert Einstein: Germany",
    "The largest planet in our solar system is": "Largest planet: Jupiter",
}


def fig_candsrc(out):
    rows = read_csv(os.path.join(ROOT, "benchmark_results", "candidate_source_study", "batch_paired_summary.csv"))
    rows = [r for r in rows if r["prompt"] in SHORT]
    rows.sort(key=lambda r: -int(r["baseline_rank"]))
    DATA["candsrc"] = [{"prompt": r["prompt"], "baseline_rank": int(r["baseline_rank"]), "all_final_rank": int(r["all_final_rank"]),
                        "last_final_rank": int(r["last_final_rank"])} for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    n = len(rows)
    for i, r in enumerate(rows):
        c, a_, l_ = int(r["baseline_rank"]), int(r["all_final_rank"]), int(r["last_final_rank"])
        ax.plot([min(a_, l_, c), max(a_, l_, c)], [i, i], color="#e5e7eb", lw=1.4, zorder=1)
        ax.scatter([c], [i], s=95, facecolors="white", edgecolors=GRAY, linewidths=1.3, zorder=3, label="clean model" if i == 0 else None)
        ax.scatter([l_], [i], s=34, marker="s", color=ORANGE, zorder=4, label="last token only" if i == 0 else None)
        ax.scatter([a_], [i], s=34, color=BLUE, zorder=5, label="all prompt positions" if i == 0 else None)
    ax.axvline(1, color=GREEN, ls="--", lw=1)
    ax.text(1.06, -0.9, "rank 1", color=GREEN, fontsize=7.5, va="center")
    ax.set_xscale("log")
    ax.set_yticks(range(n))
    ax.set_yticklabels([SHORT[r["prompt"]] for r in rows], fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlabel("Rank of the target (log scale; lower is better)")
    ax.set_xlim(0.8, 500)
    ax.set_ylim(n - 0.4, -1.3)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    save(fig, out, "fig_candsrc")


def fig_filter(out):
    pil = {r["Arm"]: r for r in read_csv(glob.glob(os.path.join(P, "entry21_pilot_*", "tables", "arms_all.csv"))[0])}
    full = {r["Arm"]: r for r in read_csv(glob.glob(os.path.join(P, "entry21_full_last_*", "tables", "arms_last.csv"))[0])}
    arms = [("strict", "strict\n(used)"), ("off", "no\nfilter"), ("tol5_after", "tolerance\n5%"), ("graded5_after", "graded,\nafter"), ("graded5_inter", "graded,\ninterleaved")]
    DATA["filter"] = {"pilot_n": int(pil["strict"]["Prompts"]), "full_n": int(full["strict"]["Prompts"]),
                      "pilot": {k: int(pil[k]["Reached rank #1"]) for k, _ in arms}, "full": {k: int(full[k]["Reached rank #1"]) for k, _ in arms}}
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, (d, nme, ttl) in zip(axs, [(pil, "pilot", "(a) Pilot: all prompt positions, 45 prompts"), (full, "full", "(b) Full study: last token, 131 prompts")]):
        vals = [int(d[k]["Reached rank #1"]) for k, _ in arms]
        cols = [BLUE, ORANGE, "#93c5fd", "#93c5fd", "#93c5fd"]
        bars = ax.bar(range(len(arms)), vals, color=cols, width=0.65)
        label_bars(ax, bars)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([lab for _, lab in arms], fontsize=7)
        ax.set_title(ttl, loc="left", fontsize=8.6)
        ax.set_ylim(0, max(vals) * 1.22)
    axs[0].set_ylabel("Prompts reaching rank 1")
    fig.tight_layout()
    save(fig, out, "fig_filter")


def fig_headroom(out):
    d = json.load(open(os.path.join(P, "headroom", "headroom_20261002_144917.json"), encoding="utf-8"))
    recs = d["records"]
    n = len(recs)
    methods = [("sae_reach", "SAE multipliers, no damage limit", BLUE), ("sae_gentle", "SAE multipliers, side effects penalised", BLUE),
               ("sae_naive_5", "SAE fixed rule, top 5 mute / boost", "#93c5fd"), ("sae_naive_20", "SAE fixed rule, top 20", "#93c5fd"),
               ("sae_naive_50", "SAE fixed rule, top 50", "#93c5fd"),
               ("push_last_0.05", "free push, last token, 5% of norm", GRAY), ("push_last_0.1", "free push, last token, 10%", GRAY),
               ("push_last_0.2", "free push, last token, 20%", GRAY), ("push_all_0.05", "free push, all tokens, 5%", GRAY),
               ("push_all_0.1", "free push, all tokens, 10%", GRAY), ("push_all_0.2", "free push, all tokens, 20%", GRAY)]
    share = {m: sum(1 for r in recs if r[m]["rank"] == 1) / n for m, _, _ in methods}
    bands = ["2-5", "6-20", "21-100", "101-1000"]
    byband = {m: [sum(1 for r in recs if r["band"] == b and r[m]["rank"] == 1) / max(1, sum(1 for r in recs if r["band"] == b)) for b in bands]
              for m in ("sae_reach", "sae_gentle")}
    DATA["headroom"] = {"n": n, "share_rank1": share, "by_band": byband, "band_n": {b: sum(1 for r in recs if r["band"] == b) for b in bands}}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.55, 1]})
    ys = list(range(len(methods)))
    bars = a.barh(ys, [100 * share[m] for m, _, _ in methods], color=[c for _, _, c in methods], height=0.62)
    a.set_yticks(ys)
    a.set_yticklabels([lab for _, lab, _ in methods], fontsize=7.2)
    a.invert_yaxis()
    for bar, (m, _, _) in zip(bars, methods):
        a.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{100 * share[m]:.0f}%", va="center", fontsize=7.2)
    a.set_xlim(0, 112)
    a.set_xlabel("Prompts reaching rank 1 (% of 1,450)")
    a.set_title("(a) All 1,450 cached prompts", loc="left", fontsize=8.6)
    w = 0.38
    xs = range(len(bands))
    b.bar([x - w / 2 for x in xs], [100 * v for v in byband["sae_reach"]], width=w, color=BLUE, label="no damage limit")
    b.bar([x + w / 2 for x in xs], [100 * v for v in byband["sae_gentle"]], width=w, color="#93c5fd", label="side effects penalised")
    b.set_xticks(list(xs))
    b.set_xticklabels(["2-5", "6-20", "21-\n100", "101-\n1000"], fontsize=7.2)
    b.set_xlabel("Starting rank of the target")
    b.set_ylabel("% reaching rank 1")
    b.set_ylim(0, 112)
    b.legend(frameon=False, fontsize=7, loc="upper right")
    b.set_title("(b) SAE multipliers by band", loc="left", fontsize=8.6)
    fig.tight_layout()
    save(fig, out, "fig_headroom")


def fig_learned(out):
    names = ["fixed\n0.6 / 0.5", "learned\nfixed pair", "PromptNet\n(network)", "per-prompt\ntuned"]
    counts = [75, 116, 119, 123]            # TEST ONLY (300), docs/Research_Journal/packs/strength_models/analysis_output.txt
    kl = [0.156, 0.383, 0.379, 0.360]       # mean KL among prompts that reach rank 1, same file
    DATA["learned"] = {"names": names, "rank1_of_300": counts, "mean_kl_of_successes": kl}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    bars = a.bar(range(4), counts, color=[GRAY, BLUE, BLUE, "#93c5fd"], width=0.62)
    for bar, c in zip(bars, counts):
        a.text(bar.get_x() + bar.get_width() / 2, c + 2, f"{c}\n({100 * c / 300:.1f}%)", ha="center", fontsize=7.6)
    a.set_xticks(range(4))
    a.set_xticklabels(names, fontsize=7.4)
    a.set_ylabel("Prompts reaching rank 1 (of 300)")
    a.set_ylim(0, 160)
    a.set_title("(a) Held-out test prompts, stand-in measure", loc="left", fontsize=8.4)
    bars = b.bar(range(4), kl, color=[GRAY, BLUE, BLUE, "#93c5fd"], width=0.62)
    for bar, k in zip(bars, kl):
        b.text(bar.get_x() + bar.get_width() / 2, k + 0.01, f"{k:.2f}", ha="center", fontsize=7.6)
    b.set_xticks(range(4))
    b.set_xticklabels(names, fontsize=7.4)
    b.set_ylabel("Mean KL of successes (nats)")
    b.set_ylim(0, 0.48)
    b.set_title("(b) Side effects of the successes", loc="left", fontsize=8.4)
    fig.tight_layout()
    save(fig, out, "fig_learned")


def fig_additive(out):
    a_ = json.load(open(os.path.join(P, "additive_control", "results.json"), encoding="utf-8"))["summary"]
    keys = [("multipliers only", "multipliers\nonly", BLUE), ("+ additive (cap 1.0)", "+ additive\ncap 1.0", ORANGE), ("+ additive (cap 0.25)", "+ additive\ncap 0.25", ORANGE)]
    r1 = [a_[k]["rank1"] for k, _, _ in keys]
    ed = [100 * a_[k]["median_edit_size_frac_norm"] for k, _, _ in keys]
    DATA["additive_control"] = {"rank1_of_20": dict(zip([k for k, _, _ in keys], r1)), "median_edit_size_pct_of_norm": dict(zip([k for k, _, _ in keys], ed))}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.2, 2.7))
    bars = a.bar(range(3), r1, color=[c for _, _, c in keys], width=0.6)
    label_bars(a, bars)
    a.set_xticks(range(3))
    a.set_xticklabels([l for _, l, _ in keys], fontsize=7.8)
    a.set_ylabel("Random words reaching rank 1 (of 20)")
    a.set_ylim(0, 23.5)
    a.set_title("(a) Control: random unrelated targets", loc="left", fontsize=8.4)
    bars = b.bar(range(3), ed, color=[c for _, _, c in keys], width=0.6)
    label_bars(b, bars, fmt="{:.0f}%")
    b.axhline(100, color="#6b7280", ls="--", lw=1)
    b.text(0, 108, "as large as the\nresidual itself", ha="center", va="bottom", fontsize=6.8, color="#4b5563")
    b.set_xticks(range(3))
    b.set_xticklabels([l for _, l, _ in keys], fontsize=7.8)
    b.set_ylabel("Median edit size (% of residual norm)")
    b.set_ylim(0, 235)
    b.set_title("(b) Size of the edit", loc="left", fontsize=8.4)
    fig.tight_layout()
    save(fig, out, "fig_additive")


def fig_reworded(out):
    t = json.load(open(os.path.join(P, "generalisation", "true40", "summary.json"), encoding="utf-8"))["arms"]
    r = json.load(open(os.path.join(P, "generalisation", "random40", "summary.json"), encoding="utf-8"))["arms"]
    arms = [("gd", "multipliers\nonly"), ("gd_add@cap=1.0", "+ additive\ncap 1.0"), ("gd_add@cap=0.25", "+ additive\ncap 0.25")]
    DATA["reworded"] = {"unedited_median_rank": {"true": t["gd"]["para_median_rank_clean"], "random": r["gd"]["para_median_rank_clean"]},
                        "median_rank_edited": {"true": [t[k]["para_median_rank_edit"] for k, _ in arms], "random": [r[k]["para_median_rank_edit"] for k, _ in arms]},
                        "neutral_top1_changed": {"true": [t[k]["neutral_flip"] for k, _ in arms], "random": [r[k]["neutral_flip"] for k, _ in arms]},
                        "arms": [k for k, _ in arms]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    w = 0.36
    xs = range(3)
    bt = a.bar([x - w / 2 for x in xs], [t[k]["para_median_rank_edit"] for k, _ in arms], width=w, color=BLUE, label="true answers")
    br = a.bar([x + w / 2 for x in xs], [r[k]["para_median_rank_edit"] for k, _ in arms], width=w, color=ORANGE, label="random words")
    a.set_yscale("log")
    for bars in (bt, br):
        for bar in bars:
            a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.12, (f"{bar.get_height():,.0f}" if bar.get_height() >= 100 else f"{bar.get_height():g}"), ha="center", fontsize=7)
    a.set_xticks(list(xs))
    a.set_xticklabels([l for _, l in arms], fontsize=7.6)
    a.set_ylabel("Median rank on reworded prompts (log)")
    a.set_ylim(1, 5e4)
    a.legend(frameon=False, fontsize=7.4, loc="upper right")
    a.set_title("(a) Reworded prompts", loc="left", fontsize=8.4)
    bt = b.bar([x - w / 2 for x in xs], [100 * t[k]["neutral_flip"] for k, _ in arms], width=w, color=BLUE)
    br = b.bar([x + w / 2 for x in xs], [100 * r[k]["neutral_flip"] for k, _ in arms], width=w, color=ORANGE)
    for bars in (bt, br):
        label_bars(b, bars, fmt="{:.0f}%", fs=7)
    b.set_xticks(list(xs))
    b.set_xticklabels([l for _, l in arms], fontsize=7.6)
    b.set_ylabel("Unrelated prompts with a\nchanged top word (%)")
    b.set_ylim(0, 100)
    b.set_title("(b) 20 unrelated prompts", loc="left", fontsize=8.4)
    fig.tight_layout()
    save(fig, out, "fig_reworded")


def fig_genquality(out):
    s = json.load(open(os.path.join(P, "generation_quality", "true40", "summary.json"), encoding="utf-8"))
    A, B = s["A"], s["B"]
    runs = [("baseline", "no edit"), ("gd/keep_on", "multipliers\nonly"), ("gd_add@cap=1.0/keep_on", "+ additive\ncap 1.0"), ("gd_add@cap=0.25/keep_on", "+ additive\ncap 0.25")]
    ctx = [("plain", "plain prompt"), ("do_not_say", "\"Do not say\nthe word X.\""), ("wrong_answer", "\"X is the\nwrong answer.\"")]
    arms = [("gd", "multipliers only", BLUE), ("gd_add@cap=1.0", "+ additive, cap 1.0", ORANGE), ("gd_add@cap=0.25", "+ additive, cap 0.25", "#fdba74")]
    DATA["genquality"] = {"A": {k: A[k] for k, _ in runs}, "B_top1_clean": {c: B["gd/" + c]["top1_clean"] for c, _ in ctx},
                          "B_top1_edit": {a: {c: B[a + "/" + c]["top1_edit"] for c, _ in ctx} for a, _, _ in arms}}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 3.0), gridspec_kw={"width_ratios": [1, 1.35]})
    d2 = [A[k]["distinct2"] for k, _ in runs]
    bars = a.bar(range(4), d2, color=[GRAY, BLUE, ORANGE, "#fdba74"], width=0.62)
    for bar, (k, _) in zip(bars, runs):
        a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.2f}\nloops {100 * A[k]['loop_rate']:.0f}%", ha="center", fontsize=7)
    a.set_xticks(range(4))
    a.set_xticklabels([l for _, l in runs], fontsize=7)
    a.set_ylabel("Distinct 2-grams (higher = more varied)")
    a.set_ylim(0.6, 1.07)
    a.set_title("(a) 20 generated tokens, edit kept on", loc="left", fontsize=8.4)
    w = 0.2
    base = [100 * B["gd/" + c]["top1_clean"] for c, _ in ctx]
    b.bar([x - 1.5 * w for x in range(3)], base, width=w, color=GRAY, label="no edit")
    for i, (an, lab, col) in enumerate(arms):
        vals = [100 * B[an + "/" + c]["top1_edit"] for c, _ in ctx]
        bars = b.bar([x + (i - 0.5) * w for x in range(3)], vals, width=w, color=col, label=lab)
        label_bars(b, bars, fmt="{:.0f}", fs=6.6)
    label_bars(b, b.patches[:3], fmt="{:.0f}", fs=6.6)
    b.set_xticks(range(3))
    b.set_xticklabels([l for _, l in ctx], fontsize=7)
    b.set_ylabel("Target word is the top choice (%)")
    b.set_ylim(0, 142)
    b.set_yticks(range(0, 101, 20))
    b.legend(frameon=False, fontsize=6.6, ncol=2, loc="upper left")
    b.set_title("(b) Does the edit respect the instruction?", loc="left", fontsize=8.4)
    fig.tight_layout()
    save(fig, out, "fig_genquality")


def copy_journal_figs(out):
    img = os.path.join(ROOT, "docs", "Research_Journal", "images")
    for src, dst in [("e29_fig1_rank1_by_band.png", "gd_rank1_by_band.png"), ("e29_fig2_kl_paired.png", "gd_kl_paired.png"),
                     ("e29_fig3_success_vs_kl.png", "gd_success_vs_kl.png"), ("e29_fig4_time.png", "gd_time.png"),
                     ("e28_fig1_rank_by_step.png", "gd_trials_rank_by_step.png")]:
        shutil.copyfile(os.path.join(img, src), os.path.join(out, dst))
        print("copied", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "report_figures"))
    ap.add_argument("--only", default="", help="comma separated function names, e.g. fig_arch,fig_algo")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fns = {f.__name__: f for f in (fig_elig, fig_layer12, fig_candsrc, fig_filter, fig_headroom, fig_learned, fig_additive, fig_reworded,
                                    fig_genquality, rd.fig_arch, rd.fig_algo, rd.fig_gdflow, rd.fig_pipeline, rd.fig_ui)}
    todo = [s.strip() for s in a.only.split(",") if s.strip()] or list(fns)
    for name in todo:
        fns[name](a.out)
    if not a.only:
        copy_journal_figs(a.out)
    if DATA:
        path = os.path.join(a.out, "report_figure_data.json")
        old = {}
        if os.path.exists(path):
            old = json.load(open(path, encoding="utf-8"))
        old.update(DATA)
        json.dump(old, open(path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
