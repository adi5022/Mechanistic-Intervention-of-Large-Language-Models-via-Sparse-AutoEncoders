"""
Figures for Research Journal Entry 32 (cross-model transfer, part 2), drawn only from the committed data pack
docs/Research_Journal/packs/cross_model_transfer/ (no model is run):

    .venv\\Scripts\\python.exe tools/make_transfer_figures.py

e32_fig1_country_swap.*      success of the country swap by layer pair, both directions (Export / Import), with the best control
e32_fig2_dose_curves.*       the controlled export: median rank and top-1 rate of the counterfactual target against dose, translated edit and controls
e32_fig3_rewordings_neighbours.*  paraphrase and neighbour effects of the translated edit against controls and references
e32_fig4_neural_vs_linear.*  neural against linear translator on state fit, changed-word and last-position difference
(e32_fig5_fact_or_push.*     Phase C, when its summary is in the pack)

Static light-surface images for the journal. Colour follows the entity in the fixed order of the reference palette (blue translated, orange word push,
aqua random vector, yellow wrong recipe, magenta random-word edit); aqua, yellow and magenta are below 3:1 on the light surface, so every series is also
direct-labelled, drawn with its own marker shape, and the numbers are in the journal tables. Palette validated with the dataviz skill's script.
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
C = {"translated": "#2a78d6", "word_push": "#eb6834", "random": "#1baf7a", "wrong_recipe": "#eda100", "randtarget": "#e87ba4",
     "linear": "#2a78d6", "neural": "#eb6834", "ref": "#b9b8b2", "ref_dark": "#6f6e69"}
MARK = {"translated": "o", "word_push": "s", "random": "^", "wrong_recipe": "D", "randtarget": "X"}
LABEL = {"translated": "translated edit", "word_push": "push along the word's\noutput direction", "random": "random vector (same size)",
         "wrong_recipe": "another record's edit", "randtarget": "edit toward a random word"}

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
    gates = {}
    for fn in ("b1_swap_primary.json", "b1_swap_exploratory.json"):
        gates.update(load(fn)["gates"])
    names = {"s2m": "small", "m2s": "medium"}

    def lab(k):
        d, p, q = k.split("_")[0], k.split("_L")[1], k.split("_L")[2]
        s, r = ("small", "medium") if d == "s2m" else ("medium", "small")
        return f"{s} layer {p} → {r} layer {q}"

    exp = sorted([k for k in gates if k.startswith("s2m")], key=lambda k: int(k.split("_L")[2]))
    imp = sorted([k for k in gates if k.startswith("m2s")], key=lambda k: int(k.split("_L")[1]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
    for ax, keys, title in ((axes[0], exp, "Export: read in small, written into medium"), (axes[1], imp, "Import: read in medium, written into small")):
        vals = [100 * gates[k]["translated_hit"] for k in keys]
        ctrl = [100 * max(gates[k]["controls_best_alpha_hit"].values()) for k in keys]
        y = range(len(keys))
        ax.barh(list(y), vals, color=C["translated"], height=0.55)
        ax.scatter(ctrl, list(y), color=INK2, marker="|", s=160, linewidths=2, zorder=3)
        for i, v in enumerate(vals):
            ax.text(v + 1.2, i, f"{v:.0f}%", va="center", color=INK, fontsize=9.5)
        ax.set_yticks(list(y)); ax.set_yticklabels([lab(k) + ("  (primary)" if k in ("s2m_L8_L16", "m2s_L16_L8") else "") for k in keys], fontsize=9)
        ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.set_xlabel("test cases where the swapped capital becomes the top answer (%)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Country-swap control: the translated difference moves the receiver far above every control", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.legend(handles=[plt.Line2D([], [], color=INK2, marker="|", linestyle="", markersize=12, markeredgewidth=2,
                                   label="best control (random vector, random map or another pair's difference): 0–1% in every pair")],
               loc="lower center", frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    save(fig, "e32_fig1_country_swap")


def fig2(tag="linear", key="s2m_L8_L16"):
    d = load(f"d_analysis_{tag}.json")[key]["small_ok"]
    cu, n = d["curves"], d["n"]
    doses = cu["doses"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ends = {}
    for arm in ("word_push", "translated", "randtarget", "wrong_recipe", "random"):
        for ax, metric in zip(axes, ("median_rank", "top1")):
            ys = cu[arm][metric] if metric == "median_rank" else [100 * v for v in cu[arm][metric]]
            ax.plot(doses, ys, color=C[arm], marker=MARK[arm], markersize=6.5, linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABEL[arm].replace("\n", " "),
                    zorder=3 if arm == "translated" else 2)
    ax = axes[0]
    ax.set_yscale("log"); ax.axhline(cu["baseline_median_rank"], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.set_ylabel("median rank of the target (lower = better)"); ax.set_title("Where the target ranks in medium", loc="left", fontsize=10.5, color=INK)
    axes[1].set_ylabel("target is medium's top answer (%)"); axes[1].set_title("How often it becomes the top answer", loc="left", fontsize=10.5, color=INK)
    for ax in axes:
        ax.set_xlabel("dose (multiplier on the translated change)"); ax.set_xticks(doses); ax.set_xticklabels([f"{x:g}" for x in doses])
    h, l = axes[1].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2))
    l.append(f"unchanged medium (median rank {cu['baseline_median_rank']:.0f}, left panel)")
    axes[1].legend(h, l, loc="center right", frameon=False, fontsize=8.5, bbox_to_anchor=(1.0, 0.44))
    sub = key.replace("s2m_L", "small ").replace("_L", " → medium ")
    fig.suptitle(f"Controlled export, {sub}: the translated edit lifts the target about tenfold in rank; a plain word push reaches top-1 far more often (see Figure 3 for what it costs)",
                 x=0.01, ha="left", fontsize=10.5, color=INK, wrap=True)
    fig.text(0.01, -0.045, f"{n} test records whose edit reached rank 1 in small; linear translator. Dose is the same number for every arm; controls have the same size at each position.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.0, 1, 0.93))
    save(fig, "e32_fig2_dose_curves" if tag == "linear" else f"e32_fig2_dose_curves_{tag}")


def fig3(tag="linear", key="s2m_L8_L16"):
    d = load(f"d_analysis_{tag}.json")[key]["small_ok"]
    s2, ic, n = d["stage2"], d["in_context"], d["n"]
    rows = [("medium unchanged", s2["medium unchanged"], C["ref"]), ("translated edit", s2["translated"], C["translated"]), ("random vector", s2["random"], C["random"]),
            ("another record's edit", s2["wrong_recipe"], C["wrong_recipe"]), ("push along the word's output direction", s2["word_push"], C["word_push"]),
            ("fact stated in medium's prompt", {"PS": ic["PS"], "NS": ic["NS"]}, C["ref_dark"]), ("small, with the edit itself", s2["small with the edit"], C["ref_dark"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, col, title in ((axes[0], "PS", "Reworded prompts: the new answer beats the true one\n(higher = more of the new fact)"),
                           (axes[1], "NS", "Neighbouring facts: the true answer still beats the new one\n(higher = fewer leaks)")):
        for i, (name, v, color) in enumerate(rows):
            ax.barh(i, 100 * v[col], color=color, height=0.6)
            ax.text(100 * v[col] + 1.2, i, f"{100 * v[col]:.0f}%", va="center", fontsize=9.5, color=INK)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.set_xlim(0, 105); ax.set_title(title, loc="left", fontsize=9.5, color=INK); ax.grid(axis="y", visible=False)
        ax.set_xlabel("% of prompts")
    axes[0].invert_yaxis()            # the y axis is shared between the panels: invert it once
    fig.suptitle(f"The translated edit lifts rewordings by 16 points and costs neighbours 4; a word push or the fact in the prompt damages neighbours far more", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.text(0.01, -0.03, f"Small → medium layer 16, {n} test records whose edit reached rank 1 in small, dose chosen on dev; grey = references, not tested arms.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "e32_fig3_rewordings_neighbours" if tag == "linear" else f"e32_fig3_rewordings_neighbours_{tag}")


def fig4():
    d = load("mlp_vs_linear.json")
    keys = list(d)
    names = {"s2m_L8_L12": "small 8\n→ medium 12", "s2m_L8_L16": "small 8\n→ medium 16", "m2s_L16_L8": "medium 16\n→ small 8"}
    metrics = [("State fit (held-out per-dimension R²)", lambda r, m: r["r2_linear" if m == "linear" else "r2_neural"]),
               ("Changed word: difference cosine", lambda r, m: r[m]["differences"]["entity"]["cos"]),
               ("Last position: difference cosine", lambda r, m: r[m]["differences"]["last"]["cos"])]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, (title, fn) in zip(axes, metrics):
        for j, m in enumerate(("linear", "neural")):
            vals = [fn(d[k], m) for k in keys]
            xs = [i + (j - 0.5) * 0.36 for i in range(len(keys))]
            ax.bar(xs, vals, width=0.34, color=C[m], label=m + " translator")
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
        if "Last" in title:
            for i, k in enumerate(keys):
                ax.plot([i - 0.38, i + 0.38], [d[k]["linear"]["differences"]["last"]["chance_p95"]] * 2, color=INK2, linestyle=(0, (3, 2)), linewidth=1.3)
            ax.plot([], [], color=INK2, linestyle=(0, (3, 2)), linewidth=1.3, label="chance level (95th percentile)")
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([names[k] for k in keys], fontsize=8.5)
        ax.set_ylim(0, 1.0); ax.set_title(title, loc="left", fontsize=9.5, color=INK); ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    axes[2].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("A neural translator fits states and the changed word clearly better, but barely improves the last-position difference", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "e32_fig4_neural_vs_linear")


def fig5():
    fn = os.path.join(PACK, "c_fact_or_push_summary.json")
    if not os.path.exists(fn):
        print("no Phase C summary in the pack, skipping figure 5")
        return
    d = load("c_fact_or_push_summary.json")
    rows = d["rows"]
    order = [("all", "all positions", C["translated"]), ("subject", "subject tokens only", C["ref_dark"]), ("last", "last position only", C["ref"]), ("random", "random word, all positions", C["randtarget"])]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for ax, (col, title) in zip(axes, (("success", "Edit works on the tuned prompt (% of records)"), ("ps", "Rewordings: new answer beats true (% of prompts)"),
                                      ("para_gain", "Effect on rewordings (mean log-rank gain)"))):
        for i, (k, lab, color) in enumerate(order):
            v = rows[k][col] * (100 if col in ("success", "ps") else 1)
            ax.barh(i, v if v == v else 0, color=color, height=0.6)
            ax.text((v if v == v else 0) + 0.02 * (100 if col != "para_gain" else 3), i, "n/a" if v != v else (f"{v:.0f}%" if col != "para_gain" else f"{v:.2f}"), va="center", fontsize=9.5)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([o[1] for o in order] if col == "success" else [], fontsize=9); ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=9.5, color=INK); ax.grid(axis="y", visible=False)
    fig.suptitle(f"Fact edit or word push? (GPT-2 small alone). The pre-stated rule, applied as written, gives: {d['verdict']}  (caveats in the journal entry: few records succeed when the edit is restricted)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "e32_fig5_fact_or_push")


def fig6():
    """Controlled export with the linear and with the neural translator, side by side (test records whose edit reached rank 1 in small)."""
    lin, nn_ = load("d_analysis_linear.json"), load("d_analysis_mlp.json")
    keys = ["s2m_L8_L12", "s2m_L8_L16"]
    names = {"s2m_L8_L12": "small 8\n→ medium 12", "s2m_L8_L16": "small 8\n→ medium 16"}

    def top1(d, k):
        return 100 * d[k]["small_ok"]["tuned"]["translated"]["top1"]

    def para_lift(d, k):
        s2 = d[k]["small_ok"]["stage2"]
        return 100 * (s2["translated"]["PS"] - s2["medium unchanged"]["PS"])

    def neigh_cost(d, k):
        s2 = d[k]["small_ok"]["stage2"]
        return 100 * (s2["medium unchanged"]["NS"] - s2["translated"]["NS"])

    def rank(d, k):
        return d[k]["small_ok"]["tuned"]["translated"]["median_rank_after"]

    metrics = [("Target is medium's\ntop answer (%)", top1, "{:.0f}%"), ("Median rank of the target\n(unchanged: 142; lower = better)", rank, "{:.0f}"),
               ("Rewordings: gain in\n'new beats true' (points)", para_lift, "{:+.0f}"), ("Neighbours: loss in\n'true still beats new' (points)", neigh_cost, "{:.0f}")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    for ax, (title, fn, fmt) in zip(axes, metrics):
        for j, (m, d) in enumerate((("linear", lin), ("neural", nn_))):
            vals = [fn(d, k) for k in keys]
            xs = [i + (j - 0.5) * 0.36 for i in range(len(keys))]
            ax.bar(xs, vals, width=0.34, color=C[m], label=m + " translator")
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.01 * max(1, max(vals)), fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([names[k] for k in keys], fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=9, color=INK, wrap=True); ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Controlled export: a neural translator, which fits states and the changed word clearly better, transfers the edit no better than the linear one",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "e32_fig6_export_linear_vs_neural")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6}[w]()
