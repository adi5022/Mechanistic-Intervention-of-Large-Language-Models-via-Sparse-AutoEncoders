"""
Publication-style PNG / PDF renderings of the study figures (matplotlib), drawn from the same data specs
as the SVG figures in src/batch_analysis.py so the two always agree.
"""
import math

PALETTE = ["#6b7280", "#9ca3af", "#2563eb", "#16a34a", "#ea580c", "#7c3aed", "#0891b2", "#db2777"]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                         "figure.dpi": 100, "savefig.dpi": 300, "svg.fonttype": "none", "pdf.fonttype": 42})
    return plt


def render(spec, base_path, footer=None):
    """Write base_path + '.png' and '.pdf' for one figure spec. Returns the list of files written (or [])."""
    if not spec:
        return []
    plt = _mpl()
    kind = spec["kind"]
    if kind == "bar":
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        labels, vals = spec["labels"], spec["values"]
        bars = ax.bar(range(len(vals)), vals, color=[PALETTE[(i + 2) % len(PALETTE)] for i in range(len(vals))], width=0.65)
        fmt = spec.get("fmt_fn") or (lambda v: f"{v:g}")
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), fmt(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=0 if max(len(str(l)) for l in labels) < 12 else 20, ha="center" if max(len(str(l)) for l in labels) < 12 else "right")
        if spec.get("ymax"):
            ax.set_ylim(0, spec["ymax"] * 1.08)
        if spec.get("ylabel"):
            ax.set_ylabel(spec["ylabel"])
        ax.set_title(spec["title"], fontsize=11)
        ax.grid(axis="y", alpha=0.25)
    elif kind == "grouped":
        fig, ax = plt.subplots(figsize=(7.6, 4.0))
        groups, series, values = spec["groups"], spec["series"], spec["values"]
        w = 0.8 / max(1, len(series))
        for si, sname in enumerate(series):
            xs = [gi + (si - (len(series) - 1) / 2) * w for gi in range(len(groups))]
            vals_ = [values.get(sname, {}).get(g, 0) or 0 for g in groups]
            bars_ = ax.bar(xs, vals_, width=w * 0.95, label=sname, color=PALETTE[(si + 2) % len(PALETTE)])
            for b_, v_ in zip(bars_, vals_):
                ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 1, f"{v_:.0f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        if spec.get("ymax"):
            ax.set_ylim(0, spec["ymax"] * 1.1)
            ax.set_yticks([t for t in range(0, int(spec["ymax"]) + 1, max(1, int(spec["ymax"]) // 5))])
        if spec.get("ylabel"):
            ax.set_ylabel(spec["ylabel"])
        ax.set_title(spec["title"], fontsize=11)
        ax.legend(frameon=False, fontsize=8, ncol=min(5, len(series)), loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.grid(axis="y", alpha=0.25)
        if spec.get("note"):
            ax.set_xlabel(spec["note"], fontsize=8, color="#6b7280")
    elif kind == "scatter":
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        xs, ys = spec["xs"], spec["ys"]
        cols = ["#16a34a" if y < x else ("#dc2626" if y > x else "#2563eb") for x, y in zip(xs, ys)]
        lim = max([2] + list(xs) + list(ys))
        ax.plot([1, lim], [1, lim], ls="--", color="#9ca3af", lw=1)
        ax.scatter(xs, ys, c=cols, alpha=0.75, s=28, edgecolors="none")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlim(0.9, lim * 1.15)
        ax.set_ylim(0.9, lim * 1.15)
        ax.set_xlabel(spec["xlabel"] + " (lower = better)")
        ax.set_ylabel(spec["ylabel"])
        ax.set_title(spec["title"], fontsize=10)
        ax.text(0.02, 0.02, "green: better than reference   red: worse   blue: same", transform=ax.transAxes, fontsize=7, color="#6b7280")
        ax.grid(alpha=0.2)
    else:
        return []
    if footer:
        fig.text(0.01, 0.005, footer, fontsize=5.5, color="#9ca3af")
    fig.tight_layout()
    written = []
    for ext in ("png", "pdf"):
        p = f"{base_path}.{ext}"
        fig.savefig(p, bbox_inches="tight", metadata={"Title": spec["title"], "Subject": footer or ""} if ext == "pdf" else None)
        written.append(p)
    plt.close(fig)
    return written
