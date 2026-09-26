"""
Analysis of a multi-arm batch result (safety-filter study): tables, statistics, difficulty breakdown and SVG figures.

    from src.batch_analysis import analyze
    out = analyze(result_dict)            # tables + markdown (+ SVG strings in out["figures"])

Used by the Batch tab (visuals + downloads) and by tools/analyze_safety_batch.py (command line).
"""
import math

from src.batch_runner import finalize_result

BG, INK, MUTED, GRID = "#ffffff", "#1f2937", "#6b7280", "#e5e7eb"
PALETTE = ["#6b7280", "#9ca3af", "#2563eb", "#16a34a", "#ea580c", "#7c3aed", "#0891b2", "#db2777"]
FONT = "font-family='Segoe UI, Helvetica, Arial, sans-serif'"
BINS = [(2, 3), (4, 10), (11, 50), (51, 300), (301, 1000)]


class Svg(str):
    """An SVG document (a plain str) that also remembers the data spec it was drawn from, for the matplotlib export."""
    spec = None


def _with_spec(svg_text, spec):
    o = Svg(svg_text)
    o.spec = spec
    return o


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return f"<text x='{x}' y='{y}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' font-weight='{weight}'>{_esc(s)}</text>"


def _svg(w, h, body, title):
    return f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' {FONT}><title>{_esc(title)}</title><rect width='{w}' height='{h}' fill='{BG}'/>{body}</svg>"


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def bar_svg(title, labels, values, fmt_fn=lambda v: f"{v:.1f}", ymax=None, note=None, ylabel=None):
    W, H = 760, 380
    L, R, T, B = 70, 30, 60, 90
    ymax = ymax or max(1e-9, max(values) * 1.15 if values else 1)
    pw, ph = W - L - R, H - T - B
    b = [_text(W / 2, 28, title, 15, weight="bold", anchor="middle")]
    for k in range(5):
        v = ymax * k / 4
        y = T + ph * (1 - v / ymax)
        b.append(f"<line x1='{L}' y1='{y:.1f}' x2='{W - R}' y2='{y:.1f}' stroke='{GRID}'/>" + _text(L - 8, y + 4, fmt_fn(v), 11, MUTED, "end"))
    bw = pw / max(1, len(values))
    for i, (lab, v) in enumerate(zip(labels, values)):
        h = ph * (v / ymax)
        x = L + i * bw + bw * 0.15
        b.append(f"<rect x='{x:.1f}' y='{T + ph - h:.1f}' width='{bw * 0.7:.1f}' height='{h:.1f}' fill='{PALETTE[(i + 2) % len(PALETTE)]}'/>")
        b.append(_text(x + bw * 0.35, T + ph - h - 6, fmt_fn(v), 12, INK, "middle", "bold"))
        b.append(_text(x + bw * 0.35, T + ph + 18, lab, 11, MUTED, "middle"))
    if ylabel:
        b.append(_text(14, T + ph / 2, ylabel, 11, MUTED, "middle"))
    if note:
        b.append(_text(L, H - 12, note, 11, MUTED))
    return _with_spec(_svg(W, H, "".join(b), title),
                      {"kind": "bar", "title": title, "labels": list(labels), "values": list(values), "ymax": ymax,
                       "ylabel": ylabel, "note": note, "fmt_fn": fmt_fn})


def grouped_bar_svg(title, groups, series, values, fmt_fn=lambda v: f"{v:.0f}%", ymax=100, ylabel=None, note=None):
    """values[series][group] -> number; one cluster of bars per group."""
    W, H = 860, 400
    L, R, T, B = 70, 30, 60, 110
    pw, ph = W - L - R, H - T - B
    b = [_text(W / 2, 28, title, 15, weight="bold", anchor="middle")]
    for k in range(5):
        v = ymax * k / 4
        y = T + ph * (1 - v / ymax)
        b.append(f"<line x1='{L}' y1='{y:.1f}' x2='{W - R}' y2='{y:.1f}' stroke='{GRID}'/>" + _text(L - 8, y + 4, fmt_fn(v), 11, MUTED, "end"))
    gw = pw / max(1, len(groups))
    bw = gw * 0.8 / max(1, len(series))
    for gi, g in enumerate(groups):
        for si, sname in enumerate(series):
            v = values.get(sname, {}).get(g, 0) or 0
            h = ph * (v / ymax)
            x = L + gi * gw + gw * 0.1 + si * bw
            b.append(f"<rect x='{x:.1f}' y='{T + ph - h:.1f}' width='{bw * 0.92:.1f}' height='{h:.1f}' fill='{PALETTE[(si + 2) % len(PALETTE)]}'/>")
        b.append(_text(L + gi * gw + gw / 2, T + ph + 18, g, 11, MUTED, "middle"))
    lx = L
    for si, sname in enumerate(series):
        b.append(f"<rect x='{lx}' y='{H - 52}' width='12' height='12' fill='{PALETTE[(si + 2) % len(PALETTE)]}'/>" + _text(lx + 18, H - 42, sname, 11))
        lx += 140
    if ylabel:
        b.append(_text(14, T + ph / 2, ylabel, 11, MUTED, "middle"))
    if note:
        b.append(_text(L, H - 14, note, 11, MUTED))
    return _with_spec(_svg(W, H, "".join(b), title),
                      {"kind": "grouped", "title": title, "groups": list(groups), "series": list(series), "values": values,
                       "ymax": ymax, "ylabel": ylabel, "note": note})


def scatter_svg(title, xs, ys, xlabel, ylabel):
    W, H = 560, 520
    L, R, T, B = 70, 30, 60, 70
    lg = lambda v: math.log10(max(1, v))
    mx = (max([lg(v) for v in xs + ys]) if xs else 1) * 1.05 + 0.05
    pw, ph = W - L - R, H - T - B
    px = lambda v: L + pw * lg(v) / mx
    py = lambda v: T + ph * (1 - lg(v) / mx)
    b = [_text(W / 2, 28, title, 14, weight="bold", anchor="middle")]
    for t in (1, 3, 10, 30, 100, 300, 1000):
        if lg(t) <= mx:
            b.append(f"<line x1='{px(t):.1f}' y1='{T}' x2='{px(t):.1f}' y2='{T + ph}' stroke='{GRID}'/><line x1='{L}' y1='{py(t):.1f}' x2='{L + pw}' y2='{py(t):.1f}' stroke='{GRID}'/>")
            b.append(_text(px(t), T + ph + 16, f"#{t}", 10, MUTED, "middle") + _text(L - 8, py(t) + 4, f"#{t}", 10, MUTED, "end"))
    b.append(f"<line x1='{px(1):.1f}' y1='{py(1):.1f}' x2='{L + pw}' y2='{T}' stroke='#9ca3af' stroke-dasharray='5 4'/>")
    b.append(_text(L + pw - 4, T + 14, "same result", 10, MUTED, "end"))
    for x, y in zip(xs, ys):
        col = "#16a34a" if y < x else ("#dc2626" if y > x else "#2563eb")
        b.append(f"<circle cx='{px(x):.1f}' cy='{py(y):.1f}' r='4' fill='{col}' fill-opacity='0.75'/>")
    b.append(_text(L + pw / 2, H - 28, xlabel + "  (lower = better)", 12, MUTED, "middle"))
    b.append(_text(16, T + ph / 2, ylabel, 12, MUTED, "middle"))
    b.append(_text(L, H - 10, "green = better than the reference, red = worse, blue = same", 10, MUTED))
    return _with_spec(_svg(W, H, "".join(b), title),
                      {"kind": "scatter", "title": title, "xs": list(xs), "ys": list(ys), "xlabel": xlabel, "ylabel": ylabel})


def analyze(res, tag="e21"):
    """
    Returns a dict: {"multi_arm": bool, "arms", "reference", "modes",
      "arm_rows": [...], "vs_rows": [...], "bin_rows": [...], "reasons": {...}, "discard": {...},
      "cells": [...], "figures": {filename: svg_text}, "markdown": str}
    """
    finalize_result(res)
    ac = res.get("arm_comparison")
    out = {"multi_arm": bool(ac), "meta": res["meta"], "figures": {}, "figure_meta": {}, "markdown": ""}
    if not ac:
        return out
    arms, ref, pam = ac["arms"], ac["reference"], ac["per_arm_mode"]
    modes = sorted({v["mode"] for v in pam.values()})
    md = []
    out.update({"arms": arms, "reference": ref, "modes": modes, "cells": ac["cells"]})

    arm_rows, vs_rows, bin_rows = [], [], []
    for mode in modes:
        rows = []
        for arm in arms:
            v = pam[f"{arm} | {mode}"]
            arm_rows.append({"Mode": mode, "Arm": arm, "Prompts": v["n_prompts"], "Reached rank #1": v["reached_rank1"],
                             "Success rate %": round((v["success_rate"] or 0) * 100, 1), "Mean rank gain": v["mean_rank_gain"],
                             "Median final rank": v["median_final_rank"], "Mean final prob %": v["mean_final_prob_pct"],
                             "Features at best": v["mean_features_at_best"], "Rescued in best": v["mean_rescued_in_best"],
                             "Steps": v["mean_steps"], "Time s": v["mean_time_s"], "Collateral KL (nats)": v["mean_kl_nats"],
                             "Top-1 flips": v["mean_top1_flip_rate"]})
            rows.append([arm, f"{v['reached_rank1']}/{v['n_prompts']} ({(v['success_rate'] or 0) * 100:.0f}%)", fmt(v["mean_rank_gain"], 1), v["median_final_rank"],
                         fmt(v["mean_final_prob_pct"], 2) + "%", fmt(v["mean_features_at_best"], 1), fmt(v["mean_rescued_in_best"], 1),
                         fmt(v["mean_steps"], 1), fmt(v["mean_time_s"], 1), fmt(v["mean_kl_nats"], 5), fmt(v["mean_top1_flip_rate"], 3)])
        md.append(f"### Arms, {mode} mode\n\n" + md_table(
            ["Arm", "Reached rank #1", "Mean rank gain", "Median final rank", "Mean final prob", "Features at best", "Rescued in best", "Steps", "Time s", "Collateral KL (nats)", "Top-1 flips"], rows) + "\n")

        vr = []
        for arm in arms:
            if arm == ref:
                continue
            v = ac["vs_reference"][f"{arm} vs {ref} | {mode}"]
            vs_rows.append({"Mode": mode, "Arm": arm, "Reference": ref, "Prompts": v["n"], "Better": v["arm_better_rank_or_success"],
                            "Worse": v["arm_worse_rank_or_success"], "Same": v["same"], "Newly rank #1": v["prompts_newly_reaching_rank1"],
                            "Lost rank #1": v["prompts_losing_rank1"], "Sign-test p": v["sign_test_p_rank_outcome"], "McNemar p": v["mcnemar_p_rank1"]})
            vr.append([arm, v["n"], v["arm_better_rank_or_success"], v["arm_worse_rank_or_success"], v["same"], v["prompts_newly_reaching_rank1"],
                       v["prompts_losing_rank1"], fmt(v["sign_test_p_rank_outcome"], 4), fmt(v["mcnemar_p_rank1"], 4)])
        md.append(f"### Each arm vs {ref}, {mode} mode\n\n" + md_table(
            ["Arm vs " + ref, "Prompts", "Better", "Worse", "Same", "Newly rank #1", "Lost rank #1", "Sign-test p", "McNemar p"], vr) + "\n")

        # difficulty bins
        sub = [c for c in ac["cells"] if c["mode"] == mode]
        for lo, hi in BINS:
            inb = [c for c in sub if lo <= c["baseline_rank"] <= hi]
            if not inb:
                continue
            for arm in arms:
                bin_rows.append({"Mode": mode, "Baseline rank": f"{lo}-{hi}", "Arm": arm, "Prompts": len(inb),
                                 "Success rate %": round(100 * sum(1 for c in inb if c[arm]["success"]) / len(inb), 1),
                                 "Mean final rank": round(sum(c[arm]["final_rank"] for c in inb) / len(inb), 2)})

        # figures
        prefix = f"{tag}_{mode}"
        n_ = pam[f"{ref} | {mode}"]["n_prompts"]
        modelabel = {"all": "all-prompt-positions candidate source", "last": "last-token-only candidate source"}.get(mode, mode)
        out["figure_meta"][f"{prefix}_success.svg"] = {
            "title": f"Prompts reaching rank #1 by safety-filter variant ({mode})", "table": f"arms_{mode}.csv",
            "caption": f"Share of prompts whose target token reached rank #1, per safety-filter variant, {modelabel} (n = {n_} prompts whose target did not start at rank #1)."}
        out["figure_meta"][f"{prefix}_features.svg"] = {
            "title": f"Features edited at the best result ({mode})", "table": f"arms_{mode}.csv",
            "caption": f"Mean number of SAE features muted or boosted at the best result found, per variant, {modelabel} (n = {n_})."}
        out["figure_meta"][f"{prefix}_collateral.svg"] = {
            "title": f"Collateral damage on unrelated prompts ({mode})", "table": f"arms_{mode}.csv",
            "caption": f"Mean KL divergence (nats) between the clean and edited next-token distributions on 20 unrelated prompts, per variant, {modelabel}. Lower means less collateral change."}
        out["figure_meta"][f"{prefix}_time.svg"] = {
            "title": f"Mean run time ({mode})", "table": f"arms_{mode}.csv",
            "caption": f"Mean model-compute time per run in seconds, per variant, {modelabel} (n = {n_})."}
        for arm_ in arms:
            if arm_ != ref:
                out["figure_meta"][f"{prefix}_scatter_{arm_}.svg"] = {
                    "title": f"Final rank: {arm_} vs {ref} ({mode})", "table": "per_prompt.csv",
                    "caption": f"Final target rank per prompt under '{arm_}' against the reference '{ref}', {modelabel}, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = {n_})."}
        out["figures"][f"{prefix}_success.svg"] = bar_svg(
            f"Prompts reaching rank #1 by safety filter ({mode})", arms, [(pam[f"{a} | {mode}"]["success_rate"] or 0) * 100 for a in arms],
            lambda v: f"{v:.0f}%", ymax=100, ylabel="% of prompts", note=f"n = {pam[f'{ref} | {mode}']['n_prompts']} prompts with baseline rank > 1")
        out["figures"][f"{prefix}_features.svg"] = bar_svg(
            f"Features edited at the best result ({mode})", arms, [pam[f"{a} | {mode}"]["mean_features_at_best"] or 0 for a in arms], lambda v: f"{v:.0f}", ylabel="features (mean)")
        klv = [(pam[f"{a} | {mode}"]["mean_kl_nats"] or 0) for a in arms]
        if any(klv):
            out["figures"][f"{prefix}_collateral.svg"] = bar_svg(
                f"Collateral damage on 20 unrelated prompts ({mode})", arms, klv, lambda v: f"{v:.4f}", ylabel="mean KL (nats), lower = safer")
        out["figures"][f"{prefix}_time.svg"] = bar_svg(
            f"Mean run time ({mode})", arms, [pam[f"{a} | {mode}"]["mean_time_s"] or 0 for a in arms], lambda v: f"{v:.1f}s", ylabel="seconds")
        for arm in arms:
            if arm == ref:
                continue
            out["figures"][f"{prefix}_scatter_{arm}.svg"] = scatter_svg(
                f"Final target rank: {arm} vs {ref} ({mode})", [c[ref]["final_rank"] for c in sub], [c[arm]["final_rank"] for c in sub],
                f"{ref} final rank", f"{arm} final rank")

    for mode in modes:
        bmode = [r_ for r_ in bin_rows if r_["Mode"] == mode]
        if bmode:
            groups = [f"{lo}-{hi}" for lo, hi in BINS if any(r_["Baseline rank"] == f"{lo}-{hi}" for r_ in bmode)]
            vals = {a_: {r_["Baseline rank"]: r_["Success rate %"] for r_ in bmode if r_["Arm"] == a_} for a_ in arms}
            counts = {r_["Baseline rank"]: r_["Prompts"] for r_ in bmode}
            out["figures"][f"{tag}_{mode}_difficulty.svg"] = grouped_bar_svg(
                f"Success rate by starting difficulty ({mode})", [f"{g} (n={counts[g]})" for g in groups], arms,
                {a_: {f"{g} (n={counts[g]})": vals[a_].get(g, 0) for g in groups} for a_ in arms}, ylabel="% reaching rank #1",
                note="x-axis: the target's rank before any editing")
            out["figure_meta"][f"{tag}_{mode}_difficulty.svg"] = {
                "title": f"Success rate by starting difficulty ({mode})", "table": "difficulty_bins.csv",
                "caption": f"Share of prompts reaching rank #1 per safety-filter variant, grouped by how far down the target started (baseline rank bins), {mode} mode."}

    # Step 0 diagnostic on the reference arm (round 0)
    reasons = {"ok": 0, "harm": 0, "rank": 0, "both": 0}
    disc = tot = 0
    for r in res["runs"]:
        if r.get("arm") != ref or r.get("error"):
            continue
        for rd in r.get("round_details", []):
            if rd["round"] != 0:
                continue
            sa = rd.get("safety_assessment") or {}
            for side in ("mute", "boost"):
                for c in (sa.get("candidates") or {}).get(side, []):
                    reasons[c["reason_strict"]] += 1
            if rd.get("candidate_features_total") is not None:
                tot += rd["candidate_features_total"]
                disc += rd["unusable_on_both_sides"]
            else:                                                  # older records: derive from the candidate lists
                mc = {c["feature"]: c["verdict"] for c in rd["mute_candidates"]}
                bc = {c["feature"]: c["verdict"] for c in rd["boost_candidates"]}
                feats = set(mc) | set(bc)
                tot += len(feats)
                disc += sum(1 for f in feats if mc.get(f, "rejected") == "rejected" and bc.get(f, "rejected") == "rejected")
    n_all = max(1, sum(reasons.values()))
    out["reasons"] = reasons
    out["discard"] = {"features_total": tot, "unusable_on_both_sides": disc, "share": disc / max(1, tot)}
    out["figures"][f"{tag}_strict_filter_reasons.svg"] = bar_svg(
        "Why the strict filter rejects a candidate feature (round 0)", ["accepted", "harm rule only", "rank rule only", "both rules"],
        [reasons["ok"], reasons["harm"], reasons["rank"], reasons["both"]], lambda v: f"{v:.0f}", ylabel="candidate features",
        note=f"Features unusable on BOTH sides: {disc} of {tot} ({disc / max(1, tot) * 100:.1f}%)")
    out["figure_meta"][f"{tag}_strict_filter_reasons.svg"] = {
        "title": "Why the strict filter rejects a candidate feature", "table": "strict_filter_reasons.csv",
        "caption": f"Verdicts of the strict safety filter on every round-0 candidate feature (mute and boost sides pooled) in the reference arm. Only {disc} of {tot} features ({disc / max(1, tot) * 100:.1f}%) were unusable on both sides."}
    md.append("### Why the strict filter rejects (round 0, reference arm)\n\n" + md_table(
        ["Verdict on a candidate", "Count", "Share"],
        [["accepted", reasons["ok"], f"{reasons['ok'] / n_all * 100:.1f}%"], ["rejected: harm rule only", reasons["harm"], f"{reasons['harm'] / n_all * 100:.1f}%"],
         ["rejected: rank rule only", reasons["rank"], f"{reasons['rank'] / n_all * 100:.1f}%"], ["rejected: both rules", reasons["both"], f"{reasons['both'] / n_all * 100:.1f}%"]])
        + f"\n\nFeatures unusable on **both** sides (truly discarded): **{disc} of {tot} ({disc / max(1, tot) * 100:.1f}%)**.\n")
    out["arm_rows"], out["vs_rows"], out["bin_rows"] = arm_rows, vs_rows, bin_rows
    out["markdown"] = "\n".join(md)
    return out
