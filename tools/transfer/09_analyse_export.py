"""
Paired analysis of the controlled export (step D3/D4), from the per-record numbers saved by 06_export_eval.py.

    .venv\\Scripts\\python.exe tools/transfer/09_analyse_export.py                  # linear translator
    .venv\\Scripts\\python.exe tools/transfer/09_analyse_export.py --translator mlp

Everything is recomputed per record with ONE rule: the gain in rank of a word is measured against that word's OWN starting rank in medium (the
first run of 06_export_eval.py measured the random-word control against the counterfactual target's starting rank, which is wrong for that arm;
the top-1 numbers and the Gate 3 verdict do not depend on it). Paired Wilcoxon tests, one-sided, over test records.
Writes outputs/transfer/d_analysis_<translator>.json.
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
from scipy.stats import wilcoxon

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def med(xs):
    xs = sorted(x for x in xs if x == x)
    return xs[len(xs) // 2] if xs else float("nan")


def pct(x):
    return "  n/a" if x != x else f"{100 * x:4.0f}%"


def pval(x, y, alt="greater"):
    try:
        return float(wilcoxon(x, y, alternative=alt).pvalue)
    except ValueError:
        return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translator", default="linear")
    a = ap.parse_args()
    blob = torch.load(os.path.join(OUT_DIR, f"d_eval_{a.translator}.pt"), weights_only=False)
    bench = blob["bench"]
    edits = torch.load(os.path.join(OUT_DIR, "d_edits.pt"), weights_only=False)
    summary = {}
    print("=" * 120)
    print(f"PAIRED ANALYSIS of the controlled export (translator: {a.translator}); rank gain = log(start rank) - log(rank after), each word against its own start rank")
    for key, d in blob["allres"].items():
        res1, res2, i_star = d["res1"], d["res2"], d["i_star"]
        summary[key] = {}
        for subset, label in (("small_ok", "test records whose edit reached rank 1 in small"), ("all", "all test records")):
            ids = [r["case_id"] for r in bench if r["split"] == "test" and (subset == "all" or edits[r["case_id"]]["real"]["rank"] == 1)]
            n = len(ids)
            print(f"\n  ---- {key}, dose {d['alpha_star']} (chosen on dev), {label}: {n} records ----")

            def gain(arm, idx, c):
                base = res1[c]["randtarget_base" if arm == "randtarget" else "base"]["rank_new"][0].item()
                return math.log(base) - math.log(res1[c][arm]["rank_new"][idx].item())

            rows = {}
            for arm in ["translated", "random", "wrong_recipe", "randtarget", "word_push"]:
                pick = [c for c in ids if arm in res1[c]]
                g = [gain(arm, i_star, c) for c in pick]
                top1 = mean([res1[c][arm]["top1"][i_star].item() for c in pick])
                rows[arm] = {"n": len(pick), "top1": top1, "median_gain": med(g), "mean_gain": mean(g), "median_rank_after": med([res1[c][arm]["rank_new"][i_star].item() for c in pick])}
            print(f"  tuned prompt, same dose for every arm.   {'arm':<14}{'top-1':>7}{'median rank after':>19}{'mean log-rank gain':>20}{'paired p (translated greater)':>32}")
            for arm, r in rows.items():
                pick = [c for c in ids if arm in res1[c]]
                p = pval([gain("translated", i_star, c) for c in pick], [gain(arm, i_star, c) for c in pick]) if arm != "translated" else float("nan")
                r["p_vs_translated"] = p
                print(f"  {'':<41}{arm:<14}{pct(r['top1']):>7}{r['median_rank_after']:>19.0f}{r['mean_gain']:>20.2f}{p:>32.1e}" if p == p else
                      f"  {'':<41}{arm:<14}{pct(r['top1']):>7}{r['median_rank_after']:>19.0f}{r['mean_gain']:>20.2f}")
            summary[key][subset] = {"n": n, "tuned": rows}
            # dose curves (for the figures): top-1 rate and median rank of the target at every dose, each arm
            n_doses = len(res1[ids[0]]["translated"]["top1"])
            curves = {"doses": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][:n_doses], "baseline_median_rank": med([res1[c]["base"]["rank_new"][0].item() for c in ids])}
            for arm in ["translated", "random", "wrong_recipe", "randtarget", "word_push"]:
                pk = [c for c in ids if arm in res1[c]]
                curves[arm] = {"top1": [mean([res1[c][arm]["top1"][i].item() for c in pk]) for i in range(n_doses)],
                               "median_rank": [med([res1[c][arm]["rank_new"][i].item() for c in pk]) for i in range(n_doses)]}
            summary[key][subset]["curves"] = curves

            # sweeps: best dose of each control against the translated arm at the chosen dose
            best = {}
            for arm in ["random", "wrong_recipe", "randtarget"]:
                pick = [c for c in ids if arm in res1[c]]
                j = max(range(len(pick and res1[pick[0]][arm]["top1"])), key=lambda i: (round(mean([res1[c][arm]["top1"][i].item() for c in pick]), 6), mean([gain(arm, i, c) for c in pick])))
                best[arm] = {"idx": j, "top1": mean([res1[c][arm]["top1"][j].item() for c in pick]), "mean_gain": mean([gain(arm, j, c) for c in pick])}
            print("  controls at their own best dose on test (favours them):  " + ", ".join(f"{k} top-1 {pct(v['top1'])} (mean gain {v['mean_gain']:.2f})" for k, v in best.items()))
            summary[key][subset]["controls_best"] = best

            # stage 2: rewordings, neighbours, unrelated
            def per(c, kind, arm, col, which):
                return mean([float(r[arm][col][which if which is not None else 0]) for r in res2[c][kind]])

            def agg(arm, which):
                ps = [per(c, "paraphrase", arm, "es", which) for c in ids]
                ns = [1 - per(c, "neighbour", arm, "es", which) for c in ids]
                kl = [per(c, "unrelated", arm, "kl", which) for c in ids] if "kl" in res2[ids[0]]["unrelated"][0][arm] else [float("nan")]
                fl = [per(c, "unrelated", arm, "flip", which) for c in ids] if "flip" in res2[ids[0]]["unrelated"][0][arm] else [float("nan")]
                return ps, ns, kl, fl

            tbl = {}
            for name, arm, which in [("medium unchanged", "base", None), ("small with the edit", "small_native", None), ("small unchanged", "small_base", None),
                                     ("translated", "translated", 0), ("translated, norm matched", "translated", 1), ("random", "random", 0),
                                     ("wrong_recipe", "wrong_recipe", 0), ("word_push", "word_push", 0)]:
                ps, ns, kl, fl = agg(arm, which)
                tbl[name] = {"PS": mean(ps), "NS": mean(ns), "KL": mean(kl), "flips": mean(fl), "_ps": ps, "_ns": ns}
            print(f"\n  rewordings and neighbours (per-record means; PS = P(new) > P(true) on paraphrases, NS = P(true) > P(new) on neighbours), {n} records")
            print(f"  {'':<30}{'PS':>7}{'NS':>7}{'KL':>8}{'flips':>8}")
            for name, v in tbl.items():
                print(f"  {name:<30}{pct(v['PS']):>7}{pct(v['NS']):>7}{v['KL']:>8.3f}{pct(v['flips']):>8}")
            ic = [per(c, "paraphrase", "in_context", "es", None) for c in ids], [1 - per(c, "neighbour", "in_context", "es", None) for c in ids]
            print(f"  {'medium, fact in the prompt':<30}{pct(mean(ic[0])):>7}{pct(mean(ic[1])):>7}")
            t, b, rn, wr, wp = (tbl[k] for k in ("translated", "medium unchanged", "random", "wrong_recipe", "word_push"))
            p_ps_rand = pval(t["_ps"], rn["_ps"]); p_ps_wrong = pval(t["_ps"], wr["_ps"]); p_ps_base = pval(t["_ps"], b["_ps"])
            p_ns_push = pval(t["_ns"], wp["_ns"]); p_ns_base = pval(b["_ns"], t["_ns"])
            eff = (t["PS"] - b["PS"]) / (tbl["small with the edit"]["PS"] - tbl["small unchanged"]["PS"]) if tbl["small with the edit"]["PS"] != tbl["small unchanged"]["PS"] else float("nan")
            print(f"  paraphrase lift of translated over unchanged medium: {100 * (t['PS'] - b['PS']):+.0f} points (p = {p_ps_base:.1e}); over random: p = {p_ps_rand:.1e}; over wrong_recipe: p = {p_ps_wrong:.1e}; "
                  f"transfer efficiency (lift in medium / lift of the edit in small itself) = {eff:.2f}")
            print(f"  neighbour loss of translated: {100 * (b['NS'] - t['NS']):+.0f} points (unchanged minus translated; p = {p_ns_base:.1e}); translated keeps neighbours better than word_push: p = {p_ns_push:.1e} (word_push loses {100 * (b['NS'] - wp['NS']):.0f} points)")
            summary[key][subset]["stage2"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in tbl.items()}
            summary[key][subset]["stage2_tests"] = {"ps_vs_unchanged_p": p_ps_base, "ps_vs_random_p": p_ps_rand, "ps_vs_wrong_p": p_ps_wrong, "ns_vs_push_p": p_ns_push, "transfer_efficiency": eff}
            summary[key][subset]["in_context"] = {"PS": mean(ic[0]), "NS": mean(ic[1])}

            # plausible counterfactual target against rank-matched random word
            both = [c for c in ids if "randtarget" in res1[c]]
            gt, gr = [gain("translated", i_star, c) for c in both], [gain("randtarget", i_star, c) for c in both]
            print(f"  plausible counterfactual target vs rank-matched random word ({len(both)} records, each against its own start rank): mean gain {mean(gt):.2f} vs {mean(gr):.2f}, paired p = {pval(gt, gr):.1e}; "
                  f"top-1 {pct(mean([res1[c]['translated']['top1'][i_star].item() for c in both]))} vs {pct(mean([res1[c]['randtarget']['top1'][i_star].item() for c in both]))}")
            summary[key][subset]["plausible_vs_random"] = {"n": len(both), "gain_target": mean(gt), "gain_random": mean(gr), "p": pval(gt, gr)}
    json.dump(summary, open(os.path.join(OUT_DIR, f"d_analysis_{a.translator}.json"), "w", encoding="utf8"), indent=1, default=float)
    print("=" * 120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
