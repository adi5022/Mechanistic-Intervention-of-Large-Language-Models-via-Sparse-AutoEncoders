"""
Step D5 of docs/cross_model_transfer/PLAN.md: does adding the translated edit at SEVERAL layers of GPT-2 medium at once beat adding it at one?
Gate 4 is written in the plan before this run. Needs 05_export_edits.py (edits) and 01_fit_maps.py (maps small 8 -> medium 4, 8, 12, 16, 20).

    .venv\\Scripts\\python.exe tools/transfer/10_multilayer_export.py --quick
    .venv\\Scripts\\python.exe tools/transfer/10_multilayer_export.py

Same records and recipes as 06_export_eval.py. The change the edit makes to small's layer-8 state is translated once per medium layer with that
layer's map and added at every layer of the configuration. Configurations: single layers 4, 8, 12, 16, 20; layer sets 12+16, 8+12+16, 12+16+20,
8+12+16+20, 4+8+12+16+20 with the dose SPLIT over the layers (dose/n each, same total budget as one layer); the same sets with the FULL dose at
every layer are run as an exploratory extra and cannot pass the gate. Doses 0.5 to 6 chosen per configuration on the dev records whose edit
reached rank 1 in small; one norm-matched (outcome-free) dose per layer as a robustness check. Arms: translated | random (same size) |
wrong_recipe (another record's edit). Stage 2 (rewordings, neighbours, unrelated prompts) runs on the test records for the dev-selected best
single layer and best split set only.
Gate 4 (test records whose edit reached rank 1 in small): (a) best set's top-1 >= best single's + 10 points; (b) paired Wilcoxon on the log-rank
gain, set against single, p < 0.01; (c) best set's top-1 >= 15% and >= 10 points above its random and wrong-recipe controls (each at its best dose).
Reported, not gated: the proposed rerun bar (top-1 >= 50%, rewordings lift >= 80% of the edit's own lift in small, neighbours fall <= 5 points).
Stage 1 is saved when finished (d5_stage1_<tag>.pt); --resume reuses it.
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.sae_utils import load_sae_for_layer
from src.transfer.export_eval import LinearTranslator, make_arms, make_recipe, norm_match_factors, plain_logits, small_change, summarise_logits
from src.transfer.models import load_model, pick_device
from src.transfer.multilayer import layer_share, make_configs, run_injected_multi
from src.transfer.swap import NEUTRAL, run_injected

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
ARMS = ["translated", "random", "wrong_recipe"]
GATE_MARGIN_SET, GATE_P, GATE_RATE, GATE_MARGIN_CTRL = 0.10, 0.01, 0.15, 0.10
BAR_TOP1, BAR_LIFT, BAR_NS_LOSS = 0.50, 0.80, 0.05


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rate(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def med(xs):
    xs = sorted(x for x in xs if x == x)
    return xs[len(xs) // 2] if xs else float("nan")


def pct(x):
    return "  n/a" if x != x else f"{100 * x:4.0f}%"


def multi_states(model, tokens, layers):
    names = {f"blocks.{q}.hook_resid_pre" for q in layers}
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names, stop_at_layer=max(layers) + 1)
    return {q: cache[f"blocks.{q}.hook_resid_pre"][0] for q in layers}


def translated_set(env, tok, recipe_real, recipe_wrong, seed):
    """Per medium layer: translated change, wrong-recipe change, random change (same size), norm-matching factor."""
    small, medium, sae, trs, layers, dev = env["small"], env["medium"], env["sae"], env["trs"], env["layers"], env["dev"]
    h, dS = small_change(small, sae, tok, recipe_real)
    hw, dSw = small_change(small, sae, tok, recipe_wrong)
    hm = multi_states(medium, tok, layers)
    D = {}
    for q in layers:
        Dt, Dw = trs[q](h, dS), trs[q](hw, dSw)
        arms = make_arms(Dt, Dw, torch.ones(Dt.shape[-1], device=dev), torch.Generator().manual_seed(seed * 100 + q))
        D[q] = {"translated": Dt, "random": arms["random"], "wrong_recipe": arms["wrong_recipe"], "nm": norm_match_factors(h, dS, hm[q], Dt)}
    return D


def config_deltas(D, layers, mode, arms, doses, nm):
    """Deltas {layer: [R, L, d]} for every (arm, dose) row, with a final norm-matched row per arm when nm. Row order: arm-major."""
    share = layer_share(layers, mode)
    out = {}
    for q in layers:
        rows = []
        for arm in arms:
            for a in doses:
                rows.append(share * a * D[q][arm])
            if nm:
                rows.append(share * D[q]["nm"][:, None] * D[q][arm])
        out[q] = torch.stack(rows)
    return out


def split_rows(summary, arms, n_per_arm):
    return {arm: {k: v[i * n_per_arm:(i + 1) * n_per_arm] for k, v in summary.items()} for i, arm in enumerate(arms)}


# ------------------------------------------------------------------------------------------------------------------------------
def stage1_record(env, rec, edit, wrong_edit, cfgs):
    small, medium, sae, dev = env["small"], env["medium"], env["sae"], env["dev"]
    tok = small.to_tokens(rec["prompt"])
    tn, tt = rec["tid_new"], rec["tid_true"]
    D = translated_set(env, tok, make_recipe(edit["real"], dev), make_recipe(wrong_edit["real"], dev), rec["index"])
    out = {"base": summarise_logits(plain_logits(medium, tok)[None], tn, tt)}
    _, dS_full = small_change(small, sae, tok, make_recipe(edit["real"], dev), zero_bos=False)
    out["small_native"] = summarise_logits(run_injected(small, tok, dS_full[None], 8), tn, tt)
    for name, layers, mode in cfgs:
        deltas = config_deltas(D, layers, mode, ARMS, ALPHAS, nm=True)
        n_per = len(ALPHAS) + 1                                         # the last row of every arm is the norm-matched dose
        lg = run_injected_multi(medium, tok.repeat(len(ARMS) * n_per, 1), deltas)
        out[name] = split_rows(summarise_logits(lg, tn, tt), ARMS, n_per)
    return out


def pick_dose(bench, edits, res1, cfg):
    """Dose chosen on the dev records whose edit reached rank 1 in small: top-1 rate, then mean log-rank gain, then the smaller dose."""
    ids = [r["case_id"] for r in bench if r["split"] == "dev" and edits[r["case_id"]]["real"]["rank"] == 1]
    if not ids:
        ids = [r["case_id"] for r in bench if r["split"] == "dev"]
    best = None
    for i, a in enumerate(ALPHAS):
        top1 = rate([res1[c][cfg]["translated"]["top1"][i].item() for c in ids])
        gain = rate([math.log(res1[c]["base"]["rank_new"][0].item()) - math.log(res1[c][cfg]["translated"]["rank_new"][i].item()) for c in ids])
        key = (round(top1, 6), gain, -a)
        if best is None or key > best[0]:
            best = (key, i)
    return best[1], best[0][0], best[0][1]


def test_ids(bench, edits, subset="small_ok"):
    return [r["case_id"] for r in bench if r["split"] == "test" and (subset == "all" or edits[r["case_id"]]["real"]["rank"] == 1)]


def gain_of(res1, c, cfg, arm, i):
    return math.log(res1[c]["base"]["rank_new"][0].item()) - math.log(res1[c][cfg][arm]["rank_new"][i].item())


def stage1_tables(bench, edits, res1, cfgs, sel):
    lines = []
    ids = test_ids(bench, edits)
    lines.append(f"  test records whose edit reached rank 1 in small: {len(ids)}   (unchanged medium: top-1 {pct(rate([res1[c]['base']['top1'][0].item() for c in ids]))}, "
                 f"median rank of the target {med([res1[c]['base']['rank_new'][0].item() for c in ids]):.0f};  small with the edit: top-1 {pct(rate([res1[c]['small_native']['top1'][0].item() for c in ids]))})")
    lines.append(f"  {'configuration':<22}{'dev dose':>9}{'dev top-1':>10}{'| test top-1':>13}{'median rank':>12}{'| norm-matched top-1':>21}{'median rank':>12}{'| random':>10}{'wrong rec.':>11}   (controls at the same dose)")
    for name, layers, mode in cfgs:
        i, dev_top1, _ = sel[name]
        t = rate([res1[c][name]["translated"]["top1"][i].item() for c in ids])
        mr = med([res1[c][name]["translated"]["rank_new"][i].item() for c in ids])
        tn = rate([res1[c][name]["translated"]["top1"][-1].item() for c in ids])
        mn = med([res1[c][name]["translated"]["rank_new"][-1].item() for c in ids])
        rr = rate([res1[c][name]["random"]["top1"][i].item() for c in ids])
        rw = rate([res1[c][name]["wrong_recipe"]["top1"][i].item() for c in ids])
        lines.append(f"  {name:<22}{ALPHAS[i]:>9}{pct(dev_top1):>10}{pct(t):>13}{mr:>12.0f}{pct(tn):>21}{mn:>12.0f}{pct(rr):>10}{pct(rw):>11}")
    return "\n".join(lines)


def gate4(bench, edits, res1, cfgs, sel, min_n=5):
    from scipy.stats import wilcoxon
    ids = test_ids(bench, edits)
    singles = [c for c in cfgs if c[2] == "single"]
    sets = [c for c in cfgs if c[2] == "split"]
    if len(ids) < min_n or not singles or not sets:
        return {"n": len(ids), "note": "too few records or configurations"}
    pick = lambda group: max(group, key=lambda c: (round(sel[c[0]][1], 6), sel[c[0]][2]))[0]
    bs, bm = pick(singles), pick(sets)
    i_s, i_m = sel[bs][0], sel[bm][0]
    top_s = rate([res1[c][bs]["translated"]["top1"][i_s].item() for c in ids])
    top_m = rate([res1[c][bm]["translated"]["top1"][i_m].item() for c in ids])
    x = [gain_of(res1, c, bm, "translated", i_m) for c in ids]
    y = [gain_of(res1, c, bs, "translated", i_s) for c in ids]
    try:
        p = float(wilcoxon(x, y, alternative="greater").pvalue)
    except ValueError:
        p = 1.0
    ctrl = {}
    for arm in ("random", "wrong_recipe"):
        ctrl[arm] = max(rate([res1[c][bm][arm]["top1"][i].item() for c in ids]) for i in range(len(ALPHAS)))
    a_ok = (top_m - top_s) >= GATE_MARGIN_SET
    b_ok = p < GATE_P
    c_ok = top_m >= GATE_RATE and all(top_m - v >= GATE_MARGIN_CTRL for v in ctrl.values())
    return {"n": len(ids), "best_single": bs, "best_single_dose": ALPHAS[i_s], "best_single_top1": top_s, "best_set": bm, "best_set_dose": ALPHAS[i_m], "best_set_top1": top_m,
            "wilcoxon_p": p, "controls_best_dose": ctrl, "a": bool(a_ok), "b": bool(b_ok), "c": bool(c_ok), "passes": bool(a_ok and b_ok and c_ok)}


# ------------------------------------------------------------------------------------------------------------------------------
def stage2_record(env, rec, edit, wrong_edit, chosen, clean_cache):
    """chosen: {cfg name: (layers, mode, dose index)}. Re-apply the recipe to paraphrases, neighbours and unrelated prompts."""
    small, medium, dev = env["small"], env["medium"], env["dev"]
    tn, tt = rec["tid_new"], rec["tid_true"]
    rr, rw = make_recipe(edit["real"], dev), make_recipe(wrong_edit["real"], dev)
    out = {}
    for kind, plist in (("paraphrase", rec["paraphrases"]), ("neighbour", rec["neighbours"]), ("unrelated", NEUTRAL)):
        rows = []
        for text in plist:
            tok = small.to_tokens(text)
            if text not in clean_cache:
                clean_cache[text] = plain_logits(medium, tok)
            clean = clean_cache[text]
            D = translated_set(env, tok, rr, rw, rec["index"] + len(text))
            row = {"base": summarise_logits(clean[None], tn, tt, clean[None])}
            sc = plain_logits(small, tok)
            _, dS_full = small_change(small, env["sae"], tok, rr, zero_bos=False)
            row["small_base"] = summarise_logits(sc[None], tn, tt, sc[None])
            row["small_native"] = summarise_logits(run_injected(small, tok, dS_full[None], 8), tn, tt, sc[None])
            for name, (layers, mode, i) in chosen.items():
                arms = ["translated", "wrong_recipe"]
                deltas = config_deltas(D, layers, mode, arms, [ALPHAS[i]], nm=True)             # rows: translated dose, translated nm, wrong dose, wrong nm
                lg = run_injected_multi(medium, tok.repeat(4, 1), deltas)
                s = summarise_logits(lg, tn, tt, clean[None].repeat(4, 1))
                row[name] = {"translated": {k: v[0:2] for k, v in s.items()}, "wrong_recipe": {k: v[2:4] for k, v in s.items()}}
            rows.append(row)
        out[kind] = rows
    return out


def stage2_table(bench, edits, res2, chosen):
    ids = [c for c in test_ids(bench, edits) if c in res2]
    lines = [f"  {len(ids)} test records whose edit reached rank 1 in small; cells: PS (rewordings: new beats true) / NS (neighbours: true still beats new) / unrelated KL / unrelated top-1 flips"]
    lines.append(f"    {'arm':<34}{'PS':>7}{'NS':>7}{'KL':>8}{'flips':>8}")

    def get(arm, which, kind, col):
        v = []
        for c in ids:
            for row in res2[c][kind]:
                r = row[arm] if which is None else row[arm[0]][arm[1]]
                v.append(float(r[col][0] if which is None else r[col][which]))
        return rate(v)

    summary = {}
    for label, arm in (("medium unchanged", "base"), ("small unchanged", "small_base"), ("small with the edit", "small_native")):
        ps, ns = get(arm, None, "paraphrase", "es"), 1 - get(arm, None, "neighbour", "es")
        kl, fl = get(arm, None, "unrelated", "kl"), get(arm, None, "unrelated", "flip")
        summary[label] = {"PS": ps, "NS": ns}
        lines.append(f"    {label:<34}{pct(ps):>7}{pct(ns):>7}{kl:>8.3f}{pct(fl):>8}")
    for name in chosen:
        for arm in ("translated", "wrong_recipe"):
            for which, tag in ((0, "chosen dose"), (1, "norm matched")):
                key = (name, arm)
                ps, ns = get(key, which, "paraphrase", "es"), 1 - get(key, which, "neighbour", "es")
                kl, fl = get(key, which, "unrelated", "kl"), get(key, which, "unrelated", "flip")
                summary[f"{name} {arm} {tag}"] = {"PS": ps, "NS": ns, "KL": kl, "flips": fl}
                lines.append(f"    {name + ' ' + arm + ' (' + tag + ')':<34}{pct(ps):>7}{pct(ns):>7}{kl:>8.3f}{pct(fl):>8}")
    return "\n".join(lines), summary


def rerun_bar(g, res1, bench, edits, s2sum, chosen):
    """The proposed bar on the better (by test top-1) of the two configurations that went through stage 2."""
    ids = test_ids(bench, edits)
    best = max(chosen, key=lambda n: rate([res1[c][n]["translated"]["top1"][chosen[n][2]].item() for c in ids]))
    top1 = rate([res1[c][best]["translated"]["top1"][chosen[best][2]].item() for c in ids])
    s = s2sum[f"{best} translated chosen dose"]
    lift_src = s2sum["small with the edit"]["PS"] - s2sum["small unchanged"]["PS"]
    lift_here = s["PS"] - s2sum["medium unchanged"]["PS"]
    ratio = lift_here / lift_src if lift_src > 0 else float("nan")
    ns_loss = s2sum["medium unchanged"]["NS"] - s["NS"]
    ok = top1 >= BAR_TOP1 and ratio >= BAR_LIFT and ns_loss <= BAR_NS_LOSS
    return {"config": best, "top1": top1, "lift_ratio": ratio, "ns_loss": ns_loss, "meets_bar": bool(ok)}


# ------------------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse the saved stage 1")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    suffix = "_quick" if a.quick else ""
    dev = pick_device(a.device)
    t0 = time.time()
    bench = json.load(open(os.path.join(OUT_DIR, f"d_benchmark{suffix}.json"), encoding="utf8"))
    edits = torch.load(os.path.join(OUT_DIR, f"d_edits{suffix}.pt"), weights_only=False)
    bench = [r for r in bench if r["case_id"] in edits]
    maps = torch.load(os.path.join(OUT_DIR, f"maps_gpt2_to_gpt2-medium{suffix}.pt"))["maps"]
    have = [q for q in (4, 8, 12, 16, 20) if f"s2m_L8_L{q}" in maps]
    cfgs = [c for c in make_configs() if all(q in have for q in c[1])]
    layers = sorted({q for c in cfgs for q in c[1]})
    print(f"{len(bench)} records ({sum(r['split'] == 'dev' for r in bench)} dev, {sum(r['split'] == 'test' for r in bench)} test); medium layers with a map: {have}; {len(cfgs)} configurations", flush=True)
    small, medium, sae = load_model("gpt2", dev), load_model("gpt2-medium", dev), load_sae_for_layer(8, dev)
    trs = {q: LinearTranslator(maps[f"s2m_L8_L{q}"]["W"].to(dev)) for q in layers}
    env = {"small": small, "medium": medium, "sae": sae, "trs": trs, "layers": layers, "dev": dev}

    path1 = os.path.join(OUT_DIR, f"d5_stage1{suffix}.pt")
    if a.resume and os.path.exists(path1):
        res1 = torch.load(path1, weights_only=False)["res1"]
        stage(f"stage 1 loaded from {os.path.relpath(path1, ROOT)}")
    else:
        stage("stage 1: tuned prompt, all configurations, all doses")
        res1 = {}
        for n, r in enumerate(bench):
            wrong = edits[bench[(n + 7) % len(bench)]["case_id"]]
            res1[r["case_id"]] = stage1_record(env, r, edits[r["case_id"]], wrong, cfgs)
            if (n + 1) % 25 == 0:
                print(f"   {n + 1}/{len(bench)} ({time.time() - t0:.0f} s)", flush=True)
        torch.save({"res1": res1}, path1)
    sel = {c[0]: pick_dose(bench, edits, res1, c[0]) for c in cfgs}
    g = gate4(bench, edits, res1, cfgs, sel, min_n=1 if a.quick else 5)

    res2, s2txt, s2sum, bar = {}, "", {}, None
    if "note" not in g:
        chosen = {g["best_single"]: ([c for c in cfgs if c[0] == g["best_single"]][0][1], "single", sel[g["best_single"]][0]),
                  g["best_set"]: ([c for c in cfgs if c[0] == g["best_set"]][0][1], "split", sel[g["best_set"]][0])}
        stage(f"stage 2 on test records for {list(chosen)}")
        test_recs = [r for r in bench if r["split"] == "test"]
        clean_cache = {}
        for n, r in enumerate(test_recs):
            wrong = edits[bench[(bench.index(r) + 7) % len(bench)]["case_id"]]
            res2[r["case_id"]] = stage2_record(env, r, edits[r["case_id"]], wrong, chosen, clean_cache)
            if (n + 1) % 25 == 0:
                print(f"   {n + 1}/{len(test_recs)} ({time.time() - t0:.0f} s)", flush=True)
        s2txt, s2sum = stage2_table(bench, edits, res2, chosen)
        bar = rerun_bar(g, res1, bench, edits, s2sum, chosen)
        torch.save({"res1": res1, "res2": res2, "sel": sel, "gate4": g}, os.path.join(OUT_DIR, f"d5_multilayer{suffix}.pt"))

    print("\n" + "=" * 140)
    print("RESULT  step D5 (translated edit at several layers of GPT-2 medium)" + ("  [quick: smoke test, do not read the numbers]" if a.quick else ""))
    print("\n  STAGE 1, tuned prompt (cells are on the test records whose edit reached rank 1 in small; dose chosen per configuration on dev)")
    print(stage1_tables(bench, edits, res1, cfgs, sel))
    if "note" in g:
        print(f"\n  GATE 4: {g['note']}")
    else:
        print(f"\n  GATE 4 ({g['n']} records): best single layer on dev = {g['best_single']} (dose {g['best_single_dose']}, test top-1 {pct(g['best_single_top1'])});  best split set on dev = {g['best_set']} (dose {g['best_set_dose']}, test top-1 {pct(g['best_set_top1'])})")
        print(f"    (a) set minus single >= {100 * GATE_MARGIN_SET:.0f} points: {100 * (g['best_set_top1'] - g['best_single_top1']):+.0f} -> {'ok' if g['a'] else 'NO'};   (b) paired p = {g['wilcoxon_p']:.1e} < {GATE_P}: {'ok' if g['b'] else 'NO'};   "
              f"(c) top-1 >= {GATE_RATE:.0%} and {100 * GATE_MARGIN_CTRL:.0f} points above random {pct(g['controls_best_dose']['random'])} and wrong recipe {pct(g['controls_best_dose']['wrong_recipe'])}: {'ok' if g['c'] else 'NO'}")
        print(f"    ->  {'PASS' if g['passes'] else 'FAIL'}")
        print("\n  STAGE 2, rewordings / neighbours / unrelated prompts")
        print(s2txt)
        print(f"\n  PROPOSED RERUN BAR (reported, not gated) on {bar['config']}: top-1 {pct(bar['top1'])} (>= {BAR_TOP1:.0%});  rewordings lift = {bar['lift_ratio']:.2f} of the edit's own lift in small (>= {BAR_LIFT});  "
              f"neighbours fall {100 * bar['ns_loss']:.0f} points (<= {100 * BAR_NS_LOSS:.0f})  ->  {'MEETS' if bar['meets_bar'] else 'DOES NOT MEET'} the bar")
    out_json = os.path.join(OUT_DIR, f"d5_multilayer{suffix}.json")
    json.dump({"selection": {k: {"dose": ALPHAS[v[0]], "dev_top1": v[1], "dev_gain": v[2]} for k, v in sel.items()}, "gate4": g, "stage2": s2sum, "rerun_bar": bar}, open(out_json, "w", encoding="utf8"), indent=1, default=float)
    print(f"\n  total time {time.time() - t0:.0f} s; saved {os.path.relpath(out_json, ROOT)} and the per-record numbers (d5_stage1{suffix}.pt, d5_multilayer{suffix}.pt)")
    print("=" * 140)
    return 0


if __name__ == "__main__":
    sys.exit(main())
