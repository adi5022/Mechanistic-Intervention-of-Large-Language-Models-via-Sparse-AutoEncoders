"""
Step D6 of docs/cross_model_transfer/PLAN.md: the ceiling test. Tune the edit THROUGH the translator against GPT-2 medium's output.
Gate 5 is written in the plan before this run. Needs 05_export_edits.py (benchmark, original edits) and 01_fit_maps.py (map small 8 -> medium 16).

    .venv\\Scripts\\python.exe tools/transfer/11_b_aware_edit.py --quick
    .venv\\Scripts\\python.exe tools/transfer/11_b_aware_edit.py          # 150 test records, about an hour; tuned recipes are saved as it goes (--resume reuses them)

For every test record, the multipliers of the same 200 small-layer-8 SAE features as the original edit are tuned with the original loss and settings,
but the loss is taken on medium's last-position logits after small's change has been translated (linear map small 8 -> medium 16), scaled by a
fixed dose (2.0, chosen on dev in D3, not re-tuned) and added to medium's layer 16. The tuned recipe is then judged through the same hook path as
every other export run, re-applied (relay mode) to rewordings, neighbours and the 12 unrelated prompts.
Arms: b_aware (this step) | small_tuned (the original edit at the same dose, the reference) | wrong_b (another record's b_aware recipe) |
random_word_b (a b_aware edit toward the record's rank-matched random word, judged on that word).
Gate 5 (on the test records whose original edit reached rank 1 in small): (a) b_aware top-1 on the tuned prompt >= 50%; (b) rewordings lift in
"new beats true" over unchanged medium >= 80% of the original edit's lift inside small; (c) neighbours fall <= 5 points.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.sae_utils import load_sae_for_layer
from src.transfer.b_aware_edit import tune_through_translator
from src.transfer.export_eval import LinearTranslator, make_recipe, plain_logits, small_change, summarise_logits
from src.transfer.models import load_model, pick_device
from src.transfer.runlog import tee_to
from src.transfer.swap import NEUTRAL, run_injected

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
Q, DOSE = 16, 2.0
GATE_TOP1, GATE_LIFT, GATE_NS_LOSS = 0.50, 0.80, 0.05
ARMS = ["b_aware", "small_tuned", "wrong_b", "random_word_b"]


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


def recipe_rows(env, tok, recipes):
    """Last-position medium logits [R, V] with each recipe's translated change (times the dose) added at medium layer Q."""
    small, medium, sae, tr, dev = env["small"], env["medium"], env["sae"], env["tr"], env["dev"]
    ds = []
    for r in recipes:
        h, dS = small_change(small, sae, tok, make_recipe(r, dev))
        ds.append(DOSE * tr(h, dS))
    return run_injected(medium, tok.repeat(len(recipes), 1), torch.stack(ds), Q)


def judge(lg, tn, tt, tw, has_rand, clean=None):
    """Summaries per arm: rows 0..2 on the counterfactual target, row 3 (random-word arm) on its own word."""
    n_main = 3
    kw = {} if clean is None else {"clean_logits": clean[None].repeat(n_main, 1)}
    s = summarise_logits(lg[:n_main], tn, tt, **kw)
    out = {arm: {k: v[i:i + 1] for k, v in s.items()} for i, arm in enumerate(ARMS[:n_main])}
    if has_rand:
        kw = {} if clean is None else {"clean_logits": clean[None]}
        out["random_word_b"] = summarise_logits(lg[n_main:n_main + 1], tw, tt, **kw)
    return out


def evaluate_record(env, rec, edit, b_real, b_wrong, b_rand, clean_cache):
    small, medium, sae, dev = env["small"], env["medium"], env["sae"], env["dev"]
    tn, tt = rec["tid_new"], rec["tid_true"]
    tw = edit["random"]["tid"] if "random" in edit else None
    has_rand = b_rand is not None
    recipes = [b_real, edit["real"], b_wrong] + ([b_rand] if has_rand else [])
    tok = small.to_tokens(rec["prompt"])
    out1 = {"base": summarise_logits(plain_logits(medium, tok)[None], tn, tt)}
    if has_rand:
        out1["base_rand"] = summarise_logits(plain_logits(medium, tok)[None], tw, tt)
    out1.update(judge(recipe_rows(env, tok, recipes), tn, tt, tw, has_rand))
    out2 = {}
    for kind, plist in (("paraphrase", rec["paraphrases"]), ("neighbour", rec["neighbours"]), ("unrelated", NEUTRAL)):
        rows = []
        for text in plist:
            tk = small.to_tokens(text)
            if text not in clean_cache:
                clean_cache[text] = plain_logits(medium, tk)
            clean = clean_cache[text]
            row = {"base": summarise_logits(clean[None], tn, tt, clean[None])}
            sc = plain_logits(small, tk)
            _, dS_full = small_change(small, sae, tk, make_recipe(edit["real"], dev), zero_bos=False)
            row["small_base"] = summarise_logits(sc[None], tn, tt, sc[None])
            row["small_native"] = summarise_logits(run_injected(small, tk, dS_full[None], 8), tn, tt, sc[None])
            row.update(judge(recipe_rows(env, tk, recipes), tn, tt, tw, has_rand, clean=clean))
            rows.append(row)
        out2[kind] = rows
    return out1, out2


def agg2(res2, ids, arm, kind, col):
    v = [float(row[arm][col][0]) for c in ids if c in res2 for row in res2[c][kind] if arm in row]
    return rate(v)


def tables(bench, edits, recipes, res1, res2):
    summary = {}
    for label, subset in (("PRIMARY: test records whose original edit reached rank 1 in small", "ok"), ("ALL test records", "all")):
        ids = [r["case_id"] for r in bench if r["split"] == "test" and r["case_id"] in res1 and (subset == "all" or edits[r["case_id"]]["real"]["rank"] == 1)]
        print(f"\n  ---- {label}: {len(ids)} records ----")
        print(f"  tuned prompt (unchanged medium: top-1 {pct(rate([res1[c]['base']['top1'][0].item() for c in ids]))}, median rank of the target {med([res1[c]['base']['rank_new'][0].item() for c in ids]):.0f})")
        print(f"    {'arm':<16}{'top-1':>8}{'median rank':>13}")
        for arm in ARMS:
            pick = [c for c in ids if arm in res1[c]]
            if not pick:
                continue
            base = "base_rand" if arm == "random_word_b" else "base"
            print(f"    {arm:<16}{pct(rate([res1[c][arm]['top1'][0].item() for c in pick])):>8}{med([res1[c][arm]['rank_new'][0].item() for c in pick]):>13.0f}"
                  + (f"   (that word's rank unchanged: {med([res1[c]['base_rand']['rank_new'][0].item() for c in pick]):.0f}; {len(pick)} records)" if arm == "random_word_b" else ""))
        tr_ok = [recipes[c]["real"]["train_rank"] for c in ids]
        print(f"    tuning reached rank 1 at training time on {pct(rate([float(x == 1) for x in tr_ok]))}; edit size in medium (share of its residual norm at the last position): median {100 * med([recipes[c]['real']['size_frac'] for c in ids]):.0f}%")
        if res2:
            print(f"  rewordings / neighbours / unrelated (PS: new beats true; NS: true still beats new; KL and top-1 flips on unrelated)")
            print(f"    {'arm':<22}{'PS':>7}{'NS':>7}{'KL':>8}{'flips':>8}")
            s = {}
            for label2, arm in (("medium unchanged", "base"), ("small unchanged", "small_base"), ("small with the edit", "small_native")) + tuple((a, a) for a in ARMS):
                ps, ns = agg2(res2, ids, arm, "paraphrase", "es"), 1 - agg2(res2, ids, arm, "neighbour", "es")
                kl, fl = agg2(res2, ids, arm, "unrelated", "kl"), agg2(res2, ids, arm, "unrelated", "flip")
                s[label2] = {"PS": ps, "NS": ns, "KL": kl, "flips": fl}
                print(f"    {label2:<22}{pct(ps):>7}{pct(ns):>7}{kl:>8.3f}{pct(fl):>8}")
            summary[subset] = {"n": len(ids), "top1": {arm: rate([res1[c][arm]["top1"][0].item() for c in ids if arm in res1[c]]) for arm in ARMS},
                               "median_rank": {arm: med([res1[c][arm]["rank_new"][0].item() for c in ids if arm in res1[c]]) for arm in ARMS}, "stage2": s,
                               "median_size_frac": med([recipes[c]["real"]["size_frac"] for c in ids])}
    return summary


def gate5(summary):
    s = summary["ok"]
    top1 = s["top1"]["b_aware"]
    lift_src = s["stage2"]["small with the edit"]["PS"] - s["stage2"]["small unchanged"]["PS"]
    lift_here = s["stage2"]["b_aware"]["PS"] - s["stage2"]["medium unchanged"]["PS"]
    ratio = lift_here / lift_src if lift_src > 0 else float("nan")
    ns_loss = s["stage2"]["medium unchanged"]["NS"] - s["stage2"]["b_aware"]["NS"]
    a, b, c = top1 >= GATE_TOP1, ratio >= GATE_LIFT, ns_loss <= GATE_NS_LOSS
    reading = "all three hold: the channel carries a rank-1 edit that generalises" if (a and b and c) else \
        ("(a) holds but (b) or (c) fails: a push on the tuned prompt, not a generalising change" if a else "(a) fails: the channel cannot carry a rank-1 edit in this feature family")
    return {"n": s["n"], "top1": top1, "lift_ratio": ratio, "ns_loss": ns_loss, "a": bool(a), "b": bool(b), "c": bool(c), "passes": bool(a and b and c), "reading": reading}


# ------------------------------------------------------------------------------------------------------------------------------
def main():
    log_path = tee_to("11_b_aware_edit")
    print(f"log: {os.path.relpath(log_path, ROOT)}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore saved tuned recipes")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    suffix = "_quick" if a.quick else ""
    dev = pick_device(a.device)
    t0 = time.time()
    bench = json.load(open(os.path.join(OUT_DIR, f"d_benchmark{suffix}.json"), encoding="utf8"))
    edits = torch.load(os.path.join(OUT_DIR, f"d_edits{suffix}.pt"), weights_only=False)
    bench = [r for r in bench if r["case_id"] in edits]
    test = [r for r in bench if r["split"] == "test"]
    maps = torch.load(os.path.join(OUT_DIR, f"maps_gpt2_to_gpt2-medium{suffix}.pt"))["maps"]
    W = maps[f"s2m_L8_L{Q}"]["W"].to(dev)
    small, medium, sae = load_model("gpt2", dev), load_model("gpt2-medium", dev), load_sae_for_layer(8, dev)
    env = {"small": small, "medium": medium, "sae": sae, "tr": LinearTranslator(W), "dev": dev}
    print(f"{len(test)} test records; injection layer {Q}, dose {DOSE}, translator linear", flush=True)

    path = os.path.join(OUT_DIR, f"d6_recipes{suffix}.pt")
    recipes = {} if (a.fresh or not os.path.exists(path)) else torch.load(path, weights_only=False)
    stage(f"tuning through the translator ({len(recipes)} records already done)")
    for n, r in enumerate(test):
        cid = r["case_id"]
        if cid in recipes:
            continue
        tok = small.to_tokens(r["prompt"])
        fids = edits[cid]["real"]["fids"]
        entry = {"real": tune_through_translator(small, medium, sae, W, fids, tok, r["tid_new"], Q, DOSE)}
        if "random" in edits[cid]:
            entry["rand"] = tune_through_translator(small, medium, sae, W, edits[cid]["random"]["fids"], tok, edits[cid]["random"]["tid"], Q, DOSE)
        recipes[cid] = entry
        if (n + 1) % 10 == 0:
            torch.save(recipes, path)
            print(f"   tuned {n + 1}/{len(test)} ({time.time() - t0:.0f} s)", flush=True)
    torch.save(recipes, path)

    stage("judging the tuned recipes (tuned prompt, rewordings, neighbours, unrelated prompts)")
    res1, res2, clean_cache = {}, {}, {}
    for n, r in enumerate(test):
        cid = r["case_id"]
        other = test[(n + 7) % len(test)]["case_id"]
        res1[cid], res2[cid] = evaluate_record(env, r, edits[cid], recipes[cid]["real"], recipes[other]["real"], recipes[cid].get("rand"), clean_cache)
        if (n + 1) % 25 == 0:
            print(f"   {n + 1}/{len(test)} ({time.time() - t0:.0f} s)", flush=True)
    torch.save({"res1": res1, "res2": res2}, os.path.join(OUT_DIR, f"d6_eval{suffix}.pt"))

    print("\n" + "=" * 140)
    print("RESULT  step D6 (edit tuned through the translator against medium's output)" + ("  [quick: smoke test, do not read the numbers]" if a.quick else ""))
    summary = tables(bench, edits, recipes, res1, res2)
    g = gate5(summary) if "ok" in summary else {"note": "no primary records"}
    if "note" in g:
        print(f"\n  GATE 5: {g['note']}")
    else:
        print(f"\n  GATE 5 ({g['n']} records): (a) top-1 {pct(g['top1'])} >= {GATE_TOP1:.0%}: {'ok' if g['a'] else 'NO'};  (b) rewordings lift = {g['lift_ratio']:.2f} of the edit's own lift in small (>= {GATE_LIFT}): {'ok' if g['b'] else 'NO'};  "
              f"(c) neighbours fall {100 * g['ns_loss']:.0f} points (<= {100 * GATE_NS_LOSS:.0f}): {'ok' if g['c'] else 'NO'}   ->  {'PASS' if g['passes'] else 'FAIL'}")
        print(f"  reading (fixed in advance): {g['reading']}")
    mism = sum(1 for r in test if recipes[r["case_id"]]["real"]["train_rank"] == 1 and res1[r["case_id"]]["b_aware"]["rank_new"][0].item() != 1)
    print(f"  check: records that reached rank 1 at training time but not through the hook path: {mism}")
    out_json = os.path.join(OUT_DIR, f"d6_b_aware{suffix}.json")
    json.dump({"layer": Q, "dose": DOSE, "summary": summary, "gate5": g}, open(out_json, "w", encoding="utf8"), indent=1, default=float)
    print(f"\n  total time {time.time() - t0:.0f} s; saved {os.path.relpath(out_json, ROOT)} (+ d6_recipes{suffix}.pt, d6_eval{suffix}.pt)")
    print("=" * 140)
    return 0


if __name__ == "__main__":
    sys.exit(main())
