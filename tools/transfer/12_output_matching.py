"""
Step D7 of docs/cross_model_transfer/PLAN.md: a translator trained on edits with an output-matching loss. Gate 6 is written in the plan before this run.
Needs 05_export_edits.py (benchmark, original edits) and 01_fit_maps.py (ridge map small 8 -> medium 16).

    .venv\\Scripts\\python.exe tools/transfer/12_output_matching.py --quick       # smoke test, a few minutes
    .venv\\Scripts\\python.exe tools/transfer/12_output_matching.py               # about 1.5 to 2 hours; every stage is saved and resumable

Stages (progress is printed all the time):
  1. training pool: up to --n-train (800) CounterFact records disjoint from every dev/test/pilot/ROME-dev record AND whose counterfactual target word
     and subject are not those of any dev or test record (so the map cannot learn word-level pushes). CounterFact reuses target words a lot: in the
     full run 829 of 1,450 candidates were dropped by this rule, leaving 621 records, of which 289 had an edit that reached rank 1 in small;
  2. the original gradient-descent edit is tuned in small for each (about 3 s per record, saved as it goes); only edits that reached rank 1 are used;
  3. training of a linear map small 8 -> medium 16, started at the ridge solution, with loss KL(goal || medium after injection), goal = medium's clean
     logits + (small's edited logits - small's clean logits), on the tuned prompt, 2 rewordings and 2 neighbours of every training record;
  4. choice of epoch and dose on the 50 dev records; 5. judgement on the 150 test records (same small-tuned recipes as D3) and Gate 6.
"""
import argparse
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.editing import build_clean_context
from src.gradient_editing import run_gradient_descent_edit
from src.sae_utils import load_sae_for_layer
from src.transfer.benchmark import build_benchmark
from src.transfer.export_eval import make_recipe, plain_logits, small_change, summarise_logits
from src.transfer.models import load_model, pick_device
from src.transfer.output_matching import build_item, dev_scores, item_loss
from src.transfer.swap import NEUTRAL, run_injected

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
HOOK = "blocks.8.hook_resid_pre"
Q = 16
DOSES = [0.5, 1.0, 1.5, 2.0, 3.0]
RIDGE_DOSE = 2.0
GATE_TOP1, GATE_LIFT = 0.50, 0.80
ARMS = ["om_real", "ridge_real", "om_wrong", "om_rand"]


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


# ------------------------------------------------------------------------------------------------------------------------------
def build_pool(small, medium, bench, n_train, suffix):
    path = os.path.join(OUT_DIR, f"d7_pool{suffix}.json")
    n_pool = int(n_train * 1.8) + 10
    pool = json.load(open(path, encoding="utf8")) if os.path.exists(path) else []
    if len(pool) < n_pool:
        pilot = {r["case_id"] for r in json.load(open(os.path.join(OUT_DIR, "d_pilot.json"), encoding="utf8"))["records"]}
        exclude = pilot | {r["case_id"] for r in bench}
        pool = build_benchmark(small, medium, 0, n_pool, exclude_case_ids=exclude, seed=2)
        json.dump(pool, open(path, "w", encoding="utf8"))
    held_words = {r["tid_new"] for r in bench}
    held_subjects = {r["subject"] for r in bench}
    keep = [r for r in pool if r["tid_new"] not in held_words and r["subject"] not in held_subjects]
    print(f"   pool of {len(pool)} candidate records; {len(pool) - len(keep)} dropped because their target word or subject is used by a dev/test record; using {min(len(keep), n_train)}", flush=True)
    return keep[:n_train]


def tune_edits(small, sae, pool, suffix):
    path = os.path.join(OUT_DIR, f"d7_train_edits{suffix}.pt")
    edits = torch.load(path, weights_only=False) if os.path.exists(path) else {}
    todo = [r for r in pool if r["case_id"] not in edits]
    print(f"   {len(edits)} edits already done, {len(todo)} to tune (about 3 s each)", flush=True)
    t0 = time.time()
    for n, r in enumerate(todo):
        ctx = build_clean_context(small, sae, r["prompt"], r["tid_new"])
        real = run_gradient_descent_edit(small, sae, ctx, r["tid_new"], r["prompt"], 8, HOOK, top_n=200, steps=100)
        edits[r["case_id"]] = {"fids": real["fids"], "a": real["a"], "start_rank": ctx.clean_rank, "rank": real["real_path"]["rank"], "kl": real["real_path"]["kl"]}
        if (n + 1) % 5 == 0 or n + 1 == len(todo):
            torch.save(edits, path)
        if (n + 1) % 10 == 0 or n + 1 == len(todo):
            el = time.time() - t0
            ok = sum(e["rank"] == 1 for e in edits.values())
            print(f"   edits {len(edits)}/{len(pool)}; rank 1 in small on {ok} ({ok / len(edits):.0%}); {el:.0f} s, about {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min left", flush=True)
    torch.save(edits, path)
    return edits


# ------------------------------------------------------------------------------------------------------------------------------
def recipe_rows(env, tok, specs):
    """Last-position medium logits [R, V]; spec = (recipe, W, dose): that recipe's change in small, through W, times dose, added at medium layer Q."""
    small, medium, sae, dev = env["small"], env["medium"], env["sae"], env["dev"]
    ds = []
    for recipe, W, dose in specs:
        _, dS = small_change(small, sae, tok, make_recipe(recipe, dev))
        ds.append(dose * (dS @ W))
    return run_injected(medium, tok.repeat(len(specs), 1), torch.stack(ds), Q)


def judge(lg, tn, tt, tw, has_rand, clean=None):
    kw = {} if clean is None else {"clean_logits": clean[None].repeat(3, 1)}
    s = summarise_logits(lg[:3], tn, tt, **kw)
    out = {arm: {k: v[i:i + 1] for k, v in s.items()} for i, arm in enumerate(ARMS[:3])}
    if has_rand:
        kw = {} if clean is None else {"clean_logits": clean[None]}
        out["om_rand"] = summarise_logits(lg[3:4], tw, tt, **kw)
    return out


def evaluate_record(env, rec, edit, wrong_edit, W_tr, dose, clean_cache):
    small, medium, sae, dev, W0 = env["small"], env["medium"], env["sae"], env["dev"], env["W0"]
    tn, tt = rec["tid_new"], rec["tid_true"]
    has_rand = "random" in edit
    tw = edit["random"]["tid"] if has_rand else None
    specs = [(edit["real"], W_tr, dose), (edit["real"], W0, RIDGE_DOSE), (wrong_edit["real"], W_tr, dose)] + ([(edit["random"], W_tr, dose)] if has_rand else [])
    tok = small.to_tokens(rec["prompt"])
    out1 = {"base": summarise_logits(plain_logits(medium, tok)[None], tn, tt)}
    if has_rand:
        out1["base_rand"] = summarise_logits(plain_logits(medium, tok)[None], tw, tt)
    out1.update(judge(recipe_rows(env, tok, specs), tn, tt, tw, has_rand))
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
            row.update(judge(recipe_rows(env, tk, specs), tn, tt, tw, has_rand, clean=clean))
            rows.append(row)
        out2[kind] = rows
    return out1, out2


def agg2(res2, ids, arm, kind, col):
    return rate([float(row[arm][col][0]) for c in ids if c in res2 for row in res2[c][kind] if arm in row])


def tables(bench, edits, res1, res2):
    summary = {}
    for label, subset in (("PRIMARY: test records whose original edit reached rank 1 in small", "ok"), ("ALL test records", "all")):
        ids = [r["case_id"] for r in bench if r["split"] == "test" and r["case_id"] in res1 and (subset == "all" or edits[r["case_id"]]["real"]["rank"] == 1)]
        print(f"\n  ---- {label}: {len(ids)} records ----")
        print(f"  tuned prompt (unchanged medium: top-1 {pct(rate([res1[c]['base']['top1'][0].item() for c in ids]))}, median rank of the target {med([res1[c]['base']['rank_new'][0].item() for c in ids]):.0f})")
        print(f"    {'arm':<14}{'top-1':>8}{'median rank':>13}")
        for arm in ARMS:
            pick = [c for c in ids if arm in res1[c]]
            if pick:
                print(f"    {arm:<14}{pct(rate([res1[c][arm]['top1'][0].item() for c in pick])):>8}{med([res1[c][arm]['rank_new'][0].item() for c in pick]):>13.0f}"
                      + (f"   (that word's rank unchanged: {med([res1[c]['base_rand']['rank_new'][0].item() for c in pick]):.0f}; {len(pick)} records)" if arm == "om_rand" else ""))
        print(f"  rewordings / neighbours / unrelated (PS: new beats true; NS: true still beats new; KL and top-1 flips on unrelated)")
        print(f"    {'arm':<22}{'PS':>7}{'NS':>7}{'KL':>8}{'flips':>8}")
        s = {}
        for label2, arm in (("medium unchanged", "base"), ("small unchanged", "small_base"), ("small with the edit", "small_native")) + tuple((a_, a_) for a_ in ARMS):
            ps, ns = agg2(res2, ids, arm, "paraphrase", "es"), 1 - agg2(res2, ids, arm, "neighbour", "es")
            kl, fl = agg2(res2, ids, arm, "unrelated", "kl"), agg2(res2, ids, arm, "unrelated", "flip")
            s[label2] = {"PS": ps, "NS": ns, "KL": kl, "flips": fl}
            print(f"    {label2:<22}{pct(ps):>7}{pct(ns):>7}{kl:>8.3f}{pct(fl):>8}")
        summary[subset] = {"n": len(ids), "top1": {arm: rate([res1[c][arm]["top1"][0].item() for c in ids if arm in res1[c]]) for arm in ARMS},
                           "median_rank": {arm: med([res1[c][arm]["rank_new"][0].item() for c in ids if arm in res1[c]]) for arm in ARMS}, "stage2": s}
    return summary


def gate6(summary, bench, edits, res1):
    from scipy.stats import wilcoxon
    s = summary["ok"]
    top1 = s["top1"]["om_real"]
    lift_src = s["stage2"]["small with the edit"]["PS"] - s["stage2"]["small unchanged"]["PS"]
    lift_here = s["stage2"]["om_real"]["PS"] - s["stage2"]["medium unchanged"]["PS"]
    ratio = lift_here / lift_src if lift_src > 0 else float("nan")
    ns_here = s["stage2"]["medium unchanged"]["NS"] - s["stage2"]["om_real"]["NS"]
    ns_src = s["stage2"]["small unchanged"]["NS"] - s["stage2"]["small with the edit"]["NS"]
    a, b, c = top1 >= GATE_TOP1, ratio >= GATE_LIFT, ns_here <= ns_src
    ids = [r["case_id"] for r in bench if r["split"] == "test" and r["case_id"] in res1 and edits[r["case_id"]]["real"]["rank"] == 1]
    gain = lambda arm: [math.log(res1[c]["base"]["rank_new"][0].item()) - math.log(res1[c][arm]["rank_new"][0].item()) for c in ids]
    try:
        p = float(wilcoxon(gain("om_real"), gain("ridge_real"), alternative="greater").pvalue)
    except ValueError:
        p = 1.0
    reading = "all three hold: one trained map carries a small-tuned edit into medium with no tuning on medium" if (a and b and c) else \
        ("(a) holds but (b) or (c) fails: a push on the tuned prompt" if a else "(a) fails: output matching with this map and data does not rescue the export")
    return {"n": len(ids), "top1": top1, "lift_ratio": ratio, "ns_loss": ns_here, "ns_loss_source": ns_src, "a": bool(a), "b": bool(b), "c": bool(c), "passes": bool(a and b and c),
            "paired_p_vs_ridge": p, "reading": reading}


# ------------------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--retrain", action="store_true", help="ignore the saved trained map")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.quick:
        a.n_train, a.epochs = 6, 2
    suffix = "_quick" if a.quick else ""
    dev = pick_device(a.device)
    t0 = time.time()
    bench = json.load(open(os.path.join(OUT_DIR, f"d_benchmark{suffix}.json"), encoding="utf8"))
    edits = torch.load(os.path.join(OUT_DIR, f"d_edits{suffix}.pt"), weights_only=False)
    bench = [r for r in bench if r["case_id"] in edits]
    test = [r for r in bench if r["split"] == "test"]
    dev_recs = [r for r in bench if r["split"] == "dev" and edits[r["case_id"]]["real"]["rank"] == 1]
    maps = torch.load(os.path.join(OUT_DIR, f"maps_gpt2_to_gpt2-medium{suffix}.pt"))["maps"]
    W0 = maps[f"s2m_L8_L{Q}"]["W"].to(dev)
    small, medium, sae = load_model("gpt2", dev), load_model("gpt2-medium", dev), load_sae_for_layer(8, dev)
    for m in (small, medium):
        for p_ in m.parameters():
            p_.requires_grad_(False)
    env = {"small": small, "medium": medium, "sae": sae, "dev": dev, "W0": W0}
    print(f"{len(bench)} benchmark records ({len(dev_recs)} dev with a successful edit in small, {len(test)} test); map small 8 -> medium {Q}", flush=True)

    # ---------------- 1. pool, 2. edits ---------------------------------------------------------------------------------------
    stage("1/5  training pool (disjoint from every dev/test/pilot/ROME-dev record and from their target words)")
    pool = build_pool(small, medium, bench, a.n_train, suffix)
    stage("2/5  tuning the original edit in small for every training record")
    tedits = tune_edits(small, sae, pool, suffix)
    good = [r for r in pool if tedits[r["case_id"]]["rank"] == 1]
    print(f"   {len(good)} of {len(pool)} edits reached rank 1 in small and are used for training", flush=True)

    # ---------------- 3. training ---------------------------------------------------------------------------------------------
    map_path = os.path.join(OUT_DIR, f"d7_map{suffix}.pt")
    if os.path.exists(map_path) and not a.retrain:
        saved = torch.load(map_path, weights_only=False)
        W_best, dose_star = saved["W"].to(dev), saved["dose"]
        stage(f"3/5  trained map loaded from {os.path.relpath(map_path, ROOT)} (epoch {saved['epoch']}, dose {dose_star})")
    else:
        stage("3/5  building training items (tuned prompt, 2 rewordings, 2 neighbours per record)")
        items = []
        for n, r in enumerate(good):
            for text in [r["prompt"]] + r["paraphrases"][:2] + r["neighbours"][:2]:
                items.append(build_item(small, medium, sae, text, tedits[r["case_id"]], Q, dev))
            if (n + 1) % 100 == 0:
                print(f"   {n + 1}/{len(good)} records ({len(items)} items, {time.time() - t0:.0f} s)", flush=True)
        dev_items = []
        for r in dev_recs:
            it = build_item(small, medium, sae, r["prompt"], edits[r["case_id"]]["real"], Q, dev)
            lg = plain_logits(medium, it["tok"])
            it["tid"], it["base_rank"] = r["tid_new"], int((lg > lg[r["tid_new"]]).sum().item()) + 1
            dev_items.append(it)
        print(f"   {len(items)} training items from {len(good)} records; {len(dev_items)} dev items", flush=True)
        W = torch.nn.Parameter(W0.clone())
        opt = torch.optim.Adam([W], lr=a.lr)
        w0n = (W0 ** 2).sum()
        g = torch.Generator().manual_seed(0)
        t1, g1, r1 = dev_scores(medium, dev_items, W0, 1.0, Q)
        print(f"   start (ridge map, dose 1): dev top-1 {pct(t1)}, mean log-rank gain {g1:.2f}, median rank {r1}", flush=True)
        best_gain, W_best, best_ep = -1e9, W0.clone(), 0
        stage(f"4/5  training {a.epochs} epochs ({len(items)} items per epoch, batch {a.batch})")
        for ep in range(1, a.epochs + 1):
            perm = torch.randperm(len(items), generator=g).tolist()
            tot, nb, te = 0.0, 0, time.time()
            for s in range(0, len(perm), a.batch):
                batch = [items[i] for i in perm[s:s + a.batch]]
                loss = sum(item_loss(medium, it, W, 1.0, Q) for it in batch) / len(batch)
                reg = a.reg * ((W - W0) ** 2).sum() / w0n
                opt.zero_grad()
                (loss + reg).backward()
                opt.step()
                tot, nb = tot + float(loss.item()), nb + 1
            t1, g1, r1 = dev_scores(medium, dev_items, W.detach(), 1.0, Q)
            moved = float(((W - W0) ** 2).sum().sqrt() / w0n.sqrt())
            flag = ""
            if g1 > best_gain:
                best_gain, W_best, best_ep, flag = g1, W.detach().clone(), ep, "  <- best so far"
            print(f"   epoch {ep:>2}/{a.epochs}: train KL {tot / nb:.4f}, map moved {100 * moved:.1f}% from ridge; dev top-1 {pct(t1)}, gain {g1:.2f}, median rank {r1}   [{time.time() - te:.0f} s]{flag}", flush=True)
        stage(f"choosing the dose on the dev records (epoch {best_ep})")
        best = None
        for d in DOSES:
            t1, g1, r1 = dev_scores(medium, dev_items, W_best, d, Q)
            print(f"   dose {d}: dev top-1 {pct(t1)}, gain {g1:.2f}, median rank {r1}", flush=True)
            key = (round(t1, 6), g1, -d)
            if best is None or key > best[0]:
                best = (key, d)
        dose_star = best[1]
        torch.save({"W": W_best.cpu(), "epoch": best_ep, "dose": dose_star}, map_path)
        print(f"   chosen: epoch {best_ep}, dose {dose_star}", flush=True)

    # ---------------- 5. judgement --------------------------------------------------------------------------------------------
    stage(f"5/5  judging on the {len(test)} test records (dose {dose_star})")
    res1, res2, clean_cache = {}, {}, {}
    for n, r in enumerate(test):
        cid = r["case_id"]
        other = test[(n + 7) % len(test)]["case_id"]
        res1[cid], res2[cid] = evaluate_record(env, r, edits[cid], edits[other], W_best, dose_star, clean_cache)
        if (n + 1) % 25 == 0:
            print(f"   {n + 1}/{len(test)} ({time.time() - t0:.0f} s)", flush=True)
    torch.save({"res1": res1, "res2": res2}, os.path.join(OUT_DIR, f"d7_eval{suffix}.pt"))

    print("\n" + "=" * 140)
    print("RESULT  step D7 (translator trained on edits with an output-matching loss)" + ("  [quick: smoke test, do not read the numbers]" if a.quick else ""))
    print(f"  training records used: {len(good)} (target words and subjects disjoint from every dev/test record); chosen dose {dose_star}")
    summary = tables(bench, edits, res1, res2)
    g6 = gate6(summary, bench, edits, res1) if "ok" in summary and summary["ok"]["n"] else {"note": "no primary records"}
    if "note" in g6:
        print(f"\n  GATE 6: {g6['note']}")
    else:
        print(f"\n  GATE 6 ({g6['n']} records): (a) top-1 {pct(g6['top1'])} >= {GATE_TOP1:.0%}: {'ok' if g6['a'] else 'NO'};  (b) rewordings lift = {g6['lift_ratio']:.2f} of the edit's own lift in small (>= {GATE_LIFT}): {'ok' if g6['b'] else 'NO'};  "
              f"(c) neighbours fall {100 * g6['ns_loss']:.0f} points, the edit itself in small {100 * g6['ns_loss_source']:.0f}: {'ok' if g6['c'] else 'NO'}   ->  {'PASS' if g6['passes'] else 'FAIL'}")
        print(f"  reading (fixed in advance): {g6['reading']}")
        print(f"  improvement over the ridge map (paired Wilcoxon on the log-rank gain of the target): p = {g6['paired_p_vs_ridge']:.1e}")
    d6 = os.path.join(OUT_DIR, "d6_b_aware.json")
    if os.path.exists(d6):
        print(f"  reference, D6 ceiling (edit tuned per sentence against medium): top-1 {pct(json.load(open(d6))['summary']['ok']['top1']['b_aware'])}")
    out_json = os.path.join(OUT_DIR, f"d7_output_matching{suffix}.json")
    json.dump({"layer": Q, "dose": dose_star, "n_train_records": len(good), "summary": summary, "gate6": g6}, open(out_json, "w", encoding="utf8"), indent=1, default=float)
    print(f"\n  total time {time.time() - t0:.0f} s; saved {os.path.relpath(out_json, ROOT)} (+ d7_map{suffix}.pt, d7_eval{suffix}.pt)")
    print("=" * 140)
    return 0


if __name__ == "__main__":
    sys.exit(main())
