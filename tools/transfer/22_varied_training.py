"""
Step D13 of docs/cross_model_transfer/PLAN.md, part 2: train the linear map on the 289 real-target edits plus the synthetic-target edits made by 21_varied_edits.py, and judge
it on the unseen real counterfactual test records. The design, the data curve, Gate 9 (= Gate 7 unchanged) and the readings are in the plan, written before this tool was built.

    .venv\\Scripts\\python.exe tools/transfer/22_varied_training.py --smoke     # rehearsal (2 epochs, 1 seed, 2 data sizes; numbers mean nothing); run it AFTER 21_varied_edits.py has finished
    .venv\\Scripts\\python.exe tools/transfer/22_varied_training.py             # the real run: about an hour, progress printed all the time

Training sets: the 289 real edits only (replicates D7 to D10) and with 25%, 50%, 100% of the synthetic edits (random subsets by seed). Everything else as in D7 to D10: linear map small 8 to
medium 16 from the ridge solution, the D7 output-matching KL loss, Adam 2e-4, batch 32, penalty 0.1, training dose 1.0, 15 epochs, 3 seeds; epoch and dose chosen on the 28 dev records by dev
gain; judged on the same test records with the same recipes. For every run the top-1 on the TRAINING prompts is printed next to the test top-1 (the overfitting check). The full set is judged
on rewordings, neighbours and unrelated prompts and against the baseline (real only), with Gate 9. Saves outputs/transfer/d13_training.json, the maps (d13_map_*.pt) and the log.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.sae_utils import load_sae_for_layer
from src.transfer.models import load_model, pick_device
from src.transfer.om_batched import eval_ranks, pack, score, train_map_hinge
from src.transfer.om_judge import judge_map, pack_judge, per_record_gain, summarize
from src.transfer.output_matching import build_item
from src.transfer.runlog import tee_to

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
Q = 16
DOSES = [0.5, 1.0, 1.5, 2.0, 3.0]
FRACTIONS = [0.0, 0.25, 0.5, 1.0]
GATE_TOP1, GATE_LIFT, GATE_SPEC, READ_POINTS = 0.50, 0.80, 0.10, 0.05


def pct(x):
    return "  n/a" if x != x else f"{100 * x:3.0f}%"


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_synthetic():
    path = os.path.join(OUT_DIR, "d13_edits.pt")
    if os.path.exists(path):
        merged = torch.load(path, weights_only=False)
    else:
        merged = {}
        for p in sorted(glob.glob(os.path.join(OUT_DIR, "d13_edits_shard[0-9].pt"))):
            merged.update(torch.load(p, weights_only=False))
        print(f"   (d13_edits.pt not found: using the {len(merged)} edits of the shard files)", flush=True)
    return {k: v for k, v in merged.items() if v.get("rank") == 1}


def build_syn_items(ed, small, medium, sae, pool_by_case, dev):
    path = os.path.join(OUT_DIR, "d13_items.pt")
    cache = torch.load(path, weights_only=False) if os.path.exists(path) else {}
    todo = [k for k in ed if k not in cache]
    print(f"   {len(ed)} synthetic edits reached rank 1 in small; {len(cache)} already built, {len(todo)} to build (5 texts each)", flush=True)
    t0 = time.time()
    for n, k in enumerate(todo):
        r = pool_by_case[ed[k]["case_id"]]
        recipe = {"fids": ed[k]["fids"], "a": ed[k]["a"]}
        items = []
        for text in [r["prompt"]] + r["paraphrases"][:2] + r["neighbours"][:2]:
            it = build_item(small, medium, sae, text, recipe, Q, dev)
            items.append({"tok": it["tok"][0].cpu(), "dS": it["dS"].cpu(), "hm": it["hm"][0].cpu(), "goal": it["goal"].cpu()})
        cache[k] = items
        if (n + 1) % 100 == 0 or n + 1 == len(todo):
            torch.save(cache, path)
            print(f"   built {n + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    return {k: cache[k] for k in ed}


def main():
    log_path = tee_to("22_varied_training")
    print(f"log: {os.path.relpath(log_path, ROOT)}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    fractions, epochs, seeds, tag = FRACTIONS, a.epochs, a.seeds, ""
    if a.smoke:
        fractions, epochs, seeds, tag = [0.0, 1.0], 2, 1, "_smoke"
    dev = pick_device(a.device)
    t0 = time.time()
    c = torch.load(os.path.join(OUT_DIR, "d8_items.pt"), weights_only=False)
    bench = json.load(open(os.path.join(OUT_DIR, "d_benchmark.json"), encoding="utf8"))
    edits = torch.load(os.path.join(OUT_DIR, "d_edits.pt"), weights_only=False)
    bench = [r for r in bench if r["case_id"] in edits]
    test = [r for r in bench if r["split"] == "test"]
    if a.smoke:
        test = test[:12]
    tedits = torch.load(os.path.join(OUT_DIR, "d7_train_edits.pt"), weights_only=False)
    pool = json.load(open(os.path.join(OUT_DIR, "d7_pool.json"), encoding="utf8"))
    held_words, held_subjects = {r["tid_new"] for r in bench}, {r["subject"] for r in bench}
    keep = [r for r in pool if r["tid_new"] not in held_words and r["subject"] not in held_subjects][:800]
    good = [r for r in keep if tedits[r["case_id"]]["rank"] == 1]
    assert len(good) == c["n_good"], "real training items do not match the records (rebuild d8_items.pt with 14_datasize_check.py)"
    real_tids = [good[i // 5]["tid_new"] for i in range(len(c["train"]))]
    maps = torch.load(os.path.join(OUT_DIR, "maps_gpt2_to_gpt2-medium.pt"))["maps"]
    W0 = maps[f"s2m_L8_L{Q}"]["W"].to(dev)
    d7 = json.load(open(os.path.join(OUT_DIR, "d7_output_matching.json"), encoding="utf8"))["summary"]
    small, medium, sae = load_model("gpt2", dev), load_model("gpt2-medium", dev), load_sae_for_layer(8, dev)
    for m in (small, medium):
        for p_ in m.parameters():
            p_.requires_grad_(False)

    stage("1/5  synthetic-target edits (from 21_varied_edits.py) and their training texts")
    ed = load_synthetic()
    if a.smoke:
        ed = dict(list(ed.items())[:40])
    pool_by_case = {r["case_id"]: r for r in pool}
    syn_cache = build_syn_items(ed, small, medium, sae, pool_by_case, dev)
    syn_keys = sorted(syn_cache)
    print(f"   {len(syn_keys)} synthetic edits with {len({ed[k]['tid'] for k in syn_keys})} distinct target words (the real set has {len({g['tid_new'] for g in good})})", flush=True)
    if not syn_keys:
        print("   no synthetic edits yet: run 21_varied_edits.py first", flush=True)
        return 1
    small.cpu()                                              # not needed after this point
    del small, sae
    torch.cuda.empty_cache()

    # ---------------- evaluation sets ---------------------------------------------------------------------------------------------------
    jitems = torch.load(os.path.join(OUT_DIR, "d10_judge_items.pt"), weights_only=False)
    if a.smoke:
        jitems = [it for it in jitems if it["rec"] < len(test)]
    T0 = pack_judge([it for it in jitems if it["kind"] == 0], dev)
    rec_primary = torch.tensor([edits[r["case_id"]]["real"]["rank"] == 1 for r in test])
    rec_all = torch.ones(len(test), dtype=torch.bool)
    arm0 = T0["arm"].cpu()
    sel = {(a_, p_): torch.nonzero((arm0 == a_) & (m_[T0["rec"].cpu()])).flatten().to(dev) for a_ in (0, 1, 2) for p_, m_ in (("prim", rec_primary), ("all", rec_all))}
    devs = pack(c["dev"], dev)
    real_tuned = [dict(c["train"][i], tid=real_tids[i], base_rank=1) for i in range(0, len(c["train"]), 5)]
    for it in real_tuned:
        it.pop("goal", None)
    S_real_tuned = pack(real_tuned, dev)
    syn_tuned = [dict(syn_cache[k][0], tid=ed[k]["tid"], base_rank=1) for k in syn_keys]
    for it in syn_tuned:
        it.pop("goal", None)
    S_syn_tuned = pack(syn_tuned, dev)
    print(f"   {len(c['dev'])} dev records, {len(test)} test records ({int(rec_primary.sum())} primary)", flush=True)

    # ---------------- sweep ---------------------------------------------------------------------------------------------------------------
    stage(f"2/5  sweep: training sets {[f'real + {int(100 * f)}% of the synthetic edits' if f else 'real only' for f in fractions]} x {seeds} seeds, {epochs} epochs each")
    runs, maps_out = [], {}
    for f in fractions:
        for seed in range(seeds):
            ts = time.time()
            chosen = [] if f == 0 else sorted(random.Random(1000 * seed + int(100 * f)).sample(range(len(syn_keys)), max(1, round(f * len(syn_keys)))))
            items, tids, tuned = list(c["train"]), list(real_tids), [i % 5 == 0 for i in range(len(c["train"]))]
            for j in chosen:
                for t_, it in enumerate(syn_cache[syn_keys[j]]):
                    items.append(it)
                    tids.append(ed[syn_keys[j]]["tid"])
                    tuned.append(t_ == 0)
            S = pack(items, dev, with_goal=True)
            S["tid"], S["tuned"] = torch.tensor(tids, device=dev), torch.tensor(tuned, device=dev)
            W, ep, hist = train_map_hinge(medium, W0, S, devs, Q, lam=0.0, epochs=epochs, seed=seed)
            del S
            torch.cuda.empty_cache()
            rows = []
            for d in DOSES:
                dv = score(devs, eval_ranks(medium, devs, W, d, Q))
                rk = {a_: eval_ranks(medium, T0, W, d, Q) for a_ in (0, 1, 2)}
                row = {"dose": d, "dev": dv}
                for a_, name in ((0, "real"), (1, "wrong"), (2, "rand")):
                    row[name] = {"prim": score(T0, rk[a_], sel[(a_, "prim")]) if len(sel[(a_, "prim")]) else (float("nan"),) * 3,
                                 "all": score(T0, rk[a_], sel[(a_, "all")]) if len(sel[(a_, "all")]) else (float("nan"),) * 3}
                rows.append(row)
            best = max(rows, key=lambda r: (r["dev"][1], -r["dose"]))
            tr_real = float((eval_ranks(medium, S_real_tuned, W, best["dose"], Q) == 1).float().mean())
            tr_syn = float((eval_ranks(medium, S_syn_tuned, W, best["dose"], Q)[torch.tensor(chosen, device=dev)] == 1).float().mean()) if chosen else float("nan")
            runs.append({"fraction": f, "seed": seed, "n_syn_edits": len(chosen), "best_epoch": ep, "dose": best["dose"], "dev_score": best["dev"][1], "rows": rows, "history": hist,
                         "train_top1_real": tr_real, "train_top1_syn": tr_syn})
            maps_out[(f, seed)] = (W.cpu(), best["dose"])
            if not a.smoke:
                torch.save({"W": W.cpu(), "dose": best["dose"], "epoch": ep}, os.path.join(OUT_DIR, f"d13_map_f{f}_s{seed}.pt"))
            r_, w_, n_ = best["real"]["prim"], best["wrong"]["prim"], best["rand"]["prim"]
            print(f"   real + {int(100 * f):>3}% synthetic ({len(chosen)} edits), seed {seed}: epoch {ep:>2}, dose {best['dose']}, dev gain {best['dev'][1]:.2f} | TEST (primary): top-1 {pct(r_[0])}, gain {r_[1]:.2f}, "
                  f"median rank {r_[2]:.0f} | controls: wrong {pct(w_[0])}, random-word {pct(n_[0])} | TRAINING top-1: real prompts {pct(tr_real)}, synthetic {pct(tr_syn)}   [{time.time() - ts:.0f} s]", flush=True)

    # ---------------- summary and judging -------------------------------------------------------------------------------------------------
    stage("3/5  data curve (mean over seeds; test = primary records)")
    print(f"   {'training set':<28}{'dev score':>11}{'test top-1 (range)':>26}{'test gain':>11}{'med rank':>10}{'wrong':>8}{'random word':>13}{'train top-1 (real)':>20}")
    summ = {}
    pick = lambda r: [x for x in r["rows"] if x["dose"] == r["dose"]][0]
    for f in fractions:
        rs = [r for r in runs if r["fraction"] == f]
        t = [pick(r)["real"]["prim"][0] for r in rs]
        summ[f] = {"n_syn": mean([r["n_syn_edits"] for r in rs]), "dev_score": mean([r["dev_score"] for r in rs]), "top1": mean(t), "top1_min": min(t), "top1_max": max(t), "gain": mean([pick(r)["real"]["prim"][1] for r in rs]),
                   "median_rank": mean([pick(r)["real"]["prim"][2] for r in rs]), "wrong_top1": mean([pick(r)["wrong"]["prim"][0] for r in rs]), "rand_top1": mean([pick(r)["rand"]["prim"][0] for r in rs]),
                   "train_top1_real": mean([r["train_top1_real"] for r in rs]), "train_top1_syn": mean([r["train_top1_syn"] for r in rs])}
        s_ = summ[f]
        print(f"   real + {int(100 * f):>3}% synthetic ({int(s_['n_syn']):>4}){s_['dev_score']:>11.2f}{pct(s_['top1']):>14} ({pct(s_['top1_min'])} to {pct(s_['top1_max'])}){s_['gain']:>11.2f}{s_['median_rank']:>10.1f}{pct(s_['wrong_top1']):>8}{pct(s_['rand_top1']):>13}{pct(s_['train_top1_real']):>20}")
    full, base = max(fractions), 0.0
    stage(f"4/5  judging the full set (real + {int(100 * full)}% synthetic) and the baseline (real only) on rewordings, neighbours and unrelated prompts")
    del S_real_tuned, S_syn_tuned
    torch.cuda.empty_cache()
    J = pack_judge(jitems, dev)
    judged = {}
    for f in (base, full):
        for seed in range(seeds):
            W, dose = maps_out[(f, seed)]
            res = judge_map(medium, J, W.to(dev), dose, Q)
            judged[(f, seed)] = {"prim": summarize(J, res, rec_primary), "all": summarize(J, res, rec_all), "gain_rec": per_record_gain(J, res, 0)}
            print(f"   fraction {f}, seed {seed} judged ({time.time() - t0:.0f} s)", flush=True)

    def avg(f, subset, arm, key):
        return mean([judged[(f, s)][subset][arm][key] for s in range(seeds) if arm in judged[(f, s)][subset]])

    ref = {k: d7["ok"]["stage2"][k] for k in ("medium unchanged", "small unchanged", "small with the edit")}
    refa = {k: d7["all"]["stage2"][k] for k in ("medium unchanged", "small unchanged", "small with the edit")}
    print("\n" + "=" * 130)
    print("RESULT  step D13 (a map trained on many more, and more varied, edits)" + ("  [smoke: numbers mean nothing]" if a.smoke else ""))
    for subset, label, rf in (("prim", "PRIMARY: 88 test records whose original edit reached rank 1 in small", ref), ("all", "ALL test records", refa)):
        print(f"\n  ---- {label} (mean over seeds) ----")
        print(f"    {'arm':<62}{'top-1':>7}{'med rank':>10}{'PS':>7}{'NS':>7}{'KL':>8}{'flips':>8}")
        for k in ("medium unchanged", "small unchanged", "small with the edit"):
            print(f"    {k:<62}{'':>7}{'':>10}{pct(rf[k]['PS']):>7}{pct(rf[k]['NS']):>7}{rf[k]['KL']:>8.3f}{pct(rf[k]['flips']):>8}")
        for f in (base, full):
            tag_f = "real only (baseline)" if f == base else f"real + {int(100 * f)}% synthetic"
            for arm, nm in (("real", "real recipe"), ("wrong", "another record's recipe"), ("rand", "random-word edit (own word)")):
                kl = avg(f, subset, arm, "unrelated_KL") if arm == "real" else float("nan")
                fl = avg(f, subset, arm, "unrelated_flips") if arm == "real" else float("nan")
                print(f"    {tag_f + ': ' + nm:<62}{pct(avg(f, subset, arm, 'top1')):>7}{avg(f, subset, arm, 'median_rank'):>10.0f}{pct(avg(f, subset, arm, 'PS')):>7}{pct(avg(f, subset, arm, 'NS')):>7}{kl:>8.3f}{pct(fl):>8}")
    lift_src = ref["small with the edit"]["PS"] - ref["small unchanged"]["PS"]
    lift = avg(full, "prim", "real", "PS") - ref["medium unchanged"]["PS"]
    ratio = lift / lift_src if lift_src > 0 else float("nan")
    ns_fall = ref["medium unchanged"]["NS"] - avg(full, "prim", "real", "NS")
    ns_src = ref["small unchanged"]["NS"] - ref["small with the edit"]["NS"]
    top1, rand1 = avg(full, "prim", "real", "top1"), avg(full, "prim", "rand", "top1")
    ga, gb, gc, gd = top1 >= GATE_TOP1, ratio >= GATE_LIFT, ns_fall <= ns_src, (top1 - rand1) >= GATE_SPEC
    gap_base = summ[base]["train_top1_real"] - summ[base]["top1"]
    gap_full = summ[full]["train_top1_real"] - summ[full]["top1"]
    rise = summ[full]["top1"] - summ[base]["top1"]
    rand_rise = summ[full]["rand_top1"] - summ[base]["rand_top1"]
    if ga and gb and gc and gd:
        reading = "all four hold: a map trained on varied edits carries unseen real edits into medium"
    elif rise >= READ_POINTS and gap_full < gap_base:
        reading = "(a) fails but the test top-1 rises by at least 5 points and the training-test gap shrinks: variety helps but not enough (next: a regularised neural map on this set)"
    elif rise < READ_POINTS:
        reading = "(a) fails and the test top-1 stays within 5 points of the baseline: the variety of target words was not the limit (the limit is the edit or the map: amortised D6, or stop)"
    else:
        reading = "(a) fails; the test top-1 rises by at least 5 points but the training-test gap does not shrink: see the numbers"
    if rand_rise >= max(rise, 0.0) and rise > 0:
        reading += "; NOTE: the random-word edit rises as much as the real target: the map has learned to push any word"
    print(f"\n  GATE 9 (full set, 88 primary records, mean over {seeds} seeds): (a) top-1 {pct(top1)} >= {GATE_TOP1:.0%}: {'ok' if ga else 'NO'};  (b) rewordings lift = {ratio:.2f} of the edit's own lift in small (>= {GATE_LIFT}): {'ok' if gb else 'NO'};  "
          f"(c) neighbours fall {100 * ns_fall:.0f} points, the edit itself in small {100 * ns_src:.1f}: {'ok' if gc else 'NO'};  (d) real minus random-word top-1 = {100 * (top1 - rand1):.0f} points (>= {int(100 * GATE_SPEC)}): {'ok' if gd else 'NO'}   ->  {'PASS' if (ga and gb and gc and gd) else 'FAIL'}")
    print(f"  overfitting check (top-1 on the real training prompts minus top-1 on unseen test records): baseline {100 * gap_base:.0f} points, full set {100 * gap_full:.0f} points")
    print(f"  reading (fixed in the plan before the run): {reading}")
    p_val = float("nan")
    try:
        from scipy.stats import wilcoxon
        g_full = torch.stack([judged[(full, s)]["gain_rec"] for s in range(seeds)]).mean(0)
        g_base = torch.stack([judged[(base, s)]["gain_rec"] for s in range(seeds)]).mean(0)
        ok = rec_primary & ~torch.isnan(g_full) & ~torch.isnan(g_base)
        if ok.sum() > 5:
            p_val = float(wilcoxon(g_full[ok].numpy(), g_base[ok].numpy(), alternative="greater").pvalue)
    except Exception as ex:
        print(f"  (paired test not run: {ex})")
    print(f"  improvement of the full set over the baseline in the same sweep: top-1 {pct(summ[base]['top1'])} -> {pct(summ[full]['top1'])}, paired Wilcoxon on the per-record log-rank gain (88 records, seed-averaged) p = {p_val:.1e}")
    out = {"fractions": fractions, "n_syn_available": len(syn_keys), "n_syn_distinct_words": len({ed[k]["tid"] for k in syn_keys}), "runs": runs, "summary": {str(k): v for k, v in summ.items()},
           "gate9": {"top1": top1, "lift_ratio": ratio, "ns_fall": ns_fall, "ns_fall_source": ns_src, "rand_top1": rand1, "a": bool(ga), "b": bool(gb), "c": bool(gc), "d": bool(gd), "reading": reading,
                     "gap_base": gap_base, "gap_full": gap_full, "paired_p_vs_baseline": p_val},
           "judged": {f"{k[0]}_{k[1]}": {"prim": v["prim"], "all": v["all"]} for k, v in judged.items()}}
    path = os.path.join(OUT_DIR, f"d13_training{tag}.json")
    json.dump(out, open(path, "w", encoding="utf8"), indent=1, default=float)
    stage(f"5/5  done in {time.time() - t0:.0f} s; saved {os.path.relpath(path, ROOT)}, the maps (d13_map_*.pt) and the log {os.path.relpath(log_path, ROOT)}")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
