"""
Step D10 of docs/cross_model_transfer/PLAN.md: a rank-1 term in the training loss of the map. The design, the lambda grid, the selection rule, Gate 7 and the
readings are written in the plan BEFORE this tool was built; this tool follows them.

    .venv\\Scripts\\python.exe tools/transfer/16_rank1_term.py --smoke     # one-minute rehearsal (numbers mean nothing)
    .venv\\Scripts\\python.exe tools/transfer/16_rank1_term.py             # the real run: about 30 minutes, progress printed all the time

What it does (all as in the plan):
  1. Training data: the same 289 edits and 1,445 texts as D7 to D9b (cache d8_items.pt from 14_datasize_check.py; the first run of that tool builds it).
  2. Judge items for the 150 test records (the original small-tuned recipe, another record's recipe and the random-word edit, on the tuned prompt, rewordings,
     neighbours and unrelated prompts); built once (about 3 minutes) and cached in d10_judge_items.pt.
  3. Sweep: lambda in {0, 0.01, 0.03, 0.1, 0.3, 1.0} x seeds {0, 1, 2}: train 15 epochs with the D7 loss plus lambda * rank-1 hinge on the tuned prompts; epoch by dev
     gain at dose 1; dose by dev gain over {0.5, 1, 1.5, 2, 3}; the run's dev score is the dev gain at that dose. Test result and the two controls printed per run.
  3b. lambda = 0 must reproduce D8/D9b (top-1 about 20%): it is the same-run baseline and the replication check.
  4. lambda-star = the lambda with the best mean dev score over the 3 seeds (ties: smaller lambda).
  5. Judging of lambda-star and lambda = 0 (3 seeds each): rewordings, neighbours, unrelated prompts, controls; Gate 7 (a) to (d) and its fixed reading.
Everything is saved: outputs/transfer/d10_rank1_term.json (per-epoch curves, per-dose tables, controls, judging), the trained maps (d10_map_*.pt) and the log.
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

from src.transfer.models import load_model, pick_device
from src.transfer.om_batched import eval_ranks, pack, score, train_map_hinge
from src.transfer.om_judge import build_judge_items, judge_map, pack_judge, per_record_gain, summarize
from src.transfer.runlog import tee_to

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
Q = 16
DOSES = [0.5, 1.0, 1.5, 2.0, 3.0]
LAMBDAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
GATE_TOP1, GATE_LIFT, GATE_SPEC = 0.50, 0.80, 0.10


def pct(x):
    return "  n/a" if x != x else f"{100 * x:3.0f}%"


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs():
    c = torch.load(os.path.join(OUT_DIR, "d8_items.pt"), weights_only=False)
    bench = json.load(open(os.path.join(OUT_DIR, "d_benchmark.json"), encoding="utf8"))
    edits = torch.load(os.path.join(OUT_DIR, "d_edits.pt"), weights_only=False)
    bench = [r for r in bench if r["case_id"] in edits]
    tedits = torch.load(os.path.join(OUT_DIR, "d7_train_edits.pt"), weights_only=False)
    pool = json.load(open(os.path.join(OUT_DIR, "d7_pool.json"), encoding="utf8"))
    held_words, held_subjects = {r["tid_new"] for r in bench}, {r["subject"] for r in bench}
    keep = [r for r in pool if r["tid_new"] not in held_words and r["subject"] not in held_subjects][:800]
    good = [r for r in keep if tedits[r["case_id"]]["rank"] == 1]
    assert len(good) == c["n_good"] and all(c["rec_of"][i] == i // 5 for i in range(len(c["rec_of"]))), "training items do not match the records (rebuild d8_items.pt)"
    return c, bench, edits, good


def main():
    log_path = tee_to("16_rank1_term")
    print(f"log: {os.path.relpath(log_path, ROOT)}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    lambdas, epochs, seeds, tag = LAMBDAS, a.epochs, a.seeds, ""
    if a.smoke:
        lambdas, epochs, seeds, tag = [0.0, 0.3], 2, 1, "_smoke"
    dev = pick_device(a.device)
    t0 = time.time()
    c, bench, edits, good = load_inputs()
    test = [r for r in bench if r["split"] == "test"]
    if a.smoke:
        test = test[:12]
    maps = torch.load(os.path.join(OUT_DIR, "maps_gpt2_to_gpt2-medium.pt"))["maps"]
    W0 = maps[f"s2m_L8_L{Q}"]["W"].to(dev)
    d7 = json.load(open(os.path.join(OUT_DIR, "d7_output_matching.json"), encoding="utf8"))["summary"]
    medium = load_model("gpt2-medium", dev)
    for p_ in medium.parameters():
        p_.requires_grad_(False)

    # ---------------- judge items (built once) -------------------------------------------------------------------------------------
    stage("1/5  judge items for the test records (another record's recipe and the random-word edit as controls; cached)")
    jpath = os.path.join(OUT_DIR, f"d10_judge_items{tag}.pt")
    if os.path.exists(jpath):
        items = torch.load(jpath, weights_only=False)
        print(f"   loaded {len(items)} items from {os.path.relpath(jpath, ROOT)}", flush=True)
    else:
        from src.sae_utils import load_sae_for_layer
        small, sae = load_model("gpt2", dev), load_sae_for_layer(8, dev)
        items = build_judge_items(small, medium, sae, test, edits, Q, dev, log=lambda i, n: print(f"   {i}/{n} test records ({time.time() - t0:.0f} s)", flush=True))
        torch.save(items, jpath)
        del small, sae
        print(f"   built {len(items)} items ({time.time() - t0:.0f} s)", flush=True)
    kind0 = [it for it in items if it["kind"] == 0]
    T0 = pack_judge(kind0, dev)
    rec_primary = torch.tensor([edits[r["case_id"]]["real"]["rank"] == 1 for r in test])
    rec_all = torch.ones(len(test), dtype=torch.bool)
    arm0 = T0["arm"].cpu()
    sel = {(a_, p_): torch.nonzero((arm0 == a_) & (rec_mask[T0["rec"].cpu()])).flatten().to(dev) for a_ in (0, 1, 2) for p_, rec_mask in (("prim", rec_primary), ("all", rec_all))}

    # ---------------- training data and dev set ---------------------------------------------------------------------------------------
    train = pack(c["train"], dev, with_goal=True)
    train["tid"] = torch.tensor([good[i // 5]["tid_new"] for i in range(len(c["train"]))], device=dev)
    train["tuned"] = torch.tensor([i % 5 == 0 for i in range(len(c["train"]))], device=dev)
    devs = pack(c["dev"], dev)
    print(f"{len(good)} training records ({len(c['train'])} texts, {int(train['tuned'].sum())} tuned prompts), {len(c['dev'])} dev records, {len(test)} test records ({int(rec_primary.sum())} primary)", flush=True)

    # ---------------- sweep ----------------------------------------------------------------------------------------------------------
    stage(f"2/5  sweep: lambda {lambdas} x {seeds} seeds, {epochs} epochs each (dose and epoch chosen on the 28 dev records by dev gain)")
    runs, maps_out = [], {}
    for lam in lambdas:
        for seed in range(seeds):
            ts = time.time()
            W, ep, hist = train_map_hinge(medium, W0, train, devs, Q, lam=lam, epochs=epochs, seed=seed)
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
            runs.append({"lambda": lam, "seed": seed, "best_epoch": ep, "dose": best["dose"], "dev_score": best["dev"][1], "rows": rows, "history": hist})
            maps_out[(lam, seed)] = (W.cpu(), best["dose"])
            if not a.smoke:
                torch.save({"W": W.cpu(), "dose": best["dose"], "epoch": ep}, os.path.join(OUT_DIR, f"d10_map_lam{lam}_s{seed}.pt"))
            r_, w_, n_ = best["real"]["prim"], best["wrong"]["prim"], best["rand"]["prim"]
            print(f"   lambda {lam:<5} seed {seed}: epoch {ep:>2}, dose {best['dose']}, dev gain {best['dev'][1]:.2f} | test (primary records): top-1 {pct(r_[0])}, gain {r_[1]:.2f}, median rank {r_[2]:.0f} | "
                  f"controls: wrong recipe {pct(w_[0])}, random-word {pct(n_[0])}   [{time.time() - ts:.0f} s]", flush=True)

    # ---------------- sweep table, lambda-star -----------------------------------------------------------------------------------------
    stage("3/5  sweep summary (mean over seeds; test = primary records, read only, descriptive except lambda-star which is chosen by the dev score)")
    print(f"   {'lambda':<8}{'dev score':>11}{'test top-1 (range)':>26}{'test gain':>11}{'median rank':>13}{'wrong recipe':>14}{'random word':>13}")
    summ = {}
    for lam in lambdas:
        rs = [r for r in runs if r["lambda"] == lam]
        pick = lambda r: [x for x in r["rows"] if x["dose"] == r["dose"]][0]
        t = [pick(r)["real"]["prim"][0] for r in rs]
        summ[lam] = {"dev_score": mean([r["dev_score"] for r in rs]), "top1": mean(t), "top1_min": min(t), "top1_max": max(t),
                     "gain": mean([pick(r)["real"]["prim"][1] for r in rs]), "median_rank": mean([pick(r)["real"]["prim"][2] for r in rs]),
                     "wrong_top1": mean([pick(r)["wrong"]["prim"][0] for r in rs]), "rand_top1": mean([pick(r)["rand"]["prim"][0] for r in rs])}
        s_ = summ[lam]
        print(f"   {lam:<8}{s_['dev_score']:>11.2f}{pct(s_['top1']):>14} ({pct(s_['top1_min'])} to {pct(s_['top1_max'])}){s_['gain']:>11.2f}{s_['median_rank']:>13.1f}{pct(s_['wrong_top1']):>14}{pct(s_['rand_top1']):>13}")
    star = max(lambdas, key=lambda l: (round(summ[l]["dev_score"], 6), -l))
    print(f"\n   replication check, lambda 0 against D8/D9b (top-1 about 20%, gain about 2.66): top-1 {pct(summ[0.0]['top1'])}, gain {summ[0.0]['gain']:.2f}")
    print(f"   lambda-star (best mean dev score, ties to the smaller lambda) = {star}", flush=True)

    # ---------------- judging ----------------------------------------------------------------------------------------------------------
    stage(f"4/5  judging lambda-star ({star}) and lambda 0 on rewordings, neighbours and unrelated prompts (3 seeds each)")
    del train
    torch.cuda.empty_cache()
    J = pack_judge(items, dev)
    judged = {}
    for lam in sorted({star, 0.0}):
        for seed in range(seeds):
            W, dose = maps_out[(lam, seed)]
            res = judge_map(medium, J, W.to(dev), dose, Q)
            judged[(lam, seed)] = {"prim": summarize(J, res, rec_primary), "all": summarize(J, res, rec_all), "gain_rec": per_record_gain(J, res, 0)}
            print(f"   lambda {lam}, seed {seed} judged ({time.time() - t0:.0f} s)", flush=True)

    def avg(lam, subset, arm, key):
        return mean([judged[(lam, s)][subset][arm][key] for s in range(seeds) if arm in judged[(lam, s)][subset]])

    ref = {k: d7["ok"]["stage2"][k] for k in ("medium unchanged", "small unchanged", "small with the edit")}
    refa = {k: d7["all"]["stage2"][k] for k in ("medium unchanged", "small unchanged", "small with the edit")}
    print("\n" + "=" * 130)
    print("RESULT  step D10 (a rank-1 term in the training loss of the map)" + ("  [smoke: numbers mean nothing]" if a.smoke else ""))
    for subset, label, rf in (("prim", "PRIMARY: 88 test records whose original edit reached rank 1 in small", ref), ("all", "ALL test records", refa)):
        print(f"\n  ---- {label} (mean over 3 seeds) ----")
        print(f"    {'arm':<58}{'top-1':>7}{'med rank':>10}{'PS':>7}{'NS':>7}{'KL':>8}{'flips':>8}")
        for k in ("medium unchanged", "small unchanged", "small with the edit"):
            print(f"    {k:<58}{'':>7}{'':>10}{pct(rf[k]['PS']):>7}{pct(rf[k]['NS']):>7}{rf[k]['KL']:>8.3f}{pct(rf[k]['flips']):>8}")
        for lam in sorted({0.0, star}):
            tag_l = "lambda 0 (D7 baseline)" if lam == 0.0 else f"lambda {lam} (lambda-star)"
            for arm, nm in (("real", "real recipe"), ("wrong", "another record's recipe"), ("rand", "random-word edit (own word)")):
                kl = avg(lam, subset, arm, "unrelated_KL") if arm == "real" else float("nan")
                fl = avg(lam, subset, arm, "unrelated_flips") if arm == "real" else float("nan")
                print(f"    {tag_l + ': ' + nm:<58}{pct(avg(lam, subset, arm, 'top1')):>7}{avg(lam, subset, arm, 'median_rank'):>10.0f}{pct(avg(lam, subset, arm, 'PS')):>7}{pct(avg(lam, subset, arm, 'NS')):>7}"
                      f"{kl:>8.3f}{pct(fl):>8}")
    lift_src = ref["small with the edit"]["PS"] - ref["small unchanged"]["PS"]
    lift = avg(star, "prim", "real", "PS") - ref["medium unchanged"]["PS"]
    ns_fall = ref["medium unchanged"]["NS"] - avg(star, "prim", "real", "NS")
    ns_src = ref["small unchanged"]["NS"] - ref["small with the edit"]["NS"]
    top1, rand1 = avg(star, "prim", "real", "top1"), avg(star, "prim", "rand", "top1")
    ga, gb, gc, gd = top1 >= GATE_TOP1, (lift / lift_src if lift_src > 0 else float("nan")) >= GATE_LIFT, ns_fall <= ns_src, (top1 - rand1) >= GATE_SPEC
    ratio = lift / lift_src if lift_src > 0 else float("nan")
    if ga and gb and gc and gd:
        reading = "all four hold: one trained map carries edits it has never seen into medium with no tuning on medium and is specific to the target"
    elif ga and not gd:
        reading = "(a) holds but (d) fails: the map has become a steering mechanism (it pushes whatever word the edit points at)"
    elif ga:
        reading = "(a) and (d) hold, (b) or (c) fails: a push on the tuned prompt that does not generalise"
    else:
        reading = "(a) fails: a rank-1 term does not rescue the export with this map and data"
    print(f"\n  GATE 7 (lambda-star = {star}, 88 primary records, mean over 3 seeds): (a) top-1 {pct(top1)} >= {GATE_TOP1:.0%}: {'ok' if ga else 'NO'};  (b) rewordings lift = {ratio:.2f} of the edit's own lift in small "
          f"(>= {GATE_LIFT}): {'ok' if gb else 'NO'};  (c) neighbours fall {100 * ns_fall:.0f} points, the edit itself in small {100 * ns_src:.1f}: {'ok' if gc else 'NO'};  "
          f"(d) real minus random-word top-1 = {100 * (top1 - rand1):.0f} points (>= {int(100 * GATE_SPEC)}): {'ok' if gd else 'NO'}   ->  {'PASS' if (ga and gb and gc and gd) else 'FAIL'}")
    print(f"  reading (fixed in the plan before the run): {reading}")
    p_val = float("nan")
    try:
        from scipy.stats import wilcoxon
        g_star = torch.stack([judged[(star, s)]["gain_rec"] for s in range(seeds)]).mean(0)
        g_zero = torch.stack([judged[(0.0, s)]["gain_rec"] for s in range(seeds)]).mean(0)
        ok = rec_primary & ~torch.isnan(g_star) & ~torch.isnan(g_zero)
        if star != 0.0 and ok.sum() > 5:
            p_val = float(wilcoxon(g_star[ok].numpy(), g_zero[ok].numpy(), alternative="greater").pvalue)
    except Exception as ex:
        print(f"  (paired test not run: {ex})")
    print(f"  improvement of lambda-star over lambda 0 in the same sweep: top-1 {pct(summ[0.0]['top1'])} -> {pct(summ[star]['top1'])}, paired Wilcoxon on the per-record log-rank gain (88 records, seed-averaged) p = {p_val:.1e}")
    out = {"lambdas": lambdas, "star": star, "runs": [{k: v for k, v in r.items()} for r in runs], "summary": {str(k): v for k, v in summ.items()},
           "gate7": {"top1": top1, "lift_ratio": ratio, "ns_fall": ns_fall, "ns_fall_source": ns_src, "rand_top1": rand1, "a": bool(ga), "b": bool(gb), "c": bool(gc), "d": bool(gd), "reading": reading, "paired_p_vs_lambda0": p_val},
           "judged": {f"{k[0]}_{k[1]}": {"prim": v["prim"], "all": v["all"]} for k, v in judged.items()}}
    path = os.path.join(OUT_DIR, f"d10_rank1_term{tag}.json")
    json.dump(out, open(path, "w", encoding="utf8"), indent=1, default=float)
    stage(f"5/5  done in {time.time() - t0:.0f} s; saved {os.path.relpath(path, ROOT)}, the maps (d10_map_*.pt) and the log {os.path.relpath(log_path, ROOT)}")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
