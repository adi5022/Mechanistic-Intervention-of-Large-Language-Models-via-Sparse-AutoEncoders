"""
Step D14 of docs/cross_model_transfer/PLAN.md (chain step B): a map that cannot memorise as easily, trained on the D13 full set (289 real edits + the synthetic-target edits). The design, the four
arms, the arm choice, Gate 9 (unchanged) and the readings are in the plan, written before this tool was built.

    .venv\\Scripts\\python.exe tools/transfer/25_regularised_map.py --smoke     # rehearsal (1 epoch, 1 seed, 12 test records, 100 real texts + 20 synthetic edits; numbers mean nothing)
    .venv\\Scripts\\python.exe tools/transfer/25_regularised_map.py             # the real sweep: 4 arms x 2 seeds, resumable, progress printed all the time

Needs the outputs of 21_varied_edits.py and 22_varied_training.py (d13_edits.pt, d13_items.pt, and d13_training.json for the comparison). Saves outputs/transfer/d14_regularised.json, d14_runs.json
(every finished run, used to resume), the maps (d14_map_*.pt) and the log.
"""
import argparse
import copy
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.transfer.models import load_model, pick_device
from src.transfer.om_batched import eval_ranks, pack, score, train_map_hinge
from src.transfer.om_judge import judge_map, pack_judge, summarize
from src.transfer.reg_maps import LowRankMap, ResidualMlpDelta, identity, train_module, translated, typical_sizes
from src.transfer.runlog import tee_to

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
Q = 16
DOSES = [0.5, 1.0, 1.5, 2.0, 3.0]
GATE_TOP1, GATE_LIFT, GATE_SPEC, READ_POINTS = 0.50, 0.80, 0.10, 0.05
ARMS = {"R1": "linear, penalty 1.0", "R2": "linear, penalty 10", "LR": "low-rank nudge (rank 16)", "NN": "small neural map (256, dropout 0.2)"}


def pct(x):
    return "  n/a" if x != x else f"{100 * x:3.0f}%"


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare(S, spec, I):
    """(packed set, map matrix) ready for the evaluation functions: a plain matrix is used as it is; a module's translated change is computed once and the map is the identity."""
    if torch.is_tensor(spec):
        return S, spec
    return translated(S, spec), I


def main():
    log_path = tee_to("25_regularised_map")
    print(f"log: {os.path.relpath(log_path, ROOT)}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore the runs already finished")
    a = ap.parse_args()
    epochs, seeds, tag = a.epochs, a.seeds, ""
    doses = DOSES
    if a.smoke:
        epochs, seeds, tag, doses = 1, 1, "_smoke", [1.0, 2.0]
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
    w0n = (W0 ** 2).sum()
    d7 = json.load(open(os.path.join(OUT_DIR, "d7_output_matching.json"), encoding="utf8"))["summary"]
    d13_path = os.path.join(OUT_DIR, f"d13_training{tag}.json")
    d13 = json.load(open(d13_path, encoding="utf8"))["summary"] if os.path.exists(d13_path) else None
    medium = load_model("gpt2-medium", dev)
    for p_ in medium.parameters():
        p_.requires_grad_(False)

    stage("1/5  training set: the D13 full set (real edits + synthetic edits)")
    ed_all = torch.load(os.path.join(OUT_DIR, "d13_edits.pt"), weights_only=False)
    ed = {k: v for k, v in ed_all.items() if v.get("rank") == 1}
    cache = torch.load(os.path.join(OUT_DIR, "d13_items.pt"), weights_only=False)
    syn_keys = sorted(k for k in ed if k in cache)
    if a.smoke:
        syn_keys = syn_keys[:20]
    print(f"   {len(syn_keys)} synthetic edits (of {len(ed)} that reached rank 1) on {len({ed[k]['tid'] for k in syn_keys})} distinct target words; {c['n_good']} real edits on {len({g['tid_new'] for g in good})} words", flush=True)
    n_real = 100 if a.smoke else len(c["train"])
    items, tids, tuned = list(c["train"][:n_real]), list(real_tids[:n_real]), [i % 5 == 0 for i in range(n_real)]
    for k in syn_keys:
        for t_, it in enumerate(cache[k]):
            items.append(it)
            tids.append(ed[k]["tid"])
            tuned.append(t_ == 0)
    S = pack(items, dev, with_goal=True)
    S["tid"], S["tuned"] = torch.tensor(tids, device=dev), torch.tensor(tuned, device=dev)
    s_x, s_y = typical_sizes(S, W0)
    print(f"   {len(items)} training texts; typical size of the change: {s_x:.3f} before the ridge map, {s_y:.3f} after", flush=True)

    jitems = torch.load(os.path.join(OUT_DIR, "d10_judge_items.pt"), weights_only=False)
    if a.smoke:
        jitems = [it for it in jitems if it["rec"] < len(test)]
    T0 = pack_judge([it for it in jitems if it["kind"] == 0], dev)
    rec_primary = torch.tensor([edits[r["case_id"]]["real"]["rank"] == 1 for r in test])
    rec_all = torch.ones(len(test), dtype=torch.bool)
    arm0 = T0["arm"].cpu()
    sel = {(a_, p_): torch.nonzero((arm0 == a_) & (m_[T0["rec"].cpu()])).flatten().to(dev) for a_ in (0, 1, 2) for p_, m_ in (("prim", rec_primary), ("all", rec_all))}
    devs = pack(c["dev"], dev)
    real_tuned = [dict(c["train"][i], tid=real_tids[i], base_rank=1) for i in range(0, n_real, 5)]
    for it in real_tuned:
        it.pop("goal", None)
    S_real_tuned = pack(real_tuned, dev)
    syn_tuned = [dict(cache[k][0], tid=ed[k]["tid"], base_rank=1) for k in syn_keys]
    for it in syn_tuned:
        it.pop("goal", None)
    S_syn_tuned = pack(syn_tuned, dev)
    I = identity(medium.cfg.d_model, dev)
    print(f"   {len(c['dev'])} dev records, {len(test)} test records ({int(rec_primary.sum())} primary)", flush=True)

    # ---------------- the sweep -----------------------------------------------------------------------------------------------------------
    runs_path = os.path.join(OUT_DIR, f"d14_runs{tag}.json")
    runs = {} if (a.fresh or a.smoke or not os.path.exists(runs_path)) else json.load(open(runs_path, encoding="utf8"))
    specs = {}
    stage(f"2/5  sweep: arms {list(ARMS)} x {seeds} seeds, {epochs} epochs each ({len(runs)} runs already finished)")
    order = [(arm, seed) for seed in range(seeds) for arm in ARMS]
    t_sweep = time.time()
    done_now = 0
    for arm, seed in order:
        key = f"{arm}_s{seed}"
        map_path = os.path.join(OUT_DIR, f"d14_map_{key}{tag}.pt")
        if key in runs and os.path.exists(map_path):
            saved = torch.load(map_path, weights_only=False)
            if saved["kind"] == "W":
                specs[key] = saved["W"]
            else:
                g = ResidualMlpDelta(W0, saved["s_x"], saved["s_y"])
                g.load_state_dict(saved["state"])
                specs[key] = g.cpu()
            r = runs[key]
            print(f"   {key} ({ARMS[arm]}): already done, dev {r['dev_score']:.2f}", flush=True)
            continue
        ts = time.time()
        torch.manual_seed(seed)
        hist = None
        if arm in ("R1", "R2"):
            W, ep, hist = train_map_hinge(medium, W0, S, devs, Q, lam=0.0, epochs=epochs, seed=seed, reg={"R1": 1.0, "R2": 10.0}[arm])
            spec, payload = W, {"kind": "W", "W": W.cpu()}
        elif arm == "LR":
            g = LowRankMap(W0, 16)
            state, ep, hist = train_module(medium, g, [{"params": [g.A, g.B], "lr": 1e-3}], S, devs, Q, epochs=epochs, seed=seed)
            g.load_state_dict(state)
            spec, payload = g.matrix(), {"kind": "W", "W": g.matrix().cpu()}
        else:
            g = ResidualMlpDelta(W0, s_x, s_y)
            groups = [{"params": [g.W], "lr": 2e-4, "weight_decay": 0.0}, {"params": list(g.up.parameters()) + list(g.down.parameters()), "lr": 1e-3, "weight_decay": 0.01}]
            state, ep, hist = train_module(medium, g, groups, S, devs, Q, penalty=lambda m: 0.1 * ((m.W - m.W0) ** 2).sum() / w0n, epochs=epochs, seed=seed)
            g.load_state_dict(state)
            g.eval()
            spec, payload = g, {"kind": "nn", "state": {k: v.cpu() for k, v in state.items()}, "s_x": s_x, "s_y": s_y}
        Dv, Wd = prepare(devs, spec, I)
        Tv, Wt = prepare(T0, spec, I)
        rows = []
        for d in doses:
            dv = score(devs, eval_ranks(medium, Dv, Wd, d, Q))
            rk = eval_ranks(medium, Tv, Wt, d, Q)
            row = {"dose": d, "dev": dv}
            for a_, name in ((0, "real"), (1, "wrong"), (2, "rand")):
                row[name] = {"prim": score(T0, rk, sel[(a_, "prim")]) if len(sel[(a_, "prim")]) else (float("nan"),) * 3,
                             "all": score(T0, rk, sel[(a_, "all")]) if len(sel[(a_, "all")]) else (float("nan"),) * 3}
            rows.append(row)
        best = max(rows, key=lambda r: (r["dev"][1], -r["dose"]))
        Rv, Wr = prepare(S_real_tuned, spec, I)
        Sv, Ws = prepare(S_syn_tuned, spec, I)
        tr_real = float((eval_ranks(medium, Rv, Wr, best["dose"], Q) == 1).float().mean())
        tr_syn = float((eval_ranks(medium, Sv, Ws, best["dose"], Q) == 1).float().mean())
        del Dv, Tv, Rv, Sv
        torch.cuda.empty_cache()
        runs[key] = {"arm": arm, "seed": seed, "best_epoch": ep, "dose": best["dose"], "dev_score": best["dev"][1], "rows": rows, "history": hist, "train_top1_real": tr_real, "train_top1_syn": tr_syn}
        specs[key] = spec.cpu() if torch.is_tensor(spec) else copy.deepcopy(spec).cpu()
        torch.save(payload, map_path)
        if not a.smoke:
            json.dump(runs, open(runs_path, "w", encoding="utf8"), default=float)
        r_, w_, n_ = best["real"]["prim"], best["wrong"]["prim"], best["rand"]["prim"]
        done_now += 1
        left = len(order) - len(runs)
        eta = (time.time() - t_sweep) / done_now * left
        print(f"   {key} ({ARMS[arm]}): epoch {ep:>2}, dose {best['dose']}, dev gain {best['dev'][1]:.2f} | TEST (primary): top-1 {pct(r_[0])}, gain {r_[1]:.2f}, median rank {r_[2]:.0f} | controls: wrong {pct(w_[0])}, "
              f"random-word {pct(n_[0])} | TRAINING top-1: real prompts {pct(tr_real)}, synthetic {pct(tr_syn)}   [{time.time() - ts:.0f} s; about {eta / 60:.0f} min left]", flush=True)
        del spec
        torch.cuda.empty_cache()

    # ---------------- summary -------------------------------------------------------------------------------------------------------------
    stage("3/5  the four arms (mean over seeds; test = primary records)")
    pick = lambda r: [x for x in r["rows"] if x["dose"] == r["dose"]][0]
    summ = {}
    for arm in ARMS:
        rs = [runs[f"{arm}_s{s}"] for s in range(seeds) if f"{arm}_s{s}" in runs]
        if not rs:
            continue
        t = [pick(r)["real"]["prim"][0] for r in rs]
        summ[arm] = {"n_runs": len(rs), "dev_score": mean([r["dev_score"] for r in rs]), "top1": mean(t), "top1_min": min(t), "top1_max": max(t), "gain": mean([pick(r)["real"]["prim"][1] for r in rs]),
                     "median_rank": mean([pick(r)["real"]["prim"][2] for r in rs]), "wrong_top1": mean([pick(r)["wrong"]["prim"][0] for r in rs]), "rand_top1": mean([pick(r)["rand"]["prim"][0] for r in rs]),
                     "train_top1_real": mean([r["train_top1_real"] for r in rs]), "train_top1_syn": mean([r["train_top1_syn"] for r in rs])}
    print(f"   {'arm':<40}{'dev score':>11}{'test top-1 (range)':>24}{'test gain':>11}{'med rank':>10}{'wrong':>8}{'random word':>13}{'train top-1 (real)':>20}{'gap':>6}")
    for arm, s_ in summ.items():
        print(f"   {arm + ': ' + ARMS[arm]:<40}{s_['dev_score']:>11.2f}{pct(s_['top1']):>12} ({pct(s_['top1_min'])} to {pct(s_['top1_max'])}){s_['gain']:>11.2f}{s_['median_rank']:>10.1f}{pct(s_['wrong_top1']):>8}"
              f"{pct(s_['rand_top1']):>13}{pct(s_['train_top1_real']):>20}{100 * (s_['train_top1_real'] - s_['top1']):>5.0f}")
    base13 = d13.get("1.0") if d13 else None
    if base13:
        print(f"   {'D13 full set (penalty 0.1, 3 seeds)':<40}{base13['dev_score']:>11.2f}{pct(base13['top1']):>12} ({pct(base13['top1_min'])} to {pct(base13['top1_max'])}){base13['gain']:>11.2f}{base13['median_rank']:>10.1f}"
              f"{pct(base13['wrong_top1']):>8}{pct(base13['rand_top1']):>13}{pct(base13['train_top1_real']):>20}{100 * (base13['train_top1_real'] - base13['top1']):>5.0f}")
    else:
        print("   (d13_training.json not found: no D13 comparison)")
    chosen = max(summ, key=lambda k: summ[k]["dev_score"])
    print(f"\n   arm chosen by the best mean dev score (fixed rule): {chosen}: {ARMS[chosen]}")

    # ---------------- judging the chosen arm -----------------------------------------------------------------------------------------------
    stage(f"4/5  judging the chosen arm ({chosen}) on rewordings, neighbours and unrelated prompts")
    del S, S_real_tuned, S_syn_tuned, cache
    torch.cuda.empty_cache()
    J = pack_judge(jitems, dev)
    judged = {}
    for s in range(seeds):
        key = f"{chosen}_s{s}"
        if key not in runs:
            continue
        sp = specs[key]
        sp = sp.to(dev) if torch.is_tensor(sp) else sp.to(dev)
        Jv, Wj = prepare(J, sp, I)
        res = judge_map(medium, Jv, Wj, runs[key]["dose"], Q)
        judged[key] = {"prim": summarize(J, res, rec_primary), "all": summarize(J, res, rec_all)}
        del Jv, sp
        torch.cuda.empty_cache()
        print(f"   {key} judged ({time.time() - t0:.0f} s)", flush=True)
    ks = list(judged)

    def avg(subset, arm, k):
        return mean([judged[x][subset][arm][k] for x in ks if arm in judged[x][subset]])

    ref = {k: d7["ok"]["stage2"][k] for k in ("medium unchanged", "small unchanged", "small with the edit")}
    refa = {k: d7["all"]["stage2"][k] for k in ("medium unchanged", "small unchanged", "small with the edit")}
    print("\n" + "=" * 130)
    print("RESULT  step D14 (maps with less freedom, trained on the D13 full set)" + ("  [smoke: numbers mean nothing]" if a.smoke else ""))
    for subset, label, rf in (("prim", "PRIMARY: 88 test records whose original edit reached rank 1 in small", ref), ("all", "ALL test records", refa)):
        print(f"\n  ---- {label} (chosen arm {chosen}, mean over {len(ks)} seeds) ----")
        print(f"    {'arm':<62}{'top-1':>7}{'med rank':>10}{'PS':>7}{'NS':>7}{'KL':>8}{'flips':>8}")
        for k in ("medium unchanged", "small unchanged", "small with the edit"):
            print(f"    {k:<62}{'':>7}{'':>10}{pct(rf[k]['PS']):>7}{pct(rf[k]['NS']):>7}{rf[k]['KL']:>8.3f}{pct(rf[k]['flips']):>8}")
        for arm, nm in (("real", "real recipe"), ("wrong", "another record's recipe"), ("rand", "random-word edit (own word)")):
            kl = avg(subset, arm, "unrelated_KL") if arm == "real" else float("nan")
            fl = avg(subset, arm, "unrelated_flips") if arm == "real" else float("nan")
            print(f"    {chosen + ': ' + nm:<62}{pct(avg(subset, arm, 'top1')):>7}{avg(subset, arm, 'median_rank'):>10.0f}{pct(avg(subset, arm, 'PS')):>7}{pct(avg(subset, arm, 'NS')):>7}{kl:>8.3f}{pct(fl):>8}")
    lift_src = ref["small with the edit"]["PS"] - ref["small unchanged"]["PS"]
    lift = avg("prim", "real", "PS") - ref["medium unchanged"]["PS"]
    ratio = lift / lift_src if lift_src > 0 else float("nan")
    ns_fall = ref["medium unchanged"]["NS"] - avg("prim", "real", "NS")
    ns_src = ref["small unchanged"]["NS"] - ref["small with the edit"]["NS"]
    top1, rand1 = avg("prim", "real", "top1"), avg("prim", "rand", "top1")
    ga, gb, gc, gd = top1 >= GATE_TOP1, ratio >= GATE_LIFT, ns_fall <= ns_src, (top1 - rand1) >= GATE_SPEC
    gap_chosen = summ[chosen]["train_top1_real"] - summ[chosen]["top1"]
    if ga and gb and gc and gd:
        reading = "all four hold: a map with less freedom, trained on varied edits, carries unseen real edits into medium"
    elif base13 is None:
        reading = "(a) fails; no D13 result file, so the comparison readings are not made: see the numbers"
    else:
        rise = summ[chosen]["top1"] - base13["top1"]
        gap13 = base13["train_top1_real"] - base13["top1"]
        if rise >= READ_POINTS and gap_chosen < gap13:
            reading = "(a) fails but the test top-1 is at least 5 points above D13's full set and the training-test gap is smaller: regularisation helps but not enough"
        elif rise <= -READ_POINTS:
            reading = "(a) fails and the test top-1 is more than 5 points BELOW D13's full set: regularisation hurts"
        elif abs(rise) < READ_POINTS:
            reading = "(a) fails and the test top-1 is within 5 points of D13's full set: memorisation was not what holds the map back with this data (next: amortised D6, or stop and write up)"
        else:
            reading = "(a) fails; the test top-1 is at least 5 points above D13's full set but the training-test gap does not shrink: see the numbers"
        if summ[chosen]["rand_top1"] - base13["rand_top1"] >= max(rise, 0.0) and rise > 0:
            reading += "; NOTE: the random-word edit rises as much as the real target: the map has learned to push any word"
    print(f"\n  GATE 9 (chosen arm {chosen}, 88 primary records, mean over {len(ks)} seeds): (a) top-1 {pct(top1)} >= {GATE_TOP1:.0%}: {'ok' if ga else 'NO'};  (b) rewordings lift = {ratio:.2f} of the edit's own lift in small (>= {GATE_LIFT}): {'ok' if gb else 'NO'};  "
          f"(c) neighbours fall {100 * ns_fall:.0f} points, the edit itself in small {100 * ns_src:.1f}: {'ok' if gc else 'NO'};  (d) real minus random-word top-1 = {100 * (top1 - rand1):.0f} points (>= {int(100 * GATE_SPEC)}): {'ok' if gd else 'NO'}   ->  {'PASS' if (ga and gb and gc and gd) else 'FAIL'}")
    if base13:
        print(f"  against D13's full set (same data, penalty 0.1): test top-1 {pct(base13['top1'])} -> {pct(summ[chosen]['top1'])}; training-test gap {100 * (base13['train_top1_real'] - base13['top1']):.0f} -> {100 * gap_chosen:.0f} points")
    print(f"  reading (fixed in the plan before the run): {reading}")
    out = {"arms": ARMS, "seeds": seeds, "epochs": epochs, "n_syn_edits": len(syn_keys), "runs": runs, "summary": summ, "chosen": chosen, "d13_full_set": base13,
           "gate9": {"top1": top1, "lift_ratio": ratio, "ns_fall": ns_fall, "ns_fall_source": ns_src, "rand_top1": rand1, "a": bool(ga), "b": bool(gb), "c": bool(gc), "d": bool(gd), "reading": reading, "gap_chosen": gap_chosen},
           "judged": judged}
    path = os.path.join(OUT_DIR, f"d14_regularised{tag}.json")
    json.dump(out, open(path, "w", encoding="utf8"), indent=1, default=float)
    stage(f"5/5  done in {time.time() - t0:.0f} s; saved {os.path.relpath(path, ROOT)}, the maps (d14_map_*.pt) and the log {os.path.relpath(log_path, ROOT)}")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
