"""
Step S3 of docs/cross_model_transfer/PLAN.md: the random-word arm compared with its OWN unchanged level, for a map trained on edits (D7 by default). The design and the reading rule are in the
plan, written before this tool was built.

    .venv\\Scripts\\python.exe tools/transfer/27_random_word_baseline.py                       # the D7 map (d7_map.pt), GPU if free
    .venv\\Scripts\\python.exe tools/transfer/27_random_word_baseline.py --device cpu         # a few minutes on 2 to 4 CPU threads
    .venv\\Scripts\\python.exe tools/transfer/27_random_word_baseline.py --maps d7_map.pt d10_map_lam0.0_s0.pt

A map file needs the keys W [768, 1024] and dose (the D7, D10 and D13 maps have them). Saves outputs/transfer/s3_random_word_baseline.json.
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
from scipy.stats import wilcoxon

from src.transfer.baseline_judge import judge_with_baseline, per_record
from src.transfer.models import load_model, pick_device
from src.transfer.om_judge import pack_judge
from src.transfer.runlog import tee_to

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
Q = 16
P_LINE = 0.01


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def pct(x):
    return "  n/a" if x != x else f"{100 * x:5.1f}%"


def main():
    print(f"log: {os.path.relpath(tee_to('27_random_word_baseline'), ROOT)}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=["d7_map.pt"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--dose", type=float, default=None, help="override the dose stored in the map file")
    a = ap.parse_args()
    dev = pick_device(a.device)
    t0 = time.time()
    bench = json.load(open(os.path.join(OUT_DIR, "d_benchmark.json"), encoding="utf8"))
    edits = torch.load(os.path.join(OUT_DIR, "d_edits.pt"), weights_only=False)
    bench = [r for r in bench if r["case_id"] in edits]
    test = [r for r in bench if r["split"] == "test"]
    n = len(test)
    jitems = torch.load(os.path.join(OUT_DIR, "d10_judge_items.pt"), weights_only=False)
    jitems = [it for it in jitems if it["arm"] in (0, 2) and it["kind"] in (0, 1, 2)]
    J = pack_judge(jitems, dev)
    primary = [edits[r["case_id"]]["real"]["rank"] == 1 for r in test]
    has_rand = [("random" in edits[r["case_id"]]) for r in test]
    medium = load_model("gpt2-medium", dev)
    for p_ in medium.parameters():
        p_.requires_grad_(False)
    print(f"{n} test records ({sum(primary)} primary), {len(jitems)} judge texts (real arm and random-word arm; tuned prompt, rewordings, neighbours)", flush=True)
    out = {}
    for name in a.maps:
        m = torch.load(os.path.join(OUT_DIR, name), weights_only=False)
        W, dose = m["W"].to(dev), (a.dose if a.dose is not None else float(m["dose"]))
        res = judge_with_baseline(medium, J, W, dose, Q)
        pr = {(arm, kind, key): per_record(J, res, arm, kind, key, n) for arm in (0, 2) for kind in (1, 2) for key in ("es", "es0")}
        top1 = {arm: per_record(J, {"t": (res["rank"] == 1).float()}, arm, 0, "t", n) for arm in (0, 2)}
        ps_lift = {arm: pr[(arm, 1, "es")] - pr[(arm, 1, "es0")] for arm in (0, 2)}
        ns_fall = {arm: (1 - pr[(arm, 2, "es0")]) - (1 - pr[(arm, 2, "es")]) for arm in (0, 2)}      # NS before minus NS after (positive = neighbours lost)
        o_map = {"dose": dose}
        print("\n" + "=" * 118)
        print(f"RESULT  step S3: {name} (dose {dose}); each arm against its own unchanged level")
        for label, keep in (("PRIMARY (original edit reached rank 1 in small)", [i for i in range(n) if primary[i] and has_rand[i]]), ("ALL test records with a random-word arm", [i for i in range(n) if has_rand[i]])):
            sel = lambda t: [float(t[i]) for i in keep]
            rows = {}
            print(f"\n  ---- {label}: {len(keep)} records ----")
            print(f"    {'':<26}{'top-1':>8}{'rewordings: unchanged':>24}{'with map':>11}{'lift':>8}{'neighbours kept: unchanged':>30}{'with map':>11}{'fall':>8}")
            for arm, nm in ((0, "real target"), (2, "random word (own)")):
                r = {"top1": mean(sel(top1[arm])), "ps_before": mean(sel(pr[(arm, 1, "es0")])), "ps_after": mean(sel(pr[(arm, 1, "es")])), "ps_lift": mean(sel(ps_lift[arm])),
                     "ns_before": 1 - mean(sel(pr[(arm, 2, "es0")])), "ns_after": 1 - mean(sel(pr[(arm, 2, "es")])), "ns_fall": mean(sel(ns_fall[arm]))}
                rows[nm] = r
                print(f"    {nm:<26}{pct(r['top1']):>8}{pct(r['ps_before']):>24}{pct(r['ps_after']):>11}{100 * r['ps_lift']:>7.1f}p{pct(r['ns_before']):>30}{pct(r['ns_after']):>11}{100 * r['ns_fall']:>7.1f}p")
            d = [float(ps_lift[0][i] - ps_lift[2][i]) for i in keep]
            try:
                p = float(wilcoxon(sel(ps_lift[0]), sel(ps_lift[2]), alternative="greater").pvalue)
            except ValueError:
                p = float("nan")
            g = random.Random(0)
            boots = sorted(mean([d[g.randrange(len(d))] for _ in d]) for _ in range(10000))
            lo, hi = boots[250], boots[9750]
            more, same, less = sum(v > 1e-9 for v in d), sum(abs(v) <= 1e-9 for v in d), sum(v < -1e-9 for v in d)
            reading = "p < 0.01: for this map the real target's rewordings lift is larger than a random word's (more than a push)" if p < P_LINE else "NOT SHOWN at this sample size"
            print(f"    difference of the rewordings lifts (real minus random word) {100 * mean(d):+.1f} points (bootstrap 95% interval {100 * lo:+.1f} to {100 * hi:+.1f}); records with the real lift larger / equal / smaller: {more} / {same} / {less}; one-sided Wilcoxon p = {p:.1e}")
            print(f"    reading (fixed in the plan before the run): {reading}")
            o_map[label] = {"arms": rows, "diff": mean(d), "ci95": [lo, hi], "larger": more, "equal": same, "smaller": less, "p": p, "reading": reading, "n": len(keep)}
        out[name] = o_map
    path = os.path.join(OUT_DIR, "s3_random_word_baseline.json")
    json.dump(out, open(path, "w", encoding="utf8"), indent=1, default=float)
    print(f"\n  total time {time.time() - t0:.0f} s; saved {os.path.relpath(path, ROOT)}")
    print("=" * 118)


if __name__ == "__main__":
    main()
