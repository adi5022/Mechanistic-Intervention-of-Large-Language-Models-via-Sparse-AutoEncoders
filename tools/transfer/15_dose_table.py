"""
Step D9b of docs/cross_model_transfer/PLAN.md: a POST-HOC follow-up to D9 (longer training). Labelled post-hoc because it was designed after seeing that D9's
dose choice (most dev records with the target on top, out of only 28) is fragile. The pre-set D9 verdict stays the verdict; this tool shows what the dose
was doing.

    .venv\\Scripts\\python.exe tools/transfer/15_dose_table.py --smoke      # 2 epochs, 1 seed, about a minute
    .venv\\Scripts\\python.exe tools/transfer/15_dose_table.py              # 40 epochs, 3 seeds, about 20 minutes

For each seed ONE run of 40 epochs is trained on all 289 edits (batched, D7 settings); the 15-epoch map is the best epoch among the first 15 of the same run (same
seed, so identical to a run that stops at 15). For every map and every dose in {0.5, 1, 1.5, 2, 3} the table gives the dev result (28 records) and the test result
(88 primary records), then three ways to pick the dose are compared: (A) the D7/D8/D9 rule (dev top-1, then dev gain), (B) the dose with the best dev mean log-rank
gain, (C) a fixed dose of 2.0. The 15-epoch maps with rule A should reproduce D8 (top-1 18%, 20%, 20%). Test numbers at every dose are descriptive, not used to choose.
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
from src.transfer.models import load_model, pick_device
from src.transfer.runlog import tee_to
from src.transfer.om_batched import eval_ranks, pack, score, train_map_horizons

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
Q = 16
DOSES = [0.5, 1.0, 1.5, 2.0, 3.0]


def pct(x):
    return f"{100 * x:3.0f}%"


def main():
    log_path = tee_to("15_dose_table")
    print(f"log: {os.path.relpath(log_path, ROOT)}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--horizons", default="15,40")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.horizons, a.seeds = 2, "1,2", 1
    horizons = [int(x) for x in a.horizons.split(",")]
    dev = pick_device(a.device)
    t0 = time.time()
    c = torch.load(os.path.join(OUT_DIR, "d8_items.pt"), weights_only=False)           # built by 14_datasize_check.py
    maps = torch.load(os.path.join(OUT_DIR, "maps_gpt2_to_gpt2-medium.pt"))["maps"]
    W0 = maps[f"s2m_L8_L{Q}"]["W"].to(dev)
    medium = load_model("gpt2-medium", dev)
    for p_ in medium.parameters():
        p_.requires_grad_(False)
    train, devs, test = pack(c["train"], dev, with_goal=True), pack(c["dev"], dev), pack(c["test"], dev)
    prim = torch.tensor(c["primary"], device=dev)
    allx = torch.arange(len(test["lens"]), device=dev)
    print(f"{c['n_good']} training records ({len(train['lens'])} texts), {len(devs['lens'])} dev, {len(test['lens'])} test texts ({len(prim)} primary); horizons {horizons}", flush=True)
    r0 = eval_ranks(medium, test, W0, 2.0, Q)
    ridge = score(test, r0, prim)
    print(f"reference, ridge map at dose 2.0: primary top-1 {pct(ridge[0])}, gain {ridge[1]:.2f}, median rank {ridge[2]:.0f}\n", flush=True)

    runs = []
    for seed in range(a.seeds):
        ts = time.time()
        found, hist = train_map_horizons(medium, W0, train, devs, Q, epochs=a.epochs, horizons=horizons, seed=seed)
        print(f"seed {seed}: trained {a.epochs} epochs in {time.time() - ts:.0f} s; train KL {hist[0]['train_kl']:.3f} -> {hist[-1]['train_kl']:.3f}", flush=True)
        for h in horizons:
            W, ep = found[h]
            if not a.smoke:
                torch.save(W.cpu(), os.path.join(OUT_DIR, f"d9b_map_h{h}_s{seed}.pt"))
            rows = []
            print(f"  maps up to {h} epochs (best epoch on dev: {ep}); every dose:", flush=True)
            print(f"      {'dose':>5}{'dev top-1':>11}{'dev gain':>10}{'test top-1':>12}{'test gain':>11}{'test med rank':>15}{'all-150 top-1':>15}", flush=True)
            for d in DOSES:
                dv = score(devs, eval_ranks(medium, devs, W, d, Q))
                rk = eval_ranks(medium, test, W, d, Q)
                tp, ta = score(test, rk, prim), score(test, rk, allx)
                rows.append({"dose": d, "dev": dv, "test": tp, "all": ta})
                print(f"      {d:>5}{pct(dv[0]):>11}{dv[1]:>10.2f}{pct(tp[0]):>12}{tp[1]:>11.2f}{tp[2]:>15.0f}{pct(ta[0]):>15}", flush=True)
            pick = {"A (D7/D8/D9 rule: dev top-1, then dev gain)": max(rows, key=lambda r: (round(r["dev"][0], 6), r["dev"][1], -r["dose"])),
                    "B (best dev gain)": max(rows, key=lambda r: (r["dev"][1], -r["dose"])),
                    "C (fixed dose 2.0)": [r for r in rows if r["dose"] == 2.0][0]}
            for k, r in pick.items():
                print(f"      rule {k}: dose {r['dose']} -> test top-1 {pct(r['test'][0])}, gain {r['test'][1]:.2f}", flush=True)
            runs.append({"seed": seed, "horizon": h, "best_epoch": ep, "rows": rows, "picked": {k: r["dose"] for k, r in pick.items()},
                         "picked_test": {k: {"top1": r["test"][0], "gain": r["test"][1], "median_rank": r["test"][2]} for k, r in pick.items()}, "history": hist})

    print("\n" + "=" * 120)
    print("RESULT  step D9b (post-hoc: what was the dose doing in the longer-training test?)" + ("  [smoke test: do not read the numbers]" if a.smoke else ""))
    print(f"  reference: ridge map at dose 2.0, primary top-1 {pct(ridge[0])}, gain {ridge[1]:.2f}")
    print(f"  {'epochs':>7}  {'dose rule':<48}{'top-1 mean (range)':>26}{'gain mean':>11}{'median rank mean':>18}")
    summ = {}
    for h in horizons:
        for k in runs[0]["picked_test"]:
            rs = [r["picked_test"][k] for r in runs if r["horizon"] == h]
            t, g, m = [x["top1"] for x in rs], [x["gain"] for x in rs], [x["median_rank"] for x in rs]
            summ[(h, k)] = (sum(t) / len(t), sum(g) / len(g))
            print(f"  {h:>7}  {k:<48}{pct(sum(t) / len(t)):>12} ({pct(min(t))} to {pct(max(t))}){sum(g) / len(g):>11.2f}{sum(m) / len(m):>18.1f}")
    print("\n  descriptive comparison (not the D9 verdict): does the longest horizon beat the shortest by 4 points of top-1 AND 0.10 of gain, under each dose rule?")
    lo, hi = min(horizons), max(horizons)
    for k in runs[0]["picked_test"]:
        dt, dg = summ[(hi, k)][0] - summ[(lo, k)][0], summ[(hi, k)][1] - summ[(lo, k)][1]
        print(f"    {k:<48}{100 * dt:+5.0f} points, {dg:+.2f} gain  ->  {'yes' if dt >= 0.04 and dg >= 0.10 else 'no'}")
    out = os.path.join(OUT_DIR, "d9b_dose_table" + ("_smoke" if a.smoke else "") + ".json")
    json.dump({"reference_ridge": ridge, "runs": runs}, open(out, "w", encoding="utf8"), indent=1)
    print(f"\n  total time {time.time() - t0:.0f} s; saved {os.path.relpath(out, ROOT)}")
    print("=" * 120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
