"""
Step D8 of docs/cross_model_transfer/PLAN.md: does the map trained on edits (D7) get better with more training data? A diagnostic, written before the run
with a fixed decision rule in the plan. Uses only the data D7 already produced (outputs/transfer/d7_pool.json, d7_train_edits.pt) and a BATCHED version of
the D7 training (src/transfer/om_batched.py).

    .venv\\Scripts\\python.exe tools/transfer/14_datasize_check.py --check      # equivalence and speed against the one-text-at-a-time D7 code (about 5 min the first time)
    .venv\\Scripts\\python.exe tools/transfer/14_datasize_check.py              # the data-size check: 25%, 50%, 100% of the 289 training edits, 3 seeds each

For every fraction and seed the map is trained with the D7 settings (15 epochs, Adam 2e-4, batch 32, penalty 0.1), the epoch is chosen on the 28 dev
records, the dose on the dev records, and the tuned-prompt result is read on the 150 test records (88 primary: the original edit reached rank 1 in small),
using the same small-tuned recipes as D3 and D7. The fraction-100% seed-0 run replicates D7 (18% top-1, median rank 7, epoch 15, dose 2.0).
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

from src.sae_utils import load_sae_for_layer
from src.transfer.models import load_model, pick_device
from src.transfer.runlog import tee_to
from src.transfer.om_batched import batch_kl, build_eval_item, eval_ranks, pack, score, train_map
from src.transfer.output_matching import build_item, item_loss

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
Q = 16
DOSES = [0.5, 1.0, 1.5, 2.0, 3.0]
RIDGE_DOSE = 2.0
RULE_TOP1_POINTS = 0.04
LONGER_TOP1_POINTS, LONGER_GAIN = 0.04, 0.10


def pct(x):
    return f"{100 * x:4.0f}%"


def sub(S, idx):
    return {k: v[idx] for k, v in S.items()}


def get_data(small, medium, sae, dev, rebuild):
    path = os.path.join(OUT_DIR, "d8_items.pt")
    bench = json.load(open(os.path.join(OUT_DIR, "d_benchmark.json"), encoding="utf8"))
    edits = torch.load(os.path.join(OUT_DIR, "d_edits.pt"), weights_only=False)
    bench = [r for r in bench if r["case_id"] in edits]
    tedits = torch.load(os.path.join(OUT_DIR, "d7_train_edits.pt"), weights_only=False)
    pool = json.load(open(os.path.join(OUT_DIR, "d7_pool.json"), encoding="utf8"))
    held_words, held_subjects = {r["tid_new"] for r in bench}, {r["subject"] for r in bench}
    keep = [r for r in pool if r["tid_new"] not in held_words and r["subject"] not in held_subjects][:800]
    good = [r for r in keep if tedits[r["case_id"]]["rank"] == 1]
    dev_recs = [r for r in bench if r["split"] == "dev" and edits[r["case_id"]]["real"]["rank"] == 1]
    test_recs = [r for r in bench if r["split"] == "test"]
    if os.path.exists(path) and not rebuild:
        c = torch.load(path, weights_only=False)
        print(f"items loaded from {os.path.relpath(path, ROOT)}", flush=True)
    else:
        t0 = time.time()
        train, rec_of = [], []
        for n, r in enumerate(good):
            for text in [r["prompt"]] + r["paraphrases"][:2] + r["neighbours"][:2]:
                it = build_item(small, medium, sae, text, tedits[r["case_id"]], Q, dev)
                train.append({"tok": it["tok"][0].cpu(), "dS": it["dS"].cpu(), "hm": it["hm"][0].cpu(), "goal": it["goal"].cpu()})
                rec_of.append(n)
            if (n + 1) % 50 == 0:
                print(f"   building training texts: {n + 1}/{len(good)} records ({time.time() - t0:.0f} s)", flush=True)
        ev = lambda recs: [{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in build_eval_item(small, medium, sae, r["prompt"], edits[r["case_id"]]["real"], Q, dev, r["tid_new"]).items()} for r in recs]
        c = {"train": train, "rec_of": rec_of, "dev": ev(dev_recs), "test": ev(test_recs), "n_good": len(good),
             "primary": [i for i, r in enumerate(test_recs) if edits[r["case_id"]]["real"]["rank"] == 1]}
        torch.save(c, path)
        print(f"items built and saved ({time.time() - t0:.0f} s)", flush=True)
    S = {"train": pack(c["train"], dev, with_goal=True), "dev": pack(c["dev"], dev), "test": pack(c["test"], dev)}
    S["rec_of"] = torch.tensor(c["rec_of"], device=dev)
    S["primary"] = torch.tensor(c["primary"], device=dev)
    S["n_good"] = c["n_good"]
    return S, c


def check(medium, W0, S, c, dev):
    print("\n[check] batched against the one-text-at-a-time D7 code, first 32 training texts, ridge map", flush=True)
    idx = torch.arange(32, device=dev)
    orig = [{"tok": c["train"][i]["tok"][None].to(dev), "dS": c["train"][i]["dS"].to(dev), "hm": c["train"][i]["hm"][None].to(dev), "goal": c["train"][i]["goal"].to(dev)} for i in range(32)]
    for dose in (1.0, 2.0):
        with torch.no_grad():
            a = torch.stack([item_loss(medium, it, W0, dose, Q) for it in orig])
            b = batch_kl(medium, S["train"], idx, W0, dose, Q)
            b16 = batch_kl(medium, S["train"], idx, W0, dose, Q, amp=True)
        print(f"   dose {dose}: per-text KL, largest difference batched vs single {float((a - b).abs().max()):.2e} (mean KL {float(a.mean()):.4f}); half precision vs single {float((a - b16).abs().max()):.2e}", flush=True)
    W = torch.nn.Parameter(W0.clone())
    g1 = torch.autograd.grad(sum(item_loss(medium, it, W, 1.0, Q) for it in orig) / 32, W)[0]
    g2 = torch.autograd.grad(batch_kl(medium, S["train"], idx, W, 1.0, Q).mean(), W)[0]
    g3 = torch.autograd.grad(batch_kl(medium, S["train"], idx, W, 1.0, Q, amp=True).mean(), W)[0]
    cos = lambda x, y: float((x * y).sum() / (x.norm() * y.norm()))
    print(f"   gradient on the map: cosine batched vs single {cos(g1, g2):.6f}, relative norm difference {float((g1 - g2).norm() / g1.norm()):.2e}; half precision cosine {cos(g1, g3):.6f}", flush=True)
    print("\n[check] speed, one training step of 32 texts (forward and backward)", flush=True)

    def timed(fn, reps=3):
        fn(); torch.cuda.synchronize(); t = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t) / reps

    def single():
        Wp = torch.nn.Parameter(W0.clone())
        torch.autograd.grad(sum(item_loss(medium, it, Wp, 1.0, Q) for it in orig) / 32, Wp)

    def batched(amp):
        def f():
            Wp = torch.nn.Parameter(W0.clone())
            torch.autograd.grad(batch_kl(medium, S["train"], idx, Wp, 1.0, Q, amp).mean(), Wp)
        return f

    steps = len(S["train"]["lens"]) / 32
    for name, fn in (("one text at a time (D7)", single), ("batched, float32", batched(False)), ("batched, half precision", batched(True))):
        t = timed(fn)
        print(f"   {name:<28}{t * 1000:7.0f} ms per step, about {t * steps:5.0f} s per epoch ({int(steps)} steps)", flush=True)


def main():
    log_path = tee_to("14_datasize_check")
    print(f"log: {os.path.relpath(log_path, ROOT)}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fractions", default="0.25,0.5,1.0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--amp", action="store_true", help="blocks in half precision (check the numbers first with --check)")
    ap.add_argument("--tag", default="datasize", help="name of the result file d8_<tag>.json (use another tag so the data-size result is not overwritten)")
    ap.add_argument("--longer", action="store_true", help="step D9: compare with the 15-epoch 100%% runs stored in d8_datasize.json and apply the rule fixed in the plan")
    ap.add_argument("--rebuild-items", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = pick_device(a.device)
    t0 = time.time()
    maps = torch.load(os.path.join(OUT_DIR, "maps_gpt2_to_gpt2-medium.pt"))["maps"]
    W0 = maps[f"s2m_L8_L{Q}"]["W"].to(dev)
    small, medium, sae = load_model("gpt2", dev), load_model("gpt2-medium", dev), load_sae_for_layer(8, dev)
    for m in (small, medium):
        for p_ in m.parameters():
            p_.requires_grad_(False)
    S, c = get_data(small, medium, sae, dev, a.rebuild_items)
    print(f"{S['n_good']} training records with a successful edit ({len(S['train']['lens'])} texts), {len(S['dev']['lens'])} dev, {len(S['test']['lens'])} test texts ({len(S['primary'])} primary)", flush=True)
    if a.check:
        check(medium, W0, S, c, dev)
        print(f"\ncheck done in {time.time() - t0:.0f} s")
        return 0

    prim = S["primary"]
    allx = torch.arange(len(S["test"]["lens"]), device=dev)
    r0 = eval_ranks(medium, S["test"], W0, RIDGE_DOSE, Q, amp=a.amp)
    base = (S["test"]["base_rank"].median().item(),)
    ridge_p, ridge_a = score(S["test"], r0, prim), score(S["test"], r0, allx)
    print(f"\nreference, ridge map at dose {RIDGE_DOSE} (D3): primary top-1 {pct(ridge_p[0])}, gain {ridge_p[1]:.2f}, median rank {ridge_p[2]:.0f}; all 150: top-1 {pct(ridge_a[0])}, median rank {ridge_a[2]:.0f}", flush=True)
    print("D7 recorded (all 289 edits): primary top-1 18%, median rank 7, epoch 15, dose 2.0\n", flush=True)

    n_rec = S["n_good"]
    results = {}
    for f in [float(x) for x in a.fractions.split(",")]:
        for seed in range(a.seeds):
            tr0 = time.time()
            if f >= 1.0:
                chosen = list(range(n_rec))
            else:
                chosen = sorted(random.Random(1000 * seed + int(round(f * 100))).sample(range(n_rec), int(round(f * n_rec))))
            idx = torch.nonzero(torch.isin(S["rec_of"], torch.tensor(chosen, device=dev))).flatten()
            train = sub(S["train"], idx)
            log = (lambda ep, h, best: print(f"      epoch {ep:>2}: train KL {h['train_kl']:.4f}, dev top-1 {pct(h['dev_top1'])}, gain {h['dev_gain']:.2f}, median rank {h['dev_median_rank']:.0f}{'  <- best' if best else ''}", flush=True)) if (f >= 1.0 and seed == 0) else None
            W, ep, hist = train_map(medium, W0, train, S["dev"], Q, epochs=a.epochs, seed=seed, amp=a.amp, log=log)
            best = None
            for d in DOSES:
                t1, g1, _ = score(S["dev"], eval_ranks(medium, S["dev"], W, d, Q, amp=a.amp))
                key = (round(t1, 6), g1, -d)
                if best is None or key > best[0]:
                    best = (key, d)
            dose = best[1]
            rk = eval_ranks(medium, S["test"], W, dose, Q, amp=a.amp)
            p, al = score(S["test"], rk, prim), score(S["test"], rk, allx)
            results.setdefault(f, []).append({"seed": seed, "n_records": len(chosen), "epoch": ep, "dose": dose, "primary": {"top1": p[0], "gain": p[1], "median_rank": p[2]},
                                              "all": {"top1": al[0], "gain": al[1], "median_rank": al[2]}, "history": hist})
            print(f"   {int(f * 100):>3}% of the edits ({len(chosen)} records), seed {seed}: epoch {ep}, dose {dose}: primary top-1 {pct(p[0])}, gain {p[1]:.2f}, median rank {p[2]:.0f}; all 150 top-1 {pct(al[0])}  [{time.time() - tr0:.0f} s]", flush=True)

    print("\n" + "=" * 120)
    print("RESULT  step D8 (does the map trained on edits improve with more training data?)")
    print(f"  reference: ridge map (D3) primary top-1 {pct(ridge_p[0])}, gain {ridge_p[1]:.2f}, median rank {ridge_p[2]:.0f}; D7 recorded: 18%, median rank 7")
    print(f"  {'training edits':<22}{'records':>8}{'top-1 (mean, range over seeds)':>34}{'mean log-rank gain':>20}{'median rank (mean)':>20}")
    summ = {}
    for f, runs in sorted(results.items()):
        t = [r["primary"]["top1"] for r in runs]
        g = [r["primary"]["gain"] for r in runs]
        m = [r["primary"]["median_rank"] for r in runs]
        summ[f] = {"records": runs[0]["n_records"], "top1_mean": sum(t) / len(t), "top1_min": min(t), "top1_max": max(t), "gain_mean": sum(g) / len(g), "median_rank_mean": sum(m) / len(m)}
        print(f"  {int(f * 100):>3}%{'':<18}{runs[0]['n_records']:>8}{pct(summ[f]['top1_mean']):>14}  ({pct(min(t))} to {pct(max(t))}){'':<8}{summ[f]['gain_mean']:>12.2f}{summ[f]['median_rank_mean']:>20.1f}")
    verdict = "n/a (needs the 50% and 100% fractions)"
    if 0.5 in summ and 1.0 in summ:
        up_top1 = summ[1.0]["top1_mean"] - summ[0.5]["top1_mean"] >= RULE_TOP1_POINTS
        mono = (summ[1.0]["gain_mean"] > summ[0.5]["gain_mean"]) and (0.25 not in summ or summ[0.5]["gain_mean"] > summ[0.25]["gain_mean"])
        verdict = "RISING: more data is worth trying" if (up_top1 and mono) else ("FLAT: more data will not rescue this map" if (not up_top1 and not mono) else "UNCLEAR: mixed signals (top-1 %s, gain %s)" % ("rises" if up_top1 else "does not rise", "rises" if mono else "does not rise steadily"))
    print(f"\n  decision rule (fixed in the plan before the run): top-1 at 100% at least {int(100 * RULE_TOP1_POINTS)} points above 50%, and mean log-rank gain rising from 25% to 50% to 100%  ->  {verdict}")
    longer = None
    if a.longer and 1.0 in summ and os.path.exists(os.path.join(OUT_DIR, "d8_datasize.json")):
        old = json.load(open(os.path.join(OUT_DIR, "d8_datasize.json"), encoding="utf8"))["runs"]["1.0"]
        o_top1, o_gain = sum(r["primary"]["top1"] for r in old) / len(old), sum(r["primary"]["gain"] for r in old) / len(old)
        d_top1, d_gain = summ[1.0]["top1_mean"] - o_top1, summ[1.0]["gain_mean"] - o_gain
        helps = d_top1 >= LONGER_TOP1_POINTS and d_gain >= LONGER_GAIN
        longer = {"epochs": a.epochs, "top1_15_epochs": o_top1, "gain_15_epochs": o_gain, "top1_now": summ[1.0]["top1_mean"], "gain_now": summ[1.0]["gain_mean"],
                  "chosen_epochs": [r["epoch"] for r in results[1.0]], "verdict": "LONGER TRAINING HELPS" if helps else "NO REPRODUCIBLE IMPROVEMENT FROM LONGER TRAINING"}
        print(f"\n  step D9 (longer training, {a.epochs} epochs against 15, same 289 edits): top-1 {pct(o_top1)} -> {pct(summ[1.0]['top1_mean'])} ({100 * d_top1:+.0f} points), mean log-rank gain {o_gain:.2f} -> {summ[1.0]['gain_mean']:.2f} ({d_gain:+.2f}); "
              f"epochs chosen on dev: {longer['chosen_epochs']}")
        print(f"  rule fixed in the plan before the run: top-1 at least {int(100 * LONGER_TOP1_POINTS)} points higher AND mean log-rank gain at least {LONGER_GAIN} higher than the 15-epoch runs  ->  {longer['verdict']}")
    out = os.path.join(OUT_DIR, f"d8_{a.tag}.json")
    json.dump({"reference_ridge_primary": {"top1": ridge_p[0], "gain": ridge_p[1], "median_rank": ridge_p[2]}, "runs": {str(k): v for k, v in results.items()}, "summary": {str(k): v for k, v in summ.items()}, "verdict": verdict, "longer_training": longer}, open(out, "w", encoding="utf8"), indent=1)
    print(f"\n  total time {time.time() - t0:.0f} s; saved {os.path.relpath(out, ROOT)}")
    print("=" * 120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
