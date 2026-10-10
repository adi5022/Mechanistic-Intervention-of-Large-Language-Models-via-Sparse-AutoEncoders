"""
Step S2 of docs/cross_model_transfer/PLAN.md: is D6's rewordings lift larger than the lift a push toward a random word gives? (a significance test of an existing result; the rule is in the plan,
written before this tool was built.)

    .venv\\Scripts\\python.exe tools/transfer/26_d6_rewordings_test.py              # about 1 minute on the GPU, a few minutes on the CPU (--device cpu)

Per test record that has a random-word arm: score_real = share of the rewordings where D6's real-target dials make the new answer beat the true one, minus the same share in unchanged medium;
score_rand = the same for the random-word dials and the random word, minus that word's own share in unchanged medium (computed here, one forward pass per rewording). One-sided paired Wilcoxon
of score_real against score_rand, the mean difference with a bootstrap 95% interval, and counts. Saves outputs/transfer/s2_d6_rewordings_test.json.
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

from src.transfer.export_eval import plain_logits
from src.transfer.models import load_model, pick_device
from src.transfer.runlog import tee_to

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
P_LINE = 0.01


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    print(f"log: {os.path.relpath(tee_to('26_d6_rewordings_test'), ROOT)}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = pick_device(a.device)
    t0 = time.time()
    bench = json.load(open(os.path.join(OUT_DIR, "d_benchmark.json"), encoding="utf8"))
    edits = torch.load(os.path.join(OUT_DIR, "d_edits.pt"), weights_only=False)
    res2 = torch.load(os.path.join(OUT_DIR, "d6_eval.pt"), weights_only=False)["res2"]
    test = [r for r in bench if r["split"] == "test" and r["case_id"] in res2 and r["case_id"] in edits]
    medium = load_model("gpt2-medium", dev)
    rows = []
    for r in test:
        cid = r["case_id"]
        e = edits[cid]
        if "random" not in e:
            continue
        prow = res2[cid]["paraphrase"]
        assert len(prow) == len(r["paraphrases"])
        tw, tt = e["random"]["tid"], r["tid_true"]
        base_rand = []
        for text in r["paraphrases"]:
            lp = torch.log_softmax(plain_logits(medium, medium.to_tokens(text)), -1)
            base_rand.append(float(lp[tw] > lp[tt]))
        real_lift = mean([float(row["b_aware"]["es"][0]) - float(row["base"]["es"][0]) for row in prow])
        rand_lift = mean([float(row["random_word_b"]["es"][0]) - b for row, b in zip(prow, base_rand)])
        rows.append({"case_id": cid, "primary": e["real"]["rank"] == 1, "real_lift": real_lift, "rand_lift": rand_lift, "real_after": mean([float(row["b_aware"]["es"][0]) for row in prow]),
                     "real_before": mean([float(row["base"]["es"][0]) for row in prow]), "rand_after": mean([float(row["random_word_b"]["es"][0]) for row in prow]), "rand_before": mean(base_rand)})
    out = {}
    print("\n" + "=" * 110)
    print("RESULT  step S2 (D6 rewordings: real target against a random word, each against its own unchanged level)")
    for label, sub in (("PRIMARY (original edit reached rank 1 in small)", [x for x in rows if x["primary"]]), ("ALL test records with a random-word arm", rows)):
        d = [x["real_lift"] - x["rand_lift"] for x in sub]
        try:
            p = float(wilcoxon([x["real_lift"] for x in sub], [x["rand_lift"] for x in sub], alternative="greater").pvalue)
        except ValueError:
            p = float("nan")
        g = random.Random(0)
        boots = sorted(mean([d[g.randrange(len(d))] for _ in d]) for _ in range(10000))
        lo, hi = boots[250], boots[9750]
        more, same, less = sum(v > 1e-9 for v in d), sum(abs(v) <= 1e-9 for v in d), sum(v < -1e-9 for v in d)
        reading = "p < 0.01: the real-target lift is larger than a push toward a random word gives (more than a push, with D6's dials)" if p < P_LINE else "NOT SHOWN at this sample size"
        o = {"n": len(sub), "mean_real_before": mean([x["real_before"] for x in sub]), "mean_real_after": mean([x["real_after"] for x in sub]), "mean_rand_before": mean([x["rand_before"] for x in sub]),
             "mean_rand_after": mean([x["rand_after"] for x in sub]), "mean_real_lift": mean([x["real_lift"] for x in sub]), "mean_rand_lift": mean([x["rand_lift"] for x in sub]), "mean_diff": mean(d),
             "ci95": [lo, hi], "records_real_larger": more, "records_equal": same, "records_rand_larger": less, "p_wilcoxon_one_sided": p, "reading": reading}
        out[label] = o
        print(f"\n  ---- {label}: {len(sub)} records ----")
        print(f"    real target: rewordings where the new answer beats the true one {100 * o['mean_real_before']:.1f}% unchanged -> {100 * o['mean_real_after']:.1f}% with D6's dials (lift {100 * o['mean_real_lift']:.1f} points)")
        print(f"    random word: {100 * o['mean_rand_before']:.1f}% unchanged -> {100 * o['mean_rand_after']:.1f}% with its dials (lift {100 * o['mean_rand_lift']:.1f} points)")
        print(f"    difference of the lifts {100 * o['mean_diff']:+.1f} points (bootstrap 95% interval {100 * lo:+.1f} to {100 * hi:+.1f}); records with the real lift larger / equal / smaller: {more} / {same} / {less}; one-sided Wilcoxon p = {p:.1e}")
        print(f"    reading (fixed in the plan before the run): {reading}")
    out["rows"] = rows
    path = os.path.join(OUT_DIR, "s2_d6_rewordings_test.json")
    json.dump(out, open(path, "w", encoding="utf8"), indent=1, default=float)
    print(f"\n  total time {time.time() - t0:.0f} s; saved {os.path.relpath(path, ROOT)}")
    print("=" * 110)


if __name__ == "__main__":
    main()
