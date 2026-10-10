"""
Post-hoc check after D10 (Entry 33 section 4.7): does the map memorise its training targets? Top-1 on the 289 TRAINING tuned prompts (target words the map was
trained on) against the top-1 on the 88 primary TEST records (unseen target words), for the ridge map and the saved D10 maps (seed 0; lambda 0, 0.1, 1.0).

    .venv\\Scripts\\python.exe tools/transfer/17_train_vs_test.py

Uses d8_items.pt (training texts), d7_train_edits.pt, d10_map_lam*_s0.pt and d10_rank1_term.json; about one minute. Exploratory (designed after seeing D10).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.transfer.models import load_model, pick_device
from src.transfer.om_batched import eval_ranks, pack
from src.transfer.runlog import tee_to

O = os.path.join(ROOT, "outputs", "transfer")


def main():
    print(f"log: {os.path.relpath(tee_to('17_train_vs_test'), ROOT)}", flush=True)
    dev = pick_device(None)
    c = torch.load(os.path.join(O, "d8_items.pt"), weights_only=False)
    bench = json.load(open(os.path.join(O, "d_benchmark.json"), encoding="utf8"))
    edits = torch.load(os.path.join(O, "d_edits.pt"), weights_only=False)
    bench = [r for r in bench if r["case_id"] in edits]
    tedits = torch.load(os.path.join(O, "d7_train_edits.pt"), weights_only=False)
    pool = json.load(open(os.path.join(O, "d7_pool.json"), encoding="utf8"))
    held_words, held_subjects = {r["tid_new"] for r in bench}, {r["subject"] for r in bench}
    keep = [r for r in pool if r["tid_new"] not in held_words and r["subject"] not in held_subjects][:800]
    good = [r for r in keep if tedits[r["case_id"]]["rank"] == 1]
    assert len(good) == c["n_good"]
    medium = load_model("gpt2-medium", dev)
    for p_ in medium.parameters():
        p_.requires_grad_(False)
    items = []
    for i in range(0, len(c["train"]), 5):                                  # one tuned prompt per training record
        it = {k: v for k, v in c["train"][i].items() if k != "goal"}
        it["tid"], it["base_rank"] = good[i // 5]["tid_new"], 1
        items.append(it)
    S = pack(items, dev)
    W0 = torch.load(os.path.join(O, "maps_gpt2_to_gpt2-medium.pt"))["maps"]["s2m_L8_L16"]["W"].to(dev)
    d10 = json.load(open(os.path.join(O, "d10_rank1_term.json"), encoding="utf8"))
    print("top-1 on the 289 TRAINING tuned prompts (target words the map was trained on) against top-1 on the 88 TEST records (unseen target words):")
    print(f"{'map':<28}{'training top-1':>16}{'test top-1 (primary)':>24}")
    r0 = eval_ranks(medium, S, W0, 2.0, 16)
    print(f"{'ridge map, dose 2.0':<28}{100 * float((r0 == 1).float().mean()):>15.0f}%{'8%':>24}")
    for lam in (0.0, 0.1, 1.0):
        m = torch.load(os.path.join(O, f"d10_map_lam{lam}_s0.pt"), weights_only=False)
        r = eval_ranks(medium, S, m["W"].to(dev), m["dose"], 16)
        run = [x for x in d10["runs"] if x["lambda"] == lam and x["seed"] == 0][0]
        row = [x for x in run["rows"] if x["dose"] == run["dose"]][0]
        print(f"{'lambda ' + str(lam) + ', dose ' + str(m['dose']):<28}{100 * float((r == 1).float().mean()):>15.0f}%{100 * row['real']['prim'][0]:>23.0f}%")


if __name__ == "__main__":
    main()
