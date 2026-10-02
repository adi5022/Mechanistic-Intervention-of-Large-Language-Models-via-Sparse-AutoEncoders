"""
Check a strength-cache folder (written by tools/build_strength_cache.py).

    .venv\\Scripts\\python.exe tools/verify_strength_cache.py --expected 725
    .venv\\Scripts\\python.exe tools/verify_strength_cache.py --expected 1450 --dir outputs/strength_cache

Loads every .pt file and checks: it opens, the residual and token lengths agree, every per-candidate array has the
same length, there is at least one candidate, no NaN/inf, and the case_id matches the file name.
With --split-file it also reports which split sets each file belongs to and which expected prompts are missing.
Exit code 0 only if the count matches --expected and no file has a problem.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_CAND = ["cand_ids", "cand_act_max", "cand_act_last", "cand_n_pos", "cand_pos_max", "rm_target_prob", "rm_blocker_prob",
            "dt", "db", "align_target", "align_blocker", "dec_norm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "outputs", "strength_cache"))
    ap.add_argument("--expected", type=int, default=0, help="number of files this folder should contain")
    ap.add_argument("--split-file", default=None, help="optional: also report missing prompts of the 1,450-prompt plan")
    ap.add_argument("--train-n", type=int, default=1000)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.pt")))
    problems, sets, seconds, mismatches = [], Counter(), [], 0
    for f in files:
        cid = os.path.splitext(os.path.basename(f))[0]
        try:
            e = torch.load(f, weights_only=False)
            K = e["cand_ids"].numel()
            assert str(e["case_id"]) == cid, "case_id does not match the file name"
            assert K > 0, "no candidate features"
            assert e["resid8"].shape[0] == e["tokens"].shape[0], "residual and token lengths differ"
            assert e["cand_acts"].shape == (e["tokens"].shape[0], K), "cand_acts has the wrong shape"
            for k in PER_CAND:
                assert e[k].shape[0] == K, f"{k} has the wrong length"
            for k in PER_CAND[1:] + ["cand_acts", "resid8"]:
                assert torch.isfinite(e[k].float()).all(), f"{k} contains NaN or inf"
            assert all(isinstance(v, float) and v == v for v in e["summary"].values()), "summary has a bad value"
            sets[e.get("set", "?")] += 1
            seconds.append(e["seconds"])
        except Exception as ex:                                  # report every bad file, do not stop at the first
            problems.append((cid, repr(ex)[:120]))
    print(f"Folder: {a.dir}")
    print(f".pt files found: {len(files)}" + (f" (expected {a.expected})" if a.expected else ""))
    print(f"Files with a problem: {len(problems)}")
    for cid, msg in problems[:20]:
        print(f"   case {cid}: {msg}")
    if seconds:
        print(f"Per-prompt compute time recorded in the files: mean {sum(seconds) / len(seconds):.2f} s")
    print("Files by set:", dict(sets))
    metas = glob.glob(os.path.join(a.dir, "_run_*.json"))
    for m in metas:
        d = json.load(open(m))
        print(f"Run info {os.path.basename(m)}: gpu={d.get('gpu')} torch={d.get('torch')} prompts={d.get('prompts_computed')} "
              f"seconds={d.get('seconds')} rank_mismatches={d.get('baseline_rank_mismatches_vs_hard_set')}")
    if a.split_file:
        sp = json.load(open(a.split_file))
        want = set(sp["val"]) | set(sp["test_seen"]) | set(sp["test_unseen"]) | set(sp["train"][:a.train_n])
        have = {int(os.path.splitext(os.path.basename(f))[0]) for f in files}
        print(f"Of the {len(want)} prompts in the plan, {len(want & have)} are present, {len(want - have)} missing, "
              f"{len(have - want)} extra.")
    ok = not problems and (not a.expected or len(files) == a.expected)
    print("RESULT:", "OK" if ok else "NOT OK")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
