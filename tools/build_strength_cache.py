"""
Step 3 of the learned-strength study: build the per-prompt cache (docs/Research_Journal/23.md, Phase 2 and Section 13).

Reads data/counterfact_split.json and data/counterfact_hard_set.json and, for every prompt in the chosen sets, runs
GPT-2 small + the layer-8 SAE once and saves one file per prompt: outputs/strength_cache/<case_id>.pt
(see src/strength_cache.py for exactly what is stored). This is DATA PREPARATION, not training.

Safe to stop and restart: prompts whose file already exists are skipped. Files are written atomically.

Work can be shared between machines: each machine runs a different shard (--shard i/n) and writes into its own
outputs/strength_cache folder; afterwards copy all the .pt files into one folder (file names never clash).

    one machine:   .venv\\Scripts\\python.exe tools/build_strength_cache.py
    two equal machines:  A: ... --shard 0/2        B: ... --shard 1/2
    machines 4x apart:   fast: ... --shard 0,1,2,3/5     slow: ... --shard 4/5
    quick test:    ... tools/build_strength_cache.py --limit 5 --out-dir outputs/strength_cache_test

Default sets: val, test_seen, test_unseen and the first 1000 training prompts (--train-n). Training prompts are a
balanced, nested order, so later you can add --train-n 2000 and only the missing prompts are computed.
"""
import argparse
import json
import os
import platform
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device  # noqa: E402
from src.strength_cache import build_one, cache_path, sanity_check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-set", default=os.path.join(ROOT, "data", "counterfact_hard_set.json"))
    ap.add_argument("--split-file", default=os.path.join(ROOT, "data", "counterfact_split.json"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "strength_cache"))
    ap.add_argument("--train-n", type=int, default=1000, help="first N prompts of the balanced training order")
    ap.add_argument("--sets", default="val,test_seen,test_unseen,train", help="comma-separated: train,val,test_seen,test_unseen")
    ap.add_argument("--shard", default=None, help="i/n: this machine handles every n-th prompt starting at i. "
                    "Several slices for a faster machine: 0,1,2,3/5 (and 4/5 on the slower one)")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts of this shard (quick test)")
    a = ap.parse_args()

    for path in (a.hard_set, a.split_file):
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run tools/build_counterfact_set.py and tools/split_counterfact_set.py first.")
    with open(a.hard_set, encoding="utf-8") as f:
        by_id = {p["case_id"]: p for p in json.load(f)["prompts"]}
    with open(a.split_file, encoding="utf-8") as f:
        split = json.load(f)

    work = []
    for name in [s.strip() for s in a.sets.split(",") if s.strip()]:
        ids = split[name][:a.train_n] if name == "train" else split[name]
        work += [(name, by_id[i]) for i in ids]
    total_all = len(work)
    if a.shard:
        i_, n_ = a.shard.split("/")
        slices, n_ = {int(x) for x in i_.split(",")}, int(n_)         # e.g. 0,1,2,3/5 = four of five slices (a faster machine)
        work = [w for k, w in enumerate(work) if k % n_ in slices]
    if a.limit:
        work = work[:a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    todo = [w for w in work if not os.path.exists(cache_path(a.out_dir, w[1]["case_id"]))]
    print(f"{total_all} prompts in the chosen sets | this run's share: {len(work)} | already done: {len(work) - len(todo)} "
          f"| to compute: {len(todo)}", flush=True)
    if not todo:
        print("Nothing to do.")
        return

    device = get_default_device()
    model = load_base_model(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sae = load_sae_for_layer(layer=a.layer)
    hook_name = getattr(sae.cfg, "hook_name", f"blocks.{a.layer}.hook_resid_pre")
    print(f"Device: {device} | torch {torch.__version__} | {platform.node()}", flush=True)

    t0 = time.time()
    done = rank_mismatch = 0
    checked = False
    for name, rec in todo:
        entry = build_one(model, sae, rec, hook_name, a.top_n)
        entry["set"] = name
        if entry["baseline_rank"] != rec["baseline_rank"]:
            rank_mismatch += 1                        # exact ties can differ by one place; anything large would be a bug
        if not checked:
            ok, diff = sanity_check(model, sae, entry, hook_name)
            print(f"Sanity check (batched vs one-at-a-time removal, first prompt): {'OK' if ok else 'MISMATCH'} (diff {diff:.2e})", flush=True)
            if not ok:
                sys.exit("Batched removal does not match single removal; stopping so bad data is not written.")
            checked = True
        path = cache_path(a.out_dir, rec["case_id"])
        torch.save(entry, path + ".tmp")
        os.replace(path + ".tmp", path)
        done += 1
        if done % 25 == 0 or done == len(todo):
            el = time.time() - t0
            rate = done / el
            print(f"[{done}/{len(todo)}] {rate:.2f} prompts/s | elapsed {el / 60:.1f} min | ETA {(len(todo) - done) / rate / 60:.1f} min", flush=True)

    meta = {"host": platform.node(), "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "shard": a.shard, "prompts_computed": done, "seconds": round(time.time() - t0, 1),
            "baseline_rank_mismatches_vs_hard_set": rank_mismatch, "layer": a.layer, "top_n": a.top_n}
    shard_tag = (a.shard or "all").replace("/", "of")
    with open(os.path.join(a.out_dir, f"_run_{platform.node()}_{shard_tag}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nDone: {done} prompts in {meta['seconds']} s ({meta['seconds'] / max(done, 1):.2f} s per prompt). "
          f"Rank mismatches vs the hard-set file: {rank_mismatch}.\nFiles are in {a.out_dir}")


if __name__ == "__main__":
    main()
