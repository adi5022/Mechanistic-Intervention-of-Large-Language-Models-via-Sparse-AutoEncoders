"""
Step 4 of the learned-strength study: strength tables (docs/Research_Journal/23.md, Phase 3).

WHY: the strict safety filter decides, for every candidate feature, whether it is "safe" at the set strength. So WHICH
features are allowed depends on the mute and boost strengths. Training a strength network needs to know the allowed
features at ANY strength it proposes, not only at 0.6 / 0.5. This script records, once per prompt, what the filter
would see at a grid of strengths.

WHAT: for every prompt in the cache and every candidate feature (the same 200, in the same order as the cache), the
candidate is applied ALONE and the target's probability and rank are recorded:
    mute side:   scale = 1 - mu     for mu   in MUTE_GRID   (0.2 0.4 0.6 0.8 1.0)
    boost side:  scale = 1 + beta   for beta in BOOST_GRID  (0.25 0.5 1.0 1.5 2.0)
This is exactly the call the sweep's strict filter makes in round 0 (batched_ablation_probs_and_ranks, no features
applied beforehand). The strict rule (accept if target prob drops by at most 1e-6 and the rank does not worsen) is
NOT applied here; the raw numbers are stored, so the strict mask, the tolerance variants and the overlap rule can all
be derived later.

Run (needs outputs/strength_cache; resumable; one file per prompt: outputs/strength_tables/<case_id>.pt):
    .venv\\Scripts\\python.exe tools/build_strength_tables.py --shard 0,1,2,3,4/7      (faster machine, RTX 4050)
    .venv\\Scripts\\python.exe tools/build_strength_tables.py --shard 5,6/7            (slower machine)
    .venv\\Scripts\\python.exe tools/build_strength_tables.py                          (everything on one machine)
Check afterwards:
    .venv\\Scripts\\python.exe tools/build_strength_tables.py --verify --expected 1450
Quick test: add --limit 3 --out-dir outputs/strength_tables_test
"""
import argparse
import glob
import json
import os
import platform
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MUTE_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]
BOOST_GRID = [0.25, 0.5, 1.0, 1.5, 2.0]
HOOK = "blocks.8.hook_resid_pre"


def table_path(out_dir, case_id):
    return os.path.join(out_dir, f"{case_id}.pt")


def build_one(model, sae, cache_entry):
    """Strength table for one prompt. Candidate order = the cache's candidate order."""
    from src.batched_eval import batched_ablation_probs_and_ranks
    tokens = model.to_tokens(cache_entry["prompt"])
    fids = [int(f) for f in cache_entry["cand_ids"]]
    tid = int(cache_entry["target_id"])
    out = {"case_id": cache_entry["case_id"], "cand_ids": cache_entry["cand_ids"].clone(), "target_id": tid,
           "clean_prob": float(cache_entry["baseline_prob"]), "clean_rank": int(cache_entry["baseline_rank"]),
           "mute_grid": MUTE_GRID, "boost_grid": BOOST_GRID}
    for side, grid, sign in (("mute", MUTE_GRID, -1.0), ("boost", BOOST_GRID, 1.0)):
        probs, ranks = [], []
        for s in grid:
            p, r = batched_ablation_probs_and_ranks(model, sae, tokens, fids, 1.0 + sign * s, tid, HOOK)
            probs.append(p.detach().float().cpu())
            ranks.append(r.detach().cpu().to(torch.int32))
        out[f"{side}_prob"] = torch.stack(probs)        # [n_strengths, K] target probability with that candidate alone at that strength
        out[f"{side}_rank"] = torch.stack(ranks)        # [n_strengths, K] target rank
    return out


def sanity(model, sae, cache_entry, tbl, tol=2e-4):
    """Compare the table with (a) the cache's full-removal probability and (b) a one-feature-at-a-time hook at mu = 0.6."""
    from src.hooks import make_ablation_hook
    k = 0
    a = float((tbl["mute_prob"][MUTE_GRID.index(1.0), :5] - cache_entry["rm_target_prob"][:5]).abs().max())
    tokens = model.to_tokens(cache_entry["prompt"])
    model.reset_hooks()
    with torch.no_grad():
        model.add_hook(HOOK, make_ablation_hook(int(tbl["cand_ids"][k]), sae, 0.6))
        single = float(torch.softmax(model(tokens)[0, -1], dim=-1)[tbl["target_id"]])
    model.reset_hooks()
    b = abs(single - float(tbl["mute_prob"][MUTE_GRID.index(0.6), k]))
    return a < tol and b < tol, a, b


def verify(out_dir, expected):
    files = sorted(glob.glob(os.path.join(out_dir, "*.pt")))
    bad, mute_pass, boost_pass, n = [], [0.0] * len(MUTE_GRID), [0.0] * len(BOOST_GRID), 0
    for f in files:
        try:
            t = torch.load(f, weights_only=False)
            K = t["cand_ids"].numel()
            assert t["mute_prob"].shape == (len(MUTE_GRID), K) and t["boost_prob"].shape == (len(BOOST_GRID), K)
            assert t["mute_rank"].shape == t["mute_prob"].shape and t["boost_rank"].shape == t["boost_prob"].shape
            assert torch.isfinite(t["mute_prob"]).all() and torch.isfinite(t["boost_prob"]).all()
            for side, pas in (("mute", mute_pass), ("boost", boost_pass)):
                ok = ((t[f"{side}_prob"] - t["clean_prob"]) >= -1e-6) & (t[f"{side}_rank"] <= t["clean_rank"])   # the strict rule
                for i in range(ok.shape[0]):
                    pas[i] += float(ok[i].sum())
            n += 1
        except Exception as ex:
            bad.append((os.path.basename(f), repr(ex)[:100]))
    print(f"Folder: {out_dir}\n.pt files: {len(files)}" + (f" (expected {expected})" if expected else "") + f"\nProblems: {len(bad)}")
    for b in bad[:20]:
        print("  ", b)
    if n:
        print("\nAverage number of candidates (of 200) that PASS the strict filter, by strength:")
        print("  mute  mu   :", "  ".join(f"{m}: {mute_pass[i] / n:5.1f}" for i, m in enumerate(MUTE_GRID)))
        print("  boost beta :", "  ".join(f"{b}: {boost_pass[i] / n:5.1f}" for i, b in enumerate(BOOST_GRID)))
    ok = not bad and (not expected or len(files) == expected)
    print("RESULT:", "OK" if ok else "NOT OK")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "outputs", "strength_cache"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "strength_tables"))
    ap.add_argument("--shard", default=None, help="i/n, or several slices for a faster machine: 0,1,2,3,4/7")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--expected", type=int, default=0)
    a = ap.parse_args()
    if a.verify:
        verify(a.out_dir, a.expected)

    from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
    files = sorted(glob.glob(os.path.join(a.cache_dir, "*.pt")), key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    if not files:
        sys.exit(f"No cache files in {a.cache_dir}. Build and merge the cache first.")
    total_all = len(files)
    if a.shard:
        i_, n_ = a.shard.split("/")
        slices, n_ = {int(x) for x in i_.split(",")}, int(n_)
        files = [f for k, f in enumerate(files) if k % n_ in slices]
    if a.limit:
        files = files[:a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    todo = [f for f in files if not os.path.exists(table_path(a.out_dir, os.path.splitext(os.path.basename(f))[0]))]
    print(f"{total_all} cache files | this run's share: {len(files)} | already done: {len(files) - len(todo)} | to compute: {len(todo)}", flush=True)
    if not todo:
        print("Nothing to do.")
        return
    device = get_default_device()
    model = load_base_model(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sae = load_sae_for_layer(layer=8)
    print(f"Device: {device} | torch {torch.__version__} | {platform.node()}", flush=True)

    t0, done, checked = time.time(), 0, False
    for f in todo:
        cid = os.path.splitext(os.path.basename(f))[0]
        entry = torch.load(f, weights_only=False)
        tbl = build_one(model, sae, entry)
        if not checked:
            ok, a1, b1 = sanity(model, sae, entry, tbl)
            print(f"Sanity check: table vs cache full-removal diff {a1:.1e}; table vs one-at-a-time hook (mu 0.6) diff {b1:.1e}: "
                  f"{'OK' if ok else 'MISMATCH'}", flush=True)
            if not ok:
                sys.exit("Table does not match the reference computation; stopping so bad data is not written.")
            checked = True
        path = table_path(a.out_dir, cid)
        torch.save(tbl, path + ".tmp")
        os.replace(path + ".tmp", path)
        done += 1
        if done % 25 == 0 or done == len(todo):
            el = time.time() - t0
            print(f"[{done}/{len(todo)}] {done / el:.2f} prompts/s | elapsed {el / 60:.1f} min | ETA {(len(todo) - done) * el / done / 60:.1f} min", flush=True)
    meta = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "shard": a.shard, "prompts_computed": done, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(a.out_dir, f"_run_{platform.node()}_{(a.shard or 'all').replace('/', 'of').replace(',', '-')}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"\nDone: {done} prompts in {meta['seconds']} s ({meta['seconds'] / done:.2f} s per prompt). Files in {a.out_dir}")


if __name__ == "__main__":
    main()
