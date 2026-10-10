"""
Steps D1 and D2 of docs/cross_model_transfer/PLAN.md: build the benchmark and run the gradient-descent edits in GPT-2 small.

    .venv\\Scripts\\python.exe tools/transfer/05_export_edits.py --quick     # 4 records
    .venv\\Scripts\\python.exe tools/transfer/05_export_edits.py             # 50 dev + 150 test records, 2 edits each (about 25 minutes)

For every record, two edits are tuned in small with the project's unchanged gradient-descent tool (200 candidate features, 100 steps,
all positions):
  real edit    toward the counterfactual target (" English" for "The mother tongue of Danielle Darrieux is")
  random edit  toward a random word whose rank on the prompt in small is close to the real target's start rank (0.7x to 1.4x; a control
               that asks whether transfer depends on the target being a plausible counterfactual or works for any word)
Only the multipliers are stored (feature ids and a_k, with m_k = 1 + a_k); the change they make at any prompt is recomputed from them,
which is what lets the edit be re-applied to reworded prompts in step D3. The run can be stopped and resumed (--resume is the default).
Writes outputs/transfer/d_benchmark.json and d_edits.pt.
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

from src.editing import build_clean_context
from src.gradient_editing import run_gradient_descent_edit
from src.sae_utils import load_sae_for_layer
from src.transfer.benchmark import build_benchmark
from src.transfer.models import load_model, pick_device

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
HOOK = "blocks.8.hook_resid_pre"


def pick_random_word(small, probs, tid_new, tid_true, start_rank, rng):
    """A random alphabetic single-token word (leading space) whose rank on this prompt is within 0.7x to 1.4x of start_rank."""
    order = torch.argsort(probs, descending=True)
    rank_of = torch.empty_like(order)
    rank_of[order] = torch.arange(1, len(order) + 1, device=order.device)
    names = small.tokenizer.convert_ids_to_tokens(list(range(len(order))))
    ok = torch.tensor([bool(n.startswith("Ġ") and n[1:].isalpha() and len(n) > 3) for n in names], device=order.device)
    for lo, hi in ((0.7, 1.4), (0.5, 2.0), (0.3, 4.0)):
        cand = ok & (rank_of >= max(2, lo * start_rank)) & (rank_of <= hi * start_rank)
        cand[tid_new] = False
        cand[tid_true] = False
        ids = cand.nonzero().flatten().tolist()
        if ids:
            return rng.choice(ids), int(rank_of[rng.choice(ids)])
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dev", type=int, default=50)
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.quick:
        a.dev, a.test = 2, 2
    suffix = "_quick" if a.quick else ""
    dev = pick_device(a.device)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    small = load_model("gpt2", dev)
    bench_path = os.path.join(OUT_DIR, f"d_benchmark{suffix}.json")
    if os.path.exists(bench_path):
        bench = json.load(open(bench_path, encoding="utf8"))
        print(f"benchmark loaded from {bench_path}: {len(bench)} records", flush=True)
    else:
        medium = load_model("gpt2-medium", dev)
        pilot = {r["case_id"] for r in json.load(open(os.path.join(OUT_DIR, "d_pilot.json"), encoding="utf8"))["records"]}
        bench = build_benchmark(small, medium, a.dev, a.test, exclude_case_ids=pilot)
        json.dump(bench, open(bench_path, "w", encoding="utf8"), indent=1)
        del medium
        torch.cuda.empty_cache() if dev.startswith("cuda") else None
        print(f"benchmark built: {sum(r['split'] == 'dev' for r in bench)} dev + {sum(r['split'] == 'test' for r in bench)} test records", flush=True)

    sae = load_sae_for_layer(8, dev)
    edits_path = os.path.join(OUT_DIR, f"d_edits{suffix}.pt")
    edits = torch.load(edits_path, weights_only=False) if os.path.exists(edits_path) else {}
    print(f"{len(edits)} records already done", flush=True)

    for n, r in enumerate(bench):
        if r["case_id"] in edits:
            continue
        rng = random.Random(1000 + r["index"])
        ctx = build_clean_context(small, sae, r["prompt"], r["tid_new"])
        out = {}
        real = run_gradient_descent_edit(small, sae, ctx, r["tid_new"], r["prompt"], 8, HOOK, top_n=200, steps=a.steps)
        out["real"] = {"fids": real["fids"], "a": real["a"], "start_rank": ctx.clean_rank, "rank": real["real_path"]["rank"],
                       "kl": real["real_path"]["kl"], "edit_size_last": real["edit_size_frac_norm"]}
        w, w_rank = pick_random_word(small, ctx.clean_probs, r["tid_new"], r["tid_true"], ctx.clean_rank, rng)
        if w is not None:
            ctx_w = build_clean_context(small, sae, r["prompt"], w)
            rnd = run_gradient_descent_edit(small, sae, ctx_w, w, r["prompt"], 8, HOOK, top_n=200, steps=a.steps)
            out["random"] = {"tid": w, "word": small.tokenizer.decode([w]), "fids": rnd["fids"], "a": rnd["a"], "start_rank": ctx_w.clean_rank,
                             "rank": rnd["real_path"]["rank"], "kl": rnd["real_path"]["kl"], "edit_size_last": rnd["edit_size_frac_norm"]}
        edits[r["case_id"]] = out
        done = len(edits)
        if done % 5 == 0 or done == len(bench):
            torch.save(edits, edits_path)
        el = time.time() - t0
        print(f"   {done:>3}/{len(bench)} [{r['split']}] {r['prompt']!r} -> {r['target_new']!r}: small rank {out['real']['start_rank']} -> {out['real']['rank']} (KL {out['real']['kl']:.2f})"
              + (f"; random word {out['random']['word']!r} rank {out['random']['start_rank']} -> {out['random']['rank']}" if "random" in out else "; no random word found")
              + f"   [{el:.0f} s]", flush=True)
    torch.save(edits, edits_path)

    ok = [e["real"]["rank"] == 1 for e in edits.values()]
    okr = [e["random"]["rank"] == 1 for e in edits.values() if "random" in e]
    print("\n" + "=" * 100)
    print("RESULT  steps D1 and D2 (benchmark and edits in small)" + ("  [quick]" if a.quick else ""))
    print(f"  records {len(edits)} ({sum(r['split'] == 'dev' for r in bench)} dev, {sum(r['split'] == 'test' for r in bench)} test); "
          f"edit toward the counterfactual target reaches rank 1 in small on {sum(ok)} ({sum(ok) / len(ok):.0%}); "
          f"edit toward a rank-matched random word reaches rank 1 on {sum(okr)} of {len(okr)} ({sum(okr) / max(len(okr), 1):.0%})")
    print(f"  total time {time.time() - t0:.0f} s; saved {os.path.relpath(edits_path, ROOT)}, {os.path.relpath(bench_path, ROOT)}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
