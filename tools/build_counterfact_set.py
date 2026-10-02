"""
Step 1 of the learned-strength study (docs/Research_Journal/23.md, Phase 1).

Turns datasets/counterfact.json (21,919 records) into the list of prompts that GPT-2 small gets "wrong" in the
project's sense: the TRUE answer is not the model's top prediction.

For every record:
  1. prompt = template.format(subject); target = " " + target_true (leading space)
  2. keep only if the target is ONE GPT-2 token (the sweep scores single tokens)
  3. run the clean model (batched, no editing) and record the target's rank and probability
  4. keep if MIN_RANK <= rank <= MAX_RANK (default 2 to 1000)
     rank 1  -> nothing to fix;  rank > 1000 -> treated as "not known"

Outputs
  data/counterfact_hard_set.json          the kept prompts (what later steps read)
  data/counterfact_hard_set_summary.json  the yield numbers (also printed)
  outputs/counterfact_all_ranks.json      rank of EVERY single-token record, so another rank band can be chosen
                                          later without re-running (not committed)

Run from the project root:
    .venv\\Scripts\\python.exe tools/build_counterfact_set.py
Options: --limit N (first N records, for a quick test)  --batch-size 64  --min-rank 2  --max-rank 1000

Batching: prompts are sorted by length and right-padded. GPT-2 is causal, so padding after the real tokens cannot
change earlier positions; each prompt's logits are read at its own last real token. A self-check compares the
batched ranks with one-prompt-at-a-time ranks on a small sample before the full pass.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.sae_utils import load_base_model, get_default_device  # noqa: E402

MAX_PROMPT_TOKENS = 64
BANDS = [(2, 5), (6, 20), (21, 100), (101, 1000)]
PAD_ID = 50256   # GPT-2 <|endoftext|>; padding sits after the real tokens and is never read


def ranks_for(model, token_lists, target_ids, device, batch_size):
    """Rank and probability of each target at the last real position of each prompt (batched, right-padded)."""
    n = len(token_lists)
    order = sorted(range(n), key=lambda i: len(token_lists[i]))
    ranks, probs, top1 = [None] * n, [None] * n, [None] * n
    top1_prob = [None] * n
    with torch.no_grad():
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            lens = [len(token_lists[i]) for i in idx]
            width = max(lens)
            batch = torch.full((len(idx), width), PAD_ID, dtype=torch.long, device=device)
            for r, i in enumerate(idx):
                batch[r, :lens[r]] = torch.tensor(token_lists[i], device=device)
            logits = model(batch)                                   # [B, width, vocab]
            last = logits[torch.arange(len(idx), device=device), torch.tensor(lens, device=device) - 1]   # [B, vocab]
            p = torch.softmax(last, dim=-1)
            tgt = torch.tensor([target_ids[i] for i in idx], device=device)
            tp = p[torch.arange(len(idx), device=device), tgt]
            rk = (p > tp.unsqueeze(1)).sum(dim=1) + 1
            tp1, t1 = p.max(dim=1)
            for r, i in enumerate(idx):
                ranks[i] = int(rk[r])
                probs[i] = float(tp[r])
                top1[i] = int(t1[r])
                top1_prob[i] = float(tp1[r])
            del logits, last, p
    return ranks, probs, top1, top1_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "datasets", "counterfact.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "counterfact_hard_set.json"))
    ap.add_argument("--all-ranks-out", default=os.path.join(ROOT, "outputs", "counterfact_all_ranks.json"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N records (quick test)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--min-rank", type=int, default=2)
    ap.add_argument("--max-rank", type=int, default=1000)
    a = ap.parse_args()

    t0 = time.time()
    with open(a.src, encoding="utf-8") as f:
        records = json.load(f)
    if a.limit:
        records = records[:a.limit]
    print(f"Loaded {len(records)} records from {a.src}", flush=True)

    device = get_default_device()
    model = load_base_model(device)
    model.eval()
    summary = {"records": len(records)}

    # ---- 1. build prompts, check the target is a single token ----
    rows = []
    n_multi = n_long = n_dupe = 0
    seen = set()
    for rec in records:
        rw = rec["requested_rewrite"]
        try:
            prompt = rw["prompt"].format(rw["subject"]).strip()
        except (KeyError, IndexError):
            continue
        target = " " + rw["target_true"]["str"].strip()
        tgt_ids = model.to_tokens(target, prepend_bos=False).reshape(-1).tolist()
        if len(tgt_ids) != 1:
            n_multi += 1
            continue
        toks = model.to_tokens(prompt).reshape(-1).tolist()           # includes BOS
        if len(toks) > MAX_PROMPT_TOKENS:
            n_long += 1
            continue
        key = (prompt, target)
        if key in seen:
            n_dupe += 1
            continue
        seen.add(key)
        rows.append({"case_id": rec.get("case_id"), "relation_id": rw["relation_id"], "subject": rw["subject"],
                     "template": rw["prompt"], "prompt": prompt, "target": target.strip(), "target_token_id": tgt_ids[0],
                     "n_tokens": len(toks), "_toks": toks,
                     "paraphrase_prompts": rec.get("paraphrase_prompts", []),
                     "neighborhood_prompts": rec.get("neighborhood_prompts", [])})
    summary.update({"multi_token_target_dropped": n_multi, "too_long_dropped": n_long, "duplicates_dropped": n_dupe,
                    "single_token_prompts": len(rows)})
    print(f"Single-token targets: {len(rows)} usable | multi-token dropped {n_multi} | too long {n_long} | duplicates {n_dupe}", flush=True)

    # ---- 2. self-check: batched ranks must equal one-prompt-at-a-time ranks ----
    probe = rows[:min(24, len(rows))]
    rb, _, _, _ = ranks_for(model, [r["_toks"] for r in probe], [r["target_token_id"] for r in probe], device, batch_size=8)
    rs = []
    with torch.no_grad():
        for r in probe:
            lg = model(torch.tensor([r["_toks"]], device=device))[0, -1]
            rs.append(int((torch.softmax(lg, -1) > torch.softmax(lg, -1)[r["target_token_id"]]).sum()) + 1)
    mism = sum(1 for x, y in zip(rb, rs) if x != y)
    print(f"Self-check on {len(probe)} prompts: {mism} rank mismatches between batched and one-by-one", flush=True)
    summary["selfcheck_prompts"] = len(probe)
    summary["selfcheck_mismatches"] = mism
    if mism:
        print("WARNING: batched and single ranks differ; do not trust the output until this is understood.", flush=True)

    # ---- 3. rank every prompt (batched) ----
    t1 = time.time()
    ranks, probs, top1, top1p = ranks_for(model, [r["_toks"] for r in rows], [r["target_token_id"] for r in rows],
                                          device, a.batch_size)
    print(f"Ranked {len(rows)} prompts in {time.time() - t1:.1f} s (batch size {a.batch_size})", flush=True)
    summary["ranking_seconds"] = round(time.time() - t1, 1)

    all_ranks = []
    kept = []
    for r, rk, pr, t1id, t1p in zip(rows, ranks, probs, top1, top1p):
        all_ranks.append({"case_id": r["case_id"], "relation_id": r["relation_id"], "rank": rk, "prob": pr})
        if a.min_rank <= rk <= a.max_rank:
            out = {k: v for k, v in r.items() if k != "_toks"}
            out.update({"baseline_rank": rk, "baseline_prob": pr,
                        "blocker_token_id": t1id, "blocker_token": model.to_string([t1id]), "blocker_prob": t1p})
            kept.append(out)

    # ---- 4. yield report ----
    n1 = sum(1 for x in ranks if x == 1)
    nbig = sum(1 for x in ranks if x > a.max_rank)
    per_rel = Counter(k["relation_id"] for k in kept)
    band_counts = {f"{lo}-{hi}": sum(1 for k in kept if lo <= k["baseline_rank"] <= hi) for lo, hi in BANDS}
    subjects = len({k["subject"] for k in kept})
    summary.update({"already_rank1": n1, f"rank_above_{a.max_rank}": nbig,
                    f"kept_rank_{a.min_rank}_to_{a.max_rank}": len(kept), "kept_distinct_subjects": subjects,
                    "kept_per_relation": dict(sorted(per_rel.items(), key=lambda kv: -kv[1])),
                    "kept_by_rank_band": band_counts,
                    "min_rank": a.min_rank, "max_rank": a.max_rank, "total_seconds": round(time.time() - t0, 1)})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(os.path.dirname(a.all_ranks_out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"source": "datasets/counterfact.json", "min_rank": a.min_rank, "max_rank": a.max_rank,
                   "prompts": kept}, f, ensure_ascii=False)
    with open(a.out.replace(".json", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    with open(a.all_ranks_out, "w", encoding="utf-8") as f:
        json.dump(all_ranks, f)

    print("\n=========== YIELD REPORT ===========")
    print(f"Records scanned:                          {len(records)}")
    print(f"One-token true target (usable):           {len(rows)}")
    print(f"  already rank 1 (nothing to fix):        {n1}")
    print(f"  rank above {a.max_rank} (treated as unknown):      {nbig}")
    print(f"  KEPT, rank {a.min_rank} to {a.max_rank}:                    {len(kept)}  ({subjects} distinct subjects)")
    print("Kept by starting rank:", band_counts)
    print(f"Kept per relation ({len(per_rel)} relations):")
    for rel, c in sorted(per_rel.items(), key=lambda kv: -kv[1]):
        print(f"   {rel:<8}{c:>6}")
    print(f"\nWrote {a.out}\nWrote {a.out.replace('.json', '_summary.json')}\nWrote {a.all_ranks_out}")
    print(f"Total time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
