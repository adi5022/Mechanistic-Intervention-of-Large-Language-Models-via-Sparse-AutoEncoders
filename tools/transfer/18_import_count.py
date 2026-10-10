"""
Step E1 of docs/cross_model_transfer/PLAN.md (Import): how many true facts does GPT-2 medium know that GPT-2 small does not? A descriptive count.
The reading rule is written in the plan before this run (300 or more candidates: worth planning; fewer than 100: bigger donor or ROME-edited counterfactuals; between: decide).

    .venv\\Scripts\\python.exe tools/transfer/18_import_count.py        # about 3 minutes

Every CounterFact record whose true answer is a single token (with a leading space) is run through both models (no edit, no translator). "Right" = the true
answer is the model's top-1 next word. Candidates = medium right and small wrong. Saves outputs/transfer/e1_import_count.json (summary and the candidate list).
"""
import json
import os
import random
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

from src.editing import get_target_token_id
from src.transfer.benchmark import load_counterfact
from src.transfer.models import load_model, pick_device
from src.transfer.runlog import tee_to

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")


def run_model(model, prompts, true_ids, lengths, bs=128):
    """(rank of the true answer, probability of the true answer) for every prompt, batching prompts of equal token length (no padding)."""
    n = len(prompts)
    rank, prob = torch.zeros(n, dtype=torch.long), torch.zeros(n)
    groups = {}
    for i, L in enumerate(lengths):
        groups.setdefault(L, []).append(i)
    done = 0
    for L, idxs in sorted(groups.items()):
        for s in range(0, len(idxs), bs):
            chunk = idxs[s:s + bs]
            toks = model.to_tokens([prompts[i] for i in chunk])
            assert toks.shape[1] == L, (toks.shape, L)                 # the tokenizer already counts the start token; no padding
            with torch.no_grad():
                lg = model(toks)[:, -1].float()
            t = torch.tensor([true_ids[i] for i in chunk], device=lg.device)[:, None]
            rank[chunk] = ((lg > lg.gather(1, t)).sum(-1) + 1).cpu()
            prob[chunk] = F.softmax(lg, -1).gather(1, t)[:, 0].cpu()
            done += len(chunk)
        print(f"   {done}/{n} prompts", flush=True)
    return rank, prob


def main():
    print(f"log: {os.path.relpath(tee_to('18_import_count'), ROOT)}", flush=True)
    dev = pick_device(None)
    t0 = time.time()
    ds = load_counterfact()
    rome_dev = set(random.Random(0).sample(range(len(ds)), 100))
    pilot = {r["case_id"] for r in json.load(open(os.path.join(OUT_DIR, "d_pilot.json"), encoding="utf8"))["records"]}
    bench = {r["case_id"] for r in json.load(open(os.path.join(OUT_DIR, "d_benchmark.json"), encoding="utf8"))}
    small, medium = load_model("gpt2", dev), load_model("gpt2-medium", dev)
    recs = []
    for i, e in enumerate(ds):
        rw = e["requested_rewrite"]
        true = rw["target_true"]["str"]
        if len(small.tokenizer(" " + true, add_special_tokens=False)["input_ids"]) != 1:
            continue
        prompt = rw["prompt"].format(rw["subject"])
        recs.append({"i": i, "case_id": e["case_id"], "prompt": prompt, "subject": rw["subject"], "relation": rw.get("relation_id", "?"), "true": true,
                     "tid": get_target_token_id(small, " " + true), "n_tok": len(small.tokenizer(prompt)["input_ids"])})
    print(f"{len(ds)} CounterFact records, {len(recs)} with a single-token true answer", flush=True)
    prompts, tids, lens = [r["prompt"] for r in recs], [r["tid"] for r in recs], [r["n_tok"] for r in recs]
    print("small:", flush=True)
    rs, ps = run_model(small, prompts, tids, lens)
    print("medium:", flush=True)
    rm, pm = run_model(medium, prompts, tids, lens)
    for k, r in enumerate(recs):
        r.update({"small_rank": int(rs[k]), "medium_rank": int(rm[k]), "medium_prob": float(pm[k]), "small_prob": float(ps[k]),
                  "excluded": r["i"] in rome_dev or r["case_id"] in pilot, "in_benchmark": r["case_id"] in bench})

    def cats(rows):
        c = Counter()
        for r in rows:
            c[("medium right" if r["medium_rank"] == 1 else "medium wrong", "small right" if r["small_rank"] == 1 else "small wrong")] += 1
        return c

    def report(rows, label):
        c = cats(rows)
        n = len(rows)
        print(f"\n  {label}: {n} records")
        print(f"    both right                          {c[('medium right', 'small right')]:>6}  ({100 * c[('medium right', 'small right')] / n:.1f}%)")
        print(f"    medium right, small wrong (IMPORT)  {c[('medium right', 'small wrong')]:>6}  ({100 * c[('medium right', 'small wrong')] / n:.1f}%)")
        print(f"    small right, medium wrong           {c[('medium wrong', 'small right')]:>6}  ({100 * c[('medium wrong', 'small right')] / n:.1f}%)")
        print(f"    both wrong                          {c[('medium wrong', 'small wrong')]:>6}  ({100 * c[('medium wrong', 'small wrong')] / n:.1f}%)")
        return c

    print("\n" + "=" * 110)
    print("RESULT  step E1 (Import: what does medium know that small does not?)")
    report(recs, "ALL records with a single-token true answer")
    use = [r for r in recs if not r["excluded"]]
    c = report(use, "outside the ROME dev records and the pilot records (the counts used for the decision)")
    cand = [r for r in use if r["medium_rank"] == 1 and r["small_rank"] != 1]
    print(f"\n  IMPORT CANDIDATES (medium right, small wrong, outside ROME dev and pilot): {len(cand)}; of them in the Export benchmark: {sum(r['in_benchmark'] for r in cand)}")
    bins = [("rank 2", lambda x: x == 2), ("rank 3 to 5", lambda x: 3 <= x <= 5), ("rank 6 to 20", lambda x: 6 <= x <= 20), ("rank 21 to 100", lambda x: 21 <= x <= 100), ("rank over 100", lambda x: x > 100)]
    print("  how far small is from right (rank of the true answer in small):  " + ";  ".join(f"{n} {sum(f(r['small_rank']) for r in cand)}" for n, f in bins))
    mp = sorted(r["medium_prob"] for r in cand)
    if cand:
        print(f"  how sure medium is (its probability for the true answer): median {mp[len(mp) // 2]:.2f}; at least 0.3 on {sum(p >= 0.3 for p in mp)}; at least 0.5 on {sum(p >= 0.5 for p in mp)}")
    rel = Counter(r["relation"] for r in cand)
    print(f"  relations represented: {len(rel)}; the ten most common: " + ", ".join(f"{k} {v}" for k, v in rel.most_common(10)))
    tg = Counter(r["true"] for r in cand)
    print(f"  distinct true answers: {len(tg)}; the ten most common: " + ", ".join(f"{k!r} {v}" for k, v in tg.most_common(10)))
    print(f"  distinct subjects: {len({r['subject'] for r in cand})}")
    n = len(cand)
    verdict = "300 or more: IMPORT IS WORTH PLANNING IN FULL" if n >= 300 else ("fewer than 100: SWITCH TO A BIGGER DONOR OR ROME-EDITED COUNTERFACTUALS" if n < 100 else "between 100 and 299: DECIDE WITH THE AUTHOR")
    print(f"\n  reading rule (fixed in the plan before the run): {n} candidates -> {verdict}")
    out = {"n_records": len(ds), "n_single_token": len(recs), "n_outside_excluded": len(use), "n_candidates": n, "verdict": verdict,
           "categories_outside_excluded": {f"{a} / {b}": v for (a, b), v in c.items()},
           "small_rank_bins": {nm: sum(f(r["small_rank"]) for r in cand) for nm, f in bins}, "relations": dict(rel.most_common()), "true_answers_top": dict(tg.most_common(30)),
           "candidates": [{k: r[k] for k in ("case_id", "subject", "relation", "true", "small_rank", "medium_prob", "in_benchmark")} for r in cand]}
    path = os.path.join(OUT_DIR, "e1_import_count.json")
    json.dump(out, open(path, "w", encoding="utf8"), indent=1)
    print(f"\n  total time {time.time() - t0:.0f} s; saved {os.path.relpath(path, ROOT)}")
    print("=" * 110)


if __name__ == "__main__":
    main()
