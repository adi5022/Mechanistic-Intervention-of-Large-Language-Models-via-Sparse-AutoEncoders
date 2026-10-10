"""
Benchmark for the controlled export (step D): CounterFact records with a counterfactual target, split into dev (choose the dose) and test.

A record is kept when: both targets are single tokens (all targets in the project's CounterFact file are), the two targets differ, neither
GPT-2 small nor GPT-2 medium already answers with the counterfactual target, the record is not one of the 100 ROME dev records
(random.Random(0).sample(range(len(ds)), 100)) and not one of the 30 pilot records. Records are taken in the order given by
random.Random(1), so the benchmark is reproducible. Each record carries its tuned prompt, 2 paraphrase prompts and 5 of its 10 neighbourhood
prompts (other subjects with the same relation, where the TRUE target should stay preferred).
"""
import json
import os
import random

import torch

from src.editing import get_target_token_id, tokens_without_bos

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COUNTERFACT = os.path.join(ROOT, "datasets", "counterfact.json")


def load_counterfact():
    return json.load(open(COUNTERFACT, encoding="utf-8"))


def build_benchmark(small, medium, dev_n=50, test_n=150, exclude_case_ids=(), seed=1, n_para=2, n_nb=5):
    ds = load_counterfact()
    rome_dev = set(random.Random(0).sample(range(len(ds)), 100))
    order = random.Random(seed).sample(range(len(ds)), len(ds))
    exclude = set(exclude_case_ids)
    recs = []
    for i in order:
        if i in rome_dev or ds[i]["case_id"] in exclude:
            continue
        rw = ds[i]["requested_rewrite"]
        new, true = rw["target_new"]["str"], rw["target_true"]["str"]
        if tokens_without_bos(small, " " + new).numel() != 1 or tokens_without_bos(small, " " + true).numel() != 1:
            continue
        tid_new, tid_true = get_target_token_id(small, " " + new), get_target_token_id(small, " " + true)
        if tid_new == tid_true:
            continue
        prompt = rw["prompt"].format(rw["subject"])
        toks = small.to_tokens(prompt)
        with torch.no_grad():
            if small(toks)[0, -1].argmax().item() == tid_new or medium(toks)[0, -1].argmax().item() == tid_new:
                continue
        recs.append({"index": i, "case_id": ds[i]["case_id"], "subject": rw["subject"], "prompt": prompt, "target_new": new, "target_true": true,
                     "tid_new": tid_new, "tid_true": tid_true, "paraphrases": ds[i]["paraphrase_prompts"][:n_para],
                     "neighbours": ds[i]["neighborhood_prompts"][:n_nb]})
        if len(recs) == dev_n + test_n:
            break
    for k, r in enumerate(recs):
        r["split"] = "dev" if k < dev_n else "test"
    return recs
