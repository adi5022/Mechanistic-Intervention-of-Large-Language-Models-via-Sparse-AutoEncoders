import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.rome_baseline import rome_env

rome_env.setup()

import numpy as np
from experiments.causal_trace import ModelAndTokenizer, calculate_hidden_flow, collect_embedding_std
from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument("--min-p", type=float, default=0.2)
p.add_argument("--limit", type=int, default=None)
p.add_argument("--window", type=int, default=3)
p.add_argument("--noise-mult", type=float, default=3.0)
a = p.parse_args()

facts = json.loads((REPO / "data" / "comparison" / "gpt2_knowns.json").read_text())
facts = [f for f in facts if f["p"] >= a.min_p]
print("facts selected:", len(facts))

mt = ModelAndTokenizer("gpt2")
noise = a.noise_mult * collect_embedding_std(mt, [f["subject"] for f in facts])
print("noise level:", round(noise, 4))

cache = REPO / "data" / "comparison" / "raw" / "trace"
cache.mkdir(parents=True, exist_ok=True)
(cache / "meta.json").write_text(json.dumps(dict(noise=noise, window=a.window, min_p=a.min_p, n_facts=len(facts))))

todo = facts[: a.limit] if a.limit else facts
t0 = time.time()
for f in tqdm(todo):
    for kind in (None, "mlp", "attn"):
        fn = cache / f"fact_{f['known_id']}_{kind or 'resid'}.npz"
        if fn.exists():
            continue
        r = calculate_hidden_flow(
            mt, f["prompt"], f["subject"], noise=noise, window=a.window, kind=kind
        )
        np.savez(
            fn,
            scores=r["scores"].numpy(),
            low=r["low_score"],
            high=r["high_score"],
            subject_range=np.array(r["subject_range"]),
            tokens=np.array(r["input_tokens"]),
            answer=r["answer"],
        )
print("done in", round(time.time() - t0), "seconds for", len(todo), "facts")
