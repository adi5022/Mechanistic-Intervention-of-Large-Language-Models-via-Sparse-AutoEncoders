import argparse
import contextlib
import csv
import dataclasses
import io
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.rome_baseline import rome_env

rome_env.setup()

import numpy as np
from dsets import CounterFactDataset
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from rome import ROMEHyperParams, apply_rome_to_model
from scipy.stats import hmean
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook
from util.globals import DATA_DIR, DEVICE, HPARAMS_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="1,3,5,7,9")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
layers = [int(x) for x in a.layers.split(",")]

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
base_hp = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2.json")

ds = CounterFactDataset(DATA_DIR)
pool = random.Random(a.seed).sample(range(len(ds)), 100)
records = [ds[i] for i in pool[: a.n]]

cache = REPO / "data" / "comparison" / "raw" / "sweep"
cache.mkdir(parents=True, exist_ok=True)


def summarize(res):
    def frac(key, cond):
        return float(np.mean([np.mean([cond(x) for x in r[key]]) for r in res]))

    es = frac("rewrite_prompts_probs", lambda x: x["target_true"] > x["target_new"])
    ps = frac("paraphrase_prompts_probs", lambda x: x["target_true"] > x["target_new"])
    ns = frac("neighborhood_prompts_probs", lambda x: x["target_true"] < x["target_new"])
    return es, ps, ns, float(hmean([max(es, 1e-9), max(ps, 1e-9), max(ns, 1e-9)]))


pre = []
for rec in records:
    f = cache / f"pre_{rec['case_id']}.json"
    if f.exists():
        pre.append(json.loads(f.read_text()))
    else:
        r = compute_rewrite_quality_counterfact(model, tok, rec, None, None)
        f.write_text(json.dumps(r))
        pre.append(r)
rows = []
es, ps, ns, s = summarize(pre)
print(f"no edit     ES {es:.3f}  PS {ps:.3f}  NS {ns:.3f}  S {s:.3f}")
rows.append(dict(layer="none", ES=es, PS=ps, NS=ns, S=s, sec_per_edit=0.0, n=len(records)))

for L in layers:
    hp = dataclasses.replace(base_hp, layers=[L])
    res, times = [], []
    for rec in tqdm(records, desc=f"layer {L}"):
        f = cache / f"L{L}_{rec['case_id']}.json"
        if f.exists():
            d = json.loads(f.read_text())
        else:
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                _, orig = apply_rome_to_model(
                    model, tok, [rec["requested_rewrite"]], hp, return_orig_weights=True
                )
            t = time.time() - t0
            post = compute_rewrite_quality_counterfact(model, tok, rec, None, None)
            with __import__("torch").no_grad():
                for k, v in orig.items():
                    nethook.get_parameter(model, k)[...] = v
            d = dict(post=post, time=t)
            f.write_text(json.dumps(d))
        res.append(d["post"])
        times.append(d["time"])
    es, ps, ns, s = summarize(res)
    print(f"layer {L:2d}    ES {es:.3f}  PS {ps:.3f}  NS {ns:.3f}  S {s:.3f}  {np.mean(times):.1f}s/edit")
    rows.append(dict(layer=L, ES=es, PS=ps, NS=ns, S=s, sec_per_edit=float(np.mean(times)), n=len(records)))

out = REPO / "benchmark_results" / "rome_vs_scalpel" / "layer_sweep_dev.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved", out)
