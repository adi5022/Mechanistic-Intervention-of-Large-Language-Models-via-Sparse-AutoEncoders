import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.rome_baseline import rome_env

rome_env.setup()

import numpy as np
import torch
from dsets import KnownsDataset
from experiments.causal_trace import ModelAndTokenizer, make_inputs, predict_from_input
from util.globals import DATA_DIR

mt = ModelAndTokenizer("gpt2")
ds = KnownsDataset(DATA_DIR)
keep = []
for rec in ds:
    inp = make_inputs(mt.tokenizer, [rec["prompt"]])
    with torch.no_grad():
        preds, p = predict_from_input(mt.model, inp)
    ans = mt.tokenizer.decode(preds[0]).strip()
    if ans == rec["attribute"]:
        keep.append(
            dict(
                known_id=rec["known_id"],
                prompt=rec["prompt"],
                subject=rec["subject"],
                attribute=rec["attribute"],
                p=float(p[0]),
            )
        )
keep.sort(key=lambda r: -r["p"])
ps = np.array([r["p"] for r in keep])
print("total:", len(ds), "kept:", len(keep))
print("p percentiles 10/50/90:", np.percentile(ps, [10, 50, 90]).round(3).tolist())
print("p>=0.1:", int((ps >= 0.1).sum()), " p>=0.2:", int((ps >= 0.2).sum()), " p>=0.3:", int((ps >= 0.3).sum()))
for r in keep[:8]:
    print(round(r["p"], 3), r["prompt"], "->", r["attribute"])
out = REPO / "data" / "comparison" / "gpt2_knowns.json"
out.write_text(json.dumps(keep, indent=1))
print("saved", out)
