"""
Re-run the six additive hand trials of docs/Research_Journal/30.md (section 6) and write the numbers to a JSON file.

The hand trials were first made in the application one prompt at a time; only the "The capital of France is" -> Hong runs are in
the stored session histories. This script repeats all six prompts in three arms with the application's defaults (Top N 200,
all prompt positions, 100 steps; additive arms: 200 silent candidates, cap factor 1.0, sparsity weight 0.005) so that the table
in the journal and in the project report (Appendix B.5) can be checked against a stored file.

    .venv\\Scripts\\python.exe tools/rerun_additive_hand_trials.py
    .venv\\Scripts\\python.exe tools/rerun_additive_hand_trials.py --out outputs/hand_trials_rerun.json

Arms: "mult" (multipliers only), "add" (multipliers and additive), "add+klall" (additive, KL charged at every step).
Each cell: final rank through the real hook, target probability (%), same-prompt KL (nats), edit size (% of the residual norm at
the last position) and the model-compute time of the run. The numbers are exploratory (six hand-picked prompts).
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
from src.editing import build_clean_context, get_target_token_id
from src.gradient_editing import run_gradient_descent_edit

TRIALS = [
    ("My nephew is a", " rapist"),
    ("The sky is", " powder"),
    ("Hello, my name is", " Aryan"),
    ("Hello, my name is", " Rajesh"),
    ("The sky is", " Adi"),
    ("The capital of France is", " Hong"),
]
ARMS = [("mult", dict(additive=False, kl_always=False)),
        ("add", dict(additive=True, kl_always=False)),
        ("add+klall", dict(additive=True, kl_always=True))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "Research_Journal", "packs", "additive_control",
                                                  "hand_trials_rerun.json"))
    ap.add_argument("--layer", type=int, default=8)
    a = ap.parse_args()

    model = load_base_model()
    sae = load_sae_for_layer(layer=a.layer)
    hook_name = getattr(sae.cfg, "hook_name", f"blocks.{a.layer}.hook_resid_pre")
    rows = []
    for prompt, target in TRIALS:
        tid = get_target_token_id(model, target)
        ctx = build_clean_context(model, sae, prompt, tid)
        row = {"prompt": prompt, "target": target, "start_rank": ctx.clean_rank}
        for name, kw in ARMS:
            t0 = time.time()
            g = run_gradient_descent_edit(model, sae, ctx, tid, prompt, a.layer, hook_name, top_n=200, positions="all", steps=100, **kw)
            rp = g["real_path"]
            row[name] = {"rank": rp["rank"], "prob_pct": round(100 * rp["prob"], 4), "kl": round(rp["kl"], 5),
                         "edit_pct": round(100 * g["edit_size_frac_norm"], 2), "time_s": round(time.time() - t0, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"date": time.strftime("%Y-%m-%d"), "device": get_default_device(), "layer": a.layer,
               "settings": {"top_n": 200, "positions": "all", "steps": 100, "add_top_m": 200, "add_cap_factor": 1.0, "lam_add": 0.005},
               "rows": rows}, open(a.out, "w", encoding="utf-8"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
