"""Run the repair / SAE-limit diagnostics headlessly and save JSON to outputs/repair_diagnostics/."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
from src.hybrid_runner import run_hybrid_sweep
from src.repair_diagnostics import diagnose_prompt, verdict

PROMPTS = [
    ("The Colosseum is located in", "Rome"),
    ("She opened the door and saw a", "cat"),
    ("The doctor said that", "she"),
    ("The location of Massachusetts Institute of Technology is in", "Cambridge"),
]
CFG = {"record_detail": "compact", "collateral": False, "combination_check": False}

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="4,6,7,8,9,10,11")
ap.add_argument("--out", default="outputs/repair_diagnostics/repair_run.json")
ap.add_argument("--n-prompts", type=int, default=4)
ap.add_argument("--steps", type=int, default=60)
a = ap.parse_args()
layers = [int(x) for x in a.layers.split(",")]
device = get_default_device()
model = load_base_model(device)
saes = {}
def get_sae(l):
    if l not in saes:
        saes[l] = load_sae_for_layer(l, device)
    return saes[l]
res = {"layers": layers, "cfg": CFG, "prompts": []}
for p, t in PROMPTS[:a.n_prompts]:
    t0 = time.time()
    r = diagnose_prompt(model, get_sae, p, t, layers, CFG, run_hybrid_sweep, device, opt_steps=a.steps,
                        progress_cb=lambda i, n, m: print(f"  [{i+1}/{n}] {m}", flush=True))
    res["prompts"].append(r)
    print(f"{p!r}: {time.time()-t0:.0f}s"); [print("  ", v) for v in verdict(r)]
    json.dump(res, open(a.out, "w"), indent=1)
print("saved", a.out)
