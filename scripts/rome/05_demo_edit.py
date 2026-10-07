import argparse
import dataclasses
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.rome_baseline import rome_env

rome_env.setup()

import torch
from rome import ROMEHyperParams, apply_rome_to_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook
from util.generate import generate_fast
from util.globals import DEVICE, HPARAMS_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=3)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
hp = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2.json")
hp = dataclasses.replace(hp, layers=[a.layer])

request = {
    "prompt": "{} is located in the city of",
    "subject": "The Eiffel Tower",
    "target_new": {"str": "Rome"},
}
new_id = tok.encode(" Rome")[0]
old_id = tok.encode(" Paris")[0]
tests = [
    ("edit", "The Eiffel Tower is located in the city of"),
    ("paraphrase", "The Eiffel Tower can be found in the city of"),
    ("paraphrase", "Where is the Eiffel Tower? It is in the city of"),
    ("neighbor", "The Statue of Liberty is located in the city of"),
    ("neighbor", "Big Ben is located in the city of"),
]


def probe():
    rows = []
    with torch.no_grad():
        for kind, p in tests:
            inp = tok(p, return_tensors="pt").to(DEVICE)
            pr = torch.softmax(model(**inp).logits[0, -1], -1)
            rows.append((kind, p, tok.decode(pr.argmax()).strip(), pr[new_id].item(), pr[old_id].item()))
    return rows


before = probe()
gen_before = generate_fast(model, tok, [tests[0][1]], max_out_len=30)[0]
model, orig = apply_rome_to_model(model, tok, [request], hp, return_orig_weights=True)
after = probe()
gen_after = generate_fast(model, tok, [tests[0][1]], max_out_len=30)[0]
with torch.no_grad():
    for k, v in orig.items():
        nethook.get_parameter(model, k)[...] = v
restored = probe()

print()
print(f"edit layer {a.layer}: The Eiffel Tower -> Rome")
for b, af, p in zip(before, after, tests):
    print(
        f"{b[0]:11s} top1 {b[2]:>8s} -> {af[2]:<8s} p(Rome) {b[3]:.3f} -> {af[3]:.3f}  "
        f"p(Paris) {b[4]:.3f} -> {af[4]:.3f}  | {b[1]}"
    )
print("sample before:", gen_before)
print("sample after: ", gen_after)
drift = max(abs(b[3] - r[3]) for b, r in zip(before, restored))
print("max drift in p(Rome) after restoring weights:", f"{drift:.2e}")
