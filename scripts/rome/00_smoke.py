import os
import sys

ROME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "rome"))
sys.path.insert(0, ROME)
os.chdir(ROME)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook
from util.globals import DEVICE

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
inp = tok("The Eiffel Tower is located in the city of", return_tensors="pt").to(DEVICE)
with torch.no_grad(), nethook.Trace(model, "transformer.h.5.mlp.c_proj", retain_input=True) as t:
    out = model(**inp)
print("device", DEVICE, "params", sum(p.numel() for p in model.parameters()))
print("c_proj input shape", tuple(t.input.shape))
p = out.logits[0, -1].softmax(-1)
top = p.topk(5)
print([(tok.decode(i), round(v.item(), 3)) for v, i in zip(top.values, top.indices)])
