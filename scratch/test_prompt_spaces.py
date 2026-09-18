import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F
from src.sae_utils import load_base_model

model = load_base_model()

prompts = [
    "Eavan Boland was born in",
    "Eavan Boland was born in ",
    "Eavan Boland was born in\n",
    "Eavan Boland was born in.",
]

for p in prompts:
    tokens = model.to_tokens(p)
    with torch.no_grad():
        logits = model(tokens)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    top1_id = torch.argmax(probs).item()
    top1_str = model.to_string([top1_id])
    print(f"Prompt {p!r} -> Top 1: {top1_str!r} (ID: {top1_id})")
