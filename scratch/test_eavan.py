import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F
from src.sae_utils import load_base_model
from src.editing import get_target_token_id

model = load_base_model()

prompt = "Eavan Boland was born in"
target_str = " Dublin"

tokens = model.to_tokens(prompt)
print("Tokens:", tokens)
print("Decoded tokens:", [model.to_string([t.item()]) for t in tokens[0]])

logits = model(tokens)
probs = F.softmax(logits[0, -1, :], dim=-1)

top5_probs, top5_indices = torch.topk(probs, k=5)
print("\nClean Baseline Top 5 for prompt:", repr(prompt))
for r, (p, idx) in enumerate(zip(top5_probs, top5_indices), 1):
    print(f"Rank {r}: {model.to_string([idx.item()])!r} (ID: {idx.item()}) - Prob: {p.item()*100:.4f}%")

target_token_id = get_target_token_id(model, target_str)
target_prob = probs[target_token_id].item()
print(f"\nTarget {target_str!r} (ID: {target_token_id}): Prob = {target_prob*100:.4f}%")
