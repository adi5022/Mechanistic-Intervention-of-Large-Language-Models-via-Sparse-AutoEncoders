import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F
from src.sae_utils import load_base_model, load_sae_for_layer
from src.editing import get_target_token_id, get_top_competitor_features, check_target_safe

model = load_base_model()
layer = 8
sae = load_sae_for_layer(layer)

prompt_4 = "Eavan Boland was born in"
target_4 = "Dublin"
target_str = target_4 if target_4.startswith(" ") else " " + target_4

tokens = model.to_tokens(prompt_4)
target_token_id = get_target_token_id(model, target_str)

model.reset_hooks()
with torch.no_grad():
    logits = model(tokens)
probs = F.softmax(logits[0, -1, :], dim=-1)
current_top1_id = torch.argmax(probs).item()
current_top1_str = model.to_string([current_top1_id])
baseline_target_prob = probs[target_token_id].item()

print(f"DEBUG Initial: Top-1 = {current_top1_str!r} (ID: {current_top1_id}), Target Prob = {baseline_target_prob*100:.4f}%")

topn_4 = 30
strength_mute_4 = 0.3
competitor_features = get_top_competitor_features(model, sae, prompt_4, current_top1_id, top_n=topn_4)

print(f"Competitor features count: {len(competitor_features)}")

comp_ids = []
for fid, delta in competitor_features[:10]:
    is_safe, t_delta = check_target_safe(model, sae, prompt_4, fid, target_token_id, strength=strength_mute_4)
    print(f"FID {fid}: is_safe={is_safe}, t_delta={t_delta}")
