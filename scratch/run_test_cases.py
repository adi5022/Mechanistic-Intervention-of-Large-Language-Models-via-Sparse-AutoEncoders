import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F
from src.sae_utils import load_base_model, load_sae_for_layer
from src.editing import (
    get_target_token_id,
    get_top_competitor_features,
    get_top_target_features,
    check_target_safe,
    check_boost_safe
)
from src.hooks import make_joint_ablation_hook, make_signed_ablation_hook

model = load_base_model()
layer = 8
sae = load_sae_for_layer(layer)
hook_name = f"blocks.{layer}.hook_resid_pre"

test_cases = [
    {
        "name": "Test Case 1 (Melbourne)",
        "prompt": "2005 Australian Open is located in",
        "target": "Melbourne",
        "mute_str": 0.3,
        "boost_str": 0.5
    },
    {
        "name": "Test Case 2 (Dublin)",
        "prompt": "Eavan Boland was born in",
        "target": "Dublin",
        "mute_str": 0.3,
        "boost_str": 0.7
    }
]

for tc in test_cases:
    print(f"\n==================================================")
    print(f"--- {tc['name']} ---")
    prompt = tc['prompt']
    target = tc['target']
    target_str = target if target.startswith(" ") else " " + target
    
    tokens = model.to_tokens(prompt)
    target_id = get_target_token_id(model, target_str)
    
    # 1. Baseline
    model.reset_hooks()
    with torch.no_grad():
        logits = model(tokens)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    
    top5_p, top5_i = torch.topk(probs, 5)
    top1_id = top5_i[0].item()
    top1_str = model.to_string([top1_id])
    target_prob = probs[target_id].item()
    target_rank = (torch.argsort(probs, descending=True) == target_id).nonzero().item() + 1
    
    print(f"CLEAN BASELINE:")
    print(f"  Top-1: {top1_str!r} ({top5_p[0].item()*100:.2f}%)")
    print(f"  Target {target_str!r}: Rank #{target_rank}, Prob = {target_prob*100:.2f}%")
    
    # 2. Get safe features
    comp_feats = get_top_competitor_features(model, sae, prompt, top1_id, top_n=30)
    safe_mutes = []
    for fid, _ in comp_feats:
        safe, _ = check_target_safe(model, sae, prompt, fid, target_id, strength=tc['mute_str'])
        if safe:
            safe_mutes.append(fid)
            
    targ_feats = get_top_target_features(model, sae, prompt, target_id, top_n=30)
    safe_boosts = []
    for fid, _ in targ_feats:
        safe, _, _ = check_boost_safe(model, sae, prompt, fid, target_id, strength=tc['boost_str'])
        if safe:
            safe_boosts.append(fid)
            
    print(f"Safe Mute Pool ({len(safe_mutes)} features): {safe_mutes[:3]}")
    print(f"Safe Boost Pool ({len(safe_boosts)} features): {safe_boosts[:3]}")
    
    # 3. Hybrid Run (Batch size 3x3)
    mute_batch = safe_mutes[:3]
    boost_batch = safe_boosts[:3]
    
    model.reset_hooks()
    joint_mute_fn = make_joint_ablation_hook(mute_batch, sae, strength=tc['mute_str'])
    model.add_hook(hook_name, joint_mute_fn)
    
    signed_boost_fn = make_signed_ablation_hook(boost_batch, sae, strength=+tc['boost_str'])
    model.add_hook(hook_name, signed_boost_fn)
    
    with torch.no_grad():
        h_logits = model(tokens)
    h_probs = F.softmax(h_logits[0, -1, :], dim=-1)
    
    h_top5_p, h_top5_i = torch.topk(h_probs, 5)
    h_top1_id = h_top5_i[0].item()
    h_top1_str = model.to_string([h_top1_id])
    h_target_prob = h_probs[target_id].item()
    h_target_rank = (torch.argsort(h_probs, descending=True) == target_id).nonzero().item() + 1
    
    print(f"HYBRID INTERVENTION (Mute {len(mute_batch)} @ -{tc['mute_str']} | Boost {len(boost_batch)} @ +{tc['boost_str']}):")
    print(f"  New Top-1: {h_top1_str!r} ({h_top5_p[0].item()*100:.2f}%)")
    print(f"  Target {target_str!r}: Rank #{h_target_rank}, Prob = {h_target_prob*100:.2f}%")
