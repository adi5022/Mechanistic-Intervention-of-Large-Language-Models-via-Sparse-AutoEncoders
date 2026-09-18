import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F
from src.sae_utils import load_base_model, load_sae_for_layer
from src.editing import get_target_token_id, check_target_safe_batch, check_boost_safe_batch, build_clean_context, get_top_competitor_features, get_top_target_features
from src.hooks import make_joint_ablation_hook, make_signed_ablation_hook

model = load_base_model()
sae = load_sae_for_layer(8)

prompt = "2005 Australian Open is located in"
target = "Melbourne"
target_str = " Melbourne"
target_id = get_target_token_id(model, target_str)
tokens = model.to_tokens(prompt)

for b_str in [0.5, 0.7, 0.8, 1.0]:
    clean_ctx = build_clean_context(model, sae, prompt, target_id)
    comp_feats = get_top_competitor_features(model, sae, prompt, torch.argmax(clean_ctx.clean_probs).item(), top_n=30, clean_ctx=clean_ctx, use_batched=True)
    targ_feats = get_top_target_features(model, sae, prompt, target_id, top_n=30, clean_ctx=clean_ctx, use_batched=True)
    
    comp_ids = [fid for fid, (safe, _) in zip([f for f, _ in comp_feats], check_target_safe_batch(model, sae, clean_ctx, [f for f, _ in comp_feats], target_id, strength=0.3)) if safe]
    targ_ids = [fid for fid, (safe, _, _) in zip([f for f, _ in targ_feats], check_boost_safe_batch(model, sae, clean_ctx, [f for f, _ in targ_feats], target_id, strength=b_str)) if safe]
    
    print(f"\nBoost Strength = {b_str}:")
    for m_n in [1, 3, 5]:
        for b_n in [1, 3, 5]:
            mute_batch = comp_ids[:m_n]
            boost_batch = targ_ids[:b_n]
            
            model.reset_hooks()
            model.add_hook("blocks.8.hook_resid_pre", make_joint_ablation_hook(mute_batch, sae, strength=0.3))
            model.add_hook("blocks.8.hook_resid_pre", make_signed_ablation_hook(boost_batch, sae, strength=+b_str))
            
            with torch.no_grad():
                logits = model(tokens)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            top1_id = torch.argmax(probs).item()
            target_rank = (torch.argsort(probs, descending=True) == target_id).nonzero().item() + 1
            print(f"  Mute {m_n} | Boost {b_n} -> Top1: {model.to_string([top1_id])!r}, Target Rank: #{target_rank}, Prob: {probs[target_id].item()*100:.2f}%")
