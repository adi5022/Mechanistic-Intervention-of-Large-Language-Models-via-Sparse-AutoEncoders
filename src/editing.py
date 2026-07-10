"""
Activation editing/steering, intervention logic, and the Causal Feature Selector.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from src.hooks import make_ablation_hook

HOOK_NAME = "blocks.8.hook_resid_pre"

def get_target_token_id(model, target_str: str) -> int:
    """
    Safely resolves a target string (e.g. ' Paris') to its token ID in the model's vocabulary.
    """
    token_ids = model.to_tokens(target_str, prepend_bos=False).squeeze()
    if token_ids.numel() > 1:
        return token_ids[-1].item()
    return token_ids.item()

def get_top_active_features(model, sae, prompt: str, top_n: int = 20) -> List[Tuple[int, float]]:
    """
    Runs the model on the prompt, extracts the final token's residual stream activation,
    encodes it with the SAE, and returns the top_n feature indices and their raw activations.
    """
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)
    
    # Get last token activation from residual stream
    resid_act = cache[HOOK_NAME]  # [batch, seq_len, d_model]
    last_token_act = resid_act[:, -1, :]  # [batch, d_model]
    
    # Encode to SAE features
    with torch.no_grad():
        feature_acts = sae.encode(last_token_act)  # [batch, n_features]
        last_token_features = feature_acts[0]  # [n_features]
        
    values, indices = torch.topk(last_token_features, top_n)
    
    return [(idx.item(), val.item()) for idx, val in zip(indices, values) if val.item() > 0.0]

def run_causal_selector(
    model, sae, prompt: str, target_str: str, top_n: int = 20
) -> List[Dict]:
    """
    Ranks the top-N active features by their causal effect on the target token's probability
    when fully ablated (strength=1.0).
    """
    target_token_id = get_target_token_id(model, target_str)
    tokens = model.to_tokens(prompt)
    
    # Get baseline (clean) probabilities
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        clean_target_prob = clean_probs[target_token_id].item()
    
    # Get top active features
    active_features = get_top_active_features(model, sae, prompt, top_n=top_n)
    
    results = []
    for feature_id, activation in active_features:
        hook_fn = make_ablation_hook(feature_id, sae, strength=1.0)
        
        with torch.no_grad():
            # Run model with feature fully ablated
            ablated_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(HOOK_NAME, hook_fn)]
            )
            ablated_probs = F.softmax(ablated_logits[0, -1, :], dim=-1)
            ablated_target_prob = ablated_probs[target_token_id].item()
            
            # Compute top-1 token after ablation
            top1_id = torch.argmax(ablated_probs).item()
            top1_token = model.to_string([top1_id])
            
            # Causal effect: change in target token probability
            prob_delta = ablated_target_prob - clean_target_prob
            
            # KL divergence: D_KL(clean || ablated) to measure distribution shift
            kl = torch.sum(clean_probs * torch.log((clean_probs + 1e-10) / (ablated_probs + 1e-10))).item()
            
        results.append({
            "feature_id": feature_id,
            "activation": activation,
            "clean_prob": clean_target_prob,
            "ablated_prob": ablated_target_prob,
            "prob_delta": prob_delta,
            "kl_divergence": kl,
            "top1_prediction": top1_token
        })
        
    # Rank by magnitude of causal effect (negative delta means ablated prob dropped, i.e., feature was positive for prediction)
    # Sorting by prob_delta ascending (largest drop first)
    results.sort(key=lambda x: x["prob_delta"])
    return results
