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
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    resid_act = cache[hook_name]  # [batch, seq_len, d_model]
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
        hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
        
        with torch.no_grad():
            # Run model with feature fully ablated
            ablated_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_fn)]
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

def get_top_competitor_features(
    model, sae, prompt: str, current_top_token_id: int, top_n: int = 20
) -> list[tuple[int, float]]:
    """
    Ranks active features by how much their ablation (strength=1.0) decreases the probability of current_top_token_id.
    Returns: list of (feature_id, prob_delta) sorted by prob_delta ascending (largest drop first).
    """
    tokens = model.to_tokens(prompt)
    
    # 1. Get baseline probability of the competitor
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        clean_competitor_prob = clean_probs[current_top_token_id].item()
        
    # 2. Get top active features
    active_features = get_top_active_features(model, sae, prompt, top_n=top_n)
    
    results = []
    for feature_id, activation in active_features:
        hook_fn = make_ablation_hook(feature_id, sae, strength=1.0)
        hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
        
        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_fn)]
            )
            ablated_probs = F.softmax(ablated_logits[0, -1, :], dim=-1)
            ablated_competitor_prob = ablated_probs[current_top_token_id].item()
            
            # Causal effect on the competitor
            prob_delta = ablated_competitor_prob - clean_competitor_prob
            
        results.append((feature_id, prob_delta))
        
    # Sort by prob_delta ascending (biggest drop / most negative delta first)
    results.sort(key=lambda x: x[1])
    return results

def get_top_target_features(
    model, sae, prompt: str, target_token_id: int, top_n: int = 30
) -> list[tuple[int, float]]:
    """
    Ranks active features by how much their ablation (strength=1.0) decreases the probability of target_token_id.
    Returns: list of (feature_id, prob_delta) sorted by prob_delta ascending (largest target drop first).
    """
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        clean_target_prob = clean_probs[target_token_id].item()
        
    active_features = get_top_active_features(model, sae, prompt, top_n=top_n)
    
    results = []
    for feature_id, activation in active_features:
        hook_fn = make_ablation_hook(feature_id, sae, strength=1.0)
        
        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_fn)]
            )
            ablated_probs = F.softmax(ablated_logits[0, -1, :], dim=-1)
            ablated_target_prob = ablated_probs[target_token_id].item()
            
            prob_delta = ablated_target_prob - clean_target_prob
            
        results.append((feature_id, prob_delta))
        
    results.sort(key=lambda x: x[1])
    return results

def check_target_safe(
    model, sae, prompt: str, feature_id: int, target_token_id: int, strength: float = 0.3
) -> tuple[bool, float]:
    """
    Temporarily applies ONLY this one feature's ablation (using make_ablation_hook),
    measures the resulting change in the TARGET token's probability, and returns
    (is_safe, target_prob_delta) where is_safe = True if target_prob_delta >= 0 (target wasn't hurt).
    Resets hooks after measurement.
    """
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    
    # Baseline target prob under current model state
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        clean_target_prob = clean_probs[target_token_id].item()
        clean_sorted_indices = torch.argsort(clean_probs, descending=True)
        clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1
        
    hook_fn = make_ablation_hook(feature_id, sae, strength=strength)
    
    # Run with temporary ablation hook for feature_id
    with torch.no_grad():
        ablated_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, hook_fn)]
        )
        ablated_probs = F.softmax(ablated_logits[0, -1, :], dim=-1)
        ablated_target_prob = ablated_probs[target_token_id].item()
        ablated_sorted_indices = torch.argsort(ablated_probs, descending=True)
        ablated_rank = (ablated_sorted_indices == target_token_id).nonzero().item() + 1
        
    target_prob_delta = ablated_target_prob - clean_target_prob
    is_safe = bool(ablated_rank <= clean_rank and target_prob_delta >= -1e-6)
    
    return is_safe, target_prob_delta

def check_boost_safe(
    model, sae, prompt: str, feature_id: int, target_token_id: int, strength: float = 0.5
) -> tuple[bool, float, int]:
    """
    Temporarily applies ONLY this one feature's boost (using make_signed_ablation_hook with +strength),
    measures the resulting change in the TARGET token's probability and rank, and returns
    (is_safe, target_prob_delta, rank_improvement) where is_safe = True if target rank improves or stays same,
    and target probability does not decrease.
    """
    from src.hooks import make_signed_ablation_hook
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        clean_target_prob = clean_probs[target_token_id].item()
        clean_sorted_indices = torch.argsort(clean_probs, descending=True)
        clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1
        
    hook_fn = make_signed_ablation_hook([feature_id], sae, strength=+strength)
    
    with torch.no_grad():
        boosted_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, hook_fn)]
        )
        boosted_probs = F.softmax(boosted_logits[0, -1, :], dim=-1)
        boosted_target_prob = boosted_probs[target_token_id].item()
        boosted_sorted_indices = torch.argsort(boosted_probs, descending=True)
        boosted_rank = (boosted_sorted_indices == target_token_id).nonzero().item() + 1
        
    target_prob_delta = boosted_target_prob - clean_target_prob
    rank_improvement = clean_rank - boosted_rank
    
    is_safe = bool(boosted_rank <= clean_rank and target_prob_delta >= -1e-6)
    
    return is_safe, target_prob_delta, rank_improvement


