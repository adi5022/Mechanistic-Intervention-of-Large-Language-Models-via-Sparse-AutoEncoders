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


def check_combination_safe(
    model, sae, prompt: str,
    mute_feature_ids: list[int], mute_strength: float,
    boost_feature_ids: list[int], boost_strength: float,
    target_token_id: int, top_k: int = 10
) -> dict:
    """
    Evaluates the joint effect of all muted and boosted features applied together.
    Detects any 'new blockers' (tokens ranked below target or absent in clean top-k,
    but ranked above target in the new list).
    """
    from src.hooks import make_joint_ablation_hook, make_signed_ablation_hook
    
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    
    model.reset_hooks()

    # 1. Clean baseline pass
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        
    clean_target_prob = clean_probs[target_token_id].item()
    clean_sorted_indices = torch.argsort(clean_probs, descending=True)
    clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1


    #Temporary Addition to test the testcase for "a"
    print("CLEAN TOP 10:")
    for i in range(10):
        idx = clean_sorted_indices[i].item()
        print(f"  Rank {i+1}: {model.to_string([idx])!r} — {clean_probs[idx].item()*100:.2f}%")
    
    # Get clean top_k token info
    clean_top_k_indices = clean_sorted_indices[:top_k]
    clean_top_k_probs = clean_probs[clean_top_k_indices]
    
    clean_top_k_tokens = [model.to_string([idx.item()]) for idx in clean_top_k_indices]
    clean_top_k_probs_list = [p.item() for p in clean_top_k_probs]
    clean_top_k_ids = [idx.item() for idx in clean_top_k_indices]
    
    # Build clean maps
    clean_id_to_rank = {idx: rank for rank, idx in enumerate(clean_top_k_ids, start=1)}
    clean_id_to_prob = {idx: prob for idx, prob in zip(clean_top_k_ids, clean_top_k_probs_list)}
    
    '''
    # Temporary Addition to test the testcase for "a"
    print("NEW TOP 10:")
    for i in range(10):
        idx = new_sorted_indices[i].item()
        print(f"  Rank {i+1}: {model.to_string([idx])!r} — {new_probs[idx].item()*100:.2f}%")
    '''
# 2. Apply ALL mute features AND all boost features TOGETHER, in one real pass
    from src.hooks import make_mute_and_boost_hook
    model.reset_hooks()
    combined_fn = make_mute_and_boost_hook(mute_feature_ids, mute_strength, boost_feature_ids, boost_strength, sae)
    model.add_hook(hook_name, combined_fn)
        
    with torch.no_grad():
        new_logits = model(tokens)
        new_probs = F.softmax(new_logits[0, -1, :], dim=-1)
        
    model.reset_hooks()  # Reset hooks immediately after measurement
    
    new_target_prob = new_probs[target_token_id].item()
    new_sorted_indices = torch.argsort(new_probs, descending=True)
    new_rank = (new_sorted_indices == target_token_id).nonzero().item() + 1
    
    # Get new top_k token info
    new_top_k_indices = new_sorted_indices[:top_k]
    new_top_k_probs = new_probs[new_top_k_indices]
    new_top_k_tokens = [model.to_string([idx.item()]) for idx in new_top_k_indices]
    new_top_k_probs_list = [p.item() for p in new_top_k_probs]
    new_top_k_ids = [idx.item() for idx in new_top_k_indices]
    
    # 3. Identify new_blockers
    new_blockers = []
    # A blocker is any token ranked ABOVE the target in the new list (rank < new_rank)
    # AND must have been ranked BELOW the target in the clean baseline (clean_tok_rank_val > clean_rank).
    for i, tok_id in enumerate(new_top_k_ids):
        new_tok_rank = i + 1
        if new_tok_rank >= new_rank:
            continue  # Not ranked above target
            
        tok_str = new_top_k_tokens[i]
        new_tok_prob = new_top_k_probs_list[i]
        
        # Get actual clean rank and probability from clean baseline
        clean_tok_rank_val = (clean_sorted_indices == tok_id).nonzero().item() + 1
        clean_tok_prob_val = clean_probs[tok_id].item()
        
        # Determine if it was in the clean top-k
        in_clean_top_k = tok_id in clean_id_to_rank
        clean_rank_report = clean_tok_rank_val if in_clean_top_k else "absent"
        clean_prob_report = clean_tok_prob_val if in_clean_top_k else "absent"
        
        # Blocker condition: must have been ranked below the target in clean baseline
        is_blocker = bool(clean_tok_rank_val > clean_rank)
        
        if is_blocker:
            new_blockers.append({
                "token": tok_str,
                "clean_prob_or_absent": clean_prob_report,
                "new_prob": new_tok_prob,
                "clean_rank_or_absent": clean_rank_report,
                "new_rank": new_tok_rank
            })
            
    # 4. is_safe condition
    is_safe = False
    if new_rank == 1:
        is_safe = True
    elif len(new_blockers) == 0 and new_rank <= clean_rank:
        is_safe = True
        
    return {
        "is_safe": is_safe,
        "target_clean_rank": clean_rank,
        "target_new_rank": new_rank,
        "target_clean_prob": clean_target_prob,
        "target_new_prob": new_target_prob,
        "new_blockers": new_blockers
    }



