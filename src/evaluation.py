"""
Evaluation metrics and scripts for measuring intervention success.
"""

import torch
import torch.nn.functional as F
import pandas as pd
from src.editing import get_target_token_id, run_causal_selector, get_top_competitor_features

from src.hooks import make_ablation_hook, make_joint_ablation_hook

def classify_fact(model, prompt: str, target_str: str) -> dict:
    """
    Classifies a (prompt, target_str) fact pair based on GPT-2-small's baseline behavior.
    
    Labels:
      - "already_correct": target is already the top-1 prediction.
      - "suppressed": target is not top-1, but clean_prob > 0.01 OR rank <= 50.
      - "absent": target is not top-1, clean_prob <= 0.01, and rank > 50.
    """
    target_token_id = get_target_token_id(model, target_str)
    tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        logits = model(tokens)
    
    # Get probabilities for the last token position
    probs = F.softmax(logits[0, -1, :], dim=-1)
    
    # Raw probability of target token
    clean_prob = probs[target_token_id].item()
    
    # Rank of target token (1-indexed: 1 is the highest probability)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    rank = (sorted_indices == target_token_id).nonzero().item() + 1
    
    # Check if target is top-1
    is_top1 = (rank == 1)
    
    # Classify
    if is_top1:
        label = "already_correct"
    elif clean_prob > 0.01 or rank <= 50:
        label = "suppressed"
    else:
        label = "absent"
        
    return {
        "clean_prob": clean_prob,
        "rank": rank,
        "label": label
    }

def scan_fact_batch(model, sae, fact_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Scans a batch of fact pairs and returns a pandas DataFrame.
    """
    results = []
    for prompt, target in fact_pairs:
        res = classify_fact(model, prompt, target)
        results.append({
            "prompt": prompt,
            "target": target,
            "clean_prob": res["clean_prob"],
            "rank": res["rank"],
            "label": res["label"]
        })
    return pd.DataFrame(results)


def iterative_ablate(
    model, sae, prompt: str, target_str: str, max_rounds: int = 5, strength: float = 0.3
) -> dict:
    """
    Finds the ordered list of features to ablate until the target becomes top-1,
    targeting competitor features.
    """
    model.reset_hooks()
    
    target_token_id = get_target_token_id(model, target_str)
    tokens = model.to_tokens(prompt)
    
    features_ablated = []
    success = False
    rounds_used = 0
    
    # Check if already correct before any ablation
    with torch.no_grad():
        logits = model(tokens)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    current_top1_id = torch.argmax(probs).item()
    current_top1_token = model.to_string([current_top1_id])
    
    initial_top1_token = current_top1_token
    
    if current_top1_id == target_token_id:
        return {
            "prompt": prompt,
            "target": target_str,
            "success": True,
            "rounds_used": 0,
            "features_ablated": [],
            "final_target_rank": 1,
            "final_target_prob": probs[target_token_id].item(),
            "initial_top1_token": initial_top1_token,
            "final_top1_token": current_top1_token
        }

    for r_idx in range(1, max_rounds + 1):
        # 1. Determine the model's CURRENT top-1 predicted token under existing hooks
        with torch.no_grad():
            logits = model(tokens)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        current_top1_id = torch.argmax(probs).item()
        
        # If current top-1 == target token, stop immediately (Success)
        if current_top1_id == target_token_id:
            success = True
            break
            
        # 2. Get top competitor features driving this current wrong guess
        competitor_results = get_top_competitor_features(
            model, sae, prompt, current_top1_id, top_n=20
        )
        
        # 3. Find the top competitor feature that has not been ablated yet
        next_feature = None
        for fid, prob_delta in competitor_results:
            if fid not in features_ablated:
                next_feature = fid
                break
                
        if next_feature is None:
            break
            
        features_ablated.append(next_feature)
        rounds_used = r_idx
        
        # 4. Add the hook for this new feature permanently (stacking)
        model.reset_hooks()
        for fid in features_ablated:
            hook_fn = make_ablation_hook(fid, sae, strength)
            model.add_hook("blocks.8.hook_resid_pre", hook_fn)
            
    # Final check of the model's prediction/prob/rank
    with torch.no_grad():
        logits = model(tokens)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    sorted_indices = torch.argsort(probs, descending=True)
    final_rank = (sorted_indices == target_token_id).nonzero().item() + 1
    final_prob = probs[target_token_id].item()
    final_top1_id = torch.argmax(probs).item()
    final_top1_token = model.to_string([final_top1_id])
    
    # We achieved success if the final top-1 is target
    if final_top1_id == target_token_id:
        success = True
        
    # Clean up hooks on the model before returning
    model.reset_hooks()
    
    return {
        "prompt": prompt,
        "target": target_str,
        "success": success,
        "rounds_used": rounds_used,
        "features_ablated": features_ablated,
        "final_target_rank": final_rank,
        "final_target_prob": final_prob,
        "initial_top1_token": initial_top1_token,
        "final_top1_token": final_top1_token
    }



def check_specificity(
    model, sae, features_ablated: list[int], control_pairs: list[tuple[str, str]], strength: float = 0.3
) -> float:
    """
    Checks if applying features_ablated breaks control_pairs.
    Returns: fraction of controls that remained unaffected (top-1 prediction unchanged).
    """
    if not features_ablated:
        return 1.0
        
    model.reset_hooks()
    
    # Apply all ablation hooks
    for fid in features_ablated:
        hook_fn = make_ablation_hook(fid, sae, strength)
        model.add_hook("blocks.8.hook_resid_pre", hook_fn)
        
    unaffected = 0
    for prompt, target in control_pairs:
        tokens = model.to_tokens(prompt)
        target_token_id = get_target_token_id(model, target)
        
        with torch.no_grad():
            logits = model(tokens)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        if torch.argmax(probs).item() == target_token_id:
            unaffected += 1
            
    model.reset_hooks()
    return unaffected / len(control_pairs) if control_pairs else 1.0


def run_stage_1_evaluation(model, sae, max_rounds: int = 5, strength: float = 0.3) -> pd.DataFrame:
    """
    Runs the full stage 1 evaluation across all 46 facts from the fact bank.
    Specifically:
    1. Runs the eligibility scanner to categorize them.
    2. Identifies suppressed facts (eval targets) and already_correct facts (controls).
    3. Runs iterative_ablate on evaluate targets.
    4. Evaluates specificity of successful edits against controls.
    """
    from src.fact_bank import FACT_BANK
    
    # Step 1: Scan and categorize
    df_scan = scan_fact_batch(model, sae, FACT_BANK)
    
    # Control pool: already_correct facts
    control_df = df_scan[df_scan["label"] == "already_correct"]
    control_pairs = list(zip(control_df["prompt"], control_df["target"]))
    
    # Target pool: suppressed facts
    target_df = df_scan[df_scan["label"] == "suppressed"]
    
    results = []
    for _, row in target_df.iterrows():
        prompt = row["prompt"]
        target = row["target"]
        
        # Run iterative ablation
        res = iterative_ablate(model, sae, prompt, target, max_rounds=max_rounds, strength=strength)
        
        # Specificity check if successful
        specificity_score = None
        if res["success"] and res["rounds_used"] > 0:
            specificity_score = check_specificity(model, sae, res["features_ablated"], control_pairs, strength=strength)
        elif res["success"] and res["rounds_used"] == 0:
            specificity_score = 1.0 # No features needed, so specificity is perfect
            
        results.append({
            "prompt": prompt,
            "target": target,
            "clean_prob": row["clean_prob"],
            "clean_rank": row["rank"],
            "success": res["success"],
            "rounds_used": res["rounds_used"],
            "features_ablated": res["features_ablated"],
            "final_prob": res["final_target_prob"],
            "final_rank": res["final_target_rank"],
            "specificity": specificity_score,
            "initial_top1_token": res["initial_top1_token"],
            "final_top1_token": res["final_top1_token"]
        })
        
    return pd.DataFrame(results)


def run_grid_sweep(
    model,
    sae,
    fact_pairs: list[tuple[str, str]],
    strengths: list[float],
    batch_sizes: list[int],
    top_n: int = 30,
    max_rounds_note: str = "single-shot, not iterative"
) -> pd.DataFrame:
    """
    Runs a bounded grid sweep of single-shot joint (compound) ablation across a list of fact pairs,
    strengths, and batch sizes.
    """
    records = []
    
    for prompt, target_str in fact_pairs:
        target_token_id = get_target_token_id(model, target_str)
        tokens = model.to_tokens(prompt)
        
        # Clean baseline pass
        model.reset_hooks()
        with torch.no_grad():
            logits = model(tokens)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        baseline_top1_id = torch.argmax(probs).item()
        
        # Rank competitor features against baseline top-1 token
        competitor_features = get_top_competitor_features(
            model, sae, prompt, baseline_top1_id, top_n=top_n
        )
        ranked_ids = [fid for fid, _ in competitor_features]
        
        for strength in strengths:
            for req_size in batch_sizes:
                actual_used = min(req_size, len(ranked_ids))
                features_used = ranked_ids[:actual_used]
                
                # Apply joint ablation hook
                model.reset_hooks()
                joint_hook_fn = make_joint_ablation_hook(features_used, sae, strength=strength)
                model.add_hook("blocks.8.hook_resid_pre", joint_hook_fn)
                
                with torch.no_grad():
                    logits = model(tokens)
                probs = F.softmax(logits[0, -1, :], dim=-1)
                
                resulting_top1_id = torch.argmax(probs).item()
                resulting_top1_token = model.to_string([resulting_top1_id])
                resulting_top1_prob = probs[resulting_top1_id].item()
                
                target_prob = probs[target_token_id].item()
                sorted_indices = torch.argsort(probs, descending=True)
                target_rank = (sorted_indices == target_token_id).nonzero().item() + 1
                
                success = bool(resulting_top1_id == target_token_id)
                
                records.append({
                    "prompt": prompt,
                    "target": target_str,
                    "strength": strength,
                    "requested_batch_size": req_size,
                    "actual_batch_size_used": actual_used,
                    "features_used": features_used,
                    "resulting_top1_token": resulting_top1_token,
                    "resulting_top1_prob": resulting_top1_prob,
                    "target_prob": target_prob,
                    "target_rank": target_rank,
                    "success": success
                })
                
    model.reset_hooks()
    return pd.DataFrame(records)

