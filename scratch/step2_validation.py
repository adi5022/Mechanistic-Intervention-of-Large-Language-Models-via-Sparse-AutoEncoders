"""
Step 2 Validation & Performance Comparison Script
Compares Reference vs Optimized implementations for equivalence and execution cost.
"""

import time
import json
import torch
import torch.nn.functional as F
from collections import defaultdict
from src.sae_utils import load_model_and_sae
import src.editing as editing

class ForwardPassTracker:
    def __init__(self):
        self.counts = defaultdict(int)
        self.runtimes = defaultdict(float)
        self.current_category = "Other"
        self.total_forwards = 0

    def reset(self):
        self.counts.clear()
        self.runtimes.clear()
        self.current_category = "Other"
        self.total_forwards = 0

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def record_forward(self, category: str, fn, *args, **kwargs):
        self.sync()
        t0 = time.perf_counter()
        res = fn(*args, **kwargs)
        self.sync()
        elapsed = (time.perf_counter() - t0) * 1000.0
        
        self.counts[category] += 1
        self.runtimes[category] += elapsed
        self.total_forwards += 1
        return res

tracker = ForwardPassTracker()

def instrument_model(model):
    orig_forward = model.forward
    orig_run_with_hooks = model.run_with_hooks
    orig_run_with_cache = model.run_with_cache

    def tracked_forward(*args, **kwargs):
        return tracker.record_forward(tracker.current_category, orig_forward, *args, **kwargs)

    def tracked_run_with_hooks(*args, **kwargs):
        return tracker.record_forward(tracker.current_category, orig_run_with_hooks, *args, **kwargs)

    def tracked_run_with_cache(*args, **kwargs):
        return tracker.record_forward(tracker.current_category, orig_run_with_cache, *args, **kwargs)

    model.forward = tracked_forward
    model.run_with_hooks = tracked_run_with_hooks
    model.run_with_cache = tracked_run_with_cache
    return model

def run_step2_validation():
    prompt = "The location of Massachusetts Institute of Technology is in"
    target = "Cambridge"
    target_str = " " + target
    layer = 8
    
    print("=== STARTING STEP 2 VALIDATION & BENCHMARK ===")
    
    model, sae = load_model_and_sae(layer=layer)
    instrument_model(model)

    # Clean Baseline Pass
    tracker.reset()
    tracker.current_category = "Clean baseline"
    target_token_id = editing.get_target_token_id(model, target_str)
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", f"blocks.{layer}.hook_resid_pre")
    
    model.reset_hooks()
    with torch.no_grad():
        logits = model(tokens)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    
    current_top1_id = torch.argmax(probs).item()
    top_prediction_before = model.to_string([current_top1_id])
    clean_probability = probs[target_token_id].item()
    clean_sorted_indices = torch.argsort(probs, descending=True)
    clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1
    
    # Feature Candidate Selection
    competitor_features = editing.get_top_competitor_features(model, sae, prompt, current_top1_id, top_n=30)
    target_features = editing.get_top_target_features(model, sae, prompt, target_token_id, top_n=30)

    # 1. RUN REFERENCE SAFETY PATH (Forcing clean_target_prob=None to recompute clean baseline)
    tracker.reset()
    tracker.current_category = "Safety Filtering (Reference Path)"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0_ref = time.perf_counter()

    ref_comp_ids = []
    for fid, _ in competitor_features:
        is_safe, _ = editing.check_target_safe(
            model, sae, prompt, fid, target_token_id, strength=0.3,
            clean_target_prob=None, clean_rank=None  # FORCING REFERENCE RECOMPUTE
        )
        if is_safe:
            ref_comp_ids.append(fid)
    ref_mute_batch = ref_comp_ids[:3]

    ref_target_ids = []
    for fid, _ in target_features:
        is_safe, _, _ = editing.check_boost_safe(
            model, sae, prompt, fid, target_token_id, strength=0.5,
            clean_target_prob=None, clean_rank=None  # FORCING REFERENCE RECOMPUTE
        )
        if is_safe:
            ref_target_ids.append(fid)
    ref_boost_batch = ref_target_ids[:3]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_ref_safety_ms = (time.perf_counter() - t0_ref) * 1000.0
    ref_forwards = tracker.total_forwards

    # 2. RUN OPTIMIZED SAFETY PATH (Passing precomputed clean_probability and clean_rank)
    tracker.reset()
    tracker.current_category = "Safety Filtering (Optimized Path)"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0_opt = time.perf_counter()

    opt_comp_ids = []
    for fid, _ in competitor_features:
        is_safe, _ = editing.check_target_safe(
            model, sae, prompt, fid, target_token_id, strength=0.3,
            clean_target_prob=clean_probability, clean_rank=clean_rank  # PASSED PRECOMPUTED
        )
        if is_safe:
            opt_comp_ids.append(fid)
    opt_mute_batch = opt_comp_ids[:3]

    opt_target_ids = []
    for fid, _ in target_features:
        is_safe, _, _ = editing.check_boost_safe(
            model, sae, prompt, fid, target_token_id, strength=0.5,
            clean_target_prob=clean_probability, clean_rank=clean_rank  # PASSED PRECOMPUTED
        )
        if is_safe:
            opt_target_ids.append(fid)
    opt_boost_batch = opt_target_ids[:3]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_opt_safety_ms = (time.perf_counter() - t0_opt) * 1000.0
    opt_forwards = tracker.total_forwards

    # 3. Final Intervention Pass comparison
    from src.hooks import make_mute_and_boost_hook
    model.reset_hooks()
    joint_hook_ref = make_mute_and_boost_hook(ref_mute_batch, 0.3, ref_boost_batch, 0.5, sae)
    model.add_hook(hook_name, joint_hook_ref)
    with torch.no_grad():
        logits_ref = model(tokens)
    probs_ref = F.softmax(logits_ref[0, -1, :], dim=-1)
    ref_final_prob = probs_ref[target_token_id].item()
    ref_final_rank = (torch.argsort(probs_ref, descending=True) == target_token_id).nonzero().item() + 1
    model.reset_hooks()

    joint_hook_opt = make_mute_and_boost_hook(opt_mute_batch, 0.3, opt_boost_batch, 0.5, sae)
    model.add_hook(hook_name, joint_hook_opt)
    with torch.no_grad():
        logits_opt = model(tokens)
    probs_opt = F.softmax(logits_opt[0, -1, :], dim=-1)
    opt_final_prob = probs_opt[target_token_id].item()
    opt_final_rank = (torch.argsort(probs_opt, descending=True) == target_token_id).nonzero().item() + 1
    model.reset_hooks()

    print("\n=== EQUIVALENCE & PERFORMANCE METRICS ===")
    print(f"Reference Safety Forwards: {ref_forwards}")
    print(f"Optimized Safety Forwards: {opt_forwards}")
    print(f"Forwards Eliminated:       {ref_forwards - opt_forwards}")
    print(f"Reference Safety Runtime:  {t_ref_safety_ms:.2f} ms")
    print(f"Optimized Safety Runtime:  {t_opt_safety_ms:.2f} ms")
    print(f"Safety Speedup:            {t_ref_safety_ms / t_opt_safety_ms:.2f}x faster")
    
    val_data = {
        "clean_probability": (clean_probability, clean_probability),
        "clean_rank": (clean_rank, clean_rank),
        "competitor_safety_decisions_match": ref_comp_ids == opt_comp_ids,
        "target_safety_decisions_match": ref_target_ids == opt_target_ids,
        "mute_batch": (ref_mute_batch, opt_mute_batch),
        "boost_batch": (ref_boost_batch, opt_boost_batch),
        "final_probability": (ref_final_prob, opt_final_prob),
        "final_rank": (ref_final_rank, opt_final_rank),
        "ref_safety_forwards": ref_forwards,
        "opt_safety_forwards": opt_forwards,
        "ref_safety_ms": t_ref_safety_ms,
        "opt_safety_ms": t_opt_safety_ms
    }
    with open("scratch/step2_validation.json", "w") as f:
        json.dump(val_data, f, indent=2)
    print("\nSaved step2_validation.json")

if __name__ == "__main__":
    run_step2_validation()
