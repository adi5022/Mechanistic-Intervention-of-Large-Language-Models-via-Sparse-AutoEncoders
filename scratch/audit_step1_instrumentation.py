"""
Temporary Audit Instrumentation Script (Step 1 Audit Only)
Measures execution path, exact model forward calls, stage-by-stage GPU/CPU timings, and redundancies.
DOES NOT MODIFY ANY ALGORITHMS, HOOKS, OR MODEL TENSORS.
"""

import time
import json
import torch
import torch.nn.functional as F
from collections import defaultdict
from src.sae_utils import load_model_and_sae
from src.benchmark.layer_benchmark_runner import run_layer_benchmark
import src.editing as editing

# 1. Forward Pass Counter & Profiler
class ForwardPassTracker:
    def __init__(self):
        self.counts = defaultdict(int)
        self.runtimes = defaultdict(float)
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

# 2. Patch model forward / run_with_hooks to intercept and count actual forward passes
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

# 3. Patch editing.py helper functions to set tracker category
orig_get_top_active_features = editing.get_top_active_features
orig_get_top_competitor_features = editing.get_top_competitor_features
orig_get_top_target_features = editing.get_top_target_features
orig_check_target_safe = editing.check_target_safe
orig_check_boost_safe = editing.check_boost_safe

def patched_get_top_active_features(model, sae, prompt, top_n=20):
    prev_cat = tracker.current_category
    tracker.current_category = "Feature Activation / Candidate Discovery"
    res = orig_get_top_active_features(model, sae, prompt, top_n=top_n)
    tracker.current_category = prev_cat
    return res

def patched_get_top_competitor_features(model, sae, prompt, current_top_token_id, top_n=20):
    prev_cat = tracker.current_category
    tokens = model.to_tokens(prompt)
    
    tracker.current_category = "Competitor Causal Ranking (Internal Clean Pass)"
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        clean_competitor_prob = clean_probs[current_top_token_id].item()
        
    active_features = editing.get_top_active_features(model, sae, prompt, top_n=top_n)
    
    results = []
    for feature_id, activation in active_features:
        hook_fn = editing.make_ablation_hook(feature_id, sae, strength=1.0)
        hook_name = getattr(sae.cfg, "hook_name", editing.HOOK_NAME)
        
        tracker.current_category = "Competitor Causal Feature Ranking"
        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_fn)]
            )
            ablated_probs = F.softmax(ablated_logits[0, -1, :], dim=-1)
            ablated_competitor_prob = ablated_probs[current_top_token_id].item()
            prob_delta = ablated_competitor_prob - clean_competitor_prob
            
        results.append((feature_id, prob_delta))
        
    results.sort(key=lambda x: x[1])
    tracker.current_category = prev_cat
    return results

def patched_get_top_target_features(model, sae, prompt, target_token_id, top_n=30):
    prev_cat = tracker.current_category
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", editing.HOOK_NAME)
    
    tracker.current_category = "Target Causal Ranking (Internal Clean Pass)"
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        clean_target_prob = clean_probs[target_token_id].item()
        
    active_features = editing.get_top_active_features(model, sae, prompt, top_n=top_n)
    
    results = []
    for feature_id, activation in active_features:
        hook_fn = editing.make_ablation_hook(feature_id, sae, strength=1.0)
        
        tracker.current_category = "Target Causal Feature Ranking"
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
    tracker.current_category = prev_cat
    return results

def patched_check_target_safe(model, sae, prompt, feature_id, target_token_id, strength=0.3):
    prev_cat = tracker.current_category
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", editing.HOOK_NAME)
    
    tracker.current_category = "Competitor Safety Filtering (Internal Clean Pass)"
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        clean_target_prob = clean_probs[target_token_id].item()
        clean_sorted_indices = torch.argsort(clean_probs, descending=True)
        clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1
        
    hook_fn = editing.make_ablation_hook(feature_id, sae, strength=strength)
    
    tracker.current_category = "Competitor Safety Filtering"
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
    
    tracker.current_category = prev_cat
    return is_safe, target_prob_delta

def patched_check_boost_safe(model, sae, prompt, feature_id, target_token_id, strength=0.5):
    from src.hooks import make_signed_ablation_hook
    prev_cat = tracker.current_category
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", editing.HOOK_NAME)
    
    tracker.current_category = "Target Safety Filtering (Internal Clean Pass)"
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        clean_target_prob = clean_probs[target_token_id].item()
        clean_sorted_indices = torch.argsort(clean_probs, descending=True)
        clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1
        
    hook_fn = make_signed_ablation_hook(feature_id, sae, strength=strength)
    
    tracker.current_category = "Target Safety Filtering"
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
    
    tracker.current_category = prev_cat
    return is_safe, target_prob_delta, rank_improvement

editing.get_top_competitor_features = patched_get_top_competitor_features
editing.get_top_target_features = patched_get_top_target_features
editing.check_target_safe = patched_check_target_safe
editing.check_boost_safe = patched_check_boost_safe
editing.get_top_active_features = patched_get_top_active_features

def run_audit():
    prompt = "The location of Massachusetts Institute of Technology is in"
    target = "Cambridge"
    layer = 8
    
    print("=== STARTING STEP 1 EXECUTION COST AUDIT ===")
    print(f"Prompt: {prompt}")
    print(f"Target: {target}")
    print(f"Layer: {layer}")
    
    # Custom model loader that instruments the model instance
    def instrumented_loader(l):
        model, sae = load_model_and_sae(layer=l)
        instrument_model(model)
        return model, sae

    # Sync CUDA before starting wall-clock measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start_wall = time.perf_counter()

    # Category for clean baseline in layer_benchmark_runner
    tracker.current_category = "Clean baseline"
    
    res = run_layer_benchmark(
        prompts=[{"prompt": prompt, "target": target}],
        layers=[layer],
        mute_strength=0.3,
        boost_strength=0.5,
        mute_batch_size=3,
        boost_batch_size=3,
        use_safety=True,
        algorithm="hybrid",
        model_sae_loader=instrumented_loader,
        results_dir="benchmark_results_audit"
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end_wall = time.perf_counter()
    total_wall_ms = (t_end_wall - t_start_wall) * 1000.0

    print("\n=== AUDIT RESULTS ===")
    print(f"Total Wall-Clock Runtime: {total_wall_ms:.2f} ms ({total_wall_ms/1000.0:.2f} s)")
    print(f"Total Model Forward Passes Counted: {tracker.total_forwards}")
    print("\n--- Forward-Pass & Runtime Breakdown ---")
    print(f"{'Stage Category':<50} | {'Calls':<8} | {'Total ms':<12} | {'Avg ms/call':<10}")
    print("-" * 88)
    
    for cat in [
        "Clean baseline",
        "Feature Activation / Candidate Discovery",
        "Competitor Causal Ranking (Internal Clean Pass)",
        "Competitor Causal Feature Ranking",
        "Target Causal Ranking (Internal Clean Pass)",
        "Target Causal Feature Ranking",
        "Competitor Safety Filtering (Internal Clean Pass)",
        "Competitor Safety Filtering",
        "Target Safety Filtering (Internal Clean Pass)",
        "Target Safety Filtering",
        "Final Intervention Pass",
        "Other"
    ]:
        c = tracker.counts[cat]
        tot = tracker.runtimes[cat]
        avg = (tot / c) if c > 0 else 0.0
        if c > 0 or "Intervention" in cat:
            print(f"{cat:<50} | {c:<8} | {tot:<12.2f} | {avg:<10.2f}")

    print("-" * 88)
    
    # Save audit summary JSON
    audit_data = {
        "wall_clock_ms": total_wall_ms,
        "total_forward_passes": tracker.total_forwards,
        "category_counts": dict(tracker.counts),
        "category_runtimes_ms": dict(tracker.runtimes),
        "benchmark_output": res
    }
    with open("audit_step1_results.json", "w") as f:
        json.dump(audit_data, f, indent=2)
    print("Audit JSON saved to audit_step1_results.json")

if __name__ == "__main__":
    run_audit()
