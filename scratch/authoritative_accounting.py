"""
Authoritative Single-Pass Forward Pass Counter Script
Measures the EXACT call stack and model forward pass breakdown for:
1) Reference Path (unoptimized check_target_safe / check_boost_safe recomputing clean baseline)
2) Optimized Path (Step 2 passing precomputed clean baseline)
"""

import time
import json
import torch
import torch.nn.functional as F
from collections import defaultdict
from src.sae_utils import load_model_and_sae
from src.benchmark.layer_benchmark_runner import run_layer_benchmark
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

orig_get_top_active_features = editing.get_top_active_features
orig_get_top_competitor_features = editing.get_top_competitor_features
orig_get_top_target_features = editing.get_top_target_features

def tracked_get_top_active_features(model, sae, prompt, top_n=20):
    prev_cat = tracker.current_category
    tracker.current_category = "Stage 3: Candidate Discovery (run_with_cache)"
    res = orig_get_top_active_features(model, sae, prompt, top_n=top_n)
    tracker.current_category = prev_cat
    return res

def tracked_get_top_competitor_features(model, sae, prompt, current_top_token_id, top_n=20):
    tokens = model.to_tokens(prompt)
    
    tracker.current_category = "Stage 4: Competitor Causal Ranking (Internal Clean Baseline)"
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        clean_competitor_prob = clean_probs[current_top_token_id].item()
        
    active_features = orig_get_top_active_features(model, sae, prompt, top_n=top_n)
    
    results = []
    for feature_id, activation in active_features:
        hook_fn = editing.make_ablation_hook(feature_id, sae, strength=1.0)
        hook_name = getattr(sae.cfg, "hook_name", editing.HOOK_NAME)
        
        tracker.current_category = "Stage 4: Competitor Causal Ranking (Ablation Passes)"
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
    return results

def tracked_get_top_target_features(model, sae, prompt, target_token_id, top_n=30):
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", editing.HOOK_NAME)
    
    tracker.current_category = "Stage 5: Target Causal Ranking (Internal Clean Baseline)"
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        clean_target_prob = clean_probs[target_token_id].item()
        
    active_features = orig_get_top_active_features(model, sae, prompt, top_n=top_n)
    
    results = []
    for feature_id, activation in active_features:
        hook_fn = editing.make_ablation_hook(feature_id, sae, strength=1.0)
        
        tracker.current_category = "Stage 5: Target Causal Ranking (Ablation Passes)"
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

def tracked_check_target_safe(model, sae, prompt, feature_id, target_token_id, strength=0.3, clean_target_prob=None, clean_rank=None):
    prev_cat = tracker.current_category
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", editing.HOOK_NAME)
    
    if clean_target_prob is None or clean_rank is None:
        tracker.current_category = "Stage 6: Competitor Safety Filtering (Redundant Clean Baseline)"
        with torch.no_grad():
            clean_logits = model(tokens)
            clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
            clean_target_prob = clean_probs[target_token_id].item()
            clean_sorted_indices = torch.argsort(clean_probs, descending=True)
            clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1
        
    hook_fn = editing.make_ablation_hook(feature_id, sae, strength=strength)
    
    tracker.current_category = "Stage 6: Competitor Safety Filtering (Ablation Passes)"
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

def tracked_check_boost_safe(model, sae, prompt, feature_id, target_token_id, strength=0.5, clean_target_prob=None, clean_rank=None):
    from src.hooks import make_signed_ablation_hook
    prev_cat = tracker.current_category
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    
    if clean_target_prob is None or clean_rank is None:
        tracker.current_category = "Stage 7: Target Safety Filtering (Redundant Clean Baseline)"
        with torch.no_grad():
            clean_logits = model(tokens)
            clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
            clean_target_prob = clean_probs[target_token_id].item()
            clean_sorted_indices = torch.argsort(clean_probs, descending=True)
            clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1
        
    hook_fn = make_signed_ablation_hook([feature_id], sae, strength=+strength)
    
    tracker.current_category = "Stage 7: Target Safety Filtering (Boost Passes)"
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

editing.get_top_competitor_features = tracked_get_top_competitor_features
editing.get_top_target_features = tracked_get_top_target_features
editing.check_target_safe = tracked_check_target_safe
editing.check_boost_safe = tracked_check_boost_safe
editing.get_top_active_features = tracked_get_top_active_features

def run_accounting():
    prompt = "The location of Massachusetts Institute of Technology is in"
    target = "Cambridge"
    layer = 8

    def instrumented_loader(l):
        model, sae = load_model_and_sae(layer=l)
        instrument_model(model)
        return model, sae

    print("=== AUTHORITATIVE FORWARD-PASS ACCOUNTING AUDIT ===")

    # 1. REFERENCE PATH RUN (FORCING CLEAN_TARGET_PROB=NONE IN SAFETY CHECKS)
    # Monkey-patch check_target_safe / check_boost_safe in runner loop to pass None
    tracker.reset()
    tracker.current_category = "Stage 2: Clean Baseline (layer_benchmark_runner)"
    
    def ref_check_target_safe(model, sae, prompt, feature_id, target_token_id, strength=0.3, **kwargs):
        return tracked_check_target_safe(model, sae, prompt, feature_id, target_token_id, strength=strength, clean_target_prob=None, clean_rank=None)

    def ref_check_boost_safe(model, sae, prompt, feature_id, target_token_id, strength=0.5, **kwargs):
        return tracked_check_boost_safe(model, sae, prompt, feature_id, target_token_id, strength=strength, clean_target_prob=None, clean_rank=None)

    import src.benchmark.layer_benchmark_runner as runner_module
    runner_module.check_target_safe = ref_check_target_safe
    runner_module.check_boost_safe = ref_check_boost_safe
    editing.check_target_safe = ref_check_target_safe
    editing.check_boost_safe = ref_check_boost_safe

    res_ref = run_layer_benchmark(
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

    ref_counts = dict(tracker.counts)
    ref_total = tracker.total_forwards

    # 2. OPTIMIZED PATH RUN (PASSING PRECOMPUTED CLEAN BASELINE IN SAFETY CHECKS)
    tracker.reset()
    tracker.current_category = "Stage 2: Clean Baseline (layer_benchmark_runner)"
    runner_module.check_target_safe = tracked_check_target_safe
    runner_module.check_boost_safe = tracked_check_boost_safe
    editing.check_target_safe = tracked_check_target_safe
    editing.check_boost_safe = tracked_check_boost_safe

    res_opt = run_layer_benchmark(
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

    opt_counts = dict(tracker.counts)
    opt_total = tracker.total_forwards

    output_data = {
        "reference_total": ref_total,
        "optimized_total": opt_total,
        "reference_breakdown": ref_counts,
        "optimized_breakdown": opt_counts
    }
    with open("scratch/authoritative_breakdown.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("\nSaved scratch/authoritative_breakdown.json")

if __name__ == "__main__":
    run_accounting()
