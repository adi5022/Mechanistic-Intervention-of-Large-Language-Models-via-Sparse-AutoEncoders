import sys
import os
import torch
import torch.nn.functional as F

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sae_utils import load_model_and_sae
from src.editing import (
    get_target_token_id,
    get_top_competitor_features,
    get_top_target_features,
    build_clean_context,
    CleanContext,
    HOOK_NAME,
)
from src.hooks import make_ablation_hook


def count_forward_calls(model):
    """
    Wraps model.forward exclusively to count forward calls accurately without
    double-counting higher-level helper wrappers (like run_with_hooks or run_with_cache).
    """
    counter = {"count": 0}
    orig_forward = model.forward
    
    def wrapped_forward(*args, **kwargs):
        counter["count"] += 1
        return orig_forward(*args, **kwargs)
        
    model.forward = wrapped_forward
    
    def reset():
        counter["count"] = 0
        
    def get_count():
        return counter["count"]
        
    return reset, get_count


def main():
    print("=== Step 3A Validation Script ===")
    prompt = "The location of Massachusetts Institute of Technology is in"
    target = " Cambridge"
    layer = 8
    
    print(f"Loading model and Layer {layer} SAE...")
    model, sae = load_model_and_sae(layer=layer)
    target_token_id = get_target_token_id(model, target)
    tokens = model.to_tokens(prompt)
    
    reset_counter, get_count = count_forward_calls(model)
    
    # -------------------------------------------------------------
    # 1. REFERENCE PATH (clean_ctx=None)
    # -------------------------------------------------------------
    reset_counter()
    
    # Clean baseline
    with torch.no_grad():
        ref_logits = model(tokens)
        ref_probs = F.softmax(ref_logits[0, -1, :], dim=-1)
        ref_clean_prob = ref_probs[target_token_id].item()
        ref_sorted = torch.argsort(ref_probs, descending=True)
        ref_clean_rank = (ref_sorted == target_token_id).nonzero().item() + 1
        ref_top1_id = torch.argmax(ref_probs).item()
        
    # Candidate screening & ranking
    ref_competitor_features = get_top_competitor_features(model, sae, prompt, ref_top1_id, top_n=30, clean_ctx=None)
    ref_target_features = get_top_target_features(model, sae, prompt, target_token_id, top_n=30, clean_ctx=None)
    
    ref_forward_calls = get_count()
    
    # -------------------------------------------------------------
    # 2. OPTIMIZED PATH (clean_ctx supplied)
    # -------------------------------------------------------------
    reset_counter()
    
    clean_ctx = build_clean_context(model, sae, prompt, target_token_id)
    opt_clean_prob = clean_ctx.clean_target_prob
    opt_clean_rank = clean_ctx.clean_rank
    opt_top1_id = torch.argmax(clean_ctx.clean_probs).item()
    
    opt_competitor_features = get_top_competitor_features(model, sae, prompt, opt_top1_id, top_n=30, clean_ctx=clean_ctx)
    opt_target_features = get_top_target_features(model, sae, prompt, target_token_id, top_n=30, clean_ctx=clean_ctx)
    
    opt_forward_calls = get_count()
    
    # -------------------------------------------------------------
    # 3. HOOK DELTA MATH ISOLATION TEST (Change 2 verification)
    # -------------------------------------------------------------
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        resid = cache[hook_name]  # [1, seq, d_model]
        
    test_feature_id = 8459
    test_strength = 0.3
    
    # Hook implementation under test
    hook_fn = make_ablation_hook(test_feature_id, sae, test_strength)
    hook_out = hook_fn(resid.clone(), None)
    
    # Hand-computed inline reference
    with torch.no_grad():
        f_acts = sae.encode(resid)
        baseline_recon = sae.decode(f_acts)
        m_acts = f_acts.clone()
        m_acts[..., test_feature_id] = m_acts[..., test_feature_id] * (1.0 - test_strength)
        mod_recon = sae.decode(m_acts)
        hand_out = resid + (mod_recon - baseline_recon)
        
    hook_max_diff = torch.max(torch.abs(hook_out - hand_out)).item()
    hook_math_match = hook_max_diff < 1e-9
    
    # -------------------------------------------------------------
    # 4. SIDE-BY-SIDE VERIFICATION TABLE
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print(f"{'Metric / Field':<35} | {'Reference':<20} | {'Optimized':<20} | {'Status':<10}")
    print("="*80)
    
    # Clean prob check
    prob_diff = abs(ref_clean_prob - opt_clean_prob)
    prob_status = "MATCH" if prob_diff < 1e-9 else "MISMATCH"
    print(f"{'Clean Target Probability':<35} | {ref_clean_prob:<20.9f} | {opt_clean_prob:<20.9f} | {prob_status:<10}")
    
    # Clean rank check
    rank_status = "MATCH" if ref_clean_rank == opt_clean_rank else "MISMATCH"
    print(f"{'Clean Target Rank':<35} | {ref_clean_rank:<20} | {opt_clean_rank:<20} | {rank_status:<10}")
    
    # Top competitor feature IDs
    ref_comp_fids = [fid for fid, _ in ref_competitor_features]
    opt_comp_fids = [fid for fid, _ in opt_competitor_features]
    comp_fid_status = "MATCH" if ref_comp_fids == opt_comp_fids else "MISMATCH"
    print(f"{'Competitor Feature IDs (30)':<35} | {'Count=' + str(len(ref_comp_fids)):<20} | {'Count=' + str(len(opt_comp_fids)):<20} | {comp_fid_status:<10}")
    
    # Competitor prob deltas float check
    ref_comp_deltas = [d for _, d in ref_competitor_features]
    opt_comp_deltas = [d for _, d in opt_competitor_features]
    max_comp_delta_diff = max(abs(r - o) for r, o in zip(ref_comp_deltas, opt_comp_deltas))
    comp_delta_status = "MATCH" if max_comp_delta_diff < 1e-9 else "MISMATCH"
    print(f"{'Competitor Max Delta Diff':<35} | {0.0:<20.9f} | {max_comp_delta_diff:<20.9e} | {comp_delta_status:<10}")
    
    # Top target feature IDs
    ref_tgt_fids = [fid for fid, _ in ref_target_features]
    opt_tgt_fids = [fid for fid, _ in opt_target_features]
    tgt_fid_status = "MATCH" if ref_tgt_fids == opt_tgt_fids else "MISMATCH"
    print(f"{'Target Feature IDs (30)':<35} | {'Count=' + str(len(ref_tgt_fids)):<20} | {'Count=' + str(len(opt_tgt_fids)):<20} | {tgt_fid_status:<10}")
    
    # Target prob deltas float check
    ref_tgt_deltas = [d for _, d in ref_target_features]
    opt_tgt_deltas = [d for _, d in opt_target_features]
    max_tgt_delta_diff = max(abs(r - o) for r, o in zip(ref_tgt_deltas, opt_tgt_deltas))
    tgt_delta_status = "MATCH" if max_tgt_delta_diff < 1e-9 else "MISMATCH"
    print(f"{'Target Max Delta Diff':<35} | {0.0:<20.9f} | {max_tgt_delta_diff:<20.9e} | {tgt_delta_status:<10}")

    # Ablation hook delta math check
    hook_status = "MATCH" if hook_math_match else "MISMATCH"
    print(f"{'Ablation Hook Math (max diff)':<35} | {0.0:<20.9f} | {hook_max_diff:<20.9e} | {hook_status:<10}")
    
    print("="*80)
    print(f"\nIsolated Function Calls (Screening + Ranking stages):")
    print(f"  Reference path forward calls: {ref_forward_calls}")
    print(f"  Optimized path forward calls: {opt_forward_calls}")
    print(f"  Forward call reduction:       {ref_forward_calls} -> {opt_forward_calls}")
    print("="*80)

    # -------------------------------------------------------------
    # 5. FULL BENCHMARK RUNNER TRACE VERIFICATION (126 -> 122)
    # -------------------------------------------------------------
    from src.benchmark.layer_benchmark_runner import run_layer_benchmark
    
    reset_counter()
    bench_results = run_layer_benchmark(
        prompts=prompt,
        target=target,
        layers=[8],
        use_safety=True
    )
    bench_layer_data = bench_results["prompts"][0]["layers"][0]
    bench_fwd_acc = bench_layer_data["forward_passes"]
    
    print("\n" + "="*80)
    print("Full Benchmark Runner Execution Trace Accounting (Post-Step 3A):")
    print("="*80)
    for k, v in bench_fwd_acc.items():
        print(f"  {k:<30}: {v}")
    print("="*80)
    
    all_passed = (
        prob_status == "MATCH" and
        rank_status == "MATCH" and
        comp_fid_status == "MATCH" and
        comp_delta_status == "MATCH" and
        tgt_fid_status == "MATCH" and
        tgt_delta_status == "MATCH" and
        hook_status == "MATCH" and
        bench_fwd_acc["total_model_forwards"] == 122
    )
    
    if all_passed:
        print("\nSUCCESS: All Gate 3A checks passed! (Exact 1e-9 precision, total_model_forwards = 122)")
    else:
        print("\nFAILURE: Mismatch detected in Gate 3A validation checks.")
        sys.exit(1)


if __name__ == "__main__":
    main()
