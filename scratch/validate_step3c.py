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
    check_target_safe,
    check_boost_safe,
    check_target_safe_batch,
    check_boost_safe_batch,
    build_clean_context,
    CleanContext,
    HOOK_NAME,
)
from src.benchmark.layer_benchmark_runner import run_layer_benchmark


def count_forward_calls(model):
    """
    Wraps model.forward exclusively to count forward calls accurately.
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
    print("=== Step 3C Validation Script (Batched Safety Filters) ===")
    prompt = "The location of Massachusetts Institute of Technology is in"
    target = " Cambridge"
    layer = 8
    mute_strength = 0.3
    boost_strength = 0.5

    print(f"Loading model and Layer {layer} SAE...")
    model, sae = load_model_and_sae(layer=layer)
    target_token_id = get_target_token_id(model, target)

    reset_counter, get_count = count_forward_calls(model)

    # 1. Build clean context once
    clean_ctx = build_clean_context(model, sae, prompt, target_token_id)
    top1_id = torch.argmax(clean_ctx.clean_probs).item()

    # 2. Get top 30 competitor and target features (batched ranking)
    competitor_features = get_top_competitor_features(
        model, sae, prompt, top1_id, top_n=30, clean_ctx=clean_ctx, use_batched=True
    )
    target_features = get_top_target_features(
        model, sae, prompt, target_token_id, top_n=30, clean_ctx=clean_ctx, use_batched=True
    )

    comp_fids = [fid for fid, _ in competitor_features]
    tgt_fids = [fid for fid, _ in target_features]

    # -------------------------------------------------------------
    # 3. SEQUENTIAL SAFETY FILTER CHECKS
    # -------------------------------------------------------------
    seq_comp_safe_results = []
    for fid in comp_fids:
        is_safe, delta = check_target_safe(
            model, sae, prompt, fid, target_token_id, strength=mute_strength,
            clean_target_prob=clean_ctx.clean_target_prob, clean_rank=clean_ctx.clean_rank
        )
        seq_comp_safe_results.append((is_safe, delta))

    seq_tgt_safe_results = []
    for fid in tgt_fids:
        is_safe, delta, rank_imp = check_boost_safe(
            model, sae, prompt, fid, target_token_id, strength=boost_strength,
            clean_target_prob=clean_ctx.clean_target_prob, clean_rank=clean_ctx.clean_rank
        )
        seq_tgt_safe_results.append((is_safe, delta, rank_imp))

    # -------------------------------------------------------------
    # 4. BATCHED SAFETY FILTER CHECKS
    # -------------------------------------------------------------
    batch_comp_safe_results = check_target_safe_batch(
        model, sae, clean_ctx, comp_fids, target_token_id, strength=mute_strength
    )

    batch_tgt_safe_results = check_boost_safe_batch(
        model, sae, clean_ctx, tgt_fids, target_token_id, strength=boost_strength
    )

    # -------------------------------------------------------------
    # 5. COMPARE 60 SAFETY BOOLEANS ELEMENT-BY-ELEMENT
    # -------------------------------------------------------------
    comp_boolean_matches = 0
    comp_row_diffs = []
    for idx, (fid, (s_safe, s_delta), (b_safe, b_delta)) in enumerate(zip(comp_fids, seq_comp_safe_results, batch_comp_safe_results)):
        match = (s_safe == b_safe)
        if match:
            comp_boolean_matches += 1
        else:
            comp_row_diffs.append(f"Row {idx} (fid={fid}): seq_safe={s_safe} (delta={s_delta:.6e}) vs batch_safe={b_safe} (delta={b_delta:.6e})")

    tgt_boolean_matches = 0
    tgt_row_diffs = []
    for idx, (fid, (s_safe, s_delta, s_imp), (b_safe, b_delta, b_imp)) in enumerate(zip(tgt_fids, seq_tgt_safe_results, batch_tgt_safe_results)):
        match = (s_safe == b_safe)
        if match:
            tgt_boolean_matches += 1
        else:
            tgt_row_diffs.append(f"Row {idx} (fid={fid}): seq_safe={s_safe} (delta={s_delta:.6e}) vs batch_safe={b_safe} (delta={b_delta:.6e})")

    total_boolean_matches = comp_boolean_matches + tgt_boolean_matches

    # -------------------------------------------------------------
    # 6. PRINT FULL 60-ROW SAFETY-DECISION AGREEMENT TABLE
    # -------------------------------------------------------------
    print("\n" + "="*95)
    print(f"Competitor Mute Safety Filter Decisions (30 Features, strength={mute_strength}):")
    print("="*95)
    print(f"{'Index':<5} | {'Feature ID':<10} | {'Seq Safe':<10} | {'Batch Safe':<10} | {'Seq Delta':<14} | {'Batch Delta':<14} | {'Match':<8}")
    print("-" * 95)
    for idx, (fid, (s_safe, s_delta), (b_safe, b_delta)) in enumerate(zip(comp_fids, seq_comp_safe_results, batch_comp_safe_results)):
        match_str = "MATCH" if s_safe == b_safe else "MISMATCH"
        print(f"{idx:<5} | {fid:<10} | {str(s_safe):<10} | {str(b_safe):<10} | {s_delta:<14.6e} | {b_delta:<14.6e} | {match_str:<8}")

    print("\n" + "="*95)
    print(f"Target Boost Safety Filter Decisions (30 Features, strength={boost_strength}):")
    print("="*95)
    print(f"{'Index':<5} | {'Feature ID':<10} | {'Seq Safe':<10} | {'Batch Safe':<10} | {'Seq Delta':<14} | {'Batch Delta':<14} | {'Match':<8}")
    print("-" * 95)
    for idx, (fid, (s_safe, s_delta, _), (b_safe, b_delta, _)) in enumerate(zip(tgt_fids, seq_tgt_safe_results, batch_tgt_safe_results)):
        match_str = "MATCH" if s_safe == b_safe else "MISMATCH"
        print(f"{idx:<5} | {fid:<10} | {str(s_safe):<10} | {str(b_safe):<10} | {s_delta:<14.6e} | {b_delta:<14.6e} | {match_str:<8}")

    # -------------------------------------------------------------
    # 7. RUN END-TO-END BENCHMARK RUNNER (Sequential vs Batched)
    # -------------------------------------------------------------
    print("\n" + "="*95)
    print("End-to-End Benchmark Runner Trace & Metric Validation:")
    print("="*95)

    reset_counter()
    seq_bench = run_layer_benchmark(
        prompts=prompt, target=target, layers=[layer], use_safety=True, use_batched_ranking=False
    )
    seq_fwd_calls = get_count()
    seq_data = seq_bench["prompts"][0]["layers"][0]

    reset_counter()
    batch_bench = run_layer_benchmark(
        prompts=prompt, target=target, layers=[layer], use_safety=True, use_batched_ranking=True
    )
    batch_fwd_calls = get_count()
    batch_data = batch_bench["prompts"][0]["layers"][0]

    print(f"Sequential Benchmark Run:")
    print(f"  Clean Rank: {seq_data['clean_rank']} -> Final Rank: {seq_data['final_rank']} (Improvement: +{seq_data['rank_improvement']})")
    print(f"  Clean Prob: {seq_data['clean_probability']:.6f} -> Final Prob: {seq_data['final_probability']:.6f} (Gain: +{seq_data['probability_gain']*100:.3f}%)")
    print(f"  Selected Mute Features:  {seq_data['mute_features']}")
    print(f"  Selected Boost Features: {seq_data['boost_features']}")
    print(f"  Total Model Forward Calls: {seq_bench['prompts'][0]['layers'][0]['forward_passes']['total_model_forwards']} (Measured: {seq_fwd_calls})")

    print(f"\nBatched Benchmark Run:")
    print(f"  Clean Rank: {batch_data['clean_rank']} -> Final Rank: {batch_data['final_rank']} (Improvement: +{batch_data['rank_improvement']})")
    print(f"  Clean Prob: {batch_data['clean_probability']:.6f} -> Final Prob: {batch_data['final_probability']:.6f} (Gain: +{batch_data['probability_gain']*100:.3f}%)")
    print(f"  Selected Mute Features:  {batch_data['mute_features']}")
    print(f"  Selected Boost Features: {batch_data['boost_features']}")
    print(f"  Total Model Forward Calls: {batch_bench['prompts'][0]['layers'][0]['forward_passes']['total_model_forwards']} (Measured: {batch_fwd_calls})")

    print("\nForward Pass Accounting Comparison:")
    for k in seq_data['forward_passes']:
        seq_val = seq_data['forward_passes'][k]
        batch_val = batch_data['forward_passes'][k]
        print(f"  {k:<28}: Seq = {seq_val:<5} | Batched = {batch_val}")
    print("="*95)

    all_booleans_match = (total_boolean_matches == 60)
    outcomes_match = (
        seq_data['final_rank'] == batch_data['final_rank'] and
        seq_data['mute_features'] == batch_data['mute_features'] and
        seq_data['boost_features'] == batch_data['boost_features'] and
        batch_data['forward_passes']['total_model_forwards'] == 6
    )

    if all_booleans_match and outcomes_match:
        print("\nSUCCESS: All Gate 3C checks passed! (100% agreement on all 60 safety booleans, forward calls = 6)")
        sys.exit(0)
    else:
        print("\nFAILURE: Mismatch detected in Gate 3C validation checks.")
        if not all_booleans_match:
            print(f"  Safety booleans matched: {total_boolean_matches} / 60")
        sys.exit(1)


if __name__ == "__main__":
    main()
