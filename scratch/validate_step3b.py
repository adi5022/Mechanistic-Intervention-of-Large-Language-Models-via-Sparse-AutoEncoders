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
from src.batched_eval import batched_ablation_probs, batched_ablation_probs_and_ranks
from src.fact_bank import FACT_BANK


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


def compute_stability_margin(prob_deltas: list[float]) -> float:
    if len(prob_deltas) < 2:
        return float("inf")
    # Smallest gap between consecutive values in sorted list
    sorted_deltas = sorted(prob_deltas)
    gaps = [sorted_deltas[i+1] - sorted_deltas[i] for i in range(len(sorted_deltas)-1)]
    return min(gaps) if gaps else float("inf")


def compare_ordered_lists(seq_list: list[int], batch_list: list[int]) -> tuple[bool, list[str]]:
    """
    Compares two feature ID lists element-by-element in order.
    Returns (is_exact_match, list_of_mismatch_descriptions).
    """
    if len(seq_list) != len(batch_list):
        return False, [f"Length mismatch: seq length {len(seq_list)} != batch length {len(batch_list)}"]

    mismatches = []
    for idx, (s_val, b_val) in enumerate(zip(seq_list, batch_list)):
        if s_val != b_val:
            mismatches.append(f"Index {idx}: sequential={s_val} != batched={b_val}")

    return len(mismatches) == 0, mismatches


def run_single_pass(model, sae, prompt, target_token_id, hook_name, reset_counter, get_count):
    clean_ctx = build_clean_context(model, sae, prompt, target_token_id)
    top1_id = torch.argmax(clean_ctx.clean_probs).item()

    reset_counter()
    seq_comp = get_top_competitor_features(model, sae, prompt, top1_id, top_n=30, clean_ctx=clean_ctx, use_batched=False)
    seq_tgt = get_top_target_features(model, sae, prompt, target_token_id, top_n=30, clean_ctx=clean_ctx, use_batched=False)
    seq_forward_calls = get_count()

    reset_counter()
    batch_comp = get_top_competitor_features(model, sae, prompt, top1_id, top_n=30, clean_ctx=clean_ctx, use_batched=True)
    batch_tgt = get_top_target_features(model, sae, prompt, target_token_id, top_n=30, clean_ctx=clean_ctx, use_batched=True)
    batch_forward_calls = get_count()

    seq_comp_fids = [f for f, _ in seq_comp]
    batch_comp_fids = [f for f, _ in batch_comp]
    seq_comp_deltas = [d for _, d in seq_comp]
    batch_comp_deltas = [d for _, d in batch_comp]

    seq_tgt_fids = [f for f, _ in seq_tgt]
    batch_tgt_fids = [f for f, _ in batch_tgt]
    seq_tgt_deltas = [d for _, d in seq_tgt]
    batch_tgt_deltas = [d for _, d in batch_tgt]

    comp_margin = compute_stability_margin(batch_comp_deltas)
    tgt_margin = compute_stability_margin(batch_tgt_deltas)
    overall_margin = min(comp_margin, tgt_margin)

    return {
        "clean_ctx": clean_ctx,
        "seq_comp_fids": seq_comp_fids,
        "batch_comp_fids": batch_comp_fids,
        "seq_comp_deltas": seq_comp_deltas,
        "batch_comp_deltas": batch_comp_deltas,
        "seq_tgt_fids": seq_tgt_fids,
        "batch_tgt_fids": batch_tgt_fids,
        "seq_tgt_deltas": seq_tgt_deltas,
        "batch_tgt_deltas": batch_tgt_deltas,
        "seq_forward_calls": seq_forward_calls,
        "batch_forward_calls": batch_forward_calls,
        "comp_margin": comp_margin,
        "tgt_margin": tgt_margin,
        "overall_margin": overall_margin,
    }


def run_multi_prompt_check(model, sae, reset_counter, get_count):
    """
    Evaluates 6 prompts (1 canonical MIT + 5 spread-out prompts from FACT_BANK) at Layer 8.
    """
    # 5 additional prompts selected across FACT_BANK by index spread (0, 2, 6, 11, 17)
    selected_indices = [5, 0, 2, 6, 11, 17] # Index 5 is MIT/Cambridge (canonical)
    prompts_to_test = [FACT_BANK[idx] for idx in selected_indices]

    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)

    results_table = []
    has_mismatch = False
    mismatch_details = []

    print("\n" + "="*115)
    print("Multi-Prompt Rank Stability Check (6 Prompts at Layer 8)")
    print("="*115)

    for i, (prompt, target) in enumerate(prompts_to_test):
        target_token_id = get_target_token_id(model, target)
        res = run_single_pass(model, sae, prompt, target_token_id, hook_name, reset_counter, get_count)

        comp_match, comp_diffs = compare_ordered_lists(res["seq_comp_fids"], res["batch_comp_fids"])
        tgt_match, tgt_diffs = compare_ordered_lists(res["seq_tgt_fids"], res["batch_tgt_fids"])

        overall_match = comp_match and tgt_match
        if not overall_match:
            has_mismatch = True
            mismatch_details.append((prompt, comp_diffs, tgt_diffs))

        is_below_1e5 = res["overall_margin"] < 1e-5

        prompt_disp = prompt[:45] + "..." if len(prompt) > 45 else prompt
        results_table.append({
            "idx": i,
            "prompt": prompt_disp,
            "target": target,
            "comp_margin": res["comp_margin"],
            "tgt_margin": res["tgt_margin"],
            "overall_margin": res["overall_margin"],
            "below_1e5": "YES" if is_below_1e5 else "NO",
            "match": "MATCH" if overall_match else "MISMATCH",
        })

    print(f"{'Row':<4} | {'Prompt':<48} | {'Comp Margin':<12} | {'Target Margin':<13} | {'Below 1e-5?':<11} | {'Ordered IDs':<12}")
    print("-" * 115)
    for row in results_table:
        print(f"{row['idx']:<4} | {row['prompt']:<48} | {row['comp_margin']:<12.2e} | {row['tgt_margin']:<13.2e} | {row['below_1e5']:<11} | {row['match']:<12}")
    print("="*115)

    # Summary calculations
    num_below_1e5 = sum(1 for r in results_table if r["below_1e5"] == "YES")
    num_mismatches = sum(1 for r in results_table if r["match"] == "MISMATCH")
    num_close_but_matched = num_below_1e5 - num_mismatches

    print("\nSummary Analysis:")
    print(f"  - Total Prompts Evaluated:               6 (1 canonical + 5 spread-out from FACT_BANK)")
    print(f"  - Prompts with Margin < 1e-5:           {num_below_1e5} of 6")
    print(f"  - Prompts with Real Ordered-ID Mismatch: {num_mismatches} of 6")
    print(f"  - Prompts Close But Still Matched:       {num_close_but_matched} of 6")

    if has_mismatch:
        print("\nCRITICAL WARNING: Found an actual ordered feature-ID mismatch on at least one prompt!")
        for p, c_diff, t_diff in mismatch_details:
            print(f"  Prompt: {p!r}")
            if c_diff:
                print(f"    Competitor diffs: {c_diff}")
            if t_diff:
                print(f"    Target diffs: {t_diff}")
        print("\nVerdict: found an actual mismatch on prompt X — flagging before any further step.")
    else:
        print(f"\nVerdict: no ordered-ID mismatches observed across 6 prompts despite {num_below_1e5} prompts having tight margins.")

    return has_mismatch, num_below_1e5


def main():
    print("=== Step 3B Validation Script (Strict Multi-Run & Ordered List Check) ===")
    prompt = "The location of Massachusetts Institute of Technology is in"
    target = " Cambridge"
    layer = 8

    print(f"Loading model and Layer {layer} SAE...")
    model, sae = load_model_and_sae(layer=layer)
    target_token_id = get_target_token_id(model, target)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)

    reset_counter, get_count = count_forward_calls(model)

    # -------------------------------------------------------------
    # RUN 3 REPEATED PASSES TO INSPECT MARGIN STABILITY ACROSS RUNS
    # -------------------------------------------------------------
    print("\nExecuting 3 consecutive runs to analyze margin stability across runs...")
    run_results = []
    for run_idx in range(1, 4):
        res = run_single_pass(model, sae, prompt, target_token_id, hook_name, reset_counter, get_count)
        run_results.append(res)
        print(f"  Run {run_idx}: comp_margin={res['comp_margin']:.9e}, tgt_margin={res['tgt_margin']:.9e}, overall_margin={res['overall_margin']:.9e}")

    # Use Run 1 for detailed gate reporting
    r1 = run_results[0]
    clean_ctx = r1["clean_ctx"]

    # -------------------------------------------------------------
    # ELEMENT-BY-ELEMENT ORDERED FEATURE ID COMPARISON
    # -------------------------------------------------------------
    comp_fid_match, comp_mismatches = compare_ordered_lists(r1["seq_comp_fids"], r1["batch_comp_fids"])
    tgt_fid_match, tgt_mismatches = compare_ordered_lists(r1["seq_tgt_fids"], r1["batch_tgt_fids"])

    max_comp_delta_diff = max(abs(s - b) for s, b in zip(r1["seq_comp_deltas"], r1["batch_comp_deltas"]))
    max_tgt_delta_diff = max(abs(s - b) for s, b in zip(r1["seq_tgt_deltas"], r1["batch_tgt_deltas"]))

    comp_delta_match = max_comp_delta_diff < 1e-5
    tgt_delta_match = max_tgt_delta_diff < 1e-5

    # -------------------------------------------------------------
    # CROSS-CHECK ON-DEVICE RANK FORMULA VS ARGSORT RANK
    # -------------------------------------------------------------
    test_fids = r1["seq_tgt_fids"][:30]
    target_probs_tensor, ondevice_ranks_tensor = batched_ablation_probs_and_ranks(
        model, sae, clean_ctx.tokens, test_fids, scale=0.0,
        target_token_id=target_token_id, hook_name=hook_name
    )

    batch_tokens = clean_ctx.tokens.repeat(len(test_fids), 1)
    from src.hooks import make_per_row_scale_hook
    hook_fn = make_per_row_scale_hook(test_fids, sae, scale=0.0)
    with torch.no_grad():
        ablated_logits = model.run_with_hooks(batch_tokens, fwd_hooks=[(hook_name, hook_fn)])
        all_probs = torch.softmax(ablated_logits[:, -1, :], dim=-1)

    argsort_ranks = []
    for row in range(len(test_fids)):
        row_probs = all_probs[row]
        sorted_indices = torch.argsort(row_probs, descending=True)
        r = (sorted_indices == target_token_id).nonzero().item() + 1
        argsort_ranks.append(r)

    argsort_ranks_tensor = torch.tensor(argsort_ranks, device=ondevice_ranks_tensor.device)
    rank_formula_match = torch.equal(ondevice_ranks_tensor, argsort_ranks_tensor)

    # -------------------------------------------------------------
    # CHUNKING CROSS-CHECK (MAX_EVAL_BATCH = 7 vs 32)
    # -------------------------------------------------------------
    chunked_probs_7 = batched_ablation_probs(
        model, sae, clean_ctx.tokens, test_fids, scale=0.0,
        token_ids_of_interest=[target_token_id], hook_name=hook_name, max_eval_batch=7
    )
    chunked_probs_32 = batched_ablation_probs(
        model, sae, clean_ctx.tokens, test_fids, scale=0.0,
        token_ids_of_interest=[target_token_id], hook_name=hook_name, max_eval_batch=32
    )
    max_chunk_diff = torch.max(torch.abs(chunked_probs_7 - chunked_probs_32)).item()
    chunking_match = max_chunk_diff < 1e-5

    # -------------------------------------------------------------
    # PRINT DETAILED TABLE FOR CANONICAL CASE
    # -------------------------------------------------------------
    print("\n" + "="*85)
    print(f"{'Metric / Field':<38} | {'Sequential':<18} | {'Batched':<18} | {'Status':<10}")
    print("="*85)

    comp_status_str = "MATCH" if comp_fid_match else f"MISMATCH ({len(comp_mismatches)} diffs)"
    tgt_status_str = "MATCH" if tgt_fid_match else f"MISMATCH ({len(tgt_mismatches)} diffs)"

    print(f"{'Competitor Feature IDs (Element-Exact)':<38} | {'[Ordered List N=30]':<18} | {'[Ordered List N=30]':<18} | {comp_status_str:<10}")
    print(f"{'Competitor Max Delta Diff (1e-5)':<38} | {0.0:<18.9f} | {max_comp_delta_diff:<18.9e} | {'MATCH' if comp_delta_match else 'MISMATCH':<10}")

    print(f"{'Target Feature IDs (Element-Exact)':<38} | {'[Ordered List N=30]':<18} | {'[Ordered List N=30]':<18} | {tgt_status_str:<10}")
    print(f"{'Target Max Delta Diff (1e-5)':<38} | {0.0:<18.9f} | {max_tgt_delta_diff:<18.9e} | {'MATCH' if tgt_delta_match else 'MISMATCH':<10}")

    print(f"{'On-Device Rank Formula Match':<38} | {'argsort-oracle':<18} | {'count-formula':<18} | {'MATCH' if rank_formula_match else 'MISMATCH':<10}")
    print(f"{'Chunking Cross-Check (7 vs 32)':<38} | {0.0:<18.9f} | {max_chunk_diff:<18.9e} | {'MATCH' if chunking_match else 'MISMATCH':<10}")
    print("="*85)

    print("\nSequential Competitor Feature IDs:", r1["seq_comp_fids"])
    print("Batched Competitor Feature IDs:   ", r1["batch_comp_fids"])
    print("Sequential Target Feature IDs:    ", r1["seq_tgt_fids"])
    print("Batched Target Feature IDs:       ", r1["batch_tgt_fids"])

    # -------------------------------------------------------------
    # 7. MULTI-PROMPT RANK STABILITY CHECK (6 PROMPTS)
    # -------------------------------------------------------------
    has_mismatch, num_below_1e5 = run_multi_prompt_check(model, sae, reset_counter, get_count)

    all_exact_passed = (
        comp_fid_match and
        tgt_fid_match and
        comp_delta_match and
        tgt_delta_match and
        rank_formula_match and
        chunking_match
    )

    if not all_exact_passed or has_mismatch:
        print("\nRESULT: FAILURE (Exact or tolerance check mismatch detected).")
        sys.exit(1)
    elif num_below_1e5 > 0:
        print(f"\nRESULT: SOFT-FAIL / REQUIRES EXPLICIT SIGN-OFF")
        print(f"  Reason: {num_below_1e5} of 6 prompts have rank stability margins below 1e-5 threshold.")
        print(f"  Note: All feature IDs match element-by-element in exact order across ALL 6 prompts (0 real misorderings).")
        sys.exit(2)
    else:
        print("\nRESULT: SUCCESS (All Gate 3B checks passed with safe margins!)")
        sys.exit(0)


if __name__ == "__main__":
    main()
