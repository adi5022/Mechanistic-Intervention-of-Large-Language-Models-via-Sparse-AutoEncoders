"""
Sequential vs GPU-Batched candidate safety-filtering benchmark.

Runs the OLD per-candidate loop (check_target_safe / check_boost_safe) and the
NEW batched path (check_target_safe_batch / check_boost_safe_batch) back-to-back
on identical inputs, times each with time.perf_counter(), and verifies both
paths produce the same is_safe verdicts before reporting any speed numbers.

Usage:
    python benchmark_sequential_vs_batched.py
    python benchmark_sequential_vs_batched.py --top-n 30 --repeats 3 --device cuda
"""

import argparse
import time
import statistics

import torch

from src.sae_utils import load_model_and_sae, get_default_device
from src.editing import (
    get_target_token_id,
    get_top_competitor_features,
    get_top_target_features,
    check_target_safe,
    check_boost_safe,
    check_target_safe_batch,
    check_boost_safe_batch,
    build_clean_context,
)


def sync_if_cuda(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def run_sequential(model, sae, clean_ctx, prompt, fids, target_token_id, strength, kind, device):
    sync_if_cuda(device)
    t0 = time.perf_counter()
    results = []
    for fid in fids:
        if kind == "mute":
            is_safe, delta = check_target_safe(
                model, sae, prompt, fid, target_token_id, strength=strength,
                clean_target_prob=clean_ctx.clean_target_prob, clean_rank=clean_ctx.clean_rank
            )
            results.append((fid, is_safe, delta))
        else:
            is_safe, delta, rank_imp = check_boost_safe(
                model, sae, prompt, fid, target_token_id, strength=strength,
                clean_target_prob=clean_ctx.clean_target_prob, clean_rank=clean_ctx.clean_rank
            )
            results.append((fid, is_safe, delta))
    sync_if_cuda(device)
    elapsed = time.perf_counter() - t0
    return results, elapsed, len(fids)  # len(fids) == number of forward passes issued


def run_batched(model, sae, clean_ctx, fids, target_token_id, strength, kind, device):
    sync_if_cuda(device)
    t0 = time.perf_counter()
    if kind == "mute":
        batch_res = check_target_safe_batch(model, sae, clean_ctx, fids, target_token_id, strength=strength)
        results = [(fid, is_safe, delta) for fid, (is_safe, delta) in zip(fids, batch_res)]
    else:
        batch_res = check_boost_safe_batch(model, sae, clean_ctx, fids, target_token_id, strength=strength)
        results = [(fid, is_safe, delta) for fid, (is_safe, delta, _) in zip(fids, batch_res)]
    sync_if_cuda(device)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def verify_parity(seq_results, batch_results, label):
    seq_map = {fid: (is_safe, delta) for fid, is_safe, delta in seq_results}
    batch_map = {fid: (is_safe, delta) for fid, is_safe, delta in batch_results}
    if set(seq_map.keys()) != set(batch_map.keys()):
        print(f"  [WARN]  [{label}] Feature ID sets differ between sequential and batched runs!")
        return False
    mismatches = []
    for fid, (seq_safe, seq_delta) in seq_map.items():
        batch_safe, batch_delta = batch_map[fid]
        if seq_safe != batch_safe:
            mismatches.append((fid, seq_safe, batch_safe, seq_delta, batch_delta))
    if mismatches:
        print(f"  [WARN]  [{label}] {len(mismatches)} is_safe MISMATCHES between sequential and batched:")
        for fid, s, b, sd, bd in mismatches[:10]:
            print(f"      feature {fid}: sequential={s} (delta={sd:.6f})  batched={b} (delta={bd:.6f})")
        return False
    max_delta_diff = max(abs(seq_map[fid][1] - batch_map[fid][1]) for fid in seq_map)
    print(f"  [OK] [{label}] {len(seq_map)}/{len(seq_map)} is_safe verdicts match. Max prob_delta diff: {max_delta_diff:.2e}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Benchmark sequential vs GPU-batched safety filtering")
    parser.add_argument("--prompt", default="Seiyu Group's headquarters are in", help="Prompt to test")
    parser.add_argument("--target", default="Tokyo", help="Target completion token")
    parser.add_argument("--layer", type=int, default=8, help="SAE layer")
    parser.add_argument("--top-n", type=int, default=30, help="Number of candidate features per pool")
    parser.add_argument("--mute-strength", type=float, default=0.3)
    parser.add_argument("--boost-strength", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per method, for stable timing")
    parser.add_argument("--device", default=None, help="cuda / cpu / mps (default: auto-detect)")
    args = parser.parse_args()

    device = args.device or get_default_device()
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt!r}  ->  Target: {args.target!r}")
    print(f"Layer: {args.layer}  |  top_n per pool: {args.top_n}  |  repeats: {args.repeats}\n")

    model, sae = load_model_and_sae(device=device, layer=args.layer)
    target_str = " " + args.target.strip()
    target_token_id = get_target_token_id(model, target_str)

    model.reset_hooks()
    clean_ctx = build_clean_context(model, sae, args.prompt, target_token_id)
    current_top1_id = torch.argmax(clean_ctx.clean_probs).item()

    competitor_features = get_top_competitor_features(
        model, sae, args.prompt, current_top1_id, top_n=args.top_n, clean_ctx=clean_ctx, use_batched=False
    )
    target_features = get_top_target_features(
        model, sae, args.prompt, target_token_id, top_n=args.top_n, clean_ctx=clean_ctx, use_batched=False
    )
    comp_fids = [fid for fid, _ in competitor_features]
    tgt_fids = [fid for fid, _ in target_features]
    print(f"Candidate pools resolved: {len(comp_fids)} mute candidates, {len(tgt_fids)} boost candidates\n")

    results_table = []

    for label, fids, strength, kind in [
        ("MUTE safety filter", comp_fids, args.mute_strength, "mute"),
        ("BOOST safety filter", tgt_fids, args.boost_strength, "boost"),
    ]:
        print(f"--- {label} ({len(fids)} candidates) ---")

        seq_times, batch_times = [], []
        seq_results = batch_results = None
        for r in range(args.repeats):
            seq_results, t_seq, n_calls = run_sequential(
                model, sae, clean_ctx, args.prompt, fids, target_token_id, strength, kind, device
            )
            seq_times.append(t_seq)
            print(f"  run {r+1}: sequential = {t_seq:.4f}s ({n_calls} forward passes)")

        for r in range(args.repeats):
            batch_results, t_batch = run_batched(
                model, sae, clean_ctx, fids, target_token_id, strength, kind, device
            )
            batch_times.append(t_batch)
            print(f"  run {r+1}: batched    = {t_batch:.4f}s (1 batched forward pass)")

        parity_ok = verify_parity(seq_results, batch_results, label)

        seq_mean = statistics.mean(seq_times)
        batch_mean = statistics.mean(batch_times)
        speedup = seq_mean / batch_mean if batch_mean > 0 else float("inf")

        print(f"  Sequential mean: {seq_mean:.4f}s (min {min(seq_times):.4f}s, max {max(seq_times):.4f}s)")
        print(f"  Batched    mean: {batch_mean:.4f}s (min {min(batch_times):.4f}s, max {max(batch_times):.4f}s)")
        print(f"  Speedup: {speedup:.2f}x   |   Results match: {parity_ok}\n")

        results_table.append({
            "label": label,
            "n_candidates": len(fids),
            "seq_mean_s": seq_mean,
            "batch_mean_s": batch_mean,
            "speedup": speedup,
            "parity_ok": parity_ok,
        })

    print("=" * 70)
    print(f"{'Stage':<22}{'#Cand':>7}{'Sequential':>14}{'Batched':>12}{'Speedup':>10}{'Match':>8}")
    total_seq = total_batch = 0.0
    for row in results_table:
        total_seq += row["seq_mean_s"]
        total_batch += row["batch_mean_s"]
        print(f"{row['label']:<22}{row['n_candidates']:>7}{row['seq_mean_s']:>13.4f}s{row['batch_mean_s']:>11.4f}s{row['speedup']:>9.2f}x{str(row['parity_ok']):>8}")
    overall_speedup = total_seq / total_batch if total_batch > 0 else float("inf")
    print("-" * 70)
    print(f"{'TOTAL':<22}{'':>7}{total_seq:>13.4f}s{total_batch:>11.4f}s{overall_speedup:>9.2f}x")
    print("=" * 70)
    all_match = all(row["parity_ok"] for row in results_table)
    print(f"\nAll verdicts matched between sequential and batched paths: {all_match}")


if __name__ == "__main__":
    main()
