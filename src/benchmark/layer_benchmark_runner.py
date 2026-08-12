"""
Layer Benchmark Runner
Contains the algorithm-agnostic core logic for characterization of intervention success across transformer depth.
"""

import os
import json
import time
import datetime
import torch
import torch.nn.functional as F

from src.sae_utils import load_model_and_sae
from src.editing import (
    get_target_token_id,
    get_top_competitor_features,
    get_top_target_features,
    check_target_safe,
    check_boost_safe
)
from src.hooks import (
    make_mute_and_boost_hook,
    make_joint_ablation_hook,
    make_signed_ablation_hook
)


def run_layer_benchmark(
    prompts: list[dict] | str,
    target: str = None,
    layers: list[int] = None,
    mute_strength: float = 0.3,
    boost_strength: float = 0.5,
    mute_batch_size: int = 3,
    boost_batch_size: int = 3,
    use_safety: bool = True,
    algorithm: str = "hybrid",
    model_sae_loader=None,
    results_dir: str = "benchmark_results",
    layer_callback=None
) -> dict:
    """
    Sweeps the Hybrid Mute & Boost intervention across selected layers for one or multiple prompt-target pairs.
    Includes runtime profiling per layer, extended metadata tracking, and safe CUDA memory cleanup.
    Saves a SINGLE consolidated JSON file into results_dir and returns the master results dictionary.
    """
    if layers is None:
        layers = [2, 5, 8, 10]

    # Standardize input to list of prompt-target items
    if isinstance(prompts, str):
        prompt_items = [{"prompt": prompts, "target": target}]
    else:
        prompt_items = prompts

    benchmark_name = "Layer Intervention Benchmark"
    benchmark_version = "0.3"
    now = datetime.datetime.utcnow()
    timestamp = now.isoformat() + "Z"
    model_name = "gpt2"
    algorithm_name = "Hybrid Mute & Boost"
    sae_release_name = "gpt2-small-res-jb"
    hook_location_name = "hook_resid_pre"

    prompt_runs = []
    total_prompts = len(prompt_items)
    total_layers = len(layers)

    for p_idx, p_item in enumerate(prompt_items, start=1):
        prompt_text = p_item["prompt"].strip()
        target_text = p_item["target"].strip()

        # Format target completion (ensure leading space for standard GPT-2 tokenization)
        target_str = target_text if target_text.startswith(" ") else " " + target_text
        layer_results = []

        for l_idx, layer in enumerate(layers, start=1):
            start_time = time.perf_counter()
            
            try:
                # STAGE 1: Load SAE & Base Model
                if layer_callback is not None:
                    try:
                        layer_callback(p_idx, total_prompts, l_idx, total_layers, layer, prompt_text, "Loading SAE...")
                    except Exception:
                        pass

                t_sae_start = time.perf_counter()
                if model_sae_loader is not None:
                    model, sae = model_sae_loader(layer)
                else:
                    model, sae = load_model_and_sae(layer=layer)
                sae_loading_ms = (time.perf_counter() - t_sae_start) * 1000.0

                target_token_id = get_target_token_id(model, target_str)
                tokens = model.to_tokens(prompt_text)
                hook_name = getattr(sae.cfg, "hook_name", f"blocks.{layer}.hook_resid_pre")
                release_name = getattr(sae.cfg, "release", sae_release_name)

                # STAGE 2: Clean Baseline Pass
                if layer_callback is not None:
                    try:
                        layer_callback(p_idx, total_prompts, l_idx, total_layers, layer, prompt_text, "Running clean baseline...")
                    except Exception:
                        pass

                t_clean_start = time.perf_counter()
                model.reset_hooks()
                with torch.no_grad():
                    logits = model(tokens)
                probs = F.softmax(logits[0, -1, :], dim=-1)
                
                current_top1_id = torch.argmax(probs).item()
                top_prediction_before = model.to_string([current_top1_id])
                clean_probability = probs[target_token_id].item()
                clean_sorted_indices = torch.argsort(probs, descending=True)
                clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1
                clean_baseline_ms = (time.perf_counter() - t_clean_start) * 1000.0

                # STAGE 3: Feature Candidate Selection
                if layer_callback is not None:
                    try:
                        layer_callback(p_idx, total_prompts, l_idx, total_layers, layer, prompt_text, "Selecting candidate features...")
                    except Exception:
                        pass

                t_feat_start = time.perf_counter()
                competitor_features = get_top_competitor_features(model, sae, prompt_text, current_top1_id, top_n=30)
                target_features = get_top_target_features(model, sae, prompt_text, target_token_id, top_n=30)
                feature_selection_ms = (time.perf_counter() - t_feat_start) * 1000.0

                # STAGE 4: Safety Filtering
                if layer_callback is not None:
                    try:
                        layer_callback(p_idx, total_prompts, l_idx, total_layers, layer, prompt_text, "Applying safety filter...")
                    except Exception:
                        pass

                t_safe_start = time.perf_counter()
                comp_ids = []
                for fid, _ in competitor_features:
                    if use_safety:
                        is_safe, _ = check_target_safe(
                            model, sae, prompt_text, fid, target_token_id, strength=mute_strength,
                            clean_target_prob=clean_probability, clean_rank=clean_rank
                        )
                        if is_safe:
                            comp_ids.append(fid)
                    else:
                        comp_ids.append(fid)
                
                mute_batch = comp_ids[:mute_batch_size]

                target_ids = []
                for fid, _ in target_features:
                    if use_safety:
                        is_safe, _, _ = check_boost_safe(
                            model, sae, prompt_text, fid, target_token_id, strength=boost_strength,
                            clean_target_prob=clean_probability, clean_rank=clean_rank
                        )
                        if is_safe:
                            target_ids.append(fid)
                    else:
                        target_ids.append(fid)

                
                boost_batch = target_ids[:boost_batch_size]
                safety_filtering_ms = (time.perf_counter() - t_safe_start) * 1000.0

                # STAGE 5: Apply Intervention Pass
                if layer_callback is not None:
                    try:
                        layer_callback(p_idx, total_prompts, l_idx, total_layers, layer, prompt_text, "Running intervention pass...")
                    except Exception:
                        pass

                t_int_start = time.perf_counter()
                model.reset_hooks()
                joint_hook = make_mute_and_boost_hook(
                    mute_feature_ids=mute_batch,
                    mute_strength=mute_strength,
                    boost_feature_ids=boost_batch,
                    boost_strength=boost_strength,
                    sae=sae
                )
                model.add_hook(hook_name, joint_hook)


                with torch.no_grad():
                    logits_int = model(tokens)
                probs_int = F.softmax(logits_int[0, -1, :], dim=-1)
                model.reset_hooks()

                top1_id_after = torch.argmax(probs_int).item()
                top_prediction_after = model.to_string([top1_id_after])
                final_probability = probs_int[target_token_id].item()
                sorted_indices_after = torch.argsort(probs_int, descending=True)
                final_rank = (sorted_indices_after == target_token_id).nonzero().item() + 1
                intervention_ms = (time.perf_counter() - t_int_start) * 1000.0

                # STAGE 6: Result Packaging
                t_pkg_start = time.perf_counter()
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                success = (final_rank < clean_rank) or (final_probability > clean_probability)
                result_packaging_ms = (time.perf_counter() - t_pkg_start) * 1000.0

                num_competitor_candidates = len(competitor_features)
                num_target_candidates = len(target_features)
                num_competitor_safety_checks = len(competitor_features) if use_safety else 0
                num_target_safety_checks = len(target_features) if use_safety else 0

                # Actual Forward Pass Counter Accounting (Exact Execution Trace)
                fwd_clean_baseline = 1
                fwd_candidate_screening = 2  # 1 pass in get_top_competitor_features + 1 pass in get_top_target_features
                fwd_competitor_ranking = 1 + num_competitor_candidates  # 1 internal clean pass + N candidate ablations (31)
                fwd_target_ranking = 1 + num_target_candidates          # 1 internal clean pass + N candidate ablations (31)
                fwd_competitor_safety = num_competitor_safety_checks    # 1 pass per check (clean baseline is precomputed/passed) (30)
                fwd_target_safety = num_target_safety_checks            # 1 pass per check (clean baseline is precomputed/passed) (30)
                fwd_final_intervention = 1

                total_fwd_passes = (
                    fwd_clean_baseline +
                    fwd_candidate_screening +
                    fwd_competitor_ranking +
                    fwd_target_ranking +
                    fwd_competitor_safety +
                    fwd_target_safety +
                    fwd_final_intervention
                )

                forward_passes_acc = {
                    "clean_baseline": fwd_clean_baseline,
                    "candidate_screening": fwd_candidate_screening,
                    "competitor_ranking": fwd_competitor_ranking,
                    "target_ranking": fwd_target_ranking,
                    "competitor_safety": fwd_competitor_safety,
                    "target_safety": fwd_target_safety,
                    "final_intervention": fwd_final_intervention,
                    "total_model_forwards": total_fwd_passes
                }

                counts_meta = {
                    "competitor_candidates_evaluated": num_competitor_candidates,
                    "target_candidates_evaluated": num_target_candidates,
                    "competitor_safety_checks": num_competitor_safety_checks,
                    "target_safety_checks": num_target_safety_checks,
                    "selected_mute_features_count": len(mute_batch),
                    "selected_boost_features_count": len(boost_batch)
                }

                profile = {
                    "sae_loading_ms": float(sae_loading_ms),
                    "clean_baseline_ms": float(clean_baseline_ms),
                    "feature_selection_ms": float(feature_selection_ms),
                    "safety_filtering_ms": float(safety_filtering_ms),
                    "intervention_ms": float(intervention_ms),
                    "result_packaging_ms": float(result_packaging_ms),
                    "total_layer_ms": float(duration_ms)
                }

                layer_results.append({
                    "layer": layer,
                    "hook": hook_name,
                    "release": release_name,
                    "clean_rank": int(clean_rank),
                    "final_rank": int(final_rank),
                    "rank_improvement": int(clean_rank - final_rank),
                    "clean_probability": float(clean_probability),
                    "final_probability": float(final_probability),
                    "probability_gain": float(final_probability - clean_probability),
                    "runtime_ms": float(duration_ms),
                    "profile": profile,
                    "forward_passes": forward_passes_acc,
                    "counts": counts_meta,
                    "success": success,
                    "top_prediction_before": top_prediction_before,
                    "top_prediction_after": top_prediction_after,
                    "mute_features": [int(f) for f in mute_batch],
                    "boost_features": [int(f) for f in boost_batch]
                })

                # TASK 5: Safe Memory Cleanup of intermediate PyTorch tensors
                del logits, probs, clean_sorted_indices, logits_int, probs_int, sorted_indices_after, tokens
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                layer_results.append({
                    "layer": layer,
                    "hook": f"blocks.{layer}.hook_resid_pre",
                    "release": sae_release_name,
                    "clean_rank": -1,
                    "final_rank": -1,
                    "rank_improvement": 0,
                    "clean_probability": 0.0,
                    "final_probability": 0.0,
                    "probability_gain": 0.0,
                    "runtime_ms": float(duration_ms),
                    "profile": {
                        "sae_loading_ms": 0.0,
                        "clean_baseline_ms": 0.0,
                        "feature_selection_ms": 0.0,
                        "safety_filtering_ms": 0.0,
                        "intervention_ms": 0.0,
                        "result_packaging_ms": 0.0,
                        "total_layer_ms": float(duration_ms)
                    },
                    "success": False,
                    "top_prediction_before": "ERROR",
                    "top_prediction_after": "ERROR",
                    "mute_features": [],
                    "boost_features": [],
                    "error": str(e)
                })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        prompt_runs.append({
            "prompt": prompt_text,
            "target": target_text,
            "layers": layer_results
        })

    # TASK 2: Assemble master JSON-serializable benchmark output dictionary with extended metadata
    output = {
        "benchmark_name": benchmark_name,
        "benchmark_version": benchmark_version,
        "timestamp": timestamp,
        "model": model_name,
        "algorithm": algorithm_name,
        "sae_release": sae_release_name,
        "hook_location": hook_location_name,
        "safety_enabled": use_safety,
        "use_safety": use_safety,  # Preserve existing key for backward compatibility
        "selected_layers": sorted(layers),
        "parameters": {
            "mute_strength": mute_strength,
            "boost_strength": boost_strength,
            "mute_batch_size": mute_batch_size,
            "boost_batch_size": boost_batch_size,
            "use_safety": use_safety,
            "algorithm": algorithm
        },
        "total_prompts": total_prompts,
        "prompts": prompt_runs
    }

    # Automatically save a SINGLE master JSON file into benchmark results directory
    try:
        os.makedirs(results_dir, exist_ok=True)
        ts_filename = now.strftime("layer_benchmark_%Y%m%d_%H%M%S_%f.json")
        saved_filepath = os.path.join(results_dir, ts_filename)
        with open(saved_filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        output["saved_filepath"] = saved_filepath
    except Exception as e:
        output["saved_filepath_error"] = str(e)

    return output
