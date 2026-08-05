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
    make_joint_ablation_hook,
    make_signed_ablation_hook
)

def run_layer_benchmark(
    prompt: str,
    target: str,
    layers: list[int],
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
    Sweeps the Hybrid Mute & Boost intervention across selected layers for a single prompt-target pair.
    Saves the JSON result automatically into results_dir and returns the results dictionary.
    """
    benchmark_name = "Layer Intervention Benchmark"
    benchmark_version = "0.1"
    now = datetime.datetime.utcnow()
    timestamp = now.isoformat() + "Z"
    model_name = "gpt2"
    algorithm_name = "Hybrid Mute & Boost"

    layer_results = []

    # Format target completion (ensure leading space for standard GPT-2 tokenization)
    target_str = target if target.startswith(" ") else " " + target

    for l_idx, layer in enumerate(layers, start=1):
        if layer_callback is not None:
            try:
                layer_callback(l_idx, layer)
            except Exception:
                pass

        start_time = time.perf_counter()
        
        try:
            # Load model and SAE (either using user-supplied cached loader or default function)
            if model_sae_loader is not None:
                model, sae = model_sae_loader(layer)
            else:
                model, sae = load_model_and_sae(layer=layer)

            target_token_id = get_target_token_id(model, target_str)
            tokens = model.to_tokens(prompt)
            hook_name = getattr(sae.cfg, "hook_name", f"blocks.{layer}.hook_resid_pre")
            release_name = getattr(sae.cfg, "release", "gpt2-small-res-jb")

            # 1. Clean Baseline Pass
            model.reset_hooks()
            with torch.no_grad():
                logits = model(tokens)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            
            current_top1_id = torch.argmax(probs).item()
            top_prediction_before = model.to_string([current_top1_id])
            clean_probability = probs[target_token_id].item()
            clean_sorted_indices = torch.argsort(probs, descending=True)
            clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1

            # 2. Candidate Feature Selection (with optional Safety Filtering)
            # Pool candidate size is fixed to 30 as in Tab 4
            competitor_features = get_top_competitor_features(model, sae, prompt, current_top1_id, top_n=30)
            comp_ids = []
            for fid, _ in competitor_features:
                if use_safety:
                    is_safe, _ = check_target_safe(model, sae, prompt, fid, target_token_id, strength=mute_strength)
                    if is_safe:
                        comp_ids.append(fid)
                else:
                    comp_ids.append(fid)
            
            target_features = get_top_target_features(model, sae, prompt, target_token_id, top_n=30)
            target_ids = []
            for fid, _ in target_features:
                if use_safety:
                    is_safe, _, _ = check_boost_safe(model, sae, prompt, fid, target_token_id, strength=boost_strength)
                    if is_safe:
                        target_ids.append(fid)
                else:
                    target_ids.append(fid)

            # Slice batches
            mute_batch = comp_ids[:mute_batch_size]
            boost_batch = target_ids[:boost_batch_size]

            # 3. Apply Intervention Pass
            model.reset_hooks()
            joint_mute_fn = make_joint_ablation_hook(mute_batch, sae, strength=mute_strength)
            model.add_hook(hook_name, joint_mute_fn)

            signed_boost_fn = make_signed_ablation_hook(boost_batch, sae, strength=boost_strength)
            model.add_hook(hook_name, signed_boost_fn)

            with torch.no_grad():
                logits = model(tokens)
            probs = F.softmax(logits[0, -1, :], dim=-1)

            final_sorted_indices = torch.argsort(probs, descending=True)
            final_rank = (final_sorted_indices == target_token_id).nonzero().item() + 1
            final_probability = probs[target_token_id].item()
            new_top1_id = final_sorted_indices[0].item()
            top_prediction_after = model.to_string([new_top1_id])

            model.reset_hooks()

            # Execution metrics
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            success = bool(final_rank == 1)

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
                "success": success,
                "top_prediction_before": top_prediction_before,
                "top_prediction_after": top_prediction_after,
                "mute_features": [int(f) for f in mute_batch],
                "boost_features": [int(f) for f in boost_batch]
            })

        except Exception as e:
            # Log failure but continue sweep
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            layer_results.append({
                "layer": layer,
                "hook": f"blocks.{layer}.hook_resid_pre",
                "release": "gpt2-small-res-jb",
                "clean_rank": -1,
                "final_rank": -1,
                "rank_improvement": 0,
                "clean_probability": 0.0,
                "final_probability": 0.0,
                "probability_gain": 0.0,
                "runtime_ms": float(duration_ms),
                "success": False,
                "top_prediction_before": "ERROR",
                "top_prediction_after": "ERROR",
                "mute_features": [],
                "boost_features": [],
                "error": str(e)
            })

    # Assemble JSON-serializable benchmark output dictionary
    output = {
        "benchmark_name": benchmark_name,
        "benchmark_version": benchmark_version,
        "timestamp": timestamp,
        "model": model_name,
        "algorithm": algorithm_name,
        "use_safety": use_safety,
        "prompt": prompt,
        "target": target,
        "layer_metadata": {
            "hook": "hook_resid_pre",
            "release": "gpt2-small-res-jb"
        },
        "layers": layer_results
    }

    # Automatically save JSON file into benchmark results directory
    try:
        os.makedirs(results_dir, exist_ok=True)
        # Small delay/suffix if microsecond matches
        ts_filename = now.strftime("layer_benchmark_%Y%m%d_%H%M%S_%f.json")
        saved_filepath = os.path.join(results_dir, ts_filename)
        with open(saved_filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        output["saved_filepath"] = saved_filepath
    except Exception as e:
        output["saved_filepath_error"] = str(e)

    return output
