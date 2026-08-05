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
    benchmark_version = "0.2"
    now = datetime.datetime.utcnow()
    timestamp = now.isoformat() + "Z"
    model_name = "gpt2"
    algorithm_name = "Hybrid Mute & Boost"

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
            if layer_callback is not None:
                try:
                    layer_callback(p_idx, total_prompts, l_idx, total_layers, layer, prompt_text)
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
                tokens = model.to_tokens(prompt_text)
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
                competitor_features = get_top_competitor_features(model, sae, prompt_text, current_top1_id, top_n=30)
                comp_ids = []
                for fid, _ in competitor_features:
                    if use_safety:
                        is_safe, _ = check_target_safe(model, sae, prompt_text, fid, target_token_id, strength=mute_strength)
                        if is_safe:
                            comp_ids.append(fid)
                    else:
                        comp_ids.append(fid)
                
                mute_batch = comp_ids[:mute_batch_size]

                target_features = get_top_target_features(model, sae, prompt_text, target_token_id, top_n=30)
                target_ids = []
                for fid, _ in target_features:
                    if use_safety:
                        is_safe, _ = check_boost_safe(model, sae, prompt_text, fid, target_token_id, strength=boost_strength)
                        if is_safe:
                            target_ids.append(fid)
                    else:
                        target_ids.append(fid)
                
                boost_batch = target_ids[:boost_batch_size]

                # 3. Apply Intervention Hook
                model.reset_hooks()
                joint_hook = make_joint_ablation_hook(
                    sae=sae,
                    mute_feature_indices=mute_batch,
                    mute_coeff=mute_strength,
                    boost_feature_indices=boost_batch,
                    boost_coeff=boost_strength
                )
                model.add_hook(hook_name, joint_hook)

                # 4. Measure Intervened Output
                with torch.no_grad():
                    logits_int = model(tokens)
                probs_int = F.softmax(logits_int[0, -1, :], dim=-1)
                model.reset_hooks()

                top1_id_after = torch.argmax(probs_int).item()
                top_prediction_after = model.to_string([top1_id_after])
                final_probability = probs_int[target_token_id].item()
                sorted_indices_after = torch.argsort(probs_int, descending=True)
                final_rank = (sorted_indices_after == target_token_id).nonzero().item() + 1

                duration_ms = (time.perf_counter() - start_time) * 1000.0
                success = (final_rank < clean_rank) or (final_probability > clean_probability)

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

        prompt_runs.append({
            "prompt": prompt_text,
            "target": target_text,
            "layers": layer_results
        })

    # Assemble master JSON-serializable benchmark output dictionary
    output = {
        "benchmark_name": benchmark_name,
        "benchmark_version": benchmark_version,
        "timestamp": timestamp,
        "model": model_name,
        "algorithm": algorithm_name,
        "use_safety": use_safety,
        "parameters": {
            "mute_strength": mute_strength,
            "boost_strength": boost_strength,
            "mute_batch_size": mute_batch_size,
            "boost_batch_size": boost_batch_size
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
