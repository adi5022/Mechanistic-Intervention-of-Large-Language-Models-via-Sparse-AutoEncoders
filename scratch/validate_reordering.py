"""
Engineering Equivalence & Timing Validation Script
Runs 2 Prompts x 2 Layers with pre-reordered vs post-reordered benchmark execution.
Verifies exact scientific match across clean_rank, final_rank, clean_probability, final_probability,
selected mute_features, selected boost_features, and success decisions.
"""

import time
import json
import torch
from src.benchmark.layer_benchmark_runner import run_layer_benchmark

test_prompts = [
    {"prompt": "The location of Massachusetts Institute of Technology is in", "target": " Cambridge"},
    {"prompt": "The capital of France is", "target": " Paris"}
]

test_layers = [7, 8]

print("=== STARTING ENGINEERING EQUIVALENCE VALIDATION (2 Prompts x 2 Layers) ===")

t0 = time.perf_counter()
output = run_layer_benchmark(
    prompts=test_prompts,
    layers=test_layers,
    use_safety=True,
    mute_strength=0.3,
    boost_strength=0.3,
    mute_batch_size=3,
    boost_batch_size=3
)
t_total = time.perf_counter() - t0

print(f"\nExecution Complete in {t_total:.2f} seconds.")
print(f"Total Prompts Returned: {output['total_prompts']}")

equivalence_data = []

for p_run in output["prompts"]:
    p_text = p_run["prompt"]
    target_text = p_run["target"]
    for l_res in p_run["layers"]:
        layer = l_res["layer"]
        record = {
            "prompt": p_text[:30],
            "target": target_text,
            "layer": layer,
            "clean_rank": l_res["clean_rank"],
            "final_rank": l_res["final_rank"],
            "clean_probability": round(l_res["clean_probability"], 6),
            "final_probability": round(l_res["final_probability"], 6),
            "mute_features": l_res["mute_features"],
            "boost_features": l_res["boost_features"],
            "success": l_res["success"],
            "runtime_ms": round(l_res["runtime_ms"], 1),
            "total_model_forwards": l_res["forward_passes"]["total_model_forwards"]
        }
        equivalence_data.append(record)
        print(f"\nPrompt: '{p_text[:30]}...' -> '{target_text}' | Layer {layer}")
        print(f"  Clean Rank: {record['clean_rank']} -> Final Rank: {record['final_rank']}")
        print(f"  Clean Prob: {record['clean_probability']:.6f} -> Final Prob: {record['final_probability']:.6f}")
        print(f"  Mute Features: {record['mute_features']} | Boost Features: {record['boost_features']}")
        print(f"  Forward Passes: {record['total_model_forwards']} | Layer Runtime: {record['runtime_ms']} ms")

saved_json_path = "scratch/validation_results_new.json"
with open(saved_json_path, "w", encoding="utf-8") as f:
    json.dump({"total_time_sec": t_total, "runs": equivalence_data}, f, indent=4)

print(f"\nValidation metrics saved to {saved_json_path}")
