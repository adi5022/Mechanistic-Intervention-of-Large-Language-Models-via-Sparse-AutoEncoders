"""
Speed Analysis Benchmark Script
Compares GPU (NVIDIA GeForce GTX 1660 Ti) vs CPU execution speed for 3 prompts across 4 layers.
Provides live console progress feedback during execution.
"""

import sys
import time
import torch

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.sae_utils import load_base_model, load_sae_for_layer
from src.benchmark.layer_benchmark_runner import run_layer_benchmark

TEST_PROMPTS = [
    {"prompt": "Eavan Boland was born in", "target": "Dublin"},
    {"prompt": "The capital of Roman Republic is", "target": "Rome"},
    {"prompt": "Henri Debain was born in", "target": "Paris"}
]

LAYERS = [2, 5, 8, 10]

MODEL_CACHE = {}
SAE_CACHE = {}

def get_cached_model_and_sae(layer: int, device: str):
    """Caches base model and layer SAEs in RAM/VRAM to avoid redundant reload disk I/O."""
    if "model" not in MODEL_CACHE or MODEL_CACHE.get("device") != device:
        MODEL_CACHE["model"] = load_base_model(device=device)
        MODEL_CACHE["device"] = device
    key = (layer, device)
    if key not in SAE_CACHE:
        SAE_CACHE[key] = load_sae_for_layer(layer=layer, device=device)
    return MODEL_CACHE["model"], SAE_CACHE[key]

def benchmark_device(device_name: str) -> dict:
    print(f"\n" + "=" * 60)
    print(f" STARTING BENCHMARK ON: {device_name.upper()}")
    print("=" * 60)
    
    start_total = time.perf_counter()
    
    def console_callback(p_idx, total_p, l_idx, total_l, current_layer, prompt_text, stage):
        print(f"  [{device_name.upper()}] Prompt {p_idx}/{total_p} ('{prompt_text[:25]}...') | Layer {current_layer} ({l_idx}/{total_l}) | {stage}")
    
    results = run_layer_benchmark(
        prompts=TEST_PROMPTS,
        layers=LAYERS,
        mute_strength=0.3,
        boost_strength=0.5,
        mute_batch_size=3,
        boost_batch_size=3,
        use_safety=True,
        algorithm="hybrid",
        model_sae_loader=lambda l: get_cached_model_and_sae(layer=l, device=device_name),
        results_dir="benchmark_results",
        layer_callback=console_callback
    )
    
    end_total = time.perf_counter()
    total_sec = end_total - start_total
    
    peak_vram_mb = 0.0
    if device_name == "cuda" and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
    return {
        "device": device_name,
        "total_seconds": total_sec,
        "per_prompt_seconds": total_sec / len(TEST_PROMPTS),
        "peak_vram_mb": peak_vram_mb
    }

def main():
    print("=" * 60)
    print(" HARDWARE SPEED COMPARISON ANALYSIS (GPU vs CPU)")
    print(" Prompts: 3 | Layers Tested: 4 [2, 5, 8, 10] | Safety: Enabled")
    print("=" * 60)
    
    # 1. GPU Benchmark
    gpu_stats = benchmark_device("cuda")
    print(f"\n [OK] GPU Run Complete in {gpu_stats['total_seconds']:.2f}s ({gpu_stats['per_prompt_seconds']:.2f}s/prompt)")
    
    # 2. CPU Benchmark
    cpu_stats = benchmark_device("cpu")
    print(f"\n [OK] CPU Run Complete in {cpu_stats['total_seconds']:.2f}s ({cpu_stats['per_prompt_seconds']:.2f}s/prompt)")
    
    # 3. Speedup Calculation
    speedup = cpu_stats['total_seconds'] / gpu_stats['total_seconds'] if gpu_stats['total_seconds'] > 0 else 0
    
    print("\n" + "=" * 60)
    print(" FINAL HARDWARE BENCHMARK COMPARISON SUMMARY")
    print("=" * 60)
    print(f" Device 1 (GPU - GTX 1660 Ti) : {gpu_stats['total_seconds']:.2f} seconds ({gpu_stats['per_prompt_seconds']:.2f}s per prompt)")
    print(f" Device 2 (CPU Host)          : {cpu_stats['total_seconds']:.2f} seconds ({cpu_stats['per_prompt_seconds']:.2f}s per prompt)")
    print(f" Peak GPU VRAM Usage          : {gpu_stats['peak_vram_mb']:.1f} MB / 6144 MB")
    print("-" * 60)
    print(f" GPU SPEEDUP FACTOR         : {speedup:.2f}x FASTER ON GPU!")
    print("=" * 60)

if __name__ == "__main__":
    main()
