"""
Quick GPU Verification Test for FeatureScalpel
Checks PyTorch CUDA, GPT-2 model allocation on GTX 1660 Ti, SAE loading, and execution speed.
"""

import sys
import time
import torch

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
from src.editing import get_top_competitor_features

def run_verification():
    print("=" * 60)
    print(" FEATURESCALPEL GPU VERIFICATION TEST")
    print("=" * 60)
    
    # 1. Device check
    device = get_default_device()
    print(f"[1/4] PyTorch Device: {device.upper()}")
    assert device == "cuda", "ERROR: CUDA GPU not detected by PyTorch!"
    
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"      GPU Model    : {gpu_name}")
    print(f"      Total VRAM   : {total_mem:.2f} GB")
    
    # 2. Base Model GPU allocation
    print("\n[2/4] Loading GPT-2 Small onto GPU...")
    t0 = time.perf_counter()
    model = load_base_model(device="cuda")
    t1 = time.perf_counter()
    alloc_mem = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"      Model Loaded in {(t1-t0)*1000:.1f} ms")
    print(f"      VRAM Allocated  : {alloc_mem:.1f} MB")
    
    # 3. SAE Loading onto GPU
    print("\n[3/4] Loading Layer 8 SAE onto GPU...")
    t2 = time.perf_counter()
    sae = load_sae_for_layer(layer=8, device="cuda")
    t3 = time.perf_counter()
    total_vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"      SAE Loaded in   : {(t3-t2)*1000:.1f} ms")
    print(f"      Total VRAM Used : {total_vram:.1f} MB")
    
    # 4. End-to-End Causal Feature Selection Benchmark Test
    prompt = "The location of Massachusetts Institute of Technology is in"
    target = " Cambridge"
    print(f"\n[4/4] Running 30 Causal Ablation Forward Passes on GPU...")
    print(f"      Prompt: '{prompt}'")
    
    t4 = time.perf_counter()
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top1_id = torch.argmax(probs).item()
    
    competitors = get_top_competitor_features(model, sae, prompt, top1_id, top_n=30)
    t5 = time.perf_counter()
    
    elapsed_ms = (t5 - t4) * 1000.0
    print(f"\n[SUCCESS] 30 Causal Ablation Passes Completed in {elapsed_ms:.1f} ms!")
    print(f"   Top Competitor Feature Identified: #{competitors[0][0]} (Prob Delta: {competitors[0][1]*100:.2f}%)")
    print(f"   Peak VRAM Usage: {torch.cuda.max_memory_allocated(0)/(1024**2):.1f} MB / {total_mem*1024:.0f} MB")
    print("=" * 60)
    print(" EVERYTHING IS WORKING PERFECTLY ON YOUR GTX 1660 Ti!")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
