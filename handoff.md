# Project Handoff: Transient Steering & Layer Intervention Benchmark

**Date:** August 6, 2026  
**Git Branch:** `layer-intervention-benchmark` (Pushed & Up-to-Date with Remote)  
**Repository:** `Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders`

---

## 1. Executive Summary & Session Context

This repository houses the **FeatureScalpel** project—a mechanistic intervention framework for Large Language Models using Sparse Autoencoders (SAEs).

Recently, we established the dedicated **Layer Intervention Benchmark Subsystem**, designed to experimentally characterize how intervention effectiveness (Hybrid Mute & Boost) varies across transformer layer depth ($0 \dots 11$ in GPT-2 Small).

Over the past sessions, we have completed the final infrastructure pass before large-scale benchmarking, focusing on:
1. Auditing all pretrained SAE dictionaries (`gpt2-small-res-jb`) across transformer depth.
2. Building an algorithm-agnostic benchmark engine (`src/benchmark/layer_benchmark_runner.py`) and Streamlit application (`layer_benchmark.py`).
3. Refactoring model loading to eliminate redundant base model re-initialization.
4. Adding stage-by-stage runtime profiling and extended experiment metadata.
5. Implementing batch dataset import mechanisms (file uploader, copy-paste JSON, sample template).
6. Safe memory management and artifact export flows.

---

## 2. Key Accomplishments & Architectural Milestones

### A. Pretrained SAE Dictionary Audit
- Audited `gpt2-small-res-jb` coverage via `investigate_gpt2_saes.py`.
- Confirmed **100% complete coverage** for `blocks.{layer}.hook_resid_pre` across all 12 transformer blocks ($0 \dots 11$).

### B. Decoupled & Optimized Loading Architecture (`src/sae_utils.py` & `layer_benchmark.py`)
- **Problem Solved:** Previously, `load_model_and_sae(layer)` re-instantiated a new GPT-2 base model for every layer sweep, leading to 4 separate GPT-2 loads for a sweep of `[2, 5, 8, 10]`.
- **Solution:** Separated base model caching (`get_cached_base_model()`) from layer SAE caching (`get_cached_layer_sae(layer)`).
- **Result:** GPT-2 base model is loaded **exactly ONCE per Streamlit session**. Subsequent layers and prompts reuse the cached base model in system RAM/VRAM with 0 redundant reloads.

### C. Runtime Profiling & Stage Breakdown
- Integrated lightweight stage-by-stage runtime profiling (in milliseconds) into `run_layer_benchmark`:
  - `sae_loading_ms`
  - `clean_baseline_ms`
  - `feature_selection_ms`
  - `safety_filtering_ms`
  - `intervention_ms`
  - `result_packaging_ms`
  - `total_layer_ms`
- Timings are recorded inside the JSON output under `"profile"` and displayed in the UI (**⏱️ View Layer Stage Profiling Breakdown** expander).

### D. Extended Experiment Metadata
- Extended the master JSON artifact schema with rich metadata:
  - `benchmark_name`, `benchmark_version` ("0.3"), `timestamp`
  - `model` ("gpt2"), `sae_release` ("gpt2-small-res-jb"), `hook_location` ("hook_resid_pre")
  - `safety_enabled` (bool), `selected_layers` (list[int])
  - `parameters`: `{mute_strength, boost_strength, mute_batch_size, boost_batch_size, use_safety, algorithm}`
- All existing JSON fields are preserved for backwards compatibility.

### E. Real-Time Progress Feedback & Stage Tracking
- Enhanced `layer_callback` to transmit current stage names to the UI:  
  `⏳ Prompt X / N | Layer Y (Z / M) | Stage: <stage_name>`

### F. Batch Dataset Import Mechanisms
- Added **Batch Import Prompts** in the sidebar featuring:
  1. **Copy-Paste Text Area:** Directly paste raw JSON strings (`[{"prompt": "...", "target": "..."}, ...]`).
  2. **File Uploader:** Upload `.json` files.
  3. **Sample Template Download:** Download a pre-formatted `sample_prompts_dataset.json` template.

### G. Benchmark Artifact Visibility & Export Manager
- Consolidated results into a master JSON artifact (`benchmark_results/layer_benchmark_YYYYMMDD_HHMMSS_ffffff.json`).
- Provided a **Download Master Benchmark JSON** button alongside an expandable per-prompt artifact view (**📄 Artifact #X**) with individual download buttons.

### H. Safe Memory Management
- Deletes intermediate PyTorch tensors (`logits`, `probs`, `clean_sorted_indices`, `logits_int`, `probs_int`, `sorted_indices_after`, `tokens`) after each layer run.
- Invokes `torch.cuda.empty_cache()` if CUDA is available, avoiding VRAM fragmentation without clearing cached base model or SAEs.

---

## 3. Layer Intervention Depth Findings (Phase 1 Benchmark)

| Layer | Clean Rank | Final Rank | Rank Improvement | Clean Prob | Final Prob | Prob Gain | Primary Response |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer 2** | 8 | 7 | +1 | 1.15% | 1.22% | +0.07% | Minor |
| **Layer 5** | 8 | 5 | +3 | 1.15% | 1.94% | +0.79% | Moderate |
| **Layer 8** | 8 | **3** | **+5** | 1.15% | **3.52%** | **+2.37%** | **Peak Steering Efficacy** |
| **Layer 10** | 8 | 5 | +3 | 1.15% | 1.97% | +0.82% | Moderate |

*Finding:* **Layer 8** exhibits the highest intervention sensitivity and probability gain (+2.37%, clearing 5 rank positions), establishing layer depth as a critical factor in intervention success.

---

## 4. Key Architecture Files

- **`layer_benchmark.py`**: Dedicated Streamlit dashboard for multi-prompt layer intervention benchmarking.
- **`src/benchmark/layer_benchmark_runner.py`**: Headless, algorithm-agnostic benchmark execution engine.
- **`src/sae_utils.py`**: Model loading and SAE utility functions (`load_base_model`, `load_sae_for_layer`, `load_model_and_sae`).
- **`notebooks/04_layer_intervention_benchmark.ipynb`**: Programmatic Jupyter Notebook for running layer benchmarks, inspecting DataFrames, and visualizing plots.
- **`History/Session_LayerBenchmark_History.md`**: Complete transcript/history log of benchmark subsystem development.
- **`docs/Research_Journal/10.md` & `11.md`**: Detailed research journal entries and visualizations.

---

## 5. How to Resume on Another Machine

```powershell
# 1. Fetch remote branch
git fetch --all
git checkout layer-intervention-benchmark

# 2. Activate virtual environment
.venv\Scripts\Activate.ps1

# 3. Launch the Benchmark UI
streamlit run layer_benchmark.py
```

---

## 6. Next Steps & Roadmap

1. **Large-Scale Data Collection:** Run batch benchmark sweeps across 100+ prompt-target pairs using the new batch JSON import mechanism.
2. **Cross-Prompt Aggregation:** Compute aggregate mean rank improvement and win-rate statistics across layer depth.
3. **Adaptive Layer Selection:** Use the benchmark dataset to train or calibrate layer selection heuristics.
