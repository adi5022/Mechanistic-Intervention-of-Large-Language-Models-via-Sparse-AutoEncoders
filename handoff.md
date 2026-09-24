# Project Handoff: Transient Steering & Layer Intervention Benchmark

**Date:** September 24, 2026  
**Git Branch:** `pool-refill-implementation` (from `stable-batched-workflow`)  
**Repository:** `Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders`

---

## 0. Latest Session (2026-09-24): Tab 4 Pool Refill

**Why:** Tab 4's Cumulative Sweep stopped early (e.g. `black`, rank 10 -> 5) because its length was `min(safe mute pool, safe boost pool)`, built once against the clean model. Full write-up: `docs/Research_Journal/19.md`.

**What changed (uncommitted on `pool-refill-implementation`):**
* `src/hooks.py`: `build_scale_map`, `with_extra_scale`, `make_scale_map_hook`, `make_per_row_scale_hook_with_base`.
* `src/batched_eval.py`: `base_scale_map` param on both batched primitives (empty = original behaviour).
* `src/editing.py`: `build_steered_context`; `exclude_ids` / `base_scale_map` on candidate ranking and the four safety checks.
* `experiment_app.py` (Tab 4): round loop with pool refill against a steered baseline; independent mute/boost growth inside a round; retries of earlier-rejected features; track-only regressions; optional Max Refill Rounds (0 = unlimited); live status, baseline cards, pool tables, refill log, ledger, per-round timeline, refill markers on the chart, extended run-history record.
* `scratch/validate_pool_refill.py`: validation (steered baseline, batched vs sequential parity, empty-base identity) - all pass.

* Later the same day: overlap fix (each feature used on one side only), clearer baseline cards, top-of-page live tracker, **candidate source = all prompt positions**, and Top N auto-capped to the number of active features (see Journal Entry 19, sections 8-10).

**Key finding:** candidates are limited to SAE features active at the final token (67 for the test prompt at layer 8); multiplicative steering cannot activate inactive features, so refill cannot exceed that ceiling. Any further gain needs a different mechanism (the proposed architectural change).

**Not done:** Tab 11 still uses the old single-pool sweep. Changes are not committed or pushed.

---

## 1. Executive Summary & Session Context

This session implemented **Step 3 — GPU Batched Candidate Evaluation** (Sub-steps 3A, 3B, and 3C) of the Layer Intervention Benchmark optimization plan. By stacking candidate evaluations into PyTorch batch dimensions (`batch_tokens = tokens.repeat(N, 1)`), we reduced the total model forward calls per prompt×layer evaluation from **126 down to 6** ($21\times$ reduction in forward call dispatch overhead) while maintaining **100% bit-exact equivalence** on selected features and safety decisions.

---

## 2. Key Architecture & Optimization Changes

### A. Pre-Batching Cleanup (Step 3A)
* **`src/editing.py`**:
  - Removed duplicated file header block.
  - Added `@dataclass CleanContext` and `build_clean_context()` to cache unablated forward pass outputs (`tokens`, `clean_probs`, `resid_last`, `clean_target_prob`, `clean_rank`).
  - Added optional `clean_ctx` parameter to `get_top_active_features`, `get_top_competitor_features`, and `get_top_target_features`.
* **`src/hooks.py`**:
  - Fixed double SAE encode bug in `make_ablation_hook` (encodes `resid` once, snapshots baseline reconstruction, and clones before mutating).
* **`src/benchmark/layer_benchmark_runner.py`**:
  - Integrated `build_clean_context` to reuse clean pass outputs across candidate screening and ranking.
* **Forward Calls Reduction:** **126 $\rightarrow$ 122 calls**.

### B. Batched Causal Candidate Ranking (Step 3B)
* **`src/hooks.py`**: Added `make_per_row_scale_hook(feature_ids, sae, scale)` for row-wise feature scaling across batch rows.
* **`src/batched_eval.py` (NEW)**: Created module containing GPU primitives `batched_ablation_probs` and `batched_ablation_probs_and_ranks` with configurable `MAX_EVAL_BATCH = 32` chunking.
* **`src/editing.py`**: Added `use_batched: bool = False` kwarg to `get_top_competitor_features` and `get_top_target_features`.
* **Verification**: Verified ordered feature-ID lists match element-by-element with 0 misorderings across 6 ROME prompts.

### C. Batched Safety Filter Checking (Step 3C)
* **`src/editing.py`**: Added `check_target_safe_batch` (`scale = 1 - strength`) and `check_boost_safe_batch` (`scale = 1 + strength`).
* **`src/benchmark/layer_benchmark_runner.py`**: Wired `check_target_safe_batch` and `check_boost_safe_batch` into Stage 4 safety filtering when `use_batched_ranking` is enabled.
* **Validation**: **60 / 60 (100%) safety boolean agreement** on canonical test case (`Layer 8`, MIT $\rightarrow$ Cambridge).
* **Forward Calls Reduction:** **122 $\rightarrow$ 6 calls**.

---

## 3. End-to-End Execution Trace Summary

| Stage | Sequential Forward Calls | Batched Forward Calls (Post-Step 3C) |
|---|:---:|:---:|
| **Clean Baseline** | 1 | 1 |
| **Candidate Screening** | 0 | 0 |
| **Competitor Ranking** | 30 | **1** |
| **Target Ranking** | 30 | **1** |
| **Competitor Safety Filter** | 30 | **1** |
| **Target Safety Filter** | 30 | **1** |
| **Final Intervention Pass** | 1 | 1 |
| **Total Model Forward Calls** | **122** | **6** |

---

## 4. Documentation & Validation Deliverables

* **`docs/Research_Journal/14.md`**: Created Research Journal Entry 14 detailing Step 3A/3B/3C implementation, tolerance standards, and forward call accounting.
* **`scratch/validate_step3a.py`**: Step 3A validation script ($1\text{e-}9$ precision).
* **`scratch/validate_step3b.py`**: Step 3B validation script (element-by-element list ordering, chunking, and multi-prompt margin checks).
* **`scratch/validate_step3c.py`**: Step 3C validation script ($60/60$ safety boolean agreement table).

---

## 5. Instructions to Commit & Push to GitHub

```powershell
# 1. Stage all modified core files, new batched primitives, scripts, and documentation
git add .gitignore src/hooks.py src/batched_eval.py src/editing.py src/benchmark/layer_benchmark_runner.py docs/Research_Journal/14.md handoff.md scratch/validate_step3a.py scratch/validate_step3b.py scratch/validate_step3c.py

# 2. Commit changes
git commit -m "Step 3 (3A-3C): Implement GPU Batched Candidate Evaluation and Safety Filters (126 -> 6 forward calls)"

# 3. Push to remote branch
git push origin layer_benchmark_runner
```
