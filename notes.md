# Research Log: Mechanistic Intervention of LLMs via Sparse Autoencoders

## 2026-07-10: First Automated Validation Breakthrough (Causal Feature Selector)

### 1. Overview & Objective
We successfully automated the discovery of causal features using a hook-based intervention pipeline. Previously, identifying the features responsible for factual predictions (e.g., identifying why the model predicts `" Paris"` for `"The Eiffel Tower is in the city of"`) relied on manual heuristic selection, trial-and-error editing, or external annotations (such as Neuronpedia). 

This update introduces an automated **Causal Feature Selector** (`src/editing.py`) that identifies and ranks features purely by their causal effect on target prediction probabilities. This log documents our first clean validation of this methodology.

---

### 2. Method: How the Causal Feature Selector Works
1. **Activation Extraction**: We run the prompt through the model (`gpt2`) and extract the residual stream activations at the layer where the SAE is trained (`blocks.8.hook_resid_pre`).
2. **SAE Projection**: The activations at the final prompt token are encoded through the Sparse Autoencoder (SAE) to retrieve the active latent features (sorted to retrieve the top-20 by activation value).
3. **Causal Ablation**: For each active feature, we patch the forward pass with a hook that forces the activation of that specific feature to zero (ablation strength = 1.0) while leaving all other features untouched.
4. **Logit Evaluation**: We compute the forward pass with the ablation hook active, measuring:
   - The new probability of the target token (`" Paris"`).
   - The change in probability relative to the un-ablated clean model (Probability Delta, $\Delta P$).
   - The resulting Top-1 prediction under ablation.
5. **Ranking**: Features are ranked in ascending order of $\Delta P$ (largest decrease in target token probability first).

---

### 3. Empirical Results: Validation on the Eiffel Tower Prompt
- **Prompt**: `"The Eiffel Tower is in the city of"`
- **Target Token**: `" Paris"`
- **Layer & Hook**: `blocks.8.hook_resid_pre` (GPT-2, Layer 8)

#### Ranked Causal Features (Top 20 Active)

| Rank | Feature ID | Activation | Ablated Prob | Prob Delta ($\Delta P$) | Top-1 Output under Ablation | Status / Notes |
| :--- | :--------- | :--------- | :----------- | :---------------------- | :-------------------------- | :------------- |
| **1** | **11149** | **30.1315** | **0.0350** | **-0.0337** | **Paris** | **Target Feature (Causal Driver)** |
| 2 | 2194 | 9.5312 | 0.0511 | -0.0177 | London | Secondary feature |
| 3 | 5856 | 34.1087 | 0.0614 | -0.0073 | London | High activation, low causal effect |
| 4 | 5858 | 1.9858 | 0.0641 | -0.0046 | London | Minimal effect |
| 5 | 17465 | 1.3260 | 0.0653 | -0.0034 | London | Minimal effect |
| 6 | 6807 | 1.4099 | 0.0655 | -0.0032 | London | Minimal effect |
| 7 | 1288 | 3.4975 | 0.0659 | -0.0028 | London | Minimal effect |
| 8 | 18994 | 2.6934 | 0.0659 | -0.0028 | London | Minimal effect |
| 9 | 15820 | 1.3954 | 0.0668 | -0.0019 | London | Noise |
| 10 | 16649 | 0.9991 | 0.0673 | -0.0014 | London | Noise |
| 11 | 3118 | 1.1971 | 0.0684 | -0.0003 | London | Noise |
| 12 | 19794 | 2.2413 | 0.0685 | -0.0002 | London | Noise |
| 13 | 22852 | 3.6236 | 0.0694 | +0.0007 | London | Positive Delta (inhibitory) |
| 14 | 1173 | 1.5244 | 0.0697 | +0.0010 | London | Positive Delta (inhibitory) |
| 15 | 23035 | 1.9717 | 0.0709 | +0.0022 | London | Positive Delta (inhibitory) |
| 16 | 6863 | 1.1633 | 0.0711 | +0.0024 | London | Positive Delta (inhibitory) |
| 17 | 5926 | 1.8436 | 0.0712 | +0.0025 | London | Positive Delta (inhibitory) |
| 18 | 1960 | 1.1004 | 0.0717 | +0.0029 | London | Positive Delta (inhibitory) |
| 19 | 21062 | 5.4374 | 0.0719 | +0.0032 | London | Positive Delta (inhibitory) |
| 20 | 313 | 2.0266 | 0.0722 | +0.0035 | London | Positive Delta (inhibitory) |

---

### 4. Why This Version is Better (Automated vs. Manual/Previous)

| Dimensions | Previous Version (Manual/Heuristic) | Current Version (Causal Feature Selector) |
| :--- | :--- | :--- |
| **Discovery Cost** | High. Required querying Neuronpedia, guess-and-check, or static lookup files. | Zero-shot. Automatically computed in ~1 second via dynamic forward hooks. |
| **Causal Grounding** | **Hypothetical**. High activation does not guarantee high causal importance. | **Empirical**. Measures the actual drop in target probability under ablation. |
| **Noise Filtering** | None. A highly active feature (e.g. `5856`, activation 34.10) would be assumed important, even though ablating it barely changes the output ($\Delta P = -0.0073$). | High precision. Clearly shows that feature `11149` (activation 30.13) has **~5x** the causal impact of `5856` despite having lower clean activation. |
| **User Interface Integration** | Manual text fields where the user had to input arbitrary feature IDs. | Fully auto-populated selectors that immediately highlight the top causal features. |

### 5. Takeaways & Next Steps
- **Causal Validation**: Feature `11149` independently ranked #1 with a drop in target token probability of **-0.0337**, validating that it is the principal causal driver.
- **Integration**: The Streamlit application will now ingest this causal selector ranked list, automatically pre-loading the top causal feature into the intervention slider for the user.

---

## 2026-07-24: Weighted Multi-Competitor Reduction & The Grammatical Feature Bottleneck

### 1. Objective & Hypothesis
Instead of muting only the single top competitor token (which often leads to another competitor rising up and blocking the target), we designed **Weighted Multi-Competitor Reduction**. The hypothesis was:
* Muting *all* above-target competitors simultaneously, with weights proportional to their baseline threat (probability), would clear a path for the correct target token to climb the leaderboard.
* We framed this as a **ranking problem** (making `Target > Competitor` for all competitors) rather than a raw probability maximization problem.

### 2. Diagnostic Discovery on the Cambridge Case
We validated this method on a known ground-truth benchmark:
- **Prompt**: `"The location of Massachusetts Institute of Technology is in"`
- **Target**: `" Cambridge"` (Clean rank: 8, Clean probability: 1.15%)
- **Clean competitors above target**: `" the"` (17.95%), `" a"` (10.53%), `" question"` (7.51%), `" jeopardy"` (2.71%), `" an"` (2.63%), `" danger"` (1.76%), `" doubt"` (1.21%).

When running the diagnostic to trace which feature each competitor token mapped to as its single #1 causal driver, we discovered a massive collapse:
- **Feature 313** was the top driver for:
  - `" the"` (17.95%)
  - `" a"` (10.53%)
  - `" an"` (2.63%)
- **Feature 21169** was the top driver for:
  - `" question"` (7.51%)
  - `" jeopardy"` (2.71%)
  - `" danger"` (1.76%)
  - `" doubt"` (1.21%)

As a result, all 7 competitor tokens collapsed onto only **2 unique features** (313 and 21169). Using the max-of-shared-strength rule, the resulting intervention was weak (mute strengths of 0.28 and 0.12), and the target's rank remained unchanged at 8.

### 3. The Distributed Support Hypothesis (Key Research Revelation)
This diagnostic reveals a fundamental bottleneck in token-focused mechanistic interventions:
1. **Grammatical/Functional Collapse**: Generic words like `" the"`, `" a"`, and `" an"` do not represent semantic concepts; they represent syntactic directions. Feature 313 is not a "the" feature, but a broad syntactic/grammatical driver.
2. **Distributed Support (The Speaker Analogy)**: Common grammatical tokens are supported by many independent features in the network (e.g. 150 different "speakers"). Suppressing only the single top feature (e.g. turning off the loudest speaker) leaves the remaining active features untouched, allowing the token to remain dominant.
3. **Implications**: The hypothesis that *"a single representative feature per competitor is enough to shift the ranking"* is false for highly distributed grammatical tokens.
4. **Next Steps**: A successful multi-competitor intervention must shift from a `1 competitor -> 1 feature` mapping to a `1 competitor -> N influential features` (subspace) representation.

---

## 2026-09-09T19:49:15+05:30: Fix Cleanup `del` Statement NameError in `layer_benchmark_runner.py`

### 1. Issue & Root Cause Analysis
In `src/benchmark/layer_benchmark_runner.py`, inside `run_layer_benchmark` (around line 321), the cleanup statement `del logits, probs, clean_sorted_indices, logits_int, probs_int, sorted_indices_after, tokens` referenced `logits` and `clean_sorted_indices` which were never assigned in the function scope (the function uses `clean_ctx.clean_probs` and precomputed ranks via `build_clean_context`). This line raised a `NameError` at the end of every successful iteration, triggering the `except` block and appending a duplicate bogus "ERROR" entry (`clean_rank: -1`) to `prompt_runs_dict[p_idx]["layers"]`.

### 2. Code Changes
**File**: [`src/benchmark/layer_benchmark_runner.py`](file:///d:/Work/PROJECTS/FeatureScalpel/src/benchmark/layer_benchmark_runner.py#L321)

```diff
-                del logits, probs, clean_sorted_indices, logits_int, probs_int, sorted_indices_after, tokens
+                del probs, logits_int, probs_int, sorted_indices_after, tokens
```

### 3. Empirical Validation Results
Validated via direct call to `run_layer_benchmark` with:
- Prompt: `"The location of Massachusetts Institute of Technology is in"`
- Target: `"Cambridge"`
- Layers: `[8]`
- `use_safety`: `False`
- Strengths & Batches: `mute_strength=0.3`, `boost_strength=0.5`, `mute_batch_size=3`, `boost_batch_size=3`
- `use_batched_ranking`: `False`

**Results**:
- `len(output["prompts"][0]["layers"])`: `1` (No duplicate error entry created)
- `"error"` key present: `False`
- `wall_clock_time`: `20.332201499999883` seconds

**Layer Result Entry (`layers[0]`)**:
```json
{
  "layer": 8,
  "hook": "blocks.8.hook_resid_pre",
  "release": "gpt2-small-res-jb",
  "clean_rank": 8,
  "final_rank": 3,
  "rank_improvement": 5,
  "clean_probability": 0.01152519416064024,
  "final_probability": 0.0355326384305954,
  "probability_gain": 0.024007444269955158,
  "runtime_ms": 5720.880599999873,
  "profile": {
    "sae_loading_ms": 14593.68160000031,
    "clean_baseline_ms": 880.5392999997821,
    "feature_selection_ms": 4762.711500000023,
    "safety_filtering_ms": 0.009999999747378752,
    "intervention_ms": 66.72550000030242,
    "result_packaging_ms": 0.0009000000318337698,
    "total_layer_ms": 5720.880599999873
  },
  "forward_passes": {
    "clean_baseline": 1,
    "candidate_screening": 0,
    "competitor_ranking": 30,
    "target_ranking": 30,
    "competitor_safety": 0,
    "target_safety": 0,
    "final_intervention": 1,
    "total_model_forwards": 62
  },
  "counts": {
    "competitor_candidates_evaluated": 30,
    "target_candidates_evaluated": 30,
    "competitor_safety_checks": 0,
    "target_safety_checks": 0,
    "selected_mute_features_count": 3,
    "selected_boost_features_count": 3
  },
  "success": true,
  "top_prediction_before": " the",
  "top_prediction_after": " the",
  "mute_features": [313, 8459, 21169],
  "boost_features": [3076, 19288, 8239]
}
```

