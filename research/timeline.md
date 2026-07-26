# Research Timeline

Chronological timeline of FeatureScalpel's design modifications, pivots, and milestones, reconstructed from repository Git logs and historical developer conversation records.

---

### 2026-07-08
* **Title**: Project Setup and Initialization
* **Motivation**: Establish the basic repository structure and load pretrained model (GPT-2-small) and layer-8 Sparse Autoencoder.
* **Summary**: Initialized directory structure, added basic notes file, and loaded weights.
* **Files involved**: `sae_utils.py`, `docs/notes.md`
* **Status**: Completed

---

### 2026-07-10
* **Title**: First Automated Validation (Causal Feature Selector)
* **Motivation**: Avoid manual lookup on Neuronpedia or hardcoding feature IDs. Identify causal features purely by their causal effect on prediction probability.
* **Summary**: Implemented the first version of the Causal Feature Selector ranking active features by probability delta under full ablation. Validated on Eiffel Tower prompt (Feature 11149).
* **Files involved**: `src/editing.py`, `app.py`
* **Status**: Completed

---

### 2026-07-15
* **Title**: The Target-vs-Competitor Targeting Bug Discovery & Core Logic Pivot
* **Motivation**: The initial causal selector had a core logical error where it identified features helping the *correct* target token and turned *them* down, making the correct answer weaker.
* **Summary**: Discovered that targeting the correct target's helpers caused the target to drop further in probability. Realized we must target the helpers of the **incorrect competitor** instead, or use amplification (boosting). Pivoted the search query to find features driving the top incorrect prediction.
* **Files involved**: `src/editing.py`
* **Status**: Completed

---

### 2026-07-19
* **Title**: Target-Feature Ablation Evaluation (Failure & Pivot)
* **Motivation**: Test target feature soft ablation across 22 suppressed facts.
* **Summary**: Found that muting target helpers weakens the correct answer, leading to a 4.55% success rate. Pivoted to targeting competitor features instead.
* **Files involved**: `docs/Research_Journal/2.md`
* **Status**: Completed (Hypothesis Rejected)

---

### 2026-07-20
* **Title**: Competitor-Focused Iterative Ablation & Grammatical Competitor Barrier
* **Motivation**: Suppress features helping incorrect predictions.
* **Summary**: Tripled success rate to 13.64%. Identified split between factual (success) and grammatical (failure) competitors due to distributed syntax representation. Discovered that a fixed ablation strength of 0.3 across all facts was too conservative for some cases and had no effect on others.
* **Files involved**: `docs/Research_Journal/3.md`
* **Status**: Completed

---

### 2026-07-22
* **Title**: Compound Batch Muting, Hybrid Steering (Mute & Boost), and Dublin/Melbourne Success
* **Motivation**: Outperform single-feature ablation and bypass the grammatical competitor barrier by combining interventions.
* **Summary**:
  - Tested **compound (batch) ablation**: Outperformed single-feature ablation but revealed a collapse point where muting too many features simultaneously backfired.
  - Implemented **hybrid steering (Mute & Boost)**: Mutted competitor features while simultaneously boosting target-supporting features. Succeeded on Dublin and Melbourne prompts where all prior approaches failed.
* **Files involved**: `experiment_app.py`, `src/editing.py`, `src/hooks.py`
* **Status**: Completed

---

### 2026-07-23
* **Title**: Whole-Combination Safety Checks
* **Motivation**: Avoid collateral damage and semantic regressions during joint mute/boost interventions.
* **Summary**: Introduced safety check gates (`check_target_safe`, `check_boost_safe`, and `check_combination_safe`) to ensure joint interventions do not regress the target's rank or introduce new blocker tokens.
* **Files involved**: `src/editing.py`, `experiment_app.py`
* **Status**: Completed

---

### 2026-07-24
* **Title**: Weighted Multi-Competitor Reduction & Grammatical Feature Collapse
* **Motivation**: Suppress all above-target competitor tokens in one joint pass without causing rank regression.
* **Summary**:
  - Developed Weighted Multi-Competitor Reduction. Validated against the Cambridge MIT benchmark case. Discovered that the 7 competitor tokens collapsed onto only 2 unique features (313 and 21169), confirming the Distributed Support (Loudspeaker) Hypothesis.
  - Added Neuronpedia API feature description integration to render active hyperlinks and hover tooltips for all features.
  - Built a decoupled Explainable AI layer (`src/explain.py`) integrating Groq API (`llama-3.1-8b-instant`) to output plain-English summaries explaining why the intervention worked or failed.
* **Files involved**: `src/editing.py`, `experiment_app.py`, `docs/Research_Journal/4.md`, `scratch/test_weighted_reduction.py`, `src/explain.py`
* **Status**: Completed

---

### 2026-07-25
* **Title**: Hybrid Sweep Diagnostic & Rank Regression Analysis
* **Motivation**: Diagnose why target rank regresses (drops) on larger combinations during the Cambridge hybrid sweep despite rising target probability.
* **Summary**: Ran systematic sweeps with and without safety checks. Proved that there is no state leak or implementation bug (Option A rejected via independent verification). Confirmed it is a mathematical consequence of softmax interaction (Option B), where boosting polysemantic features can inadvertently raise competitor tokens faster than the target, or muting competitors changes the denominator structure.
* **Files involved**: `scratch/test_hybrid_cambridge.py`, `scratch/test_hybrid_cambridge_large.py`, `scratch/analyze_pairs.py`, `scratch/test_independent.py`
* **Status**: Completed

---

### 2026-07-25 (Session 2)
* **Title**: Shared Feature Aggregation Experiment (max() vs sum())
* **Motivation**: Test Hypothesis H10 to see if additive accumulation with clamping resolves target rank suppression in Weighted Multi-Competitor Reduction.
* **Summary**: Replaced max-aggregation with additive sum-accumulation followed by clamping at `max_strength` ceiling. Evaluated on the Cambridge benchmark. Target probability increased slightly (from `1.29%` to `1.39%`), but the target rank remained stuck at 8. Hypothesis H10 was rejected, motivating a new hypothesis (H11) that competitor support is distributed across multiple features.
* **Files involved**: `src/editing.py`, `scratch/run_h10_cambridge.py`
* **Status**: Completed (Hypothesis H10 Rejected)

---

### 2026-07-25 (Session 3)
* **Title**: Multi-Feature Representation Diagnostic
* **Motivation**: Test Hypothesis H11 to see how concentrated or distributed competitor representations are across SAE features.
* **Summary**: Analyzed top-10 competitor feature decay profiles and overlap statistics. Verified that probability deltas are non-additive. Found that semantic competitors exhibit extremely flat decay profiles (e.g. Top-2 feature drop was 92.3% - 94.7% of Top-1 drop), suggesting distributed feature influence. Within the discovered top-10 features, 100% of the features were shared by multiple competitors.
* **Files involved**: `scratch/run_h11_diagnostic.py`
* **Status**: Completed (Hypothesis H11 Supported by Current Evidence)

---

### 2026-07-25 (Session 4)
* **Title**: Weighted Multi-Feature Competitor Reduction & Sweeps
* **Motivation**: Test Hypothesis H12 to see if distributing competitor-focused steering across multiple causal features per competitor (Top-K) improves steering performance compared to single-feature muting.
* **Summary**: Implemented `run_weighted_multi_feature_competitor_reduction` supporting Equal, Delta-Normalized, and Softmax weighting strategies. Added Tab 8 to the UI to expose Top-K and weighting parameters. Ran a validation sweep over K={1,2,3,5} and all weighting methods. Found that muting $K=3$ features recovered the target rank from 8 to 6, with Equal and Softmax weighting outperforming Delta-Normalized. Muting $K=5$ features showed dilution decay due to too many weak features.
* **Files involved**: `src/editing.py`, `experiment_app.py`, `scratch/test_tab8_matrix.py`, `docs/Research_Journal/8.md`
* **Status**: Completed (Hypothesis H12 Validated)

