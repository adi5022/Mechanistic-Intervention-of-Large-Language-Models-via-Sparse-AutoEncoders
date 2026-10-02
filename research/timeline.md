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

---

### 2026-08-09
* **Title**: Candidate Selection Formalisation, Delta Patching Literature Lineage, ROME Dataset & GPU Speed Analysis
* **Motivation**: Clarify candidate selection mechanics, formalize academic citations for Delta Patching, standardize the 22-prompt ROME dataset, and migrate PyTorch to CUDA 12.4 for hardware acceleration on NVIDIA GTX 1660 Ti.
* **Summary**:
  - Documented 2-stage sparse candidate screening (Stage 1: 1 pass sparse filter; Stage 2: N-pass causal ablation ranking).
  - Formalized Delta Patching citations (Meng et al. 2022, Bricken et al. 2023, Cunningham et al. 2023, Templeton et al. 2024).
  - Integrated 22 suppressed facts from ROME `known_1000` into `benchmark_test_prompts.json`.
  - Migrated PyTorch environment to CUDA 12.4 (`torch-2.6.0+cu124`).
  - Added interactive compute device switcher to `layer_benchmark.py` and empirically benchmarked hardware speed, demonstrating up to **20.67× speedup on GPU** (1.63s/prompt on GTX 1660 Ti vs 33.62s/prompt on CPU) with peak VRAM utilization of 1.45 GB / 6.00 GB.
* **Files involved**: `layer_benchmark.py`, `benchmark_test_prompts.json`, `test_gpu_benchmark.py`, `run_speed_analysis.py`, `docs/Research_Journal/12.md`
* **Status**: Completed

### 2026-09-30
* **Title**: Repair and SAE-limit diagnostics; new visual app tab
* **Motivation**: Test whether later layers undo the layer-8 edit, and whether the SAE (rather than the layer) limits factual correction.
* **Summary**: Added `src/repair_diagnostics.py`, `tools/run_repair_diagnostics.py` and a 6th app tab (Repair & SAE limit). Ran 3 failing prompts x layers 5-11 plus MIT at layer 8. No repair seen (held edit == plain edit in 21/21 cells); on these 3 hand-picked failed prompts the SAE-free edit of equal size reached rank 1 in 21/21 combinations vs 1/21 for the SAE edit (not a success rate; prompts were selected as failures).
* **Files involved**: `src/repair_diagnostics.py`, `tools/run_repair_diagnostics.py`, `experiment_app.py`, `docs/Research_Journal/22.md`, `docs/Research_Journal/packs/repair_diagnostics/`
* **Status**: Completed (side-effect comparison for the SAE-free edit still open)

### 2026-10-01
* **Title**: Planning: learned mute/boost strengths, and a fair method comparison on the 131 prompts
* **Motivation**: Fixed strengths cannot suit every prompt; and the SAE method needs a fair comparison with other methods (IKE, ROME, DiffMean) on the same prompts, timed.
* **Summary**: Corrected Entry 22 wording (3 hand-picked failed prompts are not a success rate). Created the reference spec (strict filter, Top-N 200, all positions, 131 prompts). Wrote the plan for learned strengths (Entry 23). No new experiments run.
* **Files involved**: `data/reference_strict_topn200_all.json`, `docs/Research_Journal/23.md`
* **Status**: Planned

### 2026-10-02
* **Title**: Step 1 of the learned-strength study: CounterFact hard-set builder
* **Motivation**: A bigger, less templated prompt set than the 131 prompts, defined by the author's condition (the true answer is not rank 1).
* **Summary**: Wrote `tools/build_counterfact_set.py` (batched ranking, self-check, yield report). Smoke test on 500 records: 373 kept, 500 prompts ranked in 0.7 s. Updated Entry 23 with the step-by-step order of work.
* **Files involved**: `tools/build_counterfact_set.py`, `docs/Research_Journal/23.md`
* **Status**: Script written; full run pending
* **Also 2026-10-02**: wrote `tools/split_counterfact_set.py` and `tools/build_strength_cache.py` (+ `src/strength_cache.py`); the cache's built-in sanity check caught a wrong `scale` argument (multiplier, not strength) before any data was written; cache rankings verified identical to the sweep's own on 6 prompts.

### 2026-10-01 to 2026-10-02 (documentation)
* **Title**: Literature check, session report and handoff rewrite
* **Motivation**: An employer questioned the SAE approach; the whole session needed to be recorded in a form a new reader can follow.
* **Summary**: Wrote `docs/literature_review_sae_editing.md` (established field; nothing conceptually new; evidence on SAE steering is mixed; what beat SAEs and how; what follows), `docs/session_report_2026-09-30_to_10-02.md` (plain-language report with glossary, numbers, mistakes and commands) and a new root `handoff.md`. The previous handoff was kept as `docs/handoff_2026-10-01_session_log.md`.
* **Files involved**: `docs/literature_review_sae_editing.md`, `docs/session_report_2026-09-30_to_10-02.md`, `handoff.md`
* **Status**: Completed
* **Also 2026-10-02**: journal entries 24 (data pipeline and headroom results), 25 (literature check and direction assessment) and a placeholder for 21 (safety-filter study, still unwritten) added; H17, EXP-016 and EXP-017 added to the logs.

* **Also 2026-10-02 (training code)**: wrote `src/strength_models.py` and `tools/train_strength_models.py` (PromptNet, learned fixed pair, oracle, prefix-cumulative proxy, four self-checks including a comparison with the real sweep's round-0 pools). Three design failures found and fixed during development are recorded in Entry 23, section 13.9. Full training run not yet done.

### 2026-10-03
* **Title**: Learned-strength models trained (prefix proxy) and analysed
* **Motivation**: Test whether choosing the mute and boost strengths (per prompt or globally) beats fixed 0.6 / 0.5.
* **Summary**: Ran `tools/train_strength_models.py` on the RTX 4050 laptop (1,195 s) and analysed per prompt with `tools/analyse_strength_models.py`. On the 300 test prompts the proxy success rate rose from 25.0% (fixed 0.6/0.5) to 38.7% (learned fixed pair, mute 0.98 / boost 1.81) and 39.7% (PromptNet; 3 prompts more than the pair, p = 0.25); side effects roughly doubled. Per-prompt adaptation not shown; real sweep not yet run.
* **Files involved**: `docs/Research_Journal/26.md`, `docs/Research_Journal/packs/strength_models/`, `tools/analyse_strength_models.py`
* **Status**: Completed (proxy only)

* **Also 2026-10-03 (version 2)**: wrote `src/feature_models.py` and `tools/train_feature_net.py` (per-feature multipliers from a shared scoring network; no new data collection). Development runs: about 31 to 32% of held-out prompts vs 37% for the version-1 pair and 25% for fixed 0.6 / 0.5 in the one-shot proxy; the free per-feature upper bound is 73%. Full run pending (Entry 27).
