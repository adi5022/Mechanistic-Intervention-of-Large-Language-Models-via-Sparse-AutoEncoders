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

* **Also 2026-10-03 (gradient-descent editing)**: new branch `gradient-descent-editing`. Added an "Editing method" choice to the Hybrid tab: the fixed mute / boost sweep (default, unchanged) or gradient descent (one multiplier per candidate feature, 0 to 3, tuned on the prompt, checked through the real hook; `src/gradient_editing.py`). Hand trials on about 12 prompts: it ended nearer rank 1 than the sweep on all six prompts where both were run and reached rank 1 on three (sky to black, Dallas, soccer), with KL 0.3 to 1.9; confounded by a larger allowed edit range; not a controlled test. Prototype vision (edit kept on during generation, per-word trace, fact probes, 300-prompt scoreboard) recorded as plan; instruction tuning of GPT-2 scrapped; gradient descent as teacher for the per-feature network recorded in `docs/future_scope.md` (Entry 28).

* **Also 2026-10-03 (equal-budget comparison)**: wrote `tools/compare_sweep_vs_gradient.py` and ran it on the 300 held-out test prompts on the real model (about 9 hours, GTX 1660 Ti). Rank 1: sweep 0.6 / 0.5 94, sweep 1.0 / 2.0 (same 0 to 3x range as gradient descent) 161, gradient descent 271 of 300; gradient descent solved 110 prompts the wide sweep did not and the wide sweep none that it did not; same-prompt KL lower for gradient descent on 160 of 161 shared successes; 5 to 6.5 times faster; weakest in the 101-1000 band (54 of 79); median target probability at rank 1 only 7.7%. Paraphrases, neighbours and generation not tested (Entry 29).

* **Also 2026-10-04 (additive edits)**: added the opt-in additive edit (silent features switched on at the last position, amounts start at exactly 0, optional always-on KL penalty) to `src/gradient_editing.py`, the scale-and-add hook to `src/hooks.py`, four self-checks (`tools/check_additive.py`, all pass), a new "Prototype lab" tab in `experiment_app.py`, and extra arms / validation split / counterfactual targets in the comparison tool. Hand trials: six failing prompts all reached rank 1 with the additive edit but with large edits (55% to 161% of residual norm). Controlled tests not yet run (Entry 30).

* **Also 2026-10-04 (additive control)**: the author noticed that every prompt reached the target in the new tab; a control with 20 random unrelated words showed the additive edit reaches rank 1 on 20 of 20 (multipliers only 1 of 20) with edits of 126% to 192% of the residual norm. The success rate is therefore not a meaningful measure for the additive arm; the planned validation tuning was suspended and the additive checkbox in the Prototype lab defaults to OFF with a warning. Entry 30 section 10.

* **Also 2026-10-04 (reworded-prompt test)**: built `tools/test_generalisation.py` (edit tuned on the original prompt applied unchanged to reworded prompts, nearby facts and neutral prompts) and ran it on 40 test prompts with true answers and with random words as the control. Additive: carries over to reworded prompts equally for random words (median rank 4 vs 4; neutral-prompt KL 2.27 for random targets), so its transfer says nothing about the fact. Multipliers only: true answers end at median rank 36 on reworded prompts (from 94), random words stay near rank 11,832 (from 17,504); not rank-matched. Entry 30 section 11.

* **Also 2026-10-04 (correction)**: the earlier rule "if a target cannot be steered near the top the model does not represent it" (Journal 1 class "knowledge absent (unfixable)", the deep-prompt reading in Entries 22 to 26) is withdrawn: the additive edit reaches rank 1 on random unrelated words (20 of 20), so reachability by steering says nothing either way. The additive edit is a narrow form of feature injection (injects the answer's output direction, not a fact). A graded measure (smallest edit needed) is proposed but must be compared with rank-matched controls; the 29% vs 177% edit-size gap seen so far is largely explained by the different starting ranks (94 vs 17,504). Entry 30 section 12, hypothesis H23.

* **Also 2026-10-04 (app bug)**: found that the author's running app server tokenised prompts without the start-of-text token (exact match: rank 28 / 0.295% / 21.80% = `prepend_bos=False`), caused by a thread race in `model.to_tokens(..., prepend_bos=False)` (reproduced with two threads). Audit: 16 of 57 recorded app runs affected, all from one server process started 3 Oct 23:27; none used in any table. Fixed with `tokens_without_bos` and a per-run self-heal (Entry 30 section 13).

* **Also 2026-10-05 (critique test run)**: `tools/test_generation_quality.py` on 40 held-out prompts, true targets (22 min): sequential collapse is mild (fluency unchanged, distinct-2 0.93 to 0.85-0.89, loops 0% to 2-12%, the target repeats about 0.65 times per continuation); the edit overrides "do not say X" (X first in 45% unedited, 68% gd, 88-90% additive). Entry 30 section 15; data `packs/generation_quality/true40/`.
* **Also 2026-10-05 (macOS port)**: device layer (`src/device_utils.py`: sync/empty-cache for cuda and mps), `mac_requirements.txt`, `scripts/*_mac.sh`, `tools/check_mac_parity.py`; Batch tab GPU panel and POSIX worker handling. Not yet run on a Mac.

### 2026-10-09
* **Title**: Repository cleanup, branch `cross-model-transfer`, and steps A0 to A2 of the cross-model transfer study
* **Motivation**: Move the most stable version (`gradient-descent-editing`) to `main`, remove stale branches, and start the new study: can the change an edit makes in one GPT-2 model be sent into another at inference time?
* **Summary**: `main` fast-forwarded from `9af68ec` to `1eeff8d` (100 commits, nothing lost); seven old branches deleted after checking each was contained in main (their SHAs were recorded); the paper draft, paper plan and cross-model idea note kept on main (`bc35482`); `gradient-descent-editing` and three side branches deleted; `rome-factual-editing` kept (worked on by a team member). New branch `cross-model-transfer` with the plan (`docs/cross_model_transfer/PLAN.md`, `DESIGN.md`). A0: both models and wikitext-2 set up, tokenizers identical, all checks passed. A1: 50 linear translators between GPT-2 small and medium fitted on 500,126 tokens (per-dimension R-squared at small 8 / medium 16: 0.581 one way, 0.661 the other; gradient descent matches the exact fit, 0.6287 vs 0.6289). A2: stitching recovers 91 to 98% of the receiver's loss on all eight pairs tested; translated sentence differences carry the word well (0.63 to 0.76) and the last position weakly (0.24 to 0.31); Gate 1 (proposed 0.3 line) passes for medium to small (2 of 4 pairs), not for small to medium (best 0.294).
* **Files involved**: `src/transfer/`, `tools/transfer/00_setup_check.py`, `01_fit_maps.py`, `02_check_maps.py`, `docs/Research_Journal/31.md`, `docs/Research_Journal/packs/cross_model_transfer/`
* **Status**: Completed (translator checks); no fact or edit has been transferred yet

### 2026-10-10
* **Title**: Country-swap positive control (B1) and a 30-record pilot of exporting a real edit (D)
* **Motivation**: Before transferring edits, show that a meaning change can be sent through the translator and moves the receiver, with controls; then look at what the project's gradient-descent edit does when its change is sent small to medium.
* **Summary**: B1 (Gate 2 fixed before the run): the translated "France became Germany" difference makes the swapped capital the receiver's top answer on 55% of 1,021 test cases (small to medium) and 33% of 695 (medium to small); random vector, random map and a different pair's difference 0 to 1%; the receiver's own difference 100%; median rank of the swapped capital 1431 to 1 and 2351 to 5. PASS in both directions. Six other layer pairs 36 to 79% (earlier layers better; exploratory). A first full run was lost to a too-short background time limit; the tool now saves each pair. Pilot (not a result): the gradient-descent edit reaches rank 1 in small on 17 of 30 counterfactual targets; 89% of its change sits at the subject's positions, 5% at the last; the translated edit raises the target in medium from median rank 165 to about 22 and makes it top-1 in up to 20% of records (random 0 to 3%, another record's edit 0%), but a plain push along the target word's own output direction does it in 87 to 100% at doses 2 to 4, so top-1 alone is not evidence of a fact.
* **Files involved**: `tools/transfer/03_swap_control.py`, `04_export_pilot.py`, `src/transfer/swap.py`, `docs/Research_Journal/31.md`, `docs/Research_Journal/packs/cross_model_transfer/`
* **Status**: B1 completed; D pilot exploratory only; Phases C, D (controlled) and E not started

### 2026-10-10 (overnight)
* **Title**: Controlled export of the gradient-descent edit, neural translator, and the fact-or-word-push test (Entry 32)
* **Motivation**: Does the edit tuned in GPT-2 small, translated and added to GPT-2 medium, make medium say the new fact, on rewordings, without damaging other knowledge, and more than controls do? Is a neural translator better than the linear one? Is the edit a fact edit or a word push?
* **Summary**: Benchmark of 200 CounterFact records (50 dev, 150 test), edit reaches rank 1 in small on 58%. Controlled export (88 test records whose edit succeeded in small): top-1 5% (layer 12) and 8% (layer 16), median rank of the target 142 to 23 and 14, rewordings +15 and +16 points, neighbours -4 to -5 points; random vector and another record's edit 0%; word push far stronger at top-1 but destroys neighbours; norm-matched dose agrees. **Gate 3 (fixed before the run) fails** on the top-1 rate and the margin over the strongest control. Neural translator: fits clearly better (R-squared +0.09 to +0.11) but exports no better. Phase C: the pre-stated rule gives read-out steering (rewordings gain not above the random word's at p < 0.01; subject-only edit succeeds on 12% against 59%); a statement of Entry 31 (edit mainly changes the subject's representation) is withdrawn. Mistakes recorded: random-word gain measured against the wrong start rank in the first printed run (fixed in the analysis tool), background time limits, harness backslash halving.
* **Files involved**: `src/transfer/{benchmark,export_eval,mlp_maps}.py`, `src/gradient_editing_masked.py`, `tools/transfer/05` to `09`, `tools/make_transfer_figures.py`, `docs/Research_Journal/32.md`, `docs/Research_Journal/images/e32_*`, `docs/Research_Journal/packs/cross_model_transfer/`
* **Status**: Completed (D, C, neural translator); Import (phase E) and the Gemma replication not started

### 2026-10-10 (daytime)
* **Title**: Several injection layers, the ceiling test (an edit tuned through the translator), a translator trained on edits (design), related work (Entry 33)
* **Motivation**: After the failed export (Entry 32), is the single injection layer the bottleneck? Can the channel carry a rank-1 edit at all? Can one trained map do it without tuning on medium?
* **Summary**: D5, translated edit at 15 configurations of medium layers: best single layer 8%, best split set 7%, Gate 4 fails; layers 8, 12 and 16 alike. D6, edit tuned through the translator against medium's output: top-1 95% against 8% for the original edit, rewordings 60% (random-word edit 42%, unchanged 36%), neighbours fall 8 points (the edit itself falls 9 in small); Gate 5 fails on the neighbour condition only; a ceiling, not a transfer. D7 (linear translator trained on 289 edits with an output-matching loss, held-out target words and subjects, Gate 6 written first): top-1 18% against 8% for the ridge map, median rank 14 to 7, rewordings 56% (ridge 52%), neighbours fall 7 points (the edit itself 10); Gate 6 fails on top-1 and on the rewordings condition; the training set was smaller than planned (621 records, 289 successful edits, not about 800). Related work: Chen et al., NeurIPS 2025 (affine stitching, includes GPT-2 small and medium, no fact edits), note in `docs/cross_model_transfer/RELATED_WORK.md`, written from a summary. Clarified with the author: the edit is the multiplier edit, not the additive one (additive transfer untested). Mistakes recorded: a proposed rerun bar written before the author answered; Gate 5's neighbour line stricter than the source edit's own collateral; a pasted review's gradient warning does not apply.
* **Files involved**: `src/transfer/{multilayer,b_aware_edit,output_matching}.py`, `tools/transfer/10` to `13`, `tools/make_transfer_figures_33.py`, `docs/Research_Journal/33.md`, `docs/Research_Journal/images/e33_*`, `docs/cross_model_transfer/{PLAN,RELATED_WORK}.md`, `docs/Research_Journal/packs/cross_model_transfer/`
* **Status**: D5, D6 and D7 completed (D7: Gate 6 fails, real improvement over the ridge map)

### 2026-10-10 (evening)
* **Title**: Batched training and the data-size check of the map trained on edits (Entry 33 section 4.4)
* **Motivation**: D7's training took an hour; before generating more edits, does the result rise with more training data?
* **Summary**: Training made about 14 times faster by batching (checked against the old code: per-text loss differs by 3e-6, gradient cosine 1.000000; the 100% run reproduces D7). Data-size check: top-1 13%, 13%, 20% at 25%, 50%, 100% of the 289 edits; the pre-set rule says RISING, with a flat-then-jump shape and noisy seeds. Best epoch at 100% was always the last, so a longer-training test (D9, rule written) is next, then a rank-1 term (D10) and a neural map (D11). A median-convention inconsistency in the new tool was found and fixed.
* **Files involved**: `src/transfer/om_batched.py`, `tools/transfer/14_datasize_check.py`, `docs/Research_Journal/33.md`, `docs/cross_model_transfer/PLAN.md`
* **Status**: D8 completed; D9 to run

### 2026-10-10 (night)
* **Title**: Longer training (D9), the dose check (D9b), run logs, figures and paper notes (Entry 33 sections 4.5 and 4.6)
* **Motivation**: D8 suggested more data helps and the best epoch was always the last, so was training too short? Then why did the longer training look 17 points worse?
* **Summary**: D9 (40 against 15 epochs): by its pre-set rule no improvement (2% against 20%), because the dose rule picked dose 1.0 in all three seeds. D9b (post-hoc, every dose): at dose 2.0 both lengths give 20%; the drop was the dose chosen on 28 dev records; longer training neither helps nor hurts, the single linear map has plateaued at about 20%. Dose now chosen by dev gain from D10 on. Infrastructure: run logs saved automatically (`src/transfer/runlog.py`) and the earlier terminal outputs saved in the pack (`logs/`), tools save epoch curves, figures 4 to 7 from saved data, `PAPER_NOTES.md`, `CLAUDE.md` with the logging rules.
* **Files involved**: `tools/transfer/15_dose_table.py`, `src/transfer/{om_batched,runlog}.py`, `tools/make_transfer_figures_33.py`, `docs/Research_Journal/33.md`, `docs/cross_model_transfer/{PLAN,PAPER_NOTES}.md`, `CLAUDE.md`, `docs/Research_Journal/packs/cross_model_transfer/logs/`
* **Status**: D9 and D9b completed; D10 (rank-1 term) and D11 (neural map) next, rules to be written first

### 2026-10-10 (late night)
* **Title**: A rank-1 term in the training loss of the map (D10, Entry 33 section 4.7)
* **Motivation**: D7 to D9b plateau at about 20% and training length is ruled out. D6 reaches 95% because its loss says directly "make the target top"; does giving the map that instruction in training help on unseen edits?
* **Summary**: Plan, six-value lambda grid, selection rule, Gate 7 and readings written and committed (`024f5f6`) before the build and run. Result: the best dev score was at lambda 0, so Gate 7 judged the D7 baseline and failed (top-1 20% against 50%, rewordings lift 0.74 against 0.8); on test every lambda gives 16 to 21%. The training shortfall from rank 1 falls from about 2.4 logits to 0.15: the map memorises the 289 training targets and does not generalise to unseen ones. A hypothesis for the next step: a map only transforms the edit it is given, while D6 changes the edit itself; amortised D6 (D12) is an untested idea, a small neural map (D11) the cheaper option.
* **Files involved**: `tools/transfer/16_rank1_term.py`, `src/transfer/{om_batched,om_judge}.py`, `tools/make_transfer_figures_33.py`, `docs/Research_Journal/33.md`, `docs/cross_model_transfer/PLAN.md`
* **Status**: D10 completed (Gate 7 fails); next candidates D11, D12 (idea), more edits, Import

### 2026-10-10 (night, Import)
* **Title**: Import, step E1: what does medium know that small does not? (Entry 34)
* **Motivation**: The author wants to test the original Import idea (medium teaches small); first, is there enough material?
* **Summary**: E1 count (Import feasibility): of 21,919 CounterFact facts, 785 are right in medium and wrong in small (780 outside the ROME dev and pilot records), 440 the other way, 1,380 both right, 19,314 both wrong; for the 780 candidates the true answer is within small's top five for about 71% (320 at rank 2), medium's median probability is 0.19 (54 at 0.5 or more), 28 relations, 169 distinct answers with the ten most common covering about 44%. Rule: 300 or more = worth planning in full -> met. Nothing about whether Import works. Rule written in the plan before the run.
* **Files involved**: `tools/transfer/18_import_count.py`, `docs/Research_Journal/34.md`, `docs/cross_model_transfer/PLAN.md`, `docs/Research_Journal/packs/cross_model_transfer/e1_import_count.json`
* **Status**: E1 completed; E2 to E4 (the Import experiment) to be designed with a gate first

### 2026-10-10 (night, Import result)
* **Title**: Import by replacement: small with medium's translated state (Entry 35)
* **Motivation**: The author's original idea: does medium's state, written into small, make small answer facts it gets wrong, without being told the answer?
* **Summary**: Plan, gate (raised to 50% after the author's comments; replacement as the primary arm) and tool committed before the run (`1719459`). Result: Import by replacement (small's layer-8 state replaced by medium's layer-16 state translated by m2s_L16_L8; no training): rescues 262 of the 780 candidates (33.6%; controls: wrong-sentence state 0.8%, average state 0.0%, random translator 0.0%; paired p = 4e-73), but loses 298 of the 1,372 both-right facts (78.3% survive) and 304 of the 437 small-right-medium-wrong facts (30.4% survive); accuracy over all 21,789 facts falls from 8.30% (small) to 7.17% (medium alone: 9.88%); agreement with medium's answers 71.1% to 74.7%; no gain on rewordings (13.1% against 13.8%); the SAE-filtered difference rescues only 6.3%. Gate 8 FAILS: (a) 33.6% against 50%, (b) 7.2% against 9.1%, (d) 78.3% against 90%; (c) holds. Reading fixed in advance: Import with this map does not work as an improvement; the information that crosses is real (controls about 0).
* **Files involved**: `tools/transfer/19_import_experiment.py`, `src/transfer/import_eval.py`, `docs/Research_Journal/35.md`, `docs/Research_Journal/images/e35_fig1_import.*`, `docs/cross_model_transfer/PLAN.md`
* **Status**: E2 to E4 completed (Gate 8 fails); softer variants (dose scored on net accuracy, gating by medium's confidence, partial positions) not tested

### 2026-10-11 (Export again, step D6b)
* **Title**: How different is D6's solution from the original edit? (Entry 33 section 4.8)
* **Motivation**: Back to Export (small to medium) after Import; the author asked for options other than the ceiling method. First a quick diagnostic with data already on disk.
* **Summary**: D6b (how different is D6's solution from the original edit? 84 primary test records where D6 reached rank 1 in training): median cosine at the last position 0.51 (quartiles 0.25 to 0.70; chance 0.02 and 0.05), all positions but the first 0.26, dial vectors 0.27 (sign agreement 0.62), D6's change is 0.71 of the original's size and its best scalar multiple of the original's direction is 0.35. By the rule fixed beforehand (at least 0.5 = same direction, different strength) the median just meets the top bucket; read with the spread and the lower all-position agreement it is partly the same direction and partly new, and D6 is smaller, not a harder push.
* **Files involved**: `tools/transfer/20_d6_vs_original.py`, `docs/Research_Journal/33.md`, `docs/cross_model_transfer/PLAN.md`
* **Status**: D6b completed; next in the chain: more varied training edits (synthetic targets), then a regularised neural map

### 2026-10-11 (Export, step D13: the edits are generated)
* **Title**: More, and more varied, training edits (Entry 36 will hold the training result)
* **Summary**: D13 edit generation (3 workers, 3,542 s): 2,894 edits toward random rank-matched words, 1,221 reached rank 1 in small (42%) on 749 sentences toward 759 distinct words (the 289 real edits cover only 116 distinct words); no target word is a true or counterfactual word of any dev or test record; the synthetic edits are easier than the real ones (median start rank 54 against 116 for the real training edits and 105 for the test records), so one exploratory training set (real plus the 457 synthetic edits starting at rank 100 or worse) was added to the training tool before its run.
* **Files involved**: `tools/transfer/21_varied_edits.py`, `tools/transfer/22_varied_training.py`, `docs/cross_model_transfer/PLAN.md`, `docs/Research_Journal/packs/cross_model_transfer/d13_edits_summary.json`
* **Status**: generation done; the training run (about an hour) is next

### 2026-10-11 (Export, steps D14 and S2: a plan queued, a statement corrected)
* **Title**: Regularised maps queued for the night; D6's rewordings contrast does not hold up
* **Summary**: While the D13 training run and the E5/S1 queue run, step D14 (the D13 full set with four maps that have less freedom: stronger pull toward the ridge map, penalty 1 and 10; a rank-16 nudge; a small dropout neural map; Gate 9 unchanged, two seeds, readings fixed before the build) was planned, built, rehearsed on a tiny random model and then on the CPU with real files, committed (`bee1e6c`, `f191bd6`) and queued behind S1 in a second runner. Step S2, a missing significance test, was run by hand: D6's real-target rewordings lift (24.4 points) against the random word's lift over its own unchanged level (17.0 points): difference +7.4 points, interval 0.0 to +15.3, p = 0.031 > 0.01, NOT SHOWN. The earlier claim of a four-fold contrast (24 against 6) compared the random word with the real target's unchanged level and is withdrawn (Entry 33 section 4.9, PAPER_NOTES).
* **Files involved**: `tools/transfer/25_regularised_map.py`, `src/transfer/reg_maps.py`, `tools/transfer/26_d6_rewordings_test.py`, `tools/transfer/queue_after.py`, `docs/cross_model_transfer/PLAN.md`, `docs/Research_Journal/33.md`, `docs/Research_Journal/packs/cross_model_transfer/s2_d6_rewordings_test.json`
* **Status**: S2 done; D14 queued (about 2.5 hours after S1)

### 2026-10-11 (Export, step S3: D7's rewordings are the same for a random word)
* **Title**: The random-word baseline recomputed for the D7 map
* **Summary**: Run on the CPU while the D13 run used the GPU (240 s; plan and tool committed first, `54b0523`). Against each arm's own unchanged level the D7 map lifts rewordings by 19.9 points for the real target and 18.2 points for a random word (difference +1.7, interval -6.8 to +10.2, p = 0.36); a random word also loses more neighbours (12.3 against 7.3 points). So what carries to rewordings is a push toward the edit's target word, not something specific to the fact; the earlier "specific to the target (rewordings 43% against 56%)" is withdrawn. Gate verdicts do not change. The tool can recompute the same for the D10, D13 and D14 maps.
* **Files involved**: `tools/transfer/27_random_word_baseline.py`, `src/transfer/baseline_judge.py`, `docs/cross_model_transfer/PLAN.md`, `docs/Research_Journal/33.md`, `docs/Research_Journal/packs/cross_model_transfer/s3_random_word_baseline.json`
* **Status**: done
