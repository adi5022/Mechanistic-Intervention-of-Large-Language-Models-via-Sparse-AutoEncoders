# Graph Report - FeatureScalpel  (2026-09-17)

## Corpus Check
- 151 files · ~571,826 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 562 nodes · 860 edges · 63 communities (30 shown, 33 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0c5a2041`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- experiment_app.py
- make_ablation_hook
- history.md
- sae_utils.py
- 4. Discussion & Next Steps
- Experiment: Weighted Multi-Competitor Reduction & Grammatical Feature Collapse on the Cambridge Case
- Experiment: Competitor-Focused Iterative Ablation
- Experiment: Multi-Feature Representation Diagnostic
- Research Log: Investigating SAE-Based Activation Manipulation
- Agent Task: Build FeatureScalpel MVP Prototype
- Experiment: Shared Feature Aggregation (max() vs sum())
- Experiment: Diagnostic Investigation of Rank Regression under Hybrid Steering
- Experiment 010: Weighted Multi-Feature Competitor Reduction Sweep
- Distributed Support (Loudspeaker) Hypothesis
- FeatureScalpel Research Directory
- Research Contribution Guidelines
- Stage 1 Report
- Research Log
- Research Paper Outline: Transient Activation Steering with Sparse Autoencoders for Factual Correction
- rules/graphify.md
- workflows/graphify.md
- Experiment Index
- abstract.md
- conclusion.md
- discussion.md
- experiments.md
- future_work.md
- introduction.md
- limitations.md
- methodology.md
- related_work.md
- results.md
- Causal Feature Selector
- Competitor-Focused Ablation
- Suppressed Knowledge
- Delta-Patching Ablation
- Full Technical Mechanics
- Project Progress & Roadmap
- Iterative Ablation & Specificity Verification
- Hybrid Sweep Diagnostics & Softmax Rank Regressions
- Shared Feature Aggregation & Clamping
- Step 1 — Iterative Ablation + Specificity Check
- GPT-2 Small
- Project Handoff Documentation
- Research Log: Mechanistic Intervention of LLMs
- Stage 1 Baseline Target Report
- README
- Residual Stream
- Sparse Autoencoder (SAE)
- validate_step3b.py
- Research Journal Entry 10: Layer Intervention Depth Characterization Benchmark
- Research Journal Entry 9: Pretrained SAE Dictionary Audit for gpt2-small-res-jb
- investigate_gpt2_saes.py
- Session History: Layer Intervention Benchmark Subsystem
- Research Journal Entry 11: Benchmark Infrastructure Optimization, Stage Profiling, and Batch Dataset Processing
- Research Journal Entry 12: Candidate Selection Mechanics, Delta Patching Formalization, ROME Dataset Integration, and Hardware Acceleration Benchmarking
- editing.py
- load_model_and_sae
- Research Journal Entry 13: Empirical Execution Cost Audit, Step 2 Safety Filter Optimization, and Benchmark UI Observability Enhancements
- Research Journal Entry 14: GPU Batched Candidate Evaluation (Steps 3A, 3B, 3C)
- ForwardPassTracker

## God Nodes (most connected - your core abstractions)
1. `load_model_and_sae()` - 29 edges
2. `get_top_competitor_features()` - 27 edges
3. `run_layer_benchmark()` - 25 edges
4. `get_target_token_id()` - 22 edges
5. `get_top_target_features()` - 18 edges
6. `Research Log: Investigating SAE-Based Activation Manipulation` - 18 edges
7. `make_ablation_hook()` - 17 edges
8. `load_base_model()` - 15 edges
9. `load_sae_for_layer()` - 13 edges
10. `main()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `get_predictions()` --calls--> `make_ablation_hook()`  [EXTRACTED]
  app.py → src/hooks.py
- `main()` --calls--> `run_layer_benchmark()`  [EXTRACTED]
  layer_benchmark.py → src/benchmark/layer_benchmark_runner.py
- `get_cached_model_and_sae()` --calls--> `load_base_model()`  [EXTRACTED]
  run_speed_analysis.py → src/sae_utils.py
- `get_cached_model_and_sae()` --calls--> `load_sae_for_layer()`  [EXTRACTED]
  run_speed_analysis.py → src/sae_utils.py
- `patched_get_top_competitor_features()` --calls--> `get_top_active_features()`  [EXTRACTED]
  scratch/audit_step1_instrumentation.py → src/editing.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Research Evolution: From Single to Multi-Feature Intervention** — docs_research_journal_2, docs_research_journal_3, docs_research_journal_4, docs_research_journal_8 [EXTRACTED 0.90]
- **Theoretical Framework** — residual_stream, sparse_autoencoder, distributed_support_hypothesis [INFERRED 0.85]

## Communities (63 total, 33 thin omitted)

### Community 0 - "experiment_app.py"
Cohesion: 0.06
Nodes (57): cache_data, get_neuronpedia_explanation(), make_feature_hover_link(), Renders a structured 6-part Educational XAI Guidance Card in the UI., Renders a structured, informative empty state card when data or metadata is…, render_empty_state_card(), render_xai_guidance_card(), slow (+49 more)

### Community 1 - "make_ablation_hook"
Cohesion: 0.11
Nodes (28): get_predictions(), DataFrame, Ranks the top-N active features by their causal effect on the target token's…, run_causal_selector(), check_specificity(), classify_fact(), iterative_ablate(), Evaluation metrics and scripts for measuring intervention success. (+20 more)

### Community 2 - "history.md"
Cohesion: 0.06
Nodes (33): 1. Core Algorithmic & Engine Changes, 2. Dashboard Improvements (`experiment_app.py`), 3. Research Infrastructure, Background, Cambridge Diagnostic Output, Current Working Hypothesis (Unverified), Diagnostic Experiment, `docs/Research_Journal/4.md` (+25 more)

### Community 3 - "sae_utils.py"
Cohesion: 0.14
Nodes (21): get_cached_base_model(), get_cached_sae(), cache_resource, get_cached_base_model(), get_cached_model_and_sae(), get_cached_sae(), main(), cache_resource (+13 more)

### Community 4 - "4. Discussion & Next Steps"
Cohesion: 0.11
Nodes (18): 1. Context & Motivation, 2. Hypothesis & Setup, 3. Results & Observations, 4. Discussion & Next Steps, Background, Evidence & Artifacts, Experiment: [Title], Experimental Setup (+10 more)

### Community 5 - "Experiment: Weighted Multi-Competitor Reduction & Grammatical Feature Collapse on the Cambridge Case"
Cohesion: 0.12
Nodes (15): 1. Context & Motivation, 2. Hypothesis & Setup, 3. Results & Observations, 4. Discussion & Next Steps, Background, Experiment: Weighted Multi-Competitor Reduction & Grammatical Feature Collapse on the Cambridge Case, Experimental Setup, Hypotheses (+7 more)

### Community 6 - "Experiment: Competitor-Focused Iterative Ablation"
Cohesion: 0.13
Nodes (14): 1. Context & Motivation, 2. Hypothesis & Setup, 3. Results & Observations, 4. Discussion & Next Steps, Background, Experiment: Competitor-Focused Iterative Ablation, Experimental Setup, Hypothesis (+6 more)

### Community 7 - "Experiment: Multi-Feature Representation Diagnostic"
Cohesion: 0.13
Nodes (14): 1. Context & Objective, 2. Methodology & Verification of Delta Additivity, 3. Results & Diagnostics, 4. Discussion & Research Findings, A. Concentration Decay Profile Table, B. Shared Feature Overlap Table, Background, C. Shared-Feature Statistics (+6 more)

### Community 8 - "Research Log: Investigating SAE-Based Activation Manipulation"
Cohesion: 0.11
Nodes (18): 2026-07-08: Project Setup, 2026-07-10: First Automated Validation Breakthrough (Causal Feature Selector), 2026-07-19: Iterative Ablation & Specificity Verification, 2026-07-20: Competitor-Focused Iterative Ablation, 2026-07-24: Whole-Combination Safety & Weighted Multi-Competitor Reduction, 2026-07-25: Diagnostic Investigation of Rank Regression, 2026-07-25 (Session 2): Shared Feature Aggregation Experiment, 2026-07-25 (Session 3): Multi-Feature Representation Diagnostic (+10 more)

### Community 9 - "Agent Task: Build FeatureScalpel MVP Prototype"
Cohesion: 0.17
Nodes (11): Agent Task: Build FeatureScalpel MVP Prototype, Deliverable, Explicit non-goals for this prototype (do not build these; they would blow, Project Context (read this fully before writing any code), Section 1 — Feature Selector (the new, automated part), Section 2 — The Slider (manual intervention, already validated), Section 3 — Specificity / Side-Effect Check, Technical requirements (+3 more)

### Community 10 - "Experiment: Shared Feature Aggregation (max() vs sum())"
Cohesion: 0.17
Nodes (11): 1. Context & Objective, 2. Hypothesis & Setup, 3. Results & Diagnostics, 4. Discussion & Scientific Conclusions, Aggregation Diagnostics, Background, Cambridge Evaluation Metrics, Experiment: Shared Feature Aggregation (max() vs sum()) (+3 more)

### Community 11 - "Experiment: Diagnostic Investigation of Rank Regression under Hybrid Steering"
Cohesion: 0.20
Nodes (9): 1. Context & Investigation Goal, 2. Experimental Data & Reproduction Logs, 3. Analysis & Key Findings, Diagnosis of Regressions, Experiment: Diagnostic Investigation of Rank Regression under Hybrid Steering, Question, Sweep Results (Raw Candidate Features - Unsafe Included), Sweep Results (Safe Features Only) (+1 more)

### Community 12 - "Experiment 010: Weighted Multi-Feature Competitor Reduction Sweep"
Cohesion: 0.29
Nodes (6): 1. Regression Test ($K = 1$), 2. Full Sweep Results Table, Experiment 010: Weighted Multi-Feature Competitor Reduction Sweep, Experimental Parameters, Interpretations & Conclusions, Observations

### Community 13 - "Distributed Support (Loudspeaker) Hypothesis"
Cohesion: 0.33
Nodes (6): Hybrid Mute & Boost Steering, Distributed Support (Loudspeaker) Hypothesis, Whole-Combination Safety & Weighted Multi-Competitor Reduction, Multi-Feature Representation Diagnostics, Multi-Feature Steering Evaluation, Hypothesis Log

### Community 14 - "FeatureScalpel Research Directory"
Cohesion: 0.40
Nodes (4): Directory Structure, Distinction of Concepts, FeatureScalpel Research Directory, Purpose of this Directory

### Community 15 - "Research Contribution Guidelines"
Cohesion: 0.50
Nodes (3): Research Contribution Guidelines, Research Workflow, Writing Guidelines

### Community 41 - "Step 1 — Iterative Ablation + Specificity Check"
Cohesion: 0.25
Nodes (7): 1. Iterative ablation loop (`editing.py`), 2. Specificity check (new: `specificity.py`, or a function added to `evaluation.py`), 3. Batch runner, Explicit constraints (do not let this expand), Step 1 — Iterative Ablation + Specificity Check, What "done" looks like, What gets built

### Community 49 - "validate_step3b.py"
Cohesion: 0.18
Nodes (17): compare_ordered_lists(), compute_stability_margin(), count_forward_calls(), main(), Evaluates 6 prompts (1 canonical MIT + 5 spread-out prompts from FACT_BANK) at…, Wraps model.forward exclusively to count forward calls accurately., Compares two feature ID lists element-by-element in order. Returns…, run_multi_prompt_check() (+9 more)

### Community 50 - "Research Journal Entry 10: Layer Intervention Depth Characterization Benchmark"
Cohesion: 0.22
Nodes (8): 1. Layman's Summary & Purpose, 2. Experimental Data & Results Table, 3. Visualization Graphs & Tracked Progress, 4. Key Observations & Research Progress, A. Probability Gain (%) vs Layer Depth, B. Rank Improvement vs Layer Depth, C. Execution Runtime (s) vs Layer Depth, Research Journal Entry 10: Layer Intervention Depth Characterization Benchmark

### Community 51 - "Research Journal Entry 9: Pretrained SAE Dictionary Audit for gpt2-small-res-jb"
Cohesion: 0.25
Nodes (7): 1. Executive Summary, 2. Experimental / Audit Methodology, 3. Results & Availability Matrix, 4. Architectural Findings & Key Takeaways, A. Pretrained Hook Availability Table for `gpt2-small-res-jb`, B. Registered SAE IDs under `gpt2-small-res-jb`, Research Journal Entry 9: Pretrained SAE Dictionary Audit for gpt2-small-res-jb

### Community 52 - "investigate_gpt2_saes.py"
Cohesion: 0.50
Nodes (4): get_release_saes_map(), main(), Standalone script to investigate available pretrained SAE dictionaries in…, Returns dictionary mapping sae_id -> location for a given release name if…

### Community 54 - "Session History: Layer Intervention Benchmark Subsystem"
Cohesion: 0.14
Nodes (13): 1. Pretrained SAE Dictionary Audit (`gpt2-small-res-jb`), 2. Layer Intervention Benchmark Subsystem (Phase 1 Infrastructure), 3. Multi-Prompt Dataset Extension, 4. Verification & Validation, 6. Hardware Acceleration, Unpacking Bug Fixes & ROME Dataset Integration, Execution & Findings, Execution & Findings, Implementation (+5 more)

### Community 55 - "Research Journal Entry 11: Benchmark Infrastructure Optimization, Stage Profiling, and Batch Dataset Processing"
Cohesion: 0.18
Nodes (10): 1. Objective, 2.1 Decoupled Caching Architecture, 2.2 Stage Profiling & Extended Metadata, 2.3 Batch Dataset Processing & UI Management, 2. Architecture & Implementation Highlights, 3.1 Layer Depth Sensitivity Summary (Hybrid Mute & Boost), 3.2 Profiling Breakdown (Layer 8 Baseline), 3. Empirical Verification & Performance Breakdown (+2 more)

### Community 56 - "Research Journal Entry 12: Candidate Selection Mechanics, Delta Patching Formalization, ROME Dataset Integration, and Hardware Acceleration Benchmarking"
Cohesion: 0.17
Nodes (11): 1. Objective, 2.1 Two-Stage Candidate Selection Mechanics, 2.2 Delta Patching Formulation & Literature Lineage, 2. Theoretical & Algorithmic Foundations, 3.1 Speed & Resource Results, 3.2 VRAM Footprint & Efficiency, 3. Empirical Hardware Speed Analysis, 4. UI & Tooling Enhancements (+3 more)

### Community 57 - "editing.py"
Cohesion: 0.09
Nodes (47): benchmark_device(), get_cached_model_and_sae(), main(), Speed Analysis Benchmark Script Compares GPU (NVIDIA GeForce GTX 1660 Ti) vs…, Caches base model and layer SAEs in RAM/VRAM to avoid redundant reload disk I/O., instrument_model(), Step 2 Validation & Performance Comparison Script Compares Reference vs…, run_step2_validation() (+39 more)

### Community 58 - "load_model_and_sae"
Cohesion: 0.07
Nodes (20): get_cached_model_and_sae(), cache_resource, HookedTransformer, ForwardPassTracker, patched_check_boost_safe(), patched_get_top_competitor_features(), patched_get_top_target_features(), Temporary Audit Instrumentation Script (Step 1 Audit Only) Measures execution… (+12 more)

### Community 59 - "Research Journal Entry 13: Empirical Execution Cost Audit, Step 2 Safety Filter Optimization, and Benchmark UI Observability Enhancements"
Cohesion: 0.17
Nodes (11): 1. Executive Summary, 2. Step 1: Execution-Cost Audit & Baseline Accounting, 3. Step 2: Safety Filter Redundancy Elimination, 4. Performance & Execution Cost Comparison, 5. Authoritative Stage-by-Stage Forward Pass Accounting, 6. Benchmark UI Observability & Dashboard Enhancements, 7. Git Verification & Commit History, Empirical Audit Findings: (+3 more)

### Community 60 - "Research Journal Entry 14: GPU Batched Candidate Evaluation (Steps 3A, 3B, 3C)"
Cohesion: 0.20
Nodes (9): 1. Accounting Framework: Forward Calls vs. Sequence Evaluations, 2. Implemented Sub-Steps & Verification Gates, 3. End-to-End Execution Trace Comparison, 4. Scientific Equivalence Results (Canonical Case), Executive Summary, Research Journal Entry 14: GPU Batched Candidate Evaluation (Steps 3A, 3B, 3C), Step 3A — Pre-Batching Cleanup, Step 3B — Batched Causal Ranking Stages (+1 more)

## Knowledge Gaps
- **199 isolated node(s):** `MockConfig`, `graphify`, `Workflow: graphify`, `Project Context (read this fully before writing any code)`, `What has already been done and confirmed (do not re-derive from scratch —` (+194 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **33 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_model_and_sae()` connect `load_model_and_sae` to `experiment_app.py`, `make_ablation_hook`, `sae_utils.py`, `validate_step3b.py`, `editing.py`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Why does `get_top_competitor_features()` connect `editing.py` to `experiment_app.py`, `make_ablation_hook`, `sae_utils.py`, `validate_step3b.py`, `load_model_and_sae`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Why does `get_target_token_id()` connect `editing.py` to `experiment_app.py`, `validate_step3b.py`, `sae_utils.py`, `make_ablation_hook`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **What connects `MockConfig`, `graphify`, `Workflow: graphify` to the rest of the system?**
  _199 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `experiment_app.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05711849957374254 - nodes in this community are weakly interconnected._
- **Should `make_ablation_hook` be split into smaller, more focused modules?**
  _Cohesion score 0.10795454545454546 - nodes in this community are weakly interconnected._
- **Should `history.md` be split into smaller, more focused modules?**
  _Cohesion score 0.058823529411764705 - nodes in this community are weakly interconnected._