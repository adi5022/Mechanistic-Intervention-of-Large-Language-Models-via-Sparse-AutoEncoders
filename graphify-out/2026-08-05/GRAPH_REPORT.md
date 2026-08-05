# Graph Report - FeatureScalpel  (2026-08-04)

## Corpus Check
- 76 files · ~509,706 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 402 nodes · 542 edges · 54 communities (22 shown, 32 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.7)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4be9f689`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- experiment_app.py
- editing.py
- history.md
- evaluation.py
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
- GPT-2 Small
- Project Handoff Documentation
- Research Log: Mechanistic Intervention of LLMs
- Stage 1 Baseline Target Report
- README
- Residual Stream
- Sparse Autoencoder (SAE)
- Research Journal Entry 10: Layer Intervention Depth Characterization Benchmark
- Research Journal Entry 9: Pretrained SAE Dictionary Audit for gpt2-small-res-jb
- investigate_gpt2_saes.py

## God Nodes (most connected - your core abstractions)
1. `load_model_and_sae()` - 17 edges
2. `get_top_competitor_features()` - 15 edges
3. `make_ablation_hook()` - 15 edges
4. `Research Log: Investigating SAE-Based Activation Manipulation` - 14 edges
5. `get_target_token_id()` - 13 edges
6. `run_layer_benchmark()` - 12 edges
7. `make_joint_ablation_hook()` - 10 edges
8. `get_top_target_features()` - 9 edges
9. `query_groq()` - 9 edges
10. `make_signed_ablation_hook()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `get_cached_model_and_sae()` --calls--> `load_model_and_sae()`  [EXTRACTED]
  app.py → src/sae_utils.py
- `get_predictions()` --calls--> `make_ablation_hook()`  [EXTRACTED]
  app.py → src/hooks.py
- `run_trace()` --calls--> `make_ablation_hook()`  [EXTRACTED]
  test.py → src/hooks.py
- `test_monosemanticity_helpers_run_on_real_model_and_sae()` --calls--> `load_model_and_sae()`  [EXTRACTED]
  tests/test_monosemanticity.py → src/sae_utils.py
- `get_cached_base_model()` --calls--> `load_base_model()`  [EXTRACTED]
  experiment_app.py → src/sae_utils.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Research Evolution: From Single to Multi-Feature Intervention** — docs_research_journal_2, docs_research_journal_3, docs_research_journal_4, docs_research_journal_8 [EXTRACTED 0.90]
- **Theoretical Framework** — residual_stream, sparse_autoencoder, distributed_support_hypothesis [INFERRED 0.85]

## Communities (54 total, 32 thin omitted)

### Community 0 - "experiment_app.py"
Cohesion: 0.06
Nodes (57): cache_data, get_neuronpedia_explanation(), make_feature_hover_link(), Renders a structured 6-part Educational XAI Guidance Card in the UI., Renders a structured, informative empty state card when data or metadata is…, render_empty_state_card(), render_xai_guidance_card(), slow (+49 more)

### Community 1 - "editing.py"
Cohesion: 0.07
Nodes (52): get_cached_base_model(), get_cached_sae(), cache_resource, get_cached_model_and_sae(), main(), cache_resource, Layer Intervention Benchmark UI Dedicated Streamlit application for…, Cached loader for HookedTransformer and Layer-specific SAE. (+44 more)

### Community 2 - "history.md"
Cohesion: 0.06
Nodes (33): 1. Core Algorithmic & Engine Changes, 2. Dashboard Improvements (`experiment_app.py`), 3. Research Infrastructure, Background, Cambridge Diagnostic Output, Current Working Hypothesis (Unverified), Diagnostic Experiment, `docs/Research_Journal/4.md` (+25 more)

### Community 3 - "evaluation.py"
Cohesion: 0.13
Nodes (24): get_cached_model_and_sae(), get_predictions(), cache_resource, DataFrame, get_target_token_id(), Safely resolves a target string (e.g. ' Paris') to its token ID in the model's…, Safely resolves a target string (e.g. ' Paris') to its token ID in the model's…, Ranks the top-N active features by their causal effect on the target token's… (+16 more)

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
Cohesion: 0.13
Nodes (14): 2026-07-08: Project Setup, 2026-07-10: First Automated Validation Breakthrough (Causal Feature Selector), 2026-07-19: Iterative Ablation & Specificity Verification, 2026-07-20: Competitor-Focused Iterative Ablation, 2026-07-24: Whole-Combination Safety & Weighted Multi-Competitor Reduction, 2026-07-25: Diagnostic Investigation of Rank Regression, 2026-07-25 (Session 2): Shared Feature Aggregation Experiment, 2026-07-25 (Session 3): Multi-Feature Representation Diagnostic (+6 more)

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

### Community 50 - "Research Journal Entry 10: Layer Intervention Depth Characterization Benchmark"
Cohesion: 0.22
Nodes (8): 1. Layman's Summary & Purpose, 2. Experimental Data & Results Table, 3. Visualization Graphs & Tracked Progress, 4. Key Observations & Research Progress, A. Probability Gain (%) vs Layer Depth, B. Rank Improvement vs Layer Depth, C. Execution Runtime (s) vs Layer Depth, Research Journal Entry 10: Layer Intervention Depth Characterization Benchmark

### Community 51 - "Research Journal Entry 9: Pretrained SAE Dictionary Audit for gpt2-small-res-jb"
Cohesion: 0.25
Nodes (7): 1. Executive Summary, 2. Experimental / Audit Methodology, 3. Results & Availability Matrix, 4. Architectural Findings & Key Takeaways, A. Pretrained Hook Availability Table for `gpt2-small-res-jb`, B. Registered SAE IDs under `gpt2-small-res-jb`, Research Journal Entry 9: Pretrained SAE Dictionary Audit for gpt2-small-res-jb

### Community 52 - "investigate_gpt2_saes.py"
Cohesion: 0.50
Nodes (4): get_release_saes_map(), main(), Standalone script to investigate available pretrained SAE dictionaries in…, Returns dictionary mapping sae_id -> location for a given release name if…

## Knowledge Gaps
- **153 isolated node(s):** `MockConfig`, `graphify`, `Workflow: graphify`, `Project Context (read this fully before writing any code)`, `What has already been done and confirmed (do not re-derive from scratch —` (+148 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **32 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_model_and_sae()` connect `editing.py` to `experiment_app.py`, `evaluation.py`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Why does `get_target_token_id()` connect `evaluation.py` to `experiment_app.py`, `editing.py`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Why does `get_top_competitor_features()` connect `editing.py` to `experiment_app.py`, `evaluation.py`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **What connects `MockConfig`, `graphify`, `Workflow: graphify` to the rest of the system?**
  _153 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `experiment_app.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05711849957374254 - nodes in this community are weakly interconnected._
- **Should `editing.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06721215663354763 - nodes in this community are weakly interconnected._
- **Should `history.md` be split into smaller, more focused modules?**
  _Cohesion score 0.058823529411764705 - nodes in this community are weakly interconnected._