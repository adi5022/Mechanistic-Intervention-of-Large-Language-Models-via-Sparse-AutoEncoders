# Research Log: Investigating SAE-Based Activation Manipulation

## 2026-07-08: Project Setup
* Established the project title, repository, and initial directory structure.

## 2026-07-10: First Automated Validation Breakthrough (Causal Feature Selector)
* Automated the discovery of causal features using a hook-based intervention pipeline.
* Built the **Causal Feature Selector** (`src/editing.py`) to rank features by their actual causal effect on target prediction probabilities, validating successfully on the Eiffel Tower prompt.

## 2026-07-19: Iterative Ablation & Specificity Verification
* Evaluated multi-feature iterative ablation at strength 0.3 on 22 suppressed facts.
* Achieved a 4.55% success rate. Discovered a logical error in the intervention strategy: muting the correct target token's helper features made it even weaker. Realized we must target the incorrect competitor features instead, or use amplification (boosting).

## 2026-07-20: Competitor-Focused Iterative Ablation
* Refined the strategy to target features driving the incorrect competitor token instead of the target token's helpers.
* Successfully tripled the correction rate to 13.64% (3 out of 22 facts).
* Identified the split between factual competitors (easily corrected in 1 round) and grammatical competitors (generic tokens like `" the"` or `" a"` that are supported by multiple redundant syntax features and resist single-feature muting).

## 2026-07-24: Whole-Combination Safety & Weighted Multi-Competitor Reduction
* Implemented **Weighted Multi-Competitor Reduction** to mute all above-target competitor features simultaneously, weighted by their threat level.
* Discovered **Shared-Feature Collapsing** during the Cambridge test case: 7 grammatical competitors collapsed onto only 2 unique features (313 and 21169).
* Formulated the **Distributed Support (Loudspeaker) Hypothesis**: grammatical/functional tokens are supported by many independent features in the network. Muting only the top representative feature is insufficient for highly distributed competitors, pointing to a subspace-based intervention requirement (see [Research Journal Entry 4](file:///d:/Work/PROJECTS/FeatureScalpel/docs/Research_Journal/4.md)).

## 2026-07-25: Diagnostic Investigation of Rank Regression
* Investigated rank regression (target rank dropping on larger combinations despite rising target probabilities) in the Hybrid Mute & Boost tab during the Cambridge MIT case sweep.
* Rejected Option A (implementation bugs/state leaks) via independent clean baseline verification, validating hook manager integrity.
* Confirmed Option B (mathematical consequence of softmax interaction): boosting polysemantic features (e.g. Feature 24181) amplifies competitor tokens faster than target tokens, leading to relative probability suppression. Documented findings in [Research Journal Entry 5](file:///d:/Work/PROJECTS/FeatureScalpel/docs/Research_Journal/5.md).

## 2026-07-25 (Session 2): Shared Feature Aggregation Experiment
* Conducted Experiment H10 (Shared Feature Aggregation) in Weighted Multi-Competitor Reduction.
* Replaced max-aggregation with additive sum-accumulation followed by clamping at the `max_strength` ceiling.
* Rejected Hypothesis H10: Changing aggregation from max() to additive accumulation did not materially improve intervention performance despite substantially increasing Feature 313 mute strength from 0.284 to 0.492 (~73% stronger intervention).
* Concluded that the strongest identified SAE feature alone is insufficient to substantially suppress these competitors. This motivated a new hypothesis (H11) that competitor support is distributed across multiple features (see [Research Journal Entry 6](file:///d:/Work/PROJECTS/FeatureScalpel/docs/Research_Journal/6.md)).

## 2026-07-25 (Session 3): Multi-Feature Representation Diagnostic
* Conducted Experiment H11 (Multi-Feature Representation Diagnostic) to measure the concentration of competitor representation across SAE features.
* Verified that `prob_delta` is a post-softmax probability difference and is therefore not additive for cumulative attribution analysis.
* Measured decay profiles: Factual competitors (e.g. `" question"`, `" doubt"`) exhibit extremely flat decay profiles (Top-2 feature drops are 92.3% - 94.7% of the Top-1 drop), supporting H11. Grammatical competitors (e.g. `" the"`, `" a"`) show steep decay (~33% - 40%), indicating concentration on a single dominant shared feature (Feature 313).
* Confirmed 100% feature overlap: within the discovered Top-10 features, every identified feature was shared by multiple competitors (see [Research Journal Entry 7](file:///d:/Work/PROJECTS/FeatureScalpel/docs/Research_Journal/7.md)).

## 2026-07-30: Towards Monosemanticity Baseline & Monosemanticity Analysis Tab
* Rebuilt **Tab 10 (Towards Monosemanticity)** to faithfully reproduce the 3 fundamental diagnostic measurements from Anthropic (Bricken et al., 2023): Feature Activation Spectrum ($L_0$ norm), Direct Logit Attribution ($W_{\text{dec}} \cdot W_{\text{U}}$), and SAE activation space clamping ($f_i \leftarrow C$).
* Enhanced **Tab 6 (Monosemanticity Analysis)** with plain-language explanation blocks across all four evaluation sections (Max-Activating Examples, Autointerp Interpretability Scoring via Groq, Sparsity Statistics, and Nearest Decoder Directions).
* Added automatic pre-filling of `GROQ_API_KEY` from environment variables, `.env`, or `.streamlit/secrets.toml`.

## 2026-07-31: GPU Acceleration, Decoupled Caching, and HF_TOKEN Authentication
* Implemented automatic PyTorch hardware device selection (`get_default_device()`), supporting CUDA GPU (detected NVIDIA GeForce GTX 1660 Ti), Apple MPS, and CPU fallback for a 10x–20x execution speedup.
* Decoupled base model loading (`load_base_model`) from SAE dictionary loading (`load_sae_for_layer`). Cached the base transformer model once per session in Streamlit (`@st.cache_resource`), reducing layer-switching time from ~10s down to **< 0.2s**.
* Integrated automated `HF_TOKEN` environment loading and `huggingface_hub.login` to eliminate rate-limiting during model and SAE dictionary weight downloads.

## 2026-08-01: Synthetic Corpus Generator, Bundled Corpus Viewer, & Robust Autointerp Parsing
* Added **✨ Synthetic Corpus Generator** (`generate_synthetic_corpus`) in `src/monosemanticity.py`, empowering researchers to synthesize topic-focused benchmark corpora on demand via Groq LLM agents.
* Integrated **👁️ Bundled Corpus Inspector** in Tab 6, rendering an interactive dataframe preview of all 210 corpus sentences across 10 categories.
* Upgraded **Autointerp Parsing Robustness** (`score_feature_interpretability`) with 800-token allocations and 3-stage normalized fuzzy string matching (`_norm(text)`) to eliminate formatting mismatch drops.
* Enhanced **UI Credential Security** by removing on-screen password input widgets, loading secrets silently from environment variables / `.env` in backend memory.

## 2026-08-04: Pretrained SAE Dictionary Audit for gpt2-small-res-jb
* Audited available pretrained SAE dictionaries in SAELens for `gpt2-small-res-jb` using standalone verification script `investigate_gpt2_saes.py`.
* Verified 100% complete coverage for `blocks.{layer}.hook_resid_pre` across all 12 transformer blocks (Layers 0 to 11).
* Mapped external release structure for other hook locations (`gpt2-small-resid-post-jb`, `gpt2-small-mlp-out-jb`, `gpt2-small-attn-out-jb`, `gpt2-small-hook-z-jb`). Documented findings in [Research Journal Entry 9](file:///d:/Work/PROJECTS/FeatureScalpel/docs/Research_Journal/9.md).

## 2026-08-04 (Session 2): Layer Intervention Depth Characterization Benchmark
* Implemented dedicated **Layer Intervention Benchmark** (`layer_benchmark.py` and `src/benchmark/layer_benchmark_runner.py`).
* Rendered high-resolution benchmark visualization plots (Probability Gain vs Layer, Rank Improvement vs Layer, Runtime vs Layer) in `docs/Research_Journal/images/`.
* Created [Research Journal Entry 10](file:///d:/Work/PROJECTS/FeatureScalpel/docs/Research_Journal/10.md) tracking progress across transformer depth.

## 2026-08-05: Multi-Prompt Dataset Benchmark & Research Logging Subsystem
* Extended `layer_benchmark.py` with an interactive **Prompts Dataset Editor** (`➕ Add Prompt`, `➖ Remove Prompt`).
* Added automatic timestamped JSON artifact persistence to `benchmark_results/` per prompt.
* Added programmatic notebook `notebooks/04_layer_intervention_benchmark.ipynb` and session transcript log `History/Session_LayerBenchmark_History.md`.

## 2026-08-09 / 2026-08-10: Hardware Acceleration, Safety Unpacking Fix, and ROME Dataset Integration
* Upgraded PyTorch to CUDA 12.4 (`torch-2.6.0+cu124`) enabling GPU hardware acceleration (NVIDIA GeForce GTX 1660 Ti), achieving 3.83x–20.67x faster execution speed.
* Resolved unpacking signature mismatch in `check_boost_safe` (`is_safe, _, _`) and updated hook call in `src/benchmark/layer_benchmark_runner.py` to `make_mute_and_boost_hook`.
* Standardized 22 factual ROME prompts dataset ([benchmark_test_prompts.json](file:///d:/Work/PROJECTS/FeatureScalpel/benchmark_test_prompts.json)).
* Added hardware compute device indicator (`⚡ GPU` / `💻 CPU`) to Streamlit sidebar and header metadata in `layer_benchmark.py`.
* Created [Research Journal Entry 12](file:///d:/Work/PROJECTS/FeatureScalpel/docs/Research_Journal/12.md) documenting hardware speed analysis and delta patching formalization.

## 2026-09-09T19:49:15+05:30: Fix Cleanup `del` Statement NameError in `layer_benchmark_runner.py`
* Fixed `del` statement in [`src/benchmark/layer_benchmark_runner.py`](file:///d:/Work/PROJECTS/FeatureScalpel/src/benchmark/layer_benchmark_runner.py#L321) by removing unassigned local variables `logits` and `clean_sorted_indices`.
* Eliminates `NameError` exceptions causing bogus duplicate `ERROR` entries (`clean_rank: -1`) in benchmark prompt results.
* Validated standalone execution on Layer 8 (`gpt2-small-res-jb`) confirming single layer output entry (`len == 1`), zero error keys, and wall-clock execution time of 20.33s.

## 2026-08-12: Empirical Execution Cost Audit, Step 2 Optimization, and UI Observability
* Conducted **Step 1 Execution Cost Audit** on Layer 8 (`gpt2-small-res-jb`) identifying 186 model forward passes on GPU.
* Implemented **Step 2 Safety Filter Optimization**, passing precomputed clean baseline probabilities into `check_target_safe()` and `check_boost_safe()` in `src/editing.py`.
* Removed **60 redundant GPU forward passes**, reducing safety check passes from 120 down to 60 (achieving a **1.98× safety filtering speedup**).
* Verified 100% mathematical and scientific equivalence across all target probabilities, ranks, and safety decisions.
* Enhanced `layer_benchmark.py` with live **Execution Cost / Forward-Pass Accounting** expander and **Stage vs Time Breakdown** table.
* Created [Research Journal Entry 13](file:///d:/Projects/transient_steering/docs/Research_Journal/13.md) documenting the complete audit and optimization results.






