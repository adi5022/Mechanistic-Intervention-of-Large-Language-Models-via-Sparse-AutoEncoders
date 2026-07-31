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

