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
