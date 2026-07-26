# Research Paper Outline: Transient Activation Steering with Sparse Autoencoders for Factual Correction

## Tentative Paper Structure

1. **Abstract**
   - Summary of problem: permanently editing weights is risky; activation steering is safer but blunt.
   - Core method: causal feature selector & multi-competitor ablation.
   - Core finding: 13.64% factual correction success rate, grammatical collapse / distributed support bottleneck.

2. **Introduction**
   - The reliability and alignment challenges of Large Language Models.
   - The trade-offs between weight editing (ROME/MEMIT) and activation steering (representation engineering).
   - Our focus: transient, zero-shot factual editing in base models.

3. **Related Work**
   - Locating and Editing Factual Associations (ROME/MEMIT).
   - Mechanistic Interpretability & Superposition (Anthropic).
   - Sparse Autoencoders for Monosemanticity (Cunningham et al.).
   - Activation Steering Benchmarks (AxBench).

4. **Methodology**
   - SAE Representation and Hooking Mechanics.
   - Causal Feature Selector formulation.
   - Competitor Muting vs. Target Boosting.
   - Weighted Multi-Competitor Reduction (Mathematical formulation).
   - Joint Safety Verification and Blocker Detection.

5. **Experiments**
   - Evaluation on 22 suppressed facts from ROME's `known_1000`.
   - Single-feature target vs. competitor soft-ablation.
   - Joint safety combinations and blocker tracing.
   - Case Study: The Cambridge prompt benchmark.

6. **Results & Discussion**
   - Triple success rate under competitor muting (13.64%).
   - The Shared-Feature grammatical collapse observation (Feature 313).
   - Validation of the Distributed Support (Loudspeaker) Hypothesis.

7. **Limitations & Future Work**
   - Syntactic stopword resilience to single-feature interventions.
   - Scaling to instruction-tuned and larger models.
   - Moving from 1-to-1 mappings to subspace-based multi-feature interventions.

8. **Conclusion**
   - Summary of findings and contributions.
