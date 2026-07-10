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
