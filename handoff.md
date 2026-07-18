# FeatureScalpel — Project Handoff Documentation

This document provides a comprehensive transfer of the **FeatureScalpel** project, detailing what has been built, the underlying architecture, how the code is structured, and how to run the interactive dashboard.

---

## 1. Project Overview & Objective

**FeatureScalpel** is an MVP prototype for investigating and performing **Mechanistic Activation Interventions** on Large Language Models via Sparse Autoencoders (SAEs) without altering the model's weights. 

Unlike traditional methods (such as ROME or MEMIT) that edit model weights permanently, FeatureScalpel performs transient, single-inference interventions by:
1. Decomposing the model's residual stream into interpretable concepts using a trained Sparse Autoencoder (SAE).
2. Identifying the specific causal feature responsible for a given prediction or factual error.
3. Editing the activation of that feature in the residual stream (ablation or steering) in a targeted, temporary manner.

---

## 2. Core Architecture & Mathematical Mechanics

### 2.1 The Edit Site: Residual Stream
The intervention target is the **residual stream** at Layer 8 (`blocks.8.hook_resid_pre`) of GPT-2 Small. The residual stream serves as the model's shared additive workspace. We intercept the activation vector $x \in \mathbb{R}^{d_{model}}$ during the forward pass.

### 2.2 Sparse Autoencoder (SAE) Projection
To make the dense activation vector interpretable, we project it into a sparse, high-dimensional latent space:

$$f = \text{ReLU}(W_{enc} \cdot x + b_{enc})$$

where $f \in \mathbb{R}^{d_{sae}}$ ($d_{sae} = 24,576$ for GPT-2 Small) is a sparse vector where only a few features are active (non-zero). Each column of the decoder weight matrix $W_{dec} \in \mathbb{R}^{d_{model} \times d_{sae}}$ represents a direction in the residual stream corresponding to a human-interpretable concept.

### 2.3 Delta-Patching Ablation Hook
To prevent injecting the SAE's reconstruction error back into the model (which degrades baseline generation quality), FeatureScalpel uses a **delta-patching** intervention hook:
1. Encode the original activation: $f_{base} = \text{encode}(x)$
2. Compute the baseline reconstruction: $\hat{x}_{base} = \text{decode}(f_{base})$
3. Construct the ablated feature activations by scaling the target feature $i$ by strength $\theta \in [0, 1]$:
   $$f_{ablated}[i] = f_{base}[i] \times (1 - \theta)$$
4. Compute the ablated reconstruction: $\hat{x}_{ablated} = \text{decode}(f_{ablated})$
5. Compute the delta update and apply it to the original residual stream:
   $$\Delta x = \hat{x}_{ablated} - \hat{x}_{base}$$
   $$x_{new} = x + \Delta x$$

This delta-patching method ensures we only edit the component of interest, leaving the rest of the residual stream untouched.

---

## 3. Key Components & Implementation Details

The codebase is modularized as follows:

```
FeatureScalpel/
│
├── app.py                      # Interactive Streamlit Web Dashboard
├── requirements.txt            # Project Dependency Specifications
├── notes.md                    # Research log & Empirical verification notes
│
├── src/
│   ├── __init__.py
│   ├── sae_utils.py            # Model loading utilities (TransformerLens + SAELens)
│   ├── hooks.py                # Activation intervention hooks (Delta-patching)
│   └── editing.py              # Causal Feature Selector & Ranking Engine
│
└── docs/
    └── architecture.md         # Design philosophy & comparison analysis
```

### 3.1 Causal Feature Selector (`src/editing.py`)
This module implements the **zero-shot causal discovery engine**. Given a prompt and target token (e.g. `"The Eiffel Tower is in the city of"` and `" Paris"`):
1. **Activation Extraction**: Extracts residual stream activations at the final token.
2. **Feature Encoder**: Project activations to find the top $N$ (default: 20) active SAE features.
3. **Iterative Intervention**: Runs a forward pass for each candidate feature with its activation fully ablated ($\theta = 1.0$).
4. **Metric Logging**: Records the target token probability drop ($\Delta P$), the new Top-1 token prediction, and the Kullback-Leibler (KL) divergence of the resulting vocabulary distribution.
5. **Ranking**: Sorts candidates in ascending order of $\Delta P$ (highlighting features whose ablation causes the largest drop in the target token's probability).

### 3.2 Interactive Web Dashboard (`app.py`)
Built with Streamlit, the frontend provides:
* **Interactive Selector**: Runs the causal feature selector live and displays results in a tabular and visual bar-chart format.
* **Live Soft Ablation Control**: An interactive slider ($\theta \in [0, 1]$) with live feedback showing how the top predicted tokens shift.
* **Specificity Check Panel**: Evaluates the intervention's side effects against a suite of target and control prompts to identify the "sweet spot" (e.g. fixing Eiffel Tower $\rightarrow$ Paris without breaking Colosseum $\rightarrow$ Rome).

---

## 4. Verification & Findings

During empirical verification of the target prompt `"The Eiffel Tower is in the city of"` (Target: `" Paris"`):
* **Causal Feature 11149** was automatically discovered as the principal driver (activation: `30.13`, $\Delta P$: `-0.0337`).
* A highly active feature (e.g. **5856**, activation `34.10`) was shown to be non-causal ($\Delta P$ of only `-0.0073`), proving the efficacy of causal validation over simple activation heuristics.
* **Ablation Sweet Spot**: Setting the slider to $\theta = 0.3$ successfully corrected the Eiffel Tower prediction while preserving the control case ("Rome" for Colosseum). At higher strengths ($\theta \ge 0.5$), side-effects were detected on control prompts.

---

## 5. Getting Started & Setup

### 5.1 Installation
Ensure you are using the virtual environment and install the pinned dependencies:

```bash
# Activate your virtual environment
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 5.2 Running the Application
Launch the Streamlit web server:

```bash
streamlit run app.py
```
