# FeatureScalpel: Mechanistic Interpretability and Intervention of Large Language Models via Sparse Autoencoders

[![CI](https://github.com/adi5022/Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders/actions/workflows/ci.yml/badge.svg)](https://github.com/adi5022/Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders/actions/workflows/ci.yml)

## Abstract

Language models often finish a factual prompt with a generic word even though the right answer is only a few places down the ranking. The usual fixes have drawbacks: fine-tuning and closed-form weight edits (such as ROME and MEMIT) are permanent and degrade as edits accumulate, whereas retrieval and in-context editing leave weights untouched but supply the fact as additional input context.

**FeatureScalpel** investigates a transient, weight-free alternative. During a single forward pass, the internal state of GPT-2 Small is steered through the features of a pretrained Sparse Autoencoder (SAE) at layer 8 residual stream, and the edit is discarded as soon as the forward pass completes. Across 300 held-out CounterFact prompts, gradient-descent optimization of feature multipliers achieved rank 1 on 271 (90.3%) prompts compared to 161 for a wide sweep and 94 for original settings, with 5 to 6.5 times less compute. In parallel, cross-model translation investigations study transferring activation-steering edits from GPT-2 Small to GPT-2 Medium.

---

## Team Members

Department of Computer Science and Engineering  
**Mar Baselios College of Engineering and Technology (Autonomous), Thiruvananthapuram**  
Course: **23CTB40B Software Engineering** | **Assignment 2: Main Project on GitHub**

| Name | Roll Number | KTU Reg. No. | GitHub Account | Assigned Work |
|---|---|---|---|---|
| **Adithyan R S** | B23CS2109 | MBT23CA009 | [@adi5022](https://github.com/adi5022) | SAE intervention engine, causal ranking, algorithm reliability |
| **Alona Biju** | B23CS2113 | MBT23CA013 | [@Alonabiju](https://github.com/Alonabiju) | Evaluation metrics, CounterFact benchmark curation, reproducibility |
| **Alvin Binu** | B23CS2114 | MBT23CA014 | [@Alvinbinu](https://github.com/Alvinbinu) | Streamlit dashboard validation, PyTest unit test suite, CI/CD pipeline |
| **Darsan M** | B23CS2125 | MBT23CA025 | [@DarsanM](https://github.com/DarsanM) | Workflow integration, cross-model transfer benchmarking, docs |

---

## Technology Stack

- **Core Frameworks**: Python 3.11 / 3.12, PyTorch (CUDA / MPS / CPU)
- **Mechanistic Interpretability**: TransformerLens 3.5.1, SAELens 6.45.3 (`gpt2-small-res-jb`)
- **Web Application & UI**: Streamlit 1.59.1, Altair, Starlette, Uvicorn, React / Node.js
- **Quality Assurance & Testing**: PyTest 9.2 (CPU-deterministic unit test suite with BVA & EP)
- **CI/CD & DevOps**: GitHub Actions (`.github/workflows/ci.yml`), Multi-Python Matrix Testing (3.11, 3.12)
- **External References**: ROME (Rank-One Model Editing), Neuronpedia API

---

## System Architecture

FeatureScalpel is organized into a modular three-tier architecture:

1. **Presentation Tier (`experiment_app.py`, `transfer_demo/`)**:
   - 7 Streamlit dashboard tabs: *Hybrid Mute & Boost*, *Prototype Lab*, *Batch: Last vs All Tokens*, *Repair & SAE Limit Diagnostics*, *Sequential vs Batched Proof*, *Monosemanticity Analysis*, and *Session History*.
   - Real-time rank progression charts, token probability odds indicators, and XAI educational guidance cards.
2. **Intervention Engine Tier (`src/`)**:
   - `src/hooks.py`: Single and joint PyTorch residual stream delta-patching hooks ($\Delta_p = \sum_k (s_k - 1) f_{p,k} W_{dec}[k]$).
   - `src/editing.py`: Causal feature ranking, batched GPU candidate ablation, strict safety filter ($safe(k) \iff r_k(t) \le r_0(t) \land p_k(t) - p_0(t) \ge -10^{-6}$).
   - `src/gradient_editing.py`: Per-feature continuous multiplier tuning ($m_k \in (0, 3)$) via Adam optimization and optional silent-feature additive edits.
   - `src/hybrid_runner.py`: Cumulative sweep orchestration, best-so-far tracking, and dynamic pool refill.
3. **Model & Data Tier (`data/`, `packs/`, `third_party/`)**:
   - TransformerLens `HookedTransformer` (`gpt2-small`, `gpt2-medium`).
   - SAELens layer-8 sparse autoencoders (24,576 latent directions).
   - Benchmarks: CounterFact hard set, ROME known facts, cached prompt states.

---

## How to Run

### 1. Local Application Setup
```bash
# Clone the repository
git clone https://github.com/adi5022/Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders.git
cd Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders

# Switch to the assignment integration branch
git checkout se_assignment

# Set up virtual environment
python -m venv .venv
.\.venv\Scripts\activate          # On Windows
# source .venv/bin/activate       # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit Experiment Dashboard
streamlit run experiment_app.py
```

### 2. Running Unit Tests
The test suite is located in `tests/test_assignment_a2.py`. All tests run on CPU in seconds without requiring GPU hardware or model downloads:
```bash
pytest -v tests/test_assignment_a2.py
```

---

## Research Benchmarks & Status

- **Candidate Source Headline Result**: Drawing candidate features from *all prompt positions* brought the target token to rank #1 on **10/13** prompts vs **4/13** for *last token only* (9 wins, 4 ties, 0 losses; identical across independent runs).
- **Gradient Descent vs Sweep**: On 300 held-out CounterFact prompts, gradient descent reached rank #1 on **271/300 (90.3%)**, compared to 161 for equal-range sweep and 94 for original settings, with lower same-prompt side effects (KL) and 5–6.5x faster runtime.
- **Cross-Model Transfer**: Linear ridge translators between GPT-2 Small (layer 8) and Medium (layer 16) demonstrate that inference-time steering transfers as a directional push, verified with country-swap positive controls and negative controls.
