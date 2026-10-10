# Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders
Mechanistic interpretability research project exploring inference-time activation editing using Sparse Autoencoders (SAEs) to improve factual reasoning in Large Language Models without retraining.


## Status and benchmarks

- Current app: `experiment_app.py` (Streamlit) with five tabs - Hybrid Mute & Boost, Monosemanticity Analysis, Session History & Benchmarks, Sequential vs Batched Proof, and **Batch: Last vs All Tokens**.
- Latest benchmark write-up: [Research Journal Entry 20](docs/Research_Journal/20.md) - worked run, control prompts and a 34-run paired study, with tables and figures. Data: `benchmark_results/candidate_source_study/`.
- **Headline result (GPT-2 small, layer 8, 13 valid prompts):** drawing candidate features from *all prompt positions* brought the target to rank #1 on **10/13** prompts vs **4/13** for *last token only* (9 wins, 4 ties, 0 losses; identical and reproducible across two independent runs). Last-token failed in every case because it ran out of usable features (mean 56 candidates vs 281).
- Limits: one model and layer, hand-picked prompts, success = rank #1 on the next token only, side effects not yet measured (see Entry 20, Sections 7 and 9).
- Reproduce: `python run_batch.py data/candidate_source_batch_spec.json` (about 5 minutes on a CUDA GPU), or use the Batch tab.
- Earlier results: layer benchmarks in `benchmark_results/`, journal entries in `docs/Research_Journal/`.

## Running it

**Windows + NVIDIA GPU** (the machines the project was built on): `python -m venv .venv`, `.venv\Scripts\pip install -r requirements.txt` (note: the `torch==` line there does not match the working environment, torch 2.6.0+cu124), then `.venv\Scripts\streamlit.exe run experiment_app.py`.

**macOS (Apple Silicon)**, uses the Apple GPU through PyTorch's MPS backend, falls back to the CPU:
```bash
bash scripts/setup_mac.sh            # creates .venv, installs mac_requirements.txt, prints the device it will use
bash scripts/get_data_mac.sh         # only for the test tools: downloads CounterFact and builds the hard set (the app's Prototype lab does not need it)
python tools/check_mac_parity.py     # checks this Mac reproduces the saved Windows results (add --full for the 5 additive self-checks)
bash scripts/run_app_mac.sh          # app at http://localhost:8501
```
- The code picks `cuda`, then `mps`, then `cpu`. Force one with `FEATURESCALPEL_DEVICE=cpu|mps|cuda` (use `cpu` if MPS misbehaves).
- Timings are not comparable across machines (CUDA, MPS and CPU differ); compare methods only on one machine.
- The Prototype lab tab (gradient-descent edit, additive edit, baseline-vs-edited text, follow-up prompts) is the current main tab and works on any device. The Batch tab's GPU memory panel is NVIDIA-specific; on a Mac it shows a short unified-memory note and caps background workers at 2.
- Not ported: the GPU-timing benchmark scripts at the repository root (`test_gpu_benchmark.py`, `run_speed_analysis.py`, `layer_benchmark.py`); they measure CUDA hardware specifically.
- Status of the Mac port: written and tested on the CPU path of a Windows PC; **not yet run on a real Mac** (see `docs/handoff_2026-10-05_mac.md`).

## Where the work stands
Read `docs/handoff_2026-10-05_mac.md` first (state, results, planned work), then `docs/research_roadmap.md` and the latest Research Journal entries (28 to 30).

**Cross-model transfer (branch `cross-model-transfer`, from 2026-10-09):** sending the change an edit makes in GPT-2 small into GPT-2 medium (and the reverse) through a translator fitted on ordinary text, at inference time, with no weight changed. Start with `docs/handoff_2026-10-10_cross_model.md`, the runbook `docs/cross_model_transfer/PLAN.md` and Research Journal Entry 31. Done so far: translators (steps A0 to A2), a country-swap control that passes its pre-set gate (B1), and, overnight 2026-10-10 (Research Journal Entry 32), a controlled export of the gradient-descent edit from small to medium: it fails its pre-set gate (top-1 5% and 8%) but moves the target about tenfold in rank, lifts rewordings 15 to 16 points and costs neighbouring facts 4 to 5 points, far more specifically than a plain word push; a neural translator exports no better; the fact-or-word-push test gives read-out steering as its rule is written (weak). Import of real facts is not done. Later on 2026-10-10 (Research Journal Entry 33): adding the translated edit at several layers of medium does not help (Gate 4 fails); an edit tuned directly through the translator against medium's output makes the target medium's top answer on 95% of records (the original edit: 8%), so the channel can carry a rank-1 edit and the mismatch is between what moves small and what moves medium (Gate 5 fails only on the neighbour line, 8 points against 5); a linear translator trained on edits with an output-matching loss (step D7, nothing tuned on medium at test time) roughly doubles top-1 over the ridge map (8% to 18%) but fails its pre-set gate (50%). Related work: `docs/cross_model_transfer/RELATED_WORK.md` (Chen et al., NeurIPS 2025, affine stitching between GPT-2 small and medium and others, no fact edits). Later (Entry 33 sections 4.4 to 4.6): more training edits help unevenly, longer training neither helps nor hurts (a first look that suggested otherwise came from a fragile dose rule), and the single linear map plateaus at about 20% top-1; the paper's skeleton, claims and caveats are in `docs/cross_model_transfer/PAPER_NOTES.md`, the logging rules in `CLAUDE.md`.
