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

**Cross-model transfer (branch `cross-model-transfer`, from 2026-10-09):** sending the change an edit makes in GPT-2 small into GPT-2 medium (and the reverse) through a translator fitted on ordinary text, at inference time, with no weight changed. Start with `docs/handoff_2026-10-10_cross_model.md`, the runbook `docs/cross_model_transfer/PLAN.md` and Research Journal Entry 31. Done so far: translators (steps A0 to A2), a country-swap control that passes its pre-set gate (B1), and a 30-record pilot of a real edit (not a result); nothing about facts has been shown yet.
