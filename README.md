# Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders
Mechanistic interpretability research project exploring inference-time activation editing using Sparse Autoencoders (SAEs) to improve factual reasoning in Large Language Models without retraining.


## Status and benchmarks

- Current app: `experiment_app.py` (Streamlit) with five tabs - Hybrid Mute & Boost, Monosemanticity Analysis, Session History & Benchmarks, Sequential vs Batched Proof, and **Batch: Last vs All Tokens**.
- Latest benchmark write-up: [Research Journal Entry 20](docs/Research_Journal/20.md) - worked run, control prompts and a 34-run paired study, with tables and figures. Data: `benchmark_results/candidate_source_study/`.
- **Headline result (GPT-2 small, layer 8, 13 valid prompts):** drawing candidate features from *all prompt positions* brought the target to rank #1 on **10/13** prompts vs **4/13** for *last token only* (9 wins, 4 ties, 0 losses; identical and reproducible across two independent runs). Last-token failed in every case because it ran out of usable features (mean 56 candidates vs 281).
- Limits: one model and layer, hand-picked prompts, success = rank #1 on the next token only, side effects not yet measured (see Entry 20, Sections 7 and 9).
- Reproduce: `python run_batch.py data/candidate_source_batch_spec.json` (about 5 minutes on a CUDA GPU), or use the Batch tab.
- Earlier results: layer benchmarks in `benchmark_results/`, journal entries in `docs/Research_Journal/`.
