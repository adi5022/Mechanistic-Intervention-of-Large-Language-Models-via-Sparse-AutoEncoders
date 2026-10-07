# ROME baseline: progress log

Goal: ROME as a benchmark baseline against Hybrid Mute and Boost on GPT-2 small (CPU), with a shared dataset, metrics and evaluation script.

Branch: rome-factual-editing (from gradient-descent-editing). Environment: .venv-rome (Python 3.11), requirements-rome.lock.txt.

| Step | What was done | Status |
|------|---------------|--------|
| 0 | Vendored official ROME at commit 0874014 into third_party/rome; created skeleton folders; .gitignore; PATCHES.md | done, pushed |
| 1 | Python 3.11 venv; replaced hardcoded cuda with DEVICE (default cpu); removed unused baselines from evaluate.py; added gpt2 choice; smoke test passed (124,439,808 params, Paris top-1 p=0.064) | done, pushed |
| 2a | Causal tracing of one prompt: src/rome_baseline/{rome_env,tracing}.py, scripts/rome/01_trace_one.py | done on 3 prompts; pipeline works; single weak or wrong facts are uninformative, need averaging (2b) |
| 2b | Tracing over about 100 known facts, averaged heatmaps, choose edit layer | todo |
| 3 | Covariance statistics C at the chosen layer | todo |
| 4 | gpt2.json hparams and a single ROME edit demo | todo |
| 5 | Edit-layer sweep on dev split | todo |
| 6 | Frozen benchmark, shared metrics, adapters | todo |
| 7 | Adapter for Hybrid Mute and Boost, full comparison run | todo |
| 8 | Fluency, results table, plots, trade-off write-up, merge | todo |

## Notes and decisions
- zsh does not split unquoted variables and does not treat pasted comments as comments, so command blocks avoid both.
- transformers 5.x is in use, newer than ROME's original pinned version; compatibility fixes go in PATCHES.md.
- ROME float64 statistics require CPU (MPS has no float64).
- Single-prompt tracing: Eiffel shows only a faint early MLP bump at the last subject token (layers 1-3, about 0.008 vs floor 0.005) and a strong late site at the last token (layers 8-10). Space Needle (model answers T, p=0.014) and Windows Media Player (answers Sony, p=0.078) are unknown or wrong facts, so no signal.
- Noise level is about 0.39 (3 x embedding std 0.13).
