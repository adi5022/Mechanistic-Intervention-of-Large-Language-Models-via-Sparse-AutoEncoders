# ROME baseline: progress log

Goal: ROME as a benchmark baseline against Hybrid Mute and Boost on GPT-2 small (CPU), with a shared dataset, metrics and evaluation script.

Branch: rome-factual-editing (from gradient-descent-editing). Environment: .venv-rome (Python 3.11), requirements-rome.lock.txt.

| Step | What was done | Status |
|------|---------------|--------|
| 0 | Vendored official ROME at commit 0874014 into third_party/rome; created skeleton folders; .gitignore; PATCHES.md | done, pushed |
| 1 | Python 3.11 venv; replaced hardcoded cuda with DEVICE (default cpu); removed unused baselines from evaluate.py; added gpt2 choice; smoke test passed (124,439,808 params, Paris top-1 p=0.064) | done, pushed |
| 2a | Causal tracing of one prompt: src/rome_baseline/{rome_env,tracing}.py, scripts/rome/01_trace_one.py | done on 3 prompts; pipeline works; single weak or wrong facts are uninformative, need averaging (2b) |
| 2b | Tracing over about 100 known facts, averaged heatmaps, choose edit layer | done, 97 facts traced; no sharp mid-layer MLP hotspot, see notes |
| 3 | Covariance statistics C at the chosen layer | todo |
| 4 | gpt2.json hparams and a single ROME edit demo | done: gpt2.json hparams, single edit works at layer 5 (Rome p=0.998, paraphrases flip, restore exact); Statue of Liberty p(Rome) rose 0.035 to 0.156 |
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
- Averaged tracing (97 facts, p>=0.2; clean p 0.358, corrupted p 0.034): late site found (last token, attention layers 8-10 about +0.10; residual at last token about +0.32 by layer 11). No sharp mid-layer MLP hotspot at last subject token: MLP effect about +0.01 at layers 0-1, negative at layers 2-3, about 0 after; residual at last subject token flat about +0.025 over layers 0-8.
- Decision: tracing does not pin the edit layer for GPT-2 small; choose it empirically in the Step 5 sweep.
- Known facts: GPT-2 small gets 342 of 1209 known_1000 prompts right; 97 with p>=0.2 were traced (data/comparison/gpt2_knowns.json).
- Covariance C: wikitext-103 (Salesforce/wikitext) instead of wikipedia (script dataset no longer loads); 20,000 samples per layer, float32, CPU; same sample count for all 12 layers; stats cached in third_party/rome/data/stats (gitignored).
- First edit (layer 5, The Eiffel Tower -> Rome): works under transformers 5.x with no extra patches. clamp_norm_factor 3 was binding (delta norm 47.7 = 3 x 15.9). hparams: v_loss_layer 11, kl 0.0625, lr 0.5, 20 steps.
