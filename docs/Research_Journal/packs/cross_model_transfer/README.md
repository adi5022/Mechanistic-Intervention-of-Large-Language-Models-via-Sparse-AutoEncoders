# Data pack: cross-model transfer (Journal Entries 31 and 32)

Copies of the result files written by `tools/transfer/*.py` into `outputs/transfer/` (which git ignores). Made 2026-10-10 on the RTX 4050 laptop GPU, torch 2.6.0+cu124, transformer-lens 3.5.1. Not included because of size: the fitted translators (`maps_gpt2_to_gpt2-medium.pt`, 160 MB; rebuild with `01_fit_maps.py`, 138 s) and the per-pair caches of step B1.

| File | Written by | What it holds |
|---|---|---|
| `setup_check.json` | `00_setup_check.py` | step A0: device, model sizes, tokenizer identity, fidelity against Hugging Face, joint GPU run, wikitext-2 token counts |
| `maps_summary.json`, `maps_summary.csv` | `01_fit_maps.py --sgd-check` | step A1: for each of the 50 translators (25 layer pairs x 2 directions) the chosen ridge strength and the held-out per-dimension R-squared, raw R-squared and cosine |
| `maps_sgd_check.json` | `01_fit_maps.py --sgd-check` | gradient descent against the exact solution on one pair (60,000 tokens): 0.6287 against 0.6289 |
| `a2_check.json` | `02_check_maps.py --feature-view` | step A2: stitching losses, difference cosines with chance levels (at the word and at the last position) for 8 layer pairs, the SAE-feature round-trip check, the Gate 1 verdicts |
| `b1_swap_primary.json` | `03_swap_control.py --pairs s2m_L8_L16,m2s_L16_L8 --tag primary` | step B1, primary pairs: for every arm x variant x dose the dev and test success rates, logit gain, KL and where the top answer goes; Gate 2 verdicts; leak onto unrelated prompts |
| `b1_swap_exploratory.json` | `03_swap_control.py --variants all --alphas 1,1.5,2,3,4 --tag exploratory` | step B1, the six other layer pairs (no gate) |
| `d_pilot.json` | `04_export_pilot.py` | step D pilot (30 records, a look and not a result): the edits' properties and, for medium 12 and 16, success of the translated edit and three controls at doses 1 to 4 |
| `d_benchmark.json`, `d_edits.json` | `05_export_edits.py` | step D1/D2 (Entry 32): the 200 benchmark records (50 dev, 150 test) and, per record, the two gradient-descent edits tuned in small (feature ids and a_k) with their ranks and KL |
| `d_eval_linear.json`, `d_eval_mlp.json` | `06_export_eval.py` | the dose chosen on dev and the Gate 3 verdict (as printed by the first run; the random-word arm's rank gain there used the wrong baseline, see Entry 32 section 5) |
| `d_analysis_linear.json`, `d_analysis_mlp.json` | `09_analyse_export.py` | the corrected paired analysis of the controlled export: tuned prompt, dose curves, rewordings, neighbours, unrelated prompts, transfer efficiency, plausible target against random word |
| `mlp_vs_linear.json` | `07_fit_mlp_maps.py` | neural against linear translator: R-squared, stitching, difference cosine at the changed word and the last position |
| `c_fact_or_push.json`, `c_fact_or_push_summary.json` | `08_fact_or_push.py` | Phase C per record and summarised (Entry 32 section 3.6) |

Reading guide: `docs/Research_Journal/31.md` and `32.md` (results and caveats) and `docs/cross_model_transfer/PLAN.md` (runbook, pre-stated gates). All numbers are exploratory proxies; B1 test cases share countries and are not independent; the pilot has no train/test split.
