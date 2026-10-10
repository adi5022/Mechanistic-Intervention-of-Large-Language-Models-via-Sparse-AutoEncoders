# Data pack: cross-model transfer (Journal Entries 31, 32 and 33)

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
| `d5_multilayer.json` | `10_multilayer_export.py` | step D5: dose chosen per configuration on dev, Gate 4 verdict, stage 2 summary and the proposed rerun bar for adding the translated edit at one or several layers of medium |
| `d5_configs.json` | `13_pack_d5_table.py` | step D5 per configuration: dose chosen on dev, dev top-1, test top-1 and median rank at that dose and at the norm-matched dose, and the two controls at the same dose (for Entry 33 Figure 1) |
| `d6_b_aware.json` | `11_b_aware_edit.py` | step D6 (ceiling test): edit tuned through the translator against medium's output; per-arm top-1 and rank, rewordings / neighbours / unrelated summary, Gate 5 verdict |
| `d7_output_matching.json` | `12_output_matching.py` | step D7: the map trained on edits against the ridge map and controls (top-1, median rank, rewordings, neighbours, unrelated), Gate 6 verdict, training-set size |
| `d8_datasize.json` | `14_datasize_check.py` | step D8: the map trained on 25%, 50%, 100% of the 289 edits, 3 seeds each: per run epoch, dose, primary and all-record top-1, gain, median rank (lower of the two middle values for even counts, unlike the other files), the decision-rule verdict |
| `d9_longer_training.json` | `14_datasize_check.py --fractions 1.0 --epochs 40 --longer --tag longer` | step D9: the 40-epoch runs (3 seeds), dose chosen on dev, the comparison with the 15-epoch runs, the verdict of the pre-set rule |
| `d9b_dose_table.json` | `15_dose_table.py` | step D9b (post-hoc): for 3 seeds and 15 and 40 epochs, the dev and test results at every dose and the three dose rules; the tool now also saves the epoch curves |
| `logs/` | the tools' terminal output | raw terminal output of the runs (B1, D2, D3, neural translator, Phase C, D5, D6, D7, D8, D9, D9b); the earliest ones were copied from temporary files, D5, D7, D9, D9b are the text pasted by the author (warnings and progress bars removed), D8 was reconstructed from the terminal scrollback. From D9b on the tools write their own logs to `outputs/transfer/logs/` |
| `d10_rank1_term.json` | `16_rank1_term.py` | step D10: for 6 lambda values x 3 seeds the epoch, dose, per-epoch curves (including the training shortfall from rank 1), per-dose tables on dev and test, controls; the lambda-star choice; the judging of lambda-star and lambda 0 (rewordings, neighbours, unrelated prompts, controls); Gate 7. Log: `logs/d10_rank1_term.log` (written by the tool itself) |
| `mlp_vs_linear.json` | `07_fit_mlp_maps.py` | neural against linear translator: R-squared, stitching, difference cosine at the changed word and the last position |
| `c_fact_or_push.json`, `c_fact_or_push_summary.json` | `08_fact_or_push.py` | Phase C per record and summarised (Entry 32 section 3.6) |

Reading guide: `docs/Research_Journal/31.md`, `32.md` and `33.md` (results and caveats) and `docs/cross_model_transfer/PLAN.md` (runbook, pre-stated gates). All numbers are exploratory proxies; B1 test cases share countries and are not independent; the pilot has no train/test split.
