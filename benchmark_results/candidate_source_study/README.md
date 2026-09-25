# Candidate-source study data (Journal Entry 20)

Main study (17 prompts x {all prompt positions, last token only}, layer 8, mute 0.6 / boost 0.5 / Top N 120):

- `batch_result_17prompts_layer8.json` - full result of the 34-run batch: every run's on-screen items (baseline top-5, per-round pools and candidate scores, every sweep step with top-5 and safety check, rank progression, rejections, timings), plus `paired_summary` and `aggregate`.
- `batch_paired_summary.csv` - one row per prompt: baseline vs both modes (final rank, success, probability, steps, rounds, features used, round-0 candidates, time, winner).
- `batch_spec_17prompts.json` - the exact input spec (also at `data/candidate_source_batch_spec.json`). Reproduce with `python run_batch.py data/candidate_source_batch_spec.json`.

Earlier manual runs:

- `control_runs_session_full_details.json` / `control_runs_session_summary.csv` - 6 control runs (3 prompts x 2 modes), all already rank #1.
- `sophia_run_progression.csv` - per-step rank and probability for "This is Sophia, she is a" -> " woman" (all positions, layer 8). Rebuilt from that run's session export (the original file was overwritten); values rounded to the app's 2 decimals.
