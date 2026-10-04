# Generation-quality / context-sensitivity test, true targets (Entry 30 section 15)

Produced by `tools/test_generation_quality.py --n-per-band 10` on 2026-10-04 (GTX 1660 Ti, about 22 minutes). 40 held-out test prompts (10 per starting-rank band), true CounterFact targets, arms `gd`, `gd_add@cap=1.0`, `gd_add@cap=0.25`, 20 generated tokens, greedy decoding.

- `summary.txt` / `summary.json`: the two result tables (A: sequential collapse; B: context sensitivity).
- `runs.jsonl`: one row per prompt, with the generated texts and per-step data.
- `meta.json`: settings.

Not included: a random-word control and a rank-matched control (not run). No confidence intervals (n = 40).
