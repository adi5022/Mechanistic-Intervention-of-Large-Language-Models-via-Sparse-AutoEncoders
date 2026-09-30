# Repair and SAE-limit diagnostics: result pack

Journal entry: `docs/Research_Journal/22.md`. Code: `src/repair_diagnostics.py`, `tools/run_repair_diagnostics.py`.

- `merged_all_layers.json`: one object with `prompts`, each holding `baseline_rank`, `clean_lens` (target logit margin at every block) and `layers` (layers 5 to 11 for Colosseum, door and doctor; layer 8 only for MIT). Each layer entry has `sae` (rank before and after, features used, relative push, reconstruction error), `trace.rows` (persistence, lens rank and margin per block, clean vs edited), `held` (rank and prob of the held edit) and `optimal` (`matched` and `scan` results for the SAE-free bound, `last` and `all` positions).
- `other_layers.log`: console log of the layers 5, 6, 7, 9, 10, 11 run.

Two runs were merged: layer 8 for 4 prompts (60 optimiser steps) and the other six layers for 3 prompts (40 steps).

To view in the app, copy `merged_all_layers.json` to `outputs/repair_diagnostics/` and open the "Repair & SAE limit" tab.
