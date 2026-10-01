# Handoff (written 2026-09-30): repair and SAE-limit diagnostics session

Read this first when resuming on another machine. Branch: `pool-refill-implementation`. Repo: `Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders` (folder name FeatureScalpel).

Older handoffs, for background: `research/paper/draft_v1/HANDOFF.md` (2026-09-26/27, paper draft and Batch tab, has the working-style lessons) and `docs/handoff_2026-09-24.md` (the previous root handoff, moved here).

## The project in one paragraph

Transient activation steering in GPT-2 small: edit SAE features (`gpt2-small-res-jb`, mostly `blocks.8.hook_resid_pre`) during one forward pass to raise a suppressed target token to rank 1. The edit applies only the *delta* of the SAE reconstruction (`x' = x + sum (s_i - 1) f_i w_i`), so the SAE reconstruction error is never injected. The main app is `experiment_app.py` (Streamlit); headless engine in `src/hybrid_runner.py`, `src/batch_runner.py`.

## The question that started this session

The user asked what the "actual issue" is, weighing three worries:
1. Editing only layer 8 may let later layers bring the ablated fact back ("cross-layer superposition").
2. Using SAEs at all for this is like driving a Lamborghini to buy groceries.
3. SAE reconstruction error is passed down and never corrected.

After reading the docs and code: #3's premise is wrong (delta patching already avoids injecting the error; Journal 1, 12, 18, `main.tex`), but the error is left *unedited*, which is a reachability limit. #1 and #2 were untested, so we built diagnostics and ran them.

## What was done this session

1. **`src/repair_diagnostics.py`** (new). For one prompt and layer: runs the normal Hybrid sweep, converts the best edit into a residual delta, then
   - **traces** the delta through every later block (persistence = `<diff,d>/<d,d>`, plus a logit lens, clean vs edited),
   - runs a **held edit** control (re-adds whatever fraction of the push later blocks removed),
   - computes an **SAE-free bound**: gradient-optimised residual delta with the same L2 norm as the SAE edit, and a scan at 2, 5, 10, 20, 40% of the residual norm, for last-token-only and all-non-BOS positions,
   - records SAE reconstruction error relative to the activation norm.
2. **`tools/run_repair_diagnostics.py`** (new): headless driver. Args `--layers`, `--n-prompts`, `--steps`, `--out`. Prompts: Colosseum to Rome, door to cat, doctor to she, MIT to Cambridge.
3. **`experiment_app.py`**: new 6th tab **"🔬 Repair & SAE limit"** (`tab13`, appended at the end of the file). Verdict cards in plain language, summary table, rank-by-layer chart, persistence curves, logit-lens view for a chosen edit layer, push-size heatmap, clean-model lens bars, and a "Run a new diagnostic" panel. It reads every `outputs/repair_diagnostics/*.json`. Checked in the preview with layer-8 data (no console errors); not reopened with the merged file.
4. **Ran the diagnostics** (on the user's GTX 1660 Ti, about 2.5 to 3 min per prompt-layer): layer 8 for 4 prompts (60 optimiser steps), layers 5, 6, 7, 9, 10, 11 for the 3 failing prompts (40 steps). Layer 4 was not run.
5. **Docs updated:** Journal Entry 22, `research/hypothesis_log.md` (H13, H14), `research/experiment_index.md` (EXP-012), `research/timeline.md` (2026-09-30), this handoff.

## Findings (details and full tables in `docs/Research_Journal/22.md`)

- **Repair (#1): not supported.** Held edit == plain edit in 21 of 21 layer/prompt cells. Push persisted at 0.79 to 1.29 of injected size at the output. Caveat: this tests the pushed direction, not regeneration along other directions.
- **SAE limit (#2): suggestive only.** On 3 prompts that had already failed, a gradient-searched residual edit of the same L2 size reached rank 1 in 21 of 21 layer/prompt combinations and the SAE edit in 1 of 21. These are hand-picked failures, so this is not a success rate and is not comparable to the 41/131 (last token) or 29/45 (all positions, pilot) figures. No baseline has been run on the 131-prompt set. The user's own project setup (mute 0.6, boost 0.5 fixed, Top-N 200, all positions, cumulative sweep, safety filter on) has not been compared with other methods yet.
- **Reconstruction error (#3):** 12 to 26% of activation norm, left unedited; not sufficient alone to explain failures, but it bounds what feature scaling can reach.
- MIT to Cambridge at layer 8 reached rank 1 with 22 features here (Entry 10 had rank 3 with older settings; not compared).

## Where the results are (all committed to git)

| What | Path |
|---|---|
| Journal entry with tables, methods, limits | `docs/Research_Journal/22.md` |
| Full result data (7 layers x 3 prompts + MIT at layer 8; traces, held, SAE-free scans) | `docs/Research_Journal/packs/repair_diagnostics/merged_all_layers.json` |
| Run log | `docs/Research_Journal/packs/repair_diagnostics/other_layers.log` |
| Pack notes | `docs/Research_Journal/packs/repair_diagnostics/README.md` |
| Code | `src/repair_diagnostics.py`, `tools/run_repair_diagnostics.py` |
| App tab | `experiment_app.py` (search `TAB 13`) |
| Hypotheses / index / timeline | `research/hypothesis_log.md` (H13, H14), `research/experiment_index.md` (EXP-012), `research/timeline.md` |

The raw originals lived in `outputs/repair_diagnostics/` (not committed; `outputs/` stays local). To view the result on a new machine: `mkdir outputs\repair_diagnostics`, copy `merged_all_layers.json` into it, run `streamlit run experiment_app.py`, open the last tab.

## State of the working tree when this was written

- Committed by this session: the files listed above plus the moved `docs/handoff_2026-09-24.md`.
- **Not committed on purpose:** `research/paper/draft_v1/main.tex` (modified before this session started; someone else's in-progress edit, review it before committing) and everything under `outputs/`.

## Open items, in priority order

1. **Side effects of the SAE-free edit.** Measure KL on `NEUTRAL_PROMPTS` (in `src/hybrid_runner.py`) for the optimised delta. Without it, "SAE-free wins" is only an upper bound.
2. **Non-SAE baseline** on the 131-prompt set (contrastive steering vector or ROME-style edit at layers 7 to 8). The paper draft says no baseline comparison has been done.
3. **Does the ceiling come from the error term?** Check whether the SAE-free direction lies outside the span of the SAE decoder.
4. **Only then** revisit a multi-layer intervention. Repair was not observed, so it is no longer motivated by repair.
5. **Paper:** add this study to `research/paper/draft_v1/main.tex` and `EVIDENCE_AUDIT.md` (not done; Entry 22 is the source). Older open items (compile in Overleaf, the questions in `RECONSTRUCTION.md` section 8, missing sources, Entry 21 for the filter study) are in `research/paper/draft_v1/HANDOFF.md`.
6. **Known weak spots to fix in code** (unchanged from the older handoff): blocker fixed per round, Groq explanation overwritten `probs`, multi-token targets scored on the last piece, `iterative_ablate` and `check_specificity` hard-code layer 8, no automated tests for editing or safety code. New: `src/repair_diagnostics.py` has no tests, and its `optimal_delta` uses a fixed step count (sensitivity untested).

## How to work here (from the user; still applies)

- **The user runs long experiments themselves** when they can; this session they asked directly for runs, so we ran them. Disclose anything you launch.
- **Plain language first**, explain where numbers come from, report null results honestly, no unsupported claims. Paper style: no em dashes, no hype, no novelty claims.
- Ask before commit and push. Never commit `outputs/`.
- Python: `.venv\Scripts\python.exe` (torch 2.6.0+cu124). The default `python` has no torch. GPU is a GTX 1660 Ti (6 GB), shared.
- Shell gotchas on Windows: bash heredocs and regex backslashes get mangled (write files with a file tool or a script file); files are CRLF, read with `newline=''` and `encoding='utf8'` (the app source has emoji, and a default-cp1252 read fails).
- TransformerLens: hooks must accept the keyword `hook` (`lambda r, hook: ...`), and `run_with_cache` does not take `fwd_hooks` (use `with model.hooks(fwd_hooks=[...]):`).
- The Streamlit preview needs about 60 to 90 s to load the model before tabs render.
