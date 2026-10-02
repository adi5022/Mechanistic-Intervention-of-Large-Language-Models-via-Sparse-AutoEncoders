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

## UPDATE 2026-10-01 (read this part first)

Decisions made by the user after the diagnostics (they take the decisions; Claude documents and executes):
1. **Reference run** = the user's original system on the 131 prompts: strict safety filter, mute 0.6 / boost 0.5, Top-N 200, all positions, cumulative sweep, pool refill, 250-step cap, layer 8. Spec: `data/reference_strict_topn200_all.json`. Run it from the Batch tab or `run_batch.py`. NOT run yet. It can be run on another machine (needs the environment, not extra VRAM; speed is limited by Python/kernel launch, not GPU memory) and the result file copied back.
2. **Fair comparison planned** on the same 131 prompts, each method timed: user system, IKE (in-context, few-shot demonstrations, since GPT-2 is not instruction tuned), ROME, DiffMean. Use EXISTING implementations (EasyEdit has ROME/MEMIT/IKE for GPT-2; AxBench has DiffMean/ReFT-r1), not code written from scratch. Columns: time, rank-1 count (of 131), interpretability/monosemanticity (a plain 'what can a person inspect' column plus the existing monosemanticity score for the SAE features). All results then go into one new comparison tab. Not started.
3. **Learned mute/boost strengths** (per-prompt first, per-feature later) is a planned variant of the Hybrid system, trained by gradient through the model on the feature sets recorded in the reference run (no searched labels). Full plan and risks: `docs/Research_Journal/23.md` (H15, EXP-013). Not implemented.
4. The Repair & SAE-limit tab is a diagnostic for failed prompts, not the benchmark. The wording in Entry 22 was corrected: its counts (1 of 21, 21 of 21) are on 3 hand-picked failed prompts and are not success rates.
5. Earlier success figures for the 131-prompt study are last token only, Top-N 120 (41/131) and the 45-prompt all-positions pilot (29/45); neither equals the reference setup.

6. **FULL STUDY PLAN written:** `docs/Research_Journal/23.md` (data from CounterFact `datasets/counterfact.json`, 45 MB, downloaded, gitignored; per-prompt cache; strength-grid filter tables; PromptNet / learned fixed pair / oracle / FeatureNet; training losses; arms and metrics; time budget; risks; open decisions). Measured costs: `tools/measure_plan_costs.py` (cache build 0.56 to 1.17 s per prompt; training step about 50 ms from layer 8, the same for a batch of 32). Nothing in the plan is built yet; the next step is the yield report from `tools/build_counterfact_set.py` (not written).

7. **Step 1 of the learned-strength study is written:** `tools/build_counterfact_set.py` (batched, self-checking). Run: `.venv\Scripts\python.exe tools/build_counterfact_set.py`. Smoke test (500 records): 373 kept, 0 batched-vs-single rank mismatches, 500 prompts in 0.7 s. Full run pending (user runs it). Outputs: `data/counterfact_hard_set.json`, `data/counterfact_hard_set_summary.json`, `outputs/counterfact_all_ranks.json`. Next: split script (Entry 23, section 13).

8. **Steps 2 and 3 scripts written (2026-10-02):** `tools/split_counterfact_set.py` (dry-run checked; run it without --dry-run to write `data/counterfact_split.json`, then commit that file) and `tools/build_strength_cache.py` with `src/strength_cache.py` (tested on 25 prompts: 0.60 s each; two-machine use via `--shard 0/2` and `--shard 1/2`; see Entry 23 section 13.2). The CounterFact yield was 16,360 kept of 21,919 (`data/counterfact_hard_set_summary.json`). Decided: train on the first 1,000 of a balanced nested order (extend later if the learning curve rises), split between the user's GTX 1660 Ti machine and an RTX 4050 machine. Caught and fixed a bug: the batched API's `scale` is a multiplier (0.0 = full removal).

9. **Machine speeds (user-measured):** cache build of 725 prompts took about 510 s on the GTX 1660 Ti machine and 129 s on the RTX 4050 laptop (about 4x faster). Split later jobs 4:1 using the new weighted sharding: `--shard 0,1,2,3/5` on the RTX 4050, `--shard 4/5` on the other (works in `tools/build_strength_cache.py` and `run_batch.py`). See Entry 23, section 13.3.

10. **CORRECTION to item 9:** the speed ratio is **2.4 to 1** (per-prompt compute 0.70 s vs 0.29 s), not 4 to 1; the 129 s figure covered only 450 resumed prompts. Use `--shard 0,1,2,3,4/7` (faster RTX 4050) and `--shard 5,6/7` (slower). The merged cache (1,450 files) passed `tools/verify_strength_cache.py`; the second machine used torch 2.5.1+cu121 and was checked against recomputation on the first machine (identical candidates, differences about 1e-6). See Entry 23, section 13.4.

11. **Headroom script written:** `tools/measure_headroom.py` (see Entry 23, section 13.5). Run `.venv\Scripts\python.exe tools/measure_headroom.py` (about 30 min, needs the merged cache of 1,450 files). Not yet run on the full cache; the user runs it. Next after that: strength tables (v1), training code, real sweeps.

12. **Headroom RESULT recorded** in Entry 23 section 13.6 (data in `docs/Research_Journal/packs/headroom/`). Headline: SAE-feature scaling can reach rank 1 on 80% of the 1,450 prompts if side effects are ignored (41% with a side-effect penalty; 7% for hard prompts of starting rank 101-1000 under the penalty); equal across train/val/test. Not comparable to the real reference run (not yet run). Next proposed: a global-vs-per-prompt strength grid on the cache.

13. **Strength tables script written** (`tools/build_strength_tables.py`, Entry 23 section 13.7). NOT built yet for the full set. Version 1 training needs these tables first. Run: faster machine `--shard 0,1,2,3,4/7`, slower machine `--shard 5,6/7`; then merge the `outputs/strength_tables` folders and run `--verify --expected 1450`. Tested on 20 prompts: 4.59 s per prompt on the GTX 1660 Ti machine.
