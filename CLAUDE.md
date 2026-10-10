# Project instructions for Claude (cross-model transfer study, branch `cross-model-transfer`)

Read first: `docs/handoff_compact.md` (newest section last), then `docs/handoff_2026-10-10_cross_model.md`, `docs/Research_Journal/33.md` and `docs/cross_model_transfer/PLAN.md`. The paper's skeleton and claims are in `docs/cross_model_transfer/PAPER_NOTES.md`.

## The goal
Everything we do must end up in a paper that survives review. So every important result is logged, reproducible and explained, every failure has its reason and evidence written down, and every figure is drawn from saved data.

## After EVERY experiment (same session, before moving on)
1. **Raw log saved.** Tools in `tools/transfer/` copy their output to `outputs/transfer/logs/` automatically (`src/transfer/runlog.py`). Copy the logs that matter into `docs/Research_Journal/packs/cross_model_transfer/logs/`. If the user ran something in their own terminal and pasted the output, save the pasted text as a log file there too (note that it was pasted and what was removed).
2. **Result files in the data pack.** Copy the result JSON to `docs/Research_Journal/packs/cross_model_transfer/` and add a row to that folder's `README.md`. Tools must save per-run details (epoch curves, per-dose tables, seeds) in their JSON, not only final numbers.
3. **Journal entry** (`docs/Research_Journal/NN.md`): what was done, the result tables, the pre-stated rule and its verdict, what the result does NOT show, mistakes and corrections, why it failed or worked (with the evidence), the next options. Failures are written up as carefully as successes.
4. **Plan, logs and index:** status row and results-log paragraph in `PLAN.md`; hypothesis row (`research/hypothesis_log.md`: text, test, result, verdict, evidence); `research/experiment_index.md`; dated `research/timeline.md` entry.
5. **Figures from saved data only** (`tools/make_transfer_figures_33.py` and its successors; they read the pack, never run a model). Follow the dataviz skill: one message per figure, direct labels, colour follows the entity, light surface, run `validate_palette.js` on a new palette, look at the image before keeping it.
6. **`PAPER_NOTES.md` updated** (headline table, reason for each failure, claims allowed / not allowed, open caveats).
7. **Handoff files updated** (`docs/handoff_compact.md` for Claude, `docs/handoff_2026-10-10_cross_model.md` for people).
8. **Commit and push** to `cross-model-transfer` when the user says so (they have said so for this work; never push `main`). Stage explicit paths; never `git add -A` (`outputs/` and `.env` stay out).

## Rules for experiments
- **Gates and decision rules are written into `PLAN.md` BEFORE the run** and never moved afterwards. A step designed after seeing a result is labelled POST-HOC and cannot change a verdict.
- Controls always accompany a result (random vector, wrong record's edit, rank-matched random word, word push, fact in the prompt); rank 1 alone is weak evidence.
- Choices (layer, dose, epoch) are made on the dev records; the test records are only read. Report seeds and ranges. Say plainly what was not tested.
- **Dose protocol (changed after D9b, 2026-10-10):** from D10 on the dose is chosen by the best dev mean log-rank gain, not by dev top-1 (28 dev records make top-1 a coin flip). D3 to D9 keep the rule they ran under.
- **New code only.** Older functions are imported, never edited, without asking the user first. New tools go in `tools/transfer/` or `src/transfer/`.
- Long jobs: set generous time limits, save per unit of work, tell the user where the log is. The user often runs jobs in their own terminal; give the exact command.
- The command harness halves double backslashes in shell text: write scripts and docs with the file tools, put patch scripts in files, and use single backslashes in strings inside Edit calls.
- Protected: branch `rome-factual-editing` belongs to a teammate (never delete or rebase). Never push to `main` unless told.

## How to talk to the user
They have about 3 months of interpretability experience; the concepts are fine. Over-complicated, number-heavy writing is what loses them, and they have said so angrily. Explain a result as a short story first: what the step was for, what happened, what it means for the question, with one concrete example. Use two or three numbers, define every term (dose, seed, dev, test, rank, arrows like "A to B") the first time, say what was not tested and what the next decision is. Tables only after the plain explanation and small. Use "dose" and not metaphors for it.

## Backups
`outputs/transfer/` (about 1 GB: trained maps, per-record numbers) is not in git. Remind the user to copy it somewhere safe before a milestone or a paper submission.
