# Handoff (2026-10-10, updated after the overnight runs): cross-model transfer study. START HERE for this branch

Repository: `Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders`. Branch: **`cross-model-transfer`** (from `main` at `bc35482`; the first commit `405aadc` is pushed, the overnight work after it is on disk and NOT committed or pushed). The older long handoff for the gradient-descent work is `docs/handoff_2026-10-05_mac.md`. This file covers only the new study. Branch `rome-factual-editing` is worked on by another team member: do not delete or rebase it.

## The study in one paragraph
Take the change an edit makes inside GPT-2 small (layer 8) and add it, translated, to GPT-2 medium while it reads the same sentence (**Export**), or take medium's state and add it, translated, to small (**Import**), with no weight changed in either model. A translator fitted on ordinary text maps one model's residual stream (`blocks.L.hook_resid_pre`) to the other's. Direction names follow the information: "small to medium" = read in small, written into medium.

## Read in this order
1. `docs/Research_Journal/32.md`: the overnight results (controlled export, neural translator, fact-or-push test), with figures `docs/Research_Journal/images/e32_fig*`.
2. `docs/Research_Journal/31.md`: translators, country swap, 30-record pilot (one statement in it is withdrawn, see section 3.5 there).
3. `docs/cross_model_transfer/PLAN.md`: runbook, status table, results log, gates; `DESIGN.md` is the earlier reasoning.
4. `docs/cross_model_patching_idea.md`: a design note from a cloud session (its "Direction 1" is Import, "Direction 2" is Export).
5. Data: `docs/Research_Journal/packs/cross_model_transfer/` (README inside).

## Where it stands (plain words)
- A linear translator keeps the receiving model working and carries which word changed well; the last-position summary weakly. Medium to small is easier than small to medium. A neural translator fits clearly better but exports an edit no better.
- Country swap (B1, Gate 2 fixed beforehand): passes, 55% (small to medium) and 33% (medium to small), controls 0 to 1%. A control of the channel, not a fact.
- **Controlled export of the real gradient-descent edit (D, Gate 3 fixed beforehand): FAILS** on top-1 (5% and 8% against 15%) and on the margin over the strongest control (0 and 3 points against 10). But the graded effect is real and specific: the target's rank in medium goes 142 to 23 and 14, "new beats true" on rewordings rises 15 to 16 points (about 60% of what the edit does in small itself), neighbours lose 4 to 5 points, random vector and another record's edit do nothing, and the norm-matched dose agrees. A plain word push makes the target the top answer in most records but destroys neighbours (23% kept at layer 16); stating the fact in medium's prompt moves rewordings to 97% but costs neighbours 23 points. In the athlete picture: the big player picks up the trick partly, without forgetting his game, but rarely does the move cleanly.
- **Phase C (small alone):** the pre-stated rule, applied as written, gives read-out steering (rewordings gain 1.04 against 0.82 for a rank-matched random-word edit, p = 0.032; an edit restricted to the subject's tokens succeeds on 12% of records against 59% for all positions), but the edit is not dominated by the target's output direction (logit-lens rank 4,922) and the evidence is weak either way. The Entry 31 statement "mainly a change of the subject's representation" is withdrawn.
- Not done: Import of real facts (E), the Gemma replication (G), the tester app (H), text quality over generated continuations, any other relation or model pair.

## Run things (Windows, repository root, `.venv\Scripts\python.exe`; the first A0 run downloads GPT-2 medium 1.5 GB and wikitext-2 8 MB)
```
tools/transfer/00_setup_check.py [--quick]                          about 25 s once cached
tools/transfer/01_fit_maps.py [--quick|--selftest] --sgd-check      138 s -> outputs/transfer/maps_gpt2_to_gpt2-medium.pt (160 MB)
tools/transfer/02_check_maps.py [--quick] --feature-view            59 s
tools/transfer/03_swap_control.py --pairs ... --tag primary         279 s (saves each pair; --fresh recomputes)
tools/transfer/04_export_pilot.py                                   158 s (30-record pilot)
tools/transfer/05_export_edits.py                                   1,263 s: 200 records x 2 edits in small (resumable)
tools/transfer/07_fit_mlp_maps.py                                   380 s: neural translators (needs step A1 maps)
tools/transfer/06_export_eval.py [--translator mlp]                 about 54 min each: controlled export (stage 1 and 2)
tools/transfer/09_analyse_export.py [--translator mlp]              seconds: the corrected paired analysis (use THIS for numbers)
tools/transfer/08_fact_or_push.py                                   1,230 s: Phase C
tools/make_transfer_figures.py [1 2 3 4 5 6]                        figures from the committed pack only
```
Times: RTX 4050 laptop GPU (6 GB). Set `HF_HUB_OFFLINE=1` once everything is cached. Everything lands in `outputs/transfer/` (ignored by git); per-record numbers of the export are in `d_eval_*.pt` there (80 MB each, local); committed summaries are in the data pack.

## Gotchas learned the hard way
- `tokenizer.encode(...)` in this transformers version adds the start token; count word tokens with `tokenizer(w, add_special_tokens=False)`.
- The plain prompt "The capital of X is" makes GPT-2 answer "the" or "a"; use "The capital of X is the city of".
- Background jobs: set a generous time limit for scripts AND for any queue script around them (a 50-minute limit killed a run; a 2-hour limit on a queue script stopped the queue while the Python job survived). Tools save per unit of work; the export and edit tools can resume.
- **The command harness halves double backslashes** inside shell command text (a `\\n` became a real newline and broke a script). Write files with the file tools, and put patch scripts in a file.
- In the first printed run of `06_export_eval.py` the random-word arm's rank gain used the counterfactual target's starting rank (the tool is fixed; the numbers in the journal come from `09_analyse_export.py`, which measures each word against its own start rank).
- GPU memory: 6.4 GB total, about 5.3 GB free; the two models need about 2 GB; the export evaluation uses about 3.5 GB; do not run two heavy GPU jobs at once.
- Sentence templates for difference checks must continue past the changed word, or "last position" is the word itself.
- New code lives in `src/transfer/`, `src/gradient_editing_masked.py`, `tools/transfer/`; older functions are imported and called, never edited (the author's rule).

## Open decisions for the author
1. What to try after the failed export (Entry 32 section 7): a translator fitted on edit differences (for example against ROME on medium, hyper-parameters exist) or with an output-matching loss; several injection layers; other edit forms. Or go to Import (E) first.
2. Whether Gate 3's numbers stay for a rerun (they were proposed after a 30-record pilot and treated as final overnight).
3. Phase G route: Gemma 3 270M and 1B (SAEs on both sides) or training a GPT-2 medium SAE.
4. When to commit and push the overnight work (nothing after `405aadc` is committed), and when to merge `cross-model-transfer` into `main`.

## Next steps, in order (proposal)
1. Commit and push the overnight work to `cross-model-transfer` when the author agrees.
2. Phase E, Import of real facts (medium's state into small, filtered by small's SAE, target-free arms first).
3. A better translator for the export (edit-difference or output-matching training) if the author wants another run at Export.
4. Entry 33 with figures from real data; paper draft last.

## How the author likes to work (still applies)
Plain words first, then detail; one concrete example when something is unclear; do exactly what is asked and propose extras; document everything with dated entries; figures and tables only from real data; label things honestly (exploratory, pilot, proxy); say plainly when something failed. The author runs long jobs and may watch them in their own terminal.
