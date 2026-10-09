# Handoff (2026-10-10): cross-model transfer study. START HERE for this branch

Repository: `Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders`. Branch: **`cross-model-transfer`** (from `main` at `bc35482`). `main` equals the old `gradient-descent-editing` branch plus four preserved documents; the older long handoff for that work is `docs/handoff_2026-10-05_mac.md`. This file covers only the new study. Branch `rome-factual-editing` is worked on by another team member: do not delete or rebase it.

## The study in one paragraph
Take the change an edit makes inside GPT-2 small (layer 8) and add it, translated, to GPT-2 medium while it reads the same sentence (**Export**), or take medium's state and add it, translated, to small (**Import**), with no weight changed in either model. A linear translator fitted on ordinary text maps one model's residual stream (`blocks.L.hook_resid_pre`) to the other's. Direction names follow the information: "small to medium" = read in small, written into medium.

## Read in this order
1. `docs/cross_model_transfer/PLAN.md`: runbook with a status table, results log, the pre-stated gates, and the proposed design of the controlled step D.
2. `docs/Research_Journal/31.md`: results of steps A0 to B1 and the D pilot, mistakes made, limits.
3. `docs/cross_model_transfer/DESIGN.md`: the earlier long reasoning (banner at its top lists what changed since).
4. `docs/cross_model_patching_idea.md`: a design note written by a cloud session; it calls Import "Direction 1" and Export "Direction 2" (the reverse of the numbering in the first draft of the plan).
5. Data: `docs/Research_Journal/packs/cross_model_transfer/` (README inside).

## Where it stands (plain words)
- A translator between small and medium exists and keeps the receiving model working (stitching recovers 91 to 98%), but it carries a sentence-to-sentence difference well only at the changed word's position (cosine 0.63 to 0.76) and weakly at the last position (0.24 to 0.31). Medium to small is easier than small to medium.
- Country swap (B1, Gate 2 fixed beforehand): the translated "France became Germany" difference flips the receiver to the swapped capital on 55% (small to medium) and 33% (medium to small) of test cases, controls 0 to 1%. This is a control of the channel; it is not a fact and not an edit.
- Pilot of a real edit (30 records, not a result): the gradient-descent edit puts 89% of its change at the subject's positions; translated into medium it raises the target from median rank 165 to about 22 (top-1 up to 20%, controls 0 to 3%), but pushing the target word's own output direction does it in 87 to 100%. So top-1 cannot show a fact; the controlled step D must be judged on rewordings, neighbours and unrelated prompts.
- Nothing about facts has been shown. Phases C, D (controlled) and E have not been run.

## Run things (Windows, repository root, `.venv\Scripts\python.exe`; the first A0 run downloads GPT-2 medium 1.5 GB and wikitext-2 8 MB)
```
tools/transfer/00_setup_check.py [--quick]            about 25 s once cached
tools/transfer/01_fit_maps.py [--quick|--selftest] --sgd-check     138 s, writes outputs/transfer/maps_gpt2_to_gpt2-medium.pt (160 MB)
tools/transfer/02_check_maps.py [--quick] --feature-view           59 s
tools/transfer/03_swap_control.py --pairs s2m_L8_L16,m2s_L16_L8 --tag primary     279 s (saves each pair as it finishes; --fresh recomputes)
tools/transfer/04_export_pilot.py [--n 6]                          158 s for 30 records
```
Timings are for the RTX 4050 laptop GPU (6 GB). Set `HF_HUB_OFFLINE=1` to avoid network calls once everything is cached. Everything lands in `outputs/transfer/` (a `.gitignore` there keeps it out of git); the committed copies of small results are in the data pack.

## Gotchas learned the hard way
- `tokenizer.encode(...)` in this transformers version adds the start token; count word tokens with `tokenizer(w, add_special_tokens=False)`.
- The plain prompt "The capital of X is" makes GPT-2 answer "the" or "a"; use "The capital of X is the city of".
- Background jobs: give a generous time limit (a 50-minute limit killed a 75-minute run and, before the fix, lost everything). The B1 tool now saves per pair; `tools/transfer/*` print one line per stage, not a progress bar inside a pair.
- GPU memory: 6.4 GB total, about 5.3 GB free; the two models need about 2 GB; do not run two GPU jobs at once.
- Sentence templates for the difference checks must continue past the changed word, or "last position" is the word itself and the cosine is inflated.
- New code lives in `src/transfer/`, `tools/transfer/`, `src/` is otherwise untouched; the older functions are imported and called, never edited (the author's rule).

## Open decisions for the author
1. **Gate 3** for the controlled step D (proposal in `PLAN.md`, written after the pilot): top-1 at least 15%, at least 10 points above the best non-push control, paired p < 0.01; rewordings, neighbours and unrelated prompts reported. Confirm or change the numbers before the run.
2. Run Phase C (fact edit or word push, small alone) before, in parallel with, or after step D.
3. Whether Phase G uses Gemma 3 270M and 1B (SAEs on both sides) or a GPT-2 medium SAE trained here.
4. When to merge `cross-model-transfer` into `main`.

## Next steps, in order
1. D1: build the CounterFact benchmark (dev 50, test 150, excluding the ROME dev records and the pilot records) and the relay-mode export; run with the controls.
2. Phase C on the same benchmark (also serves the ROME comparison, steps 6 and 7 of `docs/rome_baseline/PROGRESS.md`).
3. Phase E, import.
4. Entry 32 with figures from the real data; paper draft last.

## How the author likes to work (still applies)
Plain words first, then detail; one concrete example when something is unclear; do exactly what is asked and propose extras; document everything with dated entries; figures and tables only from real data; label things honestly (exploratory, pilot, proxy); say plainly when something failed. The author runs long jobs and may watch them in their own terminal.
