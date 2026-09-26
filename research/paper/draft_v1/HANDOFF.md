# Handoff (written 2026-09-26/27, end of a long session)

Read this first when resuming. Branch: `pool-refill-implementation`. Last commit before this file: `389b218`.

## Where things stand

**Project:** transient activation steering in GPT-2 small: edit SAE features (`gpt2-small-res-jb`, mostly layer 8, `blocks.8.hook_resid_pre`) during one forward pass to raise a suppressed target token to rank 1. Repo folder name: FeatureScalpel.

**Finished this session**
1. App trimmed to: Hybrid Mute & Boost, Monosemanticity Analysis, Session History (rewritten, auto-saves to `outputs/session_history/`), Sequential vs Batched Proof, and a new **Batch tab** (background runs, live progress, GPU panel, charts, statistics, "Build paper pack").
2. Headless engine: `src/hybrid_runner.py` (port of the Hybrid tab, plus safety modes strict / tolerance / graded, collateral KL), `src/batch_runner.py` (arms, sharding, resume, exact sign and McNemar tests), `src/batch_analysis.py`, `src/paper_pack.py` (labeled figure/table/provenance packs), `run_batch.py`.
3. Studies with packs in `docs/Research_Journal/packs/`:
   - Candidate source (Entry 20): all positions 10/13 vs last token 4/13 reach rank 1.
   - Safety-filter pilot (45 prompts, all positions) and full last-token (131 prompts): the strict filter is worth keeping; relaxing it (tolerance or graded) adds at most one prompt. Only ~2.5 to 2.9% of candidate features are unusable on both sides. Ceiling = number of active features.
4. Paper draft v1 (`research/paper/draft_v1/`): `main.tex`, `RECONSTRUCTION.md`, `EVIDENCE_AUDIT.md`, figures. **Not compiled** (no LaTeX on this machine); static checks pass (`tools/check_latex_static.py`).

## Open items, in priority order

1. **Compile `main.tex` in Overleaf** and fix whatever breaks.
2. **Answer the questions in the last message / `RECONSTRUCTION.md` section 8:** scope of the older H10 to H12 material; which stage-1 success is right (Castellacci per table vs Eiffel Tower per narrative); 45 vs 46 facts; hardware behind Entry 15 timings; author block; GPT-2 report year; ROME `known_1000` source.
3. **Missing sources:** `conversations.json` (H6 to H8), raw logs for `exp_001/003/005`, Dublin/Melbourne hybrid output. Full PDFs of Meng, Cunningham, Elhage, Wu (only abstracts were checked).
4. **Optional run:** `data/safety_filter_spec_full.json` (all-positions, 131 prompts, ~50 min with 2 workers) to complete the filter section. The user runs studies themselves through the Batch tab.
5. **Known weak spots to fix in code:** blocker is fixed per round (should re-select when top-1 changes); Groq explanation uses an overwritten `probs` for the baseline rank; multi-token targets are scored on the last piece; `iterative_ablate` / `check_specificity` hard-code layer 8; no automated tests for editing/safety code.
6. **Idea not started:** multi-layer intervention (start with a 2 to 3 layer pilot from the 6 to 9 plateau); mechanisms that can use features that are off.

## How to work here (lessons from this session)

- **The user wants to run experiments themselves.** Write code and analyse files they send; do not launch studies. Small smoke tests of new code are fine if disclosed.
- **Plain language first**, then detail. Explain where any number came from. Do not throw unexplained numbers.
- **Honest null results are wanted and documented.** No unsupported claims; flag discrepancies instead of choosing a side.
- **Paper style:** no em dashes, no hype, no novelty claims, no baseline comparisons presented as done. Every claim traceable in `EVIDENCE_AUDIT.md`.
- Ask before commit/push; the user usually says "commit and push" once work is done. Never commit `outputs/` (raw results are large and local).
- **Shell gotcha:** bash heredocs and Python regex with backslashes get mangled. Write patch scripts to a file with the Write tool, then run them. Files are CRLF on Windows; read with `newline=''` and normalize.
- Use `D:\Work\PROJECTS\FeatureScalpel\.venv\Scripts\python.exe` (torch 2.6.0+cu124, CUDA works). The default `python` has no torch.
- GPU is a GTX 1660 Ti (6 GB) shared with the user's apps; the Batch tab caps workers by free memory (usually 1 to 2).
- Do not read a batch result JSON while a run writes it (Windows file lock); the runner now retries, but poll the `.progress.json` instead.

## Key files

| Need | File |
|---|---|
| Journal | `docs/Research_Journal/1.md` to `20.md`; hypothesis log `research/hypothesis_log.md` |
| Study data + packs | `benchmark_results/candidate_source_study/`, `docs/Research_Journal/packs/` (manifests hold checksums) |
| Specs / prompt sets | `data/safety_filter_spec_*.json`, `data/safety_filter_prompts_full.json`, `tools/build_prompt_set.py`, `tools/make_safety_specs.py` |
| Analysis | `src/batch_analysis.py`, `tools/analyze_safety_batch.py`, `tools/make_paper_pack.py` |
| Paper | `research/paper/draft_v1/` |

## Not yet written
A journal Entry 21 (the filter study). Drafts exist in each pack's `journal_entry_draft.md`; the interpretation section is blank.
