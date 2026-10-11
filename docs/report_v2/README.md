# Updated project report (October 2026): FeatureScalpel, the whole project from first prototype to the cross-model side study

`Report_v2.tex` is the rewritten main project report (**73 pages**: 10 front pages, 53 pages of eight chapters, 5 of references, 5 of appendices). It follows the format of the first report in `docs/report/` (cover, certificate, declaration, acknowledgement, abstract, contents, lists, chapters, references, appendices) and tells the **whole research process in order**: first prototype, single-feature ablation, several competitors, mute and boost, depth and speed, a reliable sweep (pool refill, all positions, safety-filter study), repair diagnostics, CounterFact benchmark and headroom, learned strengths, gradient descent (with its literature lineage), the additive edit and the random-word control, behaviour beyond the next word, **the ROME reference (Chapter 5)**, and the cross-model transfer study as an **applied side study** (Chapter 6). **`Report_v2.pdf` is the compiled result.** The first report (`docs/report/`) is left untouched. The NLA ideas are deliberately not in the report: they were never worked on.

## What changed on 2026-10-12
- **Figures: 13 to 42.** Everything that already had saved data but no figure in the report was drawn or brought in; each figure caption now says how to read it. Appendix D lists, for every figure, the stored data and the script.
- **New Chapter 5, "ROME as an External Reference".** What ROME does, how the official code was set up on the branch `rome-factual-editing` (7 Oct), causal tracing of GPT-2 Small (97 facts), the edit-layer sweep (layer 3), how ROME's scores (ES, PS, NS) map onto ours, and a cross-check of our edit with ROME's own metrics. It says plainly that the head-to-head comparison (ROME plan steps 6 to 8) has **not** been run, and lists what it needs.
- **Citations.** New literature references r53 to r66 (residual stream, KL divergence, Wikitext, sign test, bootstrap, causal mediation, key-value memories, text degeneration, distinct-n, Hugging Face transformers, ROME code and data, Streamlit). Every result from our own experiments is cited to a numbered **project record** P1 to P19 (journal entries, data packs, ROME log; see the end of the reference list). Chapter 8 ends with a table of which claims the research paper may make and the evidence behind each.
- Short list-of-figures titles (the full caption stays in the text), floats placed `[!htb]`.

## Folder
- `Report_v2.tex`: main file. `Report_v2.pdf`: compiled report.
- `chapters/`: `abstract_body`, `ch1_introduction`, `ch2_literature`, `ch3_method`, `ch4_evolution`, `ch5_rome`, `ch6_transfer`, `ch7_implementation`, `ch8_conclusion`, `references`, `appendices` (A: journal entries to sections; B: pass bars of the side study; C: reproduction; D: data and script behind every figure).
- `figures/`: `h_*.png` by `tools/make_report_v3_history_figures.py`; `r_*.png` by `tools/make_report_v2_figures.py`; `v4_*.png` by `tools/make_report_v4_figures.py` (drawn from the data packs, or copied unchanged from `docs/report/figures/` and `docs/rome_baseline/tracing/`); `fig_additive`, `fig_candsrc`, `fig_headroom`, `fig_learned` copied from `docs/report/figures/`. No figure script runs a model.

## Compile
Compiled with Tectonic 0.17.0: no errors, no undefined references or citations; remaining warnings are three overfull lines of at most 5 pt and the bold small-caps font substitution on the cover. To compile elsewhere: upload the folder to Overleaf and compile `Report_v2.tex` with pdfLaTeX twice, or run `tectonic -X compile Report_v2.tex`. A missing figure prints a red box instead of stopping the compile. The PDF pages have not been inspected by eye (page fill was checked from the PDF's content streams: no half-empty page inside a chapter).

## Open items (red `[TODO: ...]` in the text)
1. Certificate: the project guide's designation.
2. Section 7.3: the AI-assistance statement is kept in case the college requires disclosure; check the policy and delete it if not.
Also: the college logo `figures/mbcet_logo.png` is not in the repository; certificate and declaration dates are empty. Head of Department is Dr. Ann Alex (the first report, `docs/report/Reportmain.tex`, still says Dr. Jisha John). The descriptions of Chen et al. (model stitching) and Gupta et al. rest on their abstracts only. The ROME paper's own numbers for GPT-2 XL were not compared with ours.

## Where each number comes from
Journal entries 1 to 37 (`docs/Research_Journal/`), the ROME log (`docs/rome_baseline/PROGRESS.md`), the plan with every pass bar (`docs/cross_model_transfer/PLAN.md`), the data packs, and `docs/cross_model_transfer/PAPER_NOTES.md`. Appendix A maps entries to sections; Appendix B lists the side study's pass bars; Appendix D traces every figure.
