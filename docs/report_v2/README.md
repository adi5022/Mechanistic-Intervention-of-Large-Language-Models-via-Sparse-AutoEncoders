# Updated project report (October 2026): FeatureScalpel, the whole project from first prototype to the cross-model side study

`Report_v2.tex` is the rewritten main project report (**46 pages**: 3 unnumbered front pages, 7 roman-numbered, 36 arabic including 3 of references and 4 of appendices; the seven chapters take 29 pages). It follows the format of the first report in `docs/report/` (cover, certificate, declaration, acknowledgement, abstract, contents, lists, chapters, references, appendices) and, unlike the first draft of this folder, tells the **whole research process in order**: first prototype, single-feature ablation, several competitors, mute and boost, depth and speed, a reliable sweep (pool refill, all positions, safety-filter study), repair diagnostics, CounterFact benchmark and headroom, learned strengths, gradient descent (with its literature lineage), the additive edit and the random-word control, behaviour beyond the next word. The cross-model transfer study is Chapter 5 and is presented as an **applied side study**, not as the evolved version of the system. **`Report_v2.pdf` is the compiled result.** The first report (`docs/report/`) is left untouched.

## Folder
- `Report_v2.tex`: main file (preamble, front matter, `\input` of the chapters). `Report_v2.pdf`: compiled report.
- `chapters/`: `abstract_body`, `ch1_introduction`, `ch2_literature`, `ch3_method` (the mathematics, checked against `src/hooks.py`, `src/editing.py`, `src/gradient_editing.py`), `ch4_evolution` (the trial-and-error record), `ch5_transfer`, `ch6_implementation`, `ch7_conclusion`, `references`, `appendices` (A: journal entries to sections; B: pass bars of the side study; C: reproduction).
- `figures/` (13 used): `h_*.png` are drawn by `tools/make_report_v3_history_figures.py` (early-stage numbers copied from journal entries 2, 3, 4, 5, 6, 8, 12 to 15; the gradient-descent bars read from `docs/Research_Journal/images/e29_figure_data.json`); `r_*.png` by `tools/make_report_v2_figures.py` from the data pack (no model is run); `fig_additive`, `fig_candsrc`, `fig_headroom`, `fig_learned` are copied unchanged from `docs/report/figures/`. Some `r_*.png` files are not used in the text.

## Compile
Compiled with Tectonic 0.17.0 (self-contained TeX engine): no errors, no undefined references or citations. Remaining warnings are harmless (one 2 pt overfull line, the bold small-caps font substitution on the cover). To compile elsewhere: upload the folder to Overleaf and compile `Report_v2.tex` with pdfLaTeX twice, or run `tectonic -X compile Report_v2.tex`. A missing figure prints a red box instead of stopping the compile; the college logo is optional.

## Open items (red `[TODO: ...]` in the text)
1. Certificate: the project guide's designation.
2. Section 6.4: the AI-assistance statement is kept in case the college requires disclosure; check the policy and delete it if not.
Also: the college logo `figures/mbcet_logo.png` is not in the repository; certificate and declaration dates are empty. Head of Department is Dr. Ann Alex (the first report, `docs/report/Reportmain.tex`, still says Dr. Jisha John).
The description of Chen et al. (model stitching) was checked against the paper's abstract only.

## Where each number comes from
Journal entries 1 to 37 (`docs/Research_Journal/`), the plan with every pass bar (`docs/cross_model_transfer/PLAN.md`), the data packs, and `docs/cross_model_transfer/PAPER_NOTES.md`. Appendix A maps entries to sections; Appendix B lists the side study's pass bars with their verdicts.
