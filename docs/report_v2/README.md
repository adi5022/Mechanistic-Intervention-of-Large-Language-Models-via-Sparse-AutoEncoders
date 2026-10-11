# Project report (October 2026): FeatureScalpel, from first prototype to the cross-model side study

`Report_v2.tex` is the main project report (**60 pages**: 10 front pages, 39 pages of eight chapters, 5 of references, 6 of appendices) and `Report_v2.pdf` is the compiled result. It follows the format of the first report in `docs/report/` and tells the whole research process in order: first prototype, single-feature ablation, several competitors, mute and boost, depth and speed, a reliable sweep, repair diagnostics, the CounterFact benchmark, learned strengths, gradient descent (with its literature lineage), the additive edit and its control, behaviour beyond the next word, a short **ROME reference chapter** (Chapter 5), and the cross-model transfer study as an **applied side study** (Chapter 6). The NLA ideas are not in the report because they were never worked on.

## Folder
- `Report_v2.tex`: main file. `Report_v2.pdf`: compiled report.
- `chapters/`: `abstract_body`, `ch1_introduction`, `ch2_literature`, `ch3_method`, `ch4_evolution`, `ch5_rome`, `ch6_transfer`, `ch7_implementation`, `ch8_conclusion`, `references`, `appendices` (A: journal entries to sections; B: pass bars of the side study; C: reproduction; D: data and script behind every figure).
- `figures/` (21 figures): `h_*.png` by `tools/make_report_v3_history_figures.py`; `r_*.png` by `tools/make_report_v2_figures.py`; `v4_*.png` by `tools/make_report_v4_figures.py` (that script also draws a few extra figures that are not used in the text); `fig_*.png` copied from `docs/report/figures/`. No figure script runs a model.

## Citations
Statements about other work cite the literature (references 1 to 66). Results from our own experiments cite numbered **project records** P1 to P19 (journal entries, data packs, the ROME log), listed at the end of the reference list.

## Compile
Compiled with Tectonic 0.17.0: no errors, no undefined references or citations. Overleaf's free plan times out on this project; use the compiled PDF, or compile locally with `tectonic -X compile Report_v2.tex`. The PDF pages have not been inspected by eye (page fill was checked from the PDF's content streams).

## Open items (red `[TODO: ...]` in the text)
1. Certificate: the project guide's designation.
2. Section 7.3: the AI-assistance statement is kept in case the college requires disclosure; check the policy and delete it if not.
Also: the college logo `figures/mbcet_logo.png` is not in the repository; certificate and declaration dates are empty. Head of Department is Dr. Ann Alex (the first report, `docs/report/Reportmain.tex`, still says Dr. Jisha John). The descriptions of Chen et al. and Gupta et al. rest on their abstracts only. References 53 to 66 were added from memory of well-known papers and should be checked against the sources.

## Not in the report
The ROME head-to-head comparison (ROME plan steps 6 to 8) has not been run; Chapter 5 says so. A fuller table of the claims the research paper may make is not included.
