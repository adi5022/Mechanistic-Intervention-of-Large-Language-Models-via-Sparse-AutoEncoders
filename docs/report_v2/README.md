# Updated project report (October 2026): FeatureScalpel, transient SAE edits and their transfer between language models

`Report_v2.tex` is a new, shorter report (**40 pages**: 3 unnumbered front pages, 7 roman-numbered, 30 arabic including references and appendices) in the same format as the first report in `docs/report/` (cover, certificate, declaration, acknowledgement, abstract, contents, lists, chapters, references, appendices). It covers the original motivation (the single-model transient edit), the cross-model transfer study (Export and Import) with all its checkpoints, and the corrections made along the way. The first report (`docs/report/`) is left untouched. **`Report_v2.pdf` is the compiled result.**

## Folder
- `Report_v2.tex`: the main file (preamble, front matter, `\input` of the chapters). `Report_v2.pdf`: the compiled report.
- `chapters/`: `ch1_introduction`, `ch2_literature`, `ch3_methodology`, `ch4_implementation`, `ch5_results`, `ch6_conclusion`, `references`, `appendices`.
- `figures/`: 12 figures. Nine (`r_*.png`) are drawn at the page width by `tools/make_report_v2_figures.py` from the committed data pack `docs/Research_Journal/packs/cross_model_transfer/` only (no model is run); three (`fig_pipeline`, `gd_rank1_by_band`, `fig_additive`) are copied unchanged from the first report.

## Compile
Compiled with Tectonic 0.17.0 (a self-contained TeX engine that fetches the packages it needs): no LaTeX errors, no undefined references or citations, 40 pages. Only harmless warnings remain (a few under- and overfull lines, the bold small-caps font substitution on the cover, duplicate `page.1` anchors from restarting the page numbers). To compile elsewhere: upload the whole folder to Overleaf and compile `Report_v2.tex` with pdfLaTeX twice (no bibliography tool is needed), or run `tectonic -X compile Report_v2.tex`. A missing figure never stops the compile: `\safegraphic` prints a red box with the file name; the college logo is optional and prints nothing while `figures/mbcet_logo.png` is absent.

Regenerate the figures: `.venv\Scripts\python.exe tools/make_report_v2_figures.py`

## Open items (red `[TODO: ...]` in the text; search for "TODO")
1. Certificate: the project guide's designation.
2. Section 4.4 (Development Process): the wording of the AI-assistance statement, to be confirmed with the project guide and the college policy.
3. Section 2.3: the description of the stitching paper (Chen et al., NeurIPS 2025) was written from its summary; check it against the full text. Its entry in the references lacks first-name initials for the same reason.
4. The college logo `figures/mbcet_logo.png` is not in the repository (the cover and certificate show nothing there until it is added).
5. Certificate and declaration dates.

## Where each number comes from
Journal entries 31 to 37 (`docs/Research_Journal/`), the plan with every pass bar (`docs/cross_model_transfer/PLAN.md`), the data pack above and `docs/cross_model_transfer/PAPER_NOTES.md`. Appendix A of the report lists every pass bar with its verdict.
