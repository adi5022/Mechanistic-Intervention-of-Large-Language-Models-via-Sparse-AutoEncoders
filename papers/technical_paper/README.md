# Technical paper

`technical_paper.tex` is a two-column, ACL-style paper (layout modelled on Gupta et al. 2024) covering the whole project: the transient SAE-feature edit of GPT-2 Small, the gradient-descent editor, the rank-1 control, the ROME cross-check and the cross-model transfer study. `technical_paper.pdf` is the compiled result (19 pages: 14 main + references, 5 appendix).

- Figures: `figures/` (copies of `docs/report_v2/figures/`).
- Compile: `pdflatex technical_paper.tex` (twice). Needs `natbib`, `times`, `microtype`, `hyperref`, `booktabs`, `caption`, `enumitem`.
- All numbers come from the report, `docs/cross_model_transfer/PAPER_NOTES.md` and the data packs; Appendix D maps each figure to its data.
