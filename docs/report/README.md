# Project report (LaTeX)

`Reportmain.tex` is the B.Tech main project report of FeatureScalpel. It was rewritten on 2026-10-07 from the research journal (entries 1 to 30), the stored result packs and the code. This folder is a snapshot of it. The copy the authors review and edit is in their own report folder (`D:\Downloads\Files\Project\Project Report\`), so the two can drift apart: ask for the snapshot to be refreshed after the authors change their copy.

## Status

- **Not compiled.** The machine it was written on has no LaTeX. Only `tools/check_latex_static.py` was run (braces, environments, labels, references, citations, special characters). Compile with pdfLaTeX twice (Overleaf works): upload `Reportmain.tex` and the `figures/` folder, and fix whatever the log reports.
- Every number in the tables was checked against the stored packs in `docs/Research_Journal/packs/`. Appendix E of the report lists where each result comes from and whether it was measured on the real model, on a stand-in, or on hand-picked prompts.

## Figures

- `figures/` holds the 20 new figures (`fig_*.png`, `gd_*.png`) and `report_figure_data.json` (every number that is plotted). They are drawn from the committed packs by `tools/make_report_figures.py` and `tools/report_diagrams.py`:

      .venv\Scripts\python.exe tools/make_report_figures.py --out docs/report/figures

- **13 figures of the first version of the report are not in the repository** and have to be added to `figures/` from the authors' own folder before the report compiles: `mbcet_logo`, `delta`, `dfd0`, `dfd1`, `strategy`, `decay`, `topk`, `hybrid`, `layer`, `passes`, `gpu`, `ui_app`, `ui_bench`.

## Open items (red `[TODO: ...]` markers in the .tex; search for "TODO")

1. Certificate: the project guide's designation (the name, Ms. Poorna B R, was taken from the paper draft and should be confirmed).
2. Chapter 2: the full literature survey. The chapter covers only the sources the project examined, mostly at the level of their abstracts.
3. Section 4.3 (Development Process): the wording of the AI-assistance statement, to be confirmed with the project guide and the college policy. The declaration's "original work" wording goes with it.
4. Sections 4.5 and 5.4: the result of `tools/check_mac_parity.py` on the MacBook. The MPS path has not been measured.
5. Appendix D: whether the GitHub repository is public.

## Choices made while rewriting (change if wrong)

- The gradient-descent editor is the main measured editor and the Hybrid sweep is the original system and baseline; the additive edit is an exploratory negative result; the first stage is kept short as development history.
- No comparison with ROME, MEMIT or in-context editing has been run, so it appears only as future work.
- The declaration and acknowledgement use "we"; the cover is dated October 2026.

`docs/report_revision_guide.md` is the audit of the first version of the report (what was wrong or outdated, and where each fact lives in the repository).
