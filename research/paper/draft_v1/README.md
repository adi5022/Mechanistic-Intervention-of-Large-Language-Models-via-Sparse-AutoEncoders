# Paper draft v1

Reference draft of the paper. Accuracy and traceability were prioritized over polish.

| File | Purpose |
|---|---|
| `main.tex` | The paper (v2: organized around the current application; development history in its own section; IEEEtran conference class, inline bibliography) |
| `figures/` | PDF/PNG figures generated from repo data by `tools/make_paper_figures.py` |
| `RECONSTRUCTION.md` | Research-state summary, implemented vs tested vs open, discrepancies, deliberate omissions |
| `EVIDENCE_AUDIT.md` | Claim by claim: evidence source, type, status |

## Building
`pdflatex main.tex` twice (Overleaf works; upload `main.tex` and `figures/`). **No LaTeX compiler was available when this was written, so the file has not been compiled.** `tools/check_latex_static.py` was run instead (braces, environments, labels, citations, figure files, tabular column counts, non-ASCII and dash characters); it passed, but it is not a compile.

## Regenerating numbers and figures
```
python tools/summarize_layer_benchmark.py     # layer table (Section VII-D)
python tools/make_paper_figures.py            # figures
python tools/check_latex_static.py research/paper/draft_v1/main.tex
```
The safety-filter tables come from `docs/Research_Journal/packs/entry21_pilot_*` and `entry21_full_last_*` (manifests record checksums of the raw result files, which stay local in `outputs/safety_batches/`).

## Open items (red `\todo` in the text and in `RECONSTRUCTION.md`)
Author block, year of the GPT-2 report, and the decisions listed in `RECONSTRUCTION.md` section 8.
