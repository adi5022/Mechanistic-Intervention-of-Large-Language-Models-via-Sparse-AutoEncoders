# Report revision guide: `Reportmain.tex` (the older report) against the project as it stands on 2026-10-07

**Status update (later on 2026-10-07).** The author then asked for the changes to be made, and `Reportmain.tex` was rewritten from this audit (the first version is kept as `Reportmain_v1_backup.tex` next to it). This guide stays as the audit trail: what was wrong in the first version, and where every fact lives in the repository. Open items of the rewritten report are marked `\todo{...}` (red) in the `.tex`: the project guide's designation on the certificate, the full literature survey, the wording of the AI-assistance statement, the macOS parity result, and whether the repository is public.

Written for the author, who will rewrite the report. This file does not rewrite anything. It says what is wrong, what is outdated, what is still fine, what to add, and where every fact lives in the repository, from the cover page to the last appendix.

**What was read for this guide.** `Reportmain.tex` (all 983 lines); Research Journal Entries 1 to 30 (all; Entry 21 is a placeholder); `research/` (hypothesis log H1 to H23, experiment index EXP-001 to EXP-022, timeline, notebooks); the paper draft `research/paper/draft_v1/` (`main.tex`, `EVIDENCE_AUDIT.md`, `RECONSTRUCTION.md`, `HANDOFF.md`); `docs/` (literature review, session report, architecture, roadmap, handoffs, the 2026-10-06 session record); the stored result files that the report's numbers come from (`benchmark_results/`, `docs/Research_Journal/packs/`); the code layout and the git history of `src/hooks.py`.
**What was not done.** The literature survey (you said later). No source paper was read in full. No experiment was run for this guide. The `.tex` file was not edited.

**Tags used below.** [WRONG] contradicts a stored result file or the code. [OUTDATED] was true when written, is not now. [WITHDRAWN] the project itself withdrew the claim. [UNVERIFIED] no artifact in the repo supports it. [OK] still true, keep. [ADD] new material.

---

## Part 1. The big picture

### 1.1 What the old report is
A report on the project **as of about mid-August 2026** (Journal Entries 1 to 14; the report's newest content is the 12 and 13 August cost audit and batching): a Hybrid Mute and Boost editor with a fixed strength, tested on 22 hand-picked ROME facts and on one prompt (the MIT to Cambridge case), plus a layer sweep and a compute-cost audit. Its conclusion is "it raises the rank, it does not reach rank 1, nothing is compared with ROME".

### 1.2 What the project is now (to 6 October 2026)
1. The Hybrid sweep grew into a full pipeline: best-so-far tracking, cumulative sweep, one joint hook, **pool refill**, an overlap rule, **candidate source = all prompt positions** (Entries 16 to 19).
2. A batch engine and studies on real prompt sets: candidate source (17 prompts), **safety-filter study** (45 and 131 prompts) (Entries 20, 21).
3. Diagnostics: do later layers undo the edit, is the SAE the limit (Entry 22). A literature check (Entry 25).
4. A **large benchmark**: CounterFact hard set of 16,360 prompts, a subject-safe split (train 1,000, validation 150, test 150 + 150), per-prompt cache, strength tables, a headroom measurement (Entries 23, 24).
5. **Learned strengths** (PromptNet, learned fixed pair, FeatureNet): a better global pair, little gain from choosing per prompt (Entries 26, 27).
6. **Per-feature gradient descent** (one multiplier per feature, 0 to 3), compared with the sweep on 300 held-out prompts: **271 of 300 against 161 and 94** (Entries 28, 29).
7. An **additive edit** (switch on silent features), then a control that showed it forces random words to rank 1 (20 of 20), then withdrawal of the old "knowledge absent" rule (Entry 30, sections 10 to 12).
8. A new **Prototype lab** tab: baseline against edited text, follow-up prompts, tests of text quality and of instruction sensitivity (Entry 30, sections 13 to 15).
9. Engineering: a thread-race bug that silently dropped the start-of-text token (fixed), and a **macOS (Apple MPS) port** with a device layer.
10. A roadmap (`docs/research_roadmap.md`): A specificity benchmark, C more specific edit, D in-context fact vector, B comparison with IKE/ROME/DiffMean.

The report currently covers only the early part of item 1 and nothing after. Most of the later results are not in it, and several of its claims no longer hold (Part 2).

### 1.3 Old story against new story

| Topic | Old report | Now |
|---|---|---|
| Main method | Hybrid Mute and Boost, 3 + 3 features, fixed strengths 0.3 / 0.5 | Hybrid sweep (pool refill etc.) is the baseline; **per-feature gradient descent** is the strongest editor measured |
| Where the edit acts | described as the final token | every token position (always was); candidates from the last token or all positions |
| Evaluation set | 22 ROME facts; one prompt for most diagnostics | CounterFact hard set; **300 held-out test prompts** (bands 2-5, 6-20, 21-100, 101-1000) |
| Headline | "raises rank, never rank 1" | gradient descent reaches rank 1 on 271 of 300 (but rank 1 is a low bar: median target probability 7.7%) |
| Side effects | specificity on 15 controls (stage 1 only) | same-prompt KL, edit size, KL and top-1 flips on 20 unrelated prompts, reworded and nearby prompts, generated text |
| Layer | "layer 8 best on every metric, 100% improved" | a plateau around layers 6 to 9; layer 8 not clearly separated (see A1) |
| Compute | 186 to 126 passes, "1.89x" | 186 to 126 to 122 to **6 forward calls**; gradient descent about 5 to 6.5 times faster than the sweep |
| Knowledge claim | "suppressed, absent" classes; absent = not in the model | **withdrawn**: steering reachability says nothing about what the model knows |
| Comparison with others | future work | still not run (EXP-015); the literature check says compare fairly |
| Platforms | Windows + NVIDIA | also macOS (MPS) and CPU |
| Honesty record | none | negative results kept (learned strengths, additive edit), bugs and corrections documented |

### 1.4 Recommended narrative for the new report
Keep the college chapter structure. Tell it as an investigation, not a product:
1. A transient, weight-free SAE-feature edit can lift a suppressed answer; here is how it was built and what each design step fixed (Hybrid sweep).
2. How far does it go, and what does it cost? (candidate source, safety filter, layer, compute).
3. Can the strengths be learned? (largely no, apart from a better global default).
4. A per-feature optimiser does much better on the same measures (the main result), with lower same-prompt side effects, partly by construction of its loss.
5. But rank 1 is a weak test: the additive control, the reworded-prompt test, and the generation tests show what the edit does and does not do. The "knowledge absent" inference was withdrawn.
6. What is not shown, and the plan (roadmap A, C, D, B).
This is honest, publishable-sized, and matches how the journal was written.

### 1.5 Suggested new table of contents (old chapters kept)
- **Front matter:** abstract (rewrite), keywords.
- **Ch 1 Introduction:** background, problem statement and research questions, objectives, scope, contributions, organisation.
- **Ch 2 Literature review:** (2.1 fact editing; 2.2 activation steering; 2.3 SAEs; 2.4 SAE steering, unlearning, evaluations and limits; 2.5 causal localisation; 2.6 evaluation methodology, text degeneration; 2.7 comparison table; 2.8 gap, carefully worded).
- **Ch 3 System analysis and design:** requirements; data (ROME facts, CounterFact pipeline); architecture; design methodology; the edit operation; editor A (Hybrid sweep); editor B (gradient descent) and its additive extension; evaluation design and metrics; DFDs; modules; environment.
- **Ch 4 Implementation:** technologies; algorithms (Algorithm 1 selection, 2 hybrid with refill, 3 layer sweep, 4 gradient descent, 5 additive); development process (extended Table 4.1); module-wise; correctness checks; engineering lessons.
- **Ch 5 Results and discussion:** 5.1 setup; 5.2 development-stage findings (one-prompt, labelled); 5.3 depth; 5.4 candidate source; 5.5 safety filter; 5.6 CounterFact data and headroom; 5.7 learned strengths; 5.8 gradient descent against the sweep (main result); 5.9 repair diagnostics; 5.10 additive edit, control, reworded prompts; 5.11 generation and instruction tests; 5.12 compute and platforms; 5.13 discussion and threats to validity.
- **Ch 6 Conclusion and future work.**
- **References; Appendices** (A screenshots; B artifacts and per-fact tables; C code listings; D reproduction commands; E evidence table).

---

## Part 2. Claims in the old report that must change (read this first)

Line numbers refer to `Reportmain.tex`.

### A. Wrong or contradicted by stored data

| # | Where | Old claim | Problem and what to say instead | Source |
|---|---|---|---|---|
| A1 | Abstract (l.145), Contribution 5 (l.207), Section 5.2.5 (l.692), limitations (l.776), conclusion (l.785) | "Pilot benchmark, **22 suppressed prompts**, layers 2, 5, 8, 10, 11. Layer 8 best on every metric, mean rank improvement 4.36, probability gain 0.0484, improved the rank on **all 22 (100%)**. **No prompt reached rank 1 at any layer.**" | (i) The 22 prompts are `benchmark_test_prompts.json`, a **different 22-prompt set** from the 22 suppressed facts of Appendix B.3, and **9 of its 22 were already at rank 1** in the clean model (rank improvement is 0 for them by construction). (ii) 4.36 and 0.0484 are means over all 22, including those 9 (file `layer_benchmark_20260809_101619_929947.json`: mute 0.5, boost 0.7, batch 4, safety on, last token). On the 13 prompts not already correct, layer 8 has mean gain 7.38, improved the rank on 11 of 13, and **4 of 13 reached rank 1**; over all 22, 13 had final rank 1 at layer 8. (iii) "100%" is the benchmark's own `success` flag, which counts a rank OR probability improvement (see RECONSTRUCTION item 8); layer 8 improved the rank on 11 of 22. (iv) "No prompt reached rank 1" is false. (v) The better-documented run is the **21-prompt, 12-layer** run of 9 September (mute 0.6, boost 0.7, 3 + 3, safety on, last token). On its **12 prompts not already correct**: mean rank gain at layer 8 is 7.33 (layers 7: 7.08, 6: 6.50, 9: 6.33), rank 1 on 3 of 12 (tied with layers 4, 5, 7, 9), improved on 9 of 12 (tied with layers 5 and 9). Say "a plateau around layers 6 to 9, layer 8 the highest mean but not clearly separated from its neighbours". | `benchmark_results/layer_benchmark_20260809_101619_929947.json`, `benchmark_results/layer_benchmark_20260909_144450_707979.json`, `benchmark_results/layer_benchmark_20260909_summary.csv`, `research/paper/draft_v1/main.tex` (Table "layers") |
| A2 | Section 3.2 and Section 5.1 | "The benchmark set is the 22 suppressed facts" for the layer sweep | Same problem as A1(i): two different 22-prompt sets. Name both: the 22 suppressed ROME facts (stage 1) and the layer-benchmark file. | `src/fact_bank.py`, `benchmark_test_prompts.json`, RECONSTRUCTION item 7 |
| A3 | Section 3.4.1 (l.344), FR2 (l.283), Algorithm 1 (l.479), Section 4.4.2 (l.535), Listing C.1 (l.942 to 953) | the SAE encodes and the edit applies to "the final-token residual" | The hook has **always** applied the delta at **every position** (`sae.encode(resid)` on the whole tensor, since its first implementation on 10 July 2026). Only candidate *screening* was last-token only, until 24 September; it is now a choice (default in the app: all prompt positions). Rewrite the equations for a sequence, fix the listing. | `src/hooks.py`, git history of `src/hooks.py`, Entry 19 section 10 |
| A4 | Section 5.2.1 (l.600) | "All three successes involved a **factual competitor, a different city**." | The three variant-B successes had initial top-1 tokens "Italy", "P", "May"; 8 of the 22 facts had top-1 " the" (none corrected). The factual against grammatical split is **not reproducible from the results table** (RECONSTRUCTION item 3). Say what the table shows: 4 successes in total, all at clean rank 2; 9 of 22 facts start at rank 2. | `outputs/stage_1/results.csv`, `RECONSTRUCTION.md` |
| A5 | Section 5.2.1 (l.580) and Table 4.1 row 1 (l.514) | the original success was "Eiffel Tower to Paris" | The result table lists Francesco Castellacci; the fact bank has no Eiffel Tower entry (RECONSTRUCTION item 2). Use the table. | `outputs/stage_1/results.csv`, `src/fact_bank.py` |
| A6 | Table 4.1 row 8 (l.521) and Section 5.2.6 (l.736) | "1.89x from batched ranking" | **No artifact in the repo supports 1.89x.** What is recorded: Entry 14, 126 (122 after cleanup) to **6 model forward calls**, same 126 sequences evaluated, outputs identical (60 of 60 safety decisions, 0 misorderings on 6 prompts); Entry 15, whole-workflow timings 12.16 / 0.90 s, 11.55 / 0.71 s, 11.75 / 0.58 s (hardware not stated). Use those, or drop the figure. | `docs/Research_Journal/14.md`, `15.md` |
| A7 | Table 5.7 stage-timing profile (l.750 to 762), Section 5.3 | Layer 8 cached: clean pass ~150 ms, selection ~350 ms, **safety ~1,200 ms**, total ~1.88 s | Matches no stored profile. The stored layer-8 means are: Aug 9 run, selection 3.0 s, safety 5.5 s, total 8.6 s; Sep 9 run, selection 3.1 s, safety 3.1 s, total 6.4 s. The source (Entry 11) is "journal only". Replace with a profile from a stored JSON, name the hardware, and reconcile with Table 5.8. | `benchmark_results/layer_benchmark_20260809_*`, `..._20260909_144450_*` |
| A8 | Listing C.3 (l.971 to 980) | `classify_fact(model, prompt, target, max_rank, min_prob)` with `rank <= max_rank and probs[tid] >= min_prob` | Real signature `classify_fact(model, prompt, target_str)`; suppressed if **probability > 0.01 OR rank <= 50**. Listing C.2 signature also differs from the code (`get_top_competitor_features(model, sae, prompt, current_top_token_id, top_n, clean_ctx, use_batched, exclude_ids, base_scale_map, positions)`). | `src/evaluation.py`, `src/editing.py` |

### B. Withdrawn or must be reworded

| # | Where | Old claim | What to do | Source |
|---|---|---|---|---|
| B1 | Abstract (l.143), Contribution 1 (l.203), Table 3.1 (l.312 to 314), l.324, Section 4.4.4 (l.543), l.167 | Three classes "already correct / suppressed / **absent**"; absent = "knowledge not present"; "the knowledge has not been lost; it is present in the model's internal computation" | **[WITHDRAWN]** as a statement about the model (Entry 30 section 12, hypothesis H23). "Could not be steered" described the weak tool (multipliers only rescale features that are already active). A strong additive edit pushed random words to rank 1 on 20 of 20 prompts, so neither failure nor success of steering shows what the model holds. Keep the classes only as an **operational rule** (rank 1; rank > 1 with probability > 1% or rank <= 50; otherwise "low prior") and say so. Soften "knowledge is present" to "the answer already receives non-trivial probability". Journal 1 carries an update note. | `docs/Research_Journal/30.md` section 12, `1.md` update note, `hypothesis_log.md` H23 |
| B2 | Scope (l.196 to 198), conclusion (l.787) | "not claimed to make the model output the correct answer at rank 1 in general" | **[OUTDATED]**. Gradient descent reaches rank 1 on 271 of 300 held-out prompts. Add the caveats: rank 1 is a low bar (median target probability 7.7%, 79 of 271 below 5%); next-word measure only; same-prompt side effects partly by construction. | Entry 29 |
| B3 | Table 4.1 row 7, abstract (l.145), Table 5.4 | "Rank 8 to rank 1 at Mute 3 / Boost 3 ... adopted as production algorithm" | That sweep (strength 1.0) used the **unfiltered top candidates** (feature 313 among the mutes). With the safety filter on, the same sweep's best was rank 2 (8.59%) at 3 + 3; at the benchmark strengths 0.3 / 0.5 the layer-8 result is rank 3. State the filter status in every table that quotes these numbers. | `research/notebook/exp_007.md`, `research/paper/draft_v1/main.tex` Table "hybrid" |
| B3b | Section 5.2.2 to 5.2.3, Table 5.3, Figure 5.2 | competitor support "grammatical = concentrated, semantic = distributed" | One prompt, one run (the project's own hypothesis log says "supported by current evidence" only). Label as a one-prompt observation. | `research/hypothesis_log.md` H11 |
| B4 | Section 2.3 Research gap (l.262 to 272) | "rarely select directions by measured causal effect", "has not been benchmarked", "addresses the second, third, fourth and fifth gaps directly" | The project's own literature check (Entry 25) concludes: **nothing is conceptually new; possibly new as a combination and use case; search not exhaustive**. Arad et al. (2025) already select SAE features by output effect and get 2 to 3 times better steering, close to the project's selector. The paper draft deliberately makes **no novelty claims**. Reword the gap as "we found no study that ...", and drop "addresses gaps directly". | `docs/Research_Journal/25.md`, `docs/literature_review_sae_editing.md`, `EVIDENCE_AUDIT.md` ("Claims intentionally left out") |
| B5 | l.169, l.222, l.251 to 252 | Gupta et al. (2024): ROME and MEMIT degrade and collapse under sequential edits, so errors compound | **[UNVERIFIED]** in the project: the paper draft deliberately left Gupta and Arad numbers out because no source PDF was read. Read the paper before stating its findings as fact. | `RECONSTRUCTION.md` section 6 |
| B6 | l.234 | "This result [AxBench] is contested in subsequent work" | Journal 1 mentions "a 2026 rebuttal paper" but the paper was never identified in the repo. Find and read it, or remove the sentence. | `docs/Research_Journal/1.md` |
| B7 | Equation 3.5 (Section 3.4.4, weighted competitor muting) and Section 4.4.3 | presented as part of the method | Not in the application any more (code remains in `src/editing.py`, no UI). Move it to the development history. | `EVIDENCE_AUDIT.md` row 19 |

### C. Outdated content that needs updating, not deleting

| # | Where | What changed |
|---|---|---|
| C1 | Abstract, objectives, Section 3.3 (architecture), modules list (l.427 to 439), "two Streamlit applications" | `experiment_app.py` (7 tabs: Hybrid mute and boost, Monosemanticity Analysis, Session history and benchmarks, Sequential vs Batched Proof, Batch: Last vs All Tokens, Repair & SAE limit, **Prototype lab**) is the maintained app. `app.py` and `layer_benchmark.py` still exist but are the early versions. New modules: `hybrid_runner`, `batch_runner`, `batch_analysis`, `paper_pack`, `gradient_editing`, `repair_diagnostics`, `strength_cache`, `strength_models`, `feature_models`, `device_utils`, plus `tools/`. |
| C2 | Algorithm 2 (l.486 to 497), Table 4.1, Figure "algo" | The production sweep is now cumulative (one mute + one feature per step), keeps the best step, refills pools against the steered model, resolves features that are safe on both sides, and has a candidate-source option. The fixed "k_m = k_b = 3" description no longer matches the app. |
| C3 | Section 3.5 and DFD level 1 and Figure "algo" | Add the pool-refill loop and the gradient-descent path. |
| C4 | Table 5.1 metrics (l.560 to 574) | Add the metrics now used: same-prompt KL (all tokens except target and original top-1, renormalised), edit size (change norm / residual norm at the last position), KL and top-1 flips on 20 unrelated prompts, paraphrase and neighbourhood measures, distinct-2, loop rate, tail perplexity, instruction sensitivity; and the statistics (exact McNemar, Wilcoxon). |
| C5 | Operating environment (l.441 to 463) | Add macOS / Apple Silicon (MPS) and CPU; torch versions seen: 2.6.0+cu124 (GTX 1660 Ti PC), 2.5.1+cu121 (RTX 4050 laptop), 2.13.0 (Mac venv, Python 3.12.14); Python 3.13.7 on the PC; branch `gradient-descent-editing` (not `layer-intervention-benchmark`); transformers 5.13.0, transformer-lens 3.5.1, sae-lens 6.45.3, streamlit 1.59.1. **Check first whether `tools/check_mac_parity.py` has been run on the Mac** before claiming MPS support. |
| C6 | Future work (l.791 to 801) | Done since: full 12-layer sweep (21 prompts), batching of safety passes (6 forward calls, on by default in the app), larger fact sets (CounterFact), graded safety filters (implemented and studied: relaxed rules add at most one prompt). Still open: ROME / MEMIT / IKE / DiffMean comparison, AtP*, Gemma Scope, multi-layer (Entry 22 says repair is not a reason for it), dynamic layer choice. New: the roadmap A, C, D, B. |
| C7 | NFR "Portability" (l.297) | CUDA, MPS, CPU; `FEATURESCALPEL_DEVICE` override; timings are never comparable across machines. |
| C8 | Section 4.3 last paragraph (l.525): AI coding agent | Keep a disclosure, but update it (Journal 14 names "Antigravity AI"; later entries name Claude). Wording is yours and your college's policy. |

### D. Front matter and format errors (fix regardless)
- **Certificate (l.90):** all four KTU registration numbers read B23CS2109. The cover gives 2109, 2113, 2114, 2125.
- **Certificate:** "carried out by **him**" (four authors); guide is a placeholder `[Project Guide Name]` (the paper draft names Ms. Poorna B R: confirm); date blank.
- **Declaration (l.115):** "original work carried out by **me**" with four authors.
- **Cover:** "September 2026"; update the month. Align the title with the paper title ("Transient Activation Steering with a Pretrained Sparse Autoencoder for Suppressed Factual Completions in GPT-2 Small") or keep the project name, but be consistent; "Causal Feature Steering (CFS)" appears once (l.173) and nowhere else.
- **Hard-coded numbers:** the text says "Table 2.1", "Figure 3.1", "Equation 3.2", "Chapter 5" by hand. Adding content will break them. Use `\ref` for every one.
- Figures and tables lists are generated; after adding figures check the captions are self-contained.

### E. Still fine, keep
Chapter 2 background on ROME, MEMIT, ActAdd, CAA, RepE, superposition, SAEs (Section 2.1.1 to 2.1.3); equations 3.1 to 3.4; delta-patching explanation (add: edits all positions); two-stage selection idea and the 31 against 24,576 pass count (99.87% fewer); "one-bounded-step" design rule; Table 5.3 numbers (match Entry 7); Table 5.5 MIT layer numbers (match Entry 10; stored file differs by at most 0.01 percentage points); Table 5.6 and Table 5.8 numbers (match Entry 13); Appendix B.2 (SAE availability) and B.3 (per-fact stage-1 table, matches Entry 3 row by row); GPU against CPU 150.07 s against 574.60 s (3.83 times; hardware GTX 1660 Ti; 1,483 MB peak).

---

## Part 3. Section by section: what to edit, from the first page to the last

### Cover, certificate, declaration, acknowledgement
- Fix everything in Part 2 D.
- Acknowledgement: add what the new work used: **CounterFact (ROME authors), Neuronpedia, Hugging Face, Groq (explanations), the two machines**; the guide. Thank the AI tools only if your college policy expects it.

### Abstract (rewrite; l.141 to 147)
Keep the first two sentences (problem, why weights are not edited). Then in order: (1) transient edit through SAE features of GPT-2 small, layer 8; (2) the Hybrid sweep and what was learned about it (pool refill, all-position candidates: 10 of 13 against 4 of 13; the strict filter: removing it lost prompts, relaxing it added at most one); (3) CounterFact benchmark, 300 held-out prompts; per-feature gradient descent **271 of 300 against 161 and 94** with lower same-prompt side effects and about 5 to 6.5 times faster; (4) honest limits: rank 1 is a low bar (median probability 7.7%), additive control (20 of 20 random words), withdrawal of the "absent" inference, generation tests (mild repetition effect, ignores "do not say X"); (5) no comparison with ROME / IKE yet. Remove: the 22-fact numbers as headline, 4.36, "inverted-U ... best on every metric", "two Streamlit applications".

### Chapter 1 Introduction
- **1.1 Background (l.167 to 173):** keep the MIT example. Soften "knowledge present". Add RAG citation (Lewis et al., see Part 6). Add one paragraph on why a transient edit matters and on the new question: *can the edit be found automatically, and is rank 1 enough?*
- **1.2 Problem statement (l.177 to 181):** make the research questions explicit, e.g. RQ1 Can a transient SAE-feature edit lift a suppressed answer to rank 1, and with which settings? RQ2 Where does the supply of features limit it (candidate source, layer)? RQ3 Can the strengths be chosen automatically (learned strengths, gradient descent)? RQ4 What does the edit do beyond the next word (same-prompt side effects, unrelated prompts, reworded prompts, generated text)? RQ5 At what compute cost, on which hardware?
- **1.3 Objectives (l.185 to 192):** keep 1 to 6 as the early objectives, relabel the classifier as an "operational eligibility rule", and add: build a CounterFact benchmark with a subject-safe split; implement per-feature gradient-descent editing; compare editors under one yardstick; measure specificity beyond the next word; tooling for reproducible studies (batch engine, packs); cross-platform support.
- **1.4 Scope (l.196 to 198):** one model (GPT-2 small), one SAE release (`gpt2-small-res-jb`, 2024), mostly layer 8; the evaluation sets (22 ROME facts; 17-prompt candidate-source set; 45 and 131 safety-filter prompts; CounterFact 16,360 hard set with 300 test prompts); single-token targets only (multi-token targets are scored on the last piece, flagged in the app); success = rank 1 on the next token; no comparison with ROME / MEMIT yet.
- **1.5 Contributions (l.202 to 209):** replace with a list that matches the evidence, for example: (1) a transient edit pipeline with documented design decisions and ablations (joint hook, best-so-far, pool refill, all-position candidates, strict filter); (2) a CounterFact-based benchmark pipeline with a safe split; (3) an equal-budget comparison in which per-feature gradient descent beats the sweep; (4) negative or partial results kept on record: learned strengths, the additive edit; (5) measurement lessons: rank 1 is not evidence, rank-matched controls are needed, reachability by steering is not evidence about knowledge; (6) the engineering: batched evaluation, a thread-safety bug and its fix, cross-platform support. Keep the 4.55% to 13.64% result as a development-stage finding, not a contribution. Drop "measured applicability boundary".
- Add **1.6 Organisation of the report.**

### Chapter 2 Literature review
Detailed in Part 6. Edits: restructure into the themes there; update Table 2.1 (add rows: IKE, DiffMean, SAE-TS, SpARE, SAE unlearning, per-feature optimisation = this work); rewrite 2.3 "Research gap" (B4); add a short paragraph that the project's own literature check concluded the field is established and the evidence on SAE steering is mixed.

### Chapter 3 System analysis and design
- **3.1 Requirements:** keep FR1 to FR7, relabel FR1 (operational rule). Add FR8 (gradient-descent editing), FR9 (batch studies with resume and sharding, paired statistics), FR10 (side-effect measures), FR11 (generation view and follow-up leak test). NFR: add correctness checks (equivalence to 1e-9 / 1e-5 against reference paths, parity checks), reproducibility (seeded splits, packs with checksums), portability (Part 2 C7), thread safety (the BOS bug).
- **3.2 Dataset and eligibility:** two datasets. (a) ROME `known_1000` facts: 45 pairs (15 / 22 / 8; the old text says 46 scanned: the code and the counts give 45); (b) **CounterFact pipeline**: 21,919 records, 21,913 usable, 1,817 already rank 1, 3,736 above rank 1000, **16,360 kept** (rank 2 to 1000, single-token true answer, 34 relations, 15,594 subjects); split by subject, five relations held out (P140, P641, P1303, P176, P190), train 1,000 / validation 150 / test-seen 150 / test-unseen 150; per-prompt cache and strength tables (1,450 each). Table 3.1 relabelled; Figure 3.1 caption.
- **3.3 Architecture:** redraw (see Part 5): data pipeline, hooks, selection, two editors, batch engine, packs, app tabs, device layer. The paper draft has a TikZ version to adapt (`research/paper/draft_v1/main.tex`, figure `fig:arch`).
- **3.4 Design methodology:** keep the hypothesis-driven cycle and the one-bounded-step rule; add the **evidence discipline** that the project developed (journal entries with limits, data packs with checksums, an evidence audit, corrections recorded).
- **3.4.1 SAE encoding:** write for a sequence; the SAE release uses `apply_b_dec_to_input=True` (the paper draft has the exact formula).
- **3.4.2 Delta patching:** `x' = x + sum_i (s_i - 1) f_i w_i` at every position; mute `s = 1 - mu`, boost `s = 1 + beta` (the old text says boosting "increases by a positive amount": it is multiplicative).
- **3.4.3 and 3.4.4:** causal effect and decay ratio, weighted muting: move to "development history" or keep short.
- **NEW 3.4.5 Candidate selection and the safety filter:** candidate source (last token or all positions, scored by maximum activation, top-N capped by the number of active features); the strict filter (rank not worse, probability drop at most 1e-6); tolerance and graded variants (batch harness only); the combination check (new blockers).
- **NEW 3.4.6 Cumulative sweep, best-so-far, overlap rule, pool refill.**
- **NEW 3.4.7 Gradient-descent editor:** parameterisation `m_k = 1 + a_k`, `a_k = -1 + 3 sigmoid(z_k)`, start at no edit; loss = rank hinge (margin 0.3) + 3.0 x KL once the target leads + 0.005 x sum |a_k| w_k; Adam, lr 0.1, 100 steps; keep the best iterate; check through the real hook. Settings table (steps 100, lr 0.1, margin 0.3, side-effect weight 3.0, edit-size weight 0.005, Top N 200, candidate source all positions). Exact text: Entry 28 sections 2 and 3, and `docs/session_2026-10-06_1623.md` section 6 (a draft write-up you wrote with the assistant).
- **NEW 3.4.8 Additive extension (exploratory):** silent features at the last position, first-order score, cap = cap factor x loudest feature, sparsity weight; present as an experiment that failed its control.
- **3.5 Flow diagrams:** update DFD level 1 (eligibility, clean baseline, screening, ranking, filtering, joint intervention / optimisation, refill loop, metrics, packs) and the control-flow figure.
- **3.6 Modules:** extend the list (C1).
- **3.7 Operating environment:** C5.

### Chapter 4 Implementation
- **4.1 Technologies:** add Hugging Face Hub, Neuronpedia API, Groq (optional explanations; model migrated, Entry 18), matplotlib; PyTorch device backends.
- **4.2 Algorithms:** Algorithm 1 (two-stage selection: N is 120 or 200 in the studies, 30 in the layer benchmark; stage 1 on all positions or the last token); Algorithm 2 rewritten (cumulative sweep, best-so-far, pool refill); Algorithm 3 (layer sweep, unchanged); **new Algorithm 4** (gradient-descent editor); **new Algorithm 5** (additive edit, optional).
- **4.3 Development process:** extend Table 4.1 with rows 9 onwards. Suggested rows (date, change, outcome, status in the current system): best-so-far tracking (Entry 16); cumulative sweep (17); joint hook replaces chained hooks, rank 66 against 59 on one case (18); pool refill (19: a run stopped at rank 5 with 24 features per side; 67 active features at the last token); all-positions candidates and Top-N cap (19, 20: 10 of 13 against 4 of 13); batch engine (20); strict / tolerance / graded filter study (21); repair diagnostics (22: not supported); learned strengths (23 to 27: better global pair, little per-prompt gain); per-feature gradient descent (28, 29); additive edit (30: failed control); critique tests (30 section 15). Keep the AI-assistance paragraph (C8).
- **4.4 Module-wise implementation:** add gradient editing, batch runner and packs, Prototype lab, device layer, tools list.
- **NEW 4.5 Correctness checks:** equivalence tests (batched against sequential to 1e-9; cache path against real hook to ~1e-5; `tools/check_additive.py`, 5 checks; `tools/check_mac_parity.py`); built-in sanity checks that stop a run on mismatch; the two bugs the checks caught (strength-cache scale, additive tiny start).
- **NEW 4.6 Engineering lessons:** the BOS thread race (`prepend_bos=False` temporarily changes shared model config; 16 of 57 recorded app runs affected; fix `tokens_without_bos` and a per-run self-heal; no table affected) (Entry 30 section 13); the Neuronpedia lookup that inflated a timer (Entry 15); the chained hooks mismatch (Entry 18).

### Chapter 5 Results and discussion
Restructure as in 1.5. Per section: what to keep, what to add, where the facts are (Part 4).
- **5.1 Setup:** model, SAE, hardware (GTX 1660 Ti for the 300-prompt comparison, the filter studies; RTX 4050 laptop for caching, tables, headroom, training), datasets, metrics (C4), statistics.
- **5.2 Development-stage findings (old 5.2.1 to 5.2.4):** keep, shorter, each labelled "one prompt, one run" where true. Fix A4, A5, B3, B3b. Table 5.4 caption must say "unfiltered candidates".
- **5.3 Depth:** replace the 22-prompt claim (A1) with the 12-layer table and the MIT four-layer table; add Entry 22 (layer 7 the only SAE success on three hard prompts; Hase et al. caution about localisation against editing); conclusion "plateau, layer 8 a reasonable default".
- **5.4 Candidate source, 5.5 Safety filter:** new (Part 4).
- **5.6 CounterFact data and headroom, 5.7 learned strengths:** new.
- **5.8 Gradient descent against the sweep:** new, the main result, with Figures from Entry 29.
- **5.9 Repair diagnostics:** new (short).
- **5.10 Additive edit and controls, 5.11 generation and instruction tests:** new.
- **5.12 Compute and platforms:** fix A6, A7; add 6 forward calls, gradient descent timing, MPS/CPU note.
- **5.13 Discussion:** supply of features is the binding factor for multiplicative edits; rank against probability; what the edit is (a narrow push toward the answer's output direction); what is not shown (list in Part 4, section M). Merge the old "Analysis" paragraphs (selection direction, competitor type, softmax coupling, depth) and mark each as one-prompt or small-sample.
- **Threats to validity** (own subsection): proxies, single model / layer / SAE, prompt-set construction, rank 1 as a low bar, no equal-damage comparison with SAE-free edits, no rank-matched control, one run per condition, uncorrected multiple comparisons, timing on one machine only.

### Chapter 6 Conclusion and future work
Rewrite the conclusion to four or five findings, each with its caveat. Future work in priority order from `docs/research_roadmap.md`: A specificity benchmark (reach, transfer, leak, text quality, instruction sensitivity; true against random against **rank-matched** targets); C paraphrase and neighbour terms in the loss; D in-context fact vector; B IKE / ROME / DiffMean on the same prompts. Then: stop-early option, follow-up control, multi-token targets, FeatureNet full run, a steering-controller front end, larger models with Gemma Scope, AtP*.

### References
See Part 6. Also fix the AxBench arXiv number (the repo has two: `research/bibliography.bib` uses 2501.07727, the literature review and paper draft use 2501.17148: check on arXiv). `research/bibliography.bib` holds only 5 entries and cannot serve as the source.

### Appendices
- **A (UI layouts):** the two schematics describe `app.py` and `layer_benchmark.py`. Replace by screenshots of `experiment_app.py`: Hybrid tab (with the Editing-method choice), Prototype lab (result, baseline against edited text, follow-up box), Batch tab, Repair & SAE limit, Session history.
- **B.1 artifact schema:** still valid for the layer benchmark (v0.3). Add the batch-result and pack formats if space allows (`docs/Research_Journal/packs/*/MANIFEST.json`, `runs.jsonl`).
- **B.2 SAE availability:** keep. **B.3 per-fact table:** keep (verified against Entry 3); add a note that this is the stage-1 set.
- **C code listings:** replace C.1 by `make_scale_map_hook` (all positions); fix C.2 and C.3 (A8); add C.4 the gradient-descent loop (`run_gradient_descent_edit`, `src/gradient_editing.py`).
- **NEW D reproduction commands:** from `docs/session_report_2026-09-30_to_10-02.md` section 10, Entry 29 section 8, and `README.md` (running on Windows and macOS).
- **NEW E evidence table:** claim, artifact, status, in the style of `research/paper/draft_v1/EVIDENCE_AUDIT.md`. It protects you in a viva.

---

## Part 4. Fact sheet for the new material (numbers with their sources)

Every number below was taken from the file named; "real" = measured on the real model through the real hook; "proxy" = a stand-in measured on cached states.

### A. Stage 1 and the Cambridge case (development stage) [OK, with A4, A5, B3 fixes]
- Variant A (ablate the target's helpers): **1 of 22** (4.55%), 3 rounds, specificity 100%. Variant B (ablate the competitor's drivers): **3 of 22** (13.64%), 1 round each, specificity 100% on 15 controls. All four successes at clean rank 2; 9 of 22 facts start at rank 2. (Entries 2, 3; `outputs/stage_1/results.csv`; paper draft Table "stage1".)
- A grid of single-shot joint ablations (3 strengths x 4 batch sizes): 2 to 4 of 22 per setting, 5 distinct facts ever, all at rank 2 (`outputs/stage_1/grid_sweep_results.csv`; no journal entry).
- Cambridge (MIT prompt): clean rank 8, 1.15%, seven competitors; features 313 (the, a, an) and 21169 (question, jeopardy, danger, doubt); max-rule strengths 0.284 / 0.119, rank 8, 1.29%; sum rule 0.492 / 0.208, rank 8, 1.39% (H10 rejected); top-K: K = 3 gives rank 6 (1.53%), K = 5 rank 7 (Entries 4 to 8).
- Hybrid sweep: unfiltered candidates, strength 1.0: 3 + 3 gives rank 1 (7.33%), 3 + 4 gives rank 2 (6.74%, feature 24181 polysemantic; isolated rerun identical); filtered candidates: best rank 2 (8.59%) (`research/notebook/exp_007.md`).

### B. Depth
- MIT, layers 2 / 5 / 8 / 10, mute 0.3, boost 0.5, 3 + 3: rank 8 to 7 / 5 / 3 / 5, gains +0.07 / 0.78 / 2.38 / 0.81 percentage points (`layer_benchmark_20260909_143704_240449.json`).
- 21 prompts, 12 layers, mute 0.6, boost 0.7, 3 + 3, last token, safety on; **12 prompts not already correct**: mean rank gain by layer 0 to 11: 0.08, 1.83, 1.17, 2.25, 3.75, 2.50, 6.50, 7.08, **7.33**, 6.33, 4.75, 2.67; reached rank 1: 1, 0, 1, 2, 3, 3, 2, 3, 3, 3, 1, 2 (`benchmark_results/layer_benchmark_20260909_summary.csv`).
- Entry 22 (three hard prompts, layers 5 to 11, Top-N 120): layer 7 the only SAE success (Colosseum to Rome); scope warning: hand-picked failures, not a success rate.

### C. Compute
- Forward calls per prompt and layer: 186 (reference) to 126 (clean baseline passed to the safety filters; safety stage 3,863 to 1,953 ms; total 6.38 to 4.39 s, RTX 4050) to 122 (clean-pass caching) to **6** (batched candidates; sequence evaluations unchanged at 126; 60 of 60 safety decisions identical; 0 misorderings on 6 prompts) (Entries 13, 14). Note Entry 14's header says 126 to 6 but its table says 122 to 6.
- GPU against CPU: 150.07 s against 574.60 s (3.83 times) for 3 prompts x 4 layers (about 960 passes), GTX 1660 Ti; peak 1,483 MB (GPT-2 478 MB, 96 MB per SAE) (Entry 12). The "9.8 to 20.7 times warm" figure is a single-layer script run; the paper draft left it out.
- App workflow, sequential against batched: 12.16 / 0.90 s, 11.55 / 0.71 s, 11.75 / 0.58 s (hardware not stated); Tab 4 about 10.1 s against 1.0 s (Entry 15). A timer originally included Neuronpedia lookups (46 s): fixed.
- Gradient descent: 100 steps, about 5 to 9 s per prompt (mean 7.7 s) against 38.5 s and 49.7 s for the sweeps (Entry 29).

### D. Candidate source (real; 17-prompt set, Top-N 120, mute 0.6, boost 0.5, refill on) (Entry 20)
13 comparable prompts (3 already correct, 1 two-token target excluded): all positions **10 of 13** reached rank 1, last token **4 of 13**; all-positions won 9, tied 4, lost 0 (sign test p about 0.004; prompts not independent, chosen by the authors); higher final probability on 12 of 13; mean round-0 candidates 281 against 56; features edited 94 against 42; time 12.2 s against 5.4 s; **all 9 last-token failures ended because no unapplied active feature remained**. The 34-run batch reproduced identically from the UI and the command line. Known issues listed in Entry 20 section 7 (multi-token targets, stale blocker within a round, explainer bug).

### E. Safety-filter study (real; mute 0.6, boost 0.5, Top-N 120, refill on, 250-step cap) (packs `entry21_*`; numbers verified in `EVIDENCE_AUDIT.md`)
| Study | strict | no filter | tolerance 5% | graded (after) | graded (interleaved) |
|---|---|---|---|---|---|
| Pilot, all positions, 45 prompts | **29** (64%) | 19 | 27 | 29 | 29 |
| Full, last token, 131 prompts | **41** (31%) | 35 | 42 | 41 | 41 |
Removing the filter lost 10 prompts and gained none in the pilot (McNemar p = 0.002); relaxing it added at most one prompt. The strict filter accepted about 50% of round-0 candidates; only 2.9% (pilot) and 2.5% (full) of candidate features were unusable on both sides, because a feature rejected as a mute is usually accepted as a boost. Collateral KL (20 neutral prompts): 0.0055 strict, 0.0069 no filter (pilot). **Entry 21 itself is an unwritten placeholder**; the interpretation is in the paper draft and the packs.

### F. Repair and SAE-limit diagnostics (Entry 22; three hand-picked failed prompts plus MIT; scope-limited)
Held edit gave the same final rank as the plain edit in **21 of 21** layer/prompt cells; the pushed direction persisted at 0.79 to 1.29 of its size at the output (H13 not supported, direction-level only); an SAE-free residual edit of the same size reached rank 1 in 21 of 21 cells against 1 of 21 for the SAE edit (suggestive, not a success rate); SAE-free edits needed 5 to 10% of the residual norm on all tokens, SAE edits pushed 8 to 28%; reconstruction error 12 to 26% of the activation norm, left unedited.

### G. CounterFact data and headroom (Entries 23, 24; headroom = proxy)
Counts in Part 3 (3.2). Headroom, one-shot edits on saved layer-8 states, 100 steps, 1,450 prompts: free per-feature multipliers (0 to 3) with no damage limit reach rank 1 on **80%**; with a side-effect penalty (one setting) **41%**; crude fixed proxies 6 to 15%; free residual push last token 5 / 10 / 20% of norm: 18 / 47 / 99%; all tokens 5 / 10 / 20%: 58 / 99 / 100%. Same on train, validation, test. Starting-rank band 101 to 1000: only 51% reachable without a damage limit. The 41% depends on arbitrary weights.

### H. Learned strengths (proxy only; the real sweep was never run with them) (Entries 26, 27)
300 test prompts: fixed 0.6 / 0.5 **75 (25.0%)**; learned fixed pair (mute 0.98, boost 1.81) **116 (38.7%)**; PromptNet 119 (39.7%, 3 more than the pair, 0 fewer, p = 0.25); per-prompt tuned oracle 123 (41.0%). Pair against reference: 42 against 1 (p < 0.0001). Side effects of successes roughly double (KL 0.156 to 0.38). Learning curve flat from 100 training prompts. FeatureNet (development runs): 31 to 32% against about 37% for the pair; free per-feature optimum 73.3%. Finding: **a better global default, not per-prompt adaptation**.

### I. Gradient descent against the sweep (real; 300 held-out prompts; GTX 1660 Ti) (Entry 29; `packs/sweep_vs_gradient/`)
| Arm | reached rank 1 | median same-prompt KL | features changed | edit size | neutral KL | time |
|---|---|---|---|---|---|---|
| sweep 0.6 / 0.5 | 94 / 300 (31.3%) | 0.259 | 247 | 0.176 | 0.0045 | 49.7 s |
| sweep 1.0 / 2.0 (same 0 to 3 range) | 161 / 300 (53.7%) | 0.600 | 182 | 0.408 | 0.0292 | 38.5 s |
| **gradient descent** | **271 / 300 (90.3%)** | **0.144** | 193 | 0.217 | 0.0088 | **7.7 s** |
Paired: gradient descent solved **110** prompts the wide sweep did not, the wide sweep none that gradient descent did (p = 2e-33); on prompts both solved, gradient-descent KL was lower on 160 of 161. By starting-rank band (rank 1 / n): 2-5: 51 / 63 / 67 of 67; 6-20: 31 / 54 / 76 of 76; 21-100: 10 / 31 / 74 of 78; 101-1000: 2 / 13 / 54 of 79. **Caveats to print next to it:** the loss contains a KL term, so part of the KL advantage is by construction; median target probability at rank 1 only 7.7% (4.9% in the hardest band); next-word at the last position only; one run, one machine; other sweep settings were not run. Hand trials (Entry 28 section 6) are exploratory.

### J. Additive edit and controls (Entry 30 sections 2, 6, 10, 11, 12)
Hand trials (six failing prompts): additive reached rank 1 on all six, at edits of 55% to 161% of the residual norm (free pushes in Entry 24 were 5%). **Control** (20 validation prompts x random unrelated words): multipliers only 1 of 20 (edit 39% of norm); additive 20 of 20 (192% at cap 1.0; 126% at cap 0.25). **Reworded-prompt test** (40 prompts, 80 paraphrases): additive carries over equally for random and true targets (median reworded rank 4 and 9 for random; 4 and 12 for true); neutral-prompt top-1 flips 82% (random, cap 1.0) against 24% (true); multiplier edit: true targets median reworded rank 94 to 36, rank 1 on 9%; random words 17,504 to 11,832, rank 1 on 0% (not rank-matched). Conclusion in the journal: **do not present the additive edit as a fix**; it injects an answer, not a fact.

### K. Generation and instruction tests (Entry 30 sections 14, 15; 40 prompts, true targets, 20 greedy tokens)
No edit: distinct-2 0.93, loops 0%, tail perplexity 4.7, target in 12% of continuations. Edited (kept on): distinct-2 0.85 (gd), 0.86 / 0.89 (additive cap 1.0 / 0.25); loops 12% / 8% / 2%; perplexity 4.4 / 4.8 / 4.8; target present in 90% / 100% / 100%, repeated about 0.6 to 0.7 times per continuation. **"Do not say the word X"**: unedited model already says X first in 45% (naming it primes copying); with the edit 68% (gd), 90% / 88% (additive). "X is the wrong answer": 30% unedited; 68%, 88%, 85% edited. n = 40, no intervals, no random control. Hand check: with the additive edit kept on, "The capital of Germany is" continued with "Hong Kong".

### L. What is NOT shown (print this list in the report)
No comparison with ROME, MEMIT, IKE, DiffMean or prompting (EXP-015); no real-sweep run with the learned strengths; FeatureNet full run not done; no rank-matched random-target control; the all-positions 131-prompt filter study not run; no confidence intervals on the 40-prompt tests; one model, one SAE release (2024), mostly layer 8; multi-token targets scored on the last piece only; nothing measured on a larger model; the Mac (MPS) path not confirmed unless you have run the parity check.

---

## Part 5. Figures and tables

**Old figures** (`elig`, `arch`, `delta`, `dfd0`, `dfd1`, `algo`, `strategy`, `decay`, `topk`, `hybrid`, `layer`, `passes`, `gpu`, `ui_app`, `ui_bench`, `mbcet_logo`) are not in the repository; you have the files. Status:
- Keep: `delta` (add "all positions" in caption), `strategy`, `decay`, `topk`, `hybrid` (add "unfiltered candidates"), `layer` (MIT), `passes` (add the batched bar), `gpu`, `dfd0`, `elig` (relabel "absent").
- Redraw: `arch`, `dfd1`, `algo` (pool refill, gradient descent path), `ui_app`, `ui_bench` (replace by screenshots).

**Existing figures you can reuse (all drawn from real stored data)**
| What | File | Format |
|---|---|---|
| Gradient descent against the sweep: rank 1 by band, paired KL, success against KL, time | `docs/Research_Journal/images/e29_fig1_rank1_by_band`, `e29_fig2_kl_paired`, `e29_fig3_success_vs_kl`, `e29_fig4_time` | PNG + SVG; plotted numbers in `e29_figure_data.json` |
| Hand trials, rank by step | `docs/Research_Journal/images/e28_fig1_rank_by_step` | PNG + SVG |
| Candidate-source study (worked example, controls, pools, cost, headline) | `docs/Research_Journal/images/e20_fig1` to `e20_fig10` | **SVG only**: convert to PNG or PDF for pdflatex |
| Safety-filter studies (success, features, collateral, time, difficulty, scatter, filter reasons) | `docs/Research_Journal/packs/entry21_pilot_*/figures/`, `entry21_full_last_*/figures/` (`Fig01` to `Fig10`) | PDF, PNG, SVG |
| Layer plots (MIT) | `docs/Research_Journal/images/probability_gain_vs_layer.png`, `rank_improvement_vs_layer.png`, `runtime_vs_layer.png` | PNG |
| 12-layer sweep, candidate source, filter (paper-style) | regenerate with `python tools/make_paper_figures.py` (output `research/paper/draft_v1/figures/`; the folder is empty in git) | PDF + PNG |

**New figures worth making (only from stored data; never invent values)**
1. Headroom by starting-rank band (`docs/Research_Journal/packs/headroom/headroom_20261002_144917.json`).
2. Learned-strength comparison: reference, pair, PromptNet, oracle (Entry 26 table; `packs/strength_models/`).
3. Additive control: rank-1 count and edit size for multipliers against additive (`packs/additive_control/results.json`).
4. Reworded-prompt transfer, true against random (`packs/generalisation/true40`, `random40`).
5. Text quality and instruction sensitivity (`packs/generation_quality/true40/summary.json`).
6. A diagram of the gradient-descent editor, and one of pool refill.
7. Repair diagnostics: persistence and rank by layer (`packs/repair_diagnostics/merged_all_layers.json`).
Use the `tools/make_journal_figures.py` pattern (reads only committed packs). Check LaTeX without a compiler using `tools/check_latex_static.py` (written for the paper draft).

**Tables to add or replace:** Table 2.1 rows; Table 3 (CounterFact counts and split); Table 4.1 rows 9+; the hyperparameter table of the gradient-descent editor; Part 4 E, H, I, J, K as result tables; the layer table (12 layers); a "what is not shown" table.

---

## Part 6. Literature: what is already in the project and what to add

You said the survey comes next; this is the starting inventory. **Nothing here has been read in full by the project** (`EVIDENCE_AUDIT.md`: most checks were abstracts; `literature_review_sae_editing.md` marks each source "fetched" or "snippet"). Read each paper before citing a number from it.

### 6.1 In the old report (r1 to r22): check status
All 22 look like real papers or tools. Verify before reuse: r4 (Gupta et al.: claims in B5), r9 (title changed between versions of the paper), r10, r11, r16 to r19, r20 (Neuronpedia citation form), r21. Add the publication year of GPT-2 (the audit asks for confirmation).

### 6.2 New sources the project used (from `docs/literature_review_sae_editing.md` and Entry 25)
| Topic | Source (as recorded in the repo) | Level checked |
|---|---|---|
| SAE research de-prioritised at Google DeepMind (dense probes beat SAE probes on harmful-intent detection out of distribution) | "Negative Results for SAEs on Downstream Tasks and Deprioritising SAE Research" (LessWrong post) | fetched |
| AxBench (prompting 0.894, SAE 0.165 on steering) | arXiv 2501.17148 | snippet |
| Selecting SAE features by output effect makes steering 2 to 3 times better | Arad, Mueller, Belinkov, "SAEs Are Good for Steering, If You Select the Right Features" (EMNLP 2025), arXiv 2505.20063 | snippet |
| SAEs on GPT-2 small struggle on factual disentanglement (RAVEL) | "Evaluating Open-Source Sparse Autoencoders on Disentangling Factual Knowledge in GPT-2 Small", arXiv 2409.04478 | fetched |
| Suppressed behaviour can be recovered after SAE intervention, attributed to the reconstruction residual | "SAE Interventions are Unreliable: Post-Intervention Recovery of Suppressed Behavior", arXiv 2606.18322 | fetched |
| SAE unlearning, conditional clamping | arXiv 2410.19278; 2503.11127 | snippet |
| SAE-targeted steering vectors | "Improving Steering Vectors by Targeting SAE Features", arXiv 2411.02193 | snippet |
| SAE knowledge-selection steering | SpARE, arXiv 2410.15999 | snippet |
| Localisation against editing | Hase et al., "Does Localization Inform Editing?", arXiv 2301.04213 | snippet |
| In-context knowledge editing | "Can We Edit Factual Knowledge by In-Context Learning?" (IKE), EMNLP 2023 | snippet |
| Editing toolkit | EasyEdit, arXiv 2308.07269 | snippet |
| Datasets | CounterFact and known_1000 (ROME); LAMA / T-REx; ParaRel | CounterFact inspected |
| Mentioned, unidentified | "SAFE, SALVE (2025)" (`docs/architecture.md`); "a 2026 rebuttal paper" on SAEs against LoRA (Journal 1) | **find or remove** |

### 6.3 Gaps the new report will need (suggestions from background knowledge; none is verified by the project: find the paper, check the claim, then cite)
- **Fact-editing evaluation and surveys:** metrics efficacy / paraphrase / neighbourhood (CounterFact, ROME paper); MEND, SERAC; "Editing Large Language Models: Problems, Methods, and Opportunities" (Yao et al.); ripple-effects evaluation (Cohen et al.); knowledge neurons and feed-forward key-value memories (Geva et al.; Dai et al.).
- **Retrieval-augmented generation:** Lewis et al. (the old report names RAG without a citation).
- **Text degeneration:** Holtzman et al., "The Curious Case of Neural Text Degeneration" (greedy decoding repetition: relevant to the generation tests).
- **Logit lens:** nostalgebraist (used in the Repair tab).
- **SAE variants and limits:** Gao et al. (TopK SAEs); gated and JumpReLU SAEs (Rajamanoharan et al.); feature absorption (Chanin et al.); Gemma Scope (already r19); SAE feature circuits (Marks et al., listed as "established" in the literature review without a source).
- **Steering:** refusal steering with SAEs (O'Brien et al.); representation fine-tuning ReFT / LoReFT (cited inside AxBench), relevant to optimisation-based editing; DAS (Geiger et al.; the "skyline" in the GPT-2 RAVEL study).
- **Unlearning:** RMU / WMDP (Li et al.), named inside the SAE unlearning paper.
- **Optimiser and statistics:** Adam (Kingma and Ba); McNemar test; Wilcoxon signed-rank (you use both).
- **Tools:** Hugging Face Transformers; Streamlit; Neuronpedia; TransformerLens; SAELens (already r13, r14, r20).

### 6.4 Bibliography file problems to fix before you build the list
`research/bibliography.bib` has 5 entries; the AxBench entry has arXiv 2501.07727 while the literature review and the paper draft use 2501.17148; the SAELens URL differs between the old report (`jbloomAus/SAELens`) and the paper draft (`decoderesearch/SAELens`).

### 6.5 Suggested survey structure
(1) Knowledge editing and its evaluation; (2) activation steering and its benchmarks; (3) SAEs and SAE steering; (4) negative and mixed evidence on SAEs; (5) causal localisation and attribution; (6) in-context and retrieval alternatives; (7) evaluation methodology (side effects, generation quality); (8) a comparison table with a column "needs target?", "persistent?", "side-effect measure", "interpretable?", and a short, carefully worded gap statement (Part 2 B4).

---

## Part 7. Decisions only you can make
1. **What is the "main system" of the report?** Recommendation: the Hybrid sweep as the original system and baseline; per-feature gradient descent as the main measured editor; the additive edit as a documented exploratory result that failed a control.
2. **How much of the history to keep.** Recommendation: keep stage 1 and the Cambridge case as a short development section; move formulas not used by the app (weighted reduction) to an appendix.
3. **Run the missing experiments before submission, or report them as future work?** The most valuable single run is the rank-matched control, then the random-target version of the generation test; the ROME / IKE comparison is larger. All are on the roadmap.
4. **Name and title.** "FeatureScalpel" against "Causal Feature Steering"; align the report title with the paper title.
5. **AI-assistance statement and authorship statements** (certificate, declaration, acknowledgement, Section 4.3): your college's policy decides wording.
6. **Entry 21 (safety-filter study) has no journal write-up**; decide whether the report's section is written from the packs and the paper draft (it can be) or whether you want the journal entry first.
7. **Whether to describe Mac / MPS support** (only after `tools/check_mac_parity.py` has been run there).
8. **Whether the stage-1 narrative conflicts (Eiffel Tower, factual against grammatical) should be told or silently corrected.** Recommendation: correct it silently in the main text and list it in the evidence table.

---

## Part 8. Suggested order of work
1. Fix Part 2 A and D (wrong numbers, front matter, `\ref`s): about half a day.
2. Rewrite abstract, Chapter 1, Chapter 6 last, once Chapter 5 is settled.
3. Chapter 5: write the new sections in the order D, E, G, H, I, J, K (Part 4), then fix the old sections (A1, A4, A6, A7).
4. Chapter 3 and 4 additions (gradient-descent editor, pool refill, batch engine, correctness checks).
5. Figures (Part 5), then Appendices.
6. Literature survey (Part 6), then rewrite Chapter 2 and the gap statement.
7. Build an evidence table as you go (Appendix E).
8. Static check: `python tools/check_latex_static.py <your .tex>`; then compile in Overleaf (no compiler on the project machines was ever used for the paper draft either).

---

## Appendix. Where each number lives

| Topic | File |
|---|---|
| Stage 1 | `outputs/stage_1/results.csv`, `report.md`, `grid_sweep_results.csv` (local), Journal 2, 3 |
| Cambridge case | Journal 4 to 8, `research/notebook/exp_004` to `exp_010` |
| Layer benchmark | `benchmark_results/*.json`, `layer_benchmark_20260909_summary.csv`, Journal 10, 11 |
| Compute | Journal 12 to 15, `scratch/*.py`, `src/batched_eval.py` |
| Candidate source | `benchmark_results/candidate_source_study/`, Journal 20 |
| Safety filter | `docs/Research_Journal/packs/entry21_*`, `research/paper/draft_v1/main.tex` |
| Repair diagnostics | `packs/repair_diagnostics/merged_all_layers.json`, Journal 22 |
| CounterFact, headroom | `data/counterfact_hard_set_summary.json`, `data/counterfact_split.json`, `packs/headroom/`, Journal 23, 24 |
| Learned strengths | `packs/strength_models/`, Journal 26, 27 |
| Gradient descent | `packs/gradient_descent/`, `packs/sweep_vs_gradient/`, Journal 28, 29 |
| Additive, controls, generation | `packs/additive_control/`, `packs/generalisation/`, `packs/generation_quality/`, Journal 30 |
| Literature | `docs/literature_review_sae_editing.md`, Journal 25, `research/paper/draft_v1/EVIDENCE_AUDIT.md` |
| Roadmap, handoff | `docs/research_roadmap.md`, `docs/handoff_2026-10-05_mac.md`, `docs/session_2026-10-06_1623.md` |
| Hypotheses, experiments | `research/hypothesis_log.md` (H1 to H23), `research/experiment_index.md` (EXP-001 to EXP-022) |
