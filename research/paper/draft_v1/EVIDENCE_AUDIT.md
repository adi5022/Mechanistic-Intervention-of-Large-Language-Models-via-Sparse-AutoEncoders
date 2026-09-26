# Evidence audit (deliverable C)

Every claim in `main.tex`, the artifact that supports it, and its status. **Verified** = checked against the raw artifact or source. **Journal only** = stated in a journal or notebook with printed numbers but no raw file in the repo. **Needs evidence** = not established; either flagged in the paper or excluded.

| # | Paper claim | Evidence source | Type | Status |
|---|---|---|---|---|
| **Related work and citations** |||||
| 1 | SAE architecture (linear encoder + ReLU, linear decoder, MSE + L1), features more interpretable than neurons (median rubric 12 vs 0), 79% loss recovered | Bricken et al. PDF `docs/Bricken...pdf`, pages 12 to 48 | Literature | Verified (text extracted) |
| 2 | Cunningham et al.: SAE features more interpretable / monosemantic than alternatives; localize causal components on IOI | arXiv 2309.08600 abstract | Literature | Verified (abstract only) |
| 3 | Superposition as a source of polysemanticity in toy models | arXiv 2209.10652 abstract | Literature | Verified (abstract only) |
| 4 | ROME locates recall in middle-layer feed-forward modules, edits weights, introduces a counterfactual dataset | arXiv 2202.05262 abstract (NeurIPS 2022) | Literature | Verified (abstract only) |
| 5 | AxBench: prompting best for steering, then finetuning; SAEs not competitive (Gemma-2-2B/9B) | arXiv 2501.17148 abstract | Literature | Verified (abstract only) |
| 6 | GPT-2 small: 117M params, 12 layers, d_model 768 | Radford et al. PDF, Table 2 | Literature | Verified; publication year needs confirmation |
| 7 | SAE release has one SAE per block, resid_pre | Bloom (2024) post; `Journal 9`; loaded SAE config | Literature / Implementation | Verified |
| 8 | SAE metadata: OpenWebText, context 128, BOS, d_sae 24,576, standard architecture, `apply_b_dec_to_input=True` | `sae.cfg` printed from the installed SAE; SAELens source `process_sae_in` | Implementation | Verified |
| 9 | Facts in the fact bank come from ROME `known_1000` | Comment in `src/fact_bank.py`; Journals 1, 12 | Repo statement | Needs evidence (dataset file itself not in repo) |
| **Method** |||||
| 10 | Delta patch equation, muting `1-s`, boosting `1+s`, edit at every position | `src/hooks.py`, `experiment_app` / `hybrid_runner.py` | Implementation | Verified |
| 11 | Joint hook replaced two chained hooks; rank 66 vs 59 on one case | Journal 18 section 4 (printed table) | Journal only | Journal only |
| 12 | Candidate screening (last token or all positions, max activation, top-N cap) | `src/editing.py: get_top_active_features` | Implementation | Verified |
| 13 | Mute candidates ordered by blocker probability drop; boost by target drop | `get_top_competitor_features`, `get_top_target_features` | Implementation | Verified |
| 14 | Strict safety rule: rank not worse and drop <= 1e-6 (absolute) | `check_target_safe*`, `check_boost_safe*` | Implementation | Verified |
| 15 | New-blocker definition | `check_combination_safe` | Implementation | Verified |
| 16 | Tolerance and graded variants (formulas, alpha guess, halving, alpha_min 0.05) | `src/hybrid_runner.py: assess_side` | Implementation | Verified |
| 17 | Cumulative sweep, best-so-far, overlap assignment, pool refill, stop reasons | `hybrid_runner.py`; Journals 16, 17, 19 | Implementation | Verified |
| 18 | Batched evaluation (chunks of 32, per-row hook, on-device rank) | `src/batched_eval.py`, `src/hooks.py` | Implementation | Verified |
| 19 | Weighted multi-competitor and multi-feature reduction formulas | `src/editing.py` (lines ~700 to 990) | Implementation | Verified; code default is the H10 sum variant |
| **Prompts and metrics** |||||
| 20 | Fact bank: 45 pairs, single-token targets; 15 / 8 / 22 by the classification rule | `src/fact_bank.py` + re-run of `classify_fact` | Experimental | Verified (journals say 46) |
| 21 | 21-prompt layer run: 9 at rank 1, 12 not; clean ranks 2 (x6), 8 (x2), 10, 16, 45, 84 | `layer_benchmark_20260909_144450_707979.json`, `tools/summarize_layer_benchmark.py` | Experimental | Verified |
| 22 | Safety-filter prompt set: 211 valid, 131 with rank 2 to 1000, 4 above 1000, 76 at rank 1; pilot 45 + 5, full 131 + 25 | `data/safety_filter_prompts_full.json`, spec files, `tools/build_prompt_set.py` | Experimental | Verified |
| 23 | Collateral KL definition (20 neutral prompts, top-1 flips) | `hybrid_runner._collateral`, `NEUTRAL_PROMPTS` | Implementation | Verified |
| 24 | Exact sign and McNemar tests, uncorrected | `src/batch_runner.py: build_arm_comparison` | Implementation | Verified |
| 25 | UI and CLI runs of the 34-run batch were identical | Journal 20 section 1 and 3; `benchmark_results/candidate_source_study/` | Experimental | Verified for that batch |
| 26 | Combination check disabled in the safety-filter studies | spec files (`combination_check: false`) | Implementation | Verified |
| **Results: stage 1** |||||
| 27 | Variant A 1/22, variant B 3/22; rounds; all four successes at clean rank 2; specificity 100% on 15 controls | `outputs/stage_1/results.csv`, `report.md`, `baseline_target_report.md` | Experimental | Verified |
| 28 | 9 of 22 facts start at rank 2; top-1 was "the" in 8 (0 successes); successes' top-1 tokens Italy, P, May | `results.csv` | Experimental | Verified (computed by us from the CSV) |
| 29 | Eiffel Tower narrative conflicts with the table | Entries 1, 2, baseline report vs table | Discrepancy | Flagged in paper |
| 30 | Grid: 2 to 4 of 22 per setting; 5 facts ever succeed, all rank 2 | `grid_sweep_results.csv` | Experimental | Verified; no journal entry |
| 31 | "Compound ablation has a collapse point at large batch" (H7) | Hypothesis log cites `conversations.json` | Journal only | Needs evidence; **not claimed in the paper** |
| 32 | "Hybrid steering fixed Dublin and Melbourne" (H8) | `exp_006.md` narrative only | Journal only | Needs evidence; **not claimed** |
| **Results: MIT prompt** |||||
| 33 | Clean rank 8, 1.15%; seven competitors and probabilities | Journals 4, 5, `exp_004`, `exp_007` | Journal only | Journal only (consistent across four entries) |
| 34 | Top feature per competitor: 313 (the, a, an), 21169 (other four) | Journals 4, 6, 7 | Journal only | Journal only |
| 35 | Max rule: strengths 0.284 / 0.119, rank 8, 1.29%; sum rule (0.492 / 0.208): 1.39%, rank 8; H10 rejected | Journal 6, `exp_008.md` | Journal only | Journal only |
| 36 | Decay ratios table; 18 shared features; 3.89 competitors per feature | Journal 7, `exp_009.md` | Journal only | Journal only |
| 37 | Top-K sweep table | Journal 8, `exp_010.md` | Journal only | Journal only |
| 38 | Hybrid sweep rows; 24181 raises "the"; clean-model rerun identical | Journal 5, `exp_007.md` | Journal only | Journal only |
| **Results: layers** |||||
| 39 | MIT layers 2/5/8/10: ranks 7/5/3/5, gains 0.07/0.78/2.38/0.81 pp, top-1 stays "the" | `layer_benchmark_20260909_143704_240449.json` | Experimental | Verified (journal plot labels differ by <= 0.01 pp) |
| 40 | 12-layer summary table and mean gains | `layer_benchmark_20260909_summary.csv` (from the 21-prompt artifact) | Experimental | Verified |
| **Results: candidate source** |||||
| 41 | 10/13 vs 4/13; 9 wins, 4 ties, 0 losses; sign p ~ 0.004; 12/13 higher probability | `batch_paired_summary.csv`, Journal 20 | Experimental | Verified |
| 42 | 281 vs 56 mean round-0 candidates; 94 vs 42 features; 12.2 s vs 5.4 s | Journal 20 section 6.1 (from the result JSON) | Experimental | Verified |
| 43 | All 9 last-token failures ended with no unapplied active feature | Journal 20 section 6.3 (from the result JSON) | Experimental | Verified |
| 44 | 67 vs 303 active features on the "leading cause of death" prompt; rank 152 to 50 / 51 | Journal 19 sections 6, 9, 10 | Journal only | Journal only |
| **Results: safety filter** |||||
| 45 | Pilot table (29/19/27/29/29 of 45; features, steps, time, KL) | pack `entry21_pilot_*/tables/arms_all.csv`, manifest | Experimental | Verified |
| 46 | Full last-token table (41/35/42/41/41 of 131) | pack `entry21_full_last_*/tables/arms_last.csv`, manifest | Experimental | Verified |
| 47 | Paired counts and p-values (both studies) | packs `vs_reference_*.csv` | Experimental | Verified |
| 48 | Rejection reasons and 2.9% / 2.5% unusable on both sides | packs `strict_filter_reasons.csv` | Experimental | Verified |
| 49 | Pilot stop-reason counts (83 no features left, 7 dead ends, 2 step limit, 133 reached, 25 controls) | `outputs/safety_batches/safety_filter_spec_pilot.json` (local, checksum in manifest) | Experimental | Verified (computed during the session; not stored in the pack) |
| 50 | "Relaxing the filter added at most one prompt; removing it lost prompts" | rows 45 to 47 | Experimental | Verified |
| **Results: compute** |||||
| 51 | 186 to 126 to 122 to 6 forward passes; equivalence (features, 1e-9, 0 misorderings, 60/60) | Journals 13, 14 | Journal only | Journal only (scripts referenced in `scratch/`) |
| 52 | Safety stage 3,863 to 1,953 ms; total 6.38 to 4.39 s (RTX 4050) | Journal 13 | Journal only | Journal only |
| 53 | Application timings 12.16/0.90, 11.55/0.71, 11.75/0.58 s | Journal 15 | Journal only | Journal only; hardware not stated |
| 54 | GPU 150.07 s vs CPU 574.60 s (3.83x); 1.48 GB peak | Journal 12 | Journal only | Journal only |
| 55 | Profile of a layer-8 run (about 350 ms selection, 1,200 ms safety, 1.88 s total) | Journal 11 | Journal only | Journal only |
| **Limitations and statements about absence** |||||
| 56 | No baseline comparison exists | Absence in repo; Journal 1 section 6 (no code for ROME / mean-difference) | Repo audit | Verified (absence) |
| 57 | No automated tests for editing/safety code | `tests/` contains only monosemanticity tests | Repo audit | Verified |
| 58 | Multi-token targets scored by last piece | `get_target_token_id` | Implementation | Verified |
| 59 | requirements.txt torch 2.13.0 vs used 2.6.0+cu124 | `requirements.txt`, pack manifests | Repo audit | Verified |
| 60 | Study hardware GTX 1660 Ti | pilot and full-last manifests | Experimental | Verified for those two studies |

## Claims intentionally left out
- Any statement about ROME, MEMIT, steering vectors, prompting, RAG, LoRA, or fine-tuning performance.
- H6, H7, H8 outcomes, "20.67x" speedups, factual-vs-grammatical success rates from the stage-1 report, and all numbers from the earlier draft's cited papers (Gupta et al., Arad et al.).
- Claims about novelty.
