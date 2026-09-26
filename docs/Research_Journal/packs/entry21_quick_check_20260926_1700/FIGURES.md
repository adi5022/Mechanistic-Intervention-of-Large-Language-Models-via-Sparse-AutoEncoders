# Figures - entry21_quick_check_20260926_1700

> **PRELIMINARY - do not cite.** Too few comparable prompts (or errors) for publication-grade claims.


Study: **Safety filter study - QUICK CHECK (sanity test, a few minutes)** - layer 8 - result file `safety_filter_spec_quick.json` (sha256 `ad2f2226987d303a...`) - code commit `a9e6302` + uncommitted changes.

## Fig01. Prompts reaching rank #1 by safety-filter variant (all)

![Fig01](figures/Fig01_all_success.svg)

**Caption.** Share of prompts whose target token reached rank #1, per safety-filter variant, all-prompt-positions candidate source (n = 6 prompts whose target did not start at rank #1).  
*Data:* `tables/arms_all.csv`  
*Also as:* [PNG](figures/Fig01_all_success.png), [PDF](figures/Fig01_all_success.pdf)

## Fig02. Features edited at the best result (all)

![Fig02](figures/Fig02_all_features.svg)

**Caption.** Mean number of SAE features muted or boosted at the best result found, per variant, all-prompt-positions candidate source (n = 6).  
*Data:* `tables/arms_all.csv`  
*Also as:* [PNG](figures/Fig02_all_features.png), [PDF](figures/Fig02_all_features.pdf)

## Fig03. Collateral damage on unrelated prompts (all)

![Fig03](figures/Fig03_all_collateral.svg)

**Caption.** Mean KL divergence (nats) between the clean and edited next-token distributions on 20 unrelated prompts, per variant, all-prompt-positions candidate source. Lower means less collateral change.  
*Data:* `tables/arms_all.csv`  
*Also as:* [PNG](figures/Fig03_all_collateral.png), [PDF](figures/Fig03_all_collateral.pdf)

## Fig04. Mean run time (all)

![Fig04](figures/Fig04_all_time.svg)

**Caption.** Mean model-compute time per run in seconds, per variant, all-prompt-positions candidate source (n = 6).  
*Data:* `tables/arms_all.csv`  
*Also as:* [PNG](figures/Fig04_all_time.png), [PDF](figures/Fig04_all_time.pdf)

## Fig05. Success rate by starting difficulty (all)

![Fig05](figures/Fig05_all_difficulty.svg)

**Caption.** Share of prompts reaching rank #1 per safety-filter variant, grouped by how far down the target started (baseline rank bins), all mode.  
*Data:* `tables/difficulty_bins.csv`  
*Also as:* [PNG](figures/Fig05_all_difficulty.png), [PDF](figures/Fig05_all_difficulty.pdf)

## Fig06. Final rank: graded5_after vs strict (all)

![Fig06](figures/Fig06_all_scatter_graded5_after.svg)

**Caption.** Final target rank per prompt under 'graded5_after' against the reference 'strict', all-prompt-positions candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 6).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig06_all_scatter_graded5_after.png), [PDF](figures/Fig06_all_scatter_graded5_after.pdf)

## Fig07. Final rank: graded5_inter vs strict (all)

![Fig07](figures/Fig07_all_scatter_graded5_inter.svg)

**Caption.** Final target rank per prompt under 'graded5_inter' against the reference 'strict', all-prompt-positions candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 6).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig07_all_scatter_graded5_inter.png), [PDF](figures/Fig07_all_scatter_graded5_inter.pdf)

## Fig08. Final rank: off vs strict (all)

![Fig08](figures/Fig08_all_scatter_off.svg)

**Caption.** Final target rank per prompt under 'off' against the reference 'strict', all-prompt-positions candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 6).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig08_all_scatter_off.png), [PDF](figures/Fig08_all_scatter_off.pdf)

## Fig09. Final rank: tol5_after vs strict (all)

![Fig09](figures/Fig09_all_scatter_tol5_after.svg)

**Caption.** Final target rank per prompt under 'tol5_after' against the reference 'strict', all-prompt-positions candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 6).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig09_all_scatter_tol5_after.png), [PDF](figures/Fig09_all_scatter_tol5_after.pdf)

## Fig10. Prompts reaching rank #1 by safety-filter variant (last)

![Fig10](figures/Fig10_last_success.svg)

**Caption.** Share of prompts whose target token reached rank #1, per safety-filter variant, last-token-only candidate source (n = 6 prompts whose target did not start at rank #1).  
*Data:* `tables/arms_last.csv`  
*Also as:* [PNG](figures/Fig10_last_success.png), [PDF](figures/Fig10_last_success.pdf)

## Fig11. Features edited at the best result (last)

![Fig11](figures/Fig11_last_features.svg)

**Caption.** Mean number of SAE features muted or boosted at the best result found, per variant, last-token-only candidate source (n = 6).  
*Data:* `tables/arms_last.csv`  
*Also as:* [PNG](figures/Fig11_last_features.png), [PDF](figures/Fig11_last_features.pdf)

## Fig12. Collateral damage on unrelated prompts (last)

![Fig12](figures/Fig12_last_collateral.svg)

**Caption.** Mean KL divergence (nats) between the clean and edited next-token distributions on 20 unrelated prompts, per variant, last-token-only candidate source. Lower means less collateral change.  
*Data:* `tables/arms_last.csv`  
*Also as:* [PNG](figures/Fig12_last_collateral.png), [PDF](figures/Fig12_last_collateral.pdf)

## Fig13. Mean run time (last)

![Fig13](figures/Fig13_last_time.svg)

**Caption.** Mean model-compute time per run in seconds, per variant, last-token-only candidate source (n = 6).  
*Data:* `tables/arms_last.csv`  
*Also as:* [PNG](figures/Fig13_last_time.png), [PDF](figures/Fig13_last_time.pdf)

## Fig14. Success rate by starting difficulty (last)

![Fig14](figures/Fig14_last_difficulty.svg)

**Caption.** Share of prompts reaching rank #1 per safety-filter variant, grouped by how far down the target started (baseline rank bins), last mode.  
*Data:* `tables/difficulty_bins.csv`  
*Also as:* [PNG](figures/Fig14_last_difficulty.png), [PDF](figures/Fig14_last_difficulty.pdf)

## Fig15. Final rank: graded5_after vs strict (last)

![Fig15](figures/Fig15_last_scatter_graded5_after.svg)

**Caption.** Final target rank per prompt under 'graded5_after' against the reference 'strict', last-token-only candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 6).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig15_last_scatter_graded5_after.png), [PDF](figures/Fig15_last_scatter_graded5_after.pdf)

## Fig16. Final rank: graded5_inter vs strict (last)

![Fig16](figures/Fig16_last_scatter_graded5_inter.svg)

**Caption.** Final target rank per prompt under 'graded5_inter' against the reference 'strict', last-token-only candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 6).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig16_last_scatter_graded5_inter.png), [PDF](figures/Fig16_last_scatter_graded5_inter.pdf)

## Fig17. Final rank: off vs strict (last)

![Fig17](figures/Fig17_last_scatter_off.svg)

**Caption.** Final target rank per prompt under 'off' against the reference 'strict', last-token-only candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 6).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig17_last_scatter_off.png), [PDF](figures/Fig17_last_scatter_off.pdf)

## Fig18. Final rank: tol5_after vs strict (last)

![Fig18](figures/Fig18_last_scatter_tol5_after.svg)

**Caption.** Final target rank per prompt under 'tol5_after' against the reference 'strict', last-token-only candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 6).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig18_last_scatter_tol5_after.png), [PDF](figures/Fig18_last_scatter_tol5_after.pdf)

## Fig19. Why the strict filter rejects a candidate feature

![Fig19](figures/Fig19_strict_filter_reasons.svg)

**Caption.** Verdicts of the strict safety filter on every round-0 candidate feature (mute and boost sides pooled) in the reference arm. Only 11 of 1061 features (1.0%) were unusable on both sides.  
*Data:* `tables/strict_filter_reasons.csv`  
*Also as:* [PNG](figures/Fig19_strict_filter_reasons.png), [PDF](figures/Fig19_strict_filter_reasons.pdf)
