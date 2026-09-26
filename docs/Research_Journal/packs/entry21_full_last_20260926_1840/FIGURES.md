# Figures - entry21_full_last_20260926_1840


Study: **Safety filter study - FULL (last token only)** - layer 8 - result file `safety_filter_spec_full_last.json` (sha256 `43efeeb8464f0640...`) - code commit `3f9b18a`.

## Fig01. Prompts reaching rank #1 by safety-filter variant (last)

![Fig01](figures/Fig01_last_success.svg)

**Caption.** Share of prompts whose target token reached rank #1, per safety-filter variant, last-token-only candidate source (n = 131 prompts whose target did not start at rank #1).  
*Data:* `tables/arms_last.csv`  
*Also as:* [PNG](figures/Fig01_last_success.png), [PDF](figures/Fig01_last_success.pdf)

## Fig02. Features edited at the best result (last)

![Fig02](figures/Fig02_last_features.svg)

**Caption.** Mean number of SAE features muted or boosted at the best result found, per variant, last-token-only candidate source (n = 131).  
*Data:* `tables/arms_last.csv`  
*Also as:* [PNG](figures/Fig02_last_features.png), [PDF](figures/Fig02_last_features.pdf)

## Fig03. Collateral damage on unrelated prompts (last)

![Fig03](figures/Fig03_last_collateral.svg)

**Caption.** Mean KL divergence (nats) between the clean and edited next-token distributions on 20 unrelated prompts, per variant, last-token-only candidate source. Lower means less collateral change.  
*Data:* `tables/arms_last.csv`  
*Also as:* [PNG](figures/Fig03_last_collateral.png), [PDF](figures/Fig03_last_collateral.pdf)

## Fig04. Mean run time (last)

![Fig04](figures/Fig04_last_time.svg)

**Caption.** Mean model-compute time per run in seconds, per variant, last-token-only candidate source (n = 131).  
*Data:* `tables/arms_last.csv`  
*Also as:* [PNG](figures/Fig04_last_time.png), [PDF](figures/Fig04_last_time.pdf)

## Fig05. Success rate by starting difficulty (last)

![Fig05](figures/Fig05_last_difficulty.svg)

**Caption.** Share of prompts reaching rank #1 per safety-filter variant, grouped by how far down the target started (baseline rank bins), last mode.  
*Data:* `tables/difficulty_bins.csv`  
*Also as:* [PNG](figures/Fig05_last_difficulty.png), [PDF](figures/Fig05_last_difficulty.pdf)

## Fig06. Final rank: graded5_after vs strict (last)

![Fig06](figures/Fig06_last_scatter_graded5_after.svg)

**Caption.** Final target rank per prompt under 'graded5_after' against the reference 'strict', last-token-only candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 131).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig06_last_scatter_graded5_after.png), [PDF](figures/Fig06_last_scatter_graded5_after.pdf)

## Fig07. Final rank: graded5_inter vs strict (last)

![Fig07](figures/Fig07_last_scatter_graded5_inter.svg)

**Caption.** Final target rank per prompt under 'graded5_inter' against the reference 'strict', last-token-only candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 131).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig07_last_scatter_graded5_inter.png), [PDF](figures/Fig07_last_scatter_graded5_inter.pdf)

## Fig08. Final rank: off vs strict (last)

![Fig08](figures/Fig08_last_scatter_off.svg)

**Caption.** Final target rank per prompt under 'off' against the reference 'strict', last-token-only candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 131).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig08_last_scatter_off.png), [PDF](figures/Fig08_last_scatter_off.pdf)

## Fig09. Final rank: tol5_after vs strict (last)

![Fig09](figures/Fig09_last_scatter_tol5_after.svg)

**Caption.** Final target rank per prompt under 'tol5_after' against the reference 'strict', last-token-only candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 131).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig09_last_scatter_tol5_after.png), [PDF](figures/Fig09_last_scatter_tol5_after.pdf)

## Fig10. Why the strict filter rejects a candidate feature

![Fig10](figures/Fig10_strict_filter_reasons.svg)

**Caption.** Verdicts of the strict safety filter on every round-0 candidate feature (mute and boost sides pooled) in the reference arm. Only 195 of 7691 features (2.5%) were unusable on both sides.  
*Data:* `tables/strict_filter_reasons.csv`  
*Also as:* [PNG](figures/Fig10_strict_filter_reasons.png), [PDF](figures/Fig10_strict_filter_reasons.pdf)
