# Figures - entry21_pilot_20260926_1812


Study: **Safety filter study - PILOT (all prompt positions)** - layer 8 - result file `safety_filter_spec_pilot.json` (sha256 `3b36c8f30af7a185...`) - code commit `a9e6302` + uncommitted changes.

## Fig01. Prompts reaching rank #1 by safety-filter variant (all)

![Fig01](figures/Fig01_all_success.svg)

**Caption.** Share of prompts whose target token reached rank #1, per safety-filter variant, all-prompt-positions candidate source (n = 45 prompts whose target did not start at rank #1).  
*Data:* `tables/arms_all.csv`  
*Also as:* [PNG](figures/Fig01_all_success.png), [PDF](figures/Fig01_all_success.pdf)

## Fig02. Features edited at the best result (all)

![Fig02](figures/Fig02_all_features.svg)

**Caption.** Mean number of SAE features muted or boosted at the best result found, per variant, all-prompt-positions candidate source (n = 45).  
*Data:* `tables/arms_all.csv`  
*Also as:* [PNG](figures/Fig02_all_features.png), [PDF](figures/Fig02_all_features.pdf)

## Fig03. Collateral damage on unrelated prompts (all)

![Fig03](figures/Fig03_all_collateral.svg)

**Caption.** Mean KL divergence (nats) between the clean and edited next-token distributions on 20 unrelated prompts, per variant, all-prompt-positions candidate source. Lower means less collateral change.  
*Data:* `tables/arms_all.csv`  
*Also as:* [PNG](figures/Fig03_all_collateral.png), [PDF](figures/Fig03_all_collateral.pdf)

## Fig04. Mean run time (all)

![Fig04](figures/Fig04_all_time.svg)

**Caption.** Mean model-compute time per run in seconds, per variant, all-prompt-positions candidate source (n = 45).  
*Data:* `tables/arms_all.csv`  
*Also as:* [PNG](figures/Fig04_all_time.png), [PDF](figures/Fig04_all_time.pdf)

## Fig05. Success rate by starting difficulty (all)

![Fig05](figures/Fig05_all_difficulty.svg)

**Caption.** Share of prompts reaching rank #1 per safety-filter variant, grouped by how far down the target started (baseline rank bins), all mode.  
*Data:* `tables/difficulty_bins.csv`  
*Also as:* [PNG](figures/Fig05_all_difficulty.png), [PDF](figures/Fig05_all_difficulty.pdf)

## Fig06. Final rank: graded5_after vs strict (all)

![Fig06](figures/Fig06_all_scatter_graded5_after.svg)

**Caption.** Final target rank per prompt under 'graded5_after' against the reference 'strict', all-prompt-positions candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 45).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig06_all_scatter_graded5_after.png), [PDF](figures/Fig06_all_scatter_graded5_after.pdf)

## Fig07. Final rank: graded5_inter vs strict (all)

![Fig07](figures/Fig07_all_scatter_graded5_inter.svg)

**Caption.** Final target rank per prompt under 'graded5_inter' against the reference 'strict', all-prompt-positions candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 45).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig07_all_scatter_graded5_inter.png), [PDF](figures/Fig07_all_scatter_graded5_inter.pdf)

## Fig08. Final rank: off vs strict (all)

![Fig08](figures/Fig08_all_scatter_off.svg)

**Caption.** Final target rank per prompt under 'off' against the reference 'strict', all-prompt-positions candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 45).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig08_all_scatter_off.png), [PDF](figures/Fig08_all_scatter_off.pdf)

## Fig09. Final rank: tol5_after vs strict (all)

![Fig09](figures/Fig09_all_scatter_tol5_after.svg)

**Caption.** Final target rank per prompt under 'tol5_after' against the reference 'strict', all-prompt-positions candidate source, log scale. Points below the dashed diagonal are better than the reference; each dot is one prompt (n = 45).  
*Data:* `tables/per_prompt.csv`  
*Also as:* [PNG](figures/Fig09_all_scatter_tol5_after.png), [PDF](figures/Fig09_all_scatter_tol5_after.pdf)

## Fig10. Why the strict filter rejects a candidate feature

![Fig10](figures/Fig10_strict_filter_reasons.svg)

**Caption.** Verdicts of the strict safety filter on every round-0 candidate feature (mute and boost sides pooled) in the reference arm. Only 157 of 5400 features (2.9%) were unusable on both sides.  
*Data:* `tables/strict_filter_reasons.csv`  
*Also as:* [PNG](figures/Fig10_strict_filter_reasons.png), [PDF](figures/Fig10_strict_filter_reasons.pdf)
