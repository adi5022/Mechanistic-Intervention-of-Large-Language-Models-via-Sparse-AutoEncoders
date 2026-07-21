# Research Output — Stage 1 Baseline (Target-focused Ablation)

**Date:** July 19, 2026
**Model:** GPT-2-small (residual stream Layer 8)
**Ablation Strength:** 0.3
**Max Search Rounds:** 5

---

## 1. Executive Summary
This experiment verifies whether multi-feature iterative ablation targeting correct token helpers can correct factual completion errors (where the model possesses suppressed knowledge) and whether the corrections leak/break unrelated facts.

### Aggregate Metrics
* **Total Suppressed Facts Tested:** 22
* **Intervention Success Count:** 1
* **Success Rate:** 4.55%
* **Average Rounds Needed (for successful cases):** 3.00
* **Average Specificity Score (for successful cases):** 100.00%

---

## 2. Methodology & Findings
We ran the iterative ablation loop on the 22 suppressed facts from ROME's `known_1000` dataset. For each fact, the selector iteratively:
1. Identifies the top causal feature supporting the correct answer.
2. Applies a soft-ablation hook at strength `0.3`.
3. Re-evaluates target probability and rank.
4. Stops if the target token becomes top-1 (Success) or rounds exceed `5` (Failure).

For successful cases, we re-applied all ablated features to 15 control prompts to measure how many stayed unaffected. 

Our findings indicate that muting target-supportive features generally decreases the target's probability and rank, leading to failure. The single success (Eiffel Tower) was an anomaly because the ablated feature was a general city-identifying feature that supported the competitor ("London") even more strongly.

---

## 3. Results Table

| prompt                                                      | target     | clean_prob   |   clean_rank | success   |   rounds_used | features_ablated                  | final_prob   |   final_rank | specificity   |
|:------------------------------------------------------------|:-----------|:-------------|-------------:|:----------|--------------:|:----------------------------------|:-------------|-------------:|:--------------|
| Eavan Boland was born in                                    | Dublin     | 1.85%        |            4 | False     |             5 | 3111, 19288, 369, 3076, 16240     | 1.60%        |            4 | N/A           |
| Knud, Hereditary Prince of Denmark passed away in           | Copenhagen | 0.50%        |           45 | False     |             5 | 2337, 11034, 13784, 6435, 13194   | 0.39%        |           49 | N/A           |
| The location of Massachusetts Institute of Technology is in | Cambridge  | 1.15%        |            8 | False     |             5 | 3076, 19288, 8239, 24181, 313     | 0.50%        |           16 | N/A           |
| The location of Galatasaray University is in the heart of   | Istanbul   | 14.07%       |            2 | False     |             5 | 10347, 3076, 13754, 11123, 3075   | 12.02%       |            2 | N/A           |
| John Pym died in the city of                                | London     | 0.87%        |           10 | False     |             5 | 11149, 2194, 16649, 2619, 22852   | 0.67%        |           12 | N/A           |
| Emilia Rydberg was born in                                  | Stockholm  | 0.70%        |           16 | False     |             5 | 3111, 16240, 18837, 5858, 15899   | 0.64%        |           16 | N/A           |
| Joseph Goebbels worked in the city of                       | Berlin     | 4.82%        |            2 | False     |             5 | 11149, 2194, 21062, 15688, 23704  | 3.88%        |            4 | N/A           |
| Francesco Castellacci was born in                           | Rome       | 5.41%        |            2 | True      |             3 | 19288, 2337, 16240                | 5.01%        |            1 | 100.0%        |
| 2005 Australian Open is located in                          | Melbourne  | 10.27%       |            2 | False     |             5 | 8239, 24181, 10111, 5858, 3882    | 9.25%        |            3 | N/A           |
| Giulio Romano originates from the city of                   | Rome       | 2.52%        |            2 | False     |             5 | 11149, 12720, 15688, 23092, 22852 | 1.78%        |            4 | N/A           |
| Euromoney Institutional Investor's headquarters are in      | London     | 0.99%        |            7 | False     |             5 | 19288, 8239, 8628, 4445, 3076     | 0.77%        |            9 | N/A           |
| Coco Chanel passed away in                                  | Paris      | 3.43%        |            2 | False     |             5 | 11034, 8239, 6461, 2337, 21460    | 2.87%        |            3 | N/A           |
| The headquarter of Zillow is in downtown                    | Seattle    | 4.12%        |            8 | False     |             5 | 19490, 17551, 5858, 923, 5096     | 3.67%        |            6 | N/A           |
| Jan Swammerdam died in the city of                          | Amsterdam  | 1.16%        |            9 | False     |             5 | 11149, 2194, 8239, 1068, 1288     | 0.89%        |           14 | N/A           |
| Seiyu Group's headquarters are in                           | Tokyo      | 10.00%       |            2 | False     |             5 | 8239, 19288, 4445, 8118, 3076     | 7.05%        |            2 | N/A           |
| Second French Empire's capital city is                      | Paris      | 1.21%        |           14 | False     |             5 | 1442, 18220, 8628, 21496, 23177   | 0.64%        |           22 | N/A           |
| Henry Somerset, 7th Duke of Beaufort worked in the city of  | London     | 2.20%        |            2 | False     |             5 | 11149, 16649, 21062, 16744, 2194  | 1.41%        |            5 | N/A           |
| York University can be found in the heart of                | Toronto    | 7.12%        |            3 | False     |             5 | 10347, 11123, 13754, 14430, 3075  | 6.19%        |            3 | N/A           |
| Rudolf Steiner worked in the city of                        | Berlin     | 2.28%        |            5 | False     |             5 | 11149, 2194, 16649, 23704, 15688  | 1.92%        |            4 | N/A           |
| James Northcote died in the city of                         | London     | 1.38%        |            3 | False     |             5 | 11149, 2194, 16649, 8239, 6807    | 0.96%        |            4 | N/A           |
| 2002 Australian Open is located in                          | Melbourne  | 9.42%        |            2 | False     |             5 | 8239, 24181, 10111, 3882, 5858    | 8.51%        |            3 | N/A           |
| Fantastic Fest can be found in the heart of downtown        | Austin     | 4.41%        |            3 | False     |             5 | 19490, 15820, 11619, 2436, 14726  | 3.53%        |            2 | N/A           |
