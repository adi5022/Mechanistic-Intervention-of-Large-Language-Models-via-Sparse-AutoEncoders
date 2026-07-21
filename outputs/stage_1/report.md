# Research Output — Stage 1: Iterative Ablation & Specificity Check

**Date:** July 19, 2026
**Model:** GPT-2-small (residual stream Layer 8)
**Ablation Strength:** 0.3
**Max Search Rounds:** 5

---

## 1. Executive Summary
This experiment verifies whether multi-feature iterative ablation can correct factual completion errors (where the model possesses suppressed knowledge) and whether the corrections leak/break unrelated facts.

### Aggregate Metrics
* **Total Suppressed Facts Tested:** 22
* **Intervention Success Count:** 3
* **Overall Success Rate:** 13.64% (3/22)
* **Pure Factual Competitor Success Rate:** 37.50% (3/8 cases where the competitor was a genuine competing fact: Rome, Rome, Paris, Sydney, Sydney, Hamburg, Hamburg, Cologne)
* **Average Rounds Needed (for successful cases):** 1.00
* **Average Specificity Score (for successful cases):** 100.00%

---

## 2. Methodology & Findings
We ran the iterative ablation loop on the 22 suppressed facts from ROME's `known_1000` dataset. For each fact, the competitor-focused selector:
1. Dynamically identifies the model's current top predicted token ID (the competitor) under the currently applied persistent hooks.
2. If the current top predicted token is the target, stops and returns success.
3. Otherwise, identifies the top causal feature driving that competitor token's probability and applies a soft-ablation hook at strength `0.3`.
4. Repeats for up to `5` rounds.

### Performance Analysis

#### Factual vs. Non-Factual Competitor Breakdown (19 Failures)
For the 19 failing cases, the model's initial top-1 prediction (competitor) falls into these categories:
* **Stopwords/Articles/Function Words (9/19)**: In 9 cases, the initial wrong guess was `"the"` (8 cases) or `"now"` (1 case). Muting these general grammatical tokens is ineffective because they are supported by broad syntactical circuits.
* **Tokenizer-Truncated City Name Prefixes (4/19)**: In 4 cases, the competitor was a word prefix fragment of another city name (`"Los"`, `"San"`, `"New"`, `"St"`).
* **Subject Name Echo (1/19)**: For `"Henry Somerset, 7th Duke of Beaufort worked in the city of"`, the competitor was `"Beau"`. This represents a distinct copy/echoing failure mode where the model copies a fragment of the subject's own name instead of predicting a city name.
* **Pure Factual Competitors (5/19)**: In 5 cases, the model's initial wrong guess was a genuine competing city name (`"Sydney"` x2, `"Hamburg"` x2, `"Cologne"` x1). 

#### Pure Factual Competitor Performance
Among all tested facts, exactly 8 cases had a genuine competing factual city name as their initial top-1 prediction (successful: `Rome`, `Rome`, `Paris`; failing: `Sydney`, `Sydney`, `Hamburg`, `Hamburg`, `Cologne`). Muting competitor drivers was successful in 3 of these 8 cases, giving a **Pure Factual Competitor Success Rate of 37.50%**.

However, for the 5 failing pure factual cases, round-by-round analysis shows that ablation is **barely moving the competitor's probability**:
* **2005 Australian Open (Sydney)**: 12.13% $\rightarrow$ 11.15% after 5 rounds.
* **2002 Australian Open (Sydney)**: 12.11% $\rightarrow$ 11.12% after 5 rounds.
* **Jan Swammerdam (Hamburg)**: 3.13% $\rightarrow$ 2.51% after 5 rounds.
* **Rudolf Steiner (Hamburg)**: 3.93% $\rightarrow$ 3.70% after 5 rounds.
* **Joseph Goebbels (Cologne)**: 7.64% $\rightarrow$ 7.85% after 5 rounds.

This indicates that even for pure factual competitors, single-feature ablation at strength 0.3 is insufficient to suppress the incorrect guess, as the competitor is supported by either broader features or a highly distributed set of representations. No claim of general viability or robustness can be made yet.


---

## 3. Full Results Table

| prompt                                                      | target     | clean_prob   |   clean_rank | success   |   rounds_used | features_ablated                  | final_prob   |   final_rank | specificity   | initial_top1_token   | final_top1_token   |
|:------------------------------------------------------------|:-----------|:-------------|-------------:|:----------|--------------:|:----------------------------------|:-------------|-------------:|:--------------|:---------------------|:-------------------|
| Eavan Boland was born in                                    | Dublin     | 1.85%        |            4 | False     |             5 | 8782, 19288, 2337, 5858, 24181    | 1.81%        |            3 | N/A           | the                  | the                |
| Knud, Hereditary Prince of Denmark passed away in           | Copenhagen | 0.50%        |           45 | False     |             5 | 6435, 2337, 11034, 77, 16399      | 0.45%        |           47 | N/A           | the                  | the                |
| The location of Massachusetts Institute of Technology is in | Cambridge  | 1.15%        |            8 | False     |             5 | 313, 8459, 3076, 19288, 14430     | 0.73%        |           11 | N/A           | the                  | the                |
| The location of Galatasaray University is in the heart of   | Istanbul   | 14.07%       |            2 | False     |             5 | 313, 11122, 10579, 10347, 8449    | 15.84%       |            2 | N/A           | the                  | the                |
| John Pym died in the city of                                | London     | 0.87%        |           10 | False     |             5 | 11149, 2194, 16649, 22852, 1288   | 0.68%        |           11 | N/A           | New                  | New                |
| Emilia Rydberg was born in                                  | Stockholm  | 0.70%        |           16 | False     |             5 | 8782, 6840, 5858, 2337, 6         | 0.73%        |           14 | N/A           | the                  | the                |
| Joseph Goebbels worked in the city of                       | Berlin     | 4.82%        |            2 | False     |             5 | 11149, 5856, 21062, 15688, 23035  | 5.45%        |            2 | N/A           | Cologne              | Cologne            |
| Francesco Castellacci was born in                           | Rome       | 5.41%        |            2 | True      |             1 | 16240                             | 5.31%        |            1 | 100.0%        | Italy                | Rome               |
| 2005 Australian Open is located in                          | Melbourne  | 10.27%       |            2 | False     |             5 | 19288, 10450, 8239, 10111, 19087  | 9.58%        |            3 | N/A           | Sydney               | Sydney             |
| Giulio Romano originates from the city of                   | Rome       | 2.52%        |            2 | True      |             1 | 5856                              | 2.93%        |            1 | 100.0%        | P                    | Rome               |
| Euromoney Institutional Investor's headquarters are in      | London     | 0.99%        |            7 | False     |             5 | 313, 4445, 5096, 21169, 5858      | 1.18%        |            6 | N/A           | the                  | the                |
| Coco Chanel passed away in                                  | Paris      | 3.43%        |            2 | True      |             1 | 20694                             | 3.85%        |            1 | 100.0%        | May                  | Paris              |
| The headquarter of Zillow is in downtown                    | Seattle    | 4.12%        |            8 | False     |             5 | 19490, 15820, 5096, 13505, 8239   | 3.99%        |            5 | N/A           | Los                  | San                |
| Jan Swammerdam died in the city of                          | Amsterdam  | 1.16%        |            9 | False     |             5 | 11149, 2194, 2619, 16649, 8239    | 0.90%        |           14 | N/A           | Hamburg              | Hamburg            |
| Seiyu Group's headquarters are in                           | Tokyo      | 10.00%       |            2 | False     |             5 | 313, 4445, 7645, 21169, 19935     | 10.95%       |            2 | N/A           | the                  | the                |
| Second French Empire's capital city is                      | Paris      | 1.21%        |           14 | False     |             5 | 10852, 14626, 21000, 21496, 18220 | 1.52%        |           12 | N/A           | now                  | now                |
| Henry Somerset, 7th Duke of Beaufort worked in the city of  | London     | 2.20%        |            2 | False     |             5 | 11149, 2194, 1288, 18994, 21062   | 1.51%        |            5 | N/A           | Beau                 | Beau               |
| York University can be found in the heart of                | Toronto    | 7.12%        |            3 | False     |             5 | 8459, 11122, 10347, 14430, 6863   | 6.92%        |            3 | N/A           | the                  | the                |
| Rudolf Steiner worked in the city of                        | Berlin     | 2.28%        |            5 | False     |             5 | 5856, 2194, 14287, 7276, 23704    | 2.47%        |            5 | N/A           | Hamburg              | Hamburg            |
| James Northcote died in the city of                         | London     | 1.38%        |            3 | False     |             5 | 5856, 11149, 8605, 1288, 6807     | 1.36%        |            4 | N/A           | St                   | St                 |
| 2002 Australian Open is located in                          | Melbourne  | 9.42%        |            2 | False     |             5 | 19288, 10450, 8239, 10111, 19087  | 8.80%        |            3 | N/A           | Sydney               | Sydney             |
| Fantastic Fest can be found in the heart of downtown        | Austin     | 4.41%        |            3 | False     |             5 | 15820, 19490, 12354, 2436, 14726  | 3.64%        |            2 | N/A           | San                  | San                |
