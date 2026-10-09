# Cross-model transfer: step-by-step plan

Branch: `cross-model-transfer` (from `main` at `bc35482`). Written 2026-10-09. Background and reasoning: `DESIGN.md` (same folder). Earlier side note on the same idea: `docs/cross_model_patching_idea.md`.

## In plain words
We edit a fact in one language model, take the change that edit made inside the model, translate it into the second model's internal coordinates, and add it there while the second model reads the same sentence. No weights change in either model. The question is whether the second model's answer moves, whether it moves for the right reason (a fact, not just a pushed word), and whether other things stay put.

The first thing to find out is whether a **translator** between two models' internal states can exist at all. Everything else depends on that, so it comes first.

## How we work through this
- You run every step in a PowerShell terminal, from the repo root `D:\Projects\transient_steering`, with the project's Python: `.venv\Scripts\python.exe`.
- I write the code for one step at a time, tell you the exact command, and you run it. Each script has `--quick` (tiny smoke run, about a minute) and prints a **RESULT** block at the end. Paste that block back and I read it with you.
- After each step I update the status table below. Nothing is committed or pushed unless you ask.
- New code goes in new files (`src/transfer/`, `tools/transfer/`). Existing code is imported and called, never edited. If a step would need a change to old code, I stop and ask.
- Outputs go to `outputs/transfer/` (a `.gitignore` inside it keeps them out of git; created in step A0).
- Pass/fail numbers ("gates") are proposals. We may revise a gate **before** looking at any later-stage result, never after.
- All times are estimates for the RTX 4050 laptop GPU (6 GB), not measurements.

## Status

| Step | What | Status |
|---|---|---|
| A0 | Setup check: downloads, tokenizer match, memory | **done 2026-10-09**, all checks passed (tokenizers identical, fidelity 1e-4, peak GPU 4.18 GB) |
| A1 | Fit translators on ordinary text, score them | **done 2026-10-09** (138 s). 50 maps saved. Per-dim R2: small 8 -> medium 16 = 0.58, medium 16 -> small 8 = 0.66; best overall small 2 -> medium 4 = 0.65; medium -> small is easier by 0.03-0.08; gradient descent matches the exact solution (0.6287 vs 0.6289), so the exact fit is used |
| A2 | Stitching and difference checks, **Gate 1** | **done 2026-10-09**. 8 pairs tested. Stitching recovered 0.91-0.98 on all 8. Last-position difference cosine 0.24-0.31, above the within-template chance 95th percentile on all 8; **2 of 8 reach the proposed 0.3 line, both medium -> small** (m2s_L16_L8 0.314, m2s_L8_L6 0.308). Gate 1 passes for Import; Export (small -> medium) is above chance but under the line (best s2m_L8_L16 0.275). See results log below |
| B1 | Positive control: transfer a "Germany instead of France" change, **Gate 2** | **primary pairs done 2026-10-10, Gate 2 PASSES in both directions** (Export s2m_L8_L16 55%, Import m2s_L16_L8 33%, all controls 0-1%); the six exploratory pairs are done too (36-79%, controls at most 1%). Details in the results log. A first full run was lost to a too-short background time limit (50 min); the tool now saves each pair as it finishes |
| C1-C5 | Is the gradient-descent edit a fact edit or a word push? (GPT-2 small alone) | todo (can run in parallel with A and B) |
| D1-D4 | Export: edit in small, transfer to medium | **30-record pilot done 2026-10-10 (a look, not a result)**: translated edit raises the target in medium from median rank 165 to about 22, top-1 up to 20%, controls 0-3%, but a plain word push reaches 87-100%. Controlled run designed (Phase D section), Gate 3 proposed, **waiting for the author to confirm the numbers** |
| E1-E4 | Import: medium's state into small, small's SAE filters | todo |
| F | Combined analysis, journal entry | todo |
| G | Replication on Gemma 3 270M and 1B (SAEs on both sides) | todo |
| H | Tester app (`transfer_app.py`) | todo |

## Results log (full-size runs only; numbers copied from the terminal output, files in `outputs/transfer/`)
**A1** (500,126 fit tokens, 100,076 held-out tokens, wikitext-2). Per-dimension R-squared, small layer 8 -> medium layer 16: 0.581; medium 16 -> small 8: 0.661. Best overall small 2 -> medium 4: 0.646, medium 4 -> small 2: 0.729. Medium -> small is easier by 0.03-0.08 at matched depths. Gradient descent on one pair: 0.6287 against 0.6289 exact.

**A2** (100 text sequences the maps never saw; 210 sentence pairs from 6 templates; clean next-word loss small 4.245, medium 3.964):

| pair | stitch loss (clean / stitched / floor / wrong text) | recovered | difference cosine at the word | at the last position (chance 95th pct, share of pairs above) |
|---|---|---|---|---|
| s2m 2 -> 4 | 3.96 / 4.13 / 10.70 / 10.88 | 0.976 | 0.709 | 0.294 (0.242, 70%) |
| s2m 6 -> 8 | 3.96 / 4.15 / 9.12 / 10.76 | 0.964 | 0.656 | 0.269 (0.209, 75%) |
| s2m 8 -> 12 | 3.96 / 4.20 / 8.49 / 10.68 | 0.947 | 0.648 | 0.242 (0.199, 65%) |
| s2m 8 -> 16 | 3.96 / 4.36 / 8.17 / 10.53 | 0.906 | 0.628 | 0.275 (0.202, 83%) |
| m2s 4 -> 2 | 4.24 / 4.37 / 10.52 / 10.85 | 0.980 | 0.762 | 0.279 (0.221, 74%) |
| m2s 8 -> 6 | 4.24 / 4.42 / 8.22 / 10.59 | 0.957 | 0.711 | **0.308** (0.238, 73%) |
| m2s 12 -> 8 | 4.24 / 4.42 / 8.29 / 10.55 | 0.956 | 0.682 | 0.288 (0.238, 68%) |
| m2s 16 -> 8 | 4.24 / 4.31 / 8.29 / 10.75 | 0.984 | 0.694 | **0.314** (0.229, 82%) |

Translated differences come out 0.44-0.96 times the size of the real ones (so the injection scale `alpha` will need to be above 1). Feature view: SAE feature directions of small layer 8 keep a cosine of 0.89 after a round trip small -> medium (layer 12 or 16) -> small, against 0.81 for random directions.

**Reading.** The translated state keeps the receiver working (stitching) and is text-specific (wrong-text is worse than the floor). The translated difference carries which word changed well (cosine 0.63-0.76) but the downstream summary state at the last position only weakly (0.24-0.31); that signal is real (above chance on every pair) but modest, and the 0.3 line separates the pairs by small margins. The gate line of 0.3 was a proposal made before the data; it has not been moved.

**B1** (country swap; primary pairs; test cases = ordered country pairs the receiver answers correctly on both sentences, dose chosen on 30% dev pairs, numbers on the other 70%). Files: `outputs/transfer/b1_swap_primary.json`, per-pair caches `b1_cache_*.pt`.

| | Export: small 8 -> medium 16 (1,021 test cases, 28 countries) | Import: medium 16 -> small 8 (695 test cases, 25 countries) |
|---|---|---|
| top answer becomes the swapped capital, all positions (dose chosen on dev) | **55%** (dose 3) | **33%** (dose 1.5) |
| word's position only / last position only | 60% / 1% | 41% / 5% |
| random vector, random map, wrong pair (each at its best dose) | 0% / 0% / 1% | 0% / 0% / 1% |
| receiver's own real difference, dose 1, all positions (ceiling, exact replacement) | 100% | 100% |
| median rank of the swapped capital: unchanged / translated / random / wrong pair | 1431 / **1** / 980 / 851 | 2351 / **5** / 1574 / 3358 |
| top answer when not the swapped capital: original / another capital / not a capital | 1% / 4% / 41% | 0% / 1% / 65% |
| leak on 12 unrelated prompts, translated vs random of the same size (KL; top-1 flips) | 0.049; 14% vs 0.023; 11% | 0.044; 27% vs 0.027; 25% |

Gate 2: (a) rate >= 20%, (b) margin over the best control >= 15 points, (c) paired Wilcoxon p < 0.01 (reported p underflows to 0), sanity checks (zero dose exact, native dose 1 = 100%) all hold -> **PASS** for both directions. Reading: the translated difference carries a *meaning change* (which country the sentence names) strongly enough to flip the receiver's answer, and does it far above every control. It is carried mostly by the word's own position (word-only variant is as good as or better than all positions) and almost not by the last position alone (1% and 5%; for Export the receiver's own last-position difference reaches 53%, so the translator loses it; for Import the receiver's own last-position difference reaches 0%, so that variant cannot work there for reasons unrelated to the translator). Caveats: test cases share countries, so they are not independent; in Import 65% of top answers are not in our capital list (mostly first pieces such as ' St', ' K', ' C' and ' the'), so exact top-1 match understates how close the answer is (median rank 5); B1 sends the difference of two real sentences, not an edit, and says nothing about facts yet.

**B1 exploratory pairs** (all positions, doses 1 to 4, same cases; file `outputs/transfer/b1_swap_exploratory.json`; no gate applied, they were not named as primary before the run). Translated success, with the strongest control in brackets, and the median rank of the swapped capital (unchanged -> translated):

| source layer -> receiver layer | translated | control | median rank |
|---|---|---|---|
| small 2 -> medium 4 | 79% (dose 1.5) | 1% | 1431 -> 1 |
| small 6 -> medium 8 | 71% (dose 2) | 1% | 1431 -> 1 |
| small 8 -> medium 12 | 67% (dose 2) | 1% | 1431 -> 1 |
| small 8 -> medium 16 (primary) | 55% (dose 3) | 0% | 1431 -> 1 |
| medium 4 -> small 2 | 42% (dose 1) | 1% | 2351 -> 2 |
| medium 8 -> small 6 | 38% (dose 1.5) | 1% | 2351 -> 3 |
| medium 12 -> small 8 | 36% (dose 1.5) | 1% | 2351 -> 4 |
| medium 16 -> small 8 (primary) | 33% (dose 1.5) | 0% | 2351 -> 5 |

Pattern: the earlier the layer the receiver is fed at, the better the swap works. That fits the swap being mostly a change in *which word was named* (token identity dominates early layers and translates best); it is a reason not to read B1 as evidence about facts. For fact edits made at small layer 8 the relevant rows are small 8 -> medium 12 and 16.

## Decisions already made
- The translator is fitted on the **residual stream directly**, not on SAE features. SAE features are used afterwards to read and explain what arrived.
- Start inside the **GPT-2 family** (shared tokenizer, no porting). Partner model: **GPT-2 medium** (24 layers, 1024 wide). GPT-2 small is 12 layers, 768 wide.
- An SAE is needed only where the edit is made. GPT-2 small has one; GPT-2 medium does not, and we do not need one for the main study.
- Map-fitting text: **wikitext-2** (about 2.4 million tokens, 7.8 MB). Facts: CounterFact (already in `datasets/`).
- Directions: **Export** first (small to bigger), **Import** second (bigger to small).
- The ROME dev records (`random.Random(0).sample(range(len(ds)), 100)` over CounterFact) are excluded from every test set here.

## Still open (not needed until the step named)
- Gate numbers (proposed below, confirm after A2).
- Test-set size, proposed 150 records per fact set (before D1).
- Whether a GPT-2 large donor is worth it (decide after counting, in E1).
- Phase G route (Gemma 3 pair, or training a GPT-2 medium SAE).

---

## Phase A: can a translator exist? (no SAE, no facts)

### Step A0: setup check
**Goal:** make sure both models and the text are on this machine and behave.
**I write:** `tools/transfer/00_setup_check.py`, `src/transfer/__init__.py`, `src/transfer/models.py` (load a model by name through TransformerLens with the project's usual defaults; tokenizer check).
**You run:**
```powershell
.venv\Scripts\python.exe tools/transfer/00_setup_check.py
```
**Downloads (I will list them again when the script is ready; they happen when you run it):**
- `gpt2-medium` from huggingface.co: `model.safetensors` 1520 MB plus tokenizer files, about 3 MB.
- `Salesforce/wikitext`, config `wikitext-2-raw-v1`: train 6.4 MB, validation 0.7 MB, test 0.7 MB.
- Saved to your Hugging Face cache, `C:\Users\imalv\.cache\huggingface`. Disk free: 92 GB.

**It checks:** the two tokenizers are identical (vocabulary and 200 sample sentences); layer count and width of each model; both fit on the GPU together; GPT-2 small's output through TransformerLens matches the Hugging Face model; creates `outputs/transfer/.gitignore`.
**Expect:** 5 to 10 minutes, mostly the download.
**Pass:** every line says OK. **Stop if** the tokenizers differ.

### Step A1: fit the translators
**Goal:** learn, for many pairs of layers, a linear map from one model's residual stream to the other's, and score each map on text it never saw.
**What it does:** runs both models over the same wikitext-2 text (about 500,000 tokens for fitting, 100,000 held out, sequences of 128 tokens, the first token of each sequence skipped because its state is unusually large). It keeps only running sums, not the states themselves. The map is then solved by ridge regression (least squares with a small stabilising term; the term is picked on the held-out text).
**Layers tried:** small layers 2, 4, 6, 8, 10 (the `resid_pre` point, as the project uses) against medium layers 4, 8, 12, 16, 20. That is 25 pairs, fitted in **both** directions (small to medium, medium to small).
**I write:** `src/transfer/paired_acts.py`, `src/transfer/maps.py`, `tools/transfer/01_fit_maps.py`.
**You run:**
```powershell
.venv\Scripts\python.exe tools/transfer/01_fit_maps.py --quick
.venv\Scripts\python.exe tools/transfer/01_fit_maps.py
```
**Expect:** quick run about 1 minute; full run 10 to 20 minutes. It writes the maps and a table to `outputs/transfer/`.
**Result block:** a table of the 25 pairs per direction with two scores on held-out text:
- *Raw R-squared*: share of the target state's variance the map explains. Can look high just because a few huge dimensions dominate.
- *Per-dimension R-squared*: the same, averaged over every dimension equally. This is the honest one.
- Mean cosine between predicted and real state.

**No pass/fail here.** We read the table together and pick the 3 best pairs per direction for A2. A very flat table (all pairs similar, all poor) is itself an answer.

### Step A2: do the maps actually work? Gate 1
**Goal:** the checks that matter. A map can fit well and still be useless, and we only ever transfer *differences*, so the second check is the important one.
**Checks (on the 3 best pairs per direction):**
1. **Stitching.** Run the receiver, but at layer `L_R` replace its residual stream with the translated state from the other model. Measure how much next-word loss is recovered between two reference points: a floor (replace with the average state, the information destroyed) and the receiver's own clean loss. 1.0 means the translated state is as good as the real one.
2. **Difference fidelity.** Take pairs of near-identical sentences that differ in one word ("The capital of France is" and "The capital of Germany is"; about 200 pairs from a handful of templates, single-token words only). Translate the first model's difference and compare it with the second model's real difference (cosine). Compare against the same number with the pairs shuffled, which gives the chance level.
3. **Feature view (small side only).** Push each of GPT-2 small's SAE decoder directions through the small to medium map and report how much of the map's output survives a trip back; this is a rough readout of whether the map respects the SAE's feature structure. Diagnostic only, not a gate.

**I write:** `tools/transfer/02_check_maps.py`, the pair templates in `src/transfer/minimal_pairs.py`.
**You run:**
```powershell
.venv\Scripts\python.exe tools/transfer/02_check_maps.py --quick
.venv\Scripts\python.exe tools/transfer/02_check_maps.py
```
**Expect:** about 10 minutes.
**Gate 1 (proposed):** for at least one layer pair, stitching recovers at least 0.8 **and** the mean difference cosine is at least 0.3 and clearly above the shuffled chance level.
**If it fails**, in this order: wider layer grid; a small two-layer network instead of a linear map; medium to large instead of small to medium; stop and write up the negative result.

---

## Phase B: positive control

### Step B1: swap one country for another
**Goal:** prove the whole pipeline moves a model in a known direction before any edited fact is involved.
**What it does:** for about 25 country-capital pairs, take the first model's change for "Germany" instead of "France" (at every position, token-for-token), translate it with the Gate-1 map, add it to the second model's residual stream while it reads "The capital of France is", and see whether its answer moves from Paris toward Berlin. Scale `alpha` swept over 0.5, 1, 1.5, 2, 3.
**Comparisons (all at matched size):**
- the second model's own real change for the same swap (the ceiling),
- a random vector of the same norm,
- a translated change from a *different* pair (the wrong fact),
- no translator (a random map).

**Measures:** share of pairs where the swapped answer becomes top-1; median change in its rank; effect on 12 unrelated prompts (KL divergence, top-1 flips).
**I write:** `src/transfer/inject.py`, `src/transfer/deltas.py`, `tools/transfer/03_swap_control.py`.
**You run:**
```powershell
.venv\Scripts\python.exe tools/transfer/03_swap_control.py --quick
.venv\Scripts\python.exe tools/transfer/03_swap_control.py
```
**Design as built** (`tools/transfer/03_swap_control.py`, `src/transfer/swap.py`). Prompts: "The capital of {X} is the city of" and "Everyone knows that the capital of {X} is the city of" (the plain "The capital of X is" makes both models answer "the" or "a", so small knew no capitals; these wordings force a city name: small knows 21 capitals, medium 27). A test case is an ordered pair of countries (a, b) that the **receiver** answers correctly on both. The difference "state for b minus state for a" is read in the source model at every position, translated, scaled by `alpha`, and added to the receiver's state while it reads the sentence for a. Success: the receiver's top answer becomes b's capital. Injection variants: all positions, the word's position only, the last position only. Arms: translated; the receiver's own real difference (ceiling); a random vector of the same size; a random linear map of the source difference; the translated difference of a *different* pair. `alpha` is chosen on 30% of the country pairs (dev) and reported on the other 70% (test); controls are also shown at their own best `alpha` on test, which favours the controls. Primary pairs, named before running: **Export s2m_L8_L16** and **Import m2s_L16_L8**; the other six pairs are exploratory.

**Gate 2 (fixed 2026-10-09 before the run, per direction, all-positions variant):**
- (a) the translated difference, at the `alpha` chosen on dev, makes the swapped capital the receiver's top answer on at least **20%** of test cases;
- (b) that rate exceeds the best control (each at its own best `alpha` on test) by at least **15 percentage points**;
- (c) the gain in logit difference (swapped capital minus original capital) is larger for the translated arm than for the strongest control, paired Wilcoxon test, **p < 0.01**.

Sanity checks that must hold or the run is invalid: `alpha` = 0 reproduces the unchanged model (0% success); the receiver's own difference at `alpha` = 1 on all positions gives at least 97% success (it is a state replacement, so it should be exact).
Caveat written in advance: the test cases share countries, so they are not independent; confidence intervals are indicative only.

---

## Phase C: fact edit or word push? (GPT-2 small alone)
Can start any time; independent of A and B. It decides what the paper may *say* about the transferred edits. Steps are outlined here; commands are written when we reach them.

- **C1. Shared benchmark.** Frozen dev (about 50) and test (150) record lists from CounterFact, excluding the ROME dev records; CounterFact's reworded prompts, neighbour prompts and generation prompts. Metrics module reused by every later phase (this is also ROME plan steps 6 and 7, so we build it once). `src/transfer/benchmark.py`.
- **C2. Rank-matched random targets.** Random words picked to start at ranks similar to the true answers, so the comparison is not circular (the 17 versus 17,504 problem of Entry 30).
- **C3. Position test.** A new function (in `src/gradient_editing_masked.py`, wrapping the existing edit without changing it) that applies the edit only at the subject's tokens, or only at the last token. A fact lives at the subject; a push lives at the end.
- **C4. Direction test at scale.** For each edit, how much of it points along the target word's own output direction (the existing single-prompt tool, `tools/test_additive_mechanism.py`, looped over many records).
- **C5. Run and read.** Arms: `gd` (multiplier edit), `gd_add@cap=0.25` (additive, known to leak), ROME (already working on GPT-2 small). Report against the rank-matched random control.

**Pre-stated reading (H24, confirm before the run):** "association-like" if paraphrase success is clearly above the rank-matched random control (paired test, p < 0.01) **and** the subject-only edit keeps at least half of the full edit's effect; "read-out steering" if the paraphrase lift is close to the random control or the edit is dominated by the target's output direction; "mixed" otherwise.

---

## Phase D: Export (edit in small, transfer to medium)
Gate 2 is passed, so this can start. A 30-record pilot (`tools/transfer/04_export_pilot.py`, 2026-10-10, a look and not a result) has been run; what it showed, and how it shapes the design below, is in "Step D: pilot and analysis" at the end of this section. The numbers in "Gate 3" are PROPOSALS for the author to confirm before the controlled run, and they were written after seeing the pilot (the pilot records are excluded from dev and test).

### Step D, in plain words
A real edit is made in GPT-2 small with the project's gradient-descent tool, for example "The mother tongue of Danielle Darrieux is" -> " English" (a counterfactual target; the true answer is French). The edit changes small's internal state at layer 8, at every position, by `sum_k a_k * act_k * W_dec[k]`. We take that change, translate it with the A1 map (small layer 8 to medium layer 12 or 16), multiply by a dose, and add it to GPT-2 medium while it reads the same sentence. Then we ask four questions: (1) does medium now say " English"? (2) does it still say it when the sentence is reworded? (3) does it leave nearby facts and unrelated sentences alone? (4) is this a changed fact or just a pushed word? Question 4 cannot be answered from question 1: a plain push along the target word's own output direction makes medium say it almost every time (pilot), so the transferred edit has to be judged on questions 2 and 3 and against that push.

### Proposed design of the controlled run
- Records: CounterFact, single-token counterfactual target, neither model answers it at baseline, excluding the 100 ROME dev records and the 30 pilot records; 50 dev (choose layer and dose), 150 test (frozen first). Primary analysis on records where the edit reaches rank 1 in small (otherwise there is nothing to transfer); all records also reported.
- Maps: small 8 to medium 12 and small 8 to medium 16 only (the layers where the edit lives).
- Modes: relay (the same multipliers are re-applied in small to each reworded prompt and the new change is translated) and vector (one fixed change from the edit prompt).
- Measures on medium: target is the top answer; median rank of the target; same on CounterFact's paraphrase prompts; on the neighbourhood prompts (the target should not rise); on the 12 unrelated prompts (KL, top-1 flips); 20-word continuations with the transfer kept on.
- Controls at matched size: random vector; another record's translated edit; push along the target word's own output direction; edits toward rank-matched random words tuned in small and transferred; the fact stated in medium's prompt (the plain prompting baseline).

### Gate 3 (PROPOSED, to be confirmed; written after the pilot, so informed by it)
On the test records whose edit reaches rank 1 in small, relay mode, dose chosen on dev: (a) the target is medium's top answer on at least 15%; (b) that rate exceeds the best of random, another-record and rank-matched-random controls (each at its best dose) by at least 10 points; (c) paired Wilcoxon on the gain in the target's log-rank p < 0.01; (d) REPORTED, not gated: effect on paraphrase prompts as a share of the effect on the tuned prompt, and effect on neighbour and unrelated prompts against the word-push control. The word-push control is expected to beat the translated edit at top-1 (pilot); beating it is not required, but a result where the translated edit's top-1 is explained by a push (neighbour and unrelated prompts disturbed as much as the push does, paraphrase carry no better than random-target edits) is reported as "word push transferred", not "fact transferred".

- **D1. Fact sets.** CounterFact records where GPT-2 small ranks the true answer below first (the existing hard set), single-token answers only, plus counterfactual targets. Dev 50, test 150.
- **D2. Get the change.** Run the existing gradient-descent edit in small; read the change it made to small's residual stream at the chosen layer, for every prompt position. Relay mode: for each new input (a reworded prompt, each generated word), small is run with the same multipliers again and the change is recomputed and translated. Vector mode (one fixed vector from the edit prompt) is the cheap baseline.
- **D3. Inject and measure.** Add the translated change to medium. Tune layer and `alpha` on dev, freeze, run test. Measures: target rank 1 on the tuned prompt; reworded prompts; neighbour prompts; the 12 unrelated prompts; 20-word generations with the transfer kept on; edit size as a share of medium's residual norm; transfer efficiency (effect in medium divided by effect in small).
- **D4. Controls (all at matched size):** random vector; a push along the target word's own output direction in medium (if this does as well, the word transferred, not the fact); no translator; rank-matched random targets; the fact stated in medium's prompt; medium's own gradient-descent edit if we port it (the ceiling).

### Step D: pilot and analysis (2026-10-10; 30 CounterFact records, no split, no gate; `packs/cross_model_transfer/d_pilot.json`)
- The gradient-descent edit reaches rank 1 in small on 17 of 30 counterfactual targets (median start rank 315, median same-prompt KL 0.92, edit size at the last position 29% of the residual norm).
- **89% of the edit's squared size is at positions between the start token and the last token (the subject and relation words); 5% is at the last position.** The multiplier edit is mainly a change of the subject's representation. This is where B1 found the translator strongest (the word's position) and the last position weakest, so the edit lives where the channel is best. It is also compatible with an association edit (ROME's locus is the last subject token) but does not show one: that is Phase C.
- Transfer to medium (medium's median rank of the target before anything: 165). Small 8 to medium 16, target is medium's top answer / median rank: translated 0, 2, 6, 5 of 30 / 25, 21, 23, 89 at doses 1, 2, 3, 4; random vector 0, 1, 1, 1 / 159 to 560; another record's edit 0 of 30 / 179 to 611; push along the target word's output direction 5, 26, 28, 30 of 30 / rank 3 then 1. Small 8 to medium 12: translated 0, 2, 5, 3 of 30; push 0, 6, 17, 26 of 30.
- Reading: the translated edit does something real (rank 165 to about 22, top-1 up to 20%, controls 0 to 3%), but it is far weaker than a trivial push, and it degrades at the highest dose (the translated change is off the map's training distribution). Success at top-1 alone cannot separate fact from word push, which is why Gate 3 is judged on controls, paraphrases, neighbours and unrelated prompts.
- Expected from B1 and A2 (a prediction, to be checked): edits that live at the last position (the additive edit) transfer worse than the multiplier edit, because the last-position summary state translates weakly (0.24 to 0.31).
- Open design questions: conditional on the edit succeeding in small, how much of medium's effect remains (not computed in the pilot); whether relay mode keeps the paraphrase effect; whether a map fitted on edited-state differences, not ordinary text, would do better (a possible upgrade if the controlled run is weak).

## Phase E: Import (medium's state into small, small's SAE filters)
Starts after Phase D has a first reading. Details to be filled in.
- **E1. Count.** CounterFact true facts that medium gets right and small gets wrong; the same for large. If too few, a bigger donor or ROME-edited counterfactuals.
- **E2. Donor change.** (a) Natural: medium's own clean state versus small's, translated into small's space. (b) ROME-edited medium (official hparams exist for medium, large and XL; run in `.venv-rome`), difference between edited and unedited medium.
- **E3. Filtering with small's SAE.** Encode small's own state and the translated donor state with small's SAE; differences are `df`; accept `all`, the `top k`, or only where the SAE reconstructs the translated state well (`agree`). Inject only the decoded difference of the accepted features. The receiver is **never told the answer** in these arms. A target-aware arm (`causal`) is secondary and needs its own control.
- **E4. Same controls as D4.**

## Phase F: combined analysis
One table ranking every arm on one yardstick, with paired tests, confidence intervals, and a breakdown by starting rank. Data packs and figures from real data only. Journal entry 31, hypothesis H24 onward, experiment index and timeline updated (after asking).

## Phase G: replication on Gemma 3 270M and 1B
Both have SAEs in the installed library (resid, several layers). Needs: confirm the tokenizers match; confirm they load in the installed TransformerLens; port layer and hook choices into new files. Adds the readable table "feature A of the small model corresponds to feature B of the bigger model". GPT-2 large and XL (16-bit) can serve as extra donors without any porting.

## Phase H: tester app
New file `transfer_app.py` (leaves `experiment_app.py` untouched): pick models and layers, enter a prompt and target, run the edit or donor, see which features were sent and which arrived (with Neuronpedia links where an SAE exists), an `alpha` slider with live results, controls on and off, reworded and unrelated prompts, side-by-side generation. Built around the arms that actually worked.

---

## Files this plan will add (nothing existing is edited)
```
src/transfer/{__init__,models,paired_acts,maps,minimal_pairs,deltas,inject,benchmark,metrics,pooling}.py
src/gradient_editing_masked.py
tools/transfer/{00_setup_check,01_fit_maps,02_check_maps,03_swap_control,04_export_pilot}.py   (later: C, D, E tools)
outputs/transfer/            (local only, ignored by git)
docs/cross_model_transfer/{PLAN,DESIGN}.md   (+ PROGRESS notes in this file's status table)
```

## Words used here
- **Residual stream:** the model's running internal state at each word position.
- **SAE feature:** one readable direction in that state, found by a sparse autoencoder.
- **Translator / map:** a table of numbers that converts one model's state into the other's coordinates.
- **Delta / change:** edited state minus unedited state, for the same text.
- **Stitching:** swapping a model's internal state for a translated one and seeing how well it still predicts words.
- **Rank-matched:** controls chosen to start at the same difficulty as the real targets.
