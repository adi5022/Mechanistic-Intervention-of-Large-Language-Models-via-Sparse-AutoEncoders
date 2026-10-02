# Session report, 2026-09-30 to 2026-10-02: from "what is our actual issue?" to a finished dataset for the learned-strength study

Written for a reader who was not in the conversation. Plain language first, numbers second. Every claim here is either measured (and the file that holds the measurement is named) or marked as an estimate or an open question. Companion files: `docs/literature_review_sae_editing.md` (what the literature says), `docs/Research_Journal/22.md` and `23.md` (the detailed journal entries), `handoff.md` (how to resume).

---

## 0. The 60-second version

1. **The project:** edit GPT-2 small while it reads a prompt, using features from a sparse autoencoder (SAE), so that a true answer GPT-2 normally ranks below first becomes its top prediction. Example: "The Colosseum is located in" should predict " Rome", which GPT-2 ranks 7th.
2. **We started with three worries** (do later layers undo the edit? is the SAE overkill? does SAE reconstruction error leak?). We built a diagnostic tool and a new app tab to test them. Result: later layers did not undo the edit in our tests; the reconstruction error is not injected but cannot be edited; the "SAE is overkill" worry could not be settled because our comparison was on hand-picked failures.
3. **We checked the literature** because an employer questioned the SAE approach. Result: the field is established, nothing in the project is conceptually new, and the evidence on SAE steering is mixed. It is not a reason to stop; it is a reason to compare against other methods fairly.
4. **We chose a new idea to try:** a small network that picks the mute and boost strengths per prompt, instead of the fixed 0.6 and 0.5. It is a planned variant of the system, to be written up whether or not it works.
5. **We built the data for that study.** From CounterFact (21,919 records) we made a list of 16,360 prompts GPT-2 gets wrong, split them safely, and prepared saved data for 1,450 of them (a "cache", then "strength tables"). A headroom measurement says SAE-feature scaling could lift the target to rank 1 on up to 80% of these prompts if side effects are ignored (41% with a side-effect penalty).
6. **Not done yet:** the network itself, the real-system benchmark on the test prompts, and the comparison with other methods (IKE, ROME, DiffMean). All data they need (except the other methods' code) now exists.

---

## 1. Where things stand

```
 CounterFact (21,919 records)                                    DONE
   -> keep prompts whose true answer is NOT GPT-2's top choice   DONE   16,360 prompts
   -> split into train / validation / test (safe split)          DONE   1,000 / 150 / 150 / 150
   -> per-prompt cache  (baseline, candidates, removal effects)  DONE   1,450 files
   -> headroom measurement (how much room is there?)             DONE   1,450 prompts
   -> strength tables (which features pass the filter, by strength) DONE 1,450 files
   -> write the training code (the strength network)             NOT YET
   -> train it (minutes)                                         NOT YET
   -> real benchmark on the 300 test prompts (hours)             NOT YET
   -> other methods (IKE, ROME, DiffMean) + comparison tab       NOT YET
```

Branch: `pool-refill-implementation`. Everything is pushed.

---

## 2. Glossary in plain words

| Word | Meaning here |
|---|---|
| **Prompt, target** | The sentence start and the word that should come next. "The mother tongue of Danielle Darrieux is" and " French". |
| **Rank** | Where the target sits in GPT-2's next-word list, sorted from most to least likely. Rank 1 = GPT-2's top prediction. Rank 7 means six words are more likely. |
| **Blocker** | The word currently in first place, which stands in the target's way. |
| **Layer 8, residual stream** | GPT-2 has 12 layers. After layer 7 the model's working state (a list of 768 numbers per word) is the "residual stream at layer 8". All our edits happen there. |
| **SAE feature** | The SAE splits that 768-number state into about 24,576 features, a few hundred of which are "on" for any word. Each feature is meant to stand for one idea. |
| **Candidate** | One of the 200 strongest active features for a prompt. These are the features we may edit. |
| **Mute, boost** | Mute turns a feature down (strength 0.6 means multiply by 0.4). Boost turns it up (strength 0.5 means multiply by 1.5). |
| **Delta patching** | We add only the change the edit causes, `decode(edited) - decode(original)`, so the SAE's reconstruction error is never injected into the model. |
| **Reconstruction error** | What the SAE cannot capture of the state (12 to 26% of its size at layer 8). It is left untouched. |
| **Strict safety filter** | A candidate is tested alone at the set strength and rejected if it makes the target's probability drop or its rank worse. |
| **Sweep** | Your full procedure: rank candidates, filter them, add features step by step until the target is rank 1, refill pools if they run out. |
| **KL (side effect)** | How far the rest of GPT-2's predictions moved because of the edit, in nats. Smaller is gentler. |
| **Cache** | Saved data for each prompt so that later steps never repeat the slow work. |
| **Shard** | A share of the work given to one machine. |
| **Headroom** | How much better any strength choice could possibly do. A ceiling, not a prediction. |
| **Oracle** | Strengths found by trial and error on that exact prompt. The best a strength-picking method could hope to match. |

---

## 3. Part A: the original question

### 3.1 The three worries
1. Editing only layer 8 might let later layers bring the removed fact back ("cross-layer superposition" or self-repair).
2. Using an SAE for this is like using a Lamborghini to buy groceries.
3. SAE reconstruction error is passed down the layers and never corrected.

### 3.2 What we found by reading the code and docs
Worry 3's premise is wrong: your hooks add only the delta, so the reconstruction error is never injected (Journal 1, 12, 18). But the error is also never edited, so any information hiding in it cannot be reached by scaling features.

### 3.3 What we built and ran
- `src/repair_diagnostics.py`, `tools/run_repair_diagnostics.py`: for a prompt and layer, take the SAE edit, follow the pushed direction through the later blocks, run a control that keeps the push at full strength ("held edit"), and compare with an edit made directly in the residual stream without the SAE ("SAE-free edit") of the same size.
- A new app tab, **"Repair & SAE limit"** in `experiment_app.py`, with plain-language verdict cards and charts. It is a diagnostic for failing prompts, not the benchmark.
- Run on the three prompts that had already failed in the Entry 20 study (Colosseum to Rome, door to cat, "The doctor said that" to she) at layers 5 to 11, plus MIT to Cambridge at layer 8.

### 3.4 Results (data: `docs/Research_Journal/packs/repair_diagnostics/merged_all_layers.json`, write-up: Entry 22)
- **Later layers undoing the edit: no sign.** The pushed direction was still 0.79 to 1.29 of its original size at the output. The "held edit" control gave the same rank as the plain edit in all 21 layer/prompt combinations.
- **Reconstruction error:** 12 to 26% of the state's size; untouched by the edit.
- **A fair caution about that control:** if nothing is lost, "re-adding what was lost" adds nothing, so identical ranks were guaranteed. The real evidence is the persistence number, and it only checks the pushed direction, not a later block rebuilding the fact along a different direction. That is not ruled out.
- **SAE vs SAE-free:** on those three prompts a free residual push of the same size reached rank 1 in 21 of 21 combinations, the SAE edit in 1 of 21 (Colosseum at layer 7).

### 3.5 Correction we made (important)
That "1 of 21 versus 21 of 21" is **not a success rate**. The three prompts were chosen because the SAE method had already failed on them, and the diagnostic also used Top-N 120 instead of your Top-N 200. It cannot be compared with the 41/131 or 29/45 figures from the safety-filter studies, which tested the filter, not SAE versus other methods. Entry 22, the hypothesis log (H14 now "unresolved"), the timeline and the handoff were corrected. A fair SAE-vs-others comparison was therefore planned (Section 6 of this report).

### 3.6 Success rates that do exist for your system
| Study | Prompts | Reached rank 1 (strict filter) | Notes |
|---|---|---|---|
| Entry 20, paired | 13 valid | 10 (all positions) vs 4 (last token only) | Top-N 120 |
| Safety-filter pilot | 45 | 29 (64%), all positions | Top-N 120 |
| Safety-filter full | 131 | 41 (31%), last token only | Top-N 120 |
The "reference run" (strict filter, mute 0.6 / boost 0.5, Top-N 200, all positions) has **not yet been run** on any large prompt set. Its spec is `data/reference_strict_topn200_all.json` (131 prompts).

### 3.7 Speed fix
The SAE-free optimiser was slow (309 ms per step). Two causes: it also computed gradients for all 124 million model weights, and it ran 12 optimisations one after another. Computing only the edit's gradient and running the 12 together in one batch gave identical ranks and cut the time per layer from about 150 s to under 10 s for that part.

---

## 4. Part B: is this direction right? (summary; full note in `docs/literature_review_sae_editing.md`)

- **The field is established.** ROME and MEMIT edit facts in weights; SAE steering, SAE unlearning and SAE knowledge-selection steering all exist. Nothing in the project is conceptually new. What may be new is the combination: joint mute-and-boost with a per-candidate safety filter and pool refill, for pushing single facts to rank 1 on GPT-2 small. I found no exact match, but my search was not exhaustive.
- **The evidence on SAE steering is mixed.** Google DeepMind's team de-prioritised SAE research after linear probes beat SAEs at detecting harmful intent on new data. A benchmark (AxBench) found SAE steering far below prompting and simple vector methods on larger instruction-tuned models. Another paper found that choosing features by their causal effect on outputs (which your selector does) makes SAE steering competitive. A study on the same GPT-2 SAE family found they struggled on a factual-knowledge task.
- **Your use case differs** (single-fact next-token flip on a small base model) but is not exempt from those findings.
- **Verdict:** keep going, but (1) compare against other methods on the same prompts, (2) test more than "rank 1 on the same sentence" (paraphrases and neighbours, which CounterFact provides), (3) describe results honestly, including null results.

---

## 5. Part C: the new idea and how it was shaped

### 5.1 The idea
The mute and boost strengths are fixed (0.6, 0.5). From months of use you know the strengths change the outcome. A small network could pick them per prompt. This is a **planned variant of the system** in the same spirit as the dynamic safety filter, to be documented whether or not it works (Journal 23, hypothesis H15, experiment EXP-013).

### 5.2 How the design changed through discussion (kept on record)
| First draft | Problem you raised | Current design |
|---|---|---|
| Network reads only the last token's state | The system uses all prompt positions; important information sits on earlier words | The network reads 12 summary numbers computed over all positions (a few later variants: pooled states, attention-style weighting) |
| Train on the feature lists produced by the 0.6/0.5 run | That throws away features that would pass the filter at lower strengths | **Strength tables:** record the filter's verdict on a grid of strengths so the feature list at any strength can be looked up |
| Control = brute-force search of fixed pairs (about 25 hours) | Too slow | Control = a *learned fixed pair*: two shared numbers trained with the same method (minutes) |
| (not planned) | Strengths per feature, not just per prompt | Version 2: one multiplier per candidate, with the safety filter learned implicitly; version 1 first |
| Prompts: the 131 templated ones | Too few and too repetitive | CounterFact (21,919 records) |

### 5.3 What the network will learn and how
Input: 12 numbers about the prompt (target rank and probability, blocker probability, logit gap, number of active features, strongest removal effects, and so on). Output: a mute strength and a boost strength. Learning signal: **no stored "best strength" labels.** The edit is a smooth function of the strengths, so training runs GPT-2's last four layers (8 to 11) on a saved state with the proposed strengths, scores the result (does the target beat its strongest competitor, how much did other predictions move, how big is the edit) and adjusts. One step costs about 50 ms.

### 5.4 What it must beat
1. Your reference: fixed 0.6 / 0.5.
2. The learned fixed pair (two shared numbers).
3. The oracle (strengths found per prompt by trial and error, as an upper bound).
If the network does not beat the learned fixed pair on held-out prompts, the study can only claim a better default, not that adapting per prompt helps.

---

## 6. Part D: building the data, step by step

All commands are PowerShell, run from the project folder. Times are measured unless marked estimate.

### Step 1: the hard set
- **What:** from `datasets/counterfact.json` (21,919 records, downloaded from rome.baulab.info, 45 MB, not in git) build each prompt, keep it if the true answer is one GPT-2 token, run GPT-2 once on every prompt (in batches of 64), keep the prompts where the true answer ranks 2 to 1000.
- **Run:** `.venv\Scripts\python.exe tools/build_counterfact_set.py`
- **Result:** 21,913 usable; 1,817 already rank 1; 3,736 rank above 1000; **16,360 kept** (15,594 distinct subjects, 34 relations). Total 49.5 s. Self-check: 0 differences between batched and one-by-one ranks.
- **Files:** `data/counterfact_hard_set.json` (15.5 MB, not in git), `data/counterfact_hard_set_summary.json` (in git), `outputs/counterfact_all_ranks.json` (local).

### Step 2: the split
- **Why careful:** CounterFact has only 34 templates, each used with thousands of subjects. A random split would put near-identical prompts in train and test.
- **Rules:** a subject never appears in two sets; five whole relations are held out of training and validation (P140 religion, P641 sport, P1303 instrument, P176 manufacturer, P190 twinned city); the training order is balanced across relations and starting-rank bands so the first 250, 500 or 1,000 are each balanced and nested.
- **Run:** `.venv\Scripts\python.exe tools/split_counterfact_set.py`
- **Result:** train pool 10,292 (use the first 1,000), validation 150, test-seen 150 (new subjects of trained relations), test-unseen 150 (held-out relations). Relation P264 (6 prompts) dropped. 96 prompts removed because their subject also appears in a held-out relation. Seed 20261002.
- **Files:** `data/counterfact_split.json` and `data/counterfact_split_summary.json` (both in git).

### Step 3: the per-prompt cache
- **What it saves for each of the 1,450 prompts:** GPT-2's layer-8 state at every word; the baseline target rank and probability and the blocker; the 200 candidate features and their activations at every word; what happens to the target and blocker when each candidate is switched off alone; a few descriptors and the 12 summary numbers.
- **Why:** so training never repeats the slow work and any edit can be tried by running only layers 8 to 11.
- **Run:** `.venv\Scripts\python.exe tools/build_strength_cache.py` (resumable; `--shard i/n` splits work across machines)
- **Result:** 1,450 files, 74 MB, 0 problems (`tools/verify_strength_cache.py`). Per prompt: 0.70 s on the GTX 1660 Ti machine, 0.29 s on the RTX 4050 laptop (a 2.4 to 1 speed ratio).
- **Checks:** the saved candidate rankings were identical to your sweep's own functions on 6 of 6 prompts; the laptop's data (torch 2.5.1) was recomputed on this PC for 12 prompts: identical candidate lists, removal effects equal to about 2e-6.
- **A bug the built-in check caught:** my first version passed the wrong value to the batched function (its `scale` is a multiplier: 0.0 means removed, 1.0 untouched), so every candidate looked useless. The script's sanity check stopped it before any data was written, and it was fixed.

### Step 4: the headroom measurement
- **Question:** if the strengths could be chosen perfectly, how many prompts could reach rank 1 by scaling SAE features, and how does that compare with an edit made directly in the residual stream?
- **Run:** `.venv\Scripts\python.exe tools/measure_headroom.py` (1,060 s on the laptop). Data: `docs/Research_Journal/packs/headroom/headroom_20261002_144917.json`.
- **Method:** uses only the cache. Adds `sum of (multiplier - 1) x activation x decoder direction` to the saved state and runs layers 8 to 11. **Verified against your real hook:** the same random edit on 12 prompts gave raw outputs within 8.8e-06 and the same target rank 12 of 12.

| Method | Prompts reaching rank 1 (of 1,450) | Side effect (KL) | Edit size |
|---|---|---|---|
| `sae_reach`: free multipliers, no damage limit (ceiling) | **1,158 (80%)** | 1.04 | 35.7% |
| `sae_gentle`: damage penalised | **592 (41%)** | 0.056 | 19.4% |
| `sae_naive_5` / `_20` / `_50`: crude fixed proxy | 89 / 169 / 218 (6 / 12 / 15%) | 0.07 to 0.20 | 12 to 16% |
| free push, last token only, 5 / 10 / 20% of norm | 254 / 683 / 1,432 (18 / 47 / 99%) | 0.07 / 0.23 / 0.87 | 5 / 10 / 20% |
| free push, all tokens, 5 / 10 / 20% | 844 / 1,441 / 1,450 (58 / 99 / 100%) | 0.27 / 1.11 / 2.44 | 5 to 24% |

Results by set were the same: 79 to 85% for `sae_reach` in train, validation, test-seen and test-unseen. By starting rank: `sae_reach` reached rank 1 on 328/330 (rank 2 to 5), 340/367 (6 to 20), 296/376 (21 to 100) and 194/377 (101 to 1000); `sae_gentle` 266/330, 208/367, 92/376 and 26/377.

**What this does and does not show.**
- It shows the room: SAE scaling can in principle fix about 8 in 10 of these prompts, and a good strength choice matters (the crude fixed proxy reaches 6 to 15%).
- The 41% depends on the arbitrary penalty weights I chose (lam_kl 20, lam_size 0.01). It is an illustration, not a bound.
- The free pushes had no side-effect penalty, so their KL is not comparable with `sae_gentle`. No equal-damage comparison of SAE versus SAE-free has been made.
- None of it is your real sweep (no safety filter, no step-by-step addition, no refill). The real reference run on these prompts is still to do.

### Step 5: the strength tables
- **Why:** the strict filter's verdict on each candidate depends on the strength. The network must see the allowed features at any strength it proposes.
- **What:** for each prompt and each of its 200 candidates, the target's probability and rank when that one candidate is applied alone at 5 mute strengths (0.2, 0.4, 0.6, 0.8, 1.0) and 5 boost strengths (0.25, 0.5, 1.0, 1.5, 2.0). Uses your own `batched_ablation_probs_and_ranks`, the call the strict filter makes in round 0. The strict rule is not applied at build time (raw numbers are stored).
- **Run:** `.venv\Scripts\python.exe tools/build_strength_tables.py` (resumable, shardable). Check: `... --verify --expected 1450`.
- **Result:** 1,450 files, 0 problems; all computed on the laptop at 2.0 s per prompt (this PC: 4.6 s). Table candidate lists equal the cache's for 1,450 of 1,450. Recomputed on this PC for 8 prompts: probability difference 1.2e-06, rank difference 1 place at most. (The strict rule's threshold 1e-6 is the same size as that noise, so a candidate whose true effect is almost zero can pass on one machine and fail on the other; it affects only borderline candidates.)
- **Average candidates (of 200) passing the strict filter:** mute 0.2: 117.8, 0.4: 109.7, 0.6: 105.7, 0.8: 103.1, 1.0: 101.4; boost 0.25: 115.5, 0.5: 107.7, 1.0: 101.3, 1.5: 98.3, 2.0: 96.3.
- **The effect you described:** on 300 random prompts, a mean of 12.9 candidates (median 11) pass at mute 0.2 but not at your reference 0.6; 292 of 300 prompts have at least one. (Only this direction was counted.)

### Step 6 (measured costs the plan rests on)
| Quantity | Result |
|---|---|
| Candidate data per prompt | 0.56 to 1.17 s |
| One training step, layers 8 to 11 only | about 50 ms (86 to 91 ms through all 12 layers) |
| A batch of 32 prompts vs one prompt | the same cost |
| Layers 8 to 11 from the saved state vs the full model | identical outputs (difference 0.0) |

---

## 7. Part E: running work on two machines

- **Machines:** this PC (GTX 1660 Ti, 6 GB) and a laptop (RTX 4050, 6 GB, 24 GB RAM).
- **How:** every prompt gets its own file named by its CounterFact id, so machines never clash and a stopped job resumes by skipping finished files. `--shard i/n` splits work; several slices (`0,1,2,3,4/7`) give a faster machine a larger share.
- **Instructions for an AI agent on the second machine:** `batched_training.md` (what the project is, exact steps, hashes to check, when to stop, a report template, and a section for the strength tables).
- **Speeds:** the first estimate of "4 times faster" was wrong. It used total run times, but the laptop's 129 s covered only 450 resumed prompts. The per-prompt compute time recorded inside the files gives 0.70 s versus 0.29 s, a ratio of **2.4**. The advice was corrected to a 5:2 split.
- **Version differences:** the laptop used torch 2.5.1+cu121 instead of 2.6.0+cu124. Cross-checks showed differences of about 1e-6.
- **Timing comparisons between methods** must still be measured on one machine.

---

## 8. Mistakes and corrections (so nobody repeats them)

| What went wrong | How it was caught | Fix |
|---|---|---|
| "1 of 21 / 21 of 21" presented as if it were a success rate | The author asked where the number came from | Entry 22 rewritten with a scope warning; H14 marked unresolved |
| Diagnostic used Top-N 120, not your Top-N 200 | Review of settings | Documented; not rerun (the diagnostic is not the benchmark) |
| "Held edit disproves repair" overstated | The author asked whether the proof was valid | Documented that the control cannot differ when nothing is lost; only direction-level evidence |
| First strength-cache version passed the wrong scale | Built-in sanity check | Fixed (`scale=0.0` for full removal); check runs every time |
| Training plan used feature lists from the 0.6/0.5 run | The author pointed out it discards lower-strength features | Strength tables |
| "Days" of compute for the whole study | Review of the actual costs | About one day with two workers; most of it the real sweeps |
| 4:1 speed ratio between machines | Per-prompt times inside the files | 2.4:1, 5:2 split |
| A tab character in a log path in `batched_training.md` | Spotted on re-reading | Fixed; other docs scanned for the same damage |
| Suggested a strength-grid "ceiling check" after the author had decided strengths matter | The author objected | Dropped as unnecessary; the real test is network vs learned fixed pair |

Process note: the harness appears to halve double backslashes in commands, which broke `\\t` in one document. File contents are best written with the file-writing tool, or with `chr(9)` / `chr(92)` in scripts.

---

## 9. Where everything is

| Need | File |
|---|---|
| Plain-language report (this file) | `docs/session_report_2026-09-30_to_10-02.md` |
| Literature and novelty | `docs/literature_review_sae_editing.md` |
| Diagnostics write-up | `docs/Research_Journal/22.md` |
| Full study plan and every result since | `docs/Research_Journal/23.md` (sections 13.1 to 13.8 are the running log) |
| How to resume | `handoff.md` |
| Instructions for an AI agent on the second machine | `batched_training.md` |
| Hypotheses, experiments, timeline | `research/hypothesis_log.md` (H13 to H16), `research/experiment_index.md` (EXP-012 to EXP-015), `research/timeline.md` |
| Data pack: diagnostics | `docs/Research_Journal/packs/repair_diagnostics/` |
| Data pack: headroom | `docs/Research_Journal/packs/headroom/` |
| Scripts | `tools/build_counterfact_set.py`, `split_counterfact_set.py`, `build_strength_cache.py`, `verify_strength_cache.py`, `measure_headroom.py`, `build_strength_tables.py`, `measure_plan_costs.py`, `run_repair_diagnostics.py` |
| Library code | `src/strength_cache.py`, `src/repair_diagnostics.py` |
| App | `experiment_app.py`, last tab "Repair & SAE limit" |
| Not in git (rebuildable or local) | `datasets/counterfact.json` (download), `data/counterfact_hard_set.json`, everything under `outputs/` |

---

## 10. Rerun everything from scratch

```powershell
# 0. download counterfact.json (45 MB) from https://rome.baulab.info/data/dsets/counterfact.json into datasets/
.venv\Scripts\python.exe tools/build_counterfact_set.py                 # about 1 minute
.venv\Scripts\python.exe tools/split_counterfact_set.py                 # seconds
.venv\Scripts\python.exe tools/build_strength_cache.py                  # about 7 min on the laptop, 17 min on this PC (0.29 s and 0.70 s per prompt)
.venv\Scripts\python.exe tools/verify_strength_cache.py --expected 1450 --split-file data/counterfact_split.json
.venv\Scripts\python.exe tools/measure_headroom.py                      # about 18 min on the laptop
.venv\Scripts\python.exe tools/build_strength_tables.py                 # about 49 min on the laptop, 1.9 h on this PC
.venv\Scripts\python.exe tools/build_strength_tables.py --verify --expected 1450
```
Two machines: add `--shard 0,1,2,3,4/7` on the faster and `--shard 5,6/7` on the slower, then copy the output folders together.

---

## 11. What comes next

1. **Write the training code** (`src/strength_models.py`, `tools/train_strength_models.py`): PromptNet, the learned fixed pair, and the per-prompt oracle. Train on the 1,000 training prompts, tune on validation, with a learning-curve check (100, 250, 500, 1,000 prompts). Training takes minutes.
2. **Run the real sweeps** on the 300 test prompts with fixed 0.6/0.5, the learned fixed pair, the network and the oracle (each strength is passed per prompt through the batch spec's existing per-prompt settings, so the sweep needs no change). Hours; split 5:2 across the machines.
3. **Compare with other methods** (IKE, ROME, DiffMean) using existing implementations (EasyEdit, AxBench) in a separate virtual environment, timed on one machine; add paraphrase and neighbour checks from CounterFact.
4. **Build the comparison tab** from the saved results (time, rank-1 count, what a person can inspect).
5. **Version 2** (per-feature multipliers with an implicitly learned filter) if version 1 looks promising.
6. **Run the reference run** (131 prompts or the new test prompts) so that every headroom number has a real-system number beside it.
7. Older open items still stand (compile the paper draft, Journal Entry 21 for the filter study, add this work to the paper and its evidence audit).

### Decisions still open
- Test-set size (proposed 150 + 150) and the pre-stated reading of results (Entry 23, section 8.4).
- Whether to run version 2 in parallel with version 1.
- The size of `beta_max` (2.0) and the margin for the rank term (0.3).

---

## 12. Commits in this session (branch `pool-refill-implementation`)
`80e5409` diagnostics, tab, Entry 22 | `cf5086f` speed fix, wording correction | `5a7d787` reference spec | `0e8fb16`, `01b48fc`, `f8f65bc`, `077e251` Entry 23 plan | `bcc1ea2`, `2547612` CounterFact builder and yield | `9a53d15` split script | `523c374` cache builder | `09dbaf2` agent instructions, verifier, split files | `23e272f`, `ddf13fe` weighted sharding and speed correction | `3fdf602`, `679430c` headroom | `97d3e10`, `12ea10c` strength tables | `de532fa` strength-table results.
