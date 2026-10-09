> **Background document. The step-by-step runbook is `PLAN.md` in this folder; it supersedes this file wherever they differ.**
> Changes since this was written (2026-10-09): (1) the translator is fitted on the **residual stream directly**, not on SAE features; (2) work starts **inside the GPT-2 family** (small, medium, large), with Gemma 3 270M and 1B as a later replication; (3) an SAE is needed only on the model where the edit is made, not on both; (4) direction names are **Export** (small, edited model to bigger model) and **Import** (bigger model to small model); the side note `docs/cross_model_patching_idea.md` calls Import "Direction 1" and Export "Direction 2"; (5) the first direction to run is Export, because it reuses the existing edit tool with no new SAE; (6) GPU memory is 6 GB on the RTX 4050, not 8 GB.

# Cross-model fact transfer through the residual stream: implementation plan

Written 2026-10-09 on branch `gradient-descent-editing`. Status: plan only, nothing built or run. Every number quoted from earlier work names its journal entry; everything else is a proposal for the author to accept or change.

---

## 1. The idea in one example

Prompt: "The Eiffel Tower is in". Model S (the source) is edited so that it answers " Rome". We record how the edit changed S's internal state, translate that change into the internal language of a second model R (the receiver), and add it to R's residual stream during R's own forward pass on the same text. No weights change in either model. Question: does R now answer " Rome", does it keep answering " Rome" when the question is reworded, and does it leave unrelated facts alone?

The same machinery can run in the other direction: a model that already knows a fact (or has been edited to hold it) donates its state, and the receiver takes only the parts its SAE identifies as relevant.

## 2. Four corrections to the raw idea (read these first)

**2.1 You cannot subtract one model's features from another model's features.** The raw idea computes a delta between what fires in model 1 and what fires in model 2. Feature #11149 of the GPT-2 small SAE and feature #11149 of any other SAE are unrelated; the residual streams are in different coordinate systems (often of different widths: GPT-2 small is 768 wide, GPT-2 medium 1024). Subtracting them is like subtracting a street address from a GPS coordinate. The sound version:
1. Compute the delta **inside one model**: `delta_S = h_S(edited) - h_S(clean)`, on the same text. This is exactly what our edits already produce.
2. **Translate** it with a learned map `T` (fit once on a shared text corpus): `delta_R = T(delta_S)`.
3. **Inject**: `h_R' = h_R + alpha * delta_R` at a chosen layer of R.

Because only a difference is translated, the bias of an affine map cancels; only the linear part of `T` matters.

**2.2 A shared tokenizer buys aligned positions, not shared meaning.** With the same tokenizer, token *i* in S is token *i* in R, so per-position deltas can be injected position by position. That is a real convenience, but it does not make the residual coordinates compatible. Even GPT-2 small and GPT-Neo 125M (both 768 wide, same tokenizer) have unrelated axes. A translation map is always needed. The identity map between those two models is a useful negative control (it should fail).

**2.3 What the SAE is actually for in this system.** Not for subtracting across models. It has four legitimate jobs: (a) making the edit in S (our gradient-descent multiplier edit lives in SAE feature space); (b) cleaning what is sent or received (keep a sparse, readable set of features instead of a dense vector); (c) explaining (which features were sent, which arrived, with Neuronpedia links); (d) in the donor direction (section 5.2), the receiver's SAE decides which parts of the translated donor state to accept. Job (d) is the closest sound version of "compare the features both models fire for the same sequence and pool them".

**2.4 Which model is "superior".** GPT-2 small is the model we can edit and explain best (SAEs, all our tooling), not the strongest. So there are two honest framings, and the plan covers both:
- **Export (Direction 1):** edit in the small, interpretable model; transfer the edit to a larger model. "Edit once where you can see what you are doing, apply elsewhere."
- **Import (Direction 2):** a larger model that knows the fact (or was edited with ROME) donates; GPT-2 small receives, and its SAE filters what it accepts. "A stronger model teaches a weaker one at inference time." This is the version the raw idea describes most closely.

## 3. Feasibility facts checked today

- **SAEs available in the installed sae-lens (6.45.3):** for GPT-2-tokenizer models, only GPT-2 small (`gpt2-small-res-jb`, OpenAI `v5-32k` / `v5-128k` resid_mid / resid_post, attn/mlp SAEs). No SAE for GPT-2 medium, large, XL, DistilGPT-2 or GPT-Neo. Other SAEs exist for Pythia-70m (different tokenizer) and Gemma 2 / Gemma 3 including `gemma-3-270m` (different tokenizer). Not checked: SAEs on Hugging Face outside the sae-lens registry.
- **Consequence:** with a shared tokenizer, only GPT-2 small has an SAE. That is enough for both directions (the SAE sits at the source in Direction 1 and at the receiver in Direction 2). An SAE on both sides needs either training one (section 6, Phase 6a) or a cross-tokenizer pair (Phase 6b).
- **Candidate partner models (same tokenizer, TransformerLens supports them, fit a 6 GB GPU in float32 except XL):**

| Model | Layers | Width | Training data | Role |
|---|---|---|---|---|
| GPT-2 medium | 24 | 1024 | WebText (same as small) | **Main partner.** Same data family, so maps should fit well; knows more facts. ROME hparams exist in `third_party/rome/hparams/ROME/gpt2-medium.json`. |
| GPT-2 large | 36 | 1280 | WebText | Second partner, stronger donor; ROME hparams exist. ~3.1 GB float32. |
| GPT-Neo 125M | 12 | 768 | The Pile | Same width, different data: the identity-map negative control and a harder translation. |
| DistilGPT-2 | 6 | 768 | distilled from GPT-2 | Optional. |

- **ROME:** working on GPT-2 small (edit layer 3, `docs/rome_baseline/PROGRESS.md`, steps 0 to 5 done). Its steps 6 and 7 (frozen benchmark, shared metrics, adapters) overlap with Phase 0 below and should be built once and shared.

## 4. Phase 0: are we editing a fact or steering an output? (on GPT-2 small alone)

This answers the author's first question and decides how the whole transfer study may be described. It does **not** block the transfer work: Phases 1 to 3 can run in parallel.

**What we already know (do not re-derive):**
- Additive edit: pushes 20 of 20 random words to rank 1; transfer to rewordings is the same for random and true targets; its additive part points along the target's unembedding (lens rank 5 for " India" in `packs/additive_control/mechanism_check.json`). Reading so far: output-direction injection, not a fact (Entry 30 sections 10 to 12, H23).
- Multiplier edit: modest fact-related transfer to rewordings (median rank 36 from 94 for true answers; random words stay near 11,832), control not rank-matched (Entry 30 section 11).
- ROME on GPT-2 small, 100 dev records: efficacy 1.000, paraphrase 0.985, neighbourhood 0.663 at layer 3. That is what a weight-level fact edit looks like on this model, and the yardstick to compare against.

**Tests (same CounterFact records for every arm):**

| Test | What it separates | Built? |
|---|---|---|
| 0.1 ROME-style scores: efficacy, paraphrase, neighbourhood (CounterFact `paraphrase_prompts`, `neighborhood_prompts`), generation | Fact edits generalise to rewordings and spare neighbours; output pushes do neither cleanly | ROME side yes; adapter for our arms = ROME step 6/7 (todo) |
| 0.2 Rank-matched random-target control | Removes the "random words start at rank 17,504" confound | No (planned in roadmap A) |
| 0.3 Direction test at scale: cosine, percentile and logit-lens rank of the edit vs the target's unembedding | An edit that is mostly "the word's output direction" is steering | Yes, single prompt (`tools/test_additive_mechanism.py`); needs a batch loop |
| 0.4 **Position test**: restrict the edit to the subject tokens only vs the last token only | Facts are retrieved at the subject (ROME's locus); an edit that works only at the last token acts on the read-out. A subject-only edit that still flips the answer and its paraphrases is the strongest evidence of an association edit we can get without changing weights | No: needs a position mask in the gradient-descent edit (new function, old code untouched) |
| 0.5 Same subject, other questions: reuse the multipliers tuned on "X is in" on other relation prompts about X (CounterFact `generation_prompts`) | A changed association should move related questions; a read-out push should not | Partly (`tools/test_generalisation.py` applies tuned multipliers to new prompts) |
| 0.6 Context test ("Do not say X") | Already measured (Entry 30 section 15) | Yes |

**Pre-stated reading (proposed as H24, the author confirms before the run):**
- "Association-like" if paraphrase success is clearly above the rank-matched random control (paired test, p < 0.01) **and** the subject-only edit keeps at least half of the full edit's effect.
- "Read-out steering" if paraphrase lift is close to the random control **or** the edit is dominated by the target's unembedding (lens rank 10 or better).
- "Mixed" otherwise. Report whichever comes out.

**What it changes:** if the answer is "steering", the transfer study is still valid but must be described as transferring a steering edit, and the paraphrase and neighbourhood numbers on the receiver become the main evidence of anything fact-like. If "association-like", we may say a fact edit was transferred, within the limits of those measures. Either way avoid "the model believes"; report behaviour.

## 5. System design

### 5.1 Notation and parts
- S: source model, layer `L_S`. R: receiver model, layer `L_R`. Same input text, same tokens.
- **Edit source** (how `delta_S` is produced): `gd` (multiplier edit, `run_gradient_descent_edit`), `gd_add` (additive, cap 0.25 only, as a known-leaky arm), `rome` (ROME-edited weights, delta read at `L_S`), `context` (the fact stated in the prompt: "Fact: the Eiffel Tower is in Rome. The Eiffel Tower is in", minus the plain prompt, last position; this is roadmap item D reused as a source), `swap` (positive control, 5.4).
- **Translator** `T`: `ridge` (affine least squares with ridge, the main one), `procrustes` (orthogonal, only when widths match), `identity` (only when widths match; negative control), `random_rot` (random orthogonal; negative control), later `feature_match` (Phase 6).
- **Injector:** a TransformerLens hook on R at `blocks.{L_R}.hook_resid_pre` adding `alpha * T(delta_S)` at every position, the last position only, or the subject positions only.
- **Relay mode vs vector mode.** Relay: for every new input (paraphrase, follow-up prompt, each generated token), S runs with its edit, the delta is recomputed, translated and injected. This works because the multiplier edit is a set of feature multipliers, reusable on any text. Vector mode: one delta computed on the edit prompt, added as a fixed vector. Relay is the main mode (it is what "R receives S's state" means); vector mode is the cheap baseline.

### 5.2 The two directions

**Direction 1, Export (S = GPT-2 small with SAE, R = GPT-2 medium / large / GPT-Neo).** Our edit happens in S at layer 8. Read `delta_S` at layer 8 or a later layer of S (the edit's consequences propagate; which layer translates best is measured, not assumed). Translate, inject into R at the depth-matched layer.

**Direction 2, Import (S = GPT-2 medium / large, clean if it knows the true fact, ROME-edited for a counterfactual; R = GPT-2 small with SAE).** This is the raw idea made sound:
1. Run both models on the same text. Translate the donor state into small's space: `h_hat = T(h_S)` at layer 8 of small.
2. Encode both with small's SAE: `f = SAE(h_R)`, `f_hat = SAE(h_hat)`. Feature differences: `df = f_hat - f`.
3. **Pool**: choose which differing features to accept.
   - `all`: every differing feature.
   - `topk`: the k largest |df|.
   - `agree`: only features whose SAE reconstruction of `h_hat` is trustworthy (low SAE error on the translated state).
   - `causal` (secondary, uses the target): features ranked by first-order effect on the target margin, reusing the project's first-order candidate code. Needs its own control, since knowing the target can do the work by itself.
4. Inject only the decoded delta of the accepted features: `delta = sum_k df_k * W_dec[k]` (delta only, reconstruction error never injected, the project's existing rule).
The main arms must be target-free (`all`, `topk`, `agree`): the receiver is never told the answer; only the donor's state carries it. That is the strongest claim the system could support, so it gets the main table.

### 5.3 Controls (the lesson of Entry 30: rank 1 alone proves nothing)
| Control | Answers |
|---|---|
| Norm-matched random vector in R | Does any push of this size do it? |
| R's own target-unembedding push, same norm | Is what arrives just "say this word"? If transfer is no better, the fact did not transfer, the word did. |
| Identity and random-rotation translators | Is the learned translation doing the work? |
| Rank-matched random targets edited in S and transferred | Same as Entry 30, in the receiver |
| Fact stated in R's prompt | The plain prompting baseline a reader will ask for (AxBench) |
| R's own native edit, where R has an SAE (Direction 2: GPT-2 small's own gd edit) | Ceiling: transfer vs editing the receiver directly |

### 5.4 Positive control before any fact: subject swap
`delta_S = h_S("The capital of Germany is") - h_S("The capital of France is")` (single-token subjects, so positions line up). Translate, add to R on "The capital of France is". R should move toward " Berlin". R's own native swap delta is the ground truth to compare with (cosine and effect size). If this ladder rung fails, fact transfer will not work and the problem is the map, not the edit. Cheap, and it validates the whole pipeline end to end.

### 5.5 Measures on the receiver (one yardstick for every arm)
Efficacy (target rank 1; target_new above target_true for counterfactuals), paraphrase success, neighbourhood specificity, leak on the 12 neutral prompts already used (KL, top-1 flips), generation over 20 tokens with the relay kept on (distinct-2, loop rate, against R's own baseline), edit size as % of R's residual norm, transfer efficiency (effect in R divided by effect in S, per prompt). All reported with the S-side numbers next to them.

## 6. Phases, steps and gates

Gates are proposals: the author sets the numbers before each run.

### Phase 0: fact or steering (section 4)
Steps: build the shared CounterFact benchmark and metrics once (this is ROME step 6), adapters for `gd`, `gd_add`, `rome`, `context`; rank-matched random targets; batch direction test; subject-only / last-only position masks (new function in a new file). Run on the frozen ROME test records.
Deliverable: Entry 31, H24 reading. Cost: roughly 1 to 2 h GPU per arm on 150 records (from Entry 29 timings, 7.7 s per gd edit plus paraphrase evaluation).

### Phase 1: translation infrastructure and its quality
1. **Tokenizer check:** assert identical vocabularies and identical token ids on a test list for every pair; stop if not.
2. **Paired activations, streamed:** run S and R on the same wikitext-103 text (already used for ROME's covariance; about 200k tokens, BOS position excluded because its norm is huge), accumulate `X^T X`, `X^T Y`, means for a grid of layer pairs (every second layer). Nothing large is stored; the ridge map is solved from these sums. Both directions (S to R and R to S).
3. **Map quality, three numbers per layer pair:**
   - held-out R^2 and cosine of `T(h_S)` vs `h_R`;
   - **stitching**: replace R's residual at `L_R` with `T(h_S)` and run R forward; fraction of next-token loss recovered compared with mean-ablation at that layer;
   - **difference fidelity**: for minimal-pair prompts (subject swapped), cosine between `T(h_S(a) - h_S(b))` and `h_R(a) - h_R(b)`, against a random baseline. This is the number that matters most, because we translate differences, not states.
4. **SAE sanity in Direction 2:** SAE reconstruction error and L0 on translated states vs on small's own states (a translated state that the SAE cannot read makes feature pooling meaningless).
Gate 1 (proposed): at least one layer pair with stitching loss recovered at or above 0.8 and difference fidelity clearly above random. If no pair passes, stop and rethink (nonlinear map, or a pair from the same family only).
Cost: 20 to 40 min GPU per model pair.

### Phase 2: positive control (subject swap, 5.4), both directions
Choose `L_R` and `alpha` on dev pairs. Gate 2: transferred swap moves R toward the swapped answer clearly more than the norm-matched random and the identity / random-rotation controls. If not, return to Phase 1.

### Phase 3: Direction 2, Import into GPT-2 small (the raw idea, made sound)
Donors GPT-2 medium (main) and large. Two fact sets:
- **True facts:** CounterFact records where GPT-2 small is wrong (our hard set) and the donor is right (filter by the donor's rank; report how many qualify).
- **Counterfactuals:** donor ROME-edited (official hparams for gpt2-medium / large, run in `.venv-rome`, edited weights saved and loaded into TransformerLens in the main venv).
Arms: pooling `all` / `topk` (k in {10, 50, 200}) / `agree` / `causal`; translators `ridge` vs controls; positions all / last / subject. Tune k, `alpha`, layer on dev; freeze; test.
Gate 3: target-free pooling beats every control in 5.3 on efficacy, with neighbourhood and leak no worse than GPT-2 small's own gd edit.

### Phase 4: Direction 1, Export from GPT-2 small
Sources `gd`, `gd_add@cap=0.25`, `rome` (small, layer 3), `context`. Receivers GPT-2 medium (main), GPT-Neo 125M (different data; identity control is meaningful there), large if time. Relay mode main, vector mode baseline. Same tuning-then-freeze protocol, same controls.

### Phase 5: frozen test run and analysis
Both directions, 150 test records per fact set (proposed), dev records excluded (including the 100 ROME dev records). Paired tests (McNemar for success, Wilcoxon for ranks), bootstrap 95% intervals, per-band breakdown (start rank 2-10, 11-100, 101-1000 as in Entry 29). One combined table. Data packs and figures from real data only.

### Phase 6: SAE on both sides (optional, after Phase 5)
- **6a.** Train an SAE on GPT-2 medium at the chosen `L_R` with sae-lens (hours on the 4050; the author runs it). Then: feature correspondence by activation correlation on wikitext (plus decoder-through-map cosine), a feature-to-feature translator, and a readable table "small feature A <-> medium feature B, correlation r". Compare against the dense ridge map.
- **6b.** Or a cross-tokenizer pair where both SAEs exist: GPT-2 small and Gemma 3 270M (Gemma Scope SAEs in the registry). Last position only (string-offset alignment for other positions is extra work). This is also the step toward the author's point that newer models like Gemma are better.
- **Validation of the feature matcher (cheap, do first):** two different SAEs on the same GPT-2 small point (`gpt2-small-res-jb` at `blocks.8.hook_resid_pre` and OpenAI `v5-32k` resid_post layer 7, the same activation). The true map there is the identity, so the matcher's loss can be measured exactly.

### Phase 7: tester app
A new, separate Streamlit app (`transfer_app.py`), so `experiment_app.py` stays untouched. Panels: choose S, R, direction and layers (with the Phase 1 map-quality heatmap as a guide); prompt and target; run the S edit or donor; show the sent / accepted features with Neuronpedia links; alpha slider with a live dose-response; R before vs after (rank, probability, top-5); control toggles (random, unembedding push, identity map, fact-in-prompt); paraphrase / neighbour / neutral-prompt panel; side-by-side generation with the relay kept on. Built after Phase 5, around the arms that actually worked.

### Phase 8: write-up
Journal entries, hypothesis log (H24 onward), experiment index, timeline, data packs; paper section updated last, in the project's style (no hype, withdrawn claims stated).

## 7. Code layout (new files only; existing code is imported, not edited)

```
src/transfer/__init__.py
src/transfer/models.py        load S and R (TransformerLens), tokenizer identity check, depth-matched layer lists
src/transfer/paired_acts.py   streamed sufficient statistics on wikitext for a grid of layer pairs
src/transfer/maps.py          ridge / procrustes / identity / random_rot; save, load; R^2, stitching, difference fidelity
src/transfer/deltas.py        delta_S from gd, gd_add, rome, context, swap; per position; relay helper
src/transfer/pooling.py       Direction 2: SAE encode of translated donor state, all / topk / agree / causal
src/transfer/inject.py        receiver hook (positions, alpha); controls (random, unembedding push)
src/transfer/metrics.py       efficacy, paraphrase, neighbourhood, leak, generation, transfer efficiency
src/transfer/feature_match.py Phase 6
src/gradient_editing_masked.py  position-masked gd edit for Phase 0.4 (wraps, does not modify, gradient_editing.py)
tools/transfer/fit_maps.py, check_maps.py, test_fact_or_steer.py, run_swap_control.py,
tools/transfer/run_import.py, run_export.py, analyse_transfer.py   (arm and override syntax as in compare_sweep_vs_gradient.py)
transfer_app.py
docs/cross_model_transfer/PLAN.md (this file), PROGRESS.md (step table, as in docs/rome_baseline)
```
Every tool gets a `--quick` smoke mode and built-in self-checks, as the existing tools do. A small unit test per translator (a known random linear map must be recovered) and for the injector (alpha = 0 reproduces R's clean logits exactly).

## 8. Data and splits
- Map fitting: wikitext-103 only, never CounterFact (no leakage into the evaluation).
- Facts: CounterFact (`datasets/counterfact.json`, not in git; `scripts/get_data_mac.sh` or the Windows instructions in `handoff.md`). Single-token targets only (the shared tokenizer keeps targets single-token in both models).
- Dev: about 50 records per fact set for choosing layer pair, alpha, k. Test: 150 per fact set, frozen before any test run, ROME's 100 dev records excluded.
- Every run stores per-prompt rows (jsonl) so any table can be recomputed.

## 9. Risks and how each would show up
| Risk | Where it shows | Response |
|---|---|---|
| Linear maps between models are too lossy | Phase 1 stitching and difference fidelity | Try depth pairs more widely, then a small MLP map; stop if same-family maps fail |
| Edits are off-distribution for a map fit on ordinary text (additive edits are 100%+ of the residual norm) | Difference fidelity vs norm; transfer fails only for large edits | Use multiplier edits as the main source; report by edit size |
| What transfers is the word, not the fact | Unembedding-push control matches the transfer; random targets transfer equally | Report as steering transfer (section 4 reading) |
| GPT-2 small's SAE cannot read translated states | Phase 1 step 4 (reconstruction error) | Pool only where `agree` holds, or move Direction 2 to raw-residual injection |
| GPT-2 small cannot use a fact even when its state is pushed toward it | Native gd ceiling vs transfer | This is a property of the receiver; state it |
| Outlier dimensions and the BOS position dominate the fit | Map R^2 driven by few dims | Exclude BOS, standardise per dimension before fitting |
| Memory on 6 GB GPUs | GPT-2 large + small + SAE | float16 for the donor, or run donor and receiver in sequence and cache states |
| Prior work already does this | Literature check | See section 10; claim only what is new |

## 10. Related work to check before writing any novelty claim
From memory, to be verified (titles and details may be off; a proper search is needed before citing):
- Cross-model activation patching with learned affine maps (Patchscopes, Ghandeharioun et al., 2024).
- Transferring steering interventions between models through learned maps between activation spaces (Oozeer et al., 2025).
- Transferring SAEs, probes and steering vectors between models of one family through affine maps between residual streams ("model stitching", Chen et al., 2025).
- Universal or shared SAE feature spaces across models (Lan et al., 2024); model stitching in general (Bansal et al., 2021).
If these hold up, "transferring a steering vector across models" is not new. What could be new here: transferring a per-prompt, gradient-descent-tuned **SAE feature edit of a specific fact**; target-free, receiver-side SAE pooling of a donor's state (Direction 2); and the fact-vs-steering yardstick with rank-matched controls applied to the receiver. Same honesty rule as the rest of the project: engineering and measurement first.

## 11. Decisions for the author
1. Main partner model: GPT-2 medium (proposed), plus GPT-Neo 125M as the different-data case.
2. Which direction first: Import (Direction 2, closest to the raw idea, needs no new SAE) is proposed first after the positive control; Export (Direction 1) second.
3. Whether Phase 0 runs as ROME steps 6 and 7 (shared benchmark), in parallel with Phases 1 and 2.
4. Gate numbers (stitching 0.8, p < 0.01, "half of the effect" in the position test) and test-set size (150 per fact set).
5. Phase 6 route: train an SAE for GPT-2 medium, or a Gemma 3 270M pair, or skip.
6. The app as a new file (`transfer_app.py`) rather than a new tab in `experiment_app.py`.
7. A new branch for this work (proposed `cross-model-transfer`, from `gradient-descent-editing`).
