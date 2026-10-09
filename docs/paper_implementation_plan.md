# Implementation plan: turning the gradient-descent editing work into a paper

Written 2026-10-08 from a read of `origin/gradient-descent-editing` (tip `8b88fbf`). Meant to be handed to a person or an agent together with the context below. Everything here is a plan, not a result. Literature names are from memory and must be verified before citing.

---

## 0. Context for whoever picks this up

**Project.** Transient activation editing in GPT-2 small. SAE `gpt2-small-res-jb`, layer 8, hook `blocks.8.hook_resid_pre`. During one forward pass, SAE features are edited so a TRUE answer that GPT-2 ranks below first becomes its top prediction. No weights change. Only the SAE reconstruction delta is applied: `delta = sum_k (m_k - 1) * act_k * W_dec[k]`.

**Main method (Entries 28, 29).** `run_gradient_descent_edit` in `src/gradient_editing.py`. The top-200 candidate features (by peak activation) each get one multiplier in [0, 3], starting at 1. Adam, 100 steps, lr 0.1. Loss = rank hinge (margin 0.3) + 3.0 x KL (only once the target leads) + 0.005 x size penalty. The best iterate is kept and re-checked through the real hook.

**Optional additive edit (Entry 30).** Switches on silent features at the last position. It is too strong to count as a fix: it pushes 20 of 20 random words to rank 1.

**Results so far (exploratory, proxy-labelled).**
- 300 held-out hard CounterFact prompts, rank 1 reached: gradient descent 271, sweep 1.0/2.0 161, sweep 0.6/0.5 94. About 5-6x faster. Lower same-prompt KL on 160 of 161 shared successes (partly by construction: only gradient descent has a KL term).
- Median target probability at rank 1 is only 7.7%.
- Additive edit: random targets 20/20, transfer to rewordings equal for random and true targets, heavy leakage ("Germany" continues "Hong Kong").
- Multiplier edit: modest fact-related transfer (true answers median rank 94 -> 36 on rewordings; random words stay near 11,832), but the control is not rank-matched.
- Withdrawn: "unreachable by steering = not in the model" (Entry 30 s12, H23).
- Measurement trap already hit: comparing minimal edit sizes of true vs random targets is circular (start ranks 17 vs 17,504).

**Why it is not yet a paper (reviewer objections).**
1. No outside baselines (prompt with the fact, DiffMean, IKE, ROME, fine-tuning).
2. Success = rank 1 at one position; weak metric.
3. KL comparison with the sweep is partly circular.
4. Specificity not measured at scale (rewordings, neighbours, leakage, generation).
5. One model, one layer, one SAE; hyperparameters not validated (defaults reused from the earlier network study).
6. The method is given the target; as fact correction it competes with putting the fact in the prompt.
7. `research/paper/draft_v1` does not contain Entries 22-30.

**Author's working rules (keep to these).** Plain language first. Do exactly what is asked; propose extras and wait. New code must not change old code (ask first). No em dashes, no hype, no novelty claims in paper text; every claim traceable (`research/paper/draft_v1/EVIDENCE_AUDIT.md`). Figures and tables only from real data. Do not launch long jobs without saying so; give the command and let the author run it. Timing comparisons valid only on one machine. Never commit `outputs/`. Ask before committing or pushing.

---

## 1. The paper in one sentence

> How to tell a real activation edit from merely forcing a word out: a specificity-aware evaluation of SAE feature editing, with a gradient-descent editor, checked against prompting and established editors.

Frame it as an evaluation-and-methods study. Do not frame it as "a new state-of-the-art editor". Negative results (an arm that fails the controls) are reported as findings.

### Research questions
- **RQ1 (specificity).** Does the edit work for true facts but not for random words matched on starting rank, and does it leave unrelated prompts alone?
- **RQ2 (baselines).** Does it beat putting the fact in the prompt, DiffMean, and one of IKE or ROME on the same measures?
- **RQ3 (candidate choice).** Does choosing candidate features by output effect beat choosing them by peak activation?
- **RQ4 (usability).** Can stored edits be applied only when a matching prompt appears (a gate) without false triggers?

### Work order
Phase 1 is required. Phases 2-3 are required for the paper to be credible. Phases 4-5 are optional strengtheners. Phase 6 is the write-up.

---

## 2. Phase 0: setup (about half a day)

1. Check out the branch: `git fetch origin && git checkout gradient-descent-editing` (or reset the working branch to it). In this cloud session the local branch was still at the old commit `9af68ec`; the branch content was only read through `git show`.
2. Environment: `bash scripts/setup_mac.sh` on the Mac, or `requirements.txt` on Windows (note: it pins `torch==2.13.0`, which does not match the Windows environment that produced the numbers). Use `.venv/bin/python`.
3. Data: `bash scripts/get_data_mac.sh` (downloads CounterFact, checks SHA-256 `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`, builds `data/counterfact_hard_set.json`, expected SHA-256 `657b475e3cc37c7fb7c24a2c5c9d376791c70370cc894dca5ac3db31a3526ec8`). `data/counterfact_split.json` is already in git.
4. Parity check, FIRST: `python tools/check_mac_parity.py` (add `--full` for the additive checks). If it fails on `mps`, use `FEATURESCALPEL_DEVICE=cpu` and note which line failed. The MPS path has never been verified.
5. Pick ONE machine for all timing comparisons and write its spec in the paper.
6. Create `docs/Research_Journal/31.md` and `research/hypothesis_log.md` entries (H24+) before running anything, with the predictions below written down in advance.

---

## 3. Phase 1: specificity benchmark (required; answers RQ1)

Most pieces already exist: `tools/test_generalisation.py`, `tools/test_generation_quality.py`, `tools/control_random_targets.py`, `tools/compare_sweep_vs_gradient.py`. Verify each tool's CLI flags by reading the file before running (not read in detail yet).

### 1.1 Rank-matched random control (new tool, new file)
- For each true-target prompt with starting rank r, draw a random target word whose starting rank on that prompt is within a band around r (for example 0.8r to 1.25r, and also log-spaced bands).
- Targets must be single tokens and not equal to the true answer or the blocker.
- Write `tools/make_rank_matched_controls.py` (new file). Output: a JSON of (prompt, true target, matched random target, start ranks).
- Pre-registered prediction to record in Entry 31 before running: if the edit is fact-specific, true targets reach rank 1 more often or with smaller edits than matched random targets. If the rates are equal, the edit is just a strong push and the paper says so.

### 1.2 Arms
Run all arms on the same prompts with the same measures:
| Arm | How |
|---|---|
| `sweep_ref` | fixed 0.6 / 0.5 |
| `sweep_wide` | 1.0 / 2.0 |
| `gd` | gradient-descent multipliers (default) |
| `gd_klall` | `kl_always=True` |
| `gd_add@cap=0.25`, `gd_add@cap=1.0` | additive, two caps (use the override syntax from `compare_sweep_vs_gradient.py`) |

The sweep's final edits are not stored in the Entry 29 pack, so the sweep must be re-run and its multipliers saved.

### 1.3 Measures (per prompt)
1. **Reach:** target rank 1 on the tuned prompt; also report target probability, not just rank.
2. **Transfer:** target rank/probability on reworded prompts (not tuned).
3. **Neighbours:** prompts about a related but different fact; the target should NOT rise.
4. **Leak:** full-vocabulary KL and top-1 flip rate on unrelated prompts (use at least 100, not 20).
5. **Text quality:** distinct-2, loop rate, tail perplexity over 20 generated tokens, judged against the unedited baseline's own repetition (greedy GPT-2 loops on its own).
6. **Context sensitivity:** is the target still forced when the prompt says "Do not say X"?

### 1.4 Scale and statistics
- Transfer and leak tests on about 150 prompts (the 40-prompt runs are too small).
- Paired tests across arms on the same prompts (McNemar for rank-1 success; Wilcoxon on KL and target probability). Report confidence intervals (bootstrap) for every headline rate.
- Rough cost from earlier runs: 1-1.5 h of GPU per 40-prompt run, so budget several hours for 150 prompts times the arms. The author runs the long jobs; give the command and the expected time.

### 1.5 Deliverable
One combined table (arm x measure) with paired statistics, committed under `docs/Research_Journal/packs/specificity_benchmark/` with a README, plus Entry 31.

**Exit criterion for Phase 1:** a clear statement for each arm: "specific", "strong but not specific", or "unclear", backed by the matched-control numbers.

---

## 4. Phase 2: outside baselines (required; answers RQ2)

Run on the SAME prompts and measures as Phase 1.

1. **Fact in the prompt (cheapest, do first).** Prefix "Fact: <subject> <relation> <target>." then the original prompt. Measures: rank/probability of the target, transfer to rewordings (prefix kept), leak on unrelated prompts (prefix present), fluency. This is the baseline a reviewer will ask for first. AxBench-type results suggest prompting can beat SAE steering, so expect it to be strong.
2. **DiffMean-style vector.** Mean residual difference between prompts with and without the fact, added at layer 8 with a scale alpha tuned on a validation split. Also covers the in-context fact vector idea in `docs/research_roadmap.md` section D (raw residual and SAE-feature variants).
3. **IKE** (in-context editing with demonstrations) or **ROME** (rank-one weight edit). Prefer an existing implementation (for example EasyEdit; verify it supports GPT-2 small) over writing one. Record that ROME changes weights while the gradient-descent edit does not, and report that difference rather than hiding it.
4. Optional: a short fine-tuning baseline on one fact, to answer the "fine-tuning is better" objection.

**Exit criterion:** a ranked table of all methods on reach, transfer, leak and fluency. If the gradient-descent edit loses to prompting, say so and position the contribution as inspectability and reversibility plus the evaluation lessons.

---

## 5. Phase 3: validation and fairness fixes (required)

1. **Hyperparameter validation.** On `--split val` only, sweep the rank margin {0.3, 1.0, 2.0} and side-effect weight {1, 3, 10}. Freeze the winner, then touch the test split once. The current defaults were reused from the earlier network study and never tuned for this method.
2. **`kl_always` at scale** (arm `gd_klall` above): shows whether lower KL than the sweep survives when the failing prompts are also penalised.
3. **Give the sweep a fair KL.** Compare at matched KL, or add a KL-aware selection for the sweep, so the "lower KL" claim is not by construction.
4. **Probability, not just rank.** Report the rate of target probability above a threshold (for example 20%) next to rank 1.
5. **Candidate source.** Compare "all positions" vs "last token only" (never compared).

---

## 6. Phase 4: candidate selection by output effect (optional; answers RQ3)

Current candidates are the top-200 by peak activation. Related work argues features chosen by their effect on the output steer better than features chosen by activation.

1. New code path (new file, do not change `get_top_active_features`): score each active feature by the first-order effect of its decoder direction on the target margin (the same gradient trick `silent_candidates` already uses), keep the top 200 by that score.
2. Run on the 300 held-out prompts and compare with the default using the Phase 1 measures.
3. Record both results whichever way they go.

---

## 7. Phase 5: gated edit library (optional but it is the usability story; answers RQ4)

Today every edit needs about 100 gradient steps, which is too slow to serve. The idea: tune once, store, apply only when a matching prompt appears.

1. Tune and store edits for N = 50 facts (fids, multipliers, the original prompt's SAE-activation signature at the last position).
2. Gate: cosine similarity of the incoming prompt's last-position SAE activation vector to each stored signature; apply the stored edit only above a threshold chosen on a validation split.
3. Measure on three groups: the stored prompts (hit rate), their rewordings (should trigger), and unrelated or neighbouring prompts (false-trigger rate; should not trigger).
4. Scale N to 50, 200, 1,000 and measure interference when several edits could fire together (multi-edit interference).
5. Expected risk: rewordings may fall below the threshold, or neighbours may fire it. A measured limitation is an acceptable outcome.

Related idea, a different failure: amortising the optimisation with a network. FeatureNet reached about 31-32% against 73% for direct tuning in Entry 27. Gradient descent is now a much better teacher (271 of 300). Training a network on its solutions over the 16,360 hard prompts is untried; run it only if Phase 1-3 finish early.

---

## 8. Phase 6: write-up

1. Run `tools/make_journal_figures.py` and the report-figure scripts only from committed packs. No figure without data behind it.
2. Update `research/paper/draft_v1/main.tex` (not compiled so far: no LaTeX on the author's machines) with Entries 22-31. Check claims against `EVIDENCE_AUDIT.md`.
3. Structure:
   - Introduction: the problem is telling a real edit from forcing a word.
   - Method: the gradient-descent editor (Entry 28 section 2 and the "A Gradient Descent Based Approach" draft in `docs/session_2026-10-06_1623.md` section 6).
   - Evaluation protocol: reach, transfer, neighbours, leak, text quality, context sensitivity; rank-matched controls.
   - Results: ranked table of all arms and baselines.
   - Negative and withdrawn results stated plainly: additive edit passes rank 1 for random words; the "knowledge absent" rule withdrawn; the BOS thread-race bug and which runs it affected.
   - Limitations: one model, one layer, one SAE, rank-1 metric, proxy status, target given to the method.
4. Style: no em dashes, no hype, no novelty claims. Claim only what the tables show.

---

## 9. Suggested schedule

| Day | Work | Output |
|---|---|---|
| 0 | Phase 0 setup, parity check | working environment |
| 1 | 1.1 matched controls; 1.2 sweep re-run | controls file, sweep edits stored |
| 2 | 1.3-1.4 benchmark runs (long jobs) | specificity table with CIs |
| 3 | Phase 2 baselines: prompt-fact first, then DiffMean, then IKE or ROME | ranked baseline table |
| 4 | Phase 3 validation sweep and `kl_always` | frozen settings, final test run |
| 5 | Optional: Phase 4 or Phase 5 prototype | extra table |
| 6+ | Entry 31, figures, paper draft | draft |

Phase 1 is the minimum. If time runs out, Phase 1 + prompt-fact baseline + Phase 3 still gives a defensible workshop-level paper.

---

## 10. Open decisions for the author

1. Keep the full BOS fix or cut it to the per-run self-heal only (it edited old files `src/editing.py` and `src/hybrid_runner.py` without asking first).
2. Keep the additive checkbox in the shipped prototype at all (default OFF with a warning now).
3. Merge `gradient-descent-editing` into `main`?
4. Which established editor to use: IKE (no weight change, closest in spirit) or ROME (the classic weight-edit baseline).
5. NSFW or content-filter direction (idea in `docs/session_2026-10-06_1623.md` section 11): out of scope for this paper unless the author says otherwise. It needs a banned-token list and a classifier as baselines.
6. Open housekeeping from the 2026-10-06 session: `stash@{0}`, the `.claude/launch.json` change, ignoring `outputs/session_history/`.

---

## 11. Risks

- **Prompting beats the edit.** Likely; the honest framing is inspectability, reversibility, and the evaluation lessons.
- **The multiplier edit fails the matched control.** Then the finding is "strong push, weak fact specificity", which is still a result.
- **MPS numerics differ from CUDA.** Ranks can shift by one place between near-tied words; the parity script tolerates KL within 0.05.
- **Compute.** Each arm on 300 prompts is about 40 minutes on the Windows GPU; the author runs the long jobs.
- **Unverified items.** The CLI flags of the `tools/` scripts, whether EasyEdit supports GPT-2 small, and all literature citations were not checked while writing this plan.

---

## 12. Files to read first

`docs/handoff_2026-10-05_mac.md`, `docs/Research_Journal/28.md` to `30.md`, `docs/research_roadmap.md`, `docs/session_2026-10-06_1623.md`, `src/gradient_editing.py`, `tools/compare_sweep_vs_gradient.py`, `tools/test_generalisation.py`, `tools/test_generation_quality.py`, `research/paper/draft_v1/EVIDENCE_AUDIT.md`.
