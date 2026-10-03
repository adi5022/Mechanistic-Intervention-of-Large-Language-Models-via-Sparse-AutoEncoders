# Handoff (2026-10-02): START HERE

Branch: `pool-refill-implementation` (everything is pushed). Repo: `Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders` (folder name FeatureScalpel).

## Read in this order
1. `docs/session_report_2026-09-30_to_10-02.md`: plain-language account of everything done in the last session, with numbers, mistakes and where files are.
2. `docs/Research_Journal/23.md`: the full plan for the learned-strength study and a running log of every result since (sections 13.1 to 13.8).
3. `docs/literature_review_sae_editing.md`: what the literature says about SAE editing and what is (not) novel here.
4. `docs/Research_Journal/24.md` (data pipeline and headroom results, a clean results entry), `25.md` (literature check and direction assessment), `22.md` (repair and SAE-limit diagnostics; note the scope warning at its top). `21.md` is a placeholder: the safety-filter study's interpretation is still unwritten.
5. Older background: `research/paper/draft_v1/HANDOFF.md` (paper draft, Batch tab, working-style lessons), `docs/handoff_2026-09-24.md`, `docs/handoff_2026-10-01_session_log.md` (the previous version of this file, a long running log).

## The project in one paragraph
Transient activation steering in GPT-2 small: edit SAE features (`gpt2-small-res-jb`, layer 8, `blocks.8.hook_resid_pre`) during one forward pass so that a true answer GPT-2 ranks below first becomes its top prediction. The edit applies only the delta of the SAE reconstruction (`x' = x + decode(edited) - decode(original)`), so reconstruction error is never injected. Features are chosen by the causal effect of removing them, passed through a strict safety filter, and applied step by step (cumulative sweep, pool refill). Main app: `experiment_app.py`; headless engine: `src/hybrid_runner.py`, `src/batch_runner.py`, `run_batch.py`.

## What is being done now (the learned-strength study)
A small network that picks the mute and boost strengths per prompt instead of the fixed 0.6 / 0.5. Planned as a variant and documented whether or not it works (Entry 23, H15, EXP-013). Compared against (1) fixed 0.6 / 0.5, (2) a learned fixed pair (two shared numbers), (3) an oracle (strengths optimised per prompt). Version 2 (one multiplier per candidate feature, learned implicit filter) comes after version 1.

## State of the pipeline
```
DONE   hard set: 16,360 CounterFact prompts GPT-2 gets wrong (true answer rank 2 to 1000)
DONE   split: train (first 1,000 of a balanced order) / val 150 / test-seen 150 / test-unseen 150
DONE   per-prompt cache: 1,450 files, verified
DONE   headroom measurement: 1,450 prompts (SAE scaling reaches rank 1 on up to 80%, 41% with a side-effect penalty)
DONE   strength tables: 1,450 files, verified
TODO   training code (PromptNet, learned fixed pair, oracle)
TODO   real sweeps on the 300 test prompts, 4 arms
TODO   other methods (IKE, ROME, DiffMean) with existing code; comparison tab
TODO   reference run of the real system on a large prompt set
```

## Files that are NOT in git (rebuild or copy)
| File / folder | How |
|---|---|
| `datasets/counterfact.json` (45 MB) | download from https://rome.baulab.info/data/dsets/counterfact.json; SHA-256 `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f` |
| `data/counterfact_hard_set.json` (15.5 MB) | `.venv\Scripts\python.exe tools/build_counterfact_set.py` (about 1 minute); SHA-256 `657b475e3cc37c7fb7c24a2c5c9d376791c70370cc894dca5ac3db31a3526ec8` |
| `outputs/strength_cache/` (1,450 files, 74 MB) | `tools/build_strength_cache.py` (7 to 17 min) or copy a zip from another machine; check with `tools/verify_strength_cache.py --expected 1450 --split-file data/counterfact_split.json` |
| `outputs/strength_tables/` (1,450 files, about 29 MB) | `tools/build_strength_tables.py` (49 min on the RTX 4050 laptop, about 1.9 h on the GTX 1660 Ti); check with `... --verify --expected 1450` |
In git: `data/counterfact_split.json` (SHA-256 `68638fd7193c599accb3803bbc9c21bdcf870856a92caf6902c2bc899f5f10b0`), `data/counterfact_hard_set_summary.json`, the headroom and diagnostics result packs under `docs/Research_Journal/packs/`.

## Next step in detail: the training code
Spec is in Entry 23, sections 4 to 7. In short:
- **Inputs per prompt:** the 12 summary numbers stored in each cache file (`summary` dict; names in `src/strength_cache.py: SUMMARY_NAMES`), standardised on the training prompts. The relation id is NOT an input.
- **PromptNet:** 12 -> 32 -> 32 -> 2, weight decay, dropout 0.1. Output mute strength mu = sigmoid, boost beta = 2.0 x sigmoid. Final bias initialised so it starts at exactly (0.6, 0.5).
- **Controls:** the same code with the network replaced by two free numbers (learned fixed pair), and two free numbers per prompt (oracle).
- **Which features are edited at strengths (mu, beta):** look up the strict-filter survivors in the strength tables (nearest grid point per side; strict rule: target probability drops by at most 1e-6 and rank does not worsen), rank mute candidates by the cache's `db` (blocker effect, most negative first) and boost candidates by `dt` (target effect), and resolve candidates passing on both sides as the sweep does (keep on the side with the larger target gain). The sweep's exact logic is `build_pools` in `src/hybrid_runner.py`.
- **Edit and forward pass:** `delta = sum_k (m_k - 1) * cand_acts[:, k] * W_dec[cand_ids[k]]`, added to the saved `resid8`, then run blocks 8 to 11 only: `model(resid + delta, start_at_layer=8, tokens=toks)`. This was verified equal to your real hook (max logit difference 8.8e-06, same target rank on 12 of 12 prompts). Batches of 32 prompts cost the same as one (about 50 ms). Right-pad prompts; read logits at each prompt's last real position. `tools/measure_headroom.py` already contains working batching, padding and loss code to reuse.
- **Loss:** hinge `max(0, margin - (target_logit - best_other_logit))` with margin 0.3, plus a side-effect term (KL of the distribution excluding target and blocker), plus a small strength penalty.
- **Procedure:** AdamW, lr about 1e-3, batches of 32, early stopping on validation, 5 seeds, learning curve at 100, 250, 500, 1,000 training prompts. Validation uses a one-shot proxy (all survivors applied at once); final numbers must come from real sweeps.
- **Then the real sweeps:** write a batch spec where each test prompt carries its own `settings: {mute_strength, boost_strength}` (the spec format already supports per-prompt settings, so `hybrid_runner.py` needs no change for version 1). Arms: fixed 0.6/0.5, learned fixed pair, PromptNet, oracle. Settings otherwise as the reference run: Top-N 200, all positions, strict filter, cumulative sweep, pool refill, 250-step cap. Use `run_batch.py --shard 0,1,2,3,4/7` on the laptop and `--shard 5,6/7` on the other machine.

## Open decisions (the user decides)
- Test-set size (proposed 150 + 150) and the pre-stated reading of the outcome (Entry 23, section 8.4).
- Whether version 2 runs in parallel with version 1.
- beta_max (2.0) and the rank-term margin (0.3).

## Machines and environment
- This PC: GTX 1660 Ti, 6 GB, `.venv`, Python 3.13.7, torch 2.6.0+cu124, transformer-lens 3.5.1, sae-lens 6.45.3, transformers 5.13.0. `requirements.txt` pins `torch==2.13.0`, which does not match the working environment.
- Second machine: laptop with an RTX 4050 (6 GB, 24 GB RAM), torch 2.5.1+cu121. It is about 2.4 times faster (0.29 s vs 0.70 s per cache prompt; 2.0 s vs 4.6 s per strength-table prompt). Instructions for an AI agent there: `batched_training.md`.
- Timing comparisons between methods must be measured on ONE machine.

## How to work here (from the user; still applies)
- The user takes the decisions; Claude documents and implements. Say plainly what a number is and where it came from. Do not argue against the SAE approach to prove a point; compare fairly and report null results.
- Plain language first, detail second. If the user says "I don't get it", restate with one concrete example, not more jargon.
- Ask before committing or pushing unless the user has just asked for it. Never commit `outputs/`.
- Do not launch long jobs; give the command and let the user run it (small smoke tests are fine if disclosed). The user has two machines and runs the long jobs.
- Paper style: no em dashes, no hype, no novelty claims; every claim traceable (`research/paper/draft_v1/EVIDENCE_AUDIT.md`).

## Technical gotchas
- Use `.venv\Scripts\python.exe` (the default `python` has no torch). Files are CRLF; read with `newline=''` and `encoding='utf8'` (the app source has emoji).
- Batched API: `batched_ablation_probs(...)`'s `scale` is a MULTIPLIER (0.0 = feature removed, 1.0 = untouched); `make_ablation_hook`'s `strength` is the opposite (1.0 = removed).
- TransformerLens: hooks must accept the keyword `hook` (`lambda r, hook: ...`); `run_with_cache` does not take `fwd_hooks` (use `with model.hooks(fwd_hooks=[...]):`); `model(resid, start_at_layer=8, tokens=toks)` runs from a saved layer-8 state.
- The command harness can halve double backslashes (a `\\t` once became a tab). Write documents with the file-writing tool; in scripts use `chr(9)` / `chr(92)`.
- The Streamlit app needs 60 to 90 s to load the model before tabs render. New tab: "Repair & SAE limit" (diagnostic only).
- Windows file locks: do not read a batch result JSON while a run writes it; poll the `.progress.json` instead.

## Known weak spots in the code (unchanged)
Blocker fixed per round (should re-select when top-1 changes); Groq explanation uses an overwritten `probs`; multi-token targets scored on the last piece (CounterFact targets are all single-token, so not an issue there); `iterative_ablate` and `check_specificity` hard-code layer 8; no automated tests for editing or safety code; `src/repair_diagnostics.py`, `src/strength_cache.py` and the new tools have no unit tests (they have built-in sanity checks instead).

## Not committed on purpose / not done
- Entry 21 (write-up of the safety-filter study) is still unwritten; drafts are in each pack's `journal_entry_draft.md`.
- The paper draft (`research/paper/draft_v1/main.tex`) does not yet contain the Entry 22 or Entry 23 work or the literature note. It has not been compiled (no LaTeX on the machines).
- No SAE-versus-other-methods comparison has been run. The reference run of the real system on a large prompt set has not been run.

## UPDATE 2026-10-02 (late): training code written
- `src/strength_models.py` and `tools/train_strength_models.py` exist and pass their smoke test (Entry 23, section 13.9). **Next: run it** (`.venv\Scripts\python.exe tools/train_strength_models.py`, about an hour; add `--quick` for a 4-minute smoke test), then read `outputs/strength_models/<time>/results.json`.
- The proxy is prefix-cumulative (it mimics the sweep's step-by-step additions); the side-effect penalty applies only where the target already leads; the control has its own larger learning rate. These three were failures found during development (section 13.9). Early smoke reading: most of the gain over fixed 0.6/0.5 comes from a better GLOBAL pair (about 0.95 / 1.6), not from per-prompt adaptation; one seed, 10 epochs, NOT a result.
- Still to do after the full run: per-arm specs for the real sweeps (each arm needs its own spec) plus a paired analysis across arms; version 2; the reference run of the real system.

## UPDATE 2026-10-03: training results are in
- Full training ran on the RTX 4050 laptop; results, analysis and the exported per-prompt strengths are in `docs/Research_Journal/packs/strength_models/`; write-up: **`docs/Research_Journal/26.md`** (read it). Proxy result on the 300 test prompts: fixed 0.6/0.5 25.0%, learned fixed pair (mute 0.98, boost 1.81) 38.7%, PromptNet 39.7% (not distinguishable from the pair, p = 0.25), per-prompt tuned upper bound 41.0%; side effects about doubled. So: a better global default, per-prompt adaptation not shown. This is the PROXY, not the real sweep.
- **Next:** (1) real sweeps on the 300 test prompts, 4 arms (reference; pair mute 0.98 / boost 1.81; PromptNet per-prompt from `strengths_export.json`; oracle per-prompt from the same file), each arm its own spec (per-prompt settings apply to all arms of one spec), then a paired analysis across the result files: a spec generator and an analysis script still have to be written; (2) map success vs side effects by re-training with larger `--lam-kl` (30, 100); (3) extend the strength tables past boost 2.0 if the real sweep agrees; (4) version 2; (5) other methods.
- `tools/analyse_strength_models.py --dir <folder with results.json and strengths_export.json>` reproduces the per-prompt analysis (needs the local cache and tables).

## UPDATE 2026-10-03 (later): version 2 (per-feature) built
- `tools/train_feature_net.py` + `src/feature_models.py` (Entry 27). No new data needed. Run `.venv\Scripts\python.exe tools/train_feature_net.py` (about 30 min here, about half on the laptop; `--quick` = smoke test). It compares FeatureNet, FeatureNet + strict guard, the version-1 rules (re-evaluated one-shot from `docs/Research_Journal/packs/strength_models/strengths_export.json`) and a free per-feature upper bound, with paired tests.
- Development runs only so far: FeatureNet reaches about 31 to 32% of held-out prompts vs 37% for the version-1 pair and 25% for fixed 0.6 / 0.5; the free per-feature upper bound is 73%. The defaults (lam_kl 3, size weight 0.005, patience 20) were chosen on validation, with test numbers visible. Ideas not yet tried: start from the version-1 rule and learn corrections; a bigger network; the per-feature override in `hybrid_runner.py` for real-sweep tests.

## UPDATE 2026-10-03 (latest): gradient-descent editing in the Hybrid tab (branch `gradient-descent-editing`)
- New branch `gradient-descent-editing` (from `4618a30`; code commit `af485bb`). Hybrid tab has an "Editing method" choice: fixed mute / boost sweep (default, unchanged) or **gradient descent: one multiplier (0 to 3) per candidate feature, tuned on the prompt, final result checked through the real hook**. Code: `src/gradient_editing.py`, `experiment_app.py`. Full write-up: `docs/Research_Journal/28.md` (method, settings, checks, hand trials, limitations, decisions). Data: `docs/Research_Journal/packs/gradient_descent/session_20261003_015115.json` (11 gradient-descent and 3 sweep runs).
- Hand trials only (about 12 prompts, no statistics): gradient descent ended nearer rank 1 than the sweep on all 6 shared prompts and reached rank 1 on 3 (sky to black, Frankie Lee Sims to Dallas, Kaka to soccer), with side effects (KL) 0.3 to 1.9 on the harder ones. **Confounded: it may scale features 0 to 3, the sweep only about 0.4 to 1.8 at the strengths used; the equal-budget sweep (mute 1.0 / boost 2.0) has NOT been run.** Side effects measured for gradient descent only. The loss does not penalise side effects on prompts that fail (KL 2.7 on "The sky is" to "Adi").
- Edit applies at every prompt position, but the tuning scores only the next-token prediction at the last position; nothing about the continuation is optimised.
- Decisions: prototype to be shown to others should keep the edit on while GPT-2 generates (side-by-side clean vs edited, per-word trace, fact probes with paraphrases and neighbours, a 300-prompt scoreboard with equal budgets); things to fix first: failing-prompt loss, paraphrase / neighbour terms in the objective, equal budget. Instruction tuning of GPT-2 scrapped. Gradient descent as the teacher for the per-feature network is recorded in `docs/future_scope.md` (not started).
- To run: `.venv\Scripts\streamlit.exe run experiment_app.py`, Hybrid tab, Editing method = Gradient descent, Top N 200.

## UPDATE 2026-10-03 (night): equal-budget comparison done (Entry 29)
- `tools/compare_sweep_vs_gradient.py` (commit `11eb05e`) ran the real sweep at 0.6 / 0.5 and at 1.0 / 2.0 (same 0 to 3x range as gradient descent) and gradient descent on all 300 held-out prompts, with one yardstick (rank, same-prompt KL, features, edit size, unrelated-prompt KL, time). Data: `docs/Research_Journal/packs/sweep_vs_gradient/`.
- Result: rank 1 on 94 / 161 / 271 of 300 (sweep 0.6/0.5 / sweep 1.0/2.0 / gradient descent). Gradient descent solved 110 prompts the wide sweep did not, the wide sweep none that it did not (p = 2e-33); its same-prompt KL was lower on 160 of 161 shared successes; 5 to 6.5 times faster; weakest in the 101-1000 band (54 of 79). The earlier worry that its side effects are larger is not supported. **Caveats:** its loss contains a KL term (the sweep has none); median target probability at rank 1 is 7.7%; paraphrases, neighbour prompts, generation and other sweep settings (tolerance / graded safety modes, learned pair) are NOT tested.
- Next (not started): generation view with the edit kept on; paraphrase and neighbour checks; a sampling-based success measure.
