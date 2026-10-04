# Handoff (2026-10-05): START HERE, written for moving to a MacBook

Repository: `Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders` (folder name FeatureScalpel), https://github.com/adi5022/Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders
Working branch: **`gradient-descent-editing`** (everything is pushed; not merged into `main`). Older long-running log: `handoff.md` (written for the branch `pool-refill-implementation`, with dated updates appended at the bottom).

---------------------------------------------------------------------------------------------------------------------------------------

## 1. Five-command start on the Mac
```bash
git clone <repo> && cd <repo> && git checkout gradient-descent-editing
bash scripts/setup_mac.sh            # venv + mac_requirements.txt, prints which device will be used (mps or cpu)
python tools/check_mac_parity.py     # FIRST thing to run: does this Mac reproduce the saved Windows results? (add --full for 5 more self-checks)
bash scripts/run_app_mac.sh          # http://localhost:8501, open the tab "Prototype lab"
bash scripts/get_data_mac.sh         # only when you want to run the test tools (they need CounterFact); the app does not
```
If parity fails on `mps`, run with `FEATURESCALPEL_DEVICE=cpu` (everything works on the CPU, slower) and note which line failed.
The first run downloads GPT-2 small and the SAE from Hugging Face (a few hundred MB); a token in `.env` (`HF_TOKEN=...`, not in git) only avoids rate limits. Neuronpedia explanations and the optional Groq explainer need internet.

## 2. What the project is (one paragraph)
Transient activation editing in GPT-2 small (layer 8, SAE `gpt2-small-res-jb`, hook `blocks.8.hook_resid_pre`): during one forward pass, edit SAE features so that a TRUE answer that GPT-2 ranks below first becomes its top prediction, without changing the model's weights and with few side effects. The edit is applied as a delta in the residual stream (`delta = sum (m_k - 1) * act_k * W_dec[k]`), so the SAE reconstruction error is not injected. Success is measured as target rank 1, side effects as KL(clean || edited) over the other tokens at the last position.

## 3. Current working state (what exists and works)
**App** (`experiment_app.py`, Streamlit, 7 tabs). Tabs 1-6 are older: Hybrid mute and boost (the fixed sweep and, as a choice, per-feature gradient descent), Monosemanticity Analysis, Session history and benchmarks, Sequential vs Batched Proof, Batch: Last vs All Tokens, Repair & SAE limit. **Tab 7 "🧪 Prototype lab" is the current main tab**: gradient-descent multiplier edit, optional additive edit (switch on silent features; default OFF with a warning), rank-by-step chart, **Baseline vs edited text** (greedy generation that continues past the target; option to keep the edit on while generating), **Follow-up prompts** (a leak test: run other prompts with the same edit on and off; option to append them to the generated text), switched-on feature list and multiplier table with Neuronpedia links (shown LAST because the Neuronpedia lookups are slow).
**Core code**: `src/gradient_editing.py` (`run_gradient_descent_edit`, `generate_greedy`, `next_token_shift`, additive machinery), `src/hooks.py` (`make_scale_map_hook`, `make_scale_and_add_hook`), `src/editing.py` (clean context, target ids, `tokens_without_bos`, `ensure_bos_default`), `src/hybrid_runner.py` (the sweep), `src/device_utils.py` (NEW: device sync/cache/describe for cuda, mps, cpu), `src/sae_utils.py` (device pick: cuda > mps > cpu, override `FEATURESCALPEL_DEVICE`).
**Verified where**: all numbers in the journal were produced on the Windows PC (GTX 1660 Ti, torch 2.6.0+cu124, Python 3.13.7, transformer-lens 3.5.1, sae-lens 6.45.3, transformers 5.13.0). After the Mac port the CUDA path was re-run: `tools/check_mac_parity.py` reproduced the saved ranks exactly (16 of 16 baseline ranks, KL equal to 4 decimals on 4 gradient-descent runs). The CPU path was exercised on the same PC with `--device cpu`: parity OK (16 of 16 baseline ranks identical, KL equal to 4 decimals on 4 gradient-descent runs, about 7 s per run), all 5 checks of `tools/check_additive.py` pass, and a Streamlit AppTest with `FEATURESCALPEL_DEVICE=cpu` ran the Prototype lab (result, text comparison, follow-up box) with no exceptions. **The MPS (Apple GPU) path has NOT been run by anyone yet.** Treat the first Mac session as the verification.

## 4. Results so far, in plain terms (details: Research Journal Entries 28 to 30 in `docs/Research_Journal/`)
- **Gradient descent beats the fixed sweep** (Entry 29, 300 held-out prompts, same 0 to 3x multiplier range): rank 1 on 271 (gradient descent) vs 161 (best sweep setting) vs 94 (sweep 0.6/0.5); gradient descent solved 110 that the wide sweep missed and lost 0; lower side-effect KL on 160 of 161 shared successes; about 4 to 6.5x faster. The KL term is part of its loss (partly by construction). Median target probability at rank 1 is only 7.7%.
- **The additive edit is too strong to count as a fix** (Entry 30): it pushed 20 of 20 random unrelated words to rank 1 (multipliers: 1 of 20), at edits of 126 to 192% of the residual norm; its transfer to reworded prompts is the same for random and true targets; neutral-prompt KL 0.229 and 24% top-1 flips (cap 1.0) vs 0.010 and 6% for multipliers. Rank 1 alone is therefore NOT evidence of anything for it.
- **Withdrawn claim**: the old rule "target unreachable by steering = not in the model" is withdrawn (Entry 30 section 12, H23). Reachability by steering is not evidence about knowledge.
- **Critique test, true targets, 40 prompts** (Entry 30 section 15, data in `docs/Research_Journal/packs/generation_quality/true40/`): sequential collapse is mild (fluency unchanged, variety 0.93 to 0.85-0.89, loops 0% to 2-12%, the target word repeats about 0.65 times per continuation); the edit overrides "Do not say the word X" (X first in 45% of unedited cases because naming it primes copying, 68% with the multiplier edit, 88-90% with the additive edit). The critique's "88% of the residual stream" figure is imprecise: edit size is measured at the last position only.
- **Follow-up leak (hand check, not a measurement)**: with the additive edit kept on, "The capital of Germany is" continues "Hong Kong".
- **BOS bug found and fixed** (Entry 30 section 13): `model.to_tokens(x, prepend_bos=False)` flips shared `model.cfg.default_prepend_bos` and is not thread-safe; in a multi-threaded Streamlit server the flag could stay False and silently drop the start token (16 of 57 recorded app runs, all from one server process; none used in a table). Fixed with `tokens_without_bos` and a per-run self-heal.
- Greedy GPT-2 small loops by itself over longer texts; repetition has to be judged against the baseline's own repetition.

## 5. Current thinking
- The open question is **specificity**: does an edit change only what was intended? Reaching rank 1 is easy for a strong enough edit. The multiplier edit looks usable and fairly specific; the additive edit behaves like a narrow injection of the answer word's output direction (not of a fact) and leaks into unrelated prompts.
- Novelty (honest, snippet-level literature check, not exhaustive): SAE steering, ROME/MEMIT, IKE and prompting already exist; AxBench found prompting beats SAE steering. What is new here is mostly engineering and measurement: per-feature multipliers tuned by gradient descent with a side-effect term, a transient (no weight change) edit, and an equal-budget comparison with the sweep. A reader will ask for the plain-prompt baseline and for ROME/IKE/DiffMean on the same measures (EXP-015, not run).
- A measurement trap already hit once: comparing minimal edit sizes of true vs random targets was circular because their start ranks differ (17 vs 17,504); controls must be rank-matched.
- Everything is exploratory and proxy-labelled. Paper draft updates (`research/paper/draft_v1/main.tex`) are deliberately deferred.

## 6. Planned work, in order (details: `docs/research_roadmap.md`)
1. **First on the Mac**: run `tools/check_mac_parity.py --full`; run the Prototype lab on "The capital of France is" / " Hong" (Windows reference: rank 3419 to 1 with additive on; the multiplier-only edit also reaches rank 1) and compare; note timings but never compare them across machines.
2. **A. Specificity benchmark** (the yardstick): reach, transfer to rewordings, leak to unrelated prompts, text quality, context sensitivity, with true vs random vs **rank-matched random** targets and arms sweep / gd / gd_add. Built already: `tools/test_generalisation.py`, `tools/test_generation_quality.py`. Missing: the random-target run of the generation-quality test (`--targets random`, about 20-40 min), the rank-matched control, a larger transfer test (about 150 prompts, with confidence intervals), the sweep arm (the Entry 29 pack does not store the sweep's final edits, needs a rerun), one combined table with paired statistics, then Entry 31.
3. **C. Make the multiplier edit more specific**: add paraphrase / neighbour-prompt / unrelated-prompt terms to the gradient-descent loss (the KL machinery exists), then re-run A.
4. **D. In-context fact vector**: run the model with the fact stated ("Fact: ... ") and without, subtract the layer-8 residuals at the last token, add the difference scaled by alpha (raw residual and SAE-feature variants). The told run doubles as the prompt-engineering baseline. Risk: it may behave like the additive edit.
5. **B. Established methods** (IKE, ROME, DiffMean) on the same prompts and measures.
6. **Write-up**: one ranked table, honest framing, withdrawn claims stated, then the paper draft.
**Queued smaller ideas**: stop-early option for the gradient descent (after N steps without improvement; off by default; the run currently uses all 100 steps and keeps the best iterate, which cannot be worse than an early one); a control in the follow-up box (unedited model on the prompt that contains the edited text, to separate the edit from ordinary copying); repetition-penalty / no-repeat-ngram / sampling toggle for the generation view; `kl_always` evaluated at scale; multi-token target scoring; gradient descent as teacher for FeatureNet (`docs/future_scope.md`); a steering-controller LLM front end (last).

## 7. Open decisions for the author
- Keep the full BOS fix or cut it back to only the per-run self-heal (the fix edited old files `src/editing.py` and `src/hybrid_runner.py` without asking first; the author's rule is that new code must not change old code).
- Whether the additive checkbox should stay in the shipped prototype at all (default OFF, with a warning, now).
- Whether to merge `gradient-descent-editing` into `main`.

## 8. Files NOT in git (rebuild or copy), and what needs them
| File / folder | How | Needed by |
|---|---|---|
| `datasets/counterfact.json` (45 MB) | `scripts/get_data_mac.sh` (downloads, checks SHA-256 `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`) | test tools |
| `data/counterfact_hard_set.json` (15.5 MB) | `scripts/get_data_mac.sh` runs `tools/build_counterfact_set.py`; expected SHA-256 `657b475e3cc37c7fb7c24a2c5c9d376791c70370cc894dca5ac3db31a3526ec8` (a different hash on another machine can be harmless float noise; check `data/counterfact_hard_set_summary.json`) | test tools, `tools/test_*` |
| `outputs/` (all run results, strength cache/tables, session history) | local only; the important results are copied into `docs/Research_Journal/packs/` | only older FeatureNet/strength work needs `outputs/strength_cache` and `strength_tables` (see `handoff.md`) |
| `.env` (HF_TOKEN, optional Groq key) | create by hand | rate limits, explainer |
In git: `data/counterfact_split.json`, `data/counterfact_hard_set_summary.json`, all data packs under `docs/Research_Journal/packs/` (the parity check reads `packs/sweep_vs_gradient/runs.jsonl`).

## 9. Port notes: what changed for macOS and what is untested
- New: `src/device_utils.py` (`sync_device`, `empty_device_cache`, `device_memory_gb`, `describe_device`), `mac_requirements.txt`, `scripts/setup_mac.sh`, `scripts/run_app_mac.sh`, `scripts/get_data_mac.sh`, `.gitattributes` (`*.sh` stay LF), `tools/check_mac_parity.py`.
- Changed (behaviour on CUDA/Windows unchanged): the `if device == "cuda": torch.cuda.synchronize()` blocks now call `sync_device(device)` (also correct on MPS, which is asynchronous: without it timings would be under-measured) in `experiment_app.py`, `src/hybrid_runner.py`, `tools/compare_sweep_vs_gradient.py`, `tools/measure_plan_costs.py`, `benchmark_sequential_vs_batched.py`; `src/sae_utils.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` (unsupported MPS ops run on the CPU instead of erroring) and honours `FEATURESCALPEL_DEVICE`; the Batch tab shows an Apple-GPU note and caps workers at 2 on MPS, and on POSIX starts workers in their own session, stops the whole group and reaps finished workers (a finished child otherwise looks alive); `run_stage1.py` uses a relative output path.
- Not ported on purpose: `test_gpu_benchmark.py`, `run_speed_analysis.py`, `layer_benchmark.py` (they measure CUDA hardware). The NVIDIA memory panel in the Batch tab is CUDA only.
- Known MPS risks to watch: (a) MPS has no float64 (none is used in app/src); (b) small numeric differences vs CUDA can move a rank by a place when two words are nearly tied, and can change which gradient-descent step is "best", so exact equality with the Windows numbers is not expected on the gradient-descent runs (the parity script allows for this; KL within 0.05, rank 1 still reached); (c) `torch.mps` memory is shared with the system; the Batch tab's parallel workers each load a copy of the model; (d) the `.venv\Scripts\...` commands in older docs are Windows spellings, on a Mac use `.venv/bin/python` and `.venv/bin/streamlit`; (e) the older files use CRLF line endings and the app source contains emoji: when scripting edits read/write with `newline=''` and `encoding='utf-8'`.

## 10. Tools map (all run as `python tools/<name>.py`)
| Tool | What it does | Rough run time on the GTX 1660 Ti (estimates, except `test_generation_quality` which was timed) |
|---|---|---|
| `check_mac_parity.py` | reproduces saved ranks/KL on this device | 1 to 3 min (`--full` adds `check_additive.py`) |
| `check_additive.py` | 5 self-checks of the additive edit and generation | a few min |
| `compare_sweep_vs_gradient.py` | equal-budget comparison, arms `sweep_ref, sweep_wide, gd, gd_klall, gd_add, gd_add_klall` | 300 prompts: about 40 min per arm |
| `control_random_targets.py` | random-word control (20 prompts) | ~20 min |
| `test_generalisation.py` | edit tuned on one prompt, applied to rewordings / neighbours / neutral prompts | 40 prompts: about 1 h |
| `test_generation_quality.py` | sequential collapse + "do not say X" context test | 40 prompts: about 22 min |
| `make_journal_figures.py` | regenerates journal figures from the committed packs only | seconds |
| `build_counterfact_set.py`, `split_counterfact_set.py` | build the hard set and the split | ~1 min |
| older: `build_strength_*`, `train_strength_models.py`, `train_feature_net.py`, `measure_headroom.py`, ... | the learned-strength study (Entries 23 to 27) | see `handoff.md` |

## 11. Journal map
`docs/Research_Journal/28.md` gradient descent: method and hand trials; `29.md` equal-budget comparison (300 prompts) and its figures (`images/e29_fig*`); `30.md` additive edit, controls, reworded-prompt test, correction of the "knowledge absent" rule, BOS bug, Prototype-lab features (sections 13 to 15); `1.md` carries the update note withdrawing the old rule. Hypotheses H20 to H23 in `research/hypothesis_log.md`; experiments EXP-020 to EXP-022 in `research/experiment_index.md`; `research/timeline.md` is the dated log.

## 12. How the author likes to work (please keep to this)
- Explain in plain words with a concrete example; do not water it down.
- Do exactly what is asked; propose extras and wait. New code must not change old code (ask before touching old files).
- Document everything and push to the branch; figures and tables only from real data; label things honestly (exploratory, proxy); say plainly when something failed or was not verified.
- The author runs very long jobs on their own machines; the paper draft is updated later, not now.
- Timing comparisons between methods are valid on one machine only.
