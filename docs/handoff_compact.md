# handoff_compact.md: for Claude, to resume after compaction (rewritten 2026-10-11, state: D13 edits generated, training run not yet done)

Read this first. Then, only if you need detail: `CLAUDE.md` (the working and logging rules, loaded automatically), `docs/cross_model_transfer/PAPER_NOTES.md` (headline table, claims allowed and not allowed, caveats), `docs/cross_model_transfer/PLAN.md` (runbook, every gate, results log, newest sections at the end of the D-series and Phase E), `docs/Research_Journal/33.md` (D5 to D10, D6b), `34.md` (Import count), `35.md` (Import result), `docs/handoff_2026-10-10_cross_model.md` (project handoff for people). Everything below is the short version plus how to behave.

## 0. The user and how to talk to them (most important)
- About 3 months of interpretability experience; understands the concepts. What loses them: number piles, jargon, tables with no explanation, and **many directions at once** ("we are going in seventeen different directions"). They have lost patience more than once.
- What works: lead with what happened and what it means in 3 to 5 plain lines, one concrete example ("The mother tongue of Danielle Darrieux is" with the wrong target " English"), two or three numbers, define every term the first time (dose = the multiplier on the translated change before it is added; seed = the number fixing the shuffle of training batches; dev = the records used to choose settings; test = records only read; rank 1 / top-1), tables only after the explanation and small, say what was NOT tested, end with ONE recommended next step and one question. Say "dose", never a metaphor for it. Offer a short list of options only when asked, then recommend one.
- They run long jobs in their own terminal to watch progress: give the exact PowerShell command (`$env:HF_HUB_OFFLINE=1; .venv\Scripts\python.exe tools/transfer/NN_name.py`), what they will see, how long (a rough guess, say so), and what to paste back. They paste the terminal output into the chat; the pasted text is sometimes garbled with old scrollback: check the result files.
- They want the paper "rock solid": after every experiment log everything (CLAUDE.md lists the steps). They said a pass bar of 20% is too low; bars are 50%.
- Protected: branch `rome-factual-editing` belongs to a teammate (never delete or rebase). Never push `main`. Push to `cross-model-transfer` only; the user has authorised pre-run commits and result commits there repeatedly; stage explicit paths (never `git add -A`; `outputs/` and `.env` stay out).
- Memory files exist in `C:\Users\imalv\.claude\projects\D--Projects-transient-steering\memory\` (plain-explanation feedback, project rules).

## 1. The study in one paragraph
Send the change an edit makes inside GPT-2 small (layer 8, SAE `gpt2-small-res-jb`, 200 candidate features, one multiplier each: "the multiplier edit", never the additive edit) into GPT-2 medium (layer 16, no SAE) through a map fitted on ordinary text, at inference time, no weight change. **Export** = edit in small, add the translated change to medium (the user's "athlete showing a few skill shots to a bigger athlete"). **Import** = medium's state into small. The map is a linear (ridge) map of the residual stream `blocks.L.hook_resid_pre`; the SAE is used only where the edit is made.

## 2. All results so far (88 primary test records unless stated; "top-1" = the counterfactual target becomes medium's top answer)
| Step | Result | Verdict |
|---|---|---|
| A1/A2 translator | per-dim R-squared 0.58 (small to medium), 0.66 (medium to small); stitching recovers 91 to 98% | Gate 1 passes only medium to small |
| B1 country swap | 55% (small to medium), 33% (medium to small), controls 0 to 1% | Gate 2 PASS |
| C (small alone) | edit leans to read-out steering, weakly; "mainly the subject's representation" claim withdrawn | rule: read-out steering |
| D1 to D4 export | top-1 5% and 8% (medium layers 12, 16); rank 142 to 23 and 14; rewordings +15 points; neighbours -4 | Gate 3 FAIL (15%) |
| neural translator | fits better (R-squared +0.1), exports no better (6 to 8%) | H29 rejected |
| D5 several layers | best single layer 8%, best set 7% | Gate 4 FAIL |
| D6 ceiling | edit tuned through the translator against medium's output: **95%** (original 8%); neighbours -8 points | Gate 5 FAIL on neighbours only; a ceiling, not a transfer (looks at medium per sentence) |
| D7 map trained on 289 edits | 18% (ridge map 8%), median rank 7, controls 0 and 6% | Gate 6 FAIL |
| D8 more data | 25%, 50%, 100% of the edits: 13%, 13%, 20% | RISING (flat then jump, noisy) |
| D9, D9b longer training | 40 epochs: pre-set rule said 2% vs 20%, but that was the dose rule on 28 dev records (picked dose 1.0); with dose 2.0 both lengths give 20% | no improvement; dose rule fragile (H35) |
| D10 rank-1 term | lambda 0 to 1.0: test top-1 16 to 21% for all; training shortfall 2.4 to 0.15 logits; training top-1 95% vs test 16% | Gate 7 FAIL; **the map memorises its training targets** |
| D6b | D6's solution vs original edit: last-position cosine 0.51 (chance 0.02), D6 smaller (0.71), partly the same direction | descriptive |
| E1 count | medium right and small wrong: 780 facts (outside ROME dev and pilot); 71% are small's rank 2 to 5 | rule met |
| E2 to E4 Import by replacement | replacing small's layer-8 state with medium's translated layer-16 state rescues 262 of 780 (33.6%; controls about 0); loses 22% of facts both knew and 70% of those only small knew; accuracy 7.2% vs small 8.3%; SAE-filtered 6.3% | Gate 8 FAIL; real information crosses, net negative |
**Plateau:** every map-only fix for Export stops at about 20%. The dose rule from D10 on: pick the dose by dev mean log-rank gain, not dev top-1.

## 3. Where we are now: step D13 (more, and more varied, training edits)
- Idea (the user chose the chain C, then A, then B): the map memorised 289 training edits that cover only **116 distinct target words**; so tune extra edits in small toward RANDOM rank-matched words (never a true or counterfactual word of any dev or test record) and retrain the map on the bigger set. Plan, Gate 9 (= Gate 7 unchanged: top-1 at least 50%, rewordings lift at least 80% of the edit's own, neighbours no worse than the edit's own 9.5 points, real minus random-word top-1 at least 10 points) and fixed readings are in `PLAN.md` step D13 (committed `129c3cc`, tool `19b48e6`).
- **Done:** `tools/transfer/21_varied_edits.py` (3 workers, 3,542 s): 2,894 edits, **1,221 reached rank 1 in small (42%)**, 749 sentences, **759 distinct target words**, none banned. Saved `outputs/transfer/d13_edits.pt` (local), summary `packs/.../d13_edits_summary.json`, log `packs/.../logs/d13_varied_edits.log`. Finding: the synthetic edits are easier (median start rank 54) than the real training edits (116) and the test records (105); so `22_varied_training.py` got one EXPLORATORY set (real + the 457 synthetic edits starting at rank 100 or worse). This patch is in the working tree: commit it before the run if `git status` shows it.
- **NOT done: the training run.** `tools/transfer/22_varied_training.py` (built, rehearsed on the CPU with `--smoke --device cpu`): trains the linear map on real only, +25%, +50%, +100% synthetic, plus the exploratory "hard" set, 3 seeds each, 15 epochs; prints test top-1 next to training top-1 for every run; judges the full set and the baseline on rewordings, neighbours, unrelated prompts; applies Gate 9 and the fixed readings; about an hour. The command for the user: `$env:HF_HUB_OFFLINE=1; .venv\Scripts\python.exe tools/transfer/22_varied_training.py` (the GPU must be free of other jobs). It builds `d13_items.pt` (about 6,000 training texts) first (about 10 minutes, cached).
- **After the run:** explain it as a story; then record it (Entry 36 in `docs/Research_Journal/36.md`, result JSON and log into the pack, a figure from saved data in `tools/make_transfer_figures_33.py`, PLAN status and results log, H38 and EXP-039 verdicts, timeline, PAPER_NOTES, this handoff); commit and push (to `cross-model-transfer` only).
- **Fixed readings:** all four hold = varied edits solve it; (a) fails but test top-1 rises at least 5 points above the baseline's and the training-test gap shrinks = variety helps, go to step B; (a) fails and top-1 stays within 5 points = variety was not the limit, then amortised D6 or stop. If the random-word edit rises as much as the real target the map learned to push any word.

## 4. The chain after D13 (do not present all of these at once; recommend ONE)
- **B: a regularised neural map** (dropout, weight decay, small hidden layer, noise, early stopping) trained on the D13 set, with the training-vs-test table. Needs its own plan and gate written first. Modest odds alone, better with the bigger data.
- **F: the SAE-basis control** (for the paper, not accuracy): D6 with 200 random directions instead of 200 SAE features, to show whether the SAE matters for steering medium. We never tested that.
- **D12 amortised D6** (idea only): learn the change D6 finds for medium from small's edit, supervised by D6 solutions; same unseen-words problem; D6b says D6 differs partly in direction.
- **Import softer variants** (untested, cheap): a dose between none and full replacement scored on NET accuracy, use medium's state only where medium is confident (rescue doubles there), partial positions.
- **Write up:** the figures, `PAPER_NOTES.md`, logs and the failure analysis already exist.
- Related work: Chen et al., NeurIPS 2025 (arXiv 2506.06609), affine residual maps incl. GPT-2 small to medium, no fact edits; `docs/cross_model_transfer/RELATED_WORK.md` was written from a summary, to be checked against the full paper before any novelty claim.

## 5. Explanations the user keeps asking for (keep consistent)
- **Three maps:** the ridge map (fitted once on wikitext to copy states); the map trained on edits (same matrix, nudged using 289 edits so medium's answers move as small's do; saved in `d7_map.pt` and `d10_map_*`); D6 (not a map: the 200 dials are re-tuned per sentence while watching medium's answer).
- **Why "ceiling":** D6 is the most the channel can carry when allowed to look at medium; it is not a transfer, because medium's output is used for every sentence. It also gets 83% on a random word, so rank 1 alone is not evidence of a fact.
- **Why 289 not 800:** 1,450 candidate sentences, 829 dropped by the hold-out rule (target word or subject used by a dev or test record), 621 edited, the edit reached rank 1 on 47% = 289.
- **Import vs Export:** the translator copies states well (medium to small fits better) but the swap control favoured small to medium because small is a weak receiver. Import by replacement is a hybrid: medium's first 16 layers, a map, small's last 4.
- **What the SAE did:** it is the interface for making the edit in small; in Import the SAE-filtered version carried almost nothing; whether the SAE is needed for steering medium was never tested (step F).

## 6. Repo, environment, git
- Repo `D:\Projects\transient_steering`, branch `cross-model-transfer`; remote GitHub `adi5022/Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders`. Windows 11, RTX 4050 laptop GPU (6.4 GB, about 5.3 GB free), `.venv\Scripts\python.exe` (torch 2.6.0+cu124, transformer-lens 3.5.1, sae-lens 6.45.3). Set `HF_HUB_OFFLINE=1`.
- Commits on the branch (all pushed): `a8683b0` (Entry 32), `9109e4a` (D5 to D7 part), `e4d6649` (D7 to D9b, logs, figures, PAPER_NOTES, CLAUDE.md), `024f5f6` (D10 plan), `170e0d9` (D10 result), `1719459` (Import plan), `129c3cc` (Import result, D6b, D13 plan), `19b48e6` (D13 training tool); later commits: see `git log`. `main` = `bc35482`, untouched.
- NOT in git: `outputs/transfer/` (about 1 GB: maps, per-record numbers, caches, d13_edits.pt; remind the user to back it up). Raw logs of the runs are in `docs/Research_Journal/packs/cross_model_transfer/logs/`; the tools now write their own logs to `outputs/transfer/logs/`.
- Code (new files only): `src/transfer/{models,paired_acts,maps,stitch,minimal_pairs,swap,benchmark,export_eval,mlp_maps,multilayer,b_aware_edit,output_matching,om_batched,om_judge,import_eval,runlog}.py`, `src/gradient_editing_masked.py`, `tools/transfer/00` to `22`, `tools/make_transfer_figures.py` (e32) and `tools/make_transfer_figures_33.py` (e33 fig1 to fig9 incl. `e35_fig1_import`). Older code is imported, never edited, without asking.
- Docs to keep updated after every experiment: `PLAN.md`, `research/{hypothesis_log,experiment_index,timeline}.md` (H1 to H38, EXP-001 to EXP-039), `PAPER_NOTES.md`, the journal entry, the pack README, this file.

## 7. Gotchas
- The command harness halves double backslashes in SHELL command text: write scripts and docs with the Write/Edit tools, put patch scripts in files (scratchpad: `C:\Users\imalv\AppData\Local\Temp\claude\D--Projects-transient-steering\150e3cfc-7fd8-40d5-abf1-74290f7f88c1\scratchpad\`), and use single backslashes inside Edit calls. Do not chain `sleep` in Bash; run long jobs in the background and wait for the notification.
- `tokenizer(prompt)` / `tokenizer.encode` already include the start token in this setup; use `add_special_tokens=False` for single words.
- The median-rank convention: the upper middle value for an even count (the D8 file keeps the lower one).
- GPU: do not run two heavy jobs at once (3 edit workers use about 4 GB); the training run needs about 3 GB plus the models.
- A tool run that the user declined must not be retried: wait for their instruction.

## 8. How to resume
1. `git status -sb` and `git log --oneline -5`; check whether the D13 hard-arm patch and the generation record are committed. 2. Read the user's newest message. If they pasted the D13 training RESULT block: explain it as a story, then record it (section 3). If they ask something else, answer in the plain style (section 0). 3. Before any new run: plan and gate into `PLAN.md` first, new code as new files, rehearse with `--smoke`, then give the user the command; commit and push the plan and tool before the run.
