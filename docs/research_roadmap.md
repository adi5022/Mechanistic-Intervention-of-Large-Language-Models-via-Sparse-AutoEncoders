# Research roadmap (written 2026-10-04, branch `gradient-descent-editing`)

Goal: a paper-style comparison of the ways we tried to make GPT-2 small (layer 8, `gpt2-small-res-jb`) say a chosen word, ranked on one common yardstick, and set against established methods. Status labels are honest: all measures are exploratory proxies, and "reaches rank 1" alone is not evidence of a fix (Entry 30 sections 10-12).

## Where things stand
- Per-feature multiplier edit by gradient descent beats the fixed sweep at equal range: rank 1 on 271 of 300 held-out prompts vs 161, lower same-prompt KL, about 4-6x faster (Entry 29).
- The additive edit (switching on silent features) reaches rank 1 for random unrelated words on 20 of 20 and transfers equally to reworded prompts for random and true targets, so rank 1 says it is strong, not specific (Entry 30 sections 10-11).
- The old rule "unreachable by steering = not in the model" is withdrawn (Entry 30 section 12, H23).
- Hand checks in the Prototype lab: with the additive edit kept on, "The capital of Germany is" continues "Hong Kong" (Entry 30 section 14). One prompt each, not a measurement.
- Greedy GPT-2 small loops by itself; repetition must be judged against the baseline's own repetition.

## Plan, in order

### A. Specificity benchmark (the yardstick)  *(next, to be done by the author)*
For each edit, on the same prompts, measure:
1. Reach: target rank 1 on the tuned prompt.
2. Transfer: target rise on reworded prompts.
3. Leak: next-word KL and top-1 flips on unrelated prompts.
4. Text quality: distinct-2, loop rate, tail perplexity over 20 generated tokens, against the baseline's own values.
5. Context sensitivity: still forced when the prompt says "Do not say X" / "X is the wrong answer"?

Controls: true target vs random word; a rank-matched random control (random words chosen to start at ranks similar to the true targets, since 17,504 vs 17 was an unfair comparison); arms gd, gd_add@cap=1.0, gd_add@cap=0.25, plus the fixed sweep (needs a rerun: the Entry 29 pack does not store the sweep's final edits).

Already built: `tools/test_generalisation.py` (reach, transfer, leak), `tools/test_generation_quality.py` (text quality, context sensitivity; 4-prompt smoke run passed, 40-prompt runs not yet done), the follow-up leak box in the Prototype lab.
Still to build: the rank-matched control, the sweep arm, a larger transfer test (about 150 prompts, with confidence intervals), one combined results table with paired statistics, Entry 31 with data packs and figures from real data only.
Rough cost from the smoke run: 1-1.5 h GPU per 40-prompt run.

### D. In-context fact vector (one more method to rank)
Run the model with the fact stated ("Fact: the capital of France is Hong Kong. The capital of France is") and without; take the layer-8 residual at the last token in both; subtract; add the difference to the clean run scaled by alpha. Variants: raw residual vs SAE feature space. Test with the yardstick from A against the other arms and with random-word facts as control. Expected risk: it may behave like the additive edit (push toward the word, leaks); the told-minus-clean difference also contains the reaction to the framing words. The told run doubles as the plain prompt-engineering baseline, which a reader will ask for (AxBench: prompting beat SAE steering).

### B. Established methods (EXP-015)
Run IKE, ROME and DiffMean on the same prompts and measures. Without this the paper can rank our own methods but cannot say how they compare with published ones.

### Write-up
One ranked results table, honest framing (exploratory, proxy), withdrawn claims stated, paper draft `research/paper/draft_v1/main.tex` updated last.

## Queued ideas (not started)
- Stop-early option for the gradient descent (end after N steps without improvement; off by default to stay comparable to Entry 29). The run currently uses all 100 steps and keeps the best iterate, so continuing after rank 1 costs time but cannot worsen the result.
- Paraphrase and neighbour-prompt terms in the objective to make the multiplier edit more specific (then re-run A).
- Control in the follow-up box: unedited model on the prompt that contains the edited text, to separate the edit's leak from ordinary copying of context.
- Optional repetition penalty / no-repeat-ngram / sampling toggle for the generation view.
- Fix failing-prompt KL (`kl_always` arm exists, not evaluated at scale); multi-token target scoring.
- Gradient descent as teacher for FeatureNet (`docs/future_scope.md`); steering-controller LLM front end (last).
- Open decision: keep the full BOS fix or cut it back to the self-heal only (old files `src/editing.py` and `src/hybrid_runner.py` were edited without asking first; see Entry 30 section 13).

## Added 2026-10-10: cross-model transfer (branch `cross-model-transfer`)
A new line of work next to A to D above, with its own runbook (`docs/cross_model_transfer/PLAN.md`) and handoff (`docs/handoff_2026-10-10_cross_model.md`): send the change an edit makes in GPT-2 small into GPT-2 medium (Export) or medium's state into small (Import) through a linear translator fitted on ordinary text. Translators and a country-swap control are done (Entry 31); a pilot of a real edit shows a partial, above-control effect that a plain word push exceeds. Next: Phase C (is the gradient-descent edit a fact edit or a word push; shares the CounterFact benchmark with this roadmap's item A and the ROME comparison B), then the controlled export (D) and the import (E). It reuses this roadmap's measures (reach, transfer to rewordings, leak onto unrelated prompts) and its lesson that rank 1 alone is not evidence. Update (later 2026-10-10, Entry 33): several injection layers do not help; an edit tuned through the translator against medium's output reaches rank 1 on 95% (original edit 8%); a linear translator trained on edits with an output-matching loss (D7) reaches 18% against 8% for the ridge map and fails its gate (50%). Related work note: `docs/cross_model_transfer/RELATED_WORK.md`.
