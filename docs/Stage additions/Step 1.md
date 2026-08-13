# Step 1 — Iterative Ablation + Specificity Check

**Goal of this step, in one line:** find out, per fact, how many SAE features must be
partially ablated (and in what order) before the correct answer becomes the model's
top prediction — then check that doing so doesn't break unrelated facts.

**Why this is the current priority:** everything else on the roadmap (attribution
patching, baselines, the learned predictor, model upgrades) depends on knowing whether
the basic ablation approach actually works reliably across more than one fact. Right
now it's validated on exactly one (Eiffel Tower → Paris). This step turns that from an
anecdote into either a real, general finding or a documented limitation.

---

## What gets built

### 1. Iterative ablation loop (`editing.py`)

A function, e.g. `iterative_ablate(model, sae, prompt, target_str, max_rounds=5,
strength=0.3)`, that:

- Runs the causal selector to find the current top blocking feature.
- Ablates it (using the existing `make_ablation_hook`, at a fixed, already-validated
  strength — start at ~0.3, don't re-litigate the strength question in this step).
- Re-runs the causal selector **on the now-modified forward pass** to see if the
  target token is now top-1.
  - If yes → stop. Record the ordered list of features ablated and how many rounds
    it took.
  - If no → find the new top blocking feature and repeat.
- Stops after `max_rounds` if the target still hasn't become top-1, and records this
  as a failure for that fact (not silently drops it).
- Returns a structured result per fact: `{prompt, target, success: bool,
  rounds_used: int, features_ablated: [ids in order], final_target_rank}`.

**Must reuse**, not reimplement: `run_causal_selector`, `make_ablation_hook`,
`get_target_token_id` (with its earlier leading-space/first-token fix intact).

### 2. Specificity check (new: `specificity.py`, or a function added to `evaluation.py`)

For each fact where `iterative_ablate` succeeded:

- Take the exact ordered set of ablated features.
- Re-apply that **same** set of ablations (same features, same strength) to a small
  batch of **unrelated control prompts** (a handful of the `already_correct` facts
  from the existing 46-fact classification are a natural, already-available choice —
  no new data needed).
- Check whether each control prompt's top-1 prediction is unchanged.
- Record a specificity score per fact: e.g. `X / N control prompts unaffected`.

### 3. Batch runner

Run steps 1–2 across all **22 confirmed-suppressed facts** from the existing
`scan_fact_batch` output (already generated — don't regenerate it, load/reuse it).

Output: one table with, per fact — success/failure, rounds used, features ablated,
specificity score. This table is the actual deliverable of this step.

---

## Explicit constraints (do not let this expand)

- No attribution patching in this step — brute-force search is fine at this scale (22
  facts). Attribution patching is a later, separate task, only needed if this becomes
  too slow.
- No baseline comparisons (ROME, mean-difference steering) in this step.
- No changes to the Streamlit app's UI/copy in this step — this is a backend/evaluation
  task. If results are worth surfacing in the app later, that's a separate follow-up.
- No changes to the SAE, the model, or the hook's core math (`hooks.py` stays as-is).
- Fixed ablation strength (~0.3) for this pass — don't turn this into a second
  hyperparameter search on top of everything else.

## What "done" looks like

A results table covering all 22 facts with: success rate (how many hit top-1 within
`max_rounds`), average rounds needed, and average specificity score. This is the first
piece of real evidence — good or bad — for whether the core technique generalizes.