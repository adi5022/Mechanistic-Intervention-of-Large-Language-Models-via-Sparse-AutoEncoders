# Agent Task: Build FeatureScalpel MVP Prototype

## Project Context (read this fully before writing any code)

**FeatureScalpel** is a semester research project in Knowledge Representation and
Reasoning (KRR). The core idea: use Sparse Autoencoders (SAEs) to decompose a
small language model's internal activations into interpretable features, then
apply targeted, inference-time behavioral corrections — WITHOUT any weight
editing, fine-tuning, or persistent state change.

**Why this matters (the pitch):** weight-editing methods like ROME and MEMIT
suffer documented catastrophic degradation under sequential edits. Because
FeatureScalpel never touches model weights — it only intervenes on activations
during the forward pass — this failure mode doesn't apply. Every intervention
is transient and reversible by construction.

**Important framing distinction:** this is NOT retrieval-augmented generation
(RAG). We are not injecting external information into the model. We are
rebalancing signal that already exists inside the model's internal
representations. The model already "knows" things internally that don't
surface correctly at the output layer; we're correcting the internal balance,
not adding new facts.

## What has already been done and confirmed (do not re-derive from scratch —
these are established findings from real experiments, use them as ground truth
to validate your build against)

- Model: **GPT-2-small**, loaded via **TransformerLens**.
- SAE: pretrained, loaded via **SAELens**, release `gpt2-small-res-jb`, layer 8
  residual stream, hook point `blocks.8.hook_resid_post`.
- On the prompt `"The Eiffel Tower is in the city of"`, GPT-2-small's top
  prediction is **"London"** (wrong; correct answer is "Paris").
- SAE feature **11149** was identified (initially via Neuronpedia lookup, i.e.
  by educated guess, NOT by rigorous method) as a broad "city-name prediction"
  feature active at the final token of city-related prompts.
- **Full ablation** (scaling feature 11149's activation to zero) on the Eiffel
  Tower prompt flips the top prediction from "London" to "Paris" — the
  correction works.
- **Specificity test**: the same full ablation applied to
  `"The Colosseum is in the city of"` (correct answer "Rome", already correct
  at baseline) BREAKS the correct answer — output degrades toward degenerate
  token fragments. This proved feature 11149 is NOT narrowly localized to
  "wrong city corrections" — it's a broad city-token feature, and killing it
  entirely damages unrelated correct predictions too. This was treated as a
  legitimate research finding, not a bug.
- **Partial ablation sweep** (strengths 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0) was
  run across 5 test prompts. Key confirmed results:
  - Eiffel/Paris: broken at 0.0–0.2, correct token becomes top-1 starting at
    **strength 0.3**, and improves further through 1.0.
  - Colosseum/Rome: correct through strength 0.3, degrades starting at
    **strength 0.5** (Rome drops out of top-5 by 0.7).
  - **Conclusion: there is a real "sweet spot" around 0.2–0.3 ablation
    strength where the Eiffel error gets fixed WITHOUT breaking the Colosseum
    case.** This is the core evidence that a soft "knob," not a hard on/off
    switch, is the right intervention mechanism.
  - Two other test prompts (Big Ben/London, Sagrada Familia/Barcelona) were
    already wrong at baseline with very low correct-token rank, and ablating
    feature 11149 at any strength did NOT fix them. This is an important
    **scope boundary**, not a failure: this technique corrects cases where the
    correct answer already has non-trivial probability mass in the
    distribution but is being suppressed/outcompeted — it does NOT inject
    missing knowledge the model never encoded in the first place.
- A candidate ablation hook implementation exists that: encodes the residual
  stream into SAE feature space, scales down one target feature's activation
  by `strength` (0 = untouched, 1 = fully zeroed), decodes back, and patches
  only the DELTA caused by the intervention onto the original residual stream
  (not a full replacement) — this avoids contaminating results with the SAE's
  own reconstruction error.

## The identified research gap (why today's build matters)

Feature 11149 was found by **manual guesswork**: look at top active features
by raw activation magnitude, cross-reference Neuronpedia's auto-interp label,
pick the one that "sounds relevant," test it. This is not scalable and not
rigorous — high activation magnitude does NOT imply causal responsibility for
the output (a feature can be highly active without being what's driving the
wrong prediction). Neuronpedia labels also won't exist at all once we train
our own SAE later in the project, so the whole approach must work without
relying on external documentation.

**The fix**: an automated **Feature Selector** that ranks features by measured
causal effect rather than by raw activation size or external labels:

1. Given a prompt, run the model and SAE-encode the final-token residual
   stream activation.
2. Take the top-N (e.g. N=20) features by raw activation magnitude as
   candidates (this bounds the search — do not brute-force all ~24,000
   features).
3. For each candidate feature: fully ablate it alone, rerun the model, measure
   the resulting change in the target/correct token's probability (and
   optionally KL divergence of the full output distribution vs. clean
   baseline).
4. Rank the candidates by magnitude of causal effect (|Δ probability| or KL).
5. **Validation check**: confirm that for the Eiffel Tower prompt, feature
   11149 appears at or near the top of this automatically generated ranking.
   This is the single most important thing this prototype needs to
   demonstrate — that an automated, principled selection method independently
   rediscovers the same feature found earlier by manual guesswork.

## What to build

A single-page interactive prototype (Streamlit is fine, or a lightweight
Flask/React app if you prefer — your call, but keep it to ONE cohesive page,
not a multi-tab dashboard; this is a proof-of-concept, not the final product)
with three visible sections, in this order:

### Section 1 — Feature Selector (the new, automated part)
- Prompt input (default to `"The Eiffel Tower is in the city of"`, but allow
  free text).
- On submit: run the model, SAE-encode, extract top-20 active features by
  activation magnitude.
- Ablate each candidate one at a time, measure the shift in probability of
  whatever the model currently considers the "expected correct" token (accept
  this as a user-supplied field, e.g. `" Paris"` — don't try to auto-detect
  ground truth facts, that's out of scope).
- Display a ranked table/bar chart of the top candidates by causal effect
  size, feature ID, and resulting top-1 prediction after ablating that feature
  alone.
- Visually highlight if feature 11149 is in the top few results (confirms the
  validation claim above).

### Section 2 — The Slider (manual intervention, already validated)
- Dropdown or selection of a feature ID (pre-filled with whatever the
  Feature Selector ranked #1, but editable).
- A slider from 0.0 to 1.0 for ablation strength.
- Live before/after display: top-5 tokens + probabilities at baseline vs. at
  current slider value, for the currently selected prompt.
- A small inline note or tooltip when the user is testing the Eiffel/Colosseum
  pair specifically, referencing the known 0.2–0.3 sweet spot / 0.5+
  degradation finding, so the demo visually confirms the previously-found
  result rather than just looking like an unexplained slider.

### Section 3 — Specificity / Side-Effect Check
- Given the currently selected feature and strength, also show the effect on
  1–2 OTHER stored prompts (at minimum reproduce Colosseum/Rome) so the user
  can see, side by side, "does this fix the target case while breaking this
  other case, yes or no" — this is the core scientific question of the whole
  project (specificity of intervention) and should be front and center
  visually, e.g. a red/green indicator per prompt showing "still correct" vs.
  "broken by this intervention."

## Technical requirements
- Use TransformerLens (`HookedTransformer.from_pretrained("gpt2")`) and
  SAELens (`SAE.from_pretrained("gpt2-small-res-jb", ...)`) exactly as in
  prior work — do not substitute other libraries.
- Ablation hook must use the delta-patching method described above (encode →
  modify target feature → decode → patch only the delta onto the original
  residual), not a full reconstruction replacement.
- Cache the clean (unablated) run per prompt so repeated slider movements
  don't require re-running the full forward pass unnecessarily — keep the UI
  responsive.
- All computation can assume CPU or single-GPU Colab/local environment; no
  need for distributed or batched optimization.
- Do not add authentication, persistence/database, or multi-user support —
  this is a single-session local demo.

## Explicit non-goals for this prototype (do not build these; they would blow
the time budget and aren't needed to test what's described above)
- No automatic ground-truth fact detection / error-finding across arbitrary
  prompts.
- No benchmark comparison against ROME or mean-difference steering yet.
- No training of a custom SAE.
- No multi-tab full dashboard — one page, three sections, as above.

## Deliverable
A working prototype (single Python file or minimal file set) that a user can
run locally, enter the Eiffel Tower prompt, watch the Feature Selector surface
feature 11149 near the top automatically, then move the slider and watch the
Eiffel and Colosseum cases react differently at different strengths — visually
reproducing every numeric finding listed above.
