# Session Summary — Weighted Multi-Competitor Reduction & Research Infrastructure
**Date:** 2026-07-25

---

# 1. Core Algorithmic & Engine Changes

## Joint Safety Checker (`src/editing.py`)

Implemented `check_combination_safe`, which performs a single forward pass using the complete combined intervention (all muted and boosted features simultaneously).

### Purpose

- Evaluates the actual combined edit instead of individual interventions.
- Detects only **new blockers** introduced by the edit.
- Ignores competitors that were already ranked above the target in the clean baseline.

---

## Weighted Multi-Competitor Reduction (`src/editing.py`)

Implemented weighted suppression of multiple competing tokens simultaneously.

### New Components

### `make_weighted_ablation_hook()`

- Supports simultaneous suppression of multiple SAE features.
- Allows each feature to receive an independent scaling weight.

### `run_weighted_multi_competitor_reduction()`

Pipeline:

1. Identify every token ranked above the target.
2. Compute normalized probability weights for all competitors.
3. Select each competitor's strongest driving SAE feature.
4. Aggregate shared features using the current `max()` strategy.
5. Apply weighted suppression.
6. Run the joint safety checker.

---

## Explainable AI Module (`src/explain.py`)

Created a standalone explanation layer using:

- Groq Chat Completions API
- `llama-3.1-8b-instant`

Generates concise (2–3 sentence) mechanistic summaries describing:

- Prompt context
- Target token
- Applied SAE feature interventions
- Feature descriptions
- Expected steering behavior

---

# 2. Dashboard Improvements (`experiment_app.py`)

## UI Improvements

- Added 🧪 page icon.
- Converted headers to sentence casing.
- Updated tab names with icons.

Examples:

- 🧪 Single-trace iterative ablation
- ⚖️ Weighted multi-competitor reduction
- 📊 Session history and benchmarks

---

## Neuronpedia Hover Integration

Implemented:

- Cached scraper:
  - `get_neuronpedia_explanation()`
- HTML helper:
  - `make_feature_hover_link()`

Integrated across Tabs:

- Tab 1
- Tab 2
- Tab 4
- Tab 5
- Tab 7

Features now:

- Open Neuronpedia in a new tab.
- Display explanation via HTML tooltip on hover.

---

## Tab 7 — Weighted Multi-Competitor Reduction

Added complete experiment interface including:

- Configuration controls
- Results dashboard
- Competitor table
- Baseline probabilities
- Normalized weights
- Hover-linked feature IDs

---

## Explainable AI Integration

Sidebar additions:

- Toggle explanations
- Secure Groq API key input

Explanation panel integrated into:

- Hybrid Mute & Boost
- Weighted Reduction

Narratives are generated automatically after each run.

---

# 3. Research Infrastructure

Created structured research workspace.

```
research/
├── README.md
├── CONTRIBUTING_RESEARCH.md
├── bibliography.bib
├── figures/
├── notebook/
├── hypothesis_log.md
├── timeline.md
└── experiment_index.md
```

---

## Research Notebook Entries

### Experiment 002

Competitor-Focused Iterative Ablation

- Documented methodology
- Reported 13.64% success rate

---

### Experiment 004

Weighted Multi-Competitor Reduction

Documented:

- Cambridge benchmark
- Feature 313 / 21169 collapse

---

### Experiment 006

Hybrid Mute & Boost Steering

Documented successful steering on:

- Dublin
- Melbourne

---

## Research Journal

### `docs/Research_Journal/4.md`

Added:

- Mathematical formulation of weighted reduction
- Joint safety checker
- Distributed Support ("Loudspeaker") Hypothesis

---

## Notes

Restored complete chronological project history in:

```
docs/notes.md
```

---

# Research Note — Analysis of Weighted Multi-Competitor Reduction (Cambridge Case)

**Date:** 2026-07-25

---

## Background

The Weighted Multi-Competitor Reduction algorithm was introduced to overcome the plateau observed in the original single-competitor reduction approach.

The original method only suppressed the highest-ranked competing token (for example, **"the"**), allowing other high-probability competitors (such as **"a"**) to remain unaffected.

The hypothesis was that suppressing every competitor proportionally would allow the target token (**"Cambridge"**) to continue climbing in rank.

---

# Diagnostic Experiment

A temporary diagnostic was added inside `run_weighted_multi_competitor_reduction()`.

The intervention logic was intentionally left unchanged.

For every competitor ranked above the target token, the diagnostic prints:

- Competitor token
- Competitor probability
- Selected top driving SAE feature

---

## Cambridge Diagnostic Output

```
--- DIAGNOSTIC: COMPETITORS ABOVE TARGET ---

Token: ' the'       | Prob: 17.95% | Top Feature: 313
Token: ' a'         | Prob: 10.53% | Top Feature: 313
Token: ' question'  | Prob: 7.51%  | Top Feature: 21169
Token: ' jeopardy'  | Prob: 2.71%  | Top Feature: 21169
Token: ' an'        | Prob: 2.63%  | Top Feature: 313
Token: ' danger'    | Prob: 1.76%  | Top Feature: 21169
Token: ' doubt'     | Prob: 1.21%  | Top Feature: 21169
```

---

## Intervention Results

```
Is Safe: True

Target Clean Rank: 8
Target Clean Probability: 1.1525%

Target New Rank: 8
Target New Probability: 1.2900%

Muted Feature Strengths:
{
    313:   0.28366593961499226,
    21169: 0.11863800655380938
}

New Blockers: []
```

---

# Observation

Although **seven competing tokens** were identified, they collapsed into only **two unique SAE features**.

### Feature Assignments

| SAE Feature | Competitors |
|-------------|-------------|
| **313** | the, a, an |
| **21169** | question, jeopardy, danger, doubt |

Thus, despite beginning with seven competitors, the weighted intervention ultimately edited only **two residual directions**.

---

# Interpretation

This suggests that multiple competing tokens may share common latent grammatical or semantic mechanisms rather than being represented by independent SAE features.

The intervention is therefore not operating on seven independent competitors.

Instead, it is operating on two shared latent mechanisms.

---

# Unexpected Finding

Despite targeting every competitor ranked above the target, the intervention produced only a modest improvement.

```
Target Probability

1.1525%
↓

1.2900%
```

Target rank remained unchanged:

```
8 → 8
```

This indicates that suppressing only the strongest feature is insufficient to substantially reduce the probability of these high-frequency competitors.

---

# Current Working Hypothesis (Unverified)

The current implementation assumes that a competitor's strongest SAE feature is an adequate proxy for that competitor.

The Cambridge diagnostic suggests this assumption may not hold for high-frequency grammatical tokens.

These competitors may instead be supported by multiple influential SAE features, with the top feature accounting for only part of their probability.

If true, suppressing only the strongest feature leaves much of the competitor's supporting representation untouched.

This hypothesis requires further experimental verification.

---

# Open Research Questions

1. How much of a competitor's logit is actually explained by its strongest SAE feature?

2. Are grammatical competitors represented by a more distributed feature set than factual tokens?

3. Would selecting the Top-K features per competitor produce larger ranking improvements than using only the single strongest feature?

4. Is `max()` the correct aggregation strategy when multiple competitors share the same latent feature, or should their contributions be combined differently?

These remain open research questions.