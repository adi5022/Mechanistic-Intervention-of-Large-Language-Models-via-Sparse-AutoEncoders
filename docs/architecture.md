# FeatureScalpel — Full Technical Mechanics

This document answers one question precisely: **what exactly gets touched inside the model, and why does touching it fix the output.** Everything else in the project hangs off this.

---

## 1. The Core Claim, Stated Precisely

> The model already contains the correct answer, encoded somewhere in its activations. A wrong output happens because, during this specific forward pass, the "wrong" internal signal outweighs the "right" one at the point of decoding. We don't add knowledge. We rebalance an existing internal computation, for one inference only, then discard the change.

No retraining. No fine-tuning. No saved memory. No effect on any future prompt. Every edit is scoped to a single forward pass and vanishes the instant it ends.

---

## 2. What Exactly Gets Edited — The Direct Answer

You asked the exact right question: is it the SAE latent, the residual stream, attention output, or MLP output? Here's the precise chain:

### 2.1 The residual stream is the actual edit site

A transformer's residual stream is a running vector, at every layer, that every component (attention, MLP) reads from and adds back into. It is the single shared "workspace" of the model. **This is what we edit — not the SAE, not attention weights, not MLP weights.**

### 2.2 The SAE is the diagnostic instrument, not the edit target

The SAE's job is only to tell us **which direction inside the residual stream corresponds to a specific human concept**. Mechanically:

```
residual_stream_activation (a vector, e.g. 768 numbers for GPT-2-small)
        │
        ▼
   SAE Encoder:  features = ReLU(W_enc · activation + b_enc)
        │            (features is a much longer, mostly-zero vector —
        │             e.g. 24,576 numbers, only a handful nonzero)
        ▼
   SAE Decoder:  reconstruction = W_dec · features + b_dec
        │            (should closely match the original activation)
```

Each row of `W_dec` (the decoder) is a **direction in residual-stream space**. That direction is the "feature" — e.g., one direction might correspond to "European capital cities," another to "wrong-fact-about-landmarks." The SAE's whole value is that it found these directions **automatically, by training on lots of activations**, disentangling concepts that are normally superposed (blended together) in the raw residual stream.

### 2.3 The actual edit operation

Once you know which decoder direction (`W_dec[i]`) corresponds to the wrong concept:

```
new_residual_stream = original_residual_stream − (feature_activation[i] × W_dec[i])
```

That's an **ablation**: you subtract exactly the component of the activation that the SAE attributes to that feature. If instead you want to *push toward* a correct concept, you add a scaled version of that concept's decoder direction (**amplification/steering**):

```
new_residual_stream = original_residual_stream + (α × W_dec[correct_feature])
```

You then **substitute this modified vector back into the model's residual stream at that exact layer**, via a hook, and let the forward pass continue normally through all remaining layers as if nothing happened — except the number that layer handed forward is now slightly different.

### 2.4 Where attention and MLP fit in

Attention and MLP blocks **write into** the residual stream (that's literally what "residual" means — they compute something and add it in). We don't touch their internal weights or outputs directly. By editing the residual stream itself, our intervention sits **downstream of everything that happened before that layer, and upstream of everything after it** — the simplest, least invasive point to intervene, and the same point SAEs are conventionally trained on.

**One-line summary of section 2:** *SAE latents tell you which direction in the residual stream is "the wrong concept." The actual edit modifies the residual stream itself, at one layer, for one forward pass, by adding or subtracting that direction.*

---

## 3. Why This Actually Works (the mechanism, not just the recipe)

- The residual stream is **additive** — every layer just adds its contribution on top of what's already there. This means a "wrong belief" signal and a "right belief" signal can coexist in the same vector, superposed, rather than living in separate neurons.
- A trained SAE finds a (largely) **near-orthogonal** set of directions that decompose this blend back into separate, mostly-independent concepts. Near-orthogonality is what lets you touch one direction (subtract the wrong-fact direction) **without meaningfully disturbing unrelated directions** — this is the theoretical basis for the *specificity* property you're benchmarking.
- Because nothing is retrained, this only ever affects the **current forward pass** — the next time you run the model (even on the exact same prompt), the weights are untouched, so nothing has "stuck." That's why there's no state to save: there is no persistent change, by design.

---

## 4. Full Pipeline (end to end)

```
1. Prompt in
        │
2. Model runs forward, we capture residual stream at chosen layer L
        │
3. SAE decodes that activation into sparse features
        │
4. We identify: is the "wrong concept" feature active? How strongly?
        │
5. If yes → construct edited residual stream (subtract/add feature direction)
        │
6. Hook re-injects the edited vector back into the model's forward pass at layer L
        │
7. Remaining layers (L+1 ... final) compute normally on the edited value
        │
8. Output logits reflect the correction
        │
9. Nothing is saved. Weights unchanged. Next prompt starts fresh.
```

---

## 5. Why NOT the Other Approaches — Precise Differences

| Method | What it changes | Persistence | Where the "new" info comes from |
|---|---|---|---|
| **RAG** | Nothing inside the model — appends retrieved text to the *input* | None (per-call) | External documents, fed in as text |
| **Fine-tuning** | Model weights, via gradient descent over many examples | Permanent, affects all future prompts | Training data, generalized into weights |
| **Prompting** | Nothing internal — relies on the model *choosing* to follow an instruction | None, and unreliable — no guarantee it obeys | None — just hopes the existing knowledge surfaces |
| **Model editing (ROME/MEMIT)** | Specific weight matrices (e.g. one MLP down-projection), via a closed-form rank-one update from causal tracing | **Permanent** — the weight is actually changed | None — reweights an existing association |
| **Simple activation steering (AxBench baseline)** | Residual stream, using a direction computed as a simple difference-of-means between "correct" and "incorrect" example activations | Per-inference only (like ours) | None — reweights existing signal, cheaply, without a trained dictionary |
| **Ours (SAE-based edit)** | Residual stream, using a direction identified via a trained sparse dictionary that disentangles superposed concepts | **Per-inference only** — fully reversible, nothing saved | None — reweights existing signal, but via an interpretable, disentangled direction rather than a raw statistical difference |

**The key distinguishing facts:**
- **Vs. RAG/fine-tuning/prompting:** those either inject new information externally (RAG) or permanently alter the model (fine-tuning) or hope rather than guarantee (prompting). We do neither — no new info, no permanent change, and a mechanically guaranteed effect (you directly manipulate the number that determines the output).
- **Vs. ROME/MEMIT:** they edit weights permanently; we edit activations transiently, per-call, fully reversible with a toggle. This is a real, substantive difference worth stating explicitly: **ours is a live "dial," not a permanent surgery.**
- **Vs. simple activation steering:** the *only* difference is where the edit direction comes from — a cheap, un-decomposed statistical difference (baseline) vs. a properly disentangled sparse-dictionary direction (SAE). This is exactly the comparison your project benchmarks, because AxBench found the cheap version sometimes wins anyway.

---

## 6. Novelty, Precisely Stated

Not: "we invented activation editing" (ROME did that in 2022; steering existed before that).
Not: "we invented SAEs for editing" (SAFE, SALVE, and others already tried this in 2025).

**Actual novelty claim:** a controlled, apples-to-apples comparison of all three mechanisms (permanent weight edit, cheap activation steering, SAE-based activation edit) on the same model, same facts, same metrics — reporting honestly which one wins, by how much, and why — which the existing literature does not provide together in one place.

---

## 7. Research Fields Involved

- **Mechanistic Interpretability** (core field — understanding what's inside the model)
- **Representation Engineering / Activation Steering** (the "editing the residual stream" literature)
- **Knowledge Editing** (the ROME/MEMIT line — the classical baseline you're comparing against)
- **Sparse Coding / Dictionary Learning** (the mathematical machinery underlying SAEs, borrowed from signal processing and computational neuroscience)

---

## 8. Concepts to Learn First (in order)

1. Vanilla autoencoders (encoder → bottleneck → decoder, reconstruction loss)
2. Residual stream (the transformer's shared additive workspace)
3. Superposition (why one neuron ≠ one concept)
4. Sparse Autoencoders (SAEs) — architecture + sparsity penalty
5. Activation patching / causal tracing (the general "intervene and observe" technique)
6. Rank-one weight edits (how ROME edits weights, for contrast against your approach)

---

## 9. Papers, in Reading Order

1. **Anthropic, "Towards Monosemanticity" (2023)** — superposition + SAEs, the foundational paper for everything in section 2.
2. **Meng et al., "Locating and Editing Factual Associations in GPT" (ROME, 2022)** — your baseline method, and the source of the CounterFact benchmark/metrics.
3. **Wu et al., "AxBench" (2025)** — the paper that motivates your comparative framing; read this before building anything, not after.
4. Skim: Farrell et al. (SAE-based unlearning), Rajamanoharan et al. (JumpReLU SAEs) — only needed if you reach stretch goals.

---

## 10. How to Start Today

1. Open a Colab notebook, request a T4 GPU.
2. `pip install transformer_lens sae_lens`
3. Load GPT-2-small via TransformerLens.
4. Load a pretrained GPT-2-small SAE from a public hub (SAELens quickstart shows exactly this) — don't train your own yet.
5. Run one prompt, capture the residual stream at the SAE's layer, decode into features, print the top 5 active features and their associated "meaning" (via example text snippets from the SAE's documentation/hub).
6. Pick one feature, zero it out via a hook, regenerate, see if the output changed at all.

If step 6 produces any visible change, your entire pipeline's critical path is proven — everything else is refinement.

---

## 11. MVP (Minimum Viable Project)

- One model (GPT-2-small).
- One pretrained SAE (not self-trained yet).
- One hand-picked fact where the base model is wrong.
- One ablation that fixes it, shown live, toggle on/off.
- One baseline (mean-difference steering) run on the same fact, for comparison.

That's a complete, demoable MVP — everything past this (self-trained SAEs, ROME baseline, full CounterFact benchmark) is depth added on top of a working core.

---

## 12. Experiments

1. **Feature discovery:** for a curated fact set, find which SAE feature(s) correlate with the wrong answer.
2. **Ablation efficacy:** does suppressing that feature flip the model's preferred answer (measured via output probability, not just one sampled sentence)?
3. **Generalization:** does the fix hold across paraphrased versions of the same question?
4. **Specificity:** do nearby, unrelated facts stay unchanged after the edit (checked via a small held-out set of neighborhood prompts)?
5. **Three-way comparison:** repeat 2–4 for the mean-difference steering baseline and for ROME, on the same fact set.

---

## 13. Benchmarks

- **CounterFact** (Meng et al., 2022) — provides facts, paraphrases, and neighborhood prompts, with standard efficacy/generalization/specificity metrics.
- **CounterFact+** refinement — a stricter specificity check (catches side effects the original metric misses); use if time allows, cite the limitation if not.
- **Custom mini-benchmark** — 150–300 hand-curated facts if the CounterFact format doesn't map cleanly onto your chosen model/task, with the same three metrics computed manually.

---

## 14. Performance & Caching Architecture

1. **Hardware Acceleration Engine (`get_default_device`)**:
   - Automatically detects PyTorch hardware acceleration (`cuda` GPU, Apple `mps`, or `cpu` fallback).
   - Offloads transformer forward passes, SAE dictionary encodings, and autoregressive generation to GPU (NVIDIA GTX 1660 Ti), achieving 10x–20x execution speedups.

2. **Decoupled Streamlit Caching**:
   - `get_cached_base_model()` caches the 500MB `HookedTransformer("gpt2")` base model once per session (`@st.cache_resource`).
   - `get_cached_sae(layer)` loads layer-specific SAE dictionaries (~20MB) independently, enabling instant layer switching (< 0.2s) without re-instantiating the base transformer.

3. **Hugging Face Authentication**:
   - Automatically checks `HF_TOKEN` from environment variables, `.env`, or `.streamlit/secrets.toml` and authenticates via `huggingface_hub.login`.
   - Prevents anonymous API rate limits when downloading model weights and SAE dictionaries.

---

## 15. Layer Intervention Benchmark Subsystem

1. **Decoupled Benchmark Engine (`src/benchmark/layer_benchmark_runner.py`)**:
   - Encapsulates UI-independent layer characterization sweeps (`run_layer_benchmark`).
   - Supports arbitrary transformer block depth sweeps, safety filtering toggles (`use_safety`), clean baseline passes, candidate feature safety checks (`check_target_safe`, `check_boost_safe`), and joint intervention passes.
   - Automatically persists research JSON artifacts into `benchmark_results/layer_benchmark_YYYYMMDD_HHMMSS_ffffff.json`.

2. **Standalone Benchmark Application (`layer_benchmark.py`)**:
   - Independent Streamlit dashboard featuring a dynamic **Prompts Dataset** editor (`➕ Add Prompt`, `➖ Remove Prompt`).
   - Leverages `@st.cache_resource` for zero-redundancy model/SAE reuse across multi-prompt datasets.
   - Displays real-time progress indicators, comparative performance tables, Vega-lite visualization charts, auto-saved artifact location captions, and one-click JSON download buttons per prompt.
