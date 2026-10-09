# Cross-Model Activation Patching: Idea Note

Status: idea / design note. Nothing in this file has been run. Every number below is either an estimate (marked **est.**) or a claim from a literature search that I only saw as search snippets (marked **verify**). Read the papers before citing them.

---

## 1. The idea in one paragraph

GPT-2 small is a weak model. A stronger model knows things GPT-2 does not. While GPT-2 is running on a prompt, we also run the stronger model on the **same text**, use sparse autoencoder (SAE) features to find *what* the stronger model is representing at a middle layer, turn that into a change (a **delta**) in GPT-2's residual stream, and add it. GPT-2's remaining layers then process the patched state and produce the output. The question is whether this makes GPT-2's answer better, specifically and without side effects.

This is an **inference-time intervention**. No weights change. It is **not** an ensemble: we never average the two models' outputs. The final output is always the weak model's own forward pass, after the patch.

## 2. Why this comes from the existing project

The project already edits GPT-2 small at layer 8 (`gpt2-small-res-jb`, hook `blocks.8.hook_resid_pre`) with a delta in the residual stream:

```
delta = sum_k (m_k - 1) * act_k * W_dec[k]
```

(The SAE reconstruction error is deliberately not injected.) That machinery (hooks, KL side-effect measure, specificity benchmark, controls) is exactly what this experiment needs. The new part is where the delta comes from: a second model instead of the model's own gradient-descent multipliers.

The idea grew out of the interpretability work as a side experiment. It is **not** claimed as a new invention of model stitching (see section 9).

## 3. Two directions, and which one serves the goal

The idea has two directions that are easy to mix up. Decide which one a given experiment is.

| | Direction 1: strong -> weak (the capacity goal) | Direction 2: edited model -> sibling (the transfer test) |
|---|---|---|
| Source of delta | Stronger model B's features on the prompt | An edit tuned on SAE-equipped model A |
| Receiver | Weak model A (GPT-2 small) | A sibling model B (same family) |
| Question | Can B's knowledge improve A's answers? | Does an edit tuned on A carry over to B? |
| Serves "make GPT-2 answer better"? | **Yes, directly** | Indirectly (tests whether edits are portable) |
| Needs a learned map? | Yes if widths/bases differ | No if B is a same-architecture fine-tune |

This note treats **Direction 1 as the main goal** and Direction 2 as a cheaper, cleaner side test that checks whether the shared-basis assumption holds at all. If you meant only Direction 2 (edit GPT-2, inject into a "better" model), say so and sections 5 and 6 should be reordered.

## 4. The pipeline (Direction 1)

```
                same prompt
               /           \
        Model B (strong)    Model A (GPT-2 small)
               |                  |
        layers 0..Lb         layers 0..8
               |                  |
        B's SAE features           |
        (what B represents)        |
               |                   |
   match to A's features           |
   (offline co-activation)         |
               |                   |
   decode -> map into A-space      |
               |                   |
           delta  ------------>  (+) added at A's layer 8
                                   |
                              layers 9..end   <- A's own layers read the patch
                                   |
                              A's output  <- the only output; this is what we judge
```

Steps:

1. **Offline matching (done once).** Run both models over millions of tokens of text. Record SAE feature activations. Build a co-activation table: which of B's features tend to fire on the same tokens as which of A's features. This is the "compare features that fire together" step.
2. **Mapping (done once).** Learn a map from B's residual space to A's (a linear/affine map fit on paired activations). Needed whenever widths or bases differ. This is where model stitching (Chen et al.) is the closest prior work.
3. **Online patching (per prompt).** Read B's active features, select the matched ones, decode them, map into A's space, form the delta, add it to A's residual stream at layer 8 (the delta-patching step), and let A finish.
4. **Measure.** Compare A-patched against A-clean and the controls in section 7.

## 5. Direction 2 (the cleaner side test)

Same-family pair where the shared-basis assumption is plausible, e.g. a fine-tune of GPT-2 small (same width, layers, tokenizer, initialization). Use the SAE of the original model on the original model, tune an edit with the existing gradient-descent method, compute `delta = R_edited(L) - R_clean(L)` at the same layer, add it to the sibling's own residual at layer L, and let the sibling's later layers run.

Two ways to transfer, both worth testing:

- **Transfer the delta (the vector):** copy A's delta as is.
- **Transfer the multipliers (the recipe):** apply A's per-feature multipliers `m_k` to the sibling's *own* activations encoded through A's SAE. The delta then adapts to the sibling's state.

**Gate before anything else:** measure how well A's SAE reconstructs the sibling's residual stream on the same text (variance explained). If reconstruction is poor, the bases differ and the injection results cannot be interpreted. This check is cheap and decides whether to continue.

Do **not** inject A's *final-layer* ("finished") residual into a middle layer of the other model. Inject the **same-depth delta**. Depths must match, otherwise the receiving model reads it as an earlier-stage state.

## 6. What "better output" has to mean

"The target word reaches rank 1" is **not** evidence of improvement. The project's own Entry 30 found that the additive edit pushed 20 of 20 random unrelated words to rank 1, and transferred equally to rewordings for random and true targets. A patch can raise a word without any knowledge being used. So "better" must be defined on capability, with the controls.

Primary capability test (Direction 1):

- Build a prompt set from the existing CounterFact hard set where **B answers correctly and A answers incorrectly**.
- Success = A's patched answer is correct more often, **and** the effect holds on rewordings of the prompt, **and** unrelated prompts do not change.

Report all of these, per condition:

| Measure | What it answers | Existing tool (from the handoff) |
|---|---|---|
| Target rank / probability | Reach | gradient-editing code, Prototype lab |
| Transfer to reworded prompts | Is it a fact or a word push? | `tools/test_generalisation.py` |
| Leak on neighbour + unrelated prompts (KL, top-1 flips) | Specificity | `tools/test_generalisation.py` |
| Text quality (loops, repetition, distinct-2, tail perplexity) | Fluency cost | `tools/test_generation_quality.py` |
| Context sensitivity ("Do not say X") | Does it override the prompt? | `tools/test_generation_quality.py` |
| Target rank **by layer** (logit lens) for A clean, A patched, B | Does the patch persist through A's later layers, or get undone? | new, small |

"Inheritance" must be measured **downstream**, not at the injection layer. Right after injection the added features are present by construction, which proves nothing. The informative question is whether the effect survives A's layers 9 onward and reaches the output.

## 7. Controls (all at matched norm and layer)

- **No patch** (A clean).
- **Random delta**, same norm.
- **Patch from random or unmatched B features**, same norm.
- **Rank-matched random target**, so the comparison is not circular (the project already hit this: start rank 17 vs 17,504).
- **Plain mapped-residual patch** with no SAE guidance. If this matches the SAE-guided patch, the SAE adds nothing, and that must be reported.
- **Shuffled or random map**, to show the map carries information.
- **Logit ensemble** of A and B, which reviewers will ask for. It is a baseline, not the method.
- **Prompting baseline** (state the fact in the prompt) and, per the roadmap, the in-context fact vector (roadmap item D). If the patch does not beat these, say so.
- **Dose-response:** sweep the delta scale, track side effects.
- **Layer sweep** for the injection depth. A peak at the matching layer supports a shared basis.

## 8. Models and the hardware: RTX 4050 (8 GB VRAM) + 24 GB RAM

All sizes are **est.** Run a pilot before planning around them.

| Model | Approx. weight size | Fits 8 GB VRAM? | Notes |
|---|---|---|---|
| GPT-2 small (124M) | ~0.5 GB (fp32) | Easily | Has the project's SAE (layer 8). |
| GPT-2 medium / large (355M / 774M) | ~0.7 / ~1.5 GB (fp16) | Easily | **Same tokenizer as GPT-2 small**, so no token alignment. SAE availability unknown, **verify**. |
| GPT-2 XL (1.5B) | ~3 GB (fp16) | Yes | Same tokenizer. Strongest GPT-2-family option. SAE availability unknown, **verify**. |
| Gemma-2-2B | ~5 GB (bf16) | Likely, with short contexts | Different tokenizer (256k SentencePiece), d=2304. Public SAEs (Gemma Scope), **verify** release names and layers. Running it next to GPT-2 plus SAEs leaves little headroom. |
| Gemma-2-9B | ~18 GB (bf16) | **No** | Would need 4-bit/8-bit quantization or CPU offload, or a rented GPU. |

The RTX 4050 is an Ada-generation card, so bf16 should be supported, unlike the older 1660 Ti mentioned in the project handoff. **Verify** with a small test.

Practical plan for this machine:

- **Pilot pair: GPT-2 small + GPT-2 large or XL.** Same tokenizer, fits easily, fastest to debug. A "better model" in a real sense (more parameters, same training distribution). Residual widths differ (768 vs 1280/1600), so the map is still needed, which makes it a fair test of the map.
- **Main pair: GPT-2 small + Gemma-2-2B.** Stronger gap, public SAEs, but tokenizer alignment (by character span) and tighter memory.
- Cache paired activations to disk (fp16). A few million tokens at one layer is on the order of tens of GB per model (**est.**: tokens x width x 2 bytes), which fits on disk; keep the in-RAM working set small. 24 GB of RAM is adequate if activations are streamed in chunks.
- Fitting the map is a linear regression and takes seconds to minutes. Activation collection is the expensive part.
- Nothing here needs a rented GPU until Gemma-2-9B or SAE training. Training new SAEs is out of scope: use pretrained ones.

## 9. Related work and what is actually new

Closest existing work (all from search snippets; **verify** before citing):

- **Model stitching:** Chen, Merullo, Stolfo, Pavlick, "Transferring Linear Features Across Language Models With Model Stitching" (NeurIPS 2025, arXiv 2506.06609). Affine maps between residual streams; transfers SAEs, probes and steering vectors. The reported steering-transfer results were mixed for gemma-2-2b-it to 9b-it.
- **Cross-architecture steering transfer:** arXiv 2608.05164 (2026), systematic study with learned bridges across five architectures.
- **Activation communication:** Ramesh and Li, "Communicating Activations Between Language Model Agents" (ICML 2025, arXiv 2501.14082); Bicameral Model (arXiv 2605.11167).
- **SAE feature universality and shared dictionaries:** Lan et al. (arXiv 2410.06981); SharedSAE (arXiv 2609.04344); Universal SAEs (arXiv 2502.03714).
- **SAE transfer between base and fine-tuned models:** Alignment Forum post "Do SAEs transfer across base and fine-tuned models".
- **Knowledge fusion:** FuseLLM (arXiv 2401.10491), which distills several models into one rather than patching at inference.

Honest novelty statement: the generic mechanisms (affine map between residual streams, injecting activations from one model into another, transferring steering vectors) already exist. What is specific to this project is the recipe and the measurement: SAE-feature matching to select what to patch, per-feature multiplier edits tuned by gradient descent with a side-effect term, delta patching that excludes reconstruction error, and evaluation on the specificity benchmark with rank-matched controls. Frame it as an **extension and a measurement study**, not a discovery. A literature check beyond snippets (including what cites Chen et al. and SharedSAE) is still needed before any novelty claim.

## 10. Risks and expected failure modes

- **The patch is just a push.** It may behave like the additive edit: raise a word, leak into unrelated prompts. The random-target and unmatched-feature controls are what detect this.
- **Matched features are redundant.** By definition shared features add nothing new. Any gain has to come from what B knows that A does not, and the matching step tells us where they do *not* align, which is the opposite signal. This is the main conceptual risk of the SAE-guided design, and the "plain mapped-residual patch" control is how to test it.
- **A's later layers may reject the patch.** The target-rank-by-layer plot will show a rise then a rebound if so.
- **Distribution shift.** The delta may be off-distribution for A, and a state B wrote may not be readable by A.
- **Capability gap.** A very large gap makes the map poor and the patch generic.
- **Tokenizer noise** for GPT-2 + Gemma (alignment by character span).
- **Proxy metrics.** CounterFact-style next-token accuracy is a narrow proxy for "answers better".

Negative results are reportable: if the plain mapped-residual patch matches the SAE-guided one, or the random-target control matches the true target, say that.

## 11. Suggested order of work

1. **Literature check** beyond snippets (the citation lists of Chen et al. and SharedSAE).
2. **Direction 2 gate:** reconstruction check of one model's SAE on a same-family sibling.
3. **Offline matching** for the pilot pair (GPT-2 small + GPT-2 large/XL), then the map fit and its held-out quality (R^2).
4. **Delta patch + controls** on the existing benchmark prompts; add the target-rank-by-layer plot.
5. **Move to GPT-2 + Gemma-2-2B** if the pilot is not a clear failure.
6. **Write-up** after the project's specificity benchmark (roadmap item A) exists, since this section should be judged by the same yardstick.

New code should go in new files only (for example `src/merge_patch.py` and a script under `tools/`), consistent with the project's rule that new code does not change old code.

## 12. Decisions needed

1. **Direction:** confirm Direction 1 (strong model informs GPT-2) is the main goal, with Direction 2 as a side test.
2. **Pilot pair:** GPT-2 small + GPT-2 large/XL (same tokenizer), or go straight to GPT-2 small + Gemma-2-2B.
3. **Where this goes:** a new branch, or new files on `gradient-descent-editing`. This file was written on the `ccr-56bb1a37-3pshpb` branch, which is based on `main` and does not contain the `gradient-descent-editing` code.
4. **SAE availability:** check which of GPT-2 medium/large/XL and Gemma-2-2B layers have pretrained SAEs before committing.

## 13. What this note does not claim

- It does not claim the idea is novel as a mechanism.
- It does not claim the method works. No experiment has been run.
- It does not claim a "merged model": GPT-2 plus a patch derived from a second model, with the second model running alongside, is not one model.
- It does not claim that rank-1 on a target means the model gained a fact.
