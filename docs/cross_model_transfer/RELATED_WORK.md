# Related work: cross-model transfer (note written 2026-10-10)

Status of this note: written from a web-search summary and a summary of the arXiv HTML page of the paper below, produced by a tool, **not from reading the full paper**. Every number and claim attributed to the paper must be checked against the paper itself before it is cited in anything we publish. Further searches (knowledge editing across models, other work on transferring steering vectors) have not been done yet.

## The closest paper
Chen, Merullo, Stolfo, Pavlick. *Transferring Linear Features Across Language Models With Model Stitching.* NeurIPS 2025. arXiv 2506.06609 (https://arxiv.org/abs/2506.06609).

What it does (as summarised):
- Fits **affine maps between the residual streams of two models** at chosen layers (forward and back, with an inversion penalty), by mean squared error on general internet text (OpenWebText, 180k samples, context 512, Adam, 2 epochs). Layers are chosen with SVCCA.
- Model pairs: Pythia 70m to 160m, **GPT-2 small (layer 6) to GPT-2 medium (layer 10)**, Gemma-2 2b to 9b. All main experiments go small to large, inside one model family.
- What is transferred: **SAE weights** (a small model's SAE becomes a zero-shot SAE on the larger model, or an initialisation that saves 30 to 50% of training compute), **probes**, and **steering vectors** (response-language steering from gemma-2-2b to gemma-2-9b; the transfer gap is bimodal, near 0 or near 1, correlated with how frequent the language is in training data; instruction-following steering transfers worse).
- No experiments on factual editing, counterfactual facts, ROME-style edits, or gradient-descent edits (as summarised).
- Limitations they state: stitches trained on general text only, same-family models only; downstream quality is bottlenecked by the weaker model.

## How this study overlaps with it
- Same basic tool: a linear (affine) map between two models' residual streams, fitted on ordinary text, including the GPT-2 small and medium pair. Our steps A1 (translators), A2 (stitching recovery of 91 to 98%) and B1 (the country-swap control) are in this family, and we should not present the map itself as new.
- Same kind of claim as their steering-vector transfer: a vector found in one model is carried to another through the map and changes the other model's behaviour.
- Their finding that transfer works for some cases and poorly for others fits our graded, partial export result (D3, D5).

## How this study differs (goal and direction)
| | Chen et al. | This study |
|---|---|---|
| What is carried | features that already exist as directions: SAE weights, probes, steering vectors | the **change an edit made to the source model's state**: a counterfactual fact edit (gradient descent on SAE feature multipliers at layer 8 of GPT-2 small), recomputed for any prompt |
| Purpose | cheaper SAE training, comparing representations, reusing probes and steering vectors | a transient change in a second model with no weight change in either model; can the effect of an edit move across; where does it fail |
| Direction | small to large | **Export** (small edit to medium) and **Import** (medium state into small, planned); medium to small is the better-fitting direction |
| Judged on | probe and steering performance, SAE reconstruction | target rank and top-1, rewordings, neighbouring facts, unrelated prompts, against controls (random vector, wrong record's edit, rank-matched random word, word push, fact in the prompt) |
| Result so far | transfer works, with a bimodal gap | edits tuned only in the source barely reach rank 1 in the receiver (5 to 8%) though they move the rank and rewordings above controls; an edit tuned through the map against the receiver reaches 95% (D6, a ceiling); a map trained on edits is the running test (D7) |

Possible contributions, to be claimed only after a fuller literature check: (1) a controlled test of whether an SAE-multiplier fact edit transfers through a residual-stream map, with the controls above; (2) the finding that an edit tuned for the source does not carry well, while the channel itself can carry a rank-1 edit when tuned through it; (3) training the map on edits with an output-matching loss (D7), if it works; (4) the Import direction, if it is run.

## To do before any publication claim
- Read the paper in full and correct this note.
- Search for work on transferring edits or steering vectors across models, cross-model knowledge editing, and SAE transfer between families (the earlier cloud-session note `docs/cross_model_patching_idea.md` has its own list; compare).
- Decide which of our maps to describe as ours: our map is fitted by closed-form ridge regression with held-out ridge strength; theirs by Adam with an inversion penalty. Neither is a new idea.
