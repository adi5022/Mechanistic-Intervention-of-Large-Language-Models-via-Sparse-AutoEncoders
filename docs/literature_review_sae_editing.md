# Literature check: is SAE-based factual editing an established area, what is novel here, and should the SAE approach worry us?

Date: 2026-10-01. Written because a potential employer questioned whether the SAE method is sound. This note records what was found, how well each source was checked, and what follows for the project. It is a search-based sweep, **not an exhaustive review**: "I found no exact match" means exactly that and is weaker than "none exists".

## How well each source was checked

| Level | Meaning |
|---|---|
| **Fetched** | The page or abstract was opened and read through a summarising fetch |
| **Snippet** | Only seen in search-result summaries; the paper itself was not opened |

No full paper PDF was read for any source below. Quotes of numbers come from the fetched pages or snippets named in the table.

## 1. The sources

| Topic | Source | Level | What it says (as far as checked) |
|---|---|---|---|
| Google DeepMind pivot | [Negative Results for SAEs on Downstream Tasks and Deprioritising SAE Research](https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks) (also on [Medium](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9), not fetchable) | Fetched (LessWrong version) | Task: detecting harmful intent in user prompts, tested on new jailbreaks. Dense linear probes reached AUROC about 0.999 in and out of distribution; single-feature SAE probes did poorly; k-sparse SAE probes (k about 20) closed some of the gap but stayed "distinctly worse" out of distribution; fine-tuning the SAE closed about half the gap. The team says SAEs are not useless and not a game-changer, and is de-prioritising SAE work in favour of model diffing, model organisms of deception and interpreting thinking models. It cites parallel negative results on probing, unlearning and steering. |
| Steering benchmark | [AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders](https://arxiv.org/abs/2501.17148), code: [stanfordnlp/axbench](https://github.com/stanfordnlp/axbench) | Snippet | On steering: prompting about 0.894 overall, LoReFT 0.741, SFT 0.676, LoRA 0.615, DiffMean 0.239, **SAE 0.165**. On concept detection, DiffMean, linear probes and ReFT-r1 reached AUROC about 0.94; SAEs lagged. Only ReFT-r1 was competitive with finetuning and prompting among representation methods. Models were larger instruction-tuned ones, tasks were concept steering. |
| Output vs input features | [SAEs Are Good for Steering, If You Select the Right Features](https://arxiv.org/abs/2505.20063) (Arad, Mueller, Belinkov; EMNLP 2025) | Snippet | Distinguishes "input features" (patterns in the input activations) from "output features" (causal effect on output tokens); they rarely coincide. Filtering to high output score gives a 2 to 3 times steering improvement, comparable with supervised methods. This is close to what the project's causal selector does (rank by effect of removal, not by activation size). |
| Same SAE family, facts | [Evaluating Open-Source Sparse Autoencoders on Disentangling Factual Knowledge in GPT-2 Small](https://arxiv.org/abs/2409.04478) | Fetched | Four open SAEs on GPT-2 small, RAVEL benchmark (city to country versus continent), learned binary mask of features to patch. "SAEs struggle to reach the neuron baseline, and none come close to the DAS skyline." |
| Recovery of suppressed behaviour | [SAE Interventions are Unreliable: Post-Intervention Recovery of Suppressed Behavior](https://arxiv.org/abs/2606.18322) | Fetched | Clamping harmful SAE features can suppress visible behaviour without eliminating it; optimising a residual perturbation can restore it while the clamped features stay clamped (95.8% recovery on valid refusal samples, reported). Recovery is attributed to the SAE reconstruction residual, the part the SAE does not explain. Relevant to worry 1 and 3. |
| SAE unlearning | [Applying sparse autoencoders to unlearn knowledge in language models](https://arxiv.org/html/2410.19278) | Snippet | SAE feature ablation can unlearn several topics but with similar or larger side effects than RMU; a conditional clamping follow-up ([2503.11127](https://arxiv.org/html/2503.11127)) is reported to match RMU on forgetting and retention. |
| SAE-targeted steering | [Improving Steering Vectors by Targeting SAE Features](https://arxiv.org/abs/2411.02193) | Snippet | Builds steering vectors that target chosen SAE features while limiting side effects; better than other methods on 7 of 9 tasks across scales (as summarised). |
| Knowledge-selection steering | [SpARE: Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering](https://arxiv.org/abs/2410.15999) | Snippet | Training-free edits of a small set of SAE features at inference to steer whether the model uses context or its own memory. |
| Weight editing | [ROME: Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262), MEMIT | Snippet | Causal tracing locates mid-layer MLP computation; a rank-one weight change writes a new key-value fact. Source of CounterFact and `known_1000`. |
| Localisation vs editing | [Does Localization Inform Editing?](https://arxiv.org/abs/2301.04213) (Hase et al.) | Snippet | Where causal tracing says a fact lives has almost no relation to which layer edits it best (choice of layer explains over 94% of ROME's success variance in their regression). Relevant to the layer-8 choice. |
| In-context editing | [Can We Edit Factual Knowledge by In-Context Learning? (IKE)](https://aclanthology.org/2023.emnlp-main.296/) | Snippet | Putting the corrected fact and demonstrations in the prompt edits knowledge without parameter updates; works on base models; implemented in EasyEdit. |
| Toolkit | [EasyEdit](https://arxiv.org/pdf/2308.07269) | Snippet | Implements ROME, MEMIT, IKE and others; GPT-2 listed as supported for ROME and MEMIT. |
| Datasets | [CounterFact and known_1000](https://rome.baulab.info/data/dsets/); [LAMA / T-REx](https://aclanthology.org/D19-1250.pdf); [ParaRel](https://aclanthology.org/2021.tacl-1.60.pdf) | Snippet (and the CounterFact file itself was downloaded and inspected) | CounterFact: 21,919 records. T-REx: 34,017 instances, 41 relations. ParaRel: about 199,000 paraphrased cloze questions over 23,097 facts. |

## 2. Is this an established field?
Yes. Editing facts, steering with SAE features, ablating SAE features to unlearn, and steering knowledge selection with SAEs all have prior work. The project did not copy these (the code was built independently, see the journals), but it sits inside a populated area and should say so.

## 3. Which ideas in the codebase are novel?
| Idea in the project | Status |
|---|---|
| Ablate or scale SAE features to change a prediction | Established (SAE-TS, SpARE, SAE unlearning, sparse feature circuits) |
| Choose features by causal effect of removal, not activation size | Established in spirit (Arad et al. "output features") |
| Apply only the SAE delta so reconstruction error is not injected | Standard practice; described in the SAE-intervention literature |
| Correct a factual prediction by editing at one layer | Established (ROME, MEMIT), by different means |
| **Joint mute-and-boost on one fact, with a per-candidate strict safety filter, cumulative sweep and pool refill, for pushing a single fact to rank 1 on GPT-2 small** | **Possibly novel as a combination and use case.** No exact match found; search not exhaustive |
| Dynamic (tolerance / graded) safety filtering, studied as arms | Not found elsewhere; a small, honest-null-result study (relaxing the strict filter added at most one prompt) |
| Learned, prompt-adaptive mute and boost strengths (planned) | No exact match found; planned, not built |
The project's own paper draft already avoids novelty claims, which is the right stance.

## 4. What did beat SAEs, and how do those methods work?
- **Linear probes** (detection): train a simple classifier (logistic regression) on the model's activations. Dense and supervised, so it uses every direction in the activation, not a sparse dictionary.
- **Prompting** (steering): tell the model what to do in words. Strong on large instruction-tuned models; GPT-2 small is not instruction-tuned, so the usable form is in-context editing (IKE): show the fact and a few demonstrations.
- **DiffMean** (steering and detection): average the activations on examples with the concept minus the average without; add that direction. Needs examples of the concept.
- **ReFT-r1** (steering and detection): learn a single direction with supervision and a language-modelling loss with a sparsity term; competitive with fine-tuning in AxBench.
- **Weight editing (ROME / MEMIT)**: change a layer's weights so a specific fact is stored differently; permanent, unlike the project's transient edits.
How they compare with this project: all of them use supervision or examples of the target concept, or change weights, or rely on instruction following. The project's method is training-free, transient, and works on one prompt at a time through interpretable named features. That is a different trade-off, not a reason for the SAE method to be wrong, but it has to be shown against them.

## 5. Is the project's use case different enough not to worry?
Partly. The DeepMind result is about classification on new data; AxBench is concept steering on larger instruction-tuned models; neither tests "make one specific true next-token win on a small base model". But Entry 22 and the headroom study point the same way as the literature: the SAE edit is not obviously the most efficient way to move the output (free residual pushes of the same size reach more prompts, with larger side effects; no equal-damage comparison yet). The sensible conclusion is not to stop but to **measure** against the alternatives on the same prompts.

## 6. What follows for the project
1. **Compare fairly** on the same CounterFact prompts, timed on one machine: the user's system, IKE, ROME, DiffMean (existing implementations only: EasyEdit, AxBench). Planned as EXP-015.
2. **Test more than rank 1 on the same sentence.** CounterFact supplies paraphrase and neighbourhood prompts for each record to check generalisation and spread.
3. **Report honestly,** including null results. The learned-strength idea (EXP-013) is documented as something tried whether or not it works.
4. **Use the literature** in the paper's related work and its evidence audit (not yet done): every claim above would need the full source read before it is cited.

## 6b. Open questions this note cannot answer
- Whether the combination in the project's pipeline really has no prior match (search limited to what a web search returned).
- How the methods above would score on GPT-2 small with CounterFact prompts (no one has run them here).
- Whether the newer SAEs (larger, better trained than the 2024 `gpt2-small-res-jb` release used here) would change the picture. The release used is old by 2026 standards.
