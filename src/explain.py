import os
import requests
import streamlit as st


def query_groq(prompt: str, api_key: str = None, model: str = "openai/gpt-oss-20b", temperature: float = 0.3, max_tokens: int = 200) -> str:
    """Send a prompt to Groq using the same connection pattern already used by the app."""
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key and hasattr(st, "secrets") and "groq" in st.secrets:
        try:
            api_key = st.secrets["groq"].get("api_key")
        except Exception:
            pass

    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert mechanistic interpretability researcher who explains internal neural network representation concepts in crisp, intuitive layperson terms."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # gpt-oss models spend part of max_tokens on hidden reasoning before writing the
        # final answer; without this, short max_tokens budgets can be consumed entirely by
        # reasoning and return an empty "content" field. "low" keeps reasoning minimal so the
        # token budget goes to the actual answer. Harmless (ignored) on non-gpt-oss models.
        "reasoning_effort": "low",
    }

    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=5)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        return None
    except Exception:
        return None


def generate_mechanistic_explanation(
    prompt: str,
    target: str,
    baseline_prob: float,
    baseline_rank: int,
    final_prob: float,
    final_rank: int,
    interventions: list[dict],
    api_key: str = None
) -> str:
    """
    Calls the Groq API to translate complex feature activations and interventions
    into a plain-English layperson explanation.
    """
    intervention_desc = ""
    for inv in interventions:
        action = inv.get("action", "modified")
        fid = inv.get("feature_id")
        desc = inv.get("description", "Unknown concept")
        strength = inv.get("strength", 1.0)
        intervention_desc += f"- Feature {fid} ({desc}): {action} at strength {strength:.2f}\n"

    user_prompt = f"""
Analyze this mechanistic intervention on GPT-2's internal activations.

Context:
- Prompt: "{prompt}"
- Target Token: "{target}"
- Baseline Model State (Before): Target Prob = {baseline_prob*100:.2f}%, Target Rank = {baseline_rank}
- Post-Intervention State (After): Target Prob = {final_prob*100:.2f}%, Target Rank = {final_rank}

Interventions Applied:
{intervention_desc}

Task:
Write a concise, plain-English summary (2-3 sentences max) explaining what this intervention did and why it worked or failed. Focus on translating the feature descriptions into how they influenced the model's behavior. Do not use complex neural network jargon.
"""

    res = query_groq(user_prompt, api_key=api_key, max_tokens=150)
    return res if res else "Configure your GROQ_API_KEY to enable automated AI intervention explanations."


def generate_best_result_explanation(
    prompt: str,
    target: str,
    baseline_prob: float,
    baseline_rank: int,
    best_prob: float,
    best_rank: int,
    best_mute_size: int,
    best_boost_size: int,
    reached_target: bool,
    n_combinations_run: int,
    api_key: str = None,
) -> str:
    """
    Calls Groq to explain, in plain English and grounded in this run's actual numbers, why the
    reported "best" combination is the one shown — not necessarily the last row the sweep tried —
    and what the rank/probability shift implies about how effective the intervention was.
    """
    if reached_target:
        outcome = f"the sweep succeeded: the target reached Rank #1 (the model's top prediction) after {n_combinations_run} combination(s) tried"
    else:
        outcome = f"the sweep did not reach Rank #1; it stopped after {n_combinations_run} combination(s), either exhausting its candidate pool or the configured batch sizes"

    user_prompt = f"""
A causal intervention search (muting competitor features, boosting target-aligned features) was run on GPT-2 to try to push the token "{target}" higher in the model's output ranking for the prompt: "{prompt}"

- Baseline (no intervention): target rank {baseline_rank}, probability {baseline_prob*100:.4f}%
- Best result found anywhere in the sweep: target rank {best_rank}, probability {best_prob*100:.4f}%, achieved by muting {best_mute_size} competitor feature(s) and boosting {best_boost_size} target-aligned feature(s) together in one forward pass
- Outcome: {outcome}

Task: In 2-3 plain-English sentences, explain (a) why this specific combination is reported as the "best" result rather than whichever combination happened to run last, and (b) what the rank/probability change from baseline to best tells us about how effective this intervention was for this prompt. Do not restate the raw numbers verbatim — interpret them. Avoid neural network jargon.
"""

    res = query_groq(user_prompt, api_key=api_key, max_tokens=180)
    if res:
        return res
    return (
        "This is the best combination seen across every row in the sweep — picked by lowest target rank "
        "(ties broken by higher target probability), not just whichever row ran last. The safety filter only "
        "vets individual candidate features before the sweep starts; it doesn't stop the target's rank from "
        "drifting up and down as more features get piled on within the sweep itself, so this tracks and "
        "protects the best result actually found. Configure your GROQ_API_KEY for an AI-generated explanation "
        "tailored to this specific run."
    )


def get_xai_guidance_card(section_key: str) -> dict:
    """
    Returns a standardized 6-part XAI Guidance Card dictionary for major sections in Monosemanticity Analysis.
    """
    cards = {
        "raw_neuron": {
            "trigger": "Triggered when a user initiates a monosemanticity audit to inspect baseline un-decomposed model behavior.",
            "what": "Measures the raw activation strength of a single un-decomposed MLP neuron (internal dial) across a diverse corpus.",
            "why": "Demonstrates why Sparse Autoencoders are required by proving that raw model neurons are polysemantic (doing multiple jobs at once).",
            "internal": "Directly inspects intermediate residual stream activations at `blocks.{layer}.mlp.hook_post` prior to dictionary learning.",
            "interpret": "Top-activating snippets appearing across unrelated topics indicate polysemanticity; a single clean pattern is rare.",
            "workflow": "Confirms that directly ablating raw neurons causes widespread side effects, justifying targeted SAE feature editing in downstream tabs."
        },
        "sae_feature": {
            "trigger": "Triggered automatically alongside raw neuron evaluation to benchmark SAE decomposition quality.",
            "what": "Scans the corpus for token positions that most strongly activate a specific Sparse Autoencoder (SAE) feature dial.",
            "why": "Verifies whether dictionary learning successfully isolated a clean, monosemantic concept from polysemantic raw neurons.",
            "internal": "Encodes residual activations through the SAE encoder weights ($W_{enc}$) and measures non-zero activations ($f_i > 0$).",
            "interpret": "Top-activating snippets sharing a clear semantic theme (e.g., place names or verbs) confirm a clean, monosemantic feature dial.",
            "workflow": "Identifies clean feature candidates suitable for precise ablation or boosting in the intervention pipeline."
        },
        "interpretability_score": {
            "trigger": "Triggered when autointerp scoring is enabled with an active Groq API key.",
            "what": "Evaluates how legibly an AI evaluator can predict feature firing on unseen held-out sentences after viewing reference examples.",
            "why": "Provides an objective accuracy score quantifying feature legibility and predictability.",
            "internal": "Splits the corpus into reference and held-out sets, prompts Groq with reference examples, and validates predictions against ground-truth activations.",
            "interpret": "High accuracy (80%+) confirms a predictable, monosemantic feature; low accuracy indicates erratic or messy behavior.",
            "workflow": "Filters out unreliable features, ensuring only highly interpretable dials are selected for critical model interventions."
        },
        "sparsity_stats": {
            "trigger": "Triggered during monosemanticity auditing to assess activation density across the feature dictionary.",
            "what": "Measures the $L_0$ norm (count of active features per token) and individual feature firing frequencies.",
            "why": "Ensures the SAE maintains true sparse coding rather than dense, uninterpretable activations.",
            "internal": "Counts non-zero feature activations ($f_i > 0$) per token position across a corpus sample.",
            "interpret": "Low Mean $L_0$ (~30-80 active features out of ~24,000) confirms effective sparse decomposition.",
            "workflow": "Ensures interventions touch minimal active dials, preventing cascade failures across unrelated model features."
        },
        "decoder_similarity": {
            "trigger": "Triggered during monosemanticity auditing to check geometric independence among SAE feature directions.",
            "what": "Computes cosine similarity between the current feature's decoder vector ($W_{dec}$) and all other feature vectors in the SAE.",
            "why": "Detects feature duplication or near-redundant directions in dictionary space.",
            "internal": "Calculates normalized dot products $\\frac{W_{dec,i} \\cdot W_{dec,j}}{\\|W_{dec,i}\\| \\|W_{dec,j}\\|}$ across dictionary vectors.",
            "interpret": "Low similarity (<0.5) confirms geometric independence; high similarity (>0.8) indicates duplicate or highly correlated features.",
            "workflow": "Informs joint multi-feature ablation strategies if a target concept is split across multiple redundant feature directions."
        }
    }

    return cards.get(section_key, {
        "trigger": "Triggered during analysis execution.",
        "what": "Measures model internal activation representations.",
        "why": "Helps interpret model internal state.",
        "internal": "Inspects transformer layer activations.",
        "interpret": "Compare metrics against baseline expectations.",
        "workflow": "Guides feature selection for activation interventions."
    })


def generate_polysemanticity_comparison_xai(neuron_index: int, feature_id: int, raw_tokens: list, sae_tokens: list, api_key: str = None) -> str:
    """Generates a dynamic AI synthesis comparing raw neuron polysemanticity with SAE feature dial monosemanticity."""
    raw_str = ", ".join(f"'{t}'" for t in raw_tokens[:5])
    sae_str = ", ".join(f"'{t}'" for t in sae_tokens[:5])

    prompt = f"""
Compare these two internal model activation patterns:
- Raw MLP Neuron {neuron_index} top triggering words: {raw_str}
- SAE Feature Dial {feature_id} top triggering words: {sae_str}

Task:
Write a brief, illuminating comparison (2-3 sentences max) explaining how Raw Neuron {neuron_index} exhibits polysemanticity (mixing topics) while SAE Feature Dial {feature_id} captures a far cleaner concept. Explain why this difference matters for activation editing.
"""
    ai_resp = query_groq(prompt, api_key=api_key)
    if ai_resp:
        return ai_resp

    return (
        f"**Polysemanticity Audit Verdict:** Raw Neuron {neuron_index} fires across diverse, unrelated tokens ({raw_str}), "
        f"demonstrating that single raw neurons perform multiple overlapping jobs. In contrast, SAE Feature Dial {feature_id} "
        f"activates on a far more coherent set of tokens ({sae_str}), providing the isolated control needed for precise intervention."
    )


def generate_sparsity_xai(mean_l0: float, sample_size: int, api_key: str = None) -> str:
    """Generates a contextual explanation for sparsity statistics."""
    prompt = f"""
Explain the sparsity metric Mean L0 = {mean_l0:.2f} measured over {sample_size} tokens in GPT-2's Sparse Autoencoder.
Task:
Write 2 sentences explaining why a low Mean L0 relative to ~24,576 total features proves that the SAE achieves sparse representation, and why sparsity is essential for interpretability.
"""
    ai_resp = query_groq(prompt, api_key=api_key)
    if ai_resp:
        return ai_resp

    return (
        f"**Sparsity Verdict:** With a Mean L0 of {mean_l0:.2f}, only a tiny fraction of the ~24,576 available feature dials "
        f"turn on for any given word. This extreme sparsity ensures that concepts are cleanly separated into distinct, "
        f"interpretable directions rather than blended into dense background noise."
    )


def generate_decoder_similarity_xai(feature_id: int, top_similar: list, api_key: str = None) -> str:
    """Generates a contextual explanation for decoder cosine similarity results."""
    if not top_similar:
        return "No similar decoder directions detected."

    top_id, top_sim = top_similar[0]
    prompt = f"""
Explain the decoder similarity for SAE Feature {feature_id}:
- Top neighbor: Feature {top_id} with Cosine Similarity = {top_sim:.4f}

Task:
Write 2 sentences explaining whether a cosine similarity of {top_sim:.4f} indicates geometric independence or feature duplication, and how an engineer should use this when planning an ablation intervention.
"""
    ai_resp = query_groq(prompt, api_key=api_key)
    if ai_resp:
        return ai_resp

    status = "geometrically distinct and independent" if top_sim < 0.6 else "highly redundant or duplicated"
    return (
        f"**Geometric Independence Verdict:** Feature {feature_id}'s nearest neighbor is Feature {top_id} at a similarity of {top_sim:.4f}, "
        f"indicating that this feature is {status}. "
        f"{'A single feature ablation will cleanly target this concept.' if top_sim < 0.6 else 'Consider joint multi-feature ablation to address both redundant directions.'}"
    )


def get_empty_state_guidance(state_type: str, context: dict = None) -> dict:
    """
    Returns structured empty state explanation metadata (why unavailable, expectedness, next steps).
    """
    if context is None:
        context = {}

    states = {
        "invalid_feature_id": {
            "title": "⚠️ Invalid Feature ID",
            "why": f"Feature ID `{context.get('feature_id')}` is outside the valid dictionary bounds for this SAE ({context.get('bounds_str', '0 to 24575')}).",
            "expected": "Expected whenever an out-of-bounds or non-integer ID is requested.",
            "next_steps": "Use the Feature Explorer dropdown or recommended table to select a valid Feature ID (e.g., Feature 313)."
        },
        "invalid_neuron_index": {
            "title": "⚠️ Invalid Raw Neuron Index",
            "why": f"Neuron Index `{context.get('neuron_index')}` is outside the raw MLP layer bounds ({context.get('bounds_str', '0 to 3071')}).",
            "expected": "Expected whenever an out-of-bounds raw neuron index is entered.",
            "next_steps": "Select a neuron index between 0 and 3071 for GPT-2 Small."
        },
        "no_activations": {
            "title": "ℹ️ No Activations Detected in Corpus",
            "why": f"Feature Dial `{context.get('feature_id')}` did not activate (fire > 0.0) on any token positions across the selected corpus.",
            "expected": "Normal behavior for highly specific features (or 'dead features') when tested on general sentences.",
            "next_steps": "Try selecting a broader feature (e.g. Feature 313 or 415), or paste a custom corpus with domain-specific terms matching this feature."
        },
        "missing_neuronpedia": {
            "title": "ℹ️ Neuronpedia Metadata Unavailable",
            "why": f"No online description was found on Neuronpedia for Feature `{context.get('feature_id')}` on Layer `{context.get('layer')}`.",
            "expected": "Expected if offline or if this specific feature lacks a community autointerp description on Neuronpedia.",
            "next_steps": "You can inspect the max-activating snippets directly below to infer the feature's concept manually."
        },
        "missing_decoder_similarity": {
            "title": "ℹ️ No Similar Decoder Directions Found",
            "why": f"No other feature decoder vectors ($W_{{dec}}$) met the threshold for similarity comparison.",
            "expected": "Normal for highly unique or orthogonal feature directions.",
            "next_steps": "This confirms the feature is geometrically distinct. Single-feature ablation is recommended."
        }
    }

    return states.get(state_type, {
        "title": "ℹ️ Information Unavailable",
        "why": "The requested analysis could not be computed for the current selection.",
        "expected": "May occur if data or metadata is partially missing.",
        "next_steps": "Select a recommended feature or check input parameters."
    })
