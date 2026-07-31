import os
import requests
import streamlit as st


def query_groq(prompt: str, api_key: str = None, model: str = "llama-3.1-8b-instant", temperature: float = 0.3, max_tokens: int = 200) -> str:
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
        "max_tokens": max_tokens
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
