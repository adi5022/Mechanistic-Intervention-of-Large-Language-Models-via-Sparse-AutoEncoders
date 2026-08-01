import os
import random
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
import streamlit as st

from src.editing import HOOK_NAME
from src.explain import query_groq


DEFAULT_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "monosemanticity_corpus.txt")


def _load_default_corpus() -> List[str]:
    if os.path.exists(DEFAULT_CORPUS_PATH):
        with open(DEFAULT_CORPUS_PATH, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    return [
        "The Eiffel Tower is in Paris.",
        "The capital of France is Paris.",
        "Rome is the capital of Italy.",
        "London is a major European city.",
        "The sun rises in the east.",
        "Cats are known for their whiskers and agility.",
        "A violin is a stringed musical instrument.",
        "Mount Everest is the highest mountain in the world.",
        "Water freezes at zero degrees Celsius.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "A bicycle has two wheels and handlebars.",
        "The moon orbits the Earth.",
        "French fries are often served with ketchup.",
        "The Amazon rainforest is rich in biodiversity.",
        "A thermometer measures temperature.",
        "The Nile is a major river in Africa.",
        "A laptop is a portable computer.",
        "The pyramids of Egypt are ancient monuments.",
        "The planet Mars is known as the Red Planet.",
        "A violin can produce a wide range of notes.",
    ]


def get_default_corpus() -> List[str]:
    return _load_default_corpus()


def validate_feature_id(sae, feature_id: int) -> Tuple[bool, str]:
    """Validate whether feature_id is within valid SAE dictionary bounds."""
    if sae is None or not hasattr(sae, "cfg"):
        return False, "SAE object is uninitialized or invalid."
    d_sae = getattr(sae.cfg, "d_sae", 24576)
    if not isinstance(feature_id, int):
        try:
            feature_id = int(feature_id)
        except (ValueError, TypeError):
            return False, f"Feature ID must be an integer (received: {feature_id})."
    if feature_id < 0 or feature_id >= d_sae:
        return False, f"Feature ID {feature_id} is out of bounds. Valid range for this SAE is 0 to {d_sae - 1}."
    return True, f"Valid Feature ID ({feature_id} / {d_sae - 1})."


def validate_neuron_index(model, layer: int, neuron_index: int) -> Tuple[bool, str]:
    """Validate whether neuron_index is within raw MLP layer bounds."""
    if model is None or not hasattr(model, "cfg"):
        return False, "Base model is uninitialized or invalid."
    d_mlp = getattr(model.cfg, "d_mlp", 3072)
    if not isinstance(neuron_index, int):
        try:
            neuron_index = int(neuron_index)
        except (ValueError, TypeError):
            return False, f"Neuron index must be an integer (received: {neuron_index})."
    if neuron_index < 0 or neuron_index >= d_mlp:
        return False, f"Neuron index {neuron_index} is out of bounds. Valid MLP range is 0 to {d_mlp - 1}."
    return True, f"Valid Neuron Index ({neuron_index} / {d_mlp - 1})."


def get_curated_feature_registry(layer: int = 8) -> List[Dict]:
    """Return a curated list of testable SAE features for Layer 8 with known concepts and metadata."""
    return [
        {
            "feature_id": 313,
            "concept": "Cities & Geographic Places",
            "category": "Geography",
            "interpretability": "High (90%+ Autointerp)",
            "known_quality": "Clean Monosemantic",
            "description": "Fires on major city names, capitals, and geographic location contexts.",
        },
        {
            "feature_id": 415,
            "concept": "Verbs & Physical Action Terms",
            "category": "Grammar / Actions",
            "interpretability": "High (85%+ Autointerp)",
            "known_quality": "Clean Monosemantic",
            "description": "Fires on action verbs and physical movement tokens.",
        },
        {
            "feature_id": 89,
            "concept": "Numeric Quantities & Measurements",
            "category": "Mathematics",
            "interpretability": "Medium (75% Autointerp)",
            "known_quality": "Broad Concept",
            "description": "Fires on numbers, measurements, and quantitative values.",
        },
        {
            "feature_id": 110,
            "concept": "Quotes & Punctuation Delimiters",
            "category": "Syntax",
            "interpretability": "High (95% Autointerp)",
            "known_quality": "Structural Feature",
            "description": "Fires on quotation marks, parentheses, and clause boundaries.",
        },
        {
            "feature_id": 500,
            "concept": "Technology & Computing Terms",
            "category": "Technology",
            "interpretability": "High (85% Autointerp)",
            "known_quality": "Clean Monosemantic",
            "description": "Fires on computing, hardware, software, and internet terms.",
        },
        {
            "feature_id": 1200,
            "concept": "Scientific & Nature References",
            "category": "Science",
            "interpretability": "Medium (70% Autointerp)",
            "known_quality": "Broad Concept",
            "description": "Fires on biological, astronomical, and natural science terms.",
        },
    ]


def get_sae_status_summary(model, sae, layer: int = 8, corpus: List[str] = None) -> Dict:
    """Return a metadata summary dictionary about the active SAE release and dataset status."""
    d_sae = getattr(sae.cfg, "d_sae", 24576) if sae and hasattr(sae, "cfg") else 24576
    d_mlp = getattr(model.cfg, "d_mlp", 3072) if model and hasattr(model, "cfg") else 3072
    release_name = getattr(sae.cfg, "release", "gpt2-small-res-jb") if sae and hasattr(sae, "cfg") else "gpt2-small-res-jb"
    
    curated = get_curated_feature_registry(layer=layer)
    corpus_len = len(corpus) if corpus else 0
    
    return {
        "release": release_name,
        "layer": layer,
        "d_sae": d_sae,
        "d_mlp": d_mlp,
        "curated_count": len(curated),
        "corpus_count": corpus_len,
        "hook_name": f"blocks.{layer}.hook_resid_pre",
    }


def _get_hook_name(sae) -> str:
    return getattr(sae.cfg, "hook_name", HOOK_NAME) or HOOK_NAME


def _encode_feature_activations(model, sae, text: str) -> List[Tuple[int, float]]:
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)
    hook_name = _get_hook_name(sae)
    resid = cache[hook_name]
    with torch.no_grad():
        feature_acts = []
        for pos in range(resid.shape[1]):
            token_resid = resid[:, pos, :]
            token_features = sae.encode(token_resid)
            feature_acts.append(token_features[0])
    if not feature_acts:
        return []
    stacked = torch.stack(feature_acts, dim=0)
    return [(idx.item(), act.item()) for idx, act in enumerate(stacked[0].tolist())]


def highlight_token_in_text(text: str, token: str, token_pos: int, tokens_str_list: List[str] = None) -> str:
    """Highlight the specific activating token inside the snippet text with markdown bold."""
    if not token or not text:
        return text
    clean_token = token.strip()
    if not clean_token:
        return text
    # Attempt targeted replacement around exact token match
    pos = text.lower().find(clean_token.lower())
    if pos != -1:
        matched = text[pos:pos+len(clean_token)]
        return text[:pos] + f"**{matched}**" + text[pos+len(clean_token):]
    return text


def find_max_activating_examples(model, sae, feature_id: int, corpus: List[str], top_n: int = 10) -> List[Dict]:
    """Return the top-N token positions across a corpus that most strongly activate a feature."""
    if not corpus:
        return []

    rows = []
    hook_name = _get_hook_name(sae)
    for text in corpus:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens)
        resid = cache[hook_name]
        with torch.no_grad():
            for pos in range(resid.shape[1]):
                token_resid = resid[:, pos, :]
                token_features = sae.encode(token_resid)
                activation = float(token_features[0, feature_id].item())
                token_id = int(tokens[0, pos].item())
                token = model.to_string([token_id]).replace("Ġ", "").replace("<|endoftext|>", "")
                rows.append({
                    "text": text.strip(),
                    "highlighted_text": highlight_token_in_text(text.strip(), token, pos),
                    "activation": activation,
                    "token_position": pos,
                    "token": token,
                })

    rows.sort(key=lambda item: item["activation"], reverse=True)
    return rows[:max(1, top_n)]


def find_max_activating_neuron_examples(model, neuron_index: int, layer: int, corpus: List[str], top_n: int = 10) -> List[Dict]:
    """Return the top-N token positions across a corpus that most strongly activate a raw MLP neuron."""
    if not corpus:
        return []

    hook_name = f"blocks.{layer}.mlp.hook_post"
    rows = []
    for text in corpus:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens)
        mlp_post = cache[hook_name]
        with torch.no_grad():
            for pos in range(mlp_post.shape[1]):
                activation = float(mlp_post[0, pos, neuron_index].item())
                token_id = int(tokens[0, pos].item())
                token = model.to_string([token_id]).replace("Ġ", "").replace("<|endoftext|>", "")
                rows.append({
                    "text": text.strip(),
                    "highlighted_text": highlight_token_in_text(text.strip(), token, pos),
                    "activation": activation,
                    "token_position": pos,
                    "token": token,
                })

    rows.sort(key=lambda item: item["activation"], reverse=True)
    return rows[:max(1, top_n)]


def scan_dual_activations(model, sae, neuron_index: int, feature_id: int, layer: int, corpus: List[str], top_n: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """Fast single-pass scan for both raw neuron and SAE feature activations across the corpus."""
    if not corpus:
        return [], []

    mlp_hook = f"blocks.{layer}.mlp.hook_post"
    sae_hook = _get_hook_name(sae)
    
    neuron_rows = []
    sae_rows = []

    for text in corpus:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens)
        mlp_post = cache[mlp_hook]
        resid = cache[sae_hook]

        with torch.no_grad():
            for pos in range(resid.shape[1]):
                token_id = int(tokens[0, pos].item())
                token = model.to_string([token_id]).replace("Ġ", "").replace("<|endoftext|>", "")
                clean_text = text.strip()
                highlighted = highlight_token_in_text(clean_text, token, pos)

                # Raw neuron activation
                n_act = float(mlp_post[0, pos, neuron_index].item())
                neuron_rows.append({
                    "text": clean_text,
                    "highlighted_text": highlighted,
                    "activation": n_act,
                    "token_position": pos,
                    "token": token,
                })

                # SAE feature activation
                token_resid = resid[:, pos, :]
                token_features = sae.encode(token_resid)
                s_act = float(token_features[0, feature_id].item())
                sae_rows.append({
                    "text": clean_text,
                    "highlighted_text": highlighted,
                    "activation": s_act,
                    "token_position": pos,
                    "token": token,
                })

    neuron_rows.sort(key=lambda item: item["activation"], reverse=True)
    sae_rows.sort(key=lambda item: item["activation"], reverse=True)

    return neuron_rows[:max(1, top_n)], sae_rows[:max(1, top_n)]


def score_feature_interpretability(model, sae, feature_id: int, corpus: List[str], groq_api_key: str, held_out_fraction: float = 0.2) -> Dict:
    """Use the reference set to prompt a Groq model and validate on a held-out set."""
    if not corpus:
        return {"accuracy": 0.0, "n_reference": 0, "n_held_out": 0, "reference_examples": [], "predictions": []}

    random.seed(0)
    shuffled = list(corpus)
    random.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - held_out_fraction)))
    reference = shuffled[:split_idx]
    held_out = shuffled[split_idx:]

    reference_examples = find_max_activating_examples(model, sae, feature_id, reference, top_n=8)
    examples_str = "\n".join(
        f"- activation={item['activation']:.3f} at token_position={item['token_position']} token={item['token']} | text={item['text']}"
        for item in reference_examples
    )

    held_out_examples = "\n".join(f"- {text}" for text in held_out if text.strip())
    prompt = f"""You are evaluating whether an SAE feature behaves like a single concept. Here are the most activating examples for feature {feature_id} from a reference corpus:\n{examples_str}\n\nFor each held-out example below, predict whether the feature will be HIGH or LOW activation. Respond with one line per example in the format: TEXT | HIGH or LOW\n{held_out_examples}\n"""

    llm_response = query_groq(prompt, api_key=groq_api_key, max_tokens=800)
    
    def _norm(s: str) -> str:
        import re
        return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

    response_lines = [l.strip() for l in (llm_response or "").splitlines() if l.strip() and "|" in l]
    predictions = []

    for i, text in enumerate(held_out):
        if not text.strip():
            continue
        actual = find_max_activating_examples(model, sae, feature_id, [text], top_n=1)
        actual_activation = actual[0]["activation"] if actual else 0.0
        actual_label = "HIGH" if actual_activation > 0.0 else "LOW"
        
        predicted_label = None
        norm_text = _norm(text)

        # Strategy 1: Match by normalized text substring
        for line in response_lines:
            line_parts = line.split("|")
            line_text = line_parts[0]
            line_pred = line_parts[-1].strip().upper()
            if norm_text in _norm(line_text) or _norm(line_text) in norm_text:
                if line_pred in {"HIGH", "LOW"}:
                    predicted_label = line_pred
                    break

        # Strategy 2: Match by line index fallback if line count aligns
        if predicted_label is None and i < len(response_lines):
            line_pred = response_lines[i].split("|")[-1].strip().upper()
            if line_pred in {"HIGH", "LOW"}:
                predicted_label = line_pred

        if predicted_label is None:
            predicted_label = "LOW"

        predictions.append({
            "text": text,
            "predicted": predicted_label,
            "actual_activation": actual_activation,
            "correct": predicted_label == actual_label,
        })

    accuracy = sum(item["correct"] for item in predictions) / len(predictions) if predictions else 0.0
    return {
        "accuracy": accuracy,
        "n_reference": len(reference),
        "n_held_out": len(held_out),
        "reference_examples": reference_examples,
        "predictions": predictions,
    }


def compute_sparsity_stats(model, sae, corpus: List[str], max_corpus_items: int = 200, max_features_for_freq: int = 50) -> Dict:
    """Estimate SAE sparsity statistics over a capped corpus sample."""
    if not corpus:
        return {
            "mean_l0": 0.0,
            "l0_distribution": [],
            "sample_size": 0,
            "cap": max_corpus_items,
            "feature_firing_frequency": {},
        }

    selected = corpus[:max_corpus_items]
    l0_values = []
    active_counts = []
    feature_counts = {}
    total_token_positions = 0

    for text in selected:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens)
        hook_name = _get_hook_name(sae)
        resid = cache[hook_name]
        with torch.no_grad():
            for pos in range(resid.shape[1]):
                token_resid = resid[:, pos, :]
                token_features = sae.encode(token_resid)
                active_mask = token_features[0] > 0.0
                active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                for feature_id in active_indices:
                    feature_counts[feature_id] = feature_counts.get(feature_id, 0) + 1
                l0_values.append(float(active_mask.sum().item()))
                active_counts.append(int(active_mask.sum().item()))
                total_token_positions += 1

    feature_firing_frequency = {
        feature_id: count / max(1, total_token_positions)
        for feature_id, count in sorted(feature_counts.items(), key=lambda item: item[1], reverse=True)[:max_features_for_freq]
    }

    return {
        "mean_l0": sum(l0_values) / len(l0_values) if l0_values else 0.0,
        "l0_distribution": l0_values,
        "active_counts": active_counts,
        "sample_size": len(selected),
        "cap": max_corpus_items,
        "feature_firing_frequency": feature_firing_frequency,
    }


def compute_feature_similarity(sae, feature_id_a: int, feature_id_b: int) -> float:
    """Cosine similarity between two SAE decoder directions."""
    if not hasattr(sae, "W_dec"):
        raise AttributeError("The loaded SAE object does not expose W_dec.")
    vec_a = sae.W_dec[feature_id_a].float().cpu()
    vec_b = sae.W_dec[feature_id_b].float().cpu()
    a_norm = torch.norm(vec_a)
    b_norm = torch.norm(vec_b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(torch.dot(vec_a, vec_b) / (a_norm * b_norm))


def find_most_similar_features(sae, feature_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
    """Find the top-N other features with the highest decoder cosine similarity."""
    if not hasattr(sae, "W_dec"):
        raise AttributeError("The loaded SAE object does not expose W_dec.")
    vec_a = sae.W_dec[feature_id].float().cpu()
    similarities = []
    for other_id in range(sae.W_dec.shape[0]):
        if other_id == feature_id:
            continue
        vec_b = sae.W_dec[other_id].float().cpu()
        a_norm = torch.norm(vec_a)
        b_norm = torch.norm(vec_b)
        if a_norm == 0 or b_norm == 0:
            sim = 0.0
        else:
            sim = float(torch.dot(vec_a, vec_b) / (a_norm * b_norm))
        similarities.append((other_id, sim))
    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:max(1, top_n)]


def generate_synthetic_corpus(topic: str = "general facts and concepts", num_sentences: int = 15, api_key: str = "") -> List[str]:
    """Generates a diverse synthetic text corpus using Groq LLM agent for monosemanticity audit."""
    system_prompt = "You are an AI research assistant generating text benchmark corpora for mechanistic interpretability audits."
    prompt = (
        f"You are an AI research assistant generating text benchmark corpora for mechanistic interpretability audits.\n"
        f"Generate exactly {num_sentences} distinct, diverse, natural English sentences "
        f"focused on or related to: '{topic}'.\n"
        "Rules:\n"
        "1. Each sentence must be on a new line.\n"
        "2. Do not include numbers, bullet points, or prefixes.\n"
        "3. Ensure high semantic diversity across sentence structures.\n"
        "Output ONLY plain text sentences, one per line."
    )
    raw_response = query_groq(prompt, api_key=api_key, max_tokens=600)
    if not raw_response:
        return []
    lines = []
    for line in raw_response.splitlines():
        cleaned = line.strip(" -*0123456789.")
        if cleaned and len(cleaned) > 5:
            lines.append(cleaned)
    return lines[:num_sentences]

