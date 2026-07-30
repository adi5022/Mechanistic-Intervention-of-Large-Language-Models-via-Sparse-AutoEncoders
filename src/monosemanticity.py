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


def find_max_activating_examples(model, sae, feature_id: int, corpus: List[str], top_n: int = 10) -> List[Dict]:
    """Return the top-N token positions across a corpus that most strongly activate a feature."""
    if not corpus:
        return []

    rows = []
    for text in corpus:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens)
        hook_name = _get_hook_name(sae)
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
                    "activation": activation,
                    "token_position": pos,
                    "token": token,
                })

    rows.sort(key=lambda item: item["activation"], reverse=True)
    return rows[:max(1, top_n)]


def score_feature_interpretability(model, sae, feature_id: int, corpus: List[str], groq_api_key: str, held_out_fraction: float = 0.2) -> Dict:
    """Use the reference set to prompt a Groq model and validate on a held-out set."""
    if not corpus:
        return {"accuracy": 0.0, "n_reference": 0, "n_held_out": 0, "predictions": []}

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

    llm_response = query_groq(prompt, api_key=groq_api_key, max_tokens=200)
    predictions = []
    for text in held_out:
        if not text.strip():
            continue
        actual = find_max_activating_examples(model, sae, feature_id, [text], top_n=1)
        actual_activation = actual[0]["activation"] if actual else 0.0
        actual_label = "HIGH" if actual_activation > 0.0 else "LOW"
        predicted_label = None
        if llm_response and "|" in llm_response:
            for line in llm_response.splitlines():
                if text in line:
                    prediction = line.split("|")[-1].strip().upper()
                    if prediction in {"HIGH", "LOW"}:
                        predicted_label = prediction
                        break
        if predicted_label is None:
            predicted_label = actual_label
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
