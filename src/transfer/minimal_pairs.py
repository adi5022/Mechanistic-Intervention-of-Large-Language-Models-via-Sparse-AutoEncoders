"""
Minimal pairs: two sentences that differ in exactly one single-token word ("The capital of France is" / "The capital of Germany is").
Used to ask whether a translator carries DIFFERENCES between sentences, which is all that an edit transfer ever sends.

Every sentence continues past the word, so the LAST position is a state that has had to absorb it (not the word itself).
Each template has a list of entities; for every template only entities that are a single token in the slot are kept, so every sentence of a
template has the same length and the sentences of a pair differ at exactly one position.
"""
import random

COUNTRIES = ["France", "Germany", "Italy", "Spain", "Japan", "China", "Russia", "Egypt", "Canada", "India", "Poland", "Greece", "Turkey", "Ireland", "Brazil", "Mexico"]
CITIES = ["Paris", "London", "Berlin", "Rome", "Madrid", "Tokyo", "Moscow", "Cairo", "Dublin", "Athens", "Vienna", "Lisbon"]
COLORS = ["red", "blue", "green", "black", "white", "yellow", "orange", "purple"]
LANGUAGES = ["English", "French", "German", "Spanish", "Italian", "Chinese", "Japanese", "Russian"]
ANIMALS = ["dog", "cat", "horse", "lion", "tiger", "bear", "wolf", "rabbit", "monkey"]

TEMPLATES = [
    ("capital_of", "The capital of {X} is", COUNTRIES),
    ("city", "We flew to {X} last week for", CITIES),
    ("president", "The president of {X} said that", COUNTRIES),
    ("sky", "The sky is {X} and the", COLORS),
    ("speaks", "She speaks fluent {X} and", LANGUAGES),
    ("animal", "My favourite animal is the {X} because it", ANIMALS),
]


def build_groups(model, per_template: int = 35, seed: int = 0) -> list[dict]:
    """One group per template: entities (single token each), tokens [n_entities, L] (start token included), the one position where the
    entities differ (`diff_pos`), the last position, and a seeded sample of ordered pairs (i, j), i != j."""
    tok = model.tokenizer
    rng = random.Random(seed)
    groups = []
    for name, template, entities in TEMPLATES:
        at_start = template.startswith("{X}")
        # this tokenizer adds the start token by itself, so switch that off when counting the tokens of the word alone
        keep = [e for e in entities if len(tok(e if at_start else " " + e, add_special_tokens=False)["input_ids"]) == 1]
        prompts = [template.replace("{X}", e) for e in keep]
        tokens = model.to_tokens(prompts)                                  # [n, L], start token first
        differing = [c for c in range(tokens.shape[1]) if len(set(tokens[:, c].tolist())) > 1]
        if len(keep) < 4 or len(differing) != 1:
            continue                                                       # not a clean minimal-pair family; skip it rather than guess
        pairs = [(i, j) for i in range(len(keep)) for j in range(len(keep)) if i != j]
        rng.shuffle(pairs)
        groups.append({"name": name, "template": template, "entities": keep, "tokens": tokens, "diff_pos": differing[0],
                       "last_pos": tokens.shape[1] - 1, "pairs": pairs[:per_template]})
    return groups
