import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sae_utils import load_model_and_sae
from src.monosemanticity import (
    find_max_activating_examples,
    compute_feature_similarity,
    find_most_similar_features,
)


@pytest.mark.slow
def test_monosemanticity_helpers_run_on_real_model_and_sae():
    model, sae = load_model_and_sae(device="cpu", layer=8)
    corpus = [
        "The Eiffel Tower is in Paris.",
        "The capital of France is Paris.",
        "Rome is the capital of Italy.",
        "London is a major European city.",
    ]

    examples = find_max_activating_examples(model, sae, feature_id=313, corpus=corpus, top_n=5)
    assert len(examples) == 5
    assert all("activation" in item for item in examples)
    assert all("token_position" in item for item in examples)

    similarity = compute_feature_similarity(sae, 313, 314)
    assert isinstance(similarity, float)

    similar = find_most_similar_features(sae, feature_id=313, top_n=3)
    assert len(similar) == 3
    assert all(isinstance(item[0], int) and isinstance(item[1], float) for item in similar)
