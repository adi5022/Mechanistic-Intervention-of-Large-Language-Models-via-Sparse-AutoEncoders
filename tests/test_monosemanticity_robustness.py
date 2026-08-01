import unittest
from src.monosemanticity import (
    validate_feature_id,
    validate_neuron_index,
    get_curated_feature_registry,
    get_sae_status_summary,
    find_max_activating_examples,
    find_max_activating_neuron_examples,
    scan_dual_activations,
)
from src.explain import get_empty_state_guidance


class MockConfig:
    d_sae = 24576
    d_mlp = 3072
    release = "gpt2-small-res-jb"
    hook_name = "blocks.8.hook_resid_pre"


class MockSAE:
    cfg = MockConfig()


class MockModel:
    cfg = MockConfig()


class TestMonosemanticityRobustness(unittest.TestCase):

    def test_validate_feature_id(self):
        sae = MockSAE()
        valid_0, _ = validate_feature_id(sae, 0)
        self.assertTrue(valid_0)

        valid_313, _ = validate_feature_id(sae, 313)
        self.assertTrue(valid_313)

        valid_max, _ = validate_feature_id(sae, 24575)
        self.assertTrue(valid_max)

        invalid_neg, msg_neg = validate_feature_id(sae, -1)
        self.assertFalse(invalid_neg)
        self.assertIn("out of bounds", msg_neg)

        invalid_high, msg_high = validate_feature_id(sae, 99999)
        self.assertFalse(invalid_high)
        self.assertIn("out of bounds", msg_high)

    def test_validate_neuron_index(self):
        model = MockModel()
        valid_0, _ = validate_neuron_index(model, 8, 0)
        self.assertTrue(valid_0)

        valid_max, _ = validate_neuron_index(model, 8, 3071)
        self.assertTrue(valid_max)

        invalid_neg, _ = validate_neuron_index(model, 8, -5)
        self.assertFalse(invalid_neg)

        invalid_high, _ = validate_neuron_index(model, 8, 5000)
        self.assertFalse(invalid_high)

    def test_curated_feature_registry(self):
        registry = get_curated_feature_registry(layer=8)
        self.assertIsInstance(registry, list)
        self.assertGreaterEqual(len(registry), 5)
        first = registry[0]
        self.assertIn("feature_id", first)
        self.assertIn("concept", first)
        self.assertIn("category", first)

    def test_get_sae_status_summary(self):
        model = MockModel()
        sae = MockSAE()
        summary = get_sae_status_summary(model, sae, layer=8, corpus=["Test sentence 1", "Test sentence 2"])
        self.assertEqual(summary["release"], "gpt2-small-res-jb")
        self.assertEqual(summary["d_sae"], 24576)
        self.assertEqual(summary["d_mlp"], 3072)
        self.assertEqual(summary["corpus_count"], 2)

    def test_empty_state_guidance_fallback(self):
        empty_feat = get_empty_state_guidance("invalid_feature_id", {"feature_id": 99999})
        self.assertIn("title", empty_feat)
        self.assertIn("why", empty_feat)
        self.assertIn("99999", empty_feat["why"])

        empty_no_act = get_empty_state_guidance("no_activations", {"feature_id": 313})
        self.assertIn("No Activations Detected", empty_no_act["title"])

    def test_empty_corpus_handling(self):
        model = MockModel()
        sae = MockSAE()
        neuron_ex = find_max_activating_neuron_examples(model, 0, 8, corpus=[])
        self.assertEqual(neuron_ex, [])
        n_dual, s_dual = scan_dual_activations(model, sae, 0, 313, 8, corpus=[])
        self.assertEqual(n_dual, [])
        self.assertEqual(s_dual, [])


if __name__ == "__main__":
    unittest.main()
