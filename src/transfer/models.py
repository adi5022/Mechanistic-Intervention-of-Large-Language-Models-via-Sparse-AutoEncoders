"""
Model loading and tokenizer checks for the cross-model transfer study.

Models are loaded through TransformerLens with its default settings, exactly as src/sae_utils.load_base_model does for GPT-2 small
(layer-norm weights folded in, writing weights centred), so the residual stream here is the same quantity the rest of the project uses.
"""
import os

import torch
from transformer_lens import HookedTransformer


def pick_device(forced: str | None = None) -> str:
    """cuda > mps > cpu. `forced`, or the environment variable FEATURESCALPEL_DEVICE, overrides the choice."""
    forced = (forced or os.environ.get("FEATURESCALPEL_DEVICE", "")).strip().lower()
    if forced in ("cpu", "mps", "cuda"):
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: str | None = None) -> HookedTransformer:
    """Load a TransformerLens model by its official name (for example 'gpt2' or 'gpt2-medium') and put it in eval mode."""
    model = HookedTransformer.from_pretrained(name, device=device or pick_device())
    model.eval()
    return model


def model_facts(model: HookedTransformer) -> dict:
    """The numbers that matter for pairing two models: depth, width, vocabulary, context length, size."""
    c = model.cfg
    return {
        "n_layers": int(c.n_layers),
        "d_model": int(c.d_model),
        "d_vocab": int(c.d_vocab),
        "n_ctx": int(c.n_ctx),
        # GPT-2 ties the output table to the input embedding; TransformerLens stores a second copy (unembed.W_U), which is not counted here
        "params_millions": round(sum(p.numel() for n, p in model.named_parameters() if not n.startswith("unembed.W_U")) / 1e6, 1),
    }


def tokenizers_match(tok_a, tok_b, texts: list[str]) -> tuple[bool, str]:
    """True only if both Hugging Face tokenizers have the same vocabulary, the same special tokens, and give the same ids for every text."""
    if tok_a.get_vocab() != tok_b.get_vocab():
        return False, "vocabularies differ"
    if (tok_a.bos_token_id, tok_a.eos_token_id) != (tok_b.bos_token_id, tok_b.eos_token_id):
        return False, "special token ids differ"
    for i, t in enumerate(texts):
        if tok_a(t)["input_ids"] != tok_b(t)["input_ids"]:
            return False, f"token ids differ on text #{i}"
    return True, f"same vocabulary ({len(tok_a.get_vocab())} entries), same special tokens, identical ids on {len(texts)} texts"
