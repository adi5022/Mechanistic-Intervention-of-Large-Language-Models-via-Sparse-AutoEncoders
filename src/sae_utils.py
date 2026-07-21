"""
Utilities for loading and interacting with Sparse Autoencoders (SAEs) and TransformerLens models.
"""

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

def load_model_and_sae(device: str = "cpu", layer: int = 8):
    """
    Loads GPT-2 small and the pretrained SAE for the specified layer (default Layer 8).
    Available layers for gpt2-small-res-jb: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11.
    """
    # Load GPT-2 small
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    
    # Load pretrained SAE
    sae_id = f"blocks.{layer}.hook_resid_pre"
    sae, _, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=sae_id,
        device=device
    )
    
    return model, sae

