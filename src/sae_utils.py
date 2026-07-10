"""
Utilities for loading and interacting with Sparse Autoencoders (SAEs) and TransformerLens models.
"""

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

def load_model_and_sae(device: str = "cpu"):
    """
    Loads GPT-2 small and the pretrained SAE at Layer 8.
    """
    # Load GPT-2 small
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    
    # Load pretrained SAE
    # Note: sae_id matches blocks.8.hook_resid_pre
    sae, _, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=device
    )
    
    return model, sae
