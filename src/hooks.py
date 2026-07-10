"""
Activation hooks for mechanistic intervention.
"""

import torch

def make_ablation_hook(feature_id: int, sae, strength: float):
    """
    Creates an ablation hook for a specific feature ID and ablation strength.
    strength = 0.0 -> untouched
    strength = 1.0 -> fully ablated (zeroed out)
    
    Delta-patches the residual stream to avoid adding SAE reconstruction error.
    """
    def hook_fn(resid, hook):
        # resid shape: [batch, seq_len, d_model]
        # Encode activations into SAE sparse feature space
        feature_acts = sae.encode(resid)
        original_val = feature_acts[..., feature_id].clone()
        
        # Apply ablation (scaling down by strength)
        feature_acts[..., feature_id] = original_val * (1.0 - strength)
        
        # Decode back to residual stream space
        reconstructed = sae.decode(feature_acts)
        
        # Calculate delta (difference between ablated and baseline reconstructions)
        baseline_reconstructed = sae.decode(sae.encode(resid))
        delta = reconstructed - baseline_reconstructed
        
        return resid + delta
        
    return hook_fn
