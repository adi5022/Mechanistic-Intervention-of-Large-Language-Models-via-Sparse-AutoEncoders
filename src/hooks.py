"""
Activation hooks for mechanistic intervention.
"""

import torch

def make_mute_and_boost_hook(mute_feature_ids: list[int], mute_strength: float,
                               boost_feature_ids: list[int], boost_strength: float, sae):
    """
    Applies muting and boosting to different features TOGETHER in a single pass,
    starting from the same original signal — not chained sequentially.
    """
    def hook_fn(resid, hook):
        feature_acts = sae.encode(resid)
        baseline_reconstructed = sae.decode(feature_acts)

        modified_acts = feature_acts.clone()
        for fid in mute_feature_ids:
            modified_acts[..., fid] = modified_acts[..., fid] * (1.0 - mute_strength)
        for fid in boost_feature_ids:
            modified_acts[..., fid] = modified_acts[..., fid] * (1.0 + boost_strength)

        reconstructed = sae.decode(modified_acts)
        delta = reconstructed - baseline_reconstructed
        return resid + delta

    return hook_fn

def make_ablation_hook(feature_id: int, sae, strength: float):
    """
    Creates an ablation hook for a specific feature ID and ablation strength.
    strength = 0.0 -> untouched
    strength = 1.0 -> fully ablated (zeroed out)
    
    Delta-patches the residual stream to avoid adding SAE reconstruction error.
    """
    def hook_fn(resid, hook):
        # resid shape: [batch, seq_len, d_model]
        # Encode activations into SAE sparse feature space ONCE
        feature_acts = sae.encode(resid)
        baseline_reconstructed = sae.decode(feature_acts)
        
        # Clone and apply ablation (scaling down by strength)
        modified_acts = feature_acts.clone()
        modified_acts[..., feature_id] = modified_acts[..., feature_id] * (1.0 - strength)
        
        # Decode back to residual stream space
        reconstructed = sae.decode(modified_acts)
        
        # Calculate delta (difference between ablated and baseline reconstructions)
        delta = reconstructed - baseline_reconstructed
        
        return resid + delta
        
    return hook_fn

def make_joint_ablation_hook(feature_ids: list[int], sae, strength: float):
    """
    Creates a joint ablation hook for a list of feature IDs and ablation strength.
    
    Encodes the residual stream ONCE, scales all listed feature_ids by (1 - strength)
    in that single encoded representation, decodes ONCE, and delta-patches the residual stream.
    """
    def hook_fn(resid, hook):
        if not feature_ids:
            return resid
            
        # Encode activations into SAE sparse feature space ONCE
        feature_acts = sae.encode(resid)
        baseline_reconstructed = sae.decode(feature_acts)
        
        # Create a modified copy of feature_acts and scale all listed feature_ids
        modified_acts = feature_acts.clone()
        for feature_id in feature_ids:
            modified_acts[..., feature_id] = modified_acts[..., feature_id] * (1.0 - strength)
            
        # Decode back to residual stream space ONCE
        reconstructed = sae.decode(modified_acts)
        
        # Calculate delta (difference between ablated and baseline reconstructions)
        delta = reconstructed - baseline_reconstructed
        
        return resid + delta
        
    return hook_fn

def make_signed_ablation_hook(feature_ids: list[int], sae, strength: float):
    """
    Creates a signed ablation/steering hook for a list of feature IDs.
    If strength > 0, boosts/amplifies feature activations (e.g. +0.5 -> 1.5x scaling).
    If strength < 0, mutes/ablates feature activations (e.g. -0.3 -> 0.7x scaling).
    """
    def hook_fn(resid, hook):
        if not feature_ids:
            return resid
            
        feature_acts = sae.encode(resid)
        baseline_reconstructed = sae.decode(feature_acts)
        
        modified_acts = feature_acts.clone()
        for feature_id in feature_ids:
            modified_acts[..., feature_id] = modified_acts[..., feature_id] * (1.0 + strength)
            
        reconstructed = sae.decode(modified_acts)
        delta = reconstructed - baseline_reconstructed
        return resid + delta
        
    return hook_fn

def build_scale_map(mute_ids, mute_strength: float, boost_ids, boost_strength: float) -> dict:
    """Per-feature multiplicative scale for an applied mute/boost set (product if a feature is in both)."""
    scale_map = {}
    for fid in mute_ids:
        scale_map[fid] = scale_map.get(fid, 1.0) * (1.0 - mute_strength)
    for fid in boost_ids:
        scale_map[fid] = scale_map.get(fid, 1.0) * (1.0 + boost_strength)
    return scale_map


def with_extra_scale(base_scale_map: dict | None, feature_id: int, scale: float) -> dict:
    """Copy of base_scale_map with feature_id additionally scaled by `scale`."""
    out = dict(base_scale_map or {})
    out[feature_id] = out.get(feature_id, 1.0) * scale
    return out


def make_scale_map_hook(scale_map: dict, sae):
    """Single encode/decode delta-patch hook scaling each feature id by its scale_map value."""
    def hook_fn(resid, hook):
        if not scale_map:
            return resid
        feature_acts = sae.encode(resid)
        baseline_reconstructed = sae.decode(feature_acts)
        modified = feature_acts.clone()
        idx = torch.as_tensor(list(scale_map.keys()), device=resid.device)
        scales = torch.as_tensor(list(scale_map.values()), device=resid.device, dtype=modified.dtype)
        modified[..., idx] = modified[..., idx] * scales
        return resid + (sae.decode(modified) - baseline_reconstructed)

    return hook_fn


def make_per_row_scale_hook_with_base(feature_ids: list[int], sae, scale: float, base_scale_map: dict):
    """
    Like make_per_row_scale_hook, but every row additionally has the already-applied
    features in base_scale_map scaled (the "steered state"), in the same single
    encode/decode pass. Row i then also scales feature_ids[i] by `scale`.
    """
    def hook_fn(resid, hook):
        B = resid.shape[0]
        assert B == len(feature_ids), (
            f"batch size {B} != len(feature_ids) {len(feature_ids)}"
        )
        feature_acts = sae.encode(resid)
        baseline_reconstructed = sae.decode(feature_acts)
        modified = feature_acts.clone()

        if base_scale_map:
            b_idx = torch.as_tensor(list(base_scale_map.keys()), device=resid.device)
            b_scales = torch.as_tensor(list(base_scale_map.values()), device=resid.device, dtype=modified.dtype)
            modified[:, :, b_idx] = modified[:, :, b_idx] * b_scales

        row_idx = torch.arange(B, device=resid.device)
        feat_idx = torch.as_tensor(feature_ids, device=resid.device)
        modified[row_idx, :, feat_idx] = modified[row_idx, :, feat_idx] * scale

        return resid + (sae.decode(modified) - baseline_reconstructed)

    return hook_fn


def make_per_row_scale_hook(feature_ids: list[int], sae, scale: float):
    """
    Batched counterpart of make_ablation_hook / make_signed_ablation_hook.

    resid arrives as [B, S, d_model] where B == len(feature_ids) — the prompt
    has been repeated B times along the batch dimension. Row i gets ONLY
    feature feature_ids[i] scaled by `scale`, at every sequence position
    (matching the existing single-item hooks' behavior of scaling the feature
    at all positions, not just the last).

    scale = 0.0    -> full ablation       (equivalent to make_ablation_hook strength=1.0)
    scale = 1 - s  -> mute at strength s  (equivalent to make_ablation_hook strength=s)
    scale = 1 + s  -> boost at strength s (equivalent to make_signed_ablation_hook +s)

    Delta-patching semantics are preserved exactly:
        delta = decode(modified_acts) - decode(clean_acts)
        return resid + delta
    """
    def hook_fn(resid, hook):
        B = resid.shape[0]
        assert B == len(feature_ids), (
            f"batch size {B} != len(feature_ids) {len(feature_ids)}"
        )
        feature_acts = sae.encode(resid)            # [B, S, n_features]
        baseline_reconstructed = sae.decode(feature_acts)
        modified = feature_acts.clone()

        row_idx = torch.arange(B, device=resid.device)
        feat_idx = torch.as_tensor(feature_ids, device=resid.device)
        modified[row_idx, :, feat_idx] = modified[row_idx, :, feat_idx] * scale

        return resid + (sae.decode(modified) - baseline_reconstructed)

    return hook_fn


