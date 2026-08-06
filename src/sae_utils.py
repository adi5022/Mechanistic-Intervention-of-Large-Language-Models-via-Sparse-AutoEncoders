"""
Utilities for loading and interacting with Sparse Autoencoders (SAEs) and TransformerLens models.
"""

import os
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure Hugging Face token is set to prevent rate-limiting
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
    except Exception:
        pass

def get_default_device() -> str:
    """
    Returns the best available device for PyTorch operations (CUDA GPU, Apple MPS, or CPU fallback).
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_base_model(device: str = None):
    """
    Loads GPT-2 small base model onto the specified or auto-detected device.
    """
    if device is None:
        device = get_default_device()
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    return model

def load_sae_for_layer(layer: int = 8, device: str = None):
    """
    Loads the pretrained SAE for the specified layer onto the specified or auto-detected device.
    """
    if device is None:
        device = get_default_device()
    sae_id = f"blocks.{layer}.hook_resid_pre"
    sae, _, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=sae_id,
        device=device
    )
    sae.eval()
    return sae

def load_model_and_sae(device: str = None, layer: int = 8, model: HookedTransformer = None):
    """
    Loads GPT-2 small and the pretrained SAE for the specified layer (default Layer 8).
    Auto-detects CUDA/MPS/CPU if device is None.
    Optionally accepts a pre-instantiated base model to prevent redundant re-initialization.
    """
    if device is None:
        device = get_default_device()
    if model is None:
        model = load_base_model(device=device)
    sae = load_sae_for_layer(layer=layer, device=device)
    return model, sae

