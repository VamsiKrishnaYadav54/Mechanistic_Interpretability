"""Model loading utilities."""
import os
import torch
from transformer_lens import HookedTransformer


def load_model(model_name: str, device: str = "cuda") -> HookedTransformer:
    """Load a HookedTransformer model from local cache."""
    print(f"Loading {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(
        model_name,
        cache_dir=os.environ.get("HF_HOME", "/home/cccp/25m2125/VAMSI/mech_interp/models/huggingface_cache")
    )
    model = model.to(device)
    model.eval()
    print(f"Loaded: {model_name} | Layers: {model.cfg.n_layers} | Heads: {model.cfg.n_heads}")
    return model