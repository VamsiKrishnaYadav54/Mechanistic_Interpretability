# save as: ~/mech_interp/code/scripts/download_models.py
# Run with: python download_models.py

import os

os.environ["HF_HOME"] = os.path.expanduser("/home/cccp/25m2125/VAMSI/mech_interp/models/huggingface_cache")

from transformer_lens import HookedTransformer

models_to_download = [
    "gpt2",           # 117M  — start here, fastest to iterate
    "gpt2-medium",    # 345M
    # "gpt2-large",     # 774M
    # "EleutherAI/pythia-70m",    # tiny, good for quick tests
    # "EleutherAI/pythia-160m",
    # "EleutherAI/pythia-1.4b",
    # Add larger ones once you need them:
    # "meta-llama/Llama-3.1-8B"  — needs HuggingFace token
]

for model_name in models_to_download:
    print(f"Downloading {model_name}...")
    try:
        model = HookedTransformer.from_pretrained(
            model_name
        )
        print(f"  ✓ {model_name} cached")
        del model  # free memory between downloads
    except Exception as e:
        print(f"  ✗ {model_name} failed: {e}")

print("All downloads complete.")