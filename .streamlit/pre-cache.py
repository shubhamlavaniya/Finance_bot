import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# This is a flag to only run this script when building the Streamlit app.
# It's an optional safety measure but good practice.
IS_STREAMLIT_CLOUD = "STREAMLIT_CLOUD" in os.environ

if IS_STREAMLIT_CLOUD:
    print("Pre-caching TinyLlama model for Streamlit Cloud deployment...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Use snapshot_download to get the model files. It's more robust
    # than from_pretrained for this use case.
    snapshot_download(
        repo_id=model_id,
        allow_patterns=["*"],
        cache_dir="/mount/streamlit_cache",
        local_dir_use_symlinks=False,
    )
    print("Model pre-caching complete.")

# You can also use this space to add other pre-caching logic
# for your tokenizer, etc.