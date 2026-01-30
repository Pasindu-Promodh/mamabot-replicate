#!/usr/bin/env python3
"""
Script to download the model during Docker build phase.
This ensures the model is cached in the image.
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    """Download model and tokenizer to cache them in the Docker image"""
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if not hf_token:
        print("WARNING: HUGGING_FACE_HUB_TOKEN not set. Model download may fail if the model is gated.")
        print("If the model is public, this is fine. If it's gated, the build will fail.")
    
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        'HelpMumHQ/MamaBot-Llama',
        token=hf_token
    )
    print("Tokenizer downloaded successfully!")
    
    print("Downloading model (this will take several minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        'HelpMumHQ/MamaBot-Llama',
        token=hf_token,
        torch_dtype="auto"  # Will use appropriate dtype
    )
    print("Model downloaded successfully!")
    
    print("Model and tokenizer are now cached in the image.")

if __name__ == "__main__":
    download_model()