"""
Model management for the PDF QA Chatbot
"""

from dataclasses import dataclass
from typing import List, Optional
import os
from pathlib import Path

@dataclass
class ModelInfo:
    name: str
    description: str
    size: str
    type: str  # 'gpt4all' or 'llama'
    filename: str
    recommended_for: List[str]

AVAILABLE_MODELS = [
    ModelInfo(
        name="GPT4All-J Groovy",
        description="Fast, efficient model good for general Q&A and summarization",
        size="3.8GB",
        type="gpt4all",
        filename="ggml-gpt4all-j-v1.3-groovy.bin",
        recommended_for=["summarization", "q&a", "general tasks"]
    ),
    ModelInfo(
        name="Llama 2 7B Chat",
        description="Well-balanced model for chat and comprehension",
        size="3.8GB",
        type="llama",
        filename="llama-2-7b-chat.Q4_K_M.gguf",
        recommended_for=["chat", "comprehension", "analysis"]
    ),
    ModelInfo(
        name="GPT4All Falcon",
        description="Powerful model for technical content and analysis",
        size="4GB",
        type="gpt4all",
        filename="gpt4all-falcon-q4_0.bin",
        recommended_for=["technical content", "detailed analysis"]
    )
]

def ensure_models_directory():
    """Create models directory if it doesn't exist."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    return models_dir

def get_model_path(model_info: ModelInfo) -> Optional[Path]:
    """Get the path to a model file, checking multiple locations."""
    # Check models directory first (preferred location)
    models_dir = Path("models")
    model_path = models_dir / model_info.filename
    if model_path.exists():
        return model_path
    
    # Check current directory as fallback
    current_path = Path(model_info.filename)
    if current_path.exists():
        return current_path
    
    return None

def get_available_local_models() -> List[ModelInfo]:
    """Get list of models available locally."""
    available_models = []
    for model_info in AVAILABLE_MODELS:
        if get_model_path(model_info) is not None:
            available_models.append(model_info)
    return available_models

def list_models(show_all=False):
    """List all models with their information."""
    print("\nAvailable Models:")
    print("-" * 80)
    
    if show_all:
        models_to_show = AVAILABLE_MODELS
        print("All supported models (including not downloaded):")
    else:
        models_to_show = get_available_local_models()
        print("Currently downloaded models:")
    
    for model in models_to_show:
        print(f"\nName: {model.name}")
        print(f"Type: {model.type}")
        print(f"Size: {model.size}")
        print(f"Description: {model.description}")
        print(f"Recommended for: {', '.join(model.recommended_for)}")
        print(f"Filename: {model.filename}")
        if show_all:
            path = get_model_path(model)
            status = f"Found at: {path}" if path else "Not Downloaded"
            print(f"Status: {status}")
        print("-" * 40)

def get_model_info(name: str) -> Optional[ModelInfo]:
    """Return the ModelInfo object matching the given model name."""
    for model in AVAILABLE_MODELS:
        if model.name == name:
            return model
    return None

def download_model(model_info: ModelInfo) -> Optional[Path]:
    """Download the model file for the given ModelInfo and save it to the models directory."""
    import requests
    models_dir = ensure_models_directory()
    model_path = models_dir / model_info.filename

    # If model already exists, return the path
    if model_path.exists():
        return model_path

    # Use the actual URL for GPT4All Falcon model, else fallback to example URL
    if model_info.filename == "gpt4all-falcon-q4_0.bin":
        url = "https://huggingface.co/nomic-ai/gpt4all-falcon-ggml/resolve/main/ggml-model-gpt4all-falcon-q4_0.bin?download=true"
    else:
        base_url = "https://example.com/models/"
        url = base_url + model_info.filename

    try:
        print(f"Downloading model from {url} ...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return model_path
    except Exception as e:
        print(f"Failed to download model: {e}")
        return None

if __name__ == "__main__":
    # Show all available models and their status
    list_models(show_all=True)
