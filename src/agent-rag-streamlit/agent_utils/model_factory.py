"""
model_factory.py
Factory for creating Ollama chat models
"""

import logging
from typing import Optional
from langchain_ollama.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

# --- Configure logger ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# --- Model creation function ---
def create_ollama(
    model: str = "llama3.1:latest",
    temperature: float = 0.0,
    base_url: str = "http://localhost:11434",
    api_key: str = "ollama",
    **kwargs
) -> ChatOllama:
    """
    Create an Ollama chat model instance.
    """
    logger.info(f"Creating Ollama model: {model} (temperature={temperature}) at {base_url}")
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )


# --- Factory dispatcher ---
def create_model(
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
) -> BaseChatModel:
    """
    Create a model instance (Ollama only).
    """
    model = model or "llama3.1:latest"
    return create_ollama(model=model, temperature=temperature, **kwargs)


# --- Preset configs ---
PRESETS = {
    "default": dict(model="llama3.1:latest", temperature=0.0),
    "creative": dict(model="llama3.1:latest", temperature=0.7),
}

def from_preset(name: str, **overrides) -> BaseChatModel:
    """
    Create a model from a preset configuration.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    config = {**PRESETS[name], **overrides}
    return create_model(**config)
