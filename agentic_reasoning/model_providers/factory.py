"""Factory for creating model providers."""

from typing import Dict, Type
from .base import ModelProvider
from .huggingface import HuggingFaceProvider
from ..config import ModelConfig


class ModelProviderFactory:
    """Factory for creating model provider instances."""
    
    _providers: Dict[str, Type[ModelProvider]] = {
        "huggingface": HuggingFaceProvider,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[ModelProvider]) -> None:
        """Register a new model provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create_provider(cls, config: ModelConfig) -> ModelProvider:
        """Create a model provider instance from configuration."""
        provider_class = cls._providers.get(config.provider)
        
        if not provider_class:
            raise ValueError(f"Unknown provider: {config.provider}")
        
        return provider_class(
            model_name=config.name,
            **config.model_kwargs
        )
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())