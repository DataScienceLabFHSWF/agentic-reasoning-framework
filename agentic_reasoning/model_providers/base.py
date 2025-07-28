"""Base model provider interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import BaseModel


class ModelResponse(BaseModel):
    """Standardized response from model providers."""
    
    content: str
    model_name: str
    provider: str
    metadata: Dict[str, Any] = {}
    usage: Optional[Dict[str, Any]] = None


class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model provider."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for the given text."""
        pass
    
    @abstractmethod
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream generate responses from the model."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the model provider is available."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Perform a health check on the model provider."""
        pass
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.__class__.__name__.replace("Provider", "").lower()
    
    async def ensure_initialized(self) -> None:
        """Ensure the provider is initialized."""
        if not self._initialized:
            await self.initialize()
            self._initialized = True