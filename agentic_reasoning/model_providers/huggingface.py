"""HuggingFace model provider implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator

try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        AutoModel,
        pipeline,
        Pipeline
    )
    from sentence_transformers import SentenceTransformer
    HF_AVAILABLE = True
except ImportError:
    # Mock classes for when dependencies are not available
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModel = None
    pipeline = None
    Pipeline = None
    SentenceTransformer = None
    HF_AVAILABLE = False

from .base import ModelProvider, ModelResponse

logger = logging.getLogger(__name__)


class HuggingFaceProvider(ModelProvider):
    """HuggingFace model provider with backup capabilities."""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        
        if not HF_AVAILABLE:
            logger.warning("HuggingFace dependencies not available. Install with: pip install transformers torch sentence-transformers")
        
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.pipeline: Optional[Any] = None
        self.embedding_model: Optional[Any] = None
        self.device = kwargs.get("device", "cpu")
        self.max_length = kwargs.get("max_length", 512)
        self.temperature = kwargs.get("temperature", 0.7)
        self.model_type = kwargs.get("model_type", "text-generation")
    
    async def initialize(self) -> None:
        """Initialize the HuggingFace model."""
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace dependencies not available. Install with: pip install transformers torch sentence-transformers")
        
        try:
            logger.info(f"Initializing HuggingFace model: {self.model_name}")
            
            if self.model_type == "embedding":
                # Initialize sentence transformer for embeddings
                self.embedding_model = SentenceTransformer(self.model_name)
                logger.info(f"Initialized embedding model: {self.model_name}")
            else:
                # Initialize tokenizer and model for text generation
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # Add padding token if missing
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Initialize pipeline for text generation
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() and self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                
                logger.info(f"Initialized text generation model: {self.model_name}")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model {self.model_name}: {e}")
            self._initialized = False
            raise
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response using HuggingFace model."""
        await self.ensure_initialized()
        
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace dependencies not available")
        
        if not self.pipeline:
            raise RuntimeError("Text generation pipeline not initialized")
        
        try:
            # Merge kwargs with defaults
            generation_kwargs = {
                "max_length": kwargs.get("max_length", self.max_length),
                "temperature": kwargs.get("temperature", self.temperature),
                "do_sample": kwargs.get("do_sample", True),
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False
            }
            
            # Run generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.pipeline(prompt, **generation_kwargs)
            )
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            else:
                generated_text = str(result)
            
            return ModelResponse(
                content=generated_text,
                model_name=self.model_name,
                provider="huggingface",
                metadata={
                    "generation_kwargs": generation_kwargs,
                    "device": self.device
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed for {self.model_name}: {e}")
            raise
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using HuggingFace model."""
        await self.ensure_initialized()
        
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace dependencies not available")
        
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            # Run embedding in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.embedding_model.encode(text)
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Embedding failed for {self.model_name}: {e}")
            raise
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream generate responses (simplified implementation)."""
        response = await self.generate(prompt, **kwargs)
        
        # Simple streaming simulation - yield words one by one
        words = response.content.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.1)  # Simulate streaming delay
    
    async def is_available(self) -> bool:
        """Check if the HuggingFace model is available."""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace dependencies not available")
            return False
        
        try:
            if not self._initialized:
                await self.initialize()
            return True
        except Exception as e:
            logger.warning(f"HuggingFace model {self.model_name} not available: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Perform a health check on the HuggingFace model."""
        try:
            if not await self.is_available():
                return False
            
            # Test with a simple prompt
            if self.model_type == "embedding":
                await self.embed("test")
            else:
                await self.generate("Hello", max_length=10)
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for {self.model_name}: {e}")
            return False