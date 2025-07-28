"""Backup model manager with fallback logic."""

import asyncio
import logging
from typing import List, Optional, AsyncGenerator, Dict
from .model_providers import ModelProvider, ModelProviderFactory, ModelResponse
from .config import FrameworkConfig, ModelConfig

logger = logging.getLogger(__name__)


class BackupModelManager:
    """Manages primary and backup models with automatic fallback."""
    
    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.primary_llm: Optional[ModelProvider] = None
        self.backup_llms: List[ModelProvider] = []
        self.primary_embeddings: Optional[ModelProvider] = None
        self.backup_embeddings: List[ModelProvider] = []
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all model providers."""
        logger.info("Initializing backup model manager")
        
        try:
            # Initialize primary models (if available)
            if self.config.primary_llm.enabled:
                try:
                    self.primary_llm = ModelProviderFactory.create_provider(self.config.primary_llm)
                    logger.info(f"Primary LLM configured: {self.config.primary_llm.name}")
                except Exception as e:
                    logger.warning(f"Failed to configure primary LLM: {e}")
            
            if self.config.primary_embeddings.enabled:
                try:
                    self.primary_embeddings = ModelProviderFactory.create_provider(self.config.primary_embeddings)
                    logger.info(f"Primary embeddings configured: {self.config.primary_embeddings.name}")
                except Exception as e:
                    logger.warning(f"Failed to configure primary embeddings: {e}")
            
            # Initialize backup models
            for backup_config in self.config.backup_llm:
                if backup_config.enabled:
                    try:
                        backup_provider = ModelProviderFactory.create_provider(backup_config)
                        self.backup_llms.append(backup_provider)
                        logger.info(f"Backup LLM configured: {backup_config.name}")
                    except Exception as e:
                        logger.warning(f"Failed to configure backup LLM {backup_config.name}: {e}")
            
            for backup_config in self.config.backup_embeddings:
                if backup_config.enabled:
                    try:
                        backup_provider = ModelProviderFactory.create_provider(backup_config)
                        self.backup_embeddings.append(backup_provider)
                        logger.info(f"Backup embeddings configured: {backup_config.name}")
                    except Exception as e:
                        logger.warning(f"Failed to configure backup embeddings {backup_config.name}: {e}")
            
            self._initialized = True
            logger.info("Backup model manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize backup model manager: {e}")
            raise
    
    async def generate_with_backup(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response with automatic fallback to backup models."""
        if not self._initialized:
            await self.initialize()
        
        # Try primary model first
        if self.primary_llm:
            try:
                response = await self._try_generate_with_retries(
                    self.primary_llm, prompt, **kwargs
                )
                logger.debug(f"Successfully used primary LLM: {self.primary_llm.model_name}")
                return response
            except Exception as e:
                logger.warning(f"Primary LLM failed: {e}")
        
        # Try backup models
        if self.config.backup_config.fallback_enabled and self.backup_llms:
            for backup_llm in self.backup_llms:
                try:
                    response = await self._try_generate_with_retries(
                        backup_llm, prompt, **kwargs
                    )
                    logger.info(f"Successfully used backup LLM: {backup_llm.model_name}")
                    return response
                except Exception as e:
                    logger.warning(f"Backup LLM {backup_llm.model_name} failed: {e}")
                    continue
        
        raise RuntimeError("All language models failed")
    
    async def embed_with_backup(self, text: str) -> List[float]:
        """Generate embeddings with automatic fallback to backup models."""
        if not self._initialized:
            await self.initialize()
        
        # Try primary model first
        if self.primary_embeddings:
            try:
                embeddings = await self._try_embed_with_retries(
                    self.primary_embeddings, text
                )
                logger.debug(f"Successfully used primary embeddings: {self.primary_embeddings.model_name}")
                return embeddings
            except Exception as e:
                logger.warning(f"Primary embeddings failed: {e}")
        
        # Try backup models
        if self.config.backup_config.fallback_enabled and self.backup_embeddings:
            for backup_embeddings in self.backup_embeddings:
                try:
                    embeddings = await self._try_embed_with_retries(
                        backup_embeddings, text
                    )
                    logger.info(f"Successfully used backup embeddings: {backup_embeddings.model_name}")
                    return embeddings
                except Exception as e:
                    logger.warning(f"Backup embeddings {backup_embeddings.model_name} failed: {e}")
                    continue
        
        raise RuntimeError("All embedding models failed")
    
    async def stream_generate_with_backup(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream generate with automatic fallback to backup models."""
        if not self._initialized:
            await self.initialize()
        
        # Try primary model first
        if self.primary_llm:
            try:
                async for chunk in self.primary_llm.stream_generate(prompt, **kwargs):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Primary LLM streaming failed: {e}")
        
        # Try backup models
        if self.config.backup_config.fallback_enabled and self.backup_llms:
            for backup_llm in self.backup_llms:
                try:
                    async for chunk in backup_llm.stream_generate(prompt, **kwargs):
                        yield chunk
                    return
                except Exception as e:
                    logger.warning(f"Backup LLM {backup_llm.model_name} streaming failed: {e}")
                    continue
        
        raise RuntimeError("All language models failed for streaming")
    
    async def _try_generate_with_retries(
        self, provider: ModelProvider, prompt: str, **kwargs
    ) -> ModelResponse:
        """Try to generate with retries."""
        last_exception = None
        
        for attempt in range(self.config.backup_config.max_retries):
            try:
                # Apply timeout if configured
                if self.config.backup_config.timeout:
                    response = await asyncio.wait_for(
                        provider.generate(prompt, **kwargs),
                        timeout=self.config.backup_config.timeout
                    )
                else:
                    response = await provider.generate(prompt, **kwargs)
                
                return response
                
            except Exception as e:
                last_exception = e
                logger.debug(f"Attempt {attempt + 1} failed for {provider.model_name}: {e}")
                
                if attempt < self.config.backup_config.max_retries - 1:
                    await asyncio.sleep(self.config.backup_config.retry_delay)
        
        raise last_exception or RuntimeError("All retries failed")
    
    async def _try_embed_with_retries(self, provider: ModelProvider, text: str) -> List[float]:
        """Try to embed with retries."""
        last_exception = None
        
        for attempt in range(self.config.backup_config.max_retries):
            try:
                # Apply timeout if configured
                if self.config.backup_config.timeout:
                    embeddings = await asyncio.wait_for(
                        provider.embed(text),
                        timeout=self.config.backup_config.timeout
                    )
                else:
                    embeddings = await provider.embed(text)
                
                return embeddings
                
            except Exception as e:
                last_exception = e
                logger.debug(f"Attempt {attempt + 1} failed for {provider.model_name}: {e}")
                
                if attempt < self.config.backup_config.max_retries - 1:
                    await asyncio.sleep(self.config.backup_config.retry_delay)
        
        raise last_exception or RuntimeError("All retries failed")
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all configured models."""
        if not self._initialized:
            await self.initialize()
        
        results = {}
        
        # Check primary models
        if self.primary_llm:
            results[f"primary_llm_{self.primary_llm.model_name}"] = await self.primary_llm.health_check()
        
        if self.primary_embeddings:
            results[f"primary_embeddings_{self.primary_embeddings.model_name}"] = await self.primary_embeddings.health_check()
        
        # Check backup models
        for i, backup_llm in enumerate(self.backup_llms):
            results[f"backup_llm_{i}_{backup_llm.model_name}"] = await backup_llm.health_check()
        
        for i, backup_embeddings in enumerate(self.backup_embeddings):
            results[f"backup_embeddings_{i}_{backup_embeddings.model_name}"] = await backup_embeddings.health_check()
        
        return results