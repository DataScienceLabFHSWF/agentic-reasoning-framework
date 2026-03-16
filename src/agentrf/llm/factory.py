import logging
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory to instantiate LLMs purely from configuration dictionaries."""
    
    @staticmethod
    def create(config: Dict[str, Any]) -> BaseChatModel:
        provider = config.get("provider", "ollama").lower()
        
        if provider == "ollama":
            from langchain_ollama.chat_models import ChatOllama
            
            logger.info(f"Initializing Ollama model: {config.get('model')} at {config.get('base_url')}")
            return ChatOllama(
                model=config.get("model"),
                temperature=config.get("temperature"),
                base_url=config.get("base_url"),
            )
        
        raise ValueError(f"Unsupported LLM provider: {provider}")