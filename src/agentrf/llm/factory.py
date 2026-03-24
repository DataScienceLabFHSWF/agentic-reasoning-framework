import logging
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory to instantiate LLMs based on provider and model name."""

    @staticmethod
    def create(
        provider: str,
        model: str,
        *,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> BaseChatModel:
        provider_normalized = provider.lower()

        if provider_normalized == "ollama":
            from langchain_ollama.chat_models import ChatOllama

            logger.info(f"Initializing Ollama model: {model} at {base_url}")
            return ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url,
            )

        raise ValueError(f"Unsupported LLM provider: {provider}")
