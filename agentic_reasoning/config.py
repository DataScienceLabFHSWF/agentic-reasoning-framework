"""Configuration management for the agentic reasoning framework."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    
    name: str = Field(..., description="Model name or identifier")
    provider: str = Field(..., description="Model provider (e.g., 'openai', 'huggingface')")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")
    enabled: bool = Field(default=True, description="Whether this model is enabled")


class BackupConfig(BaseModel):
    """Configuration for backup model behavior."""
    
    max_retries: int = Field(default=3, description="Maximum number of retries for primary model")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    fallback_enabled: bool = Field(default=True, description="Whether to use backup models")
    timeout: Optional[float] = Field(default=30.0, description="Timeout for model calls in seconds")


class FrameworkConfig(BaseSettings):
    """Main configuration for the agentic reasoning framework."""
    
    # Primary models
    primary_llm: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="gpt-3.5-turbo",
            provider="openai"
        ),
        description="Primary language model configuration"
    )
    
    primary_embeddings: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="text-embedding-ada-002",
            provider="openai"
        ),
        description="Primary embeddings model configuration"
    )
    
    # Backup models (HuggingFace)
    backup_llm: List[ModelConfig] = Field(
        default_factory=lambda: [
            ModelConfig(
                name="microsoft/DialoGPT-medium",
                provider="huggingface",
                model_kwargs={"max_length": 1000, "temperature": 0.7}
            ),
            ModelConfig(
                name="facebook/blenderbot-400M-distill",
                provider="huggingface",
                model_kwargs={"max_length": 1000}
            )
        ],
        description="Backup language models (HuggingFace)"
    )
    
    backup_embeddings: List[ModelConfig] = Field(
        default_factory=lambda: [
            ModelConfig(
                name="sentence-transformers/all-MiniLM-L6-v2",
                provider="huggingface"
            ),
            ModelConfig(
                name="sentence-transformers/all-mpnet-base-v2",
                provider="huggingface"
            )
        ],
        description="Backup embedding models (HuggingFace)"
    )
    
    # Backup behavior configuration
    backup_config: BackupConfig = Field(
        default_factory=BackupConfig,
        description="Backup and retry configuration"
    )
    
    # Framework settings
    log_level: str = Field(default="INFO", description="Logging level")
    enable_tracing: bool = Field(default=True, description="Enable LangSmith tracing")
    
    model_config = ConfigDict(
        env_prefix="ARF_",  # Agentic Reasoning Framework
        env_file=".env"
    )