"""Example demonstrating HuggingFace backup functionality."""

import asyncio
import logging
from agentic_reasoning.config import FrameworkConfig, ModelConfig
from agentic_reasoning.backup_manager import BackupModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function."""
    logger.info("Starting HuggingFace backup example")
    
    # Create configuration with HuggingFace backups
    config = FrameworkConfig(
        # Disable primary models to demonstrate backup usage
        primary_llm=ModelConfig(name="disabled", provider="none", enabled=False),
        primary_embeddings=ModelConfig(name="disabled", provider="none", enabled=False),
        
        # Configure HuggingFace backup models
        backup_llm=[
            ModelConfig(
                name="microsoft/DialoGPT-small",  # Smaller model for faster demo
                provider="huggingface",
                model_kwargs={
                    "max_length": 50,
                    "temperature": 0.7,
                    "model_type": "text-generation"
                },
                enabled=True
            )
        ],
        backup_embeddings=[
            ModelConfig(
                name="sentence-transformers/all-MiniLM-L6-v2",
                provider="huggingface",
                model_kwargs={
                    "model_type": "embedding"
                },
                enabled=True
            )
        ]
    )
    
    # Initialize backup manager
    manager = BackupModelManager(config)
    await manager.initialize()
    
    # Test text generation with backup
    try:
        logger.info("Testing text generation with HuggingFace backup...")
        response = await manager.generate_with_backup("Hello, how are you?")
        logger.info(f"Generated response: {response.content}")
        logger.info(f"Used model: {response.model_name} (provider: {response.provider})")
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
    
    # Test embeddings with backup
    try:
        logger.info("Testing embeddings with HuggingFace backup...")
        embeddings = await manager.embed_with_backup("This is a test sentence.")
        logger.info(f"Generated embeddings (first 5 dimensions): {embeddings[:5]}")
        logger.info(f"Embedding vector length: {len(embeddings)}")
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
    
    # Test health check
    try:
        logger.info("Running health check...")
        health_results = await manager.health_check_all()
        for model, status in health_results.items():
            logger.info(f"Health check - {model}: {'✓' if status else '✗'}")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
    
    logger.info("Example completed")


if __name__ == "__main__":
    asyncio.run(main())