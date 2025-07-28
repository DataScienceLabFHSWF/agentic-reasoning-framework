#!/usr/bin/env python3
"""Simple demo showing the HuggingFace backup framework structure."""

import asyncio
import logging
from agentic_reasoning.config import FrameworkConfig, ModelConfig
from agentic_reasoning.backup_manager import BackupModelManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def demo_framework_structure():
    """Demonstrate the framework structure without requiring HF dependencies."""
    print("🚀 Agentic Reasoning Framework - HuggingFace Backup Demo")
    print("=" * 60)
    
    # 1. Configuration Demo
    print("\n📋 1. Configuration System")
    config = FrameworkConfig(
        primary_llm=ModelConfig(name="disabled", provider="none", enabled=False),
        primary_embeddings=ModelConfig(name="disabled", provider="none", enabled=False),
        backup_llm=[
            ModelConfig(
                name="microsoft/DialoGPT-small",
                provider="huggingface",
                model_kwargs={"max_length": 100, "temperature": 0.7},
                enabled=True
            )
        ],
        backup_embeddings=[
            ModelConfig(
                name="sentence-transformers/all-MiniLM-L6-v2",
                provider="huggingface",
                model_kwargs={"model_type": "embedding"},
                enabled=True
            )
        ]
    )
    
    print(f"✅ Primary LLM: {config.primary_llm.name} ({config.primary_llm.provider})")
    print(f"✅ Backup LLMs: {len(config.backup_llm)} configured")
    print(f"✅ Backup Embeddings: {len(config.backup_embeddings)} configured")
    print(f"✅ Max retries: {config.backup_config.max_retries}")
    print(f"✅ Fallback enabled: {config.backup_config.fallback_enabled}")
    
    # 2. Manager Initialization
    print("\n⚙️  2. Manager Initialization")
    manager = BackupModelManager(config)
    
    try:
        await manager.initialize()
        print("✅ Manager initialized successfully!")
    except Exception as e:
        print(f"⚠️  Manager initialization: {e}")
        print("   This is expected without HuggingFace dependencies installed")
    
    # 3. Health Check Demo
    print("\n🔍 3. Health Check System")
    try:
        health_results = await manager.health_check_all()
        print("📊 Health Check Results:")
        for model, status in health_results.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {model}")
    except Exception as e:
        print(f"⚠️  Health check: {e}")
        print("   This demonstrates the graceful error handling")
    
    # 4. Framework Benefits
    print("\n🎯 4. Framework Benefits")
    print("✅ Automatic failover from primary to backup models")
    print("✅ Configurable retry logic with exponential backoff") 
    print("✅ Health monitoring for all configured models")
    print("✅ Flexible configuration via Python or environment variables")
    print("✅ Support for both text generation and embedding models")
    print("✅ Extensible provider system for different model backends")
    
    # 5. Usage Example
    print("\n📝 5. Usage Example (pseudo-code)")
    print("""
# Initialize with backup configuration
manager = BackupModelManager(config)
await manager.initialize()

# Generate text with automatic backup
response = await manager.generate_with_backup("Hello, how are you?")
print(response.content)  # Uses HF model if primary fails

# Generate embeddings with automatic backup  
embeddings = await manager.embed_with_backup("Document text")
print(len(embeddings))  # Uses HF embedding model if primary fails
""")
    
    print("\n🎉 Demo completed! To use with actual models:")
    print("   pip install transformers torch sentence-transformers")
    print("   python examples/hf_backup_demo.py")


if __name__ == "__main__":
    asyncio.run(demo_framework_structure())