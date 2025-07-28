#!/usr/bin/env python3
"""CLI tool for testing HuggingFace backup functionality."""

import asyncio
import argparse
import logging
import sys
from agentic_reasoning.config import FrameworkConfig, ModelConfig
from agentic_reasoning.backup_manager import BackupModelManager


async def test_text_generation(manager: BackupModelManager, prompt: str):
    """Test text generation with backup models."""
    print(f"🤖 Generating response for: '{prompt}'")
    try:
        response = await manager.generate_with_backup(prompt)
        print(f"✅ Response: {response.content}")
        print(f"📝 Model: {response.model_name} (provider: {response.provider})")
        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False


async def test_embeddings(manager: BackupModelManager, text: str):
    """Test embeddings with backup models."""
    print(f"🔢 Generating embeddings for: '{text}'")
    try:
        embeddings = await manager.embed_with_backup(text)
        print(f"✅ Embeddings generated (dimensions: {len(embeddings)})")
        print(f"📊 First 5 values: {embeddings[:5]}")
        return True
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return False


async def test_health_check(manager: BackupModelManager):
    """Test health check functionality."""
    print("🔍 Running health checks...")
    try:
        health_results = await manager.health_check_all()
        print("📋 Health Check Results:")
        for model, status in health_results.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {model}")
        return all(health_results.values())
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Test HuggingFace backup functionality")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Text generation prompt")
    parser.add_argument("--text", default="This is a test sentence.", help="Text for embedding")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--health-only", action="store_true", help="Only run health checks")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("🚀 Starting HuggingFace Backup CLI Test")
    print("=" * 50)
    
    # Configure with HuggingFace-only backups for testing
    config = FrameworkConfig(
        # Disable primary models to force backup usage
        primary_llm=ModelConfig(name="disabled", provider="none", enabled=False),
        primary_embeddings=ModelConfig(name="disabled", provider="none", enabled=False),
        
        # Use lightweight HuggingFace models for testing
        backup_llm=[
            ModelConfig(
                name="microsoft/DialoGPT-small",
                provider="huggingface",
                model_kwargs={
                    "max_length": 100,
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
                model_kwargs={"model_type": "embedding"},
                enabled=True
            )
        ]
    )
    
    # Initialize manager
    print("⚙️  Initializing backup model manager...")
    manager = BackupModelManager(config)
    
    try:
        await manager.initialize()
        print("✅ Manager initialized successfully!")
        
        # Run tests
        results = []
        
        # Health check (always run)
        results.append(await test_health_check(manager))
        
        if not args.health_only:
            print("\n" + "=" * 50)
            results.append(await test_text_generation(manager, args.prompt))
            
            print("\n" + "=" * 50)
            results.append(await test_embeddings(manager, args.text))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Summary:")
        success_count = sum(results)
        total_count = len(results)
        
        if success_count == total_count:
            print(f"🎉 All tests passed! ({success_count}/{total_count})")
            sys.exit(0)
        else:
            print(f"⚠️  Some tests failed ({success_count}/{total_count})")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())