# Agentic Reasoning Framework

A LangGraph-based framework to reason over documents with automatic HuggingFace model backup support.

## Features

- **Primary/Backup Model Architecture**: Configure primary models (OpenAI, etc.) with automatic fallback to HuggingFace models
- **Automatic Failover**: Seamless switching to backup models when primary models fail
- **Retry Logic**: Configurable retry attempts with exponential backoff
- **Health Monitoring**: Built-in health checks for all configured models
- **Flexible Configuration**: YAML/Environment variable configuration support

## Quick Start

### Installation

```bash
pip install poetry
poetry install
```

### Basic Usage

```python
import asyncio
from agentic_reasoning.config import FrameworkConfig, ModelConfig
from agentic_reasoning.backup_manager import BackupModelManager

async def main():
    # Configure with HuggingFace backups
    config = FrameworkConfig(
        backup_llm=[
            ModelConfig(
                name="microsoft/DialoGPT-small",
                provider="huggingface",
                model_kwargs={"max_length": 100},
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
    
    # Initialize backup manager
    manager = BackupModelManager(config)
    await manager.initialize()
    
    # Generate text with automatic backup
    response = await manager.generate_with_backup("Hello, how are you?")
    print(f"Response: {response.content}")
    
    # Generate embeddings with automatic backup
    embeddings = await manager.embed_with_backup("This is a test.")
    print(f"Embeddings: {embeddings[:5]}...")

asyncio.run(main())
```

### Configuration

The framework supports both Python configuration and environment variables:

```python
# Python configuration
config = FrameworkConfig(
    primary_llm=ModelConfig(
        name="gpt-3.5-turbo",
        provider="openai",
        enabled=True
    ),
    backup_llm=[
        ModelConfig(
            name="microsoft/DialoGPT-medium",
            provider="huggingface",
            model_kwargs={"max_length": 1000, "temperature": 0.7},
            enabled=True
        )
    ],
    backup_config=BackupConfig(
        max_retries=3,
        retry_delay=1.0,
        fallback_enabled=True,
        timeout=30.0
    )
)
```

Environment variables:
```bash
ARF_PRIMARY_LLM__NAME=gpt-4
ARF_PRIMARY_LLM__PROVIDER=openai
ARF_BACKUP_CONFIG__MAX_RETRIES=5
```

## HuggingFace Models as Backup

The framework automatically falls back to HuggingFace models when primary models fail:

### Supported Model Types

1. **Text Generation Models**:
   - microsoft/DialoGPT-medium
   - facebook/blenderbot-400M-distill
   - microsoft/DialoGPT-small

2. **Embedding Models**:
   - sentence-transformers/all-MiniLM-L6-v2
   - sentence-transformers/all-mpnet-base-v2

### Fallback Logic

1. Try primary model with retries
2. If primary fails, try each backup model in order
3. Each backup model has its own retry logic
4. Comprehensive error logging and health monitoring

## Examples

Run the demo:
```bash
python examples/hf_backup_demo.py
```

## Testing

```bash
poetry run pytest tests/
```

## Short-Term Planned Features

1. **Integrate with OCR and RAG modules**
2. **Add orchestrator and sub-agents**
3. **Human-in-the-loop interrupts**
4. **Memory and persistence**
5. **Add evaluations and tracings**