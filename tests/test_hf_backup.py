"""Tests for HuggingFace backup functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from agentic_reasoning.config import FrameworkConfig, ModelConfig
from agentic_reasoning.backup_manager import BackupModelManager
from agentic_reasoning.model_providers import HuggingFaceProvider, ModelResponse


class TestBackupModelManager:
    """Tests for BackupModelManager."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return FrameworkConfig(
            primary_llm=ModelConfig(name="disabled", provider="none", enabled=False),
            primary_embeddings=ModelConfig(name="disabled", provider="none", enabled=False),
            backup_llm=[
                ModelConfig(
                    name="test-model",
                    provider="huggingface",
                    model_kwargs={"model_type": "text-generation"},
                    enabled=True
                )
            ],
            backup_embeddings=[
                ModelConfig(
                    name="test-embedding-model",
                    provider="huggingface",
                    model_kwargs={"model_type": "embedding"},
                    enabled=True
                )
            ]
        )
    
    @pytest.fixture
    def manager(self, config):
        """Create test manager."""
        return BackupModelManager(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, manager):
        """Test manager initialization."""
        with patch.object(HuggingFaceProvider, 'initialize', new_callable=AsyncMock):
            await manager.initialize()
            assert manager._initialized
            assert len(manager.backup_llms) == 1
            assert len(manager.backup_embeddings) == 1
    
    @pytest.mark.asyncio
    async def test_generate_with_backup_success(self, manager):
        """Test successful generation with backup model."""
        mock_response = ModelResponse(
            content="test response",
            model_name="test-model",
            provider="huggingface"
        )
        
        with patch.object(HuggingFaceProvider, 'initialize', new_callable=AsyncMock), \
             patch.object(HuggingFaceProvider, 'generate', new_callable=AsyncMock, return_value=mock_response):
            
            response = await manager.generate_with_backup("test prompt")
            assert response.content == "test response"
            assert response.provider == "huggingface"
    
    @pytest.mark.asyncio
    async def test_embed_with_backup_success(self, manager):
        """Test successful embedding with backup model."""
        mock_embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        with patch.object(HuggingFaceProvider, 'initialize', new_callable=AsyncMock), \
             patch.object(HuggingFaceProvider, 'embed', new_callable=AsyncMock, return_value=mock_embeddings):
            
            embeddings = await manager.embed_with_backup("test text")
            assert embeddings == mock_embeddings
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, manager):
        """Test health check functionality."""
        with patch.object(HuggingFaceProvider, 'initialize', new_callable=AsyncMock), \
             patch.object(HuggingFaceProvider, 'health_check', new_callable=AsyncMock, return_value=True):
            
            health_results = await manager.health_check_all()
            assert len(health_results) == 2  # One backup LLM + one backup embedding
            assert all(status for status in health_results.values())


class TestHuggingFaceProvider:
    """Tests for HuggingFaceProvider."""
    
    @pytest.fixture
    def provider(self):
        """Create test provider."""
        return HuggingFaceProvider(
            model_name="test-model",
            model_type="text-generation",
            max_length=50
        )
    
    def test_provider_initialization(self, provider):
        """Test provider initialization parameters."""
        assert provider.model_name == "test-model"
        assert provider.model_type == "text-generation"
        assert provider.max_length == 50
        assert not provider._initialized
    
    def test_get_provider_name(self, provider):
        """Test provider name."""
        assert provider.get_provider_name() == "huggingface"
    
    @pytest.mark.asyncio
    async def test_is_available_not_initialized(self, provider):
        """Test availability check when not initialized."""
        with patch.object(provider, 'initialize', new_callable=AsyncMock, side_effect=Exception("Test error")):
            result = await provider.is_available()
            assert not result


class TestFrameworkConfig:
    """Tests for FrameworkConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FrameworkConfig()
        assert config.primary_llm.provider == "openai"
        assert config.primary_embeddings.provider == "openai"
        assert len(config.backup_llm) == 2
        assert len(config.backup_embeddings) == 2
        assert all(backup.provider == "huggingface" for backup in config.backup_llm)
        assert all(backup.provider == "huggingface" for backup in config.backup_embeddings)
    
    def test_backup_config_defaults(self):
        """Test backup configuration defaults."""
        config = FrameworkConfig()
        assert config.backup_config.max_retries == 3
        assert config.backup_config.retry_delay == 1.0
        assert config.backup_config.fallback_enabled is True
        assert config.backup_config.timeout == 30.0