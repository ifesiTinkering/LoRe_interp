"""
Unit tests for configuration module.
"""

import pytest
from pathlib import Path

from apa.config import (
    APAConfig,
    DatasetConfig,
    InferenceConfig,
    LoReConfig,
    InferenceLLMConfig,
)


class TestDatasetConfig:
    """Tests for DatasetConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DatasetConfig()

        assert config.name == "prism"
        assert "prism" in str(config.questions_pairwise_path)

    def test_path_properties(self):
        """Test path property returns Path objects."""
        config = DatasetConfig()

        assert isinstance(config.questions_pairwise_path, Path)
        assert isinstance(config.embeddings_dir, Path)
        assert isinstance(config.models_dir, Path)


class TestLoReConfig:
    """Tests for LoReConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoReConfig()

        assert config.K_list == [0, 1]
        assert config.alpha == 10000.0
        assert config.num_iterations == 20000
        assert config.learning_rate == 0.5
        assert config.logits_scale == 100.0
        assert config.threshold == 1e-2
        assert config.few_shot_iterations == 500
        assert config.few_shot_lr == 0.5
        assert config.embedding_dim == 4096
        assert config.log_interval == 2000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LoReConfig(K_list=[0, 1, 5], alpha=5000.0, num_iterations=10000)

        assert config.K_list == [0, 1, 5]
        assert config.alpha == 5000.0
        assert config.num_iterations == 10000


class TestInferenceConfig:
    """Tests for InferenceConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InferenceConfig()

        assert config.k_responses == 5
        assert config.m_voters == 10
        assert config.aggregate_strategy == "borda_count"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = InferenceConfig(
            k_responses=10,
            m_voters=20,
            aggregate_strategy="plurality",
        )

        assert config.k_responses == 10
        assert config.m_voters == 20
        assert config.aggregate_strategy == "plurality"


class TestInferenceLLMConfig:
    """Tests for InferenceLLMConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InferenceLLMConfig()

        # Model name should be a valid HuggingFace model
        assert "/" in config.model_name  # Format: org/model
        assert config.max_new_tokens == 512
        assert config.temperature == 1.2
        assert config.do_sample is True


class TestAPAConfig:
    """Tests for APAConfig class."""

    def test_default_values(self):
        """Test default configuration creates nested configs."""
        config = APAConfig()

        assert isinstance(config.dataset, DatasetConfig)
        assert isinstance(config.inference_llm, InferenceLLMConfig)
        assert isinstance(config.lore, LoReConfig)
        assert isinstance(config.inference, InferenceConfig)

    def test_historical_centuries(self):
        """Test historical centuries default."""
        config = APAConfig()

        assert "C013" in config.historical_centuries
        assert "C021" in config.historical_centuries
