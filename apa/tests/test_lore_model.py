"""
Unit tests for LoRe reward model.
"""

import pytest
import torch

from apa.train_lore_bases import LoReRewardModel, LoReTrainer


class TestLoReRewardModel:
    """Tests for LoReRewardModel class."""

    def test_init(self):
        """Test model initialization."""
        V = torch.randn(768, 8)
        model = LoReRewardModel(V)

        assert model.embedding_dim == 768
        assert model.rank == 8
        assert torch.equal(model.V, V)

    def test_score_single(self):
        """Test scoring a single embedding."""
        V = torch.randn(32, 4)
        model = LoReRewardModel(V)

        embedding = torch.randn(32)
        w = torch.randn(4)

        score = model.score(embedding, w)

        # Score should be a scalar
        assert score.shape == ()

        # Verify the computation: embedding @ V @ w
        expected = embedding @ V @ w
        assert torch.allclose(score, expected)

    def test_score_batch(self):
        """Test scoring a batch of embeddings."""
        V = torch.randn(64, 8)
        model = LoReRewardModel(V)

        batch_size = 16
        embeddings = torch.randn(batch_size, 64)
        w = torch.randn(8)

        scores = model.score(embeddings, w)

        assert scores.shape == (batch_size,)

    def test_rank_property(self):
        """Test rank property."""
        for rank in [1, 4, 8, 16]:
            V = torch.randn(32, rank)
            model = LoReRewardModel(V)
            assert model.rank == rank

    def test_embedding_dim_property(self):
        """Test embedding_dim property."""
        for dim in [32, 64, 768, 4096]:
            V = torch.randn(dim, 4)
            model = LoReRewardModel(V)
            assert model.embedding_dim == dim

    def test_save_load(self, tmp_path):
        """Test saving and loading model."""
        V = torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4)
        model = LoReRewardModel(V)

        path = str(tmp_path / "test_model.pt")
        torch.save(model.V, path)

        loaded = LoReRewardModel.load(path)

        assert loaded.embedding_dim == model.embedding_dim
        assert loaded.rank == model.rank
        assert torch.allclose(loaded.V, model.V)

    def test_load_from_dict(self, tmp_path):
        """Test loading from checkpoint dict with 'V' key."""
        V = torch.randn(32, 4)

        path = str(tmp_path / "test_model.pt")
        torch.save({'V': V, 'other': 'data'}, path)

        loaded = LoReRewardModel.load(path)

        assert torch.allclose(loaded.V, V)


class TestLoReTrainer:
    """Tests for LoReTrainer class."""

    def test_init(self):
        """Test trainer initialization."""
        V_sft = torch.randn(32, 1)

        trainer = LoReTrainer(
            V_sft=V_sft,
            alpha=10000.0,
            num_classes=10,
            num_features=32,
            num_basis_vectors=4,
            num_iterations=100,
            learning_rate=0.5,
        )

        assert trainer.alpha == 10000.0
        assert trainer.num_classes == 10
        assert trainer.num_features == 32
        assert trainer.num_basis_vectors == 4
        assert trainer.num_iterations == 100
        assert trainer.learning_rate == 0.5
        assert trainer.logits_scale == 100.0
        assert trainer.threshold == 1e-2

    def test_init_custom_params(self):
        """Test trainer initialization with custom parameters."""
        V_sft = torch.randn(64, 1)

        trainer = LoReTrainer(
            V_sft=V_sft,
            alpha=5000.0,
            num_classes=20,
            num_features=64,
            num_basis_vectors=8,
            num_iterations=500,
            learning_rate=0.1,
            logits_scale=50.0,
            threshold=1e-3,
            log_interval=100,
        )

        assert trainer.alpha == 5000.0
        assert trainer.num_classes == 20
        assert trainer.num_features == 64
        assert trainer.num_basis_vectors == 8
        assert trainer.logits_scale == 50.0
        assert trainer.threshold == 1e-3
        assert trainer.log_interval == 100

    def test_v_sft_normalized(self):
        """Test that V_sft is normalized during initialization."""
        V_sft = torch.randn(32, 1) * 10  # Large magnitude

        trainer = LoReTrainer(
            V_sft=V_sft,
            alpha=10000.0,
            num_classes=5,
            num_features=32,
            num_basis_vectors=4,
            num_iterations=100,
            learning_rate=0.5,
        )

        # V_sft_norm should be unit norm
        norm = torch.norm(trainer.V_sft_norm, dim=0)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_training_history_init(self):
        """Test that training history is initialized."""
        V_sft = torch.randn(32, 1)

        trainer = LoReTrainer(
            V_sft=V_sft,
            alpha=10000.0,
            num_classes=5,
            num_features=32,
            num_basis_vectors=4,
            num_iterations=100,
            learning_rate=0.5,
        )

        assert hasattr(trainer, 'training_history')
        assert isinstance(trainer.training_history, dict)
