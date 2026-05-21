"""Tests for apa.lore_adapt — few-shot adaptation and scoring."""

import json

import pytest
import torch

from apa.lore_adapt import (
    adapt_users,
    evaluate_adapted,
    load_adapted,
    LoReScorer,
    save_adapted,
    split_train_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 32
K = 4


@pytest.fixture
def V():
    """Small random basis matrix [DIM, K]."""
    torch.manual_seed(0)
    return torch.randn(DIM, K)


@pytest.fixture
def user_embeddings():
    """Per-user diff tensors for 3 users with 10, 6, and 2 prefs."""
    torch.manual_seed(1)
    return [
        torch.randn(10, DIM),
        torch.randn(6, DIM),
        torch.randn(2, DIM),
    ]


@pytest.fixture
def user_ids():
    return ["user_a", "user_b", "user_c"]


# ---------------------------------------------------------------------------
# split_train_test
# ---------------------------------------------------------------------------


class TestSplitTrainTest:
    def test_preserves_total_count(self, user_embeddings):
        train, test = split_train_test(user_embeddings, test_frac=0.2, seed=42)
        for i, X in enumerate(user_embeddings):
            n_train = len(train[i])
            n_test = len(test[i]) if test[i] is not None else 0
            assert n_train + n_test == len(X)

    def test_small_user_gets_no_test(self, user_embeddings):
        """User with 2 prefs (< min_total=4) should have test=None."""
        train, test = split_train_test(user_embeddings, test_frac=0.2, min_total=4)
        assert test[2] is None
        assert len(train[2]) == 2

    def test_large_user_gets_test(self, user_embeddings):
        """User with 10 prefs should have test data."""
        train, test = split_train_test(user_embeddings, test_frac=0.2, min_total=4)
        assert test[0] is not None
        assert len(test[0]) >= 1

    def test_reproducible(self, user_embeddings):
        t1, _ = split_train_test(user_embeddings, seed=123)
        t2, _ = split_train_test(user_embeddings, seed=123)
        for a, b in zip(t1, t2):
            assert torch.equal(a, b)

    def test_different_seeds_differ(self, user_embeddings):
        t1, _ = split_train_test(user_embeddings, seed=1)
        t2, _ = split_train_test(user_embeddings, seed=2)
        # At least the first user (10 prefs) should have different order
        assert not torch.equal(t1[0], t2[0])


# ---------------------------------------------------------------------------
# adapt_users
# ---------------------------------------------------------------------------


class TestAdaptUsers:
    def test_returns_correct_keys(self, user_ids, user_embeddings, V):
        train, _ = split_train_test(user_embeddings, test_frac=0.0, min_total=100)
        results = adapt_users(user_ids, train, V, num_iterations=5, learning_rate=0.5)
        assert set(results.keys()) == set(user_ids)

    def test_w_shape(self, user_ids, user_embeddings, V):
        train, _ = split_train_test(user_embeddings, test_frac=0.0, min_total=100)
        results = adapt_users(user_ids, train, V, num_iterations=5, learning_rate=0.5)
        for uid in user_ids:
            assert results[uid]['w'].shape == (K,)

    def test_n_train_prefs(self, user_ids, user_embeddings, V):
        train, _ = split_train_test(user_embeddings, test_frac=0.0, min_total=100)
        results = adapt_users(user_ids, train, V, num_iterations=5, learning_rate=0.5)
        assert results["user_a"]['n_train_prefs'] == 10
        assert results["user_b"]['n_train_prefs'] == 6
        assert results["user_c"]['n_train_prefs'] == 2


# ---------------------------------------------------------------------------
# evaluate_adapted
# ---------------------------------------------------------------------------


class TestEvaluateAdapted:
    def test_accuracy_in_range(self, user_ids, user_embeddings, V):
        train, test = split_train_test(user_embeddings, test_frac=0.2, min_total=4)
        results = adapt_users(user_ids, train, V, num_iterations=10, learning_rate=0.5)
        stats = evaluate_adapted(results, user_ids, train, test, V)

        for uid in user_ids:
            assert 0.0 <= results[uid]['train_accuracy'] <= 1.0

        assert 0.0 <= stats['train_accuracy_mean'] <= 1.0

    def test_test_accuracy_only_for_users_with_test(self, user_ids, user_embeddings, V):
        train, test = split_train_test(user_embeddings, test_frac=0.2, min_total=4)
        results = adapt_users(user_ids, train, V, num_iterations=10, learning_rate=0.5)
        evaluate_adapted(results, user_ids, train, test, V)

        # user_c has 2 prefs < min_total=4, so no test
        assert 'test_accuracy' not in results["user_c"]
        # user_a has 10 prefs, should have test
        assert 'test_accuracy' in results["user_a"]


# ---------------------------------------------------------------------------
# save_adapted / load_adapted
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self, tmp_path, user_ids, user_embeddings, V):
        train, _ = split_train_test(user_embeddings, test_frac=0.0, min_total=100)
        results = adapt_users(user_ids, train, V, num_iterations=5, learning_rate=0.5)

        path = tmp_path / "test_checkpoint.pt"
        save_adapted(results, path, metadata={'K': K, 'source': 'test'})

        loaded = load_adapted(path)
        assert 'users' in loaded
        assert 'metadata' in loaded
        assert loaded['metadata']['K'] == K

        for uid in user_ids:
            assert torch.allclose(loaded['users'][uid]['w'], results[uid]['w'])

    def test_creates_parent_dirs(self, tmp_path, user_ids, user_embeddings, V):
        train, _ = split_train_test(user_embeddings, test_frac=0.0, min_total=100)
        results = adapt_users(user_ids, train, V, num_iterations=5, learning_rate=0.5)

        path = tmp_path / "nested" / "dir" / "checkpoint.pt"
        save_adapted(results, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# LoReScorer
# ---------------------------------------------------------------------------


class TestLoReScorer:
    def test_score_embedding_correct(self, V):
        scorer = LoReScorer(V)
        w = torch.randn(K)
        scorer.user_registry["test_user"] = w

        embedding = torch.randn(DIM)
        expected = float(embedding @ V.float() @ w.float())
        actual = scorer.score_embedding("test_user", embedding)
        assert abs(actual - expected) < 1e-5

    def test_missing_user_raises(self, V):
        scorer = LoReScorer(V)
        with pytest.raises(KeyError, match="not found"):
            scorer.score_embedding("nonexistent", torch.randn(DIM))

    def test_has_user(self, V):
        scorer = LoReScorer(V)
        scorer.user_registry["u1"] = torch.randn(K)
        assert scorer.has_user("u1")
        assert not scorer.has_user("u2")

    def test_get_user_ids(self, V):
        scorer = LoReScorer(V)
        scorer.user_registry["u1"] = torch.randn(K)
        scorer.user_registry["u2"] = torch.randn(K)
        assert sorted(scorer.get_user_ids()) == ["u1", "u2"]


class TestLoReScorerLoadAdapted:
    def test_load_adapted_users(self, tmp_path, V, user_ids, user_embeddings):
        train, _ = split_train_test(user_embeddings, test_frac=0.0, min_total=100)
        results = adapt_users(user_ids, train, V, num_iterations=5, learning_rate=0.5)

        path = tmp_path / "W_adapted_test.pt"
        save_adapted(results, path)

        scorer = LoReScorer(V)
        count = scorer.load_adapted_users(path)
        assert count == len(user_ids)
        for uid in user_ids:
            assert scorer.has_user(uid)
            assert scorer.user_registry[uid].shape == (K,)


class TestLoReScorerLoadPrism:
    def test_load_prism_positional(self, tmp_path, V):
        W = torch.randn(5, K)
        w_path = tmp_path / "W_seen.pt"
        torch.save(W, w_path)

        scorer = LoReScorer(V)
        count = scorer.load_prism_users(w_path)
        assert count == 5
        for i in range(5):
            assert scorer.has_user(f"prism_user_{i}")
            assert torch.allclose(scorer.user_registry[f"prism_user_{i}"], W[i].float())

    def test_load_prism_with_mapping(self, tmp_path, V):
        W = torch.randn(3, K)
        w_path = tmp_path / "W_seen.pt"
        torch.save(W, w_path)

        mapping = {"alice": 0, "bob": 1, "carol": 2}
        mapping_path = tmp_path / "user_to_idx.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f)

        scorer = LoReScorer(V)
        scorer.load_prism_users(w_path, mapping_path)
        assert scorer.has_user("alice")
        assert scorer.has_user("bob")
        assert scorer.has_user("carol")
