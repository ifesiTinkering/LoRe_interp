"""
Tests for LoRe dataset suitability evaluation metrics.

Fast unit tests use small synthetic tensors and run in seconds.
The PRISM benchmark (marked slow) loads pre-computed PRISM embeddings
and verifies that all metrics return "green zone" values — confirming
that a dataset known to work well with LoRe is correctly identified as such.

Usage:
    pytest tests/test_suitability.py -v            # fast unit tests only
    pytest tests/test_suitability.py -v -m slow    # PRISM benchmark
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.synthetic_prefs.eval_prefs import (
    annotation_density,
    basis_space_coherence,
    basis_utilization_entropy,
    evaluate_suitability,
    fit_user_vectors,
    held_out_accuracy,
    inter_user_agreement,
    krippendorff_alpha_proxy,
    label_balance,
    nearest_neighbor_accuracy,
    population_accuracy,
    user_vector_diversity,
)


# ---------------------------------------------------------------------------
# Helpers for synthetic data
# ---------------------------------------------------------------------------

def _make_random_users(n_users: int, n_pairs: int, D: int) -> list[torch.Tensor]:
    """Entirely random preference vectors — no learnable structure."""
    return [torch.randn(n_pairs, D) for _ in range(n_users)]


def _make_consistent_user(n_pairs: int, D: int) -> torch.Tensor:
    """All preferences pointing in the same direction (rubber-stamper)."""
    direction = torch.randn(D)
    direction = direction / direction.norm()
    noise = torch.randn(n_pairs, D) * 0.05
    return direction.unsqueeze(0).expand(n_pairs, -1) + noise


def _make_low_rank_users(n_users: int, n_pairs: int, D: int, true_rank: int) -> list[torch.Tensor]:
    """Users whose preferences live in a true_rank-dimensional subspace."""
    subspace = torch.randn(D, true_rank)
    subspace = torch.linalg.qr(subspace)[0]  # orthonormal basis [D, true_rank]
    tensors = []
    for _ in range(n_users):
        coeffs = torch.randn(n_pairs, true_rank)
        tensors.append(coeffs @ subspace.T + torch.randn(n_pairs, D) * 0.05)
    return tensors


def _make_distinct_users(n_users: int, n_pairs: int, D: int) -> list[torch.Tensor]:
    """Users with clearly distinct preference directions."""
    users = []
    for _ in range(n_users):
        direction = torch.randn(D)
        direction = direction / direction.norm()
        noise = torch.randn(n_pairs, D) * 0.1
        users.append(direction.unsqueeze(0).expand(n_pairs, -1).clone() + noise)
    return users


# ---------------------------------------------------------------------------
# annotation_density
# ---------------------------------------------------------------------------

class TestAnnotationDensity:

    def test_no_warning_when_sufficient(self):
        K = 4
        X = [torch.randn(10, 32) for _ in range(5)]
        result = annotation_density(X, K)
        assert result["warn"] is False
        assert result["median_pairs"] == 10.0

    def test_warns_when_sparse(self):
        K = 8
        X = [torch.randn(1, 32)]  # Only 1 pair, need 16
        result = annotation_density(X, K)
        assert result["warn"] is True
        assert result["fraction_below_2K"] == 1.0

    def test_mixed_users(self):
        K = 4
        X = [torch.randn(1, 32), torch.randn(20, 32)]
        result = annotation_density(X, K)
        assert result["min_pairs"] == 1
        assert result["fraction_below_2K"] == 0.5  # one user below 2*K=8


# ---------------------------------------------------------------------------
# label_balance
# ---------------------------------------------------------------------------

class TestLabelBalance:

    def test_random_user_normalized_near_one(self):
        # For random iid vectors, normalized_consistency = raw * sqrt(n) ≈ 1.0
        rng = torch.Generator()
        rng.manual_seed(0)
        X = torch.randn(500, 64, generator=rng)
        result = label_balance([X])
        nc = result["mean_normalized_consistency"]
        assert 0.5 <= nc <= 2.0, f"normalized_consistency for random data should be ≈1, got {nc}"

    def test_consistent_user_scores_above_random(self):
        # Consistent user's normalized score should be well above 1.0
        X = _make_consistent_user(n_pairs=50, D=64)
        result = label_balance([X])
        nc = result["mean_normalized_consistency"]
        assert nc > 2.0, f"consistent user should have normalized_consistency >> 1, got {nc}"

    def test_consistent_user_scores_higher_than_random(self):
        consistent = _make_consistent_user(50, 64)
        rng = torch.Generator()
        rng.manual_seed(1)
        random_user = torch.randn(50, 64, generator=rng)
        result = label_balance([consistent, random_user])
        per = result["per_user_normalized"]
        assert per[0] > per[1], "consistent user should score higher than random user"


# ---------------------------------------------------------------------------
# inter_user_agreement
# ---------------------------------------------------------------------------

class TestInterUserAgreement:

    def test_identical_users_have_high_agreement(self):
        direction = torch.randn(32)
        X = [direction.unsqueeze(0).expand(5, -1) + torch.randn(5, 32) * 0.01
             for _ in range(10)]
        result = inter_user_agreement(X)
        assert result["mean_pairwise_similarity"] > 0.8

    def test_random_users_have_low_agreement(self):
        X = _make_random_users(n_users=50, n_pairs=20, D=64)
        result = inter_user_agreement(X)
        assert abs(result["mean_pairwise_similarity"]) < 0.3


# ---------------------------------------------------------------------------
# krippendorff_alpha_proxy
# ---------------------------------------------------------------------------

class TestKrippendorffProxy:

    def test_distinct_users_have_positive_corrected_ratio(self):
        # Users in clearly distinct subspaces
        X = []
        for i in range(10):
            direction = torch.zeros(64)
            direction[i * 6 % 64] = 1.0
            X.append(direction.unsqueeze(0).expand(10, -1) + torch.randn(10, 64) * 0.1)
        result = krippendorff_alpha_proxy(X)
        assert result["corrected_ratio"] > 0.01, (
            f"corrected_ratio={result['corrected_ratio']:.4f}, expected > 0.01 for distinct users"
        )

    def test_identical_users_have_near_zero_corrected_ratio(self):
        # All users have the same preference direction → sampling noise only
        direction = torch.randn(32)
        X = [direction.unsqueeze(0).expand(8, -1).clone() for _ in range(10)]
        result = krippendorff_alpha_proxy(X)
        assert result["corrected_ratio"] < 0.05, (
            f"corrected_ratio={result['corrected_ratio']:.4f}, expected ≈0 for identical users"
        )

    def test_random_data_corrected_near_zero(self):
        # Random data: corrected_ratio should be close to 0 (noise subtracted)
        rng = torch.Generator()
        rng.manual_seed(99)
        X = [torch.randn(20, 32, generator=rng) for _ in range(50)]
        result = krippendorff_alpha_proxy(X)
        assert -0.05 <= result["corrected_ratio"] <= 0.05, (
            f"corrected_ratio={result['corrected_ratio']:.4f} for random data should be ≈0"
        )

    def test_raw_ratio_in_01(self):
        X = _make_random_users(20, 10, 32)
        result = krippendorff_alpha_proxy(X)
        assert 0.0 <= result["raw_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# nearest_neighbor_accuracy
# ---------------------------------------------------------------------------

class TestNearestNeighborAccuracy:

    def test_random_data_gives_near_chance_accuracy(self):
        rng = torch.Generator()
        rng.manual_seed(42)
        # Use D=512 (closer to real D=4096) to avoid NN selection bias
        # that inflates accuracy in low-D spaces
        X = [torch.randn(30, 512, generator=rng) for _ in range(80)]
        result = nearest_neighbor_accuracy(X)
        acc = result["mean_nn_accuracy"]
        assert 0.35 <= acc <= 0.65, (
            f"random data NN accuracy should be ≈0.5, got {acc:.3f}"
        )

    def test_structured_data_gives_above_chance_accuracy(self):
        # Users with distinct, consistent directions: similar users share preferences
        X = _make_distinct_users(n_users=40, n_pairs=20, D=64)
        result = nearest_neighbor_accuracy(X)
        acc = result["mean_nn_accuracy"]
        assert acc > 0.55, f"structured data NN accuracy should be > 0.55, got {acc:.3f}"

    def test_does_not_require_V(self):
        # Should work with no V argument
        X = _make_random_users(10, 5, 32)
        result = nearest_neighbor_accuracy(X)
        assert "mean_nn_accuracy" in result
        assert "std_nn_accuracy" in result


# ---------------------------------------------------------------------------
# basis_space_coherence
# ---------------------------------------------------------------------------

class TestBasisSpaceCoherence:

    def test_distinct_users_have_positive_corrected_ratio(self):
        D, K = 64, 4
        # V aligned with distinct user directions
        V = torch.eye(D)[:, :K]
        X = []
        for i in range(20):
            direction = torch.zeros(D)
            direction[i % K] = 1.0  # each user maps to a different basis direction
            X.append(direction.unsqueeze(0).expand(10, -1).clone() + torch.randn(10, D) * 0.1)
        result = basis_space_coherence(X, V)
        assert result["corrected_ratio"] > 0.01, (
            f"corrected_ratio={result['corrected_ratio']:.4f} for structured data"
        )

    def test_random_data_corrected_near_zero(self):
        rng = torch.Generator()
        rng.manual_seed(7)
        D, K = 32, 4
        V = torch.randn(D, K, generator=rng)
        X = [torch.randn(20, D, generator=rng) for _ in range(50)]
        result = basis_space_coherence(X, V)
        assert -0.05 <= result["corrected_ratio"] <= 0.05, (
            f"corrected_ratio={result['corrected_ratio']:.4f} for random data should be ≈0"
        )


# ---------------------------------------------------------------------------
# population_accuracy
# ---------------------------------------------------------------------------

class TestPopulationAccuracy:

    def test_random_data_gives_near_chance_accuracy(self):
        rng = torch.Generator()
        rng.manual_seed(5)
        D, K = 64, 8
        V = torch.randn(D, K, generator=rng)
        X = [torch.randn(20, D, generator=rng) for _ in range(50)]
        result = population_accuracy(X, V)
        assert 0.35 <= result["accuracy"] <= 0.65, (
            f"random data population accuracy should be ≈0.5, got {result['accuracy']:.3f}"
        )

    def test_structured_data_gives_above_chance_accuracy(self):
        D, K = 32, 4
        # All users share one basis direction → pooled V captures it
        V = torch.eye(D)[:, :K]
        direction = V[:, 0]
        X = [direction.unsqueeze(0).expand(20, -1).clone() + torch.randn(20, D) * 0.1
             for _ in range(10)]
        result = population_accuracy(X, V)
        assert result["accuracy"] > 0.7, (
            f"structured data population accuracy should be > 0.7, got {result['accuracy']:.3f}"
        )


# ---------------------------------------------------------------------------
# user_vector_diversity and basis_utilization_entropy
# ---------------------------------------------------------------------------

class TestUserVectorDiversity:

    def test_identical_vectors_have_zero_distance(self):
        W = torch.zeros(10, 4)  # all the same → after softmax, all uniform
        result = user_vector_diversity(W)
        assert result["mean_pairwise_distance"] < 0.05

    def test_diverse_vectors_have_high_distance(self):
        K = 4
        # Each user concentrates all weight on a different basis
        W = torch.full((K, K), -10.0)
        W.fill_diagonal_(10.0)  # one-hot after softmax
        result = user_vector_diversity(W)
        assert result["mean_pairwise_distance"] > 0.5


class TestBasisUtilizationEntropy:

    def test_uniform_weights_give_max_entropy(self):
        W = torch.zeros(10, 4)  # after softmax: uniform
        result = basis_utilization_entropy(W)
        assert abs(result["normalized_mean_entropy"] - 1.0) < 0.01

    def test_one_hot_weights_give_low_entropy(self):
        K = 4
        W = torch.full((10, K), -100.0)
        W[:, 0] = 100.0  # all weight on basis 0
        result = basis_utilization_entropy(W)
        assert result["normalized_mean_entropy"] < 0.1


# ---------------------------------------------------------------------------
# held_out_accuracy
# ---------------------------------------------------------------------------

class TestHeldOutAccuracy:

    def test_random_data_gives_near_chance_accuracy(self):
        rng = torch.Generator()
        rng.manual_seed(7)
        D, K = 64, 8
        V = torch.randn(D, K, generator=rng)
        X = [torch.randn(20, D, generator=rng) for _ in range(40)]
        result = held_out_accuracy(X, V)
        # Should be near 0.5 for random data (no generalisation)
        assert 0.3 <= result["mean_accuracy"] <= 0.7, result["mean_accuracy"]

    def test_skips_users_with_too_few_pairs(self):
        D, K = 16, 2
        V = torch.randn(D, K)
        X = [torch.randn(3, D), torch.randn(20, D)]  # first user skipped (< 4)
        result = held_out_accuracy(X, V)
        assert result["n_users_evaluated"] == 1

    def test_accuracy_in_01(self):
        V = torch.randn(16, 4)
        X = [torch.randn(10, 16) for _ in range(10)]
        result = held_out_accuracy(X, V)
        assert 0.0 <= result["mean_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------

class TestSingleUserGuards:

    def test_inter_user_agreement_single_user(self):
        X = [torch.randn(10, 32)]
        result = inter_user_agreement(X)
        assert np.isnan(result["mean_pairwise_similarity"])
        assert result["n_users"] == 1

    def test_nearest_neighbor_accuracy_single_user(self):
        X = [torch.randn(10, 32)]
        result = nearest_neighbor_accuracy(X)
        assert np.isnan(result["mean_nn_accuracy"])
        assert result["n_users"] == 1


class TestHeldOutShuffle:

    def test_different_seeds_give_different_splits(self):
        D, K = 32, 4
        torch.manual_seed(123)
        V = torch.randn(D, K)
        # 100 pairs per user with test_frac=0.2 → 20 test pairs each,
        # giving fine-grained accuracy values that differ across shuffles
        X = [torch.randn(100, D) for _ in range(30)]
        r1 = held_out_accuracy(X, V, seed=0)
        r2 = held_out_accuracy(X, V, seed=99)
        assert r1["mean_accuracy"] != r2["mean_accuracy"], (
            "different seeds should produce different held-out splits"
        )

    def test_same_seed_is_reproducible(self):
        D, K = 32, 4
        V = torch.randn(D, K)
        X = [torch.randn(20, D) for _ in range(10)]
        r1 = held_out_accuracy(X, V, seed=7)
        r2 = held_out_accuracy(X, V, seed=7)
        assert r1["mean_accuracy"] == r2["mean_accuracy"]


class TestAnnotationDensityIntCounts:

    def test_accepts_int_counts(self):
        result = annotation_density([5, 10, 15, 20], K=4)
        assert result["median_pairs"] == 12.5
        assert result["n_users"] == 4
        assert not result["warn"]

    def test_warns_with_low_int_counts(self):
        result = annotation_density([1, 2, 3], K=8)
        assert result["warn"] is True


class TestPopulationAccuracyWarn:

    def test_small_dataset_sets_warn(self):
        D, K = 16, 4
        V = torch.randn(D, K)
        # Very few total pairs → n_test < 5
        X = [torch.randn(3, D) for _ in range(2)]
        result = population_accuracy(X, V)
        assert result["warn"] is True


# ---------------------------------------------------------------------------
# evaluate_suitability integration
# ---------------------------------------------------------------------------

class TestEvaluateSuitability:

    def test_returns_all_keys_with_V(self):
        D, K = 32, 4
        V = torch.randn(D, K)
        X = _make_random_users(20, 10, D)
        results = evaluate_suitability(X, V=V, K=K)
        expected_keys = {
            "annotation_density", "label_balance", "inter_user_agreement",
            "krippendorff_alpha_proxy", "nearest_neighbor_accuracy",
            "basis_space_coherence", "population_accuracy",
            "user_vector_diversity", "basis_utilization_entropy",
            "held_out_accuracy",
        }
        assert expected_keys.issubset(results.keys()), (
            f"missing keys: {expected_keys - results.keys()}"
        )

    def test_returns_embedding_only_keys_without_V(self):
        D = 32
        X = _make_random_users(10, 10, D)
        results = evaluate_suitability(X, V=None, K=4)
        assert "label_balance" in results
        assert "nearest_neighbor_accuracy" in results
        # V-dependent keys should be absent
        assert "basis_space_coherence" not in results
        assert "held_out_accuracy" not in results

    def test_does_not_print(self, capsys):
        D, K = 32, 4
        V = torch.randn(D, K)
        X = _make_random_users(10, 10, D)
        evaluate_suitability(X, V=V, K=K)
        captured = capsys.readouterr()
        assert captured.out == "", "evaluate_suitability should not print"


# ---------------------------------------------------------------------------
# PRISM benchmark (slow)
# ---------------------------------------------------------------------------

def _check_prism_data_available() -> bool:
    from apa.config import EMBEDDINGS_DIR, MODELS_DIR
    return (
        (EMBEDDINGS_DIR / "train.pkl").exists()
        and (MODELS_DIR / "V_K8.pt").exists()
    )


@pytest.fixture(scope="module")
def prism_data():
    """Load PRISM train embeddings grouped by seen user. Cached per module."""
    from apa.config import configure_environment, EMBEDDINGS_DIR, MODELS_DIR
    from apa.load_prism import group_embeddings_by_user

    configure_environment()
    device = "cpu"  # keep off GPU for diagnostic purposes

    train_embeddings = torch.load(EMBEDDINGS_DIR / "train.pkl", weights_only=False)
    test_embeddings = torch.load(EMBEDDINGS_DIR / "test.pkl", weights_only=False)

    train_seen, _, _, _ = group_embeddings_by_user(train_embeddings, test_embeddings, device)
    V = torch.load(MODELS_DIR / "V_K8.pt", map_location=device, weights_only=False)
    if isinstance(V, dict):
        V = V.get("V", next(iter(V.values())))
    V = V.float()

    return {"train_seen": train_seen, "V": V}


@pytest.mark.skipif(
    not _check_prism_data_available(),
    reason="PRISM embeddings or V_K8.pt not found.",
)
@pytest.mark.slow
class TestPRISMBenchmark:
    """
    Verify that all suitability metrics return 'green zone' values on PRISM.

    PRISM is a known-good dataset where LoRe achieves 87%+ accuracy at rank 8.
    Every metric here should confirm that — acting as a sanity check that the
    diagnostics are correctly calibrated.
    """

    def test_annotation_density(self, prism_data):
        result = annotation_density(prism_data["train_seen"], K=8)
        assert result["median_pairs"] >= 5, f"median pairs too low: {result['median_pairs']}"

    def test_label_balance_above_random(self, prism_data):
        result = label_balance(prism_data["train_seen"])
        nc = result["mean_normalized_consistency"]
        assert nc > 1.3, (
            f"mean_normalized_consistency={nc:.3f} — PRISM users should show "
            "consistent preferences (well above 1.0 random baseline)"
        )

    def test_krippendorff_proxy_users_are_distinct(self, prism_data):
        result = krippendorff_alpha_proxy(prism_data["train_seen"])
        assert result["corrected_ratio"] > 0.03, (
            f"corrected_ratio={result['corrected_ratio']:.4f} — "
            "PRISM users should be distinguishable above noise"
        )

    def test_nearest_neighbor_accuracy(self, prism_data):
        result = nearest_neighbor_accuracy(prism_data["train_seen"])
        assert result["mean_nn_accuracy"] > 0.55, (
            f"nn_accuracy={result['mean_nn_accuracy']:.3f} — "
            "geometrically similar PRISM users should share preferences"
        )

    def test_basis_space_coherence(self, prism_data):
        result = basis_space_coherence(prism_data["train_seen"], prism_data["V"])
        assert result["corrected_ratio"] > 0.03, (
            f"corrected_ratio={result['corrected_ratio']:.4f} — "
            "PRISM users should cluster in basis space"
        )

    def test_population_accuracy(self, prism_data):
        result = population_accuracy(prism_data["train_seen"], prism_data["V"])
        assert result["accuracy"] > 0.55, (
            f"population_accuracy={result['accuracy']:.3f} — "
            "PRISM has shared preference signal that V should capture"
        )

    def test_held_out_accuracy(self, prism_data):
        result = held_out_accuracy(prism_data["train_seen"], prism_data["V"])
        assert result["mean_accuracy"] > 0.55, (
            f"held_out_accuracy {result['mean_accuracy']:.3f} too low"
        )
        assert result["n_users_evaluated"] > 100, (
            f"too few users evaluated: {result['n_users_evaluated']}"
        )
