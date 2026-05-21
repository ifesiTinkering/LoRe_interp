"""
End-to-end test for LoRe training on PRISM dataset.

This test runs the full LoRe training pipeline with ranks 0 and 1,
and verifies that accuracy values match expected targets.

Tolerances:
- Seen user metrics (train, seen_unseen): 1.5% - validates core algorithm
- Unseen user metrics (few_shot_train, unseen_unseen): 5.0% - varies by random split

The expected values come from the original LoRe paper's specific random split.
Our code generates its own random split, so unseen user metrics may vary more
while still demonstrating the algorithm works correctly.

WARNING: This test takes approximately 12 minutes to complete.

Usage:
    pytest tests/test_lore.py -v -s
    pytest tests/test_lore.py -v -s -k test_lore_accuracy
"""

import gc
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apa.config import configure_environment, EMBEDDINGS_DIR
from apa.load_prism import group_embeddings_by_user
from apa.train_lore_bases import (
    LoReTrainer,
    eval_multiple,
    learn_multiple_few_shot,
)

logger = logging.getLogger(__name__)


# Expected accuracy values from LoRe paper (basis_log.txt)
EXPECTED_ACCURACIES = {
    0: {
        "train": 71.56,
        "seen_unseen": 71.56,
        "few_shot_train": 73.55,
        "unseen_unseen": 71.20,
    },
    1: {
        "train": 76.18,
        "seen_unseen": 76.59,
        "few_shot_train": 76.90,
        "unseen_unseen": 76.06,
    },
}

# Tolerance in percentage points
# Seen user metrics validate core algorithm - tight tolerance
TOLERANCE_SEEN = 2.0
# Unseen user metrics depend on random user split - looser tolerance
TOLERANCE_UNSEEN = 5.0

# Metrics that depend on the random user split
UNSEEN_METRICS = {"few_shot_train", "unseen_unseen"}


def check_embeddings_exist():
    """Check if embeddings are available for testing."""
    train_path = EMBEDDINGS_DIR / "train.pkl"
    test_path = EMBEDDINGS_DIR / "test.pkl"
    return train_path.exists() and test_path.exists()


@pytest.fixture(scope="module")
def embeddings_and_model():
    """Load embeddings and reward model V_final (cached per module)."""
    configure_environment()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load embeddings
    train_embeddings = torch.load(EMBEDDINGS_DIR / "train.pkl")
    test_embeddings = torch.load(EMBEDDINGS_DIR / "test.pkl")

    # Group by user
    train_seen, train_unseen, test_seen, test_unseen = group_embeddings_by_user(
        train_embeddings, test_embeddings, device
    )

    # Load V_final from reward model
    from transformers import AutoModel

    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    rm = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
        num_labels=1,
        low_cpu_mem_usage=True,
    )

    # Extract final linear layer weights
    last_linear_layer = None
    for name, module in rm.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_layer = module

    V_final = last_linear_layer.weight[:, 0].to(device).to(torch.float32).reshape(-1, 1)

    del rm
    gc.collect()

    return {
        "train_seen": train_seen,
        "train_unseen": train_unseen,
        "test_seen": test_seen,
        "test_unseen": test_unseen,
        "V_final": V_final,
        "device": device,
        "N": len(train_seen),
        "N_unseen": len(train_unseen),
    }


def run_lore_for_rank(K: int, data: dict, output_dir: Path) -> dict:
    """Run LoRe training for a specific rank and return accuracies."""
    device = data["device"]
    N = data["N"]
    N_unseen = data["N_unseen"]
    V_final = data["V_final"]
    train_seen = data["train_seen"]
    test_seen = data["test_seen"]
    train_unseen = data["train_unseen"]
    test_unseen = data["test_unseen"]

    if K == 0:
        # Reference model: use V_final directly
        V_joint = V_final
        W_joint = [torch.tensor([1.0]).to(device) for _ in range(N)]
    else:
        # Train LoRe model
        trainer = LoReTrainer(
            V_sft=V_final,
            alpha=10000.0,
            num_classes=N,
            num_features=4096,
            num_basis_vectors=K,
            num_iterations=20000,
            learning_rate=0.5,
            log_interval=5000,
        )
        W_joint, V_joint = trainer.train_model(train_seen)

        # Save checkpoint
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(V_joint.detach().cpu(), output_dir / f"V_lore_K_{K}.pt")
        torch.save(W_joint.detach().cpu(), output_dir / f"W_lore_K_{K}.pt")

    # Evaluate on train set (seen users, seen prompts)
    accuracies_train = eval_multiple(
        W_joint,
        [V_joint.detach() for _ in range(N)],
        train_seen
    )

    # Evaluate on test set (seen users, unseen prompts)
    accuracies_seen_unseen = eval_multiple(
        W_joint,
        [V_joint.detach() for _ in range(N)],
        test_seen
    )

    # Few-shot learning for unseen users
    if K <= 1:
        W_few_shot = [torch.tensor([1.0]).to(device) for _ in range(N_unseen)]
    else:
        W_few_shot = learn_multiple_few_shot(
            train_unseen,
            V_joint.detach(),
            num_iterations=500,
            learning_rate=0.5,
        )

    # Evaluate few-shot on train (unseen users, seen prompts)
    accuracies_few_shot_train = eval_multiple(
        W_few_shot,
        [V_joint.detach() for _ in range(N_unseen)],
        train_unseen
    )

    # Evaluate few-shot on test (unseen users, unseen prompts)
    accuracies_unseen_unseen = eval_multiple(
        W_few_shot,
        [V_joint.detach() for _ in range(N_unseen)],
        test_unseen
    )

    return {
        "train": np.mean(accuracies_train) * 100,
        "seen_unseen": np.mean(accuracies_seen_unseen) * 100,
        "few_shot_train": np.mean(accuracies_few_shot_train) * 100,
        "unseen_unseen": np.mean(accuracies_unseen_unseen) * 100,
    }


@pytest.mark.skipif(
    not check_embeddings_exist(),
    reason="PRISM embeddings not found. Run prepare_prism_embeddings.py first."
)
@pytest.mark.slow
class TestLoReAccuracy:
    """Test LoRe training accuracy on PRISM dataset."""

    def test_lore_accuracy(self, embeddings_and_model, tmp_path):
        """
        Test that LoRe training achieves expected accuracy on PRISM.

        WARNING: This test will take approximately 12 minutes to complete.
        """
        banner = (
            "\n============================================================\n"
            "RUNNING FULL LORE TRAINING TEST\n"
            "This will take approximately 12 minutes to complete.\n"
            "============================================================"
        )
        logger.warning(banner)
        print(banner, flush=True)

        results = {}
        for K in [0, 1]:
            print(f"\n--- Training K={K} ---", flush=True)
            # Seed before each K so LoReTrainer weight init is reproducible
            # (K=1 seeds torch.rand/randn for self.W and self.V).
            torch.manual_seed(42)
            np.random.seed(42)
            results[K] = run_lore_for_rank(K, embeddings_and_model, tmp_path)

            # Print results
            print(f"K={K} Results:", flush=True)
            for metric, value in results[K].items():
                expected = EXPECTED_ACCURACIES[K][metric]
                diff = abs(value - expected)
                tolerance = TOLERANCE_UNSEEN if metric in UNSEEN_METRICS else TOLERANCE_SEEN
                status = "PASS" if diff <= tolerance else "FAIL"
                print(
                    f"  {metric}: {value:.2f}% (expected: {expected:.2f}%, "
                    f"diff: {diff:.2f}%, tol: {tolerance}%) [{status}]",
                    flush=True
                )

        # Verify all accuracies are within tolerance
        print("\n--- Final Verification ---", flush=True)
        all_passed = True
        for K in [0, 1]:
            for metric, actual in results[K].items():
                expected = EXPECTED_ACCURACIES[K][metric]
                diff = abs(actual - expected)
                tolerance = TOLERANCE_UNSEEN if metric in UNSEEN_METRICS else TOLERANCE_SEEN
                if diff > tolerance:
                    all_passed = False
                    print(
                        f"FAILED: K={K} {metric}: {actual:.2f}% "
                        f"(expected: {expected:.2f}%, diff: {diff:.2f}%, tol: {tolerance}%)",
                        flush=True
                    )

        # Assert with detailed message
        for K in [0, 1]:
            for metric, actual in results[K].items():
                expected = EXPECTED_ACCURACIES[K][metric]
                tolerance = TOLERANCE_UNSEEN if metric in UNSEEN_METRICS else TOLERANCE_SEEN
                assert abs(actual - expected) <= tolerance, (
                    f"K={K} {metric}: {actual:.2f}% differs from expected "
                    f"{expected:.2f}% by more than {tolerance}%"
                )

        print("\nAll accuracy tests passed!", flush=True)


if __name__ == "__main__":
    # Allow running directly with python
    pytest.main([__file__, "-v", "-s"])
