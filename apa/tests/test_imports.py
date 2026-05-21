#!/usr/bin/env python3
"""
Test that all modules can be imported correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config():
    """Test config module."""
    from apa.config import (
        APAConfig,
        DatasetConfig,
        LoReConfig,
        InferenceConfig,
        configure_environment,
        get_config,
    )
    config = get_config()
    assert config.lore.alpha == 10000.0
    print("  config: OK")


def test_load_prism():
    """Test load_prism module."""
    from apa.load_prism import (
        load_prism_pairwise,
        PRISMDataset,
        group_embeddings_by_user,
        CheckpointManager,
    )
    print("  load_prism: OK")


def test_train_lore_bases():
    """Test train_lore_bases module."""
    from apa.train_lore_bases import (
        LoReRewardModel,
        LoReTrainer,
        get_embedding_model,
        embed_texts,
    )
    print("  train_lore_bases: OK")


def test_historical_prefs():
    """Test historical_prefs module."""
    from apa.synthetic_prefs.historical_prefs import (
        load_hist_llama,
        generate_historical_preferences,
        preference_from_logprobs,
        preferences_to_labels,
    )
    print("  historical_prefs: OK")


def test_democratic_response():
    """Test democratic_response module."""
    from apa.democratic_response import (
        DemocraticInference,
        InferenceResult,
        QueryCase,
        build_default_jury,
        load_query_cases,
        generate_responses,
    )
    print("  democratic_response: OK")


def test_voter_sampling():
    """Test voter_sampling module."""
    from apa.levers.voter_sampling import (
        random_sampling,
        stratified_sampling,
        weighted_sampling,
        temporal_mix_sampling,
    )
    print("  voter_sampling: OK")


def test_voter_aggregation():
    """Test voter_aggregation module."""
    from apa.levers.voter_aggregation import (
        borda_count,
        plurality,
        copeland,
        instant_runoff,
    )
    print("  voter_aggregation: OK")


def test_query_selection():
    """Test query_selection module."""
    from apa.levers.query_selection import random_subset
    print("  query_selection: OK")


def test_slate_generation():
    """Test slate_generation module."""
    from apa.levers.slate_generation import temperature_sampling
    print("  slate_generation: OK")


def test_aggregation_borda():
    """Test borda_count with sample data."""
    from apa.levers.voter_aggregation import borda_count

    rankings = {
        'user_1': [0, 1, 2],
        'user_2': [1, 0, 2],
        'user_3': [0, 2, 1],
    }
    config = {}

    result = borda_count(rankings, config)
    assert len(result) == 3
    assert result[0] == 0  # Response 0 should win
    print("  aggregation_borda: OK")


def test_sampling_random():
    """Test random_sampling."""
    from apa.levers.voter_sampling import random_sampling

    all_users = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5']
    config = {}

    result = random_sampling(all_users, None, 3, config)
    assert len(result) == 3
    assert all(u in all_users for u in result)
    print("  sampling_random: OK")


def main():
    """Run all import tests."""
    print("\nTesting APA module imports...")
    print("-" * 40)

    tests = [
        test_config,
        test_load_prism,
        test_train_lore_bases,
        test_historical_prefs,
        test_democratic_response,
        test_voter_sampling,
        test_voter_aggregation,
        test_query_selection,
        test_slate_generation,
        test_aggregation_borda,
        test_sampling_random,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  {test_fn.__name__}: FAILED - {e}")
            failed += 1

    print("-" * 40)
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
