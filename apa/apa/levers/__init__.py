"""
Strategy modules for democratic inference.

- voter_sampling: User sampling strategies
- voter_aggregation: Ranking aggregation strategies
- query_selection: Question selection strategies
- slate_generation: Response generation strategies
"""

from apa.levers.voter_sampling import (
    parse_jury_source_spec,
    per_group_sampling,
    random_sampling,
    stratified_sampling,
    weighted_sampling,
    temporal_mix_sampling,
)
from apa.levers.voter_aggregation import (
    borda_count,
    plurality,
    copeland,
    instant_runoff,
)
from apa.levers.query_selection import random_subset
from apa.levers.slate_generation import temperature_sampling

__all__ = [
    "random_sampling",
    "stratified_sampling",
    "weighted_sampling",
    "temporal_mix_sampling",
    "per_group_sampling",
    "parse_jury_source_spec",
    "borda_count",
    "plurality",
    "copeland",
    "instant_runoff",
    "random_subset",
    "temperature_sampling",
]
