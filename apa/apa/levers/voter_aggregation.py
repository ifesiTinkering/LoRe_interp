"""
Ranking aggregation strategies for democratic voting.

Implementations delegate to the pref_voting package.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def _to_profile(rankings: dict[str, list[int]]) -> Any:
    """Convert rankings dict to a pref_voting Profile."""
    from pref_voting.profiles import Profile

    ranking_tuples = [tuple(r) for r in rankings.values()]
    counter = Counter(ranking_tuples)
    unique = list(counter.keys())
    rcounts = [counter[r] for r in unique]
    return Profile(unique, rcounts=rcounts)


def borda_count(rankings: dict[str, list[int]], config: dict) -> list[int]:
    """
    Aggregate rankings using Borda count.

    Each voter awards points based on position: 1st gets k-1 points, 2nd gets k-2, etc.
    Delegates to pref_voting Profile.borda_scores().
    """
    if not rankings:
        return []

    prof = _to_profile(rankings)
    scores = prof.borda_scores()
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def plurality(rankings: dict[str, list[int]], config: dict) -> list[int]:
    """
    Aggregate rankings using plurality voting (only first-choice counts).

    Delegates to pref_voting Profile.plurality_scores().
    """
    if not rankings:
        return []

    prof = _to_profile(rankings)
    scores = prof.plurality_scores()
    return sorted(scores, key=lambda x: (scores[x], -x), reverse=True)


def copeland(rankings: dict[str, list[int]], config: dict) -> list[int]:
    """
    Aggregate rankings using Copeland's method (pairwise wins/losses).

    Delegates to pref_voting Profile.copeland_scores().
    Scoring: +1 win, +0.5 tie, 0 loss.
    """
    if not rankings:
        return []

    prof = _to_profile(rankings)
    scores = prof.copeland_scores()
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def instant_runoff(rankings: dict[str, list[int]], config: dict) -> list[int]:
    """
    Aggregate rankings using Instant-Runoff Voting (IRV).

    Iteratively finds the winner using pref_voting's instant_runoff, removes
    them, and repeats to produce a full ranking. Candidate IDs are remapped to
    0-indexed each round as required by pref_voting's Profile.
    """
    if not rankings:
        return []

    from pref_voting.profiles import Profile
    from pref_voting.voting_methods import instant_runoff as _pref_irv

    all_ranking_tuples = [tuple(r) for r in rankings.values()]
    remaining = sorted(set(all_ranking_tuples[0]))
    result = []

    while len(remaining) > 1:
        orig_to_local = {orig: local for local, orig in enumerate(remaining)}
        filtered = [tuple(orig_to_local[c] for c in r if c in orig_to_local) for r in all_ranking_tuples]
        counter = Counter(filtered)
        unique = list(counter.keys())
        rcounts = [counter[r] for r in unique]
        prof = Profile(unique, rcounts=rcounts)

        local_winner = int(_pref_irv(prof)[0])
        original_winner = remaining[local_winner]
        result.append(original_winner)
        remaining.remove(original_winner)

    if remaining:
        result.append(remaining[0])

    return result


AGGREGATION_METHODS: dict[str, Any] = {
    "borda_count": borda_count,
    "plurality": plurality,
    "copeland": copeland,
    "instant_runoff": instant_runoff,
}
