"""
User sampling strategies for democratic voting — i.e. how the jury that
votes on each query is drawn from the pool of available reward models.

Also exposes the parsing helpers (``parse_jury_source_spec``,
``_normalize_source_label``) for the ``--jury_sources`` flag, since the
labels they produce are exactly the period-tags this module's
``per_group_sampling`` filters on.
"""

from __future__ import annotations

import random
import re
from typing import Any

import numpy as np

from apa._logging import log


# =============================================================================
# Jury-source label parsing (backs the --jury_sources CLI flag)
# =============================================================================

_SOURCE_ALIASES = {"prism": "original", "original": "original"}
# Matches user-facing period labels: C21, C021, 21C, c21, 13c, etc.
_PERIOD_LABEL_RE = re.compile(r"^\s*C?0*(\d{1,3})C?\s*$", re.IGNORECASE)


def _normalize_source_label(label: str) -> str:
    """Normalise a jury-source label to the internal 'period' form.

    Accepts 'prism'/'original' and century variants like 'C21', 'C017',
    '21C', '17c'. Returns 'original' or 'NC' (e.g. '21C'). Unknown
    labels pass through unchanged.
    """
    s = label.strip()
    if not s:
        return s
    if s.lower() in _SOURCE_ALIASES:
        return _SOURCE_ALIASES[s.lower()]
    m = _PERIOD_LABEL_RE.match(s)
    if m:
        return f"{int(m.group(1))}C"
    return s


def parse_jury_source_spec(spec: str) -> tuple[str, int | None]:
    """
    Parse one ``--jury_sources`` token into (normalised_label, count).

    Accepted forms:
      - ``"C16"``       → ("16C", None)        # use all available
      - ``"prism:10"``  → ("original", 10)     # sample 10
      - ``"prism=10"``  → ("original", 10)     # `=` accepted as alias for `:`
      - ``"C16:all"``   → ("16C", None)        # explicit "all"

    Count of None means "include every available voter for this group".
    """
    raw = spec.strip()
    if not raw:
        raise ValueError("Empty jury source spec")
    for sep in (":", "="):
        if sep in raw:
            label, count_str = raw.split(sep, 1)
            label = label.strip()
            count_str = count_str.strip().lower()
            if count_str in {"", "all", "*"}:
                return _normalize_source_label(label), None
            try:
                n = int(count_str)
            except ValueError as e:
                raise ValueError(
                    f"Invalid jury source count in '{spec}': '{count_str}'"
                ) from e
            if n < 0:
                raise ValueError(f"Jury source count must be >= 0, got {n} in '{spec}'")
            return _normalize_source_label(label), n
    return _normalize_source_label(raw), None


def random_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """Sample users uniformly at random."""
    return random.sample(all_user_ids, min(m, len(all_user_ids)))


def stratified_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """Sample users with stratification by a metadata field (e.g., century)."""
    if user_metadata is None:
        return random_sampling(all_user_ids, user_metadata, m, config)

    stratify_by = config.get('stratify_by', 'century')

    groups: dict[Any, list[str]] = {}
    for user_id in all_user_ids:
        if user_id not in user_metadata:
            continue
        group = user_metadata[user_id].get(stratify_by, 'unknown')
        if group not in groups:
            groups[group] = []
        groups[group].append(user_id)

    if not groups:
        return random_sampling(all_user_ids, user_metadata, m, config)

    n_groups = len(groups)
    per_group = m // n_groups
    remainder = m % n_groups

    selected = []
    for i, (group, users) in enumerate(groups.items()):
        n_to_sample = per_group + (1 if i < remainder else 0)
        n_to_sample = min(n_to_sample, len(users))
        selected.extend(random.sample(users, n_to_sample))

    return selected[:m]


def weighted_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """Sample users with weights based on metadata 'weight' field."""
    if user_metadata is None:
        return random_sampling(all_user_ids, user_metadata, m, config)

    weights = []
    valid_users = []
    for user_id in all_user_ids:
        if user_id in user_metadata and 'weight' in user_metadata[user_id]:
            weights.append(user_metadata[user_id]['weight'])
            valid_users.append(user_id)

    if not valid_users:
        return random_sampling(all_user_ids, user_metadata, m, config)

    weights = np.array(weights)
    weights = weights / weights.sum()

    indices = np.random.choice(
        len(valid_users),
        size=min(m, len(valid_users)),
        replace=False,
        p=weights,
    )

    return [valid_users[i] for i in indices]


def temporal_mix_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any] | None,
    m: int,
    config: dict,
) -> list[str]:
    """Sample a mix of modern (C021) and historical users."""
    if user_metadata is None:
        return random_sampling(all_user_ids, user_metadata, m, config)

    historical_ratio = config.get('historical_ratio', 0.5)

    modern_users = []
    historical_users = []

    for user_id in all_user_ids:
        if user_id not in user_metadata:
            modern_users.append(user_id)
            continue

        century = user_metadata[user_id].get('century', 'C021')
        if century == 'C021' or century is None:
            modern_users.append(user_id)
        else:
            historical_users.append(user_id)

    n_historical = int(m * historical_ratio)
    n_modern = m - n_historical

    n_historical = min(n_historical, len(historical_users))
    n_modern = min(n_modern, len(modern_users))

    remainder = m - n_historical - n_modern
    if remainder > 0:
        if len(historical_users) > n_historical:
            n_historical += min(remainder, len(historical_users) - n_historical)
            remainder = m - n_historical - n_modern
        if remainder > 0 and len(modern_users) > n_modern:
            n_modern += min(remainder, len(modern_users) - n_modern)

    selected = []
    if n_modern > 0 and modern_users:
        selected.extend(random.sample(modern_users, n_modern))
    if n_historical > 0 and historical_users:
        selected.extend(random.sample(historical_users, n_historical))

    return selected


def per_group_sampling(
    all_user_ids: list[str],
    user_metadata: dict[str, Any],
    jury_sources: list[tuple[str, int | None]],
    m_voters_fallback: int,
) -> tuple[list[str], str, dict[str, Any]]:
    """Compose a jury from explicitly named groups.

    Filters ``all_user_ids`` to those whose ``user_metadata[uid]['period']``
    matches one of the labels in ``jury_sources``, then either:

      - takes the requested per-group count (when any count is explicit),
        sampling without replacement from each group; or
      - falls back to stratified-by-period sampling capped at
        ``m_voters_fallback`` (when every group's count is ``None``).

    Returns ``(sampled_user_ids, audit_strategy, audit_config)`` where the
    last two are the values to record in the vote's audit log so the
    composition is reproducible.

    Raises ``ValueError`` if any requested label has zero matching voters,
    or if an explicit count exceeds the matching pool size.
    """
    requested = [label for label, _ in jury_sources]
    counts = {label: count for label, count in jury_sources}
    pool = [
        uid for uid in all_user_ids
        if user_metadata.get(uid, {}).get("period") in counts
    ]
    by_group: dict[str, list[str]] = {g: [] for g in requested}
    for uid in pool:
        by_group[user_metadata[uid]["period"]].append(uid)
    empty = [g for g, members in by_group.items() if not members]
    if empty:
        raise ValueError(
            f"No voters in jury for requested source(s) {empty}. "
            f"Available periods: "
            f"{sorted({m.get('period') for m in user_metadata.values()})}"
        )

    explicit_counts = any(c is not None for c in counts.values())
    if explicit_counts:
        sampled_user_ids: list[str] = []
        for label in requested:
            members = by_group[label]
            requested_n = counts[label]
            if requested_n is None:
                sampled_user_ids.extend(members)
            else:
                if requested_n > len(members):
                    raise ValueError(
                        f"Requested {requested_n} voters from '{label}' "
                        f"but only {len(members)} are available."
                    )
                sampled_user_ids.extend(random.sample(members, requested_n))
        audit_strategy = "per_group"
        audit_config = {
            "per_group_counts": {
                label: (counts[label] if counts[label] is not None else len(by_group[label]))
                for label in requested
            },
        }
        log(
            f"Sampling {len(sampled_user_ids)} voters from {len(pool)} "
            f"(filtered to {requested}) via per-group selection: "
            f"{audit_config['per_group_counts']}"
        )
    else:
        m = min(m_voters_fallback, len(pool))
        audit_strategy = "stratified"
        audit_config = {"stratify_by": "period"}
        log(
            f"Sampling {m} voters from {len(pool)} (filtered to "
            f"{requested}) via stratified-by-period..."
        )
        sampled_user_ids = stratified_sampling(pool, user_metadata, m, audit_config)

    return sampled_user_ids, audit_strategy, audit_config
