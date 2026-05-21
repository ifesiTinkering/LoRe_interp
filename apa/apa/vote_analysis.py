"""
Post-process a democratic-vote audit log: per-group aggregations and
intra/inter-group rank-agreement.

Outputs:
  - ``vote_analysis.json``: structured per-case results, plus the jury
    composition (voters per group).
  - ``vote_report.txt``: short human-readable summary suitable for
    pasting into experiment notes.

CLI:
    python -m apa.vote_analysis <audit_log.json> --output_dir <dir>
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import kendalltau as _scipy_kendalltau
from scipy.stats import spearmanr as _scipy_spearmanr

from apa.levers.voter_aggregation import AGGREGATION_METHODS


# =============================================================================
# Rank-agreement metrics
# =============================================================================


def _to_position_vector(ranking: list[int]) -> list[int]:
    """Convert a preference order (response indices) to a position vector."""
    pos = [0] * len(ranking)
    for p, idx in enumerate(ranking):
        pos[idx] = p
    return pos


def spearman(rank_a: list[int], rank_b: list[int]) -> float:
    """Spearman ρ between two full rankings (lists of response indices)."""
    if len(rank_a) != len(rank_b):
        raise ValueError("Rankings have different lengths")
    rho, _ = _scipy_spearmanr(_to_position_vector(rank_a), _to_position_vector(rank_b))
    return float(rho)


def kendall_tau(rank_a: list[int], rank_b: list[int]) -> float:
    """Kendall τ-b between two full rankings (lists of response indices)."""
    if len(rank_a) != len(rank_b):
        raise ValueError("Rankings have different lengths")
    tau, _ = _scipy_kendalltau(_to_position_vector(rank_a), _to_position_vector(rank_b))
    return float(tau)


def _mean_pairwise(
    rankings: list[list[int]],
    fn: Callable[[list[int], list[int]], float],
) -> float:
    if len(rankings) < 2:
        return float("nan")
    vals = [fn(a, b) for a, b in combinations(rankings, 2)]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _mean_cross(
    a: list[list[int]],
    b: list[list[int]],
    fn: Callable[[list[int], list[int]], float],
) -> float:
    if not a or not b:
        return float("nan")
    vals = [fn(x, y) for x in a for y in b]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


# =============================================================================
# Per-case analysis
# =============================================================================


def _group_voters(
    sampled_user_ids: list[str],
    sampled_user_metadata: dict[str, dict],
    group_key: str = "period",
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for uid in sampled_user_ids:
        meta = sampled_user_metadata.get(uid, {})
        g = meta.get(group_key, "unknown")
        groups.setdefault(g, []).append(uid)
    return groups


# Sentinel scope key for the all-voters bucket. Underscored so it can't
# collide with a real metadata "period" value like "16C" or even "full".
ALL_SCOPE = "__all__"


def analyze_case(case: dict) -> dict[str, Any]:
    """Compute per-group aggregations and agreement for one InferenceResult dict."""
    per_voter_rankings = {uid: list(r) for uid, r in case["per_voter_rankings"].items()}
    sampled_meta = case.get("sampled_user_metadata", {})
    # Defensive: only group voters that actually have rankings — partial
    # logs (e.g. a crashed run) should produce a warning rather than KeyError.
    valid_uids = [uid for uid in case["sampled_user_ids"] if uid in per_voter_rankings]
    missing = set(case["sampled_user_ids"]) - set(valid_uids)
    if missing:
        print(f"[vote_analysis] warning: {len(missing)} sampled voter(s) "
              f"missing from per_voter_rankings (skipping): {sorted(missing)[:5]}"
              + ("..." if len(missing) > 5 else ""))
    groups = _group_voters(valid_uids, sampled_meta)

    per_group_aggregations: dict[str, dict[str, dict[str, Any]]] = {}
    scopes: dict[str, list[str]] = {ALL_SCOPE: list(per_voter_rankings.keys())}
    for g, users in groups.items():
        # Guard against a hypothetical period == ALL_SCOPE collision; keep
        # the all-voters scope authoritative.
        if g == ALL_SCOPE:
            print(f"[vote_analysis] warning: group label {g!r} collides with "
                  f"all-voters sentinel; renaming group to {g + '_group'!r}.")
            g = g + "_group"
        scopes[g] = users

    for scope, users in scopes.items():
        sub = {u: per_voter_rankings[u] for u in users if u in per_voter_rankings}
        per_method: dict[str, dict[str, Any]] = {}
        for method, fn in AGGREGATION_METHODS.items():
            ranking = list(fn(sub, {}))
            per_method[method] = {
                "ranking": ranking,
                "winner_idx": int(ranking[0]) if ranking else None,
            }
        per_group_aggregations[scope] = per_method

    # Structured agreement entries: {"kind": "intra"|"inter", "scope"|"groups",
    # "mean_spearman", "mean_kendall_tau", "n_pairs"}. Keeping this as a list
    # of dicts (instead of stringly-typed "intra_<g>" / "inter_<g1>_<g2>" keys)
    # lets group labels contain underscores without parsing ambiguity.
    agreement: list[dict[str, Any]] = []
    all_ranks = list(per_voter_rankings.values())
    agreement.append({
        "kind": "intra",
        "scope": ALL_SCOPE,
        "mean_spearman": _mean_pairwise(all_ranks, spearman),
        "mean_kendall_tau": _mean_pairwise(all_ranks, kendall_tau),
        "n_pairs": len(all_ranks) * (len(all_ranks) - 1) // 2,
    })
    group_names = list(groups.keys())
    for g in group_names:
        ranks = [per_voter_rankings[u] for u in groups[g]]
        agreement.append({
            "kind": "intra",
            "scope": g,
            "mean_spearman": _mean_pairwise(ranks, spearman),
            "mean_kendall_tau": _mean_pairwise(ranks, kendall_tau),
            "n_pairs": len(ranks) * (len(ranks) - 1) // 2,
        })
    for g1, g2 in combinations(group_names, 2):
        r1 = [per_voter_rankings[u] for u in groups[g1]]
        r2 = [per_voter_rankings[u] for u in groups[g2]]
        agreement.append({
            "kind": "inter",
            "groups": [g1, g2],
            "mean_spearman": _mean_cross(r1, r2, spearman),
            "mean_kendall_tau": _mean_cross(r1, r2, kendall_tau),
            "n_pairs": len(r1) * len(r2),
        })

    return {
        "query_id": case.get("query_id"),
        "query": case.get("query"),
        "n_responses": len(case.get("responses", [])),
        "groups": {g: list(users) for g, users in groups.items()},
        "per_group_aggregations": per_group_aggregations,
        "agreement": agreement,
    }


def analyze_audit_log(log: list[dict]) -> dict[str, Any]:
    """Run :func:`analyze_case` on every case in an audit log."""
    return {
        "n_cases": len(log),
        "cases": [analyze_case(c) for c in log],
    }


# =============================================================================
# Reporting
# =============================================================================


def _format_ranking(ranking: list[int]) -> str:
    return " > ".join(f"#{i + 1}" for i in ranking)


def render_report(analysis: dict[str, Any]) -> str:
    """Produce a short human-readable text report from analyze_audit_log()."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"Vote analysis — {analysis['n_cases']} case(s)")
    lines.append("=" * 72)

    for case in analysis["cases"]:
        lines.append("")
        qid = case.get("query_id")
        q = case.get("query") or ""
        lines.append(f"Q{qid}: {q}")
        lines.append(f"  n_responses={case['n_responses']}")
        lines.append("  Jury composition: " + ", ".join(
            f"{g}={len(users)}" for g, users in case["groups"].items()
        ))

        scopes = [ALL_SCOPE] + list(case["groups"].keys())
        lines.append("  Aggregate winners (response_id is 1-indexed):")
        for scope in scopes:
            row = case["per_group_aggregations"][scope]
            lines.append(
                f"    [{scope:<8}]  "
                + "  ".join(f"{m}=#{row[m]['winner_idx'] + 1}" for m in AGGREGATION_METHODS)
            )

        lines.append("  Group rankings (Borda):")
        for scope in scopes:
            r = case["per_group_aggregations"][scope]["borda_count"]["ranking"]
            lines.append(f"    [{scope:<8}] {_format_ranking(r)}")

        lines.append("  Agreement (mean pairwise):")
        for entry in case["agreement"]:
            if entry["kind"] == "intra":
                label = f"intra[{entry['scope']}]"
            else:
                g1, g2 = entry["groups"]
                label = f"inter[{g1} ↔ {g2}]"
            lines.append(
                f"    {label:<32} spearman={entry['mean_spearman']:+.3f}  "
                f"kendall={entry['mean_kendall_tau']:+.3f}  (n_pairs={entry['n_pairs']})"
            )

    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-group aggregation + rank-agreement analysis "
                    "for a democratic_response audit log.",
    )
    parser.add_argument("audit_log", type=Path,
                        help="Path to an audit log JSON written by apa.democratic_response.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory for vote_analysis.json + vote_report.txt.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.audit_log) as f:
        log = json.load(f)
    if isinstance(log, dict):
        log = [log]

    analysis = analyze_audit_log(log)

    out_json = args.output_dir / "vote_analysis.json"
    with open(out_json, "w") as f:
        json.dump(analysis, f, indent=2)

    report = render_report(analysis)
    (args.output_dir / "vote_report.txt").write_text(report)

    print(report)
    print(f"Wrote {out_json}")
    print(f"Wrote {args.output_dir / 'vote_report.txt'}")


if __name__ == "__main__":
    main()
