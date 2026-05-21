"""Three-stage preference filter for historical-persona synthetic preferences.

Filters, applied in order:

  A. drop records with ``consistency != 1.0`` (the model's two orderings
     disagreed, so the verdict is ambiguous);
  B. drop questions where every surviving record picked the same
     ``final_preference`` (no cross-persona disagreement to learn from);
  C. drop users with fewer than ``min_records_per_user`` records after A+B
     (too thin a per-user support to fit a LoRe W vector).

Reuses :func:`apa.synthetic_prefs.historical_prefs.results_to_jsonl_records`
to produce eval_prefs JSONL records (with ``chosen``/``rejected`` and
soft-preference fields) consumable by ``apa.lore_adapt``.

CLI Usage::

    uv run python -m experiments.filter_output filter \\
        --input  experiments/synthetic_prefs_C016_C020/hist_prefs_C016_raw.json \\
                 experiments/synthetic_prefs_C016_C020/hist_prefs_C020_raw.json \\
        --output experiments/synthetic_prefs_C016_C020/hist_prefs_all_filtered.jsonl \\
        --min-records-per-user 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from apa.synthetic_prefs.historical_prefs import results_to_jsonl_records


def filter_prefs(
    raw_records: list[dict],
    min_records_per_user: int = 5,
) -> tuple[list[dict], dict]:
    """Apply the A→B→C filter chain to raw historical-pref records.

    Args:
        raw_records: list of dicts as produced by
            :func:`apa.synthetic_prefs.historical_prefs.generate_century_prefs`
            (i.e. the contents of ``hist_prefs_<century>_raw.json``).
            Each record must have ``consistency``, ``final_preference``,
            ``user_id``, ``question_id``.
        min_records_per_user: filter C threshold (``< this``  →  drop user).

    Returns:
        ``(jsonl_records, stats)`` where ``jsonl_records`` is a list of
        dicts in eval_prefs JSONL format (``chosen``/``rejected``/etc.),
        and ``stats`` is a per-stage record count summary.
    """
    stats = {"input": len(raw_records)}

    # Filter A: consistency == 1.0 (and final_preference is one of the valid
    # picks; the original generator marks inconsistent records "-1").
    after_a = [
        r for r in raw_records
        if r.get("consistency") == 1.0 and r.get("final_preference") in ("1", "2")
    ]
    stats["after_consistency"] = len(after_a)

    # Filter B: drop questions where every surviving record agreed on the
    # same final_preference.
    picks_by_q: dict[object, set[str]] = defaultdict(set)
    for r in after_a:
        picks_by_q[r["question_id"]].add(r["final_preference"])
    divisive_qids = {q for q, picks in picks_by_q.items() if len(picks) >= 2}
    after_b = [r for r in after_a if r["question_id"] in divisive_qids]
    stats["after_divisive"] = len(after_b)
    stats["divisive_questions"] = len(divisive_qids)

    # Filter C: drop users with < min_records_per_user records after A+B.
    user_counts = Counter(r["user_id"] for r in after_b)
    keep_users = {u for u, n in user_counts.items() if n >= min_records_per_user}
    after_c = [r for r in after_b if r["user_id"] in keep_users]
    stats["after_user_coverage"] = len(after_c)
    stats["users_kept"] = len(keep_users)
    stats["min_records_per_user"] = min_records_per_user

    # Convert to eval_prefs JSONL schema (chosen/rejected + soft prefs).
    jsonl_records = results_to_jsonl_records(after_c)
    stats["output"] = len(jsonl_records)

    return jsonl_records, stats


def _load_raw(paths: list[str]) -> list[dict]:
    """Concatenate raw records from one or more JSON files."""
    all_records: list[dict] = []
    for p in paths:
        with open(p) as f:
            all_records.extend(json.load(f))
    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-stage preference filter (consistency → divisive Qs → user coverage).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("filter", help="Apply A→B→C filters to raw pref records.")
    f.add_argument("--input", nargs="+", required=True,
                   help="One or more hist_prefs_<century>_raw.json files.")
    f.add_argument("--output", required=True,
                   help="Output JSONL path (eval_prefs schema).")
    f.add_argument("--min-records-per-user", type=int, default=5,
                   help="Drop users with fewer than this many records after A+B.")

    args = parser.parse_args()

    if args.command == "filter":
        raw = _load_raw(args.input)
        records, stats = filter_prefs(raw, min_records_per_user=args.min_records_per_user)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

        print(f"Filter pipeline summary:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print(f"  -> wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
