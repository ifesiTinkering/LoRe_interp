"""Utilities for the experiments/ scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_question_ids(jsonl_path: Path | str, output_path: Path | str) -> int:
    """Extract ``question_id`` values from a JSONL file, one per line.

    Args:
        jsonl_path: Path to the input JSONL file (each line a JSON object
            containing a ``question_id`` field).
        output_path: Path to write the IDs file (one integer ID per line).

    Returns:
        The number of IDs written.
    """
    jsonl_path = Path(jsonl_path)
    output_path = Path(output_path)

    ids: list[int] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(int(json.loads(line)["question_id"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for qid in ids:
            f.write(f"{qid}\n")

    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiments utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract-question-ids",
        help="Extract question_ids from a JSONL file into a plain IDs file.",
    )
    extract.add_argument("--input", required=True, help="Input JSONL path")
    extract.add_argument("--output", required=True, help="Output IDs file path")

    args = parser.parse_args()

    if args.command == "extract-question-ids":
        n = extract_question_ids(args.input, args.output)
        print(f"Wrote {n} question IDs to {args.output}")


if __name__ == "__main__":
    main()
