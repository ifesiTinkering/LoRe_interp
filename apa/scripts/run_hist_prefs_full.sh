#!/usr/bin/env bash
# Generate synthetic historical preferences (20 users) and compare against PRISM/Random baselines.
#
# Uses: low temperature (0.3), enriched profiles with value statements,
#       curated value-laden PRISM questions, matched-question baselines.
#
# Expected runtime: ~20 minutes (10 profiles x 2 centuries x 20 questions x 3 reps x 2 orders
#   = 2400 model queries, plus embedding + eval).
#
# Usage:
#   bash scripts/run_hist_prefs_full.sh [output_dir]
set -euo pipefail

OUT_DIR="${1:-/nas/XXXX-9/XXXX-1/APA/synthetic_prefs}"
mkdir -p "$OUT_DIR"

echo "Expected runtime: ~20 minutes"
echo ""

echo "=== Step 1: Generate historical preferences (20 users, curated questions, temp=0.3) ==="
uv run python -m apa.synthetic_prefs.historical_prefs generate-synth \
    --centuries C013 C019 \
    --questions apa/synthetic_prefs/curated_questions.txt \
    --n-runs 3 \
    --temperature 0.3 \
    --seed 42 \
    --output-dir "$OUT_DIR"

echo ""
echo "=== Step 2: Compare Synth vs PRISM vs Random (matched questions, 20 users each) ==="
uv run python scripts/compare_metrics.py \
    --synth-path "$OUT_DIR/hist_prefs_all.jsonl" \
    --n-baseline-users 20 \
    --seed 42

echo ""
echo "=== Done. Results in $OUT_DIR ==="
