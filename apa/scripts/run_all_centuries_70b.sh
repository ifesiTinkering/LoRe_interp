#!/usr/bin/env bash
# Generate 70B synthetic historical preferences for ALL centuries (C013-C021)
# and compare against PRISM/Random baselines.
#
# Each century runs as a separate process to avoid GPU OOM when loading
# successive 70B models (~140GB each).
#
# Estimated runtime: ~3.25 hours (9 centuries x ~20 min each + eval)
#   - First run is slower due to model downloads (~8 min per new century)
#   - Subsequent runs: ~15 min per century (model cached)
#
# Usage:
#   bash scripts/run_all_centuries_70b.sh [output_dir]
set -euo pipefail

OUT_DIR="${1:-/nas/XXXX-9/XXXX-1/APA/synthetic_prefs_70b_all}"
mkdir -p "$OUT_DIR"

CENTURIES="C013 C014 C015 C016 C017 C018 C019 C020 C021"
N_CENTURIES=9
N_PROFILES=10
TOTAL_USERS=$((N_CENTURIES * N_PROFILES))

echo "Estimated runtime: ~3.25 hours (9 centuries x 10 profiles x 20 questions x 3 reps x 2 orders)"
echo "Total synthetic users: $TOTAL_USERS"
echo "Output: $OUT_DIR"
echo ""

START_TIME=$SECONDS

for CENTURY in $CENTURIES; do
    ELAPSED=$(( (SECONDS - START_TIME) / 60 ))
    echo ""
    echo "============================================================"
    echo "  $CENTURY  (elapsed: ${ELAPSED}m)"
    echo "============================================================"

    uv run python -m apa.synthetic_prefs.historical_prefs generate-synth \
        --centuries "$CENTURY" \
        --questions apa/synthetic_prefs/curated_questions.txt \
        --n-runs 3 \
        --temperature 0.3 \
        --model-size 70B \
        --seed 42 \
        --output-dir "$OUT_DIR"
done

# Combine all per-century JSONL files
echo ""
echo "=== Combining per-century JSONL files ==="
cat "$OUT_DIR"/hist_prefs_C*.jsonl > "$OUT_DIR/hist_prefs_all.jsonl"
TOTAL_LINES=$(wc -l < "$OUT_DIR/hist_prefs_all.jsonl")
echo "Combined: $TOTAL_LINES preference records in hist_prefs_all.jsonl"

# Run comparison
echo ""
echo "=== Comparing Synth ($TOTAL_USERS) vs PRISM ($TOTAL_USERS) vs Random ($TOTAL_USERS) ==="
uv run python scripts/compare_metrics.py \
    --synth-path "$OUT_DIR/hist_prefs_all.jsonl" \
    --n-baseline-users "$TOTAL_USERS" \
    --seed 42

ELAPSED=$(( (SECONDS - START_TIME) / 60 ))
echo ""
echo "=== Done in ${ELAPSED} minutes. Results in $OUT_DIR ==="
