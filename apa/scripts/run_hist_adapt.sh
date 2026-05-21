#!/usr/bin/env bash
# Few-shot LoRe adaptation on historical preferences (top 3 per century).
#
# Embeds 540 preference pairs (1080 texts) using Skywork-Reward, then fits
# per-user weight vectors via gradient descent against pre-trained bases.
#
# Expected runtime: ~10-15 minutes (mostly embedding time).
#
# Usage:
#   bash scripts/run_hist_adapt.sh [prefs_file] [K]
set -euo pipefail

PREFS="${1:-hist_prefs_top3.jsonl}"
K="${2:-8}"

echo "=== Few-shot LoRe adaptation ==="
echo "Preferences: $PREFS"
echo "Rank K: $K"
echo ""

uv run python -m apa.lore_adapt \
    "$PREFS" \
    --K "$K" \
    --num_iterations 500 \
    --learning_rate 0.5 \
    --test_frac 0.2 \
    --seed 42 \
    --name "hist_top3"

echo ""
echo "=== Done ==="
