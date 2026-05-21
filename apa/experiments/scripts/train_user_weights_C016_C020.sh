#!/usr/bin/env bash
# Few-shot LoRe adaptation: fit per-user weight vectors w_u for the 20
# C016+C020 historical personas using the filtered synthetic preferences.
# The PRISM-trained LoRe basis V_K8.pt is loaded read-only (frozen).
#
# Defaults to the repo-tracked V at experiments/checkpoints/V_K8.pt and
# writes W_adapted_hist_C016_C020_filtered.pt to experiments/checkpoints/
# so a subsequent vote uses the freshly-fit W. Set V_CHECKPOINT or
# OUTPUT_DIR to override.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFS="$REPO_ROOT/experiments/synthetic_prefs_C016_C020/hist_prefs_all_filtered.jsonl"
V_CHECKPOINT="${V_CHECKPOINT:-$REPO_ROOT/experiments/checkpoints/V_K8.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/experiments/checkpoints}"
LOG_DIR="$REPO_ROOT/experiments/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

cd "$REPO_ROOT"

uv run python -m apa.lore_adapt "$PREFS" \
    --K 8 \
    --basis "$V_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --name hist_C016_C020_filtered \
    2>&1 | tee "$LOG_DIR/train_user_weights_C016_C020.log"
