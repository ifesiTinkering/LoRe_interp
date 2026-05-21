#!/usr/bin/env bash
# Hold a fixed-jury democratic vote over experiments/query_responses.jsonl,
# then post-process the audit log into per-group aggregations + intra/inter
# group rank-agreement.
#
# Jury = all 10 C016 + all 10 C020 historical voters + 10 randomly-sampled
# PRISM voters. Frozen LoRe basis V_K8 (PRISM-trained); historical voters
# from W_adapted_hist_C016_C020_filtered.pt; PRISM voters from W_seen_K8.pt.
#
# Defaults to the repo-tracked checkpoints under experiments/checkpoints/ so
# this script reproduces the paper exactly without depending on the lab NAS.
# Override paths via --V_checkpoint / --prism_users / --adapted_users (or
# the env vars below) if you've trained your own.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESPONSES="$REPO_ROOT/experiments/query_responses.jsonl"
V_CHECKPOINT="${V_CHECKPOINT:-$REPO_ROOT/experiments/checkpoints/V_K8.pt}"
PRISM_USERS="${PRISM_USERS:-$REPO_ROOT/experiments/checkpoints/W_seen_K8.pt}"
ADAPTED="${ADAPTED:-$REPO_ROOT/experiments/checkpoints/W_adapted_hist_C016_C020_filtered.pt}"
OUT_DIR="$REPO_ROOT/experiments/vote_C016_C020"
LOG_DIR="$REPO_ROOT/experiments/logs"
AUDIT_LOG="$OUT_DIR/audit_log.json"
mkdir -p "$OUT_DIR" "$LOG_DIR"

cd "$REPO_ROOT"

# 1. Hold the vote. --jury_sources picks ALL C16 + ALL C20 + 10 random PRISM.
#    --methods runs every aggregation method we have so we can compare.
uv run python -m apa.democratic_response \
    --responses_file "$RESPONSES" \
    --V_checkpoint "$V_CHECKPOINT" \
    --prism_users "$PRISM_USERS" \
    --adapted_users "$ADAPTED" \
    --jury_sources "C16,C20,prism:10" \
    --methods borda_count,plurality,copeland,instant_runoff \
    --seed 42 \
    --log_file "$AUDIT_LOG" \
    2>&1 | tee "$LOG_DIR/vote_C016_C020.log"

# 2. Post-process: per-group aggregations + intra/inter agreement.
uv run python -m apa.vote_analysis "$AUDIT_LOG" --output_dir "$OUT_DIR" \
    2>&1 | tee -a "$LOG_DIR/vote_C016_C020.log"
