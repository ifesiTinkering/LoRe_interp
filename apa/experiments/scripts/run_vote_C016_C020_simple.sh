#!/usr/bin/env bash
# Same fixed-jury vote as run_vote_C016_C020.sh, but over the simplified
# yes/no response set in experiments/query_responses_simple.jsonl.
#
# See run_vote_C016_C020.sh for the env-var override hooks; defaults point
# at the repo-tracked checkpoints under experiments/checkpoints/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESPONSES="$REPO_ROOT/experiments/query_responses_simple.jsonl"
V_CHECKPOINT="${V_CHECKPOINT:-$REPO_ROOT/experiments/checkpoints/V_K8.pt}"
PRISM_USERS="${PRISM_USERS:-$REPO_ROOT/experiments/checkpoints/W_seen_K8.pt}"
ADAPTED="${ADAPTED:-$REPO_ROOT/experiments/checkpoints/W_adapted_hist_C016_C020_filtered.pt}"
OUT_DIR="$REPO_ROOT/experiments/vote_C016_C020_simple"
LOG_DIR="$REPO_ROOT/experiments/logs"
AUDIT_LOG="$OUT_DIR/audit_log.json"
mkdir -p "$OUT_DIR" "$LOG_DIR"

cd "$REPO_ROOT"

uv run python -m apa.democratic_response \
    --responses_file "$RESPONSES" \
    --V_checkpoint "$V_CHECKPOINT" \
    --prism_users "$PRISM_USERS" \
    --adapted_users "$ADAPTED" \
    --jury_sources "C16,C20,prism:10" \
    --methods borda_count,plurality,copeland,instant_runoff \
    --seed 42 \
    --log_file "$AUDIT_LOG" \
    2>&1 | tee "$LOG_DIR/vote_C016_C020_simple.log"

uv run python -m apa.vote_analysis "$AUDIT_LOG" --output_dir "$OUT_DIR" \
    2>&1 | tee -a "$LOG_DIR/vote_C016_C020_simple.log"
