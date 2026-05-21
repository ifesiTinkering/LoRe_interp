#!/usr/bin/env bash
# Full run: generate synthetic preferences for C016 and C020 using the
# two-stage CoT prompt with system-role persona and X/Y labels.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments"
OUT_DIR="$EXP_DIR/synthetic_prefs_C016_C020"
PROFILES="$EXP_DIR/profiles_C016_C020.jsonl"
QUESTIONS_JSONL="$EXP_DIR/chosen_questions.jsonl"
QUESTIONS_IDS="$OUT_DIR/chosen_question_ids.txt"

mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES=1,2,6,7

cd "$REPO_ROOT"

uv run python -m experiments.utils extract-question-ids \
    --input "$QUESTIONS_JSONL" \
    --output "$QUESTIONS_IDS"

uv run python -m apa.synthetic_prefs.historical_prefs generate-synth \
    --centuries C016 C020 \
    --model-size 70B \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --profiles "$PROFILES" \
    --questions "$QUESTIONS_IDS" \
    --output-dir "$OUT_DIR"
