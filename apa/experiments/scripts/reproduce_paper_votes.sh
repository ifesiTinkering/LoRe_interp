#!/usr/bin/env bash
# Exact-reproduction shortcut: hold both democratic votes (regular + simple
# slate) using the repo-tracked V_K8 / W_seen_K8 / W_adapted_*.pt
# checkpoints, then post-process. Skips PRISM data prep, V/W training, and
# 70B HistLlama generation entirely — those steps' outputs are tracked in
# experiments/checkpoints/ and experiments/synthetic_prefs_C016_C020/.
#
# Hardware: one GPU with >=16 GB free for the Skywork-Reward embedding of
# the candidate response slates. (~30 seconds per vote on an A100/A6000.)
#
# Outputs land under experiments/vote_C016_C020/ and
# experiments/vote_C016_C020_simple/. Compare audit_log.json's
# sampled_user_ids / per_voter_rankings / aggregations / average_ranks
# against the tracked versions to verify byte-equality at the rankings
# level (paths and timestamps will of course differ).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

bash experiments/scripts/run_vote_C016_C020.sh
bash experiments/scripts/run_vote_C016_C020_simple.sh

echo
echo "=== Done. Vote outputs: ==="
echo "  experiments/vote_C016_C020/{audit_log,vote_results,vote_analysis}.json"
echo "  experiments/vote_C016_C020_simple/{audit_log,vote_analysis}.json"
echo "  experiments/vote_C016_C020*/vote_report.txt"
