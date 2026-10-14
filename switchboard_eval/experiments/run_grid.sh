#!/usr/bin/env bash
# Task 7 grid: every encoder x head, trained on the live stack's examples.
# Usage: switchboard_eval/experiments/run_grid.sh [examples.jsonl]   (default devices/desktop/examples.jsonl)
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"  # repo root
PY="$HERE/.venv-bench/bin/python"
EXAMPLES="${1:-$HERE/devices/desktop/examples.jsonl}"
mkdir -p "$HERE/preds"
ENCODERS=(
  char
  st:BAAI/bge-small-en-v1.5
  st:intfloat/multilingual-e5-small
  st:sentence-transformers/all-MiniLM-L6-v2
  m2v:minishlab/potion-base-8M
  m2v:OpenVoiceOS/ovos-m2v-intents-multilingual
)
for enc in "${ENCODERS[@]}"; do
  tag="$(echo "$enc" | tr ':/' '__')"
  for head in proto knn logreg; do
    out="$HERE/preds/${tag}__${head}.jsonl"
    (cd "$HERE" && "$PY" -m switchboard_eval.experiments.bench --encoder "$enc" --head "$head" \
      --examples "$EXAMPLES" --schema "$HERE/schema/intents.yaml" --dataset "$HERE/data/dataset.jsonl" \
      --out "$out") > "${out%.jsonl}.log" 2>&1 && echo "ok   $enc $head" || echo "FAIL $enc $head"
  done
done
