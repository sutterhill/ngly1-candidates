#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/codex_ngly1_20260819
PYTHON=/opt/pytorch/bin/python
SCRIPT="$ROOT/work/ridgey_local_ensemble_score.py"
TAG="${1:-pool10}"
FASTA="${2:-$ROOT/work/ngly1_10mut_pool.fasta}"
OUT="$ROOT/work/ridgey_local_ensemble_${TAG}"
mkdir -p "$OUT"

names=(base ens1 ens2 ens3 ens4)
checkpoints=(
  "$ROOT/models/ridgey_v2_600m.pt"
  "$ROOT/models/ridgey_v2_600m_ens1.pt"
  "$ROOT/models/ridgey_v2_600m_ens2.pt"
  "$ROOT/models/ridgey_v2_600m_ens3.pt"
  "$ROOT/models/ridgey_v2_600m_ens4.pt"
)

pids=()
for index in 0 1 2 3 4; do
  name="${names[$index]}"
  CUDA_VISIBLE_DEVICES="$index" "$PYTHON" "$SCRIPT" \
    --checkpoint "${checkpoints[$index]}" \
    --model-name "ridgey_v2_600m_${name}" \
    --device cuda:0 \
    --fasta "$FASTA" \
    --output "$OUT/${name}.json" \
    >"$OUT/${name}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
