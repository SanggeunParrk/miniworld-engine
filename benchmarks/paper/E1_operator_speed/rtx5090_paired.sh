#!/bin/bash
# RTX 5090 paired E1 re-measure (paired.py: engine stack + Anthropic kit in one process, 6 interleaved rounds), through the GPU lock in
# resumable chunks (each chunk stops starting new cases after BUDGET seconds; run again until <out>.done exists).
#   MODULES="TriangleAttentionAF3 OuterProductMean" TAG=triattn_opm ./rtx5090_paired.sh
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../paper_common/env_rtx5090.sh"
GPU_SECONDS=${GPU_SECONDS:-1800}
BUDGET=${BUDGET:-$((GPU_SECONDS - 300))}
OUT=${RECORDS:-$HERE/records_paired}/$CARD/paired_${TAG:?}.r${REP:-1}.jsonl
until [ -f "$OUT.done" ]; do
  # shellcheck disable=SC2086
  "$HOME/mwe/gpu.sh" "$GPU_SECONDS" python "$HERE/paired.py" "$HERE/cfg_paired.yaml" --out "$OUT" --rounds "${ROUNDS:-6}" \
    --budget-seconds "$BUDGET" --module $MODULES $EXTRA || { echo "chunk exited $?"; exit 1; }
done
