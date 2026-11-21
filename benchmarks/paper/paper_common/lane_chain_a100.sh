#!/bin/bash
# lane_chain.sh N_JOBS [FIRST_DEP_JOBID] [sbatch args]: one lane = a chain of N 10-minute jobs of a100_lane.sbatch (afterany dependency;
# the first job waits for FIRST_DEP_JOBID if given, e.g. the last job of an earlier chain of the same lane); several lanes = concurrency.
set -euo pipefail
N=$1; prev=${2:-}; shift $(($# < 2 ? $# : 2))
P=$(cd "$(dirname "$0")/.." && pwd)
for i in $(seq 1 "$N"); do
  dep=(); [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
  prev=$(sbatch --parsable "$@" "${dep[@]}" -D "$P" --export=ALL,PAPERDIR="$P" "$P/paper_common/a100_lane.sbatch")
done
echo "$prev"
