#!/bin/bash
# Submit a resumable measurement script as a chain of N jobs (each --time=00:10:00 as the cluster rule says; every job continues where the
# previous one stopped and exits at once when the work is finished):
#   paper_common/chain.sh 12 E1_operator_speed/h100.sbatch            # repeat 1 (REP=1)
#   REP=2 paper_common/chain.sh 12 E1_operator_speed/h100.sbatch       # repeat 2: a different Slurm chain
# Never python here (login node): sbatch only.
set -euo pipefail
N=$1; SCRIPT=$2; shift 2
SCRIPT_ABS=$(realpath "$SCRIPT")
prev=""
for i in $(seq 1 "$N"); do
  dep=(); [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
  prev=$(sbatch --parsable "${dep[@]}" -D "$(dirname "$SCRIPT_ABS")" --export=ALL,REP="${REP:-1}",EXP_DIR="$(dirname "$SCRIPT_ABS")" "$@" "$SCRIPT_ABS")
  echo "job $prev ($i/$N)"
done
