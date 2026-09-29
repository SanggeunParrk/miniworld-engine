#!/bin/bash
# sbatch a job from a snapshot of this directory, so edits made while it queues or runs do not reach it:
#   ./snap_submit.sh job_x.sbatch [args...]      (the log still goes to ./logs)
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
SNAP=${A100_SNAP_ROOT:-$HOME/.cache/miniworld-a100/snap}/$(date +%m%d-%H%M%S)-$$
mkdir -p "$SNAP/experiments"
ln -s "$REPO/src" "$SNAP/src"
rsync -a --exclude logs --exclude __pycache__ --exclude 'csrc/archive' "$HERE/" "$SNAP/experiments/a100_trimul_fwd/"
ln -s "$HERE/logs" "$SNAP/experiments/a100_trimul_fwd/logs"
cd "$SNAP/experiments/a100_trimul_fwd"
sbatch --output="$HERE/logs/%x-%j.log" "$@"
