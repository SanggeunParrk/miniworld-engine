#!/bin/bash
#SBATCH --job-name=triattn-d128-fullcheck
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/miniworld-engine/experiments/triattn_local_d128/results/fullcheck-%j.log
set -euo pipefail
root=/home/psk6950/miniworld-engine/experiments/triattn_local_d128
bash "$root/env.sh" compute-sanitizer --tool memcheck --error-exitcode 87 \
  python -u "$root/dropout_check.py" > "$root/results/full-memcheck-${SLURM_JOB_ID}.log" 2>&1
bash "$root/env.sh" compute-sanitizer --tool racecheck --error-exitcode 87 \
  python -u "$root/stress.py" --sanitize --length 256 > "$root/results/racecheck-L256-${SLURM_JOB_ID}.log" 2>&1
bash "$root/env.sh" compute-sanitizer --tool synccheck --error-exitcode 87 \
  python -u "$root/stress.py" --sanitize --length 768 > "$root/results/synccheck-L768-${SLURM_JOB_ID}.log" 2>&1
