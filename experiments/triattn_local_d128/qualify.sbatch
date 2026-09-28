#!/bin/bash
#SBATCH --job-name=triattn-d128-check
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/psk6950/miniworld-engine/experiments/triattn_local_d128/results/check-%j.log
set -euo pipefail
root=/home/psk6950/miniworld-engine/experiments/triattn_local_d128
bash "$root/env.sh" python -u "$root/stress.py"
bash "$root/env.sh" python -u "$root/bench.py" --length 384
for tool in memcheck racecheck synccheck; do
  bash "$root/env.sh" compute-sanitizer --tool "$tool" --error-exitcode 87 \
    python -u "$root/stress.py" --sanitize --length 64 \
    > "$root/results/${tool}-${SLURM_JOB_ID}.log" 2>&1
done
bash "$root/env.sh" compute-sanitizer --tool memcheck --error-exitcode 87 \
  python -u "$root/stress.py" --sanitize --length 768 \
  > "$root/results/memcheck-L768-${SLURM_JOB_ID}.log" 2>&1
