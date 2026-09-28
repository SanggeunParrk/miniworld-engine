#!/bin/bash
#SBATCH --job-name=triattn-native-check
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/miniworld-engine/experiments/triattn_local_d128/results/nativecheck-%j.log
set -euo pipefail
root=/home/psk6950/miniworld-engine/experiments/triattn_local_d128
for tool in racecheck synccheck; do
  length=256
  if [[ "$tool" == synccheck ]]; then length=768; fi
  bash "$root/env.sh" compute-sanitizer --tool "$tool" --error-exitcode 87 \
    --kernel-name kns=grouped_dkdv --kernel-name kns=reduce_bias \
    python -u "$root/stress.py" --sanitize --length "$length" \
    > "$root/results/${tool}-native-L${length}-${SLURM_JOB_ID}.log" 2>&1
done
