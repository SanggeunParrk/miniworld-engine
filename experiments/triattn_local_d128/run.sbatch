#!/bin/bash
#SBATCH --job-name=triattn-d128-compact
#SBATCH --partition=h100
#SBATCH --qos=normal_h100
#SBATCH --account=cssb
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/psk6950/miniworld-engine/experiments/triattn_local_d128/results/job-%j.log
set -euo pipefail
root=/home/psk6950/miniworld-engine/experiments/triattn_local_d128
bash "$root/env.sh" python -u "$root/build.py"
bash "$root/env.sh" python -u "$root/bench.py"
