#!/bin/bash
#SBATCH --job-name=ct_ncu
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/home/psk6950/miniworld-kernels/_ct_tmp/ct_ncu_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-kernels
ENV=/home/psk6950/miniworld-kernels/.pixi/envs/default
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}; export PYTHONPATH=src
"$ENV/bin/ncu" --target-processes all -k "regex:expand_swiglu|squeeze_gate" -c 2 \
  --section SpeedOfLight --section Occupancy --section WarpStateStats --section LaunchStats \
  "$ENV/bin/python" /home/psk6950/miniworld-kernels/_ct_tmp/prof_single.py 2>&1 | \
  grep -iE "void|kernel|Duration|DRAM Through|Compute \(SM|Achieved Occup|Theoretical Occup|Registers Per|Block Limit|Shared Mem|Stall|Warp Cycles|Issued|secti|====|^  +[A-Z]" | head -120
echo "[exit=$?] NCU DONE"
