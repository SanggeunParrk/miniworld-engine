#!/bin/bash
#SBATCH --job-name=ct_repro
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/psk6950/miniworld-kernels/_ct_tmp/repro_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-kernels
ENV=/home/psk6950/miniworld-kernels/.pixi/envs/default
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src
$ENV/bin/python /home/psk6950/miniworld-kernels/_ct_tmp/repro.py 2>&1
echo "[exit=$?] REPRO DONE"
