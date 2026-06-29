#!/bin/bash
#SBATCH --job-name=tok_occ
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/home/psk6950/miniworld-kernels/_ct_tmp/tok_occ_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-kernels
ENV=/home/psk6950/miniworld-kernels/.pixi/envs/default
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}; export PYTHONPATH=src
$ENV/bin/python /home/psk6950/miniworld-kernels/_ct_tmp/tok_occ.py
echo "[exit=$?] TOK DONE"
