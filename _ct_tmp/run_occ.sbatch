#!/bin/bash
#SBATCH --job-name=ct_occ
#SBATCH -p h100 -A cssb -q cssb_h100 --gres=gpu:h100:1 -c 8 --mem=48G -t 00:12:00
#SBATCH --output=/home/psk6950/miniworld-kernels/_ct_tmp/ct_occ_%j.out
set -uo pipefail; cd /home/psk6950/miniworld-kernels
ENV=/home/psk6950/miniworld-kernels/.pixi/envs/default
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}; export PYTHONPATH=src
"$ENV/bin/python" /home/psk6950/miniworld-kernels/_ct_tmp/occ.py 2>&1
echo "[exit=$?] OCC DONE"
