#!/bin/bash
#SBATCH --job-name=mwk_uni_stage_a100
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/uni_stage_prof_a100_%j.out
set -uo pipefail; set -f
cd /home/psk6950/miniworld-kernels
echo "host=$(hostname) date=$(date)"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python submits/_uni_stage_prof.py'
echo "[exit=$?]"; echo "UNI_STAGE DONE"
