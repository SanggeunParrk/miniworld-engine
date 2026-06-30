#!/bin/bash
#SBATCH --job-name=ct_tvc_repro
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/modules/conditioned_transition/artifacts/reproduce_ct_tvc_%j.out

set -euo pipefail
cd /home/psk6950/miniworld-kernels

echo "host=$(hostname) date=$(date)"
pixi run --frozen python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/modules/conditioned_transition/diagnostics/reproduce_ct_tvc.py "$@"' _ \
  --output benchmarks/modules/conditioned_transition/artifacts/conditioned_transition_reproduce_ct_tvc_training.csv
