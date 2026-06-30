#!/bin/bash
#SBATCH --job-name=trans_repro
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/modules/transition/artifacts/reproduce_transition_module_%j.out

set -euo pipefail
cd /home/psk6950/miniworld-kernels

echo "host=$(hostname) date=$(date)"
pixi run --frozen python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/modules/transition/diagnostics/reproduce_transition_module.py "$@"' _ \
  --output-dir benchmarks/modules/transition/artifacts/transition_module_repro
