#!/usr/bin/env bash
#SBATCH --job-name=mw-h100-build
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=112
#SBATCH --mem=896G
#SBATCH --time=24:00:00
set -euo pipefail
cd /home/psk6950/miniworld-engine
export CONDA_PREFIX="$PWD/.pixi/envs/default"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MAX_JOBS=28
run_dir="$PWD/.scratch/h100-build"
export XDG_CACHE_HOME="$run_dir/cache"
export TRITON_CACHE_DIR="$run_dir/triton-cache"
export TORCH_EXTENSIONS_DIR="$run_dir/torch-extensions"
export QUACK_CACHE_DIR="$run_dir/quack-cache"
export MINIWORLD_PLAN_CACHE_DIR="$run_dir/plans"
mkdir -p "$run_dir"
nvidia-smi
srun --cpu-bind=cores python - <<'PY'
import torch
from miniworld_engine.autotune.preflight import native_dependencies
assert torch.cuda.device_count() == 4, 'Expected four allocated GPUs'
assert all(torch.cuda.get_device_capability(i)[0] == 9 for i in range(4))
native_dependencies('sm90')
PY
# A cache build is not a correctness test. Qualify masks and gradients first.
srun --cpu-bind=cores python -m pytest tests/numerics/test_trimul_fused_mask.py -q -x \
    --junitxml="$run_dir/mask-tests-${SLURM_JOB_ID}.xml"
srun --cpu-bind=cores python -m pytest tests/numerics/test_training_dropout.py \
    -k 'miniworld and not attention' -q -x --junitxml="$run_dir/dropout-tests-${SLURM_JOB_ID}.xml"
srun --cpu-bind=cores python -m pytest tests/numerics/test_numerical.py \
    -k 'transition or trimul or layernorm' -q -x --junitxml="$run_dir/kernel-tests-${SLURM_JOB_ID}.xml"
srun --cpu-bind=cores miniworld-engine build all grid --gpus all --compile-jobs 28 \
    --units-per-gpu 1 --resume --shards "$run_dir/shards" --keep-triton-cache
