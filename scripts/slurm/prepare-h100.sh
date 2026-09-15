#!/usr/bin/env bash
#SBATCH --job-name=mw-h100-prepare
#SBATCH --partition=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
cd /home/psk6950/miniworld-engine
export CONDA_PREFIX="$PWD/.pixi/envs/default"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"
export XDG_CACHE_HOME="$PWD/.scratch/cache"
export MAX_JOBS=8
run_dir="$PWD/.scratch/h100-build"
mkdir -p "$run_dir"
# Keep the locked torch/CUTLASS/quack toolchain while adding H100-only dependencies.
python - <<'PY' > "$run_dir/constraints.txt"
from importlib.metadata import version
for package in ('torch', 'triton', 'quack-kernels', 'nvidia-cutlass-dsl'):
    print(f'{package}=={version(package)}')
PY
python -m pip install --pre -c "$run_dir/constraints.txt" flash-attn-4 nvidia-mathdx
# The locked TE installation includes two cores; restore its CUDA 12 core last.
python -m pip install --no-deps --force-reinstall transformer_engine_cu12==2.16.0
python -m pip freeze > "$run_dir/environment.txt"
python -m ruff check src/miniworld_engine/modules/triangle_multiplication \
    src/miniworld_engine/kernels/trimul_inproj/cute/bidir_training.py \
    src/miniworld_engine/kernels/trimul_inproj/cute/v6_training_merged.py \
    src/miniworld_engine/kernels/trimul_inproj/whole_op.py tests/numerics/test_trimul_fused_mask.py
python -m pytest tests/builder/test_native_preflight.py tests/builder/test_build_cli_defaults.py \
    tests/builder/test_build_worker_allocation.py tests/registry/test_trimul_widths_are_implementable.py -q \
    --junitxml="$run_dir/cpu-tests.xml"
