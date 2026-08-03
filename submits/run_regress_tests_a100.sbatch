#!/bin/bash
# Full-repo regression check on A100 after the triton bidir B>1 generalisation. Confirms the
# ONE shared-helper edit (front_bwd_dW, no-op view at B=1) did not regress the OTHER paths:
# numerical fwd/bwd vs reference (all modules incl. trimul), dispatch resolution, int64 offsets.
#SBATCH --job-name=mwk_regress_a100
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=benchmarks/regress_tests_a100_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-engine
export HYDRA_FULL_ERROR=1
echo "host=$(hostname) date=$(date)"
echo "### git status (must show the 2 files MODIFIED = NEW code under test) ###"
git status --short

pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python -m pytest -q \
  tests/test_numerical.py tests/test_dispatch.py tests/test_int64_offsets.py \
  -o addopts="" 2>&1'
echo "[pytest exit=$?]"
echo "REGRESS DONE"
