#!/bin/bash
# A/B B=1 regression probe for the TRITON bidirectional trimul B>1 generalisation.
# Times B=1 on the WORKING TREE (new code), then `git stash` -> times HEAD (baseline),
# then `git stash pop`. Untracked helper scripts are NOT stashed (no -u), so both runs
# use the same bench. The two BENCH lines are printed for direct comparison.
#SBATCH --job-name=mwk_bidir_b1_ab_a100
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/bidir_b1_ab_a100_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels
export HYDRA_FULL_ERROR=1
echo "host=$(hostname) date=$(date)"

echo "### git status (before) ###"; git status --short

echo "### A = working tree (NEW, B>1-generalised) ###"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src TAG=NEW python submits/_bidir_b1_bench.py'
echo "[exit=$?]"

STASHED=0
if git stash push -- src/miniworld_engine/kernels/trimul_inproj/triton/bidirectional.py \
                     src/miniworld_engine/kernels/trimul_inproj/triton/back_fused.py; then
  STASHED=1
fi
trap '[ "$STASHED" = 1 ] && git stash pop' EXIT

echo "### verify HEAD restored ###"; git status --short
echo "### B = HEAD (BASELINE, pre-change) ###"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src TAG=BASE python submits/_bidir_b1_bench.py'
echo "[exit=$?]"

echo "AB DONE"
