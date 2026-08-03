#!/bin/bash
# B>1 verification for the remaining TRITON trimul pieces (single-direction module +
# standalone front/back kernels), plus the full pytest suite to confirm (a) B=1 regression
# stays clean through the front/back/unidirectional edits and (b) the stale dispatch test fix.
#SBATCH --job-name=mwk_trimul_uni_fb_a100
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=benchmarks/trimul_uni_fb_a100_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-engine
export HYDRA_FULL_ERROR=1
echo "host=$(hostname) date=$(date)"
echo "### git status (files under test should be MODIFIED) ###"; git status --short

echo "### B>1 verify: unidirectional module + front/back kernels ###"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python submits/_trimul_uni_fb_verify.py'
echo "[verify exit=$?]"

echo "### full pytest suite (regression + stale-test fix) ###"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python -m pytest -q \
  tests/test_numerical.py tests/test_dispatch.py tests/test_int64_offsets.py -o addopts="" 2>&1'
echo "[pytest exit=$?]"
echo "TRIMUL_UNI_FB DONE"
