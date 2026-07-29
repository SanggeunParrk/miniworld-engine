#!/bin/bash
# The GPU half of the MPNN test suite. Most of what guards these kernels cannot run on a
# login node -- no CUDA driver there -- so `pixi run test` silently skips it and the
# only place these assertions ever execute is a job like this one.
#
# A5000, and not by preference: src/miniworld_kernels/autotune/data/ carries a tuned
# config per kernel per device, and every mpnn_* kernel has an A5000 entry and nothing
# else. Every other kernel family ships A100/A6000/H100/B200 too. On an A6000 these
# tests therefore autotune from scratch -- _norm_configs alone is 324 configs, over 500
# across the four autotuned kernels in triton/main.py -- and a run that looks hung is
# just tuning. Capture an A6000 cache (run_autotune_capture_ampere.sbatch) before
# moving this off A5000.
#
#   sbatch --gres=gpu:A6000:1 submits/run_mpnn_gputests_ampere.sbatch
#   TESTS="tests/test_mpnn_edge_tail.py" sbatch ... submits/run_mpnn_gputests_ampere.sbatch
#SBATCH --job-name=mwk_mpnn_gputests
#SBATCH --partition=gpu
#SBATCH --account=cssb
#SBATCH --qos=normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --exclude=gpu01
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/modules/mpnn_gputests_%j.out
set -uo pipefail
cd /home/psk6950/practice/miniworld-kernels
export TRITON_CACHE_DIR="/tmp/triton_gputests_${SLURM_JOB_ID:-$$}"
echo "host=$(hostname) date=$(date)"
pixi run --frozen python submits/_assert_device.py --capability 8.6 --name A5000 --name A6000 || exit 3

TESTS="${TESTS:-tests/test_mpnn_edge_tail.py tests/test_mpnn_edge_tail_compute.py tests/test_mpnn_knn_backends.py tests/test_mpnn_numerical.py tests/test_mpnn_upstream_parity.py}"
echo "TESTS=$TESTS"
pixi run --frozen bash -c \
  'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python -m pytest "$@" -q -rf' _ $TESTS
echo "[exit=$?] MPNN GPUTESTS DONE"
