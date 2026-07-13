#!/bin/bash
# Verify the cache-prune wiring across live Triton kernels did NOT break anything:
#  (1) import every touched kernel module + list discovered wired op-tags (catches syntax /
#      import / bad-prune-construction errors).
#  (2) run the EXISTING test suite that exercises these modules fwd+bwd (dispatch + numerical
#      correctness + public api). The prune is transparent (cache miss -> full grid == prior
#      behavior), so these must still pass; cache-miss warnings are expected + confirm warn-on-miss.
#SBATCH --job-name=mwk_wire_verify
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/wire_verify_a100_%j.out

set -uo pipefail
cd /home/psk6950/miniworld-kernels
echo "host=$(hostname) date=$(date)"

echo "### (1) import all touched kernel modules + discover wired op-tags ###"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python submits/_wire_import_check.py'
echo "[import exit=$?]"

echo "### (2) existing test suite (dispatch + numerical + public api) ###"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python -m pytest tests/test_dispatch.py tests/test_public_api.py tests/test_numerical.py -q 2>&1 | tail -60'
echo "[pytest exit=$?]"
echo "WIRE VERIFY DONE"
