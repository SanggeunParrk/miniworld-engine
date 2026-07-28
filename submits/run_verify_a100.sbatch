#!/bin/bash
# Fast A100 correctness gate for the transition k-tiled fix. Runs BEFORE the ~1h full
# bench: confirms sm_80, the dispatch arch policy, and (crucially) that the newly-wired
# bounded-smem k-tiled fused b2b matches the pytorch reference in fwd + input-grad at
# d=256/512 (the shapes that OOM'd the old full-K expand kernel).
#SBATCH --job-name=mwk_verify_a100
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/modules/transition/artifacts/verify_a100_%x_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels
export HYDRA_FULL_ERROR=1

echo "host=$(hostname)  date=$(date)"
pixi run --frozen python submits/_assert_device.py --major 8 || exit 3

echo ""
echo "############################################################"
echo "# dispatch arch-policy tests (CPU, arch mocked)"
echo "############################################################"
pixi run --frozen bash -c 'PYTHONPATH=src python -m pytest tests/test_dispatch.py -q'
echo "[dispatch tests exit=$?]"

echo ""
echo "############################################################"
echo "# numerical fwd + input-grad vs pytorch reference (FULL suite: all modules +"
echo "#   transition_d256/d512, trimul_bidir, layernorm). -v so per-case pass/fail is visible;"
echo "#   a pre-existing per-module A100 failure is signal, not a gate."
echo "############################################################"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python -m pytest tests/test_numerical.py -v'
echo "[numerical full exit=$?]"

echo ""
echo "VERIFY DONE"
