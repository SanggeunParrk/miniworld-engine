#!/bin/bash
# Fast import + dispatch gate. The full bench sweep is submitted with
# --dependency=afterok on THIS job, so a syntax/import error (e.g. in an edited module or
# kernel file) fails here and the ~overnight bench jobs never start on broken code, instead
# of every module job dying at import. Deliberately does NOT run the full numerical suite:
# a pre-existing per-module A100 failure (what the sweep is meant to FIND) must not block
# the whole sweep. Only import-ability + the arch-policy dispatch tests gate.
#SBATCH --job-name=mwk_gate_a100
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=benchmarks/gate_a100_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels

echo "host=$(hostname)  date=$(date)"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python - <<PY
import importlib
# parse/import the edited hot files + every module class the bench uses
mods = [
    "miniworld_engine",
    "miniworld_engine.modules.dispatch",
    "miniworld_engine.modules.transition.module",
    "miniworld_engine.kernels.transition.triton.fused",
]
for m in mods:
    importlib.import_module(m)
    print("import ok:", m)
from miniworld_engine.modules import (  # noqa: E402
    Transition, TriangleMultiplication, TriangleAttention,
    AdaptiveLayerNorm, ConditionedTransition,
)
print("module classes import ok")
PY'
rc=$?
echo "[import gate exit=$rc]"
[ "$rc" -ne 0 ] && exit "$rc"

pixi run --frozen bash -c 'PYTHONPATH=src python -m pytest tests/test_dispatch.py -q'
echo "[dispatch tests exit=$?]"
echo "GATE DONE"
