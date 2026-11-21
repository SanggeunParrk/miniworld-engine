#!/bin/bash
# Run a command (the measurers on the anthropic backend) on the model kits' stack of a B200 host (card b200_sm100): a kit venv with the
# kits' package set (torch 2.13.0+cu130, triton 3.7.1: the esmfold2 kit's venv130 list) and the engine worktree on PYTHONPATH. The B200
# driver (580, CUDA 13.1) loads the release's CUDA-13.0 code natively, so no container and no forward-compatible libcuda are needed.
# The kits' B200 cells in the release tables were measured on this stack (column NVIDIA_B200:2.13.0+cu130/3.7.1).
#
#   KIT_VENV=/path/to/kitvenv130 third_party/anthropic/kit_stack_b200.sh python -m miniworld_engine.measure.speed CONFIG --out OUT.jsonl
#
# Environment: KIT_VENV (the kit venv, required), MINIWORLD_ANTHROPIC_ROOT (the release root; default the worktree's upstream copy).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
: "${KIT_VENV:?set KIT_VENV to the kit venv (torch 2.13.0+cu130)}"
export PATH=$KIT_VENV/bin:$PATH
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
export MINIWORLD_ANTHROPIC_ROOT=${MINIWORLD_ANTHROPIC_ROOT:-$HERE/upstream}
exec "$@"
