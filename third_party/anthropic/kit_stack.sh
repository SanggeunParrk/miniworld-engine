#!/bin/bash
# Run a command (e.g. the speed measurer on the anthropic backend) on the model kits' own stack: their CUDA-13.0 container (Ubuntu 22.04,
# the CUDA 13.0 forward-compatible libcuda on the host's 575 driver) with a kit venv (torch 2.13.0+cu130), the engine worktree on PYTHONPATH.
# This is the stack the kits run in-model (FoldForge 5I28 kit record 2026-11-11). It is the only stack where the release's sealed CUDA-13.0
# cubins load (trimul_native row `native`, esm_v61, the ef2_t16 transition cubin of cell pair_c256_n4), so the anthropic timings of
# TriangleMultiplication / TriangleMultiplicationRF3 and of the pair Transition at c 256 x 4 match the kits' in-model calls only here; on
# the engine's own stack (driver 575 = CUDA 12.9) those rows are excluded by name and the provider's next row serves (slower; see
# docs in modules/_anthropic.cuda13_cubins_load).
#
#   srun -A cssb -p h100 -q cssb_h100 --gres=gpu:h100:1 -c 16 --mem=96G -t 00:10:00 \
#     third_party/anthropic/kit_stack.sh python -m miniworld_engine.measure.speed CONFIG --out OUT.jsonl
#
# Environment: KITS_DIR (the kit installs: cu130run.sh, cuda13home, libstdcxx, <kit>/venv130), FF_SCRATCH (cuda13tk), KIT_VENV (default
# $KITS_DIR/esmfold2/venv130: CPython 3.12 as the engine), WORK (caches; default a temp dir), MINIWORLD_ANTHROPIC_ROOT (the release root).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
KITS_DIR=${KITS_DIR:-$HOME/foldforge-scratch/anthropic-kits}
FF_SCRATCH=${FF_SCRATCH:-$HOME/foldforge-scratch}
KIT_VENV=${KIT_VENV:-$KITS_DIR/esmfold2/venv130}
WORK=${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/kit_stack.XXXXXX")}
ROOT=${MINIWORLD_ANTHROPIC_ROOT:-$HERE/upstream}
mkdir -p "$WORK/tmp"
INNER=$WORK/inner_$$.sh
{
  echo "export XDG_CACHE_HOME=$WORK TRITON_CACHE_DIR=$WORK/triton TORCH_EXTENSIONS_DIR=$WORK/torch_ext TORCHINDUCTOR_CACHE_DIR=$WORK/inductor"
  echo "export MODEL_OPT_JIT_ROOT=$WORK/jit TMPDIR=$WORK/tmp PYTHONUNBUFFERED=1 CUDA_HOME=$KITS_DIR/cuda13home"
  echo "export CC=$FF_SCRATCH/cuda13tk/.pixi/envs/default/bin/gcc CXX=$FF_SCRATCH/cuda13tk/.pixi/envs/default/bin/g++"
  echo "export PATH=$KIT_VENV/bin:$KITS_DIR/cuda13home/bin:/usr/bin:/bin"
  echo "export PYTHONPATH=$REPO/src MINIWORLD_ANTHROPIC_ROOT=$ROOT"
  echo ". $KIT_VENV/bin/activate"
  echo "cd $PWD"
  printf '%q ' "$@"
  echo
} > "$INNER"
CU130_EXTRA_LD=$KITS_DIR/libstdcxx:$FF_SCRATCH/cuda13tk/.pixi/envs/default/lib exec "$KITS_DIR/cu130run.sh" bash "$INNER"
