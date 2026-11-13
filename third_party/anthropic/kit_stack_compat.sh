#!/bin/bash
# Run a command (e.g. the speed measurer on the anthropic backend) on the model kits' stack of a card whose host needs no container: the
# engine's own environment (already on PATH / LD_LIBRARY_PATH / PYTHONPATH) with the CUDA-13.0 forward-compatible libcuda in front of the
# host's driver library, so the release's sealed CUDA-13.0 cubins load (A100: trimul_native's sm_80 member, row `native`, the kits' fast
# TriMul row from L 768). The A100 nodes of cssb3 (Ubuntu 24.04, driver 570.211.01 = CUDA 12.8) run it this way: cuDriverGetVersion
# becomes 13000 and the package's byte gate passes. The release's sm_80 triangle-attention extension (triattn_sm80_ext, the kits' fast
# attention core on an A100) is built for this stack once by `ARCH=sm80 third_party/anthropic/build_local.sh` (an A100 job).
#
# Checked against the kits' own venv (torch 2.13.0+cu130, same compat libcuda) on one A100 80GB PCIe, 2026-11-13: TriMul d_pair 64 / 128 /
# 256 and TriAttn (4 rows) at L 768 / 1024 within 2 % of each other, either sign (run-to-run noise) -- see LOCAL_BUILD.json.
#
#   sbatch -p A100 --gres=gpu:A100:1 ... --wrap "third_party/anthropic/kit_stack_compat.sh python -m miniworld_engine.measure.speed CONFIG --out OUT.jsonl"
#
# Environment: KIT_COMPAT_DIR (the directory of the CUDA-13.0 forward-compatible libcuda.so.1: NVIDIA's cuda-compat-13-0 package,
# .../cuda-13.0/compat; default $HOME/cuda-compat-13.0), MINIWORLD_ANTHROPIC_ROOT (the release root, with the sm_80 extension installed).
set -euo pipefail
COMPAT=${KIT_COMPAT_DIR:-$HOME/cuda-compat-13.0}
[ -e "$COMPAT/libcuda.so.1" ] || { echo "kit_stack_compat.sh: no $COMPAT/libcuda.so.1 (set KIT_COMPAT_DIR to the CUDA-13.0 compat directory)" >&2; exit 2; }
export LD_LIBRARY_PATH=$COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec "$@"
