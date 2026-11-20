#!/bin/bash
# RTX 5090 (cssb-rtx, no Slurm). Run from a ~/mwe/ copy of the released tree (never ~/mwe/engine), one measurement at a time through the
# GPU lock: ~/mwe/gpu.sh SECONDS cmd (flock + 16 GB cap). Never build and time at once: run the prebuild first if the tree is new.
#   REP=1 ./rtx5090.sh            # repeat 1 (REP=1..3: three separate runs); resumable: run it again to continue
# Smoke: EXTRA="--module TriangleMultiplication --lengths 128 --max-cases 2" RECORDS=$HOME/mwe/paper-smoke ./rtx5090.sh
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../paper_common/env_rtx5090.sh"; . "$HERE/../paper_common/lib.sh"
KIND=speed
GPU_SECONDS=${GPU_SECONDS:-3000}
BUDGET=${BUDGET:-$((GPU_SECONDS - 120))}
stage_gpu() { "$HOME/mwe/gpu.sh" "$GPU_SECONDS" bash -c ". \"$HERE/../paper_common/env_rtx5090.sh\"; . \"$HERE/../paper_common/lib.sh\"; KIND=$KIND BUDGET=$BUDGET REP=$REP EXTRA=\"$EXTRA\" RECORDS=\"${RECORDS:-}\" stage $*"; }
stage_gpu E1_operator_speed engine cfg_engine.yaml
stage_gpu E1_operator_speed kit cfg_kit.yaml
