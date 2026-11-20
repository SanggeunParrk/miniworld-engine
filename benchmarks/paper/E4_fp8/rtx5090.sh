#!/bin/bash
# RTX 5090 (cssb-rtx): the engine has no sm120 FP8 route (miniworld_engine/fp8.py: the routes take inference calls on H100 sm_90a only), so
# `engine-fp8` resolves to the default `cuda` kernels there. This script still runs the pair so the record says so (routes_fired False,
# ratio ~ 1); collect.py marks those rows "no FP8 route on this card". Run from a ~/mwe/ copy of the released tree, through the GPU lock.
#   REP=1 ./rtx5090.sh
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../paper_common/env_rtx5090.sh"; . "$HERE/../paper_common/lib.sh"
GPU_SECONDS=${GPU_SECONDS:-1800}
BUDGET=${BUDGET:-$((GPU_SECONDS - 120))}
for g in shipped none; do
  "$HOME/mwe/gpu.sh" "$GPU_SECONDS" bash -c ". \"$HERE/../paper_common/env_rtx5090.sh\"; . \"$HERE/../paper_common/lib.sh\"; BUDGET=$BUDGET REP=$REP EXTRA=\"$EXTRA\" RECORDS=\"${RECORDS:-}\" fp8_stage $g"
done
