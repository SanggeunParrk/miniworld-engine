#!/bin/bash
# RTX 5090 (cssb-rtx): the opt-in e4m3-plane TriMul route (kernels/trimul_inproj/cuda/sm120_fp8.py: k1w8 front + cuBLASLt e4m3 contraction, docs/fp8.md "RTX 5090")
# takes `engine-fp8` TriMul calls (D128 / D256 / D384, L >= 256, L % 32 == 0); TriAttn has no sm120 FP8 route (its rows run the default kernels: routes_fired False,
# ratio ~ 1, marked "no FP8 route on this card" by collect.py). Run from a ~/mwe/ copy of the source tree, through the GPU lock.
#   REP=1 ./rtx5090.sh          (EXTRA="--module TriangleMultiplication" for the TriMul rows only)
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../paper_common/env_rtx5090.sh"; . "$HERE/../paper_common/lib.sh"
GPU_SECONDS=${GPU_SECONDS:-1800}
BUDGET=${BUDGET:-$((GPU_SECONDS - 120))}
for g in shipped none; do
  "$HOME/mwe/gpu.sh" "$GPU_SECONDS" bash -c ". \"$HERE/../paper_common/env_rtx5090.sh\"; . \"$HERE/../paper_common/lib.sh\"; BUDGET=$BUDGET REP=$REP EXTRA=\"$EXTRA\" RECORDS=\"${RECORDS:-}\" fp8_stage $g"
done
