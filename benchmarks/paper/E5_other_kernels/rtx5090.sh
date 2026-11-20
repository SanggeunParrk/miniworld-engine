#!/bin/bash
# RTX 5090 (cssb-rtx): from a ~/mwe/ copy of the released tree, through the GPU lock. REP=1..3.
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../paper_common/env_rtx5090.sh"; . "$HERE/../paper_common/lib.sh"
GPU_SECONDS=${GPU_SECONDS:-3000}
BUDGET=${BUDGET:-$((GPU_SECONDS - 120))}
"$HOME/mwe/gpu.sh" "$GPU_SECONDS" bash -c ". \"$HERE/../paper_common/env_rtx5090.sh\"; . \"$HERE/../paper_common/lib.sh\"; EXT='ds4sci trifast' BUDGET=$BUDGET REP=$REP EXTRA=\"$EXTRA\" RECORDS=\"${RECORDS:-}\" stage E5_other_kernels ext cfg_engine.yaml"
