#!/bin/bash
# RTX 5090 (cssb-rtx): from a ~/mwe/ copy of the released tree, through the GPU lock. REP=1..3. EXT / NAME / CFG / KIND override (see h100.sbatch).
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../paper_common/env_rtx5090.sh"; . "$HERE/../paper_common/lib.sh"
GPU_SECONDS=${GPU_SECONDS:-3000}
BUDGET=${BUDGET:-$((GPU_SECONDS - 120))}
EXT=${EXT-"ds4sci trifast fast_trimul gpumode geonn gnops of3tri megafold bionemo_ir turbopf ligo_pwa megafold_tr"}
"$HOME/mwe/gpu.sh" "$GPU_SECONDS" bash -c ". \"$HERE/../paper_common/env_rtx5090.sh\"; . \"$HERE/../paper_common/lib.sh\"; EXT='$EXT' BUDGET=$BUDGET REP=$REP EXTRA=\"$EXTRA\" KIND=${KIND:-speed} RECORDS=\"${RECORDS:-}\" stage E5_other_kernels ${NAME:-e5} ${CFG:-cfg_e5_all.yaml}"
