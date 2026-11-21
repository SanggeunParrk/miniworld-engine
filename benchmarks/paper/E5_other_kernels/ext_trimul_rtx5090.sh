#!/bin/bash
# RTX 5090 (cssb-rtx): through the GPU lock (16 GB RAM cap). REP=1..3; TM_ENV: the environment file, see ext_trimul_run.sh.
HERE=$(cd "$(dirname "$0")" && pwd)
GPU_SECONDS=${GPU_SECONDS:-1500}
~/mwe/gpu.sh "$GPU_SECONDS" env CARD=rtx5090 REP="${REP:-1}" TM_ENV="$TM_ENV" BUDGET=$((GPU_SECONDS - 120)) "$HERE/ext_trimul_run.sh"
