#!/bin/bash
# usage: sub.sh JOBNAME MODEL OUTDIR MODES TS FORCE "TARGETS"   (FORCE: kinds or -)
J=$1; M=$2; O=$3; MO=$4; TS=$5; F=$6; shift 6
export E2E_MODEL=$M E2E_MODES=$MO E2E_TS=$TS
[ "$F" != "-" ] && export E2E_FORCE=$F
cd /home/psk6950/miniworld-engine-scratch/fp8/e2e
sbatch --export=ALL -J $J -o logs/$J.%j.log scripts/ffgpu.sh e2e2.py /home/psk6950/miniworld-engine-scratch/fp8/e2e/res/$O "$@"
