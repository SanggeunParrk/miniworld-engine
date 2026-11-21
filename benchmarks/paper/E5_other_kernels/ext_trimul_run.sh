#!/bin/bash
# One resumable stage of the E5 triangle-multiplication run on this card (the caller sets the environment: TM_ENV = a file that puts the installed
# engine wheels on PYTHONPATH and cds to benchmarks/paper; CARD = h100 | a100 | rtx5090; REP = 1..3, a different Slurm job / ssh run per repeat).
#   REP=1 CARD=h100 TM_ENV=... E5_other_kernels/ext_trimul_run.sh          (through paper_common/chain.sh as the sbatch script's body, see ext_trimul_h100.sbatch)
set -uo pipefail
. "${TM_ENV:?TM_ENV: the environment file of this card}"
REP=${REP:-1}; CARD=${CARD:?CARD}; BUDGET=${BUDGET:-480}
OUT=${RECORDS:-E5_other_kernels/records}/$CARD/ext_trimul.r$REP.jsonl
echo "== E5 ext_trimul $CARD rep $REP $(date +%T)"
python E5_other_kernels/run_ext_trimul.py E5_other_kernels/cfg_trimul.yaml --kind speed --out "$OUT" --budget-seconds "$BUDGET" --ext fast_trimul gpumode geonn ${EXTRA:-}
echo "end $(date +%T)"
