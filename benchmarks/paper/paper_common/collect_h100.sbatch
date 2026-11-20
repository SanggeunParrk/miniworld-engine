#!/bin/bash
#SBATCH -A cssb -p h100 --qos=cssb_h100 --gres=none --cpus-per-task=4 --mem=16G --time=00:10:00
#SBATCH --job-name=paper-collect --output=slurm-%x-%j.log
# CPU job: raw records -> E?_h100.csv (all experiments), then the paper table shapes into $OUT (default <paper>/tables_h100).
PAPERDIR=${PAPERDIR:-$SLURM_SUBMIT_DIR}
. "$PAPERDIR/paper_common/env_h100.sh"
for e in E1_operator_speed E2_sweeps E3_accuracy E4_fp8 E5_other_kernels E6_memory; do
  echo "== collect $e"; python "$PAPERDIR/$e/collect.py" --card h100 || echo "collect $e FAILED rc=$?"
done
echo "== tables"; python "$PAPERDIR/paper_common/tables.py" --card h100 --out "${OUT:-$PAPERDIR/tables_h100}"
