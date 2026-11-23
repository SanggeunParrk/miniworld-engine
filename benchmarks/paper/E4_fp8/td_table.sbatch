#!/bin/bash
#SBATCH -A cssb -p h100 --qos=cssb_h100 --gres=none --cpus-per-task=2 --mem=8G --time=00:05:00
#SBATCH --job-name=tdtable --output=/dev/null
. "$EXP_DIR/../paper_common/env_h100.sh"
for f in $FILES; do echo "## $f"; python "$EXP_DIR/td_table.py" "$f"; echo; done > "$OUT"
