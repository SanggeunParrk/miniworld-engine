#!/bin/bash
#SBATCH -A cssb -p h100 --qos=cssb_h100 --gres=none --cpus-per-task=2 --mem=8G --time=00:05:00
#SBATCH --job-name=tdslow --output=/dev/null
. "$EXP_DIR/../paper_common/env_h100.sh"
python "$EXP_DIR/${PYF:-td_slow.py}" $FILES > "$OUT"
