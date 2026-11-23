#!/bin/bash
#SBATCH -A cssb -p h100 --qos=cssb_h100 --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=00:10:00
#SBATCH --job-name=E4tderr --output=slurm-%x-%j.log
HERE=${EXP_DIR:-$SLURM_SUBMIT_DIR}
. "$HERE/../paper_common/env_h100.sh"
python "$HERE/td_err.py" $EXTRA
