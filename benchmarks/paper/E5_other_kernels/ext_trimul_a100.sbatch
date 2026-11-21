#!/bin/bash
#SBATCH -p A100 --gres=gpu:A100:1 --cpus-per-task=16 --mem=96G --time=00:10:00
#SBATCH --job-name=E5tm-a100 --output=slurm-%x-%j.log
# Resumable: submit through paper_common/chain.sh (REP=1..3). TM_ENV: the environment file (installed 3.0.6 wheels), see ext_trimul_run.sh.
HERE=${EXP_DIR:-$SLURM_SUBMIT_DIR}
CARD=a100 "$HERE/ext_trimul_run.sh"
