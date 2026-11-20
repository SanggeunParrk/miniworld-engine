#!/bin/bash
#SBATCH -A cssb -p h100 --qos=cssb_h100 --gres=gpu:h100:1 --cpus-per-task=16 --mem=96G --time=00:10:00
#SBATCH --job-name=E4-h100 --output=slurm-%x-%j.log
# Resumable: submit through paper_common/chain.sh (REP=1..3 = repeats in different Slurm jobs). Two passes: production gates, then every gate removed.
# Smoke: EXTRA="--module TriangleMultiplication --lengths 256 --max-cases 2" sbatch --export=ALL,REP=1 <this file>
HERE=${EXP_DIR:-$SLURM_SUBMIT_DIR}   # the experiment folder: submit from it, or set EXP_DIR (paper_common/chain.sh does)
. "$HERE/../paper_common/env_h100.sh"; . "$HERE/../paper_common/lib.sh"
fp8_stage shipped
fp8_stage none
echo "end $(date +%T)"
