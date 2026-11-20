#!/bin/bash
#SBATCH -p A100 --gres=gpu:A100:1 --cpus-per-task=16 --mem=96G --time=00:10:00
#SBATCH --job-name=E6-a100 --output=slurm-%x-%j.log
# Resumable: submit through paper_common/chain.sh (N chained 10-minute jobs; REP=1..3 = repeats in different Slurm jobs).
# Smoke: EXTRA="--module TriangleMultiplication --lengths 128 --max-cases 2" sbatch --export=ALL,REP=1 <this file>
HERE=${EXP_DIR:-$SLURM_SUBMIT_DIR}   # the experiment folder: submit from it, or set EXP_DIR (paper_common/chain.sh does)
. "$HERE/../paper_common/env_a100.sh"; . "$HERE/../paper_common/lib.sh"
KIND=memory
stage E6_memory engine cfg_engine.yaml
stage E6_memory kit cfg_kit.yaml "$KIT_WRAP"
echo "end $(date +%T)"
