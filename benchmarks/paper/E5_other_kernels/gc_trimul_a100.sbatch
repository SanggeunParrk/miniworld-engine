#!/bin/bash
#SBATCH -p A100 --gres=gpu:A100:1 --cpus-per-task=16 --mem=96G --time=00:10:00
#SBATCH --job-name=E5gc-a100 --output=slurm-%x-%j.log
# graph check of the TriMul candidates only (their trees were not installed in the first pass): CARDENV=env_a100.sh for A100 (see gc_a100.sbatch for the header)
HERE=${EXP_DIR:-$SLURM_SUBMIT_DIR}
CARDENV=${CARDENV:-env_a100.sh}
. "$HERE/../paper_common/$CARDENV"
cd "$HERE"
python graph_check.py cfg_e5_all.yaml --out records/$CARD/graphcheck_trimul.r1.jsonl --budget-seconds 420 --redo-unsupported --ext fast_trimul gpumode geonn --module TriangleMultiplication
