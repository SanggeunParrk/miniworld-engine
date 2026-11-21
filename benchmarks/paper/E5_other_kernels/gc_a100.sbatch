#!/bin/bash
#SBATCH -p A100 --gres=gpu:A100:1 --cpus-per-task=16 --mem=96G --time=00:10:00
#SBATCH --job-name=E5gc-a100 --output=slurm-%x-%j.log
# CUDA-graph validity check of every candidate (graph_check.py), resumable: chain.sh. a100: gc_a100.sbatch (env_a100.sh).
# Two invocations: ds4sci needs CUTLASS_PATH / CUDA_HOME for its JIT (graphcheck_ds4sci), all others run without them (turbopf's prebuilt extension is keyed on the build environment).
HERE=${EXP_DIR:-$SLURM_SUBMIT_DIR}
CARDENV=${CARDENV:-env_a100.sh}
. "$HERE/../paper_common/$CARDENV"
cd "$HERE"
SECONDS=0
OTHERS="trifast fast_trimul gpumode geonn gnops of3tri megafold bionemo_ir turbopf ligo_pwa megafold_tr"
python graph_check.py cfg_e5_all.yaml --out records/$CARD/graphcheck.r1.jsonl --budget-seconds ${BUDGET:-400} --ext $OTHERS ${EXTRA:-}
(export CUTLASS_PATH=${CUTLASS_PATH:-/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/cutlass-4.2} CUDA_HOME=${CUDA_HOME:-$ENV} PATH=$PATH:${PAPER_EXT_PATH:-$HOME/paper-ext}/bin TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
 python graph_check.py cfg_e5_all.yaml --out records/$CARD/graphcheck_ds4sci.r1.jsonl --budget-seconds $((${BUDGET:-400} > SECONDS + 30 ? ${BUDGET:-400} - SECONDS : 30)) --ext ds4sci --module TriangleAttention --backend ds4sci ${EXTRA:-})
