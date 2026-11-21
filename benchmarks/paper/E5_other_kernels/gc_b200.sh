#!/bin/bash
# CUDA-graph validity check of every E5 candidate on the B200 (graph_check.py; counterpart of gc_h100.sbatch, no Slurm). One GPU:
#   CUDA_VISIBLE_DEVICES=<0-3> E5_other_kernels/gc_b200.sh        (resumable; writes records/b200/graphcheck*.r1.jsonl)
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../paper_common/env_b200.sh"
cd "$HERE"
export TORCH_CUDA_ARCH_LIST=10.0 PATH=$PATH:$PAPER_EXT_PATH/bin
OTHERS="trifast fast_trimul gpumode geonn gnops of3tri megafold bionemo_ir turbopf ligo_pwa megafold_tr"
python graph_check.py cfg_e5_all.yaml --out records/$CARD/graphcheck.r1.jsonl --ext $OTHERS ${EXTRA:-}
python graph_check.py cfg_e5_all.yaml --out records/$CARD/graphcheck_ds4sci.r1.jsonl --ext ds4sci --module TriangleAttention --backend ds4sci ${EXTRA:-}
echo "end $(date +%T)"
