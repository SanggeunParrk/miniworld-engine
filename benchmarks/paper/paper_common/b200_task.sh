#!/bin/bash
# b200_task.sh EXP REP STAGE : one queue task on this process's GPU (CUDA_VISIBLE_DEVICES set by the lane), unlimited budget (no Slurm job limit);
# the same stages as the h100.sbatch of each experiment, records/b200/<stage>.r<REP>.jsonl (+ .done when complete; resumable).
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/env_b200.sh"; . "$HERE/lib.sh"
exp=$1; REP=$2; st=$3; ONLY=$st; BUDGET=0
export REP ONLY BUDGET
case "$exp" in
  E3_accuracy) KIND=accuracy;;
  E6_memory) KIND=memory;;
  *) KIND=speed;;
esac
export KIND
case "$exp:$st" in
  E4_fp8:*) fp8_stage "$st";;
  E5_other_kernels:ext) export CUTLASS_PATH CUDA_HOME PATH=$PATH:${PAPER_EXT_PATH:-$W/paper-ext}/bin TORCH_CUDA_ARCH_LIST=10.0
    EXT=${EXT-"ds4sci trifast fast_trimul gpumode geonn gnops of3tri megafold bionemo_ir turbopf ligo_pwa megafold_tr"}
    stage "$exp" ext cfg_e5_all.yaml;;
  *:engine) stage "$exp" engine cfg_engine.yaml;;
  *:kit) stage "$exp" kit cfg_kit.yaml "$KIT_WRAP";;
  *) echo "unknown task $exp $st"; exit 2;;
esac
echo "end $(date +%T)"
