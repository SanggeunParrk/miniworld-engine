#!/bin/bash
#SBATCH -A cssb -p h100 --qos=cssb_h100 --gres=gpu:h100:1 --cpus-per-task=16 --mem=96G --time=00:10:00
#SBATCH --job-name=E5-h100 --output=slurm-%x-%j.log
# Resumable: submit through paper_common/chain.sh. Needs deepspeed / trifast in PAPER_EXT_PATH (README); a missing one is recorded unsupported.
HERE=${EXP_DIR:-$SLURM_SUBMIT_DIR}
. "$HERE/../paper_common/env_h100.sh"; . "$HERE/../paper_common/lib.sh"
# ds4sci JIT-builds its CUTLASS extension: CUTLASS 4.2 headers (>= 3.1.0 as DeepSpeed asks) + the env's nvcc 12.9 (+ ninja from PAPER_EXT_PATH/bin)
export CUTLASS_PATH=${CUTLASS_PATH:-/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/cutlass-4.2} CUDA_HOME=${CUDA_HOME:-$ENV} PATH=$PATH:${PAPER_EXT_PATH:-$HOME/paper-ext}/bin TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
EXT="ds4sci trifast"
stage E5_other_kernels ext cfg_engine.yaml
echo "end $(date +%T)"
