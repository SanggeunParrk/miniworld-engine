#!/bin/bash
# usage: sbatch [-J name] job.sh <tree: fp8|base> <command...>   (1 H100, 10 min; shared pixi env on PATH, never pixi run)
#SBATCH --account=cssb --partition=h100 --qos=normal_h100 --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=00:10:00
#SBATCH --output=/home/psk6950/miniworld-engine-scratch/fp8x-opm/logs/%x-%j.log
TREE=$1; shift
W=/home/psk6950/miniworld-engine-scratch/fp8x-opm
case $TREE in
  fp8) SRC=/home/psk6950/miniworld-engine-scratch/wt-fp8x-opm/src ;;
  base) SRC=$W/base/src ;;
  *) echo "tree fp8|base"; exit 2 ;;
esac
ENV=/home/psk6950/miniworld-engine/.pixi/envs/default
export PATH=$ENV/bin:/usr/bin:/bin LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=$SRC
export XDG_CACHE_HOME=$W/cache
export MINIWORLD_ENGINE_HOST_CXX=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++ CXX=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++ CC=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc
export CUTLASS_PATH=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/cutlass-4.2 MINIWORLD_MATHDX_HOME=/home/psk6950/mathdx_dl/extracted/nvidia/mathdx MATHDX_HOME=/home/psk6950/mathdx_dl/extracted/nvidia/mathdx
export MINIWORLD_ANTHROPIC_ROOT=/home/psk6950/miniworld-engine/third_party/anthropic/upstream
export MAX_JOBS=${MAX_JOBS:-8} MINIWORLD_ENGINE_TARGET_ARCH=sm90 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export DEADLINE=$(( $(date +%s) + 540 ))
cd $W/scripts
echo "node $(hostname) tree=$TREE job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES cmd: $*"
"$@"
echo "rc=$?"
