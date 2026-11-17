#!/bin/bash
#SBATCH -p h100 -A cssb --qos=cssb_h100 --gres=gpu:h100:1 --hint=nomultithread -c 8 --mem=96G --time=00:10:00
# usage: sbatch ffgpu.sh SCRIPT ARGS...   (FoldForge pixi env via run_port.sh, engine = the wt-fp8-e2e worktree)
export FF_SRC=/home/psk6950/FoldForge/src ENGINE_SRC=/home/psk6950/miniworld-engine-scratch/wt-fp8-e2e/src
export CUTLASS_PATH=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/cutlass-4.2 MINIWORLD_ENGINE_TARGET_ARCH=sm90
export MINIWORLD_ANTHROPIC_ROOT=/home/psk6950/miniworld-engine/third_party/anthropic/upstream
S=$1; shift
bash /home/psk6950/foldforge-scratch/foldbench/run_port.sh -u /home/psk6950/miniworld-engine-scratch/fp8/e2e/scripts/$S "$@" 2>&1 | grep --line-buffered -av "Warning\|warn"
