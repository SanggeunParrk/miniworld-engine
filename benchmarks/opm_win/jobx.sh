#!/bin/bash
# job.sh CMD...   engine env (STACK=engine, default) or the kits' CUDA-13 stack (STACK=kit; persistent warm WORK dir), this worktree
WT=/home/psk6950/miniworld-engine-scratch/wt-opm-win
ENV=/home/psk6950/foldjet-engine/.pixi/envs/default
export PYTHONUNBUFFERED=1 FOLDJET_ENGINE_ANTHROPIC_ROOT=/home/psk6950/foldjet-engine/third_party/anthropic/upstream
if [ "${STACK:-engine}" = kit ]; then
  export WORK=/home/psk6950/miniworld-engine-scratch/opm-win-kitwork
  exec $WT/third_party/anthropic/kit_stack.sh "$@"
fi
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=/home/psk6950/miniworld-engine-scratch/opm-win-exp/src
exec "$@"
