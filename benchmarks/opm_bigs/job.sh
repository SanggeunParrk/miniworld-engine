#!/bin/bash
# 1-GPU 10-min job runner: job.sh CMD...   (engine env, this worktree)
WT=/home/psk6950/miniworld-engine-scratch/wt-opm-bigs
ENV=/home/psk6950/foldjet-engine/.pixi/envs/default
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=$WT/src PYTHONUNBUFFERED=1
export FOLDJET_ENGINE_ANTHROPIC_ROOT=/home/psk6950/foldjet-engine/third_party/anthropic/upstream
exec "$@"
