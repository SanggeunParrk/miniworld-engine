#!/bin/bash
# jobb.sh CMD...   engine env on the BEFORE snapshot (git archive of opm-bigs HEAD 0ac874471: src only)
ENV=/home/psk6950/foldjet-engine/.pixi/envs/default
export PYTHONUNBUFFERED=1 FOLDJET_ENGINE_ANTHROPIC_ROOT=/home/psk6950/foldjet-engine/third_party/anthropic/upstream
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=/home/psk6950/miniworld-engine-scratch/opm-win-before/src
exec "$@"
