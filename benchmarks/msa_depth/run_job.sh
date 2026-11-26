#!/bin/bash
# One 1-GPU 10-minute job: python bench_msa_depth.py ARGS (stack: the engine env, or the kits' CUDA-13 stack with STACK=kit).
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
ENV=/home/psk6950/foldjet-engine/.pixi/envs/default
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=$WT/src PYTHONUNBUFFERED=1
export FOLDJET_ENGINE_ANTHROPIC_ROOT=${FOLDJET_ENGINE_ANTHROPIC_ROOT:-/home/psk6950/foldjet-engine/third_party/anthropic/upstream}
RUN=""
[ "${STACK:-engine}" = kit ] && RUN=$WT/third_party/anthropic/kit_stack.sh
exec $RUN python $WT/benchmarks/msa_depth/bench_msa_depth.py "$@"
