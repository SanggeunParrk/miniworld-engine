#!/bin/bash
# Launch a parallel autotune-cache build: plan -> shard array -> single-writer merge.
#   ./submits/build_autotune_cache_launch.sh <shards_dir> <targets> <impls> <only_ops> <gpu> "<shapes>" ["<extra>"]
set -euo pipefail
cd "$(dirname "$0")/.."
SHARDS="${1:?shards dir}"; TARGETS="${2:?comma targets}"; IMPLS="${3:?comma impls}"
ONLY="${4:-}"; GPU="${5:?gpu_key}"; SHAPES="${6:?hydra shape args}"; EXTRA="${7:-}"
mkdir -p "$SHARDS" /home/psk6950/pfcache/shardlog
CMDFILE="$SHARDS/_cmds.txt"
CONDA_OVERRIDE_CUDA=12.8 pixi run --frozen bash -c "PYTHONPATH=src python submits/build_autotune_cache.py plan --targets \"$TARGETS\" --impls \"$IMPLS\" --shards \"$SHARDS\" --shapes \"$SHAPES\" ${EXTRA:+--extra \"$EXTRA\"}" > "$CMDFILE"
N=$(wc -l < "$CMDFILE")
echo "planned $N shard commands -> $CMDFILE"
AID=$(sbatch --parsable --array=1-"$N" --export=ALL,CMDFILE="$CMDFILE" submits/build_shard.sbatch)
echo "shard array job=$AID"
MJ=$(sbatch --parsable --dependency=afterany:"$AID" --job-name=cachemerge \
  --partition=h100 --account=cssb --qos=normal_h100 --gres=gpu:h100:1 --cpus-per-task=4 --mem=32G \
  --time=00:20:00 --output=/home/psk6950/pfcache/shardlog/merge_%j.out \
  --wrap "cd $(pwd); pixi run --frozen bash -c 'export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; PYTHONPATH=src python submits/build_autotune_cache.py merge --shards $SHARDS --gpu \"$GPU\" ${ONLY:+--only-ops $ONLY}'")
echo "merge job=$MJ (afterany:$AID)"
