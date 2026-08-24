#!/bin/bash
#SBATCH -J forkless
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -w gpu01
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -o /home/psk6950/practice/miniworld-engine/tools/kernel-audit/prof/forkless-%j.log
set -u
R=/home/psk6950/practice/miniworld-engine
cd $R
export PYTHONPATH=src
export MINIWORLD_CONFIG_DIR=$R/src/miniworld_engine/autotune/configs/grid
export TRITON_CACHE_DIR=/public_data02/psk6950/tritoncache
OUT=$R/tools/kernel-audit/prof
PY=.pixi/envs/default/bin/python
OP=adaln_bwd_dx_dlnw_triton

echo "### host $(hostname)  $(date +%H:%M:%S)"
# Fresh shard stem: a leftover .compiled from the pre-fix profile run would skip the round and
# hide the effect. This is the state a NEW unit starts in, which is the 499 still queued.
rm -f $OUT/fk-$SLURM_JOB_ID.*
T0=$SECONDS
$PY -u -m miniworld_engine.autotune.builder \
    --shard $OUT/fk-$SLURM_JOB_ID.json \
    --compile-jobs 8 --bench-budget 3.0 \
    --op $OP --dtype bfloat16 --length 1024
echo "### WALL(fresh) $((SECONDS-T0)) s"

echo ""
echo "### RESUME (second attempt on the same shard -- must be forkless from the tags) $(date +%H:%M:%S)"
T0=$SECONDS
$PY -u -m miniworld_engine.autotune.builder \
    --shard $OUT/fk-$SLURM_JOB_ID.json \
    --compile-jobs 8 --bench-budget 3.0 \
    --op $OP --dtype bfloat16 --length 1024
echo "### WALL(resume) $((SECONDS-T0)) s"
