#!/bin/bash
#SBATCH -J shapekey
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -w gpu03,gpu04,gpu05
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -o .bench/_shapekey_sweep/slurm-%A_%a.out
#SBATCH -e .bench/_shapekey_sweep/slurm-%A_%a.out
set -u
R=${MINIWORLD_REPO:-$(git rev-parse --show-toplevel)}
cd "$R"
export PYTHONPATH=src
exec .pixi/envs/default/bin/python tools/kernel-audit/driver_shape_key_sweep.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS"
