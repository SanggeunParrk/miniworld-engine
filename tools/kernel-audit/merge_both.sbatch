#!/bin/bash
#SBATCH -J merge-cache
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o tools/kernel-audit/prof/merge-%j.log
set -u
R=${MINIWORLD_REPO:-$(git rev-parse --show-toplevel)}
cd "$R"
export PYTHONPATH=src
export MINIWORLD_CONFIG_DIR=$R/configs/grid
# merge_shards keys the output file on gpu_key() of the machine running it, so each cache must be
# merged ON its own card: A6000 shards on an A6000, A5000 shards on an A5000.
.pixi/envs/default/bin/python -u -m miniworld_engine.cli merge --shards "$1" --top-k 5
