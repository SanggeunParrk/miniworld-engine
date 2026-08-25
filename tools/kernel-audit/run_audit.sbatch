#!/bin/bash
#SBATCH -J audit
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 00:40:00
#SBATCH -o tools/kernel-audit/prof/audit-%j.log
set -u
R=${MINIWORLD_REPO:-$(git rev-parse --show-toplevel)}
cd "$R"
export PYTHONPATH=src
export TRITON_CACHE_DIR=${MINIWORLD_TRITON_CACHE:-$HOME/.cache/miniworld-triton}
.pixi/envs/default/bin/python -u -m miniworld_engine.cli audit \
  --shards ${MINIWORLD_SHARDS:-$HOME/.cache/miniworld-build}
