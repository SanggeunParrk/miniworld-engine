#!/bin/bash
#SBATCH -J numerical
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o tools/kernel-audit/prof/numerical-%j.log
set -u
R=${MINIWORLD_REPO:-$(git rev-parse --show-toplevel)}
cd "$R"
export PYTHONPATH=src
export TRITON_CACHE_DIR=${MINIWORLD_TRITON_CACHE:-$HOME/.cache/miniworld-triton}
.pixi/envs/default/bin/python -m pytest tests/test_numerical.py -m gpu -q -p no:cacheprovider
