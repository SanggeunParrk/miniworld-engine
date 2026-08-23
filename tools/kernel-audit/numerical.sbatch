#!/bin/bash
#SBATCH -J numerical
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o /home/psk6950/practice/miniworld-engine/tools/kernel-audit/prof/numerical-%j.log
set -u
R=/home/psk6950/practice/miniworld-engine
cd $R
export PYTHONPATH=src
export MINIWORLD_CONFIG_DIR=$R/configs/blk64
export TRITON_CACHE_DIR=/public_data02/psk6950/tritoncache
.pixi/envs/default/bin/python -m pytest tests/test_numerical.py -m gpu -q -p no:cacheprovider
