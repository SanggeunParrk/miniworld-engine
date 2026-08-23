#!/bin/bash
#SBATCH -J audit
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 00:40:00
#SBATCH -o /home/psk6950/practice/miniworld-engine/tools/kernel-audit/prof/audit-%j.log
set -u
R=/home/psk6950/practice/miniworld-engine
cd $R
export PYTHONPATH=src
export MINIWORLD_CONFIG_DIR=$R/configs/grid
export TRITON_CACHE_DIR=/public_data02/psk6950/tritoncache
.pixi/envs/default/bin/python -u -m miniworld_engine.cli audit \
  --shards /public_data02/psk6950/mwsweep
