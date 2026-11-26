#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH -o /home/psk6950/miniworld-engine-scratch/pwa-long-l/wide_%j.log
cd ~/miniworld-engine-scratch/pwa-long-l
export PATH=~/practice/miniworld-engine/.pixi/envs/default/bin:$PATH
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1
for L in 1024 2048 2176 3072 4096; do
python pwa_long_bench.py 256 512 32 $L 2>&1 | grep -v Warn
done
