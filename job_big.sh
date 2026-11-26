#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o /home/psk6950/miniworld-engine-scratch/pwa-long-l/big_%j.log
cd ~/miniworld-engine-scratch/pwa-long-l
export PATH=~/practice/miniworld-engine/.pixi/envs/default/bin:$PATH
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1
for cfg in "64 128 8" "64 256 32" "128 384 8"; do
for L in 5120 6144 8192; do
python pwa_long_bench.py $cfg $L 2>&1 | grep -v Warn | cut -c1-400
done; done
