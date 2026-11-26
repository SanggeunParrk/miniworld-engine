#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o /home/psk6950/miniworld-engine-scratch/pwa-long-l/big2_%j.log
cd ~/miniworld-engine-scratch/pwa-long-l
export PATH=~/practice/miniworld-engine/.pixi/envs/default/bin:$PATH
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1
for cfg in "64 256 32:8192" "128 384 8:6144" "128 384 8:7168" "128 384 8:8192" "256 512 32:5120" "256 512 32:6144" "256 512 32:7168" "256 512 32:8192"; do
python pwa_long_bench.py ${cfg%:*} ${cfg#*:} 2>&1 | grep -v Warn | cut -c1-400
done
