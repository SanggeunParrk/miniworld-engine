#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o /home/psk6950/miniworld-engine-scratch/pwa-long-l/ab_%j.log
cd ~/miniworld-engine-scratch/pwa-long-l
export PATH=~/practice/miniworld-engine/.pixi/envs/default/bin:$PATH
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1
for cfg in "128 384 8" "64 128 8"; do
python ../pwa_ab_old_new.py $cfg 1024 1536 2048 1024 1536 2048 2>&1 | grep -v Warn
done
