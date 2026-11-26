#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH -o /home/psk6950/miniworld-engine-scratch/pwa-long-l/edge2_%j.log
cd ~/miniworld-engine-scratch/pwa-long-l
export PATH=~/practice/miniworld-engine/.pixi/envs/default/bin:$PATH
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1
python ../pwa_edge.py 64 128 8 2050 4104 8192 8208 8704 2>&1 | grep -v Warn | cut -c1-300
python ../pwa_edge.py 128 384 8 2064 8192 8208 2>&1 | grep -v Warn | cut -c1-300
