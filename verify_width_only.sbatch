#!/bin/bash
#SBATCH -J width-only
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -w gpu05
#SBATCH -o /home/psk6950/practice/miniworld-engine/verify_width_only.log
#SBATCH -t 1:00:00
cd /home/psk6950/practice/miniworld-engine
.pixi/envs/default/bin/python -u verify_width.py
