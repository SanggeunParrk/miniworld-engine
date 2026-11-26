#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH -o /home/psk6950/miniworld-engine-scratch/pwa-long-l/test_%j.log
cd ~/miniworld-engine-scratch/pwa-long-l
export PATH=~/practice/miniworld-engine/.pixi/envs/default/bin:$PATH
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1
python -m pytest -v --durations=40 tests/gpu/test_pwa_long_inference_sm80.py 2>&1 | grep -v Warn | tail -80
