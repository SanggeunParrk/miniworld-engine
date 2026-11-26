#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o /home/psk6950/miniworld-engine-scratch/pwa-long-l/bench2_%j.log
cd ~/miniworld-engine-scratch/pwa-long-l
export PATH=~/practice/miniworld-engine/.pixi/envs/default/bin:$PATH
export PYTHONPATH=$PWD/src
python pwa_long_bench.py 64 128 8 2304 2560 3200 2>&1 | grep -v Warn
python pwa_long_bench.py 128 384 8 2304 3200 2>&1 | grep -v Warn
python -m pytest -q -x tests/gpu/test_pwa_long_inference.py tests/gpu/test_pwa_long_inference_sm80.py -k "not 4096 or 128-128-8" -p no:cacheprovider 2>&1 | tail -8
