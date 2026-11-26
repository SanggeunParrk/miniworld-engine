#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH -o /home/psk6950/miniworld-engine-scratch/pwa-long-l/bench_%j.log
cd ~/miniworld-engine-scratch/pwa-long-l
export PATH=~/practice/miniworld-engine/.pixi/envs/default/bin:$PATH
export PYTHONPATH=$PWD/src
export MAX_JOBS=8
nvidia-smi -L
python -c "import miniworld_engine;print(miniworld_engine.__file__)"
for cfg in "64 128 8" "64 128 32" "64 256 32" "128 256 16" "128 384 8"; do
  python pwa_long_bench.py $cfg 2048 2176 3072 4096 2>&1 | grep -v Warning
done
