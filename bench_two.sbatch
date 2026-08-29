#!/bin/bash
#SBATCH -J bench-adaln-ct
#SBATCH -p gpu
#SBATCH --gres=gpu:6
#SBATCH -c 48
#SBATCH -w gpu05
#SBATCH -o /home/psk6950/practice/miniworld-engine/bench_two.log
#SBATCH -t 12:00:00
cd /home/psk6950/practice/miniworld-engine
BIN=.pixi/envs/default/bin/miniworld-engine
[ -x "$BIN" ] || BIN=".pixi/envs/default/bin/python -m miniworld_engine.cli"
for target in adaptive_layernorm conditioned_transition; do
  for mode in inference training; do
    echo "############ $target / $mode"
    $BIN bench_module "$target" --mode "$mode" --gpus all --resume \
      --bench-rep-ms 25 --predict-unusable --prune-cache 2>&1 | tail -40
    echo "############ exit=$?"
  done
done
