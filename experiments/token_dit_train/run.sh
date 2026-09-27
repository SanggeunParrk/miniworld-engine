#!/bin/bash
export PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:$PATH
export LD_LIBRARY_PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}
cd /home/psk6950/miniworld-engine-dit2/experiments/token_dit_train
python -W ignore "$@"
