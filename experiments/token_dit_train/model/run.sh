#!/bin/bash
export PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/bin:$PATH
export LD_LIBRARY_PATH=/home/psk6950/MiniWorld/.pixi/envs/cu128/lib:${LD_LIBRARY_PATH:-}
cd /home/psk6950/MiniWorld/runs/token_dit_bwd_20260926
python -W ignore "$@"
