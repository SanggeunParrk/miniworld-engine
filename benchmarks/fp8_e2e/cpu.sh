#!/bin/bash
#SBATCH -A cssb -p h100 -q cssb_h100 --gres=none --hint=nomultithread -c 4 --mem=24G --time=00:10:00
ENV=/home/psk6950/miniworld-engine/.pixi/envs/default
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
cd /home/psk6950/miniworld-engine-scratch/fp8/e2e/scripts
exec python -u "$@"
