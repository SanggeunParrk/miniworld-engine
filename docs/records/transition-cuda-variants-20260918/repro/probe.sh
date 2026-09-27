#!/bin/bash
set -euo pipefail
root=/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918
bash "$root/env.sh" "$root/smoke.py" --variant "$1" --d "$2"
bash "$root/env.sh" "$root/quick.py" --variant "$1" --d "$2"
