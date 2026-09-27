#!/bin/bash
set -u
root=/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918
bash "$root/env.sh" --memcheck "$root/memcheck_norm.py" > "$root/memcheck-norm-default.log" 2>&1
LD_PRELOAD=/usr/local/cuda-12.9/lib64/libcudart.so.12 bash "$root/env.sh" --memcheck "$root/memcheck_norm.py" > "$root/memcheck-norm-preload.log" 2>&1
