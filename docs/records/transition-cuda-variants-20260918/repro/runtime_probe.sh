#!/bin/bash
set -u
root=/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918
/usr/local/cuda-12.9/bin/nvcc -ccbin /opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++ -arch=sm_90a "$root/runtime_probe.cu" -o "$root/runtime_probe"
/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode=90 "$root/runtime_probe" > "$root/runtime-probe-standalone.log" 2>&1
