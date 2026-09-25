#!/bin/bash
# prof.sh <tag> [L]: nsys timeline + ncu full sets of the fwd and bwd kernels -> profiles/<tag>/
set -e
tag=$1; L=${2:-384}; out=profiles/$tag; mkdir -p $out
nsys profile -o $out/nsys -f true --stats=false python prof_once.py --length $L > /dev/null 2>&1
nsys stats -r cuda_gpu_kern_sum -f csv -o $out/nsys $out/nsys.nsys-rep > /dev/null 2>&1 || true
ncu --set full -k regex:transition_ -c 3 -f -o $out/ncu python prof_once.py --length $L --iters 1 > $out/ncu.log 2>&1
ncu -i $out/ncu.ncu-rep --page details --csv > $out/ncu_details.csv
ncu -i $out/ncu.ncu-rep --page details | grep -E "transition_|Duration|Compute \(SM\) Throughput|Memory Throughput|DRAM Throughput|Registers Per|Achieved Occupancy|L1/TEX Hit|L2 Hit|Executed Ipc|Issue Slots Busy|Tensor|One or More|No Eligible" > $out/ncu_summary.txt
cat $out/ncu_summary.txt
