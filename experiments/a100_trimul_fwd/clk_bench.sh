#!/bin/bash
# clk_bench.sh <train_bench args>: sample SM clock / power / throttle reasons (200 ms) while train_bench runs
f=$(mktemp); nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader -lms 200 > $f & P=$!
python train_bench.py "$@" 2>&1 | grep -E '^(bidir|single)' | grep -o 'L[0-9]*:\|fwd+bwd.*'
kill $P; awk -F", " '{split($1,a," "); split($2,b," "); if (b[1] > 150) {n++; c+=a[1]; w+=b[1]; print}}' $f | sort | uniq -c | sort -k1 -n -r | head -15; rm -f $f
