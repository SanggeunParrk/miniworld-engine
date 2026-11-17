#!/bin/bash
# bit.py on the base tree, then on the fp8x/opm tree (separate processes, FP8 env unset), same cases
W=/home/psk6950/miniworld-engine-scratch/fp8x-opm
unset MINIWORLD_OPM_FP8
PYTHONPATH=$W/base/src python bit.py base "$@"
PYTHONPATH=/home/psk6950/miniworld-engine-scratch/wt-fp8x-opm/src python bit.py fp8 "$@"
