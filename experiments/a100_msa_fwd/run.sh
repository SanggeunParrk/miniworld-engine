#!/bin/bash
# run.sh <cmd...> : one A100 via srun, inside env.sh (e.g. ./run.sh python bench.py --op opm)
cd "$(dirname "$0")"
exec srun -p A100 --gres=gpu:1 --cpus-per-task=8 --mem=48G -t 30 ./env.sh "$@"
