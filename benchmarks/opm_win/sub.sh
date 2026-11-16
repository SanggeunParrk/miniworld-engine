#!/bin/bash
# sub.sh gpu|cpu NAME [--dependency=..] -- CMD...   one job with the explicit limits (1 GPU 64G 8c 10 min | no GPU 16G 4c 10 min)
D=$(cd "$(dirname "$0")" && pwd); K=$1; N=$2; shift 2
DEP=""; if [ "${1:-}" != "--" ]; then DEP=$1; shift; fi; shift
if [ "$K" = gpu ]; then R="--gres=gpu:h100:1 --mem=64G --cpus-per-task=8"; else R="--gres=none --mem=16G --cpus-per-task=4"; fi
sbatch --parsable -A cssb -p h100 -q cssb_h100 $R --time=00:10:00 $DEP -J ow-$N -o $D/logs/$N-%j.log --export=ALL --wrap="$D/job.sh $*"
