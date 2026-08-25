#!/bin/bash
#SBATCH -J tail-units
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -w gpu01
#SBATCH --gres=gpu:3
#SBATCH -c 48
#SBATCH -t 04:00:00
#SBATCH -o src/miniworld_engine/tools/prof/tail-%j.log
set -u
R=${MINIWORLD_REPO:-$(git rev-parse --show-toplevel)}
cd "$R"
export PYTHONPATH=src
export TRITON_CACHE_DIR=${MINIWORLD_TRITON_CACHE:-$HOME/.cache/miniworld-triton}
OUT=$R/src/miniworld_engine/tools/prof
PY=.pixi/envs/default/bin/python

# The three slowest COMPLETED sweep units, all measured under the old fork-per-config guard:
#   bias_only_attention_bwd_triton  L=128  3654 s   (guard 2099 s of it)
#   adaln_epilogue_triton           L=256  4156 s   (94 configs fail to compile)
#   layernorm_linear_fwd_fp32       L=128  2180 s
one () {  # one <gpu> <op> <len> <baseline>
  ( export CUDA_VISIBLE_DEVICES=$1 MINIWORLD_DRIVER_LENGTH=$3
    rm -f $OUT/tail-$2-$3-$SLURM_JOB_ID.*
    T0=$SECONDS
    $PY -u -m miniworld_engine.autotune.builder \
      --shard $OUT/tail-$2-$3-$SLURM_JOB_ID.json --compile-jobs 16 --bench-budget 3.0 \
      --op $2 --dtype bfloat16 --length $3 > $OUT/tail-$2-$3.txt 2>&1
    echo "### $2 L=$3 : WALL $((SECONDS-T0)) s   (was $4 s)" ) &
}

one 0 bias_only_attention_bwd_triton   128 3654
one 1 adaln_epilogue_triton            256 4156
one 2 layernorm_linear_fwd_fp32_triton 128 2180
wait
echo "===== breakdowns ====="
for t in bias_only_attention_bwd_triton-128 adaln_epilogue_triton-256 layernorm_linear_fwd_fp32_triton-128; do
  echo "--- $t"
  grep -E 'precompile|bench-budget|compile-guard\]|first-launch' $OUT/tail-${t%-*}-${t##*-}.txt 2>/dev/null | sed 's/^ *//'
done
