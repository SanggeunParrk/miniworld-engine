#!/bin/bash
# Build shipped autotune caches by INSTRUMENTING the Triton autotuner during real module runs
# (capture builder), instead of hand-replicating each kernel launch. For each target module we
# run the existing bench harness (miniworld backend, inference+training, seq_len + d_pair
# sweeps) with:
#   MINIWORLD_RUN_AUTOTUNE=1      -> full grid is benched (no cached top-K narrowing)
#   MINIWORLD_AUTOTUNE_CAPTURE=1  -> bench.py installs capture, records per-config timings,
#                                    flushes top-5 per (op,dtype,bucket) to the runtime cache.
# Then copy runtime cache -> committed package data. cute/cuda are sm90+ (not built here).
#
#   CAPTURE_TARGET=transition sbatch submits/run_autotune_capture_a100.sbatch   # validate first
#   CAPTURE_TARGET=all        sbatch submits/run_autotune_capture_a100.sbatch
#SBATCH --job-name=mwk_autotune_capture
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=benchmarks/autotune_capture_a100_%x_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels
export HYDRA_FULL_ERROR=1
echo "host=$(hostname) date=$(date) target=${CAPTURE_TARGET:-transition}"
pixi run --frozen python -c "import torch; cap=torch.cuda.get_device_capability(0); assert cap[0]==8, cap; print('A100 ok', torch.__version__)"

run_one() {
  target="$1"; shift
  echo ""
  echo "### capture bench kernel=${target} $* ###"
  pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-} MINIWORLD_RUN_AUTOTUNE=1 MINIWORLD_AUTOTUNE_CAPTURE=1; PYTHONPATH=src python benchmarks/runners/bench.py "$@"' _ \
    kernel="$target" implementations='[miniworld]' metric=time compile=false cudagraph=manual "$@"
  echo "[exit=$?]"
}

# lean matrix: both modes, both sweep axes, single (manual) graph mode, no plotting.
run_matrix() {
  target="$1"; extra="$2"
  for bench_mode in inference training; do
    run_one "$target" mode="$bench_mode" sweep_axis=seq_len name_suffix=cap_L $extra
    run_one "$target" mode="$bench_mode" sweep_axis=d_pair name_suffix=cap_d $extra
  done
}

SHAPES='mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
ATOM_SHAPES='mask_prob=0.0 min_seq_len=128 max_seq_len=384 seq_len_step=128 d_pair_values=[16,32,64] sweep_seq_len=384'

capture_target() {
  case "$1" in
    transition)                 run_matrix transition "$SHAPES" ;;
    triangle_attention)         run_matrix triangle_attention "$SHAPES" ;;
    bias_only_attention)        run_matrix bias_only_attention "$SHAPES" ;;
    triangle_multiplication)    run_matrix triangle_multiplication "$SHAPES" ;;
    conditioned_transition)     run_matrix conditioned_transition "precision=32 $SHAPES d_single_token=384" ;;
    adaptive_layernorm)         run_matrix adaptive_layernorm "$SHAPES" ;;
    augmented_attention_token)  run_matrix augmented_attention_token "$SHAPES" ;;
    augmented_attention_atom)   run_matrix augmented_attention_atom "$ATOM_SHAPES" ;;
    all)
      for t in transition triangle_attention bias_only_attention triangle_multiplication \
               conditioned_transition adaptive_layernorm augmented_attention_token augmented_attention_atom; do
        capture_target "$t"
      done ;;
    *) echo "unknown CAPTURE_TARGET=$1" >&2; exit 2 ;;
  esac
}

capture_target "${CAPTURE_TARGET:-transition}"

RT="${MINIWORLD_KERNELS_CACHE_DIR:-$HOME/.cache/miniworld_kernels}/autotune"
DST="src/miniworld_kernels/autotune/data"
echo ""
echo "### runtime autotune caches produced ###"
find "$RT" -name '*.json' -printf '%p\n' 2>/dev/null | sort

# Cross-check the capture builder against the hand-built PILOT shipped cache: for any op that
# both produced (e.g. transition_split_fwd), compare capture(runtime) vs hand(shipped) top-1.
echo ""
echo "### cross-check capture(runtime) vs hand-built(shipped) ###"
pixi run --frozen bash -c 'PYTHONPATH=src RT="'"$RT"'" DST="'"$DST"'" python submits/_capture_crosscheck.py' || echo "(cross-check skipped)"

# Copy runtime cache -> committed package data (gated: CAPTURE_COPY=0 for a validation-only run).
if [ "${CAPTURE_COPY:-1}" = "1" ]; then
  echo "### copy $RT -> $DST ###"
  [ -d "$RT" ] && cp -rv "$RT/." "$DST/" 2>/dev/null | tail -60
else
  echo "### CAPTURE_COPY=0 -> NOT copying to shipped (validation run) ###"
fi
echo "AUTOTUNE CAPTURE DONE target=${CAPTURE_TARGET:-transition}"
