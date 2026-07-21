#!/bin/bash
# Build the shipped Triton autotune caches for the Ampere workstation cards (RTX A5000 /
# A6000, sm_86) by INSTRUMENTING the Triton autotuner during real module runs (capture
# builder), instead of hand-replicating each kernel launch. For each target module we run
# the bench harness (miniworld backend, inference+training, seq_len + d_pair sweeps) with:
#   MINIWORLD_RUN_AUTOTUNE=1      -> full grid is benched (no cached top-K narrowing)
#   MINIWORLD_AUTOTUNE_CAPTURE=1  -> bench.py installs capture, records per-config timings,
#                                    flushes top-5 per (op,dtype,bucket) to the cache.
# The capture builder writes DIRECTLY into the committed in-repo cache
# (src/miniworld_kernels/autotune/data/<op>/<gpu_key>.json) — there is no runtime-dir copy
# step (cache.py deliberately dropped the $MINIWORLD_KERNELS_CACHE_DIR override). After the
# job, the new "NVIDIA RTX A5000 (sm86).json" / "NVIDIA RTX A6000 (sm86).json" entries are
# ready to `git add` + commit. cute/cuda kernels are sm_90+ (not built/tuned here).
#
#   CAPTURE_TARGET=transition sbatch --gres=gpu:A6000:1 submits/run_autotune_capture_ampere.sbatch  # validate first
#   CAPTURE_TARGET=all        sbatch --gres=gpu:A6000:1 submits/run_autotune_capture_ampere.sbatch
#   CAPTURE_TARGET=all        sbatch --gres=gpu:A5000:1 submits/run_autotune_capture_ampere.sbatch
#SBATCH --job-name=mwk_autotune_capture_ampere
#SBATCH --partition=gpu
#SBATCH --account=cssb
#SBATCH --qos=normal
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=benchmarks/autotune_capture_ampere_%x_%j.out

set -uo pipefail
set -f
cd /home/psk6950/practice/miniworld-kernels
export HYDRA_FULL_ERROR=1
echo "host=$(hostname) date=$(date) target=${CAPTURE_TARGET:-transition}"
pixi run --frozen python -c "
import torch
name = torch.cuda.get_device_name(0); cap = torch.cuda.get_device_capability(0)
print('capture on', name, 'sm', cap, torch.__version__)
assert cap == (8, 6), f'expected sm_86 (RTX A5000/A6000), got sm_{cap[0]}{cap[1]} ({name})'
assert ('A5000' in name) or ('A6000' in name), f'unexpected Ampere card: {name}'
"

run_one() {
  target="$1"; shift
  echo ""
  echo "### capture bench kernel=${target} $* ###"
  pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-} MINIWORLD_RUN_AUTOTUNE=1 MINIWORLD_AUTOTUNE_CAPTURE=1; PYTHONPATH=src python benchmarks/runners/bench.py "$@"' _ \
    kernel="$target" implementations='[miniworld]' metric=time \
    compile="${CAP_COMPILE:-false}" cudagraph="${CAP_GRAPH:-manual}" "$@"
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
    triangle_multiplication_bidirectional) run_matrix triangle_multiplication_bidirectional "$SHAPES" ;;
    conditioned_transition)     run_matrix conditioned_transition "precision=32 $SHAPES d_single_token=384" ;;
    adaptive_layernorm)         run_matrix adaptive_layernorm "$SHAPES" ;;
    augmented_attention_token)  run_matrix augmented_attention_token "$SHAPES" ;;
    augmented_attention_atom)   run_matrix augmented_attention_atom "$ATOM_SHAPES" ;;
    augmented)  # token + atom share augmented_attention_* op files -> run sequentially in ONE job
      run_matrix augmented_attention_token "$SHAPES"
      run_matrix augmented_attention_atom "$ATOM_SHAPES" ;;
    all)
      for t in transition triangle_attention bias_only_attention triangle_multiplication \
               conditioned_transition adaptive_layernorm augmented_attention_token augmented_attention_atom; do
        capture_target "$t"
      done ;;
    *) echo "unknown CAPTURE_TARGET=$1" >&2; exit 2 ;;
  esac
}

capture_target "${CAPTURE_TARGET:-transition}"

DST="src/miniworld_kernels/autotune/data"
echo ""
echo "### autotune cache JSONs now present for this card (grep this GPU key) ###"
DEV_KEY="$(pixi run --frozen python -c "from miniworld_kernels.autotune.cache import gpu_key; print(gpu_key())" 2>/dev/null)"
echo "gpu_key=${DEV_KEY}"
find "$DST" -name "${DEV_KEY}.json" -printf '%p\n' 2>/dev/null | sort
echo "AUTOTUNE CAPTURE DONE target=${CAPTURE_TARGET:-transition}  (git add ${DST} to commit)"
