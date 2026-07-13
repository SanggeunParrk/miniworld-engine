#!/bin/bash
# B200 (sm_100) counterpart of run_autotune_capture_a100.sbatch. The B200 dev box is a
# direct-SSH box (no Slurm), so this is a plain script, not an sbatch. It INSTRUMENTS the
# Triton autotuner during real module runs (capture builder) to ship per-GPU tuned autotune
# caches for the live Triton kernels on B200, exactly as the A100 job does for sm_80.
#
#   CAPTURE_TARGET=transition bash submits/run_autotune_capture_b200.sh   # validate first
#   CAPTURE_TARGET=all        bash submits/run_autotune_capture_b200.sh
#
# Env knobs (same semantics as the A100 job):
#   CAP_COMPILE (default false), CAP_GRAPH (default manual), CAPTURE_COPY (default 1),
#   MWK (miniworld-kernels checkout), PY (python for the quack-0.5.0 consumer env).
set -uo pipefail
set -f
MWK="${MWK:-$HOME/psk/miniworld-kernels}"
PY="${PY:-$HOME/psk/MiniWorld/.pixi/envs/default/bin/python}"   # quack 0.5.0 / cutlass 4.5.2 / FA4
cd "$MWK"
export HYDRA_FULL_ERROR=1
# Repo-local, exec-capable caches (box /tmp is noexec; keep scratch repo-local under ncu/).
export MINIWORLD_KERNELS_CACHE_DIR="${MINIWORLD_KERNELS_CACHE_DIR:-$MWK/ncu/autotune_run}"
export TRITON_CACHE_DIR="$MWK/ncu/p5cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$MWK/ncu/p5cache/inductor"
export CUTE_DSL_CACHE_DIR="$MWK/ncu/p5cache/cute"
echo "host=$(hostname) date=$(date) target=${CAPTURE_TARGET:-transition} cache=$MINIWORLD_KERNELS_CACHE_DIR"
$PY -c "import torch; cap=torch.cuda.get_device_capability(0); assert cap[0]==10, cap; print('B200 ok', torch.__version__)" || exit 1

run_one() {
  target="$1"; shift
  echo ""; echo "### capture bench kernel=${target} $* ###"
  env MINIWORLD_RUN_AUTOTUNE=1 MINIWORLD_AUTOTUNE_CAPTURE=1 PYTHONPATH=src \
    "$PY" benchmarks/runners/bench.py \
    kernel="$target" implementations='[miniworld]' metric=time \
    compile="${CAP_COMPILE:-false}" cudagraph="${CAP_GRAPH:-manual}" "$@"
  echo "[exit=$?]"
}
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
    transition)              run_matrix transition "$SHAPES" ;;
    triangle_attention)      run_matrix triangle_attention "$SHAPES" ;;
    bias_only_attention)     run_matrix bias_only_attention "$SHAPES" ;;
    triangle_multiplication) run_matrix triangle_multiplication "$SHAPES" ;;
    triangle_multiplication_bidirectional) run_matrix triangle_multiplication_bidirectional "$SHAPES" ;;
    conditioned_transition)  run_matrix conditioned_transition "precision=32 $SHAPES d_single_token=384" ;;
    adaptive_layernorm)      run_matrix adaptive_layernorm "$SHAPES" ;;
    augmented_attention_token) run_matrix augmented_attention_token "$SHAPES" ;;
    augmented_attention_atom)  run_matrix augmented_attention_atom "$ATOM_SHAPES" ;;
    all)
      for t in transition triangle_attention bias_only_attention triangle_multiplication \
               triangle_multiplication_bidirectional conditioned_transition adaptive_layernorm \
               augmented_attention_token augmented_attention_atom; do
        capture_target "$t"
      done ;;
    *) echo "unknown CAPTURE_TARGET=$1" >&2; exit 2 ;;
  esac
}
capture_target "${CAPTURE_TARGET:-transition}"

RT="$MINIWORLD_KERNELS_CACHE_DIR/autotune"
DST="src/miniworld_kernels/autotune/data"
echo ""; echo "### runtime autotune caches produced ###"
find "$RT" -name '*.json' -printf '%p\n' 2>/dev/null | sort
if [ "${CAPTURE_COPY:-1}" = "1" ]; then
  echo "### copy $RT -> $DST ###"
  [ -d "$RT" ] && cp -rv "$RT/." "$DST/" 2>/dev/null | tail -80
else
  echo "### CAPTURE_COPY=0 -> NOT copying to shipped (validation run) ###"
fi
echo "AUTOTUNE CAPTURE DONE target=${CAPTURE_TARGET:-transition}"
