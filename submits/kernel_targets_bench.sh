#!/bin/bash
# Kernel FUNCTION-operation benchmarks: each target = one kernel operation (folder name =
# operation, not module). Each target compares ALL its implementations (incl deprecated) as rows
# vs a pytorch reference, swept over seq_len and d_pair, cudagraph=manual. Fanned out across the
# node's GPUs (per-GPU TRITON_CACHE_DIR to avoid autotune-cache races that broke cudagraph capture).
# Run inside an srun holding the GPUs.
set -uo pipefail
set -f
# Repo-root relative to THIS script (submits/..), so it runs in whichever checkout it
# lives in — no hardcoded absolute path that breaks when a checkout moves or is removed.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HYDRA_FULL_ERROR=1 # target -> full variant list (rows compared inside its folder)
declare -A IMPLS=(
  [dual_gemm_epil]='[pytorch,trimul_front_triton,trimul_inproj_cute,tm1_cute,triton_tm1,trimul_front_sm100]'
  [gemm_epil]='[pytorch,layernorm_linear_triton,layernorm_linear_cute,layernorm_linear_cute_fused,layernorm_linear_te]'
  [transition_b2b]='[pytorch,triton_transition_fused,cute_transition_fused,transition_b2b_ktiled]'
  [layernorm]='[pytorch,triton_layernorm,layernorm_dispatch,quack_cute,triton_layernorm_lowreg]'
  [adaln]='[pytorch,adaln_inference,adaln_fused3,triton_adaln]'
  [tri_attn]='[pytorch,triton_tri_attn,triton_tri_attn_miniworld,triton_tri_attn_perf]'
  [bias_attn]='[pytorch,bias_only_fused,triton_bias_attn]'
  [aug_attn]='[pytorch,triton_aug_attn,aug_attn_compute_efficient]'
  [ln_mask]='[pytorch,fused_ln_mask]'
  [gemm_gate]='[pytorch,tm2_cute,triton_tm2]'
  [cond_transition_tail]='[pytorch,triton_cond_transition]'
  [layernorm_bwd]='[pytorch,triton_persistent,triton_atomic,triton_partial,cuda]'
  [gate_bwd]='[pytorch,gate_elem_bwd]'
  [dual_gemm_epil_bwd]='[pytorch,front_bwd_fused]'
  [adaln_bwd]='[pytorch,adaln_train,adaln_fused3,triton_adaln]'
  [transition_b2b_bwd]='[pytorch,triton_transition_fused,cute_transition_fused]'
  [gemm_epil_bwd]='[pytorch,layernorm_linear_cute,layernorm_linear_te]'
)
declare -A EXTRA=( [cond_transition_tail]='precision=32' )
TARGETS=(${TARGETS_OVERRIDE:-dual_gemm_epil gemm_epil transition_b2b layernorm adaln tri_attn bias_attn aug_attn ln_mask gemm_gate cond_transition_tail layernorm_bwd gate_bwd dual_gemm_epil_bwd adaln_bwd transition_b2b_bwd gemm_epil_bwd})

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NGPU:-0}" -lt 1 ] && NGPU=1
echo "GPUs=$NGPU targets=${#TARGETS[@]}"

run_one() {  # $1=gpu $2=target $3=sweep
  local extra="${EXTRA[$2]:-}"; local sfx2; [ "$3" = seq_len ] && sfx2=L_sweep || sfx2=d_sweep
  CUDA_VISIBLE_DEVICES="$1" TRITON_CACHE_DIR="/tmp/triton_gpu$1" pixi run --frozen bash -c \
    'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/bench.py "$@"' _ \
    kernel="$2" implementations="${IMPLS[$2]}" metric=time compile=true cudagraph=manual \
    mode=inference sweep_axis="$3" name_suffix="$sfx2" $extra \
    > "/tmp/ktb_${2}_${3}.log" 2>&1
  echo "[done gpu$1] $2 $3 (exit $?)"
}

TASKS=()
for t in "${TARGETS[@]}"; do for s in seq_len d_pair; do TASKS+=("$t|$s"); done; done
i=0
for t in "${TASKS[@]}"; do
  IFS='|' read -r k s <<< "$t"
  run_one "$((i % NGPU))" "$k" "$s" &
  i=$((i+1)); (( i % NGPU == 0 )) && wait
done
wait
echo "=== bench done; rendering + curating ==="
DEV="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# results/<slug>/tables slug from the card (matches results/a100 kernel layout: tables-only,
# inference seq_len + d_pair CSVs, no plots). Unknown card -> skip curation (render only).
case "$DEV" in
  *A100*) SLUG=a100 ;; *A5000*) SLUG=a5000 ;; *A6000*) SLUG=a6000 ;;
  *H100*) SLUG=h100 ;; *B200*) SLUG=b200 ;; *) SLUG="" ;;
esac
for k in "${TARGETS[@]}"; do
  A="benchmarks/kernels/${k}/artifacts/$DEV"
  find "$A" -maxdepth 1 -name "${k}_*_cudagraph-manual_*.csv" -print0 2>/dev/null | while IFS= read -r -d '' csv; do
    pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/plot_csv.py "$@"' _ "$csv" "$A" >/dev/null 2>&1
  done
  # Curate the canonical inference CSVs into results/<slug>/tables/ (source of truth = CSV).
  if [ -n "$SLUG" ]; then
    RT="benchmarks/kernels/${k}/results/${SLUG}/tables"; mkdir -p "$RT"
    for axis in seq_len d_pair; do
      sfx=L_sweep; [ "$axis" = d_pair ] && sfx=d_sweep
      csv="$(find "$A" -maxdepth 1 -name "${k}_*_inference_time_*_cudagraph-manual_${axis}_${sfx}.csv" -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1s/^[^ ]* //p')"
      [ -n "$csv" ] && cp -f "$csv" "$RT/inference_${axis}.csv"
    done
  fi
done
echo "ALL DONE"
