#!/bin/bash
set -euo pipefail

cd /home/psk6950/practice/miniworld-kernels
export PYTHONPATH=src
export HYDRA_FULL_ERROR=1

python_bin=.pixi/envs/default/bin/python
common=(
  kernel=mpnn
  n_layers=3
  n_augment=1
  mask_prob=0.0
  k_neighbors=48
  batch_size=1
  mpnn_layout=single
  mpnn_patch_size=8
  mpnn_coordinate_grad=false
  d_pair=128
  sweep_axis=seq_len
  'seq_len_values=[2048]'
  check_correctness=false
)

run_case() {
  local label="$1"
  shift
  local implementation
  # Run each implementation in a fresh process so CUDA Graph pools, allocator
  # reservations, and Inductor state cannot contaminate the next result.
  for implementation in miniworld pytorch; do
    echo "RUN_CASE ${label} implementation=${implementation}"
    "$python_bin" benchmarks/runners/bench.py \
      "${common[@]}" \
      implementations="[${implementation}]" \
      "$@" \
      name_suffix="${label}_${implementation}"
  done
}

run_case l2048_bf16_inference_time \
  mode=inference metric=time precision=bf16-mixed allow_tf32=true \
  compile=true cudagraph=manual
run_case l2048_bf16_training_time \
  mode=training metric=time precision=bf16-mixed allow_tf32=true \
  compile=true cudagraph=manual
run_case l2048_bf16_inference_memory \
  mode=inference metric=memory precision=bf16-mixed allow_tf32=true \
  compile=true cudagraph=disabled
run_case l2048_bf16_training_memory \
  mode=training metric=memory precision=bf16-mixed allow_tf32=true \
  compile=true cudagraph=disabled

# AMP-off parameter-gradient training used by the source script. Compiler and
# graph transforms remain identical so the two implementations are comparable.
run_case l2048_fp32_training_time \
  mode=training metric=time precision=32 allow_tf32=false \
  compile=true cudagraph=manual
run_case l2048_fp32_training_memory \
  mode=training metric=memory precision=32 allow_tf32=false \
  compile=true cudagraph=disabled

echo "ALL_CASES_DONE"
