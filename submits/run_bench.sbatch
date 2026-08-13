#!/bin/bash
#SBATCH --job-name=mwk_modules
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/modules/transition/artifacts/modules_%x_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-engine
export HYDRA_FULL_ERROR=1 echo "host=$(hostname)  date=$(date)"
pixi run --frozen python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

module_dir_for() {
  case "$1" in
    augmented_attention_token|augmented_attention_atom) echo "augmented_attention" ;;
    *) echo "$1" ;;
  esac
}

run_one() {
  target="$1"; shift
  impls="$1"; shift
  echo ""
  echo "############################################################"
  echo "# benchmarks/runners/bench.py  kernel=${target}  $*"
  echo "############################################################"
  pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/bench.py "$@"' _ \
    kernel="$target" \
    implementations="$impls" \
    metric=time \
    compile=true \
    "$@"
  echo "[exit=$?]"
}

render_latest() {
  target="$1"
  mode="$2"
  graph_mode="$3"
  sweep_axis="$4"
  suffix="$5"
  module_dir="$(module_dir_for "$target")"
  artifact_root="benchmarks/modules/${module_dir}/artifacts"
  if [[ "$graph_mode" == "disabled" ]]; then
    run_tail="compile_${sweep_axis}_${suffix}"
  else
    run_tail="compile_cudagraph-${graph_mode}_${sweep_axis}_${suffix}"
  fi
  csv_path="$(find "$artifact_root" -type f -name "${target}_n_layers=1_${mode}_time_*_${run_tail}.csv" -printf '%T@ %p\n' | sort -nr | sed -n '1s/^[^ ]* //p')"
  if [[ -z "${csv_path}" ]]; then
    echo "No CSV found for target=${target}, mode=${mode}, suffix=${suffix}; skipping render"
    return 1
  fi
  output_dir="$(dirname "$csv_path")"
  echo ""
  echo "############################################################"
  echo "# benchmarks/runners/plot_csv.py  csv=$csv_path"
  echo "############################################################"
  pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/plot_csv.py "$@"' _ "$csv_path" "$output_dir"
  echo "[exit=$?]"
}

run_matrix() {
  target="$1"
  impls="$2"
  extra="$3"
  for graph_mode in manual disabled; do
    for bench_mode in inference training; do
      run_one "$target" "$impls" mode="$bench_mode" cudagraph="$graph_mode" sweep_axis=seq_len name_suffix=L_sweep $extra
      render_latest "$target" "$bench_mode" "$graph_mode" seq_len L_sweep
      run_one "$target" "$impls" mode="$bench_mode" cudagraph="$graph_mode" sweep_axis=d_pair name_suffix=d_sweep $extra
      render_latest "$target" "$bench_mode" "$graph_mode" d_pair d_sweep
    done
  done
}

run_selected() {
  case "${BENCH_TARGET:-all}" in
    bias_only_attention)
      # Full triangle self-attention is not a MiniWorld-developed kernel in this repo; the
      # developed path here is the bias-only TriangleAttention case (use_self_attention=False).
      run_matrix bias_only_attention '[pytorch,cuequivariance,old_triton,miniworld]' 'mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
      ;;
    triangle_attention)
      run_matrix triangle_attention '[pytorch,cuequivariance,miniworld]' 'mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
      ;;
    transition)
      run_matrix transition '[pytorch,old_triton,miniworld]' 'mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
      ;;
    conditioned_transition)
      run_matrix conditioned_transition '[pytorch,miniworld]' 'precision=32 mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384 d_single_token=384'
      ;;
    adaptive_layernorm)
      run_matrix adaptive_layernorm '[pytorch,miniworld]' 'mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
      ;;
    augmented_attention_token)
      run_matrix augmented_attention_token '[pytorch,miniworld]' 'mask_prob=0.0 min_seq_len=128 max_seq_len=384 seq_len_step=64 d_pair_values=[128,256,512] sweep_seq_len=384'
      ;;
    augmented_attention_atom)
      run_matrix augmented_attention_atom '[pytorch,miniworld]' 'mask_prob=0.0 min_seq_len=128 max_seq_len=384 seq_len_step=128 d_pair_values=[16,32,64] sweep_seq_len=384'
      ;;
    all)
      BENCH_TARGET=bias_only_attention run_selected
      BENCH_TARGET=transition run_selected
      BENCH_TARGET=conditioned_transition run_selected
      BENCH_TARGET=adaptive_layernorm run_selected
      BENCH_TARGET=augmented_attention_token run_selected
      BENCH_TARGET=augmented_attention_atom run_selected
      ;;
    *)
      echo "Unknown BENCH_TARGET=${BENCH_TARGET}" >&2
      exit 2
      ;;
  esac
}

run_selected

echo ""
echo "MODULE BENCH DONE target=${BENCH_TARGET:-all}"
