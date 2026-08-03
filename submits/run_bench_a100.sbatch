#!/bin/bash
# A100 (sm_80) module benchmark launcher. Mirrors run_bench.sbatch but targets the
# A100 partition and, crucially, validates the pre-Hopper dispatch: on sm_80 the
# transition module must route MINIWORLD to the portable Triton family (NOT the
# Hopper-only hand-CUDA b2b / quack-cute paths, which only build/launch on sm_90).
#
# SLURM params below match the cssb A100 cluster (partition=A100, qos=normal,
# gres=gpu:A100:1, default account). Submit e.g.:
#   BENCH_TARGET=transition sbatch submits/run_bench_a100.sbatch
# Override the target on the command line: BENCH_TARGET=all sbatch ...
#SBATCH --job-name=mwk_modules_a100
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=benchmarks/modules/%x_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-engine
export HYDRA_FULL_ERROR=1

echo "host=$(hostname)  date=$(date)"
# Sanity: confirm we actually landed on an A100 (sm_80) before spending the budget.
pixi run --frozen python -c "import torch; cap=torch.cuda.get_device_capability(0); print('torch', torch.__version__, torch.cuda.get_device_name(0), 'sm', cap); assert cap[0]==8, f'expected sm_80 (A100), got sm_{cap[0]}{cap[1]}'"

# --- CPU-side correctness gate for the A100 dispatch fix (arch mocked, no kernels) ---
echo ""
echo "############################################################"
echo "# pytest: dispatch arch-policy (is_sm90 Hopper-exactly) "
echo "############################################################"
pixi run --frozen bash -c 'PYTHONPATH=src python -m pytest tests/test_dispatch.py tests/test_public_api.py -q'
echo "[dispatch tests exit=$?]"

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
  # Figures live under <bench>/plots/<gpu>/ (NOT next to the CSVs in artifacts/) — mirror
  # plot_csv.py's own default so the A100 launcher matches the convention.
  output_dir="$(dirname "$csv_path" | sed 's#/artifacts/#/plots/#')"
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
  for graph_mode in manual; do   # manual = deployment regime (default); add 'disabled' for the eager baseline
    for bench_mode in inference training; do
      run_one "$target" "$impls" mode="$bench_mode" cudagraph="$graph_mode" sweep_axis=seq_len name_suffix=L_sweep $extra
      render_latest "$target" "$bench_mode" "$graph_mode" seq_len L_sweep
      run_one "$target" "$impls" mode="$bench_mode" cudagraph="$graph_mode" sweep_axis=d_pair name_suffix=d_sweep $extra
      render_latest "$target" "$bench_mode" "$graph_mode" d_pair d_sweep
    done
  done
}

run_selected() {
  case "${BENCH_TARGET:-transition}" in
    bias_only_attention)
      run_matrix bias_only_attention '[pytorch,cuequivariance,old_triton,miniworld]' 'mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
      ;;
    triangle_attention)
      run_matrix triangle_attention '[pytorch,cuequivariance,miniworld]' 'mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
      ;;
    transition)
      run_matrix transition '[pytorch,old_triton,miniworld]' 'mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
      ;;
    triangle_multiplication)
      # A100: miniworld trimul resolves to TRITON (cute is gated behind sm90plus).
      run_matrix triangle_multiplication '[pytorch,cuequivariance,miniworld]' 'mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
      ;;
    triangle_multiplication_bidirectional)
      # A100: miniworld bidir resolves to TRITON -> main's composed-from-unidirectional
      # triton bidir path (triton_tm1 front + triton GateElem back). cute gated behind sm90plus.
      run_matrix triangle_multiplication_bidirectional '[pytorch,cuequivariance,miniworld]' 'mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
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
echo "MODULE BENCH DONE target=${BENCH_TARGET:-transition}"
