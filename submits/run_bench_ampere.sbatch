#!/bin/bash
# Ampere workstation GPU (RTX A5000 / A6000, GA102 = sm_86) module benchmark launcher
# for the CSSB cluster (login=cssb-master, partition=gpu). These are TRITON-ONLY, exactly
# like the A100 (sm_80): the cute/cutlass paths are gated behind sm_90+ (Hopper/Blackwell)
# via torch.cuda.get_device_capability()[0] < 9, so on sm_86 MINIWORLD resolves to the
# portable Triton family — same impl matrices as run_bench_a100.sbatch.
#
# One script for BOTH cards: they share partition=gpu and differ only by the gres GPU
# type, so pick the card with --gres on the command line and the script auto-detects which
# one it landed on (A5000 vs A6000) for the results slug + validates it is really sm_86.
#
#   BENCH_TARGET=transition sbatch --gres=gpu:A6000:1 submits/run_bench_ampere.sbatch
#   BENCH_TARGET=transition sbatch --gres=gpu:A5000:1 submits/run_bench_ampere.sbatch
#   BENCH_TARGET=all        sbatch --gres=gpu:A6000:1 submits/run_bench_ampere.sbatch
#
# A5000 has only 24GB VRAM (A6000 48GB, A100 80GB): the top of the sweep (L=1024, d=512)
# may OOM on A5000. That is expected and handled — bench.py records the failed points as
# status=failed rows rather than aborting the sweep, so the CSV still shows where the card
# runs out of memory.
#SBATCH --job-name=mwk_modules_ampere
#SBATCH --partition=gpu
#SBATCH --account=cssb
#SBATCH --qos=normal
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=benchmarks/modules/%x_%j.out

set -euo pipefail
set -f
cd /home/psk6950/practice/miniworld-kernels
export HYDRA_FULL_ERROR=1

echo "host=$(hostname)  date=$(date)"
# Sanity: confirm we actually landed on an sm_86 Ampere workstation card (A5000/A6000),
# NOT an A100 (also cap[0]==8 but sm_80) — the cute gating is by capability, but the
# results slug + tuned-cache key are by name, so guard the exact (8,6) + RTX Axxxx name.
pixi run --frozen python submits/_assert_device.py --capability 8.6 --name A5000 --name A6000 || exit 3

# results/<slug>/ slug from the allocated card (a5000 / a6000), mirroring results/a100 etc.
# Derive from nvidia-smi (no python-in-pixi quoting pitfalls) and HARD-FAIL on an unknown
# card so a bad slug can never collapse the path to results//{tables,plots} = results/{...}.
DEV_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
case "$DEV_NAME" in
  *A5000*) SLUG=a5000 ;;
  *A6000*) SLUG=a6000 ;;
  *) echo "FATAL: cannot derive results slug from device '$DEV_NAME'"; exit 3 ;;
esac
echo "results slug=${SLUG}  (device='${DEV_NAME}')"

module_dir_for() {
  case "$1" in
    augmented_attention_token|augmented_attention_atom) echo "augmented_attention" ;;
    *) echo "$1" ;;
  esac
}

# a100/... store augmented_attention as token_*/atom_* (two targets share one module dir);
# every other module uses a bare <mode>_<axis> stem. Match that so results/<slug> lines up.
stem_prefix_for() {
  case "$1" in
    augmented_attention_token) echo "token_" ;;
    augmented_attention_atom)  echo "atom_" ;;
    *) echo "" ;;
  esac
}

layer_count_for() {
  case "$1" in
    mpnn) echo "3" ;;
    *) echo "1" ;;
  esac
}

run_one() {
  target="$1"; shift
  impls="$1"; shift
  echo ""
  echo "############################################################"
  echo "# benchmarks/runners/bench.py  kernel=${target}  $*"
  echo "############################################################"
  LAST_RUN_STARTED_AT="$(date +%s.%N)"
  if ! pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/bench.py "$@"' _ \
      kernel="$target" \
      implementations="$impls" \
      metric=time \
      compile=true \
      "$@"; then
    echo "benchmark failed: target=${target}" >&2
    return 1
  fi
  echo "[exit=0]"
}

render_latest() {
  target="$1"
  mode="$2"
  graph_mode="$3"
  sweep_axis="$4"
  suffix="$5"
  result_prefix="${6:-}"
  module_dir="$(module_dir_for "$target")"
  n_layers="$(layer_count_for "$target")"
  artifact_root="benchmarks/modules/${module_dir}/artifacts"
  device_artifact_root="${artifact_root}/${DEV_NAME}"
  if [[ "$graph_mode" == "disabled" ]]; then
    run_tail="compile_${sweep_axis}_${suffix}"
  else
    run_tail="compile_cudagraph-${graph_mode}_${sweep_axis}_${suffix}"
  fi
  csv_path="$(find "$device_artifact_root" -type f -newermt "@${LAST_RUN_STARTED_AT}" -name "${target}_n_layers=${n_layers}_${mode}_time_*_${run_tail}.csv" -printf '%T@ %p\n' | sort -nr | sed -n '1s/^[^ ]* //p')"
  if [[ -z "${csv_path}" ]]; then
    echo "No CSV found for target=${target}, mode=${mode}, suffix=${suffix}; skipping render"
    return 1
  fi
  # Curate straight into results/<slug>/{tables,plots}/ with the canonical <stem> names
  # (this is what results/a100 etc. hold) so the Ampere results land in the committed layout
  # directly, not just in artifacts/<full-device-name>/.
  stem="$(stem_prefix_for "$target")${result_prefix}${mode}_${sweep_axis}"
  results_dir="benchmarks/modules/${module_dir}/results/${SLUG}"
  mkdir -p "${results_dir}/tables" "${results_dir}/plots"
  lock_path="/tmp/miniworld-kernels-${USER}-${SLUG}-${module_dir}-${stem}.lock"
  exec {lock_fd}>"${lock_path}"
  flock "${lock_fd}"
  canonical_csv="${results_dir}/tables/${stem}.csv"
  temporary_csv="${results_dir}/tables/.${stem}.${SLURM_JOB_ID:-$$}.tmp"
  cp "$csv_path" "$temporary_csv"
  mv -f "$temporary_csv" "$canonical_csv"
  echo ""
  echo "############################################################"
  echo "# curate -> ${results_dir}/{tables,plots}/${stem}  (csv=$csv_path)"
  echo "############################################################"
  pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/plot_csv.py "$@"' _ \
    "$csv_path" "${results_dir}/plots" --name "$stem"
  echo "[exit=$?]"
  flock -u "${lock_fd}"
}

run_matrix() {
  target="$1"
  impls="$2"
  extra="$3"
  result_prefix="${4:-}"
  run_id="${SLURM_JOB_ID:-manual_$$}"
  for graph_mode in manual; do   # manual = deployment regime (default); add 'disabled' for the eager baseline
    for bench_mode in inference training; do
      run_one "$target" "$impls" mode="$bench_mode" cudagraph="$graph_mode" sweep_axis=seq_len name_suffix="L_sweep_${run_id}" $extra
      render_latest "$target" "$bench_mode" "$graph_mode" seq_len "L_sweep_${run_id}" "$result_prefix"
      run_one "$target" "$impls" mode="$bench_mode" cudagraph="$graph_mode" sweep_axis=d_pair name_suffix="d_sweep_${run_id}" $extra
      render_latest "$target" "$bench_mode" "$graph_mode" d_pair "d_sweep_${run_id}" "$result_prefix"
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
      # sm_86: miniworld trimul resolves to TRITON (cute is gated behind sm90plus).
      run_matrix triangle_multiplication '[pytorch,cuequivariance,miniworld]' 'mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
      ;;
    triangle_multiplication_bidirectional)
      # sm_86: miniworld bidir resolves to TRITON -> main's composed-from-unidirectional
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
    mpnn)
      # One source-faithful FP32 packed parity point. Large timing runs disable
      # the in-run oracle to avoid retaining two complete autograd graphs.
      smoke_id="${SLURM_JOB_ID:-manual_$$}"
      run_one mpnn '[miniworld]' \
        mode=training cudagraph=disabled precision=32 n_layers=3 n_augment=1 \
        batch_size=8 mpnn_layout=packed mpnn_patch_size=8 \
        check_correctness=true seq_len_values=[256] d_pair=128 \
        name_suffix="packed_correctness_${smoke_id}"

      # Per-sample crop ceiling. seq_len is the physical tensor length here.
      run_matrix mpnn '[pytorch,miniworld]' \
        'precision=bf16-mixed n_layers=3 n_augment=1 mask_prob=0.0 batch_size=1 mpnn_layout=single mpnn_patch_size=8 check_correctness=false seq_len_values=[256,512,1024,1536,2048] d_pair_values=[64,128,256] sweep_seq_len=512' \
        'single_'

      # Source training layout: physical B=1, eight equal-length logical
      # samples concatenated to total length 8*seq_len.
      run_matrix mpnn '[pytorch,miniworld]' \
        'precision=32 n_layers=3 n_augment=1 mask_prob=0.0 batch_size=8 mpnn_layout=packed mpnn_patch_size=8 check_correctness=false seq_len_values=[256,384,512] d_pair_values=[64,128,256] sweep_seq_len=384' \
        'packed_b8_'
      ;;
    all)
      BENCH_TARGET=bias_only_attention run_selected
      BENCH_TARGET=transition run_selected
      BENCH_TARGET=conditioned_transition run_selected
      BENCH_TARGET=adaptive_layernorm run_selected
      BENCH_TARGET=augmented_attention_token run_selected
      BENCH_TARGET=augmented_attention_atom run_selected
      BENCH_TARGET=mpnn run_selected
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
