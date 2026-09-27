#!/bin/bash
# H100 (sm_90) MODULE benchmark sweep. gpu-8farm is whole-node (min 8 GPU / job), so we
# hold node22's 8 H100s in ONE allocation and FAN the (target × mode × axis) tasks across
# the 8 GPUs (per-GPU TRITON_CACHE_DIR to avoid triton-compile-cache races during cudagraph
# capture; the miniworld sm90 autotune caches are pre-shipped -> read-only hits, no races).
# Then render one plot per CSV. Unlike A100, this DOES exercise the sm90 cute paths
# (trimul BidirV6/V6Merged, transition b2b, triangle-attn) and includes the triangle targets.
#SBATCH --job-name=mwk_modules_h100
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=benchmarks/modules/%x_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels
export HYDRA_FULL_ERROR=1

echo "host=$(hostname)  date=$(date)"
# FAIL-FAST CUDA/sm90 gate with retry: some HGX nodes throw CUDA error 802 ("system not yet
# initialized", NVSwitch fabric race) right after allocation. Retry a few times; if still bad,
# ABORT immediately (exit) instead of running dozens of doomed tasks on a broken node.
ok=0
for try in 1 2 3 4 5; do
  if pixi run --frozen python submits/_assert_device.py --major 9; then
    ok=1; break
  fi
  echo "[cuda check try $try failed — node may not be fabric-ready; retrying in 20s]"
  sleep 20
done
[ "$ok" = 1 ] || { echo "FATAL: CUDA not usable on $(hostname) after retries — aborting (resubmit lands elsewhere)"; exit 1; }

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NGPU:-0}" -lt 1 ] && NGPU=1
echo "NGPU=$NGPU"
# per-task logs on shared FS (visible from login node), NOT node-local /tmp
LOGDIR="benchmarks/modules/logs/h100_${SLURM_JOB_ID:-manual}"; mkdir -p "$LOGDIR"

module_dir_for() {
  case "$1" in
    augmented_attention_token|augmented_attention_atom) echo "augmented_attention" ;;
    *) echo "$1" ;;
  esac
}

# target -> "impls|extra params"   (impls + sweep params copied verbatim from run_bench_a100.sbatch)
declare -A SPEC=(
  [bias_only_attention]='[pytorch,cuequivariance,old_triton,miniworld]|mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
  [triangle_attention]='[pytorch,cuequivariance,miniworld]|mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
  [transition]='[pytorch,old_triton,miniworld]|mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
  [triangle_multiplication]='[pytorch,cuequivariance,miniworld]|mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
  [triangle_multiplication_bidirectional]='[pytorch,cuequivariance,miniworld]|mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
  [conditioned_transition]='[pytorch,miniworld]|precision=32 mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384 d_single_token=384'
  [adaptive_layernorm]='[pytorch,miniworld]|mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[128,256,512] sweep_seq_len=384'
  [augmented_attention_token]='[pytorch,miniworld]|mask_prob=0.0 min_seq_len=128 max_seq_len=384 seq_len_step=64 d_pair_values=[128,256,512] sweep_seq_len=384'
  [augmented_attention_atom]='[pytorch,miniworld]|mask_prob=0.0 min_seq_len=128 max_seq_len=384 seq_len_step=128 d_pair_values=[16,32,64] sweep_seq_len=384'
)
TARGETS=(${MODULE_TARGETS_OVERRIDE:-bias_only_attention triangle_attention transition triangle_multiplication triangle_multiplication_bidirectional conditioned_transition adaptive_layernorm augmented_attention_token augmented_attention_atom})

run_one() {  # $1=gpu $2=target $3=mode $4=axis $5=suffix
  local gpu="$1" t="$2" mode="$3" axis="$4" sfx="$5"
  local spec="${SPEC[$t]}"; local impls="${spec%%|*}"; local extra="${spec#*|}"
  CUDA_VISIBLE_DEVICES="$gpu" TRITON_CACHE_DIR="/tmp/triton_gpu$gpu" \
    pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/bench.py "$@"' _ \
    kernel="$t" implementations="$impls" metric=time compile=true cudagraph=manual \
    mode="$mode" sweep_axis="$axis" name_suffix="$sfx" $extra \
    > "$LOGDIR/mb_${t}_${mode}_${axis}.log" 2>&1
  echo "[done gpu$gpu] $t $mode $axis (exit $?)  log=$LOGDIR/mb_${t}_${mode}_${axis}.log"
}

# --- Phase 1: fan all (target × mode × axis) bench runs across the GPUs ---
TASKS=()
for t in "${TARGETS[@]}"; do
  for mode in inference training; do
    TASKS+=("$t|$mode|seq_len|L_sweep")
    TASKS+=("$t|$mode|d_pair|d_sweep")
  done
done
echo "=== phase 1: ${#TASKS[@]} bench tasks across $NGPU GPUs ==="
i=0
for task in "${TASKS[@]}"; do
  IFS='|' read -r t mode axis sfx <<< "$task"
  run_one "$((i % NGPU))" "$t" "$mode" "$axis" "$sfx" &
  i=$((i+1)); (( i % NGPU == 0 )) && wait
done
wait
echo "=== phase 1 done ==="

# --- Phase 2: render one figure per latest CSV (serial; plotting is cheap/CPU) ---
render_latest() {  # $1=target $2=mode $3=axis $4=suffix
  local target="$1" mode="$2" axis="$3" suffix="$4"
  local module_dir; module_dir="$(module_dir_for "$target")"
  local artifact_root="benchmarks/modules/${module_dir}/artifacts"
  local run_tail="compile_cudagraph-manual_${axis}_${suffix}"
  local csv_path
  csv_path="$(find "$artifact_root" -type f -name "${target}_n_layers=1_${mode}_time_*_${run_tail}.csv" -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1s/^[^ ]* //p')"
  [[ -z "${csv_path}" ]] && { echo "no CSV: $target $mode $axis"; return 1; }
  local output_dir; output_dir="$(dirname "$csv_path" | sed 's#/artifacts/#/plots/#')"
  pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/plot_csv.py "$@"' _ "$csv_path" "$output_dir"
  echo "[rendered] $csv_path -> $output_dir (exit $?)"
}
echo "=== phase 2: rendering ==="
for t in "${TARGETS[@]}"; do
  for mode in inference training; do
    render_latest "$t" "$mode" seq_len L_sweep
    render_latest "$t" "$mode" d_pair d_sweep
  done
done

echo ""
echo "MODULE BENCH DONE (H100) targets=${TARGETS[*]}"
