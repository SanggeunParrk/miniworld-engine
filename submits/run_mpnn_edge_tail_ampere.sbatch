#!/bin/bash
# ProteinMPNN encoder edge tail, one Ampere GPU. Unlike the other kernel targets this one
# is a TRAINING (backward-only autograd) bench, so kernel_targets_bench.sh -- which runs
# every target at mode=inference, cudagraph=manual -- does not cover it.
#
#   sbatch --gres=gpu:A5000:1 submits/run_mpnn_edge_tail_ampere.sbatch
#   sbatch --gres=gpu:A6000:1 submits/run_mpnn_edge_tail_ampere.sbatch
#
# Three regimes, run in order of what they answer:
#   1. compile=true  cudagraph=disabled  -- COMPILED PyTorch vs Triton. The headline. This
#      is the only regime that actually compiles (cudagraph overrides compile), and until
#      the kernel benches learned to call torch.compile it silently ran eager while the
#      CSV said compiled=True.
#   2. compile=false cudagraph=manual    -- EAGER PyTorch + graph capture, the deployment
#      regime, and the eager reference to read against regime 1.
#   3. compile=false cudagraph=graphed   -- make_graphed_callables; previously died in
#      capture for BOTH rows, so it is last and its failure is not allowed to stop 1 or 2.
#
# A5000 is 24 GB and every intermediate here is edge-sized (805 MiB at seq_len 8192), so
# the top of the sweep may OOM. bench.py records those as status=failed rows.
#SBATCH --job-name=mwk_mpnn_edge_tail
#SBATCH --partition=gpu
#SBATCH --account=cssb
#SBATCH --qos=normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=benchmarks/kernels/mpnn_edge_tail_%j.out

set -uo pipefail
set -f
cd /home/psk6950/practice/miniworld-kernels
export HYDRA_FULL_ERROR=1
export TRITON_CACHE_DIR="/tmp/triton_mpnn_edge_tail_${SLURM_JOB_ID:-$$}"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-none}"
pixi run --frozen python submits/_assert_device.py --capability 8.6 --name A5000 --name A6000 || exit 3

SWEEP='[2048,4096,8192]'
RUN_ID="${SLURM_JOB_ID:-manual_$$}"

run_one() {  # $1=label $2=compile $3=cudagraph
  echo ""
  echo "############################################################"
  echo "# $1  (compile=$2 cudagraph=$3)  $(date +%T)"
  echo "############################################################"
  # Each regime runs in its OWN process: a failed CUDA-graph capture poisons the context
  # and every later row in that process inherits "operation failed due to a previous
  # error during capture", which is exactly how the earlier artifacts lost every row.
  pixi run --frozen bash -c \
    'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/bench.py "$@"' _ \
    kernel=mpnn_edge_tail \
    implementations='[pytorch,triton_compute]' \
    metric=time mode=training \
    precision=bf16-mixed allow_tf32=true check_correctness=true \
    n_layers=1 d_pair=128 k_neighbors=48 batch_size=8 \
    sweep_axis=seq_len seq_len_values="$SWEEP" sweep_seq_len=8192 \
    compile="$2" cudagraph="$3" \
    name_suffix="$1_${RUN_ID}"
  echo "[exit=$? label=$1]"
}

run_one compiled_nograph true  disabled
run_one eager_graphed   false manual
run_one make_graphed    false graphed

echo ""
echo "=== CSVs written ==="
DEV="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
find "benchmarks/kernels/mpnn_edge_tail/artifacts/${DEV}" -name "*${RUN_ID}*.csv" -print 2>/dev/null

echo ""
echo "=== summary ==="
pixi run --frozen python submits/_mpnn_edge_tail_summary.py "${DEV}" "${RUN_ID}"

echo "MPNN EDGE TAIL DONE job=${SLURM_JOB_ID:-none}"
