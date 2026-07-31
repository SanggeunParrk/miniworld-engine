#!/bin/bash
# Build/refresh the shipped Triton autotune caches for H100 (sm_90) via the capture builder:
# instrument the Triton autotuner during real module runs (miniworld backend, inference+training,
# seq_len + d_pair sweeps) and flush top-5 per (op,dtype,bucket) into the in-repo cache
#   src/miniworld_kernels/autotune/data/<op>/NVIDIA H100 80GB HBM3 (sm90).json
# d_pair sweep INCLUDES 64 so the cached ops get D=64 entries too. compile=false + cudagraph=manual
# gives clean per-config triton timings.
#   CAPTURE_TARGET=all sbatch submits/run_autotune_capture_h100.sbatch
#SBATCH --job-name=cap_h100
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/home/psk6950/cap_h100_%j.out
set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels
export HYDRA_FULL_ERROR=1
echo "host=$(hostname) date=$(date) target=${CAPTURE_TARGET:-all}"
pixi run --frozen python -c "
import torch
name=torch.cuda.get_device_name(0); cap=torch.cuda.get_device_capability(0)
print('capture on', name, 'sm', cap, torch.__version__)
assert cap[0]==9, f'expected sm_90 (H100), got sm_{cap[0]}{cap[1]} ({name})'
"
# d_pair sweep with D=64 added (cached ops get 64/128/256/512 entries)
SHAPES='mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[64,128,256,512] sweep_seq_len=384'
run_one() {
  target="$1"; shift
  echo ""; echo "### capture kernel=${target} $* ###"
  pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-} MINIWORLD_RUN_AUTOTUNE=1 MINIWORLD_AUTOTUNE_CAPTURE=1; PYTHONPATH=src python benchmarks/runners/bench.py "$@"' _ \
    kernel="$target" implementations='[miniworld]' metric=time compile=false cudagraph=manual "$@"
  echo "[exit=$?]"
}
run_matrix() {
  target="$1"
  for mode in inference training; do
    run_one "$target" mode="$mode" cudagraph=manual sweep_axis=seq_len name_suffix=cap_L $SHAPES
    run_one "$target" mode="$mode" cudagraph=manual sweep_axis=d_pair name_suffix=cap_d $SHAPES
  done
}
for t in transition triangle_multiplication triangle_multiplication_bidirectional; do
  run_matrix "$t"
done
DST="src/miniworld_kernels/autotune/data"
DEV_KEY="$(pixi run --frozen python -c "from miniworld_kernels.autotune.cache import gpu_key; print(gpu_key())" 2>/dev/null)"
echo ""; echo "### caches present for gpu_key=${DEV_KEY} ###"
find "$DST" -name "${DEV_KEY}.json" -printf '%p\n' 2>/dev/null | sort
echo "CAPTURE DONE (git add ${DST})"
