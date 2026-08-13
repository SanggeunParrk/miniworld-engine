#!/bin/bash
# Build/refresh ALL shipped Triton autotune caches for H100 (sm_90) via the capture builder.
# Every module is run (miniworld, inference+training, seq_len+d_pair sweeps) with the autotuner
# instrumented; top-5 per (op,dtype,bucket) is flushed into src/.../autotune/data/<op>/<gpu>.json.
# d_pair sweeps INCLUDE 64 so every cached op gets D=64 entries.
#   CAPTURE_TARGET=all sbatch submits/run_autotune_capture_h100.sbatch
#SBATCH --job-name=cap_h100
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=11:00:00
#SBATCH --output=/home/psk6950/cap_h100_%j.out
set -uo pipefail
set -f
cd /home/psk6950/miniworld-engine
export HYDRA_FULL_ERROR=1
echo "host=$(hostname) date=$(date) target=${CAPTURE_TARGET:-all}"
pixi run --frozen python -c "
import torch
name=torch.cuda.get_device_name(0); cap=torch.cuda.get_device_capability(0)
print('capture on', name, 'sm', cap, torch.__version__)
assert cap[0]==9, f'expected sm_90 (H100), got sm_{cap[0]}{cap[1]} ({name})'
"
declare -A SHAPES
SHAPES[transition]='mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[64,128,256,512] sweep_seq_len=384'
SHAPES[triangle_multiplication]='mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[64,128,256,512] sweep_seq_len=384'
SHAPES[triangle_multiplication_bidirectional]='mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[64,128,256,512] sweep_seq_len=384'
SHAPES[triangle_attention]='mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[64,128,256,512] sweep_seq_len=384'
SHAPES[bias_only_attention]='mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[64,128,256,512] sweep_seq_len=384'
SHAPES[adaptive_layernorm]='mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[64,128,256,512] sweep_seq_len=384'
SHAPES[conditioned_transition]='precision=32 mask_prob=0.0 min_seq_len=384 max_seq_len=1024 seq_len_step=128 d_pair_values=[64,128,256,512] sweep_seq_len=384 d_single_token=384'
SHAPES[augmented_attention_token]='mask_prob=0.0 min_seq_len=128 max_seq_len=384 seq_len_step=64 d_pair_values=[64,128,256,512] sweep_seq_len=384'
SHAPES[augmented_attention_atom]='mask_prob=0.0 min_seq_len=128 max_seq_len=384 seq_len_step=128 d_pair_values=[16,32,64] sweep_seq_len=384'
run_one() {
  target="$1"; shift
  echo ""; echo "### capture kernel=${target} $* ###"
  pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-} ; PYTHONPATH=src python benchmarks/runners/bench.py "$@"' _ \
    kernel="$target" implementations='[miniworld]' metric=time compile=false cudagraph=manual "$@"
  echo "[exit=$?]"
}
run_matrix() {
  target="$1"; sh="${SHAPES[$target]}"
  for mode in inference training; do
    run_one "$target" mode="$mode" cudagraph=manual sweep_axis=seq_len name_suffix=cap_L $sh
    run_one "$target" mode="$mode" cudagraph=manual sweep_axis=d_pair name_suffix=cap_d $sh
  done
}
TARGETS="${CAPTURE_TARGET:-all}"
if [ "$TARGETS" = "all" ]; then
  TARGETS="transition triangle_multiplication triangle_multiplication_bidirectional triangle_attention bias_only_attention adaptive_layernorm conditioned_transition augmented_attention_token augmented_attention_atom"
fi
for t in $TARGETS; do run_matrix "$t"; done
DEV_KEY="$(pixi run --frozen python -c "from miniworld_engine.autotune.cache import gpu_key; print(gpu_key())" 2>/dev/null)"
echo ""; echo "### caches for gpu_key=${DEV_KEY} ###"
find src/miniworld_engine/autotune/data -name "${DEV_KEY}.json" -printf '%p\n' 2>/dev/null | sort | wc -l
echo "CAPTURE DONE (git add src/miniworld_engine/autotune/data)"
