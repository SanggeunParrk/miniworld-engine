#!/bin/bash
# measure_b200.sh -- the B200 numbers of docs/gpus/b200/token_dit/token_dit.md: build the four sm_100a cores, then core / inference step / training block
# for L = 384 and 768, one process each; stops if anything else is on the GPU. Run under: gpuq run -g 6 -n tdit -- bash measure_b200.sh
set -u
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950
E=$W/mw-dit/experiments
R=${OUT:-$W/scratch/tdit/$(date +%F_%H%M)}
mkdir -p "$R"
export TMPDIR=$W/.tmp XDG_CACHE_HOME=$W/.cache MPLCONFIGDIR=$W/.cache/matplotlib PYTHONNOUSERSITE=1
export TRITON_CACHE_DIR=$W/.cache/triton_dit TORCH_EXTENSIONS_DIR=$W/.cache/torch_extensions_dit
export MINIWORLD_ENGINE_JIT_ROOT=$W/.cache/miniworld_engine_jit_dit CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$W/mw-dit/src:$W/refs/uplifting-biomolecular-modeling/common/opt_core OPT_CORE_CELL_CENSUS_PRINT=0
source $W/miniworld-engine/.venv/bin/activate

guard() {  # log the GPU state; stop if anything but us is on the GPU
    local uuid n
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPUQ_GPU" '$1==g{print $2}')
    n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$uuid")
    echo "## $(date +%T) $1  gpu$GPUQ_GPU foreign=$n  $(nvidia-smi -i "$GPUQ_GPU" --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv,noheader)"
    [ "$n" = 0 ] || { echo "GPU busy, abort"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$uuid"; exit 3; }
}

{
echo "# start $(date)  host $(hostname)  gpu $GPUQ_GPU  torch $(python -c 'import torch;print(torch.__version__)')"
( cd $E/augattn_sm100 && for k in attn_fwd2 attn_dqb attn_dkv attn_inf; do bash build.sh $k || exit 4; done ) || { echo "build failed"; exit 4; }
for L in 384 768; do guard "core L$L"; (cd $E/augattn_sm100 && python -W ignore bench_train.py $L); done
for L in 384 768; do guard "step L$L"; (cd $E/token_dit_fused && python -W ignore bench.py --length $L --save "$R/step-bf16-L$L.json"); done
for L in 384 768; do guard "train L$L"; (cd $E/token_dit_fused && python -W ignore train_block.py --length $L --stack 4 --variants pytorch engine-b200 tdit-b200); done
echo "# end $(date)"
} 2>&1 | tee "$R/log.txt"
