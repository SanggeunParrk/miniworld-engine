#!/bin/bash
# A/B the transition INFERENCE module: CUDA-b2b (new) vs Triton-b2b (old) vs pytorch.
set -uo pipefail
set -f
cd /home/psk6950/miniworld-kernels
CSV="benchmarks/modules/transition/artifacts/NVIDIA H100 80GB HBM3/transition_n_layers=1_inference_time_bf16-mixed_cudagraph-manual_seq_len.csv"

run() { # $1 = CUDA_B2B value (1|0)  $2 = label
  MINIWORLD_TRANSITION_CUDA_B2B="$1" pixi run --frozen bash -c '
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
    export PYTHONPATH=src
    export MINIWORLD_TRANSITION_CUDA_B2B='"$1"'
    python benchmarks/runners/bench.py \
      kernel=transition implementations=[pytorch,miniworld] mode=inference metric=time \
      compile=false cudagraph=manual allow_tf32=true precision=bf16-mixed \
      sweep_axis=seq_len d_pair=128 mask_prob=0.0 \
      min_seq_len=384 max_seq_len=1024 seq_len_step=128
  ' >/dev/null 2>&1
  echo "=== [$2  CUDA_B2B=$1]  seq_len, impl, cos, ms ==="
  python3 -c "
import csv
for r in csv.DictReader(open('$CSV')):
    L = r.get('seq_len', r.get('L', '?'))
    print('  L=%5s  %-9s  cos=%9s  ms=%s' % (L, r['implementation'], r['output_cosine'], r['value']))
"
}
echo '########## [A] CUDA-b2b ON (new dispatch) ##########'
run 1 CUDA-b2b
echo ""; echo '########## [B] CUDA-b2b OFF (Triton-b2b baseline) ##########'
run 0 Triton-b2b
echo "=== AB DONE ==="
