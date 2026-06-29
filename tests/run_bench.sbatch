#!/bin/bash
#SBATCH --job-name=mwk_uni
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/artifacts/unified_%j.out

set -uo pipefail
cd /home/psk6950/miniworld-kernels
ENV=/home/psk6950/team-gm/.pixi/envs/default
PY=$ENV/bin/python
# Use the conda env's libstdc++ (system gcc's lacks CXXABI_1.3.15 for matplotlib).
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src
export HYDRA_FULL_ERROR=1

echo "host=$(hostname)  date=$(date)"
"$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

run() {  # $1 = kernel, rest = extra hydra overrides
  k="$1"; shift
  echo ""
  echo "############################################################"
  echo "# benchmarks/runners/bench.py  kernel=$k  $*"
  echo "############################################################"
  "$PY" benchmarks/runners/bench.py kernel="$k" \
    implementations='[pytorch,triton,cuequivariance]' \
    mode=forward metric=time max_seq_len=512 seq_len_step=128 "$@"
  echo "[exit=$?]"
}

# quick verify (no compile) across the kernel ops
run triangle_multiplication compile=false
run transition            compile=false
run triangle_attention    compile=false
run adaptive_layernorm    compile=false implementations='[pytorch,triton]'

# honor the team-gm default (compile=true) on one op
run triangle_multiplication compile=true

echo ""
echo "ALL UNIFIED BENCHES DONE"
