#!/bin/bash
#SBATCH --job-name=mwk_trimul
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/modules/triangle_multiplication/artifacts/trimul_%j.out

set -uo pipefail
cd /home/psk6950/miniworld-kernels
export HYDRA_FULL_ERROR=1

echo "host=$(hostname)  date=$(date)"
pixi run --frozen python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

run() {  # $1 = kernel, rest = extra hydra overrides
  k="$1"; shift
  echo ""
  echo "############################################################"
  echo "# benchmarks/runners/bench.py  kernel=$k  $*"
  echo "############################################################"
  pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/bench.py "$@"' _ kernel="$k" \
    implementations='[pytorch,triton,cuequivariance]' \
    mode=inference metric=time max_seq_len=512 seq_len_step=128 "$@"
  echo "[exit=$?]"
}

# trimul first: final benchmark format only, compiled path only.
run triangle_multiplication compile=true

echo ""
echo "ALL UNIFIED BENCHES DONE"
