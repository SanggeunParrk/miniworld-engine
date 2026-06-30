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
    implementations='[pytorch,dtv1,cuequivariance,miniworld]' \
    metric=time "$@"
  echo "[exit=$?]"
}

render_latest() {  # $1 = mode, $2 = sweep suffix in the CSV run name
  mode="$1"
  suffix="$2"
  artifact_root="benchmarks/modules/triangle_multiplication/artifacts"
  csv_path="$(find "$artifact_root" -type f -name "triangle_multiplication_n_layers=1_${mode}_time_bf16-mixed_compile_*_${suffix}.csv" -printf '%T@ %p\n' | sort -nr | sed -n '1s/^[^ ]* //p')"
  if [[ -z "${csv_path}" ]]; then
    echo "No CSV found for mode=${mode}, suffix=${suffix}; skipping render"
    return 1
  fi
  output_dir="$(dirname "$csv_path")"
  echo ""
  echo "############################################################"
  echo "# benchmarks/runners/plot_csv.py  csv=$csv_path"
  echo "############################################################"
  pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/plot_csv.py "$@"' _ "$csv_path" "$output_dir"
  echo "[exit=$?]"
}

# trimul first: CSV generation is the benchmark step; SVG rendering is separate.
for bench_mode in inference training; do
  run triangle_multiplication mode="$bench_mode" compile=true sweep_axis=seq_len name_suffix=L_sweep
  render_latest "$bench_mode" L_sweep
  run triangle_multiplication mode="$bench_mode" compile=true sweep_axis=d_pair name_suffix=d_sweep
  render_latest "$bench_mode" d_sweep
done

echo ""
echo "ALL UNIFIED BENCHES DONE"
