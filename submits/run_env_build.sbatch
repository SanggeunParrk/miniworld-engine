#!/bin/bash
# Build the unified pixi env for THIS checkout on the CSSB A5000/A6000 cluster
# (login node = cssb-master, partitions: gpu {A5000,A6000}, cpu).
#
# The env build needs NO GPU (nvcc from the cuda-toolkit conda pkg compiles
# transformer_engine_torch from source under no-build-isolation), and the cpu
# partition's compute nodes DO have outbound network, so the whole resolve +
# download + TE compile runs off the shared login node on the cpu partition.
#
#   sbatch submits/run_env_build.sbatch
#
# --frozen installs EXACTLY the committed pixi.lock (keeps the cu12 TE core;
# a bare `pixi install` can re-pin the cu13 core, which sm86 + this driver
# cannot load). fix-te-cu12 reinstalls the cu12 core last so it wins.
#SBATCH --job-name=mwk_env_build
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/env_build_%j.out
set -uo pipefail
cd /home/psk6950/practice/miniworld-engine

echo "host=$(hostname)  date=$(date)  pixi=$(pixi --version)"

# ---------------------------------------------------------------------------
# Cold-install ordering fix for transformer-engine-torch (no-build-isolation).
# A fresh `pixi install` builds the TE-torch sdist BEFORE torch is placed in the
# env, so its setup.py `import torch` fails ("This package needs Torch to build").
# The existing ~/miniworld-engine env never hit this because it was built
# incrementally (torch already present). Fix: seed torch + prebuild TE into the
# env FIRST (torch importable), so the subsequent `pixi install --frozen` sees
# both already satisfied at the locked versions and SKIPS the problematic build,
# then just reconciles the pure-wheel deps (triton/cutlass/quack/cuequiv) + the
# editable package. Versions are pinned to pixi.lock (torch 2.10.0+cu128, TE 2.16.0).
# The conda phase already ran (python/cuda-toolkit/nvcc/ninja/pybind11 present).
# ---------------------------------------------------------------------------
ENV="$PWD/.pixi/envs/default"
if [ ! -x "$ENV/bin/python" ]; then
  echo "conda-phase env skeleton missing; running pixi install once to create it (may fail at TE — expected)"
  pixi install --frozen || true
fi
export CONDA_PREFIX="$ENV" CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"

echo "=== phase 1: seed torch (so TE-torch can import it at build time) ==="
"$ENV/bin/python" -m pip install --no-cache-dir \
  "torch==2.10.0+cu128" \
  --extra-index-url https://download.pytorch.org/whl/cu128
echo "[seed torch exit=$?]"
"$ENV/bin/python" -c "import torch; print('torch importable:', torch.__version__)"

echo "=== phase 2: prebuild transformer-engine-torch against the present torch ==="
"$ENV/bin/python" -m pip install --no-cache-dir --no-build-isolation \
  "transformer-engine-torch==2.16.0"
echo "[prebuild TE exit=$?]"

echo "=== phase 3: pixi install --frozen (TE already built+satisfied -> no rebuild; reconcile rest) ==="
pixi install --frozen
echo "[pixi install exit=$?]"

echo "=== fix-te-cu12 (reinstall cu12 TE core last so it wins over cu13) ==="
pixi run --frozen fix-te-cu12
echo "[fix-te-cu12 exit=$?]"

# CPU-only sanity (no GPU on this partition): import torch + triton, print versions.
echo "=== smoke (import only; GPU checks run separately on the gpu partition) ==="
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; python -c "import torch, triton; print(\"torch\", torch.__version__, \"triton\", triton.__version__)"'
echo "[smoke exit=$?]"
echo "ENV BUILD DONE"
