# B200 (sm_100a, host b200 = 8x B200, shared account; no Slurm): only GPUs 0-3 are ours (a colleague trains on 4-7). Run from a clone of the source tree
# (the PyPI wheels carry no sm100 kernels), through b200_lane.sh / b200_task.sh (one lane = one GPU, CUDA_VISIBLE_DEVICES set by the lane).
# Python environment: the pixi env of an earlier B200 checkout (torch 2.13.0+cu129, triton 3.7.1, cuequivariance 0.12, CUDA 12.9 toolkit), used read-only
# (never `pixi run` / install into it). All caches live under $PAPER_B200_HOME, nothing in ~ or /tmp.
CARD=b200
W=${W:-/NHNHOME/WORKSPACE/26mohw002_A/psk6950}
PAPER_B200_HOME=${PAPER_B200_HOME:-$W/paper-b200}
ENGINE=${ENGINE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
ENV=${ENGINE_ENV:-$W/eng-2511/.pixi/envs/default}
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} PYTHONPATH=${ENGINE_PYTHONPATH:-$ENGINE/src} PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export TMPDIR=$PAPER_B200_HOME/tmp XDG_CACHE_HOME=$PAPER_B200_HOME/cache MPLCONFIGDIR=$PAPER_B200_HOME/cache/matplotlib
export MINIWORLD_ENGINE_JIT_ROOT=$XDG_CACHE_HOME/jit TORCH_EXTENSIONS_DIR=$XDG_CACHE_HOME/torch_ext TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/inductor
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} CUDA_DEVICE_ORDER=PCI_BUS_ID NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++"
export PATH=$PATH:$CUDA_HOME/bin
export CUTLASS_PATH=${CUTLASS_PATH:-$W/refs/cutlass} CUTLASS_DIR=${CUTLASS_DIR:-$W/refs/cutlass}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
export MINIWORLD_ANTHROPIC_ROOT=${MINIWORLD_ANTHROPIC_ROOT:-$W/eng-2511/third_party/anthropic/upstream}
# kit stack: the B200 driver (CUDA 13.1) needs no container: kit_stack_b200.sh with KIT_VENV = a venv of torch 2.13.0+cu130 + triton 3.7.1 (built under $PAPER_B200_HOME),
# else the engine environment itself
[ -z "${KIT_VENV:-}" ] && [ -x "$PAPER_B200_HOME/kitvenv130/bin/python" ] && KIT_VENV=$PAPER_B200_HOME/kitvenv130
export KIT_VENV
if [ -n "${KIT_VENV:-}" ]; then KIT_WRAP=${KIT_WRAP:-$ENGINE/third_party/anthropic/kit_stack_b200.sh}; else KIT_WRAP=${KIT_WRAP:-}; fi

# E5 libraries (fetch.sh of every third_party/<name>/ and pip --target; never installed into the engine environment)
export PAPER_EXT_PATH=${PAPER_EXT_PATH:-$PAPER_B200_HOME/paper-ext} PAPER_EXT_ATTN=${PAPER_EXT_ATTN:-$PAPER_B200_HOME/paper-ext-attn}
cd "$ENGINE"
