# H100 (cssb, Slurm partition h100): the engine's pixi env (never `pixi run`: it re-syncs the shared env), the source tree of this checkout.
# Kit stack for the Anthropic kernels: third_party/anthropic/kit_stack.sh (CUDA-13 container, torch 2.13+cu130 kit venv).
CARD=h100
ENGINE=${ENGINE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
ENV=${ENGINE_ENV:-$HOME/miniworld-engine/.pixi/envs/default}
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} PYTHONPATH=${ENGINE_PYTHONPATH:-$ENGINE/src} PYTHONUNBUFFERED=1
export MINIWORLD_ANTHROPIC_ROOT=${MINIWORLD_ANTHROPIC_ROOT:-$HOME/miniworld-engine/third_party/anthropic/upstream}
export KITS_DIR=${KITS_DIR:-$HOME/foldforge-scratch/anthropic-kits} FF_SCRATCH=${FF_SCRATCH:-$HOME/foldforge-scratch}
KIT_WRAP=${KIT_WRAP:-$ENGINE/third_party/anthropic/kit_stack.sh}
cd "$ENGINE"
