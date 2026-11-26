# A100 (cssb3, Slurm partition A100; a separate home): the engine env of that host. Kit stack for the Anthropic kernels:
# third_party/anthropic/kit_stack_compat.sh (CUDA-13.0 forward-compatible libcuda in front of the host's; KIT_COMPAT_DIR, default ~/cuda-compat-13.0),
# FOLDJET_ENGINE_ANTHROPIC_ROOT = a release root with the cu129 / py312 prebuilt (cssb3: ~/w45-jobs/upstream).
CARD=a100
ENGINE=${ENGINE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
ENV=${ENGINE_ENV:-$HOME/practice/foldjet-engine/.pixi/envs/default}
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} PYTHONPATH=$ENGINE/src PYTHONUNBUFFERED=1
export FOLDJET_ENGINE_ANTHROPIC_ROOT=${FOLDJET_ENGINE_ANTHROPIC_ROOT:-$HOME/w45-jobs/upstream}
KIT_WRAP=$ENGINE/third_party/anthropic/kit_stack_compat.sh
cd "$ENGINE"
