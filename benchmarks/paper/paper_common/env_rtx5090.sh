# RTX 5090 (cssb-rtx, no Slurm, 30 GB RAM shared): run through ~/mwe/gpu.sh (flock + 16 GB memory cap), from a ~/mwe/ copy of the source tree
# (never ~/mwe/engine). No kit stack: the card's own driver is CUDA 13, the Anthropic kernels run in the engine environment.
CARD=rtx5090
ENGINE=${ENGINE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
. "$HOME/mwe/env.sh"
[ -f "$HOME/mwe/venv/bin/activate" ] && . "$HOME/mwe/venv/bin/activate"
export PYTHONPATH=$ENGINE/src PYTHONUNBUFFERED=1
export MINIWORLD_ANTHROPIC_ROOT=${MINIWORLD_ANTHROPIC_ROOT:-$HOME/mwe/engine/third_party/anthropic/upstream}  # the release root of the Anthropic kernels (read only)
KIT_WRAP=
cd "$ENGINE"
