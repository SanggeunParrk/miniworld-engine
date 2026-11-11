#!/bin/bash
# The repository gate: lint, types and the CPU tests; the GPU tests separately (they need an H100).
#
#   tools/check.sh            ruff check + ruff format (report only) + pyright + CPU tests       (any machine, no GPU)
#   tools/check.sh gpu        GPU tests: tests/gpu (accuracy smoke of the engine backends, determinism; an H100, < 2 min warm)
#   tools/check.sh all        both
#
# Run it where Python may run: on a Slurm cluster that is inside a job, never on the login node, e.g.
#   sbatch --gres=none --time=00:10:00 --wrap "tools/check.sh"
#   sbatch --gres=gpu:h100:1 --time=00:10:00 --wrap "tools/check.sh gpu"
# Environment: the repo's Python (pixi `default` env: its bin/ on PATH) with the `dev` extra (ruff, pyright, pytest, matplotlib).
#   PYRIGHT   the pyright command (default `pyright`; e.g. "node /path/to/pyright/index.js" where the PyPI wrapper cannot download node)
#   FORMAT_STRICT=1   fail on `ruff format --check` differences (default: report the count only; the tree is not mass-formatted)
set -u
cd "$(dirname "$0")/.." || exit 2
what=${1:-cpu}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
rc=0
step() { echo; echo "=== $*"; }

if [ "$what" = cpu ] || [ "$what" = all ]; then
  step "ruff check"
  ruff check src tools tests || rc=1
  step "ruff format --check (report)"
  ruff format --check src tools tests > /dev/null 2>&1; frc=$?
  ruff format --check src tools tests 2>&1 | tail -1
  if [ "${FORMAT_STRICT:-0}" = 1 ] && [ "$frc" != 0 ]; then rc=1; fi
  # the gate: 0 errors over [tool.pyright] (src/ and tests/, basic mode; the vendored triton/ and cuda/ kernel bodies are excluded there)
  step "pyright: src + tests"
  ${PYRIGHT:-pyright} --pythonpath "$(command -v python)" || rc=1
  step "pytest: CPU tests"
  CUDA_VISIBLE_DEVICES= python -m pytest tests/cpu -m "not gpu" -q || rc=1
fi
if [ "$what" = gpu ] || [ "$what" = all ]; then
  step "pytest: GPU tests"
  python -m pytest tests/gpu -m gpu -q || rc=1
fi
echo; [ $rc = 0 ] && echo "=== check: OK" || echo "=== check: FAILED"
exit $rc
