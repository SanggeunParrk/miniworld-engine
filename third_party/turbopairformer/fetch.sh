#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/../_attn_common.sh"
DEST=${DEST:-$HERE/upstream}
# the cp314 / torch 2.10 wheel is used for its Python and CUDA SOURCES only (its _ext/*.so are excluded): the engine env builds the extensions itself (build_local.sh, JIT)
wheel_pin https://files.pythonhosted.org/packages/dc/de/6e9e473769859153b04c59b62de7b7f1608e485912ec24ef679ed51a7da0/turbopairformer-0.1.3.post1-cp314-cp314-manylinux_2_28_x86_64.whl 61275e1063be841a555f74d21c6dd128f9dcd286e1b492e579fb521c0134d020 "$DEST" "turbopairformer/_ext/*.so"
