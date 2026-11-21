#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/../_attn_common.sh"
DEST=${DEST:-$HERE/upstream}
git_pin https://github.com/ROCm/geometric-nn-ops.git 232ed824269b55cc26f533cc951ac14ee6cd4007 "$DEST"
