#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/../_attn_common.sh"
DEST=${DEST:-$HERE/upstream}
wheel_pin https://files.pythonhosted.org/packages/2a/73/5e483d02f76c3f3e588c7d6c95b989aa5bb7da10b9d0af5bb46932aeda78/bionemo_ir-0.2.2-cp312-abi3-manylinux_2_34_x86_64.whl fa58528015d111c5c4bbacddb79110c814999f4a0d6f09ca5a5db2f3cb6e4fe0 "$DEST"
