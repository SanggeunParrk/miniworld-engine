#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/../_attn_common.sh"
DEST=${DEST:-$HERE/upstream}
wheel_pin https://files.pythonhosted.org/packages/11/b3/5b2961811e4e47fcad1e3eb2fab6cc96285ebaf03768304ef84980fb3dcf/trifast-0.1.13-py3-none-any.whl e96fc0224b85e98802b04c495380198387110e3e712ce2ca7d2e672a27e10496 "$DEST"
# trifast imports platformdirs (pure python): pinned like ~/paper-ext (platformdirs 4.14.8; pip install --target $DEST --no-deps platformdirs==4.14.8 if the engine env lacks it)
