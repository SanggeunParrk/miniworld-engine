#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/../_attn_common.sh"
DEST=${DEST:-$HERE/upstream}
git_pin https://github.com/aqlaboratory/openfold-3.git 4b59bbf39a31b4aef7c33e5b567d25d7b4a61bf3 "$DEST"
