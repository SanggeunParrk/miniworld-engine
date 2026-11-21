#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/../_attn_common.sh"
DEST=${DEST:-$HERE/upstream}
git_pin https://github.com/thuml/FlashBias.git 254a8b87b12a47f6cd369d6b3885f32ade20233d "$DEST"
