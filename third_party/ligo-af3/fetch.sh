#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/../_attn_common.sh"
DEST=${DEST:-$HERE/upstream}
git_pin https://github.com/Ligo-Biosciences/AlphaFold3.git 621531e4f36c3ce5e20157f88f27049d3f3f1fcc "$DEST"
# patches (ours, applied on top of the pinned revision; idempotent)
for p in "$HERE"/patches/*.patch; do
  if git -C "$DEST" apply --check "$p" 2>/dev/null; then git -C "$DEST" apply "$p" && echo "[fetch] applied $(basename "$p")"; else echo "[fetch] $(basename "$p") already applied (or not applicable)"; fi
done
