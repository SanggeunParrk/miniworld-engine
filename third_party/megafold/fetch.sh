#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/../_attn_common.sh"
DEST=${DEST:-$HERE/upstream}
git_pin https://github.com/Supercomputing-System-AI-Lab/MegaFold.git 845905e7c707cf4004404354156df97f2efc4fde "$DEST"
# liger-kernel 0.8.5 (BSD-2-Clause; pure Python Triton helpers imported by MegaFold's FusedTransition / FusedLayernormLinear): next to the tree, default $HERE/liger-kernel
wheel_pin https://files.pythonhosted.org/packages/8f/73/9667e4a05457e99ec921380df8d279fbcb8f8e8bd76b033ee0c82bb1d3a0/liger_kernel-0.8.5-py3-none-any.whl 60789cd576af073ab9b192b570d2597fd0cdad489cd62fb44fb16256d43754c9 "${LIGER_DEST:-$(dirname "$DEST")/liger-kernel}"
