#!/bin/bash
# Idempotent, pinned: unpack the fast_trimul 3.0.6 wheel (pure Python) into TARGET (default: ./upstream next to this script).
#   fetch.sh [TARGET]                 TARGET is added to sys.path by benchmarks/paper/E5_other_kernels/ext_fast_trimul.py
#   CUTLASS_DSL=1 fetch.sh [TARGET]   also unpacks nvidia-cutlass-dsl 4.8.0 (cp312 x86_64 cu12 wheels) + protobuf 7.36.2 for a host whose environment has none
#                                     (the 5090 venv); the engine envs of H100 / A100 already have 4.8.0 + cuda-bindings 12.9
# Nothing is installed into any shared environment; wheels are checked against the pinned sha256 before they are unpacked.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TARGET=${1:-$HERE/upstream}
DL=${DL:-$HERE/.download}
mkdir -p "$TARGET" "$DL"
PP=https://files.pythonhosted.org/packages

get() {  # name sha256 url
  local f="$DL/$1"
  [ -f "$f" ] || curl -fsSL -o "$f" "$3"
  echo "$2  $f" | sha256sum -c --quiet - || { echo "sha256 mismatch: $1" >&2; rm -f "$f"; exit 1; }
  unzip -qo "$f" -d "$TARGET"
}

get fast_trimul-3.0.6-py3-none-any.whl 2124ba64da111ec5ad5e8050a6945fc682a0c022f906b1146146b8cf0ac7a7be \
  $PP/1f/7f/4c87495abd81d55e7fdfe9c0f113b6e579cc8fbb10403c0012e312f8c9bb/fast_trimul-3.0.6-py3-none-any.whl
if [ "${CUTLASS_DSL:-0}" = 1 ]; then
  get nvidia_cutlass_dsl-4.8.0-py3-none-any.whl 24f996e0e5fa88c8b417f4893e615af135bdfdfdbdbed7e239faf36a88df711b \
    $PP/e4/f4/e96f452f7975dd2fd4d2a4af88c63205431dad7efbe63aff6e3a4b1ada17/nvidia_cutlass_dsl-4.8.0-py3-none-any.whl
  get nvidia_cutlass_dsl_libs_core-4.8.0-py3-none-any.whl b32716d2ed10c185cdbbb6b5c6fca26a3113a60e6e2fe4e2f0cb15966422b621 \
    $PP/85/66/d05f0ae372b80fc757e59d5211d93483c6ff721f574cb68cad84ceb87157/nvidia_cutlass_dsl_libs_core-4.8.0-py3-none-any.whl
  get nvidia_cutlass_dsl_libs_base-4.8.0-cp312-cp312-manylinux_2_28_x86_64.whl 5bdc80989b0fbbac3d58960057e74e2ce8676fafa76b852d9422b9c4b9b74bdd \
    $PP/12/1c/2f7a13626af1edfefb5ad69234d47e3b212064a42ad2e6db59351b5ae002/nvidia_cutlass_dsl_libs_base-4.8.0-cp312-cp312-manylinux_2_28_x86_64.whl
  get nvidia_cutlass_dsl_libs_cu12-4.8.0-cp312-cp312-manylinux_2_28_x86_64.whl 2374cfdff635ceaf7e2c5bd17b4beb410830420a4fd65b12b7f24b5d5cbb9d4c \
    $PP/13/a0/fdde64cc231f52a716107bf1a1246db9dde8cb11b28278b8d9c05f7c87bd/nvidia_cutlass_dsl_libs_cu12-4.8.0-cp312-cp312-manylinux_2_28_x86_64.whl
  get protobuf-7.36.2-py3-none-any.whl bdb3a345d48db958e6ce1f18e508beb0cc981d64f24088427549c866cd039f1e \
    $PP/e4/04/d52c7016b04b6c5108f26691f9d33ec82a9b65d041f1a9c771137693d618/protobuf-7.36.2-py3-none-any.whl
fi
echo "fast_trimul 3.0.6 unpacked into $TARGET"
