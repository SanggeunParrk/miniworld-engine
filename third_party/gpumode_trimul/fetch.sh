#!/bin/bash
# Idempotent, pinned: the GPU MODE TriMul competition kernels of arseniivanov/trimul (MIT), raw files at one commit, sha256 verified.
#   fetch.sh [TARGET]      TARGET (default ./upstream next to this script) gets the files in TARGET/gpumode_trimul/ (a package directory);
#                          benchmarks/paper/E5_other_kernels/ext_gpumode_trimul.py adds TARGET to sys.path.
# Not fetched on purpose: kernel_global of Hugging Face kernels-community/trimul_gpumode (adapted from davidberard98/gpumode-trimul, which has
# no license file); see README.md.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TARGET=${1:-$HERE/upstream}
D=$TARGET/gpumode_trimul
mkdir -p "$D"
REV=558b9f26382ec9b88b961d9f8306bf46706da0f6
BASE=https://raw.githubusercontent.com/arseniivanov/trimul/$REV
while read -r sum name; do
  f=$D/$name
  [ -f "$f" ] && echo "$sum  $f" | sha256sum -c --quiet - 2>/dev/null || curl -fsSL -o "$f" "$BASE/$name"
  echo "$sum  $f" | sha256sum -c --quiet - || { echo "sha256 mismatch: $name" >&2; rm -f "$f"; exit 1; }
done <<'LIST'
841a08280d21614c305aea6d8ab96c19d4b9bdef6a658d91fadd1900693d5e0b LICENSE
856c8f04c06fc4b0768fca5e267c34f1884355054f89c629a6aac07b3c099174 task.py
e90405e5386923d643753ac70f09d3441925835cefb41849f229a340a0e26ff6 reference.py
a8a6a725c4fcc3d5e9cd50b9299a77d9f8a39e41830a994481171033f95b2681 utils.py
5fa85c21db75661b1dd00d0050b38becced566fd3e849e35d0af80c95ab844d4 triton_a100.py
6a22a05344f6a2ebda5b2067b36e135268f261fea3b9f8f287548f5e9ddaa92b triton_h100.py
105ed18686007c003aebc4695e06271fa878fd1b96a695717938bd6f8d881ce2 triton_b200.py
LIST
touch "$D/__init__.py"
echo "gpumode_trimul @ $REV unpacked into $D"
