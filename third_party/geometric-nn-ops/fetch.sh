#!/bin/bash
# Idempotent, pinned: AMD ROCm/geometric-nn-ops release/v1.0.0 (Apache-2.0), the geometric_nn_ops_torch package (pure Python + Triton source),
# raw files at one commit, sha256 verified; one entry for both E5 adapters (geonn: triangle_multiplicative_update, gnops: triangle_attention).
#   fetch.sh [TARGET]    TARGET (default ./upstream next to this script) gets TARGET/geometric_nn_ops_torch/ (+ LICENSE, pyproject.toml)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TARGET=${1:-$HERE/upstream}
REV=c58e24cd3d776f99dfb31a8558be9a36c1b7c930
BASE=https://raw.githubusercontent.com/ROCm/geometric-nn-ops/$REV
mkdir -p "$TARGET"
while read -r sum name; do
  f=$TARGET/$name; mkdir -p "$(dirname "$f")"
  [ -f "$f" ] && echo "$sum  $f" | sha256sum -c --quiet - 2>/dev/null || curl -fsSL -o "$f" "$BASE/$name"
  echo "$sum  $f" | sha256sum -c --quiet - || { echo "sha256 mismatch: $name" >&2; rm -f "$f"; exit 1; }
done <<'LIST'
cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30 LICENSE
29181c8483738ddb2c1076c731d46fa91a9ebba01fd96003f5fe3c8cc7dd8a2a geometric_nn_ops_torch/__init__.py
bd07419e272e38b0034661de8f4cf585eaec592983ac43569734b9d45a971960 geometric_nn_ops_torch/_precision.py
2e0cf0006f557b3ee76315eab1606491bd89a28e0fa67640d4fac999b08e7f2b geometric_nn_ops_torch/triangle/__init__.py
ef1dec582bdc0c2ee9d521ea3dd4f9232c669582fd50e770b73b3b4126f3ad62 geometric_nn_ops_torch/triangle/_attention_pair_bias.py
d3b3ad0b763d5df0bc6f28f06d32bd6dee7b465621510e1b25a2863d97729f02 geometric_nn_ops_torch/triangle/_kernels/__init__.py
7feec7893db5f2d61bdde52841b8cdbefe8338ab2414fc0668524f944cbc6ccb geometric_nn_ops_torch/triangle/_kernels/triangle_attention.py
77b0b20d80d0e9c77daec5ecc6c38eb324728952d825cdc0dd42b3ecdd1427c4 geometric_nn_ops_torch/triangle/_kernels/trimul.py
fc3375cf355a2b8777f89917e7b6a8e476b685471f767ad09d1851a38e8f2785 geometric_nn_ops_torch/triangle/_masking.py
63a9f9ab57075125726be7cd6cca8f1ca16f0b0ed4a36f2a8abb111020452dab geometric_nn_ops_torch/triangle/_reference/__init__.py
29018702e6dde35d4b7a7d9c75a83a4dfef1da823fce4be8c9b73b2c52b1b7f1 geometric_nn_ops_torch/triangle/_reference/attention_pair_bias.py
937cdf9a72b135f20ea99064e9a461a98bb66891a48e0a269669b1e42a2d1263 geometric_nn_ops_torch/triangle/_reference/triangle_attention.py
f25aab054fd4a234e4567fa8736ae661ade55e0338b8cdf01f69f5aea86b6448 geometric_nn_ops_torch/triangle/_reference/trimul.py
5e3d7beec8fa8a7a1c2af6dd74db349ddb266ef1fb7f1020fac7264a0c7b4f4b geometric_nn_ops_torch/triangle/_triattn.py
6ce5336a635253c692fde96958ce34c9e344b61bb2fe62214c7afbabc7a0f02a geometric_nn_ops_torch/triangle/_trimul.py
e77593f76ed6f366adfe4809bf03d9f2065eef5c4346d9b76ff48e2473488a6f pyproject.toml
LIST
echo "geometric-nn-ops @ $REV unpacked into $TARGET"
