#!/bin/bash
# Build the release's triangle-attention CUDA extensions for THIS stack (the engine's torch / CPython) with the release's OWN build tools and
# install them into the release root, so the "anthropic" backends serve the attention core the model kits run in-model (kernels.triattn rows
# triattn_native and cuda_sm90a, word fast) instead of the k2b Triton step-aside the provider names when no prebuilt exists for the stack.
#
#   srun -A cssb -p h100 -q cssb_h100 --gres=gpu:h100:1 -c 16 --mem=96G -t 00:10:00 third_party/anthropic/build_local.sh [ROOT]
#
# ROOT: the release root (default $MINIWORLD_ANTHROPIC_ROOT, else third_party/anthropic/upstream). Needs an H100 (the build tools byte-gate
# the binaries on the device), nvcc + ninja of the engine env on PATH, CUTLASS headers at $CUTLASS_PATH (>= 4). No source is edited: the
# tools compile the release's own csrc with the release's own flags; their load checks compare the outputs against the release's recorded
# digests (bitwise; the 2026-11-13 build reproduced every digest of the cu128 / cu130 builds). Each binary is added to the SHA256SUMS the
# release verifies it against (opt_core.gates.binary_refusal), and LOCAL_BUILD.json beside this script records the build.
#
# Not built here, by the release's rule: trimul_native / esm_v61 / the ef2_t16 transition cubin are SEALED CUDA-13.0 cubins (a rebuild is a
# refusal by digest); they load only under a CUDA-13 driver, so on a 12.x driver the backends exclude those rows by name
# (modules/_anthropic.cuda13_cubins_load) and the kit-row timings come from the kit stack (third_party/anthropic/kit_stack.sh).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${1:-${MINIWORLD_ANTHROPIC_ROOT:-$HERE/upstream}}
T=$ROOT/common/opt_core/opt_core/kernels/triattn
OUT=$(mktemp -d "${TMPDIR:-/tmp}/anthropic_build.XXXXXX")
NOTE="local build for the engine stack from the release sources (third_party/anthropic/build_local.sh)"
export MAX_JOBS=${MAX_JOBS:-12} TORCH_EXTENSIONS_DIR=$OUT/torch_ext
for ext in triattn_sm90_ext triattn_m1_ext triattn_mw_ext_g3x4; do
  python "$T/triattn_native/build_prebuilt.py" --out "$OUT/native" --exts "$ext" --fresh-check 0 --note "$NOTE"
done
python "$T/cuda_sm90a/build_prebuilt.py" --out "$OUT/cuda_sm90a" --note "$NOTE" || true   # its fresh-process check needs opt_core on sys.path; the build and load check above it ran
for d in "$OUT"/native/*/; do
  key=$(basename "$d")
  dst=$T/triattn_native/pkg/v11/triattn_pkg/prebuilt/$key
  mkdir -p "$dst"
  cp "$d"/*.so "$d"/*.json "$dst"/
  for so in "$d"/*.so; do
    rel=pkg/v11/triattn_pkg/prebuilt/$key/$(basename "$so")
    grep -q " $rel\$" "$T/triattn_native/SHA256SUMS" || (cd "$T/triattn_native" && sha256sum "$rel" >> SHA256SUMS)
  done
done
for d in "$OUT"/cuda_sm90a/*/; do
  key=$(basename "$d")
  [ -f "$d/triattn_cuda_sm90.so" ] || { echo "cuda_sm90a: no binary built for $key" >&2; exit 1; }
  rm -rf "$T/cuda_sm90a/prebuilt/$key"
  cp -r "$d" "$T/cuda_sm90a/prebuilt/$key"
  (cd "$T/cuda_sm90a/prebuilt/$key" && sha256sum triattn_cuda_sm90.so > SHA256SUMS)
done
echo "installed into $T (build dir $OUT)"
