# Third-party notices

MiniWorld Engine previously developed its own inference kernels. Anthropic's
published biomolecular modeling implementation achieved substantially better
results on the workloads we studied. Version 2 builds on that work with explicit
attribution; our contribution is its integration, measured adaptations, and
training forward/backward kernels. We do not claim the upstream inference
implementation as independent MiniWorld work, or universal superiority over it.

Upstream: https://github.com/anthropics/uplifting-biomolecular-modeling
Pinned revision: `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`.
The exact original repository URL and source hashes are also recorded in
[`licenses/anthropic-trimul-native-v5-UPSTREAM.json`](anthropic-trimul-native-v5-UPSTREAM.json).

Anthropic native v5 sources and derived TMA/WGMMA device code retain Apache-2.0
terms. See [the license](Anthropic-Apache-2.0.txt) and the preserved upstream
[`NOTICE`](Anthropic-NOTICE). This includes Transition's `cuda/anthropic_v5/`
headers. The research capsules that vendored the upstream snapshot were removed from the
tree in 2.2.0 and remain at tag `archive/experiments-20260928`. Original notices in source files remain in place. Other
MiniWorld code remains under the root MIT license.

The installed TriMul runtime also includes selected native v5/overlay sources in
`src/miniworld_engine/kernels/trimul_inproj/cuda/h100_sources/` and adapted native
Python driver/launch helpers in its parent directory. `PROVENANCE.json` maps
selected source bodies to the archived experiments (tag `archive/experiments-20260928`) and their hashes.
The corresponding Apache-2.0 license applies to these inherited portions.

## Compiled kernels in binary wheels

A binary (platform) wheel of miniworld-engine carries prebuilt kernels under `miniworld_engine/_prebuilt/<arch>/`
(docs/kernels/prebuilt-kernels.md): torch extensions and cubins compiled from this repository's CUDA sources. Besides
the MIT and Anthropic Apache-2.0 code above, they contain object code compiled from:

- **NVIDIA CUTLASS / CuTe 4.2** headers (BSD-3-Clause, Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES):
  the H100 TriangleAttention kernels that include `cute/` / `cutlass/` (built with `CUTLASS_PATH`). License:
  [`CUTLASS-LICENSE.txt`](CUTLASS-LICENSE.txt). The headers themselves are not shipped.
- **FlashAttention-3** helpers (`hopper/utils.h`, `hopper/softmax.h`; BSD-3-Clause, Copyright (c) 2024 Tri Dao),
  adapted in `kernels/triangle_attention/cuda/fa3_utils.h` and `h100_sources/fa3_utils.h`. License:
  [`FlashAttention-LICENSE.txt`](FlashAttention-LICENSE.txt).
- **NVIDIA CUDA Toolkit** headers (`cuda_runtime.h`, `cuda_bf16.h`, cooperative groups, ...) and the device code nvcc
  generates from them, which the CUDA Toolkit EULA allows to distribute as part of an application. The extensions link
  the CUDA runtime, cuBLAS / cuBLASLt and libtorch dynamically; the wheel carries none of those libraries (they come
  with torch).

No shipped artifact compiles NVIDIA MathDx / cuBLASDx: the only kernels that include it (the mathdx-based
`transition/cuda` extensions) are not reached by any production path and are never bundled; they JIT-build if used.
