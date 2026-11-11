# Third-party notices

MiniWorld Engine is distributed as the `miniworld-engine` source distribution and pure-Python wheel and, for H100, the
`miniworld-engine-kernels-sm90-cu129` wheel of prebuilt kernels. All of them carry this file, the root `LICENSE` and the
licence texts in `licenses/`. The package licence expression is `MIT AND Apache-2.0 AND BSD-3-Clause`:

- **MIT** (root `LICENSE`, Copyright (c) 2026 Sanggeun Park): MiniWorld Engine's own code, unless a file says otherwise.
- **Apache-2.0**: the Anthropic-derived code below. A source file whose header carries `SPDX-License-Identifier: Apache-2.0`
  is distributed under the Apache License, Version 2.0 ([text](Anthropic-Apache-2.0.txt)); a file's own licence line takes
  precedence over the root MIT default.
- **BSD-3-Clause**: the FlashAttention-3 portions and the object code instantiated from NVIDIA CUTLASS / CuTe below.

## Anthropic: uplifting-biomolecular-modeling (Apache-2.0)

MiniWorld Engine previously developed its own inference kernels. Anthropic's
published biomolecular modeling implementation achieved substantially better
results on the workloads we studied. Version 2 builds on that work with explicit
attribution; our contribution is its integration, measured adaptations, and
training forward/backward kernels. We do not claim the upstream inference
implementation as independent MiniWorld work, or universal superiority over it.

Upstream: https://github.com/anthropics/uplifting-biomolecular-modeling, revision
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e` (Copyright 2026 Anthropic, PBC). Licence: [Apache-2.0](Anthropic-Apache-2.0.txt);
the upstream `NOTICE` file is carried verbatim as [`Anthropic-NOTICE`](Anthropic-NOTICE). The source hashes of the TriMul
native v5 files taken from it are in [`anthropic-trimul-native-v5-UPSTREAM.json`](anthropic-trimul-native-v5-UPSTREAM.json).
The upstream release itself is not redistributed: `third_party/anthropic/upstream/` (used only by the `anthropic` baseline
backends) is a local clone that is git-ignored and is in no sdist or wheel; `third_party/anthropic/` in the repository holds
only its pin (`UPSTREAM.json`), a source inventory and local build records.

What MiniWorld Engine carries from it (all paths under `src/miniworld_engine/kernels/`; upstream paths under
`common/opt_core/opt_core/kernels/`):

| MiniWorld file(s) | Upstream file(s) | Changes |
|---|---|---|
| `transition/cuda/anthropic_v5/tmn_kernels.cuh`, `tmn_ptx.cuh`, `common/tmn_math.cuh` | `trimul/native/pkg/v5/csrc/` (same names) | none (byte-identical) |
| `trimul_inproj/cuda/h100_sources/upstream/csrc/tmn_kernels.cuh` | `trimul/native/pkg/v5/csrc/tmn_kernels.cuh` | modified: K1 token-mask mode, batched-sample row count (marked in the file) |
| `trimul_inproj/cuda/h100_sources/inference/tmn_kernels.cuh`, `tmn_k3_wide.cuh`, `tmn_sm90_unit.cuh`, `common/tmn_math.cuh` | `trimul/native/pkg/v5/csrc/` (same names) | modified: batched samples, schedules and instantiation lists of the packaged H100 inference (header line in each file) |
| `trimul_inproj/cuda/_h100_launch.py`, `_h100_driver.py`, `_h100_manifest.py`, `_h100_infer_kernel.py` | `trimul/native/pkg/v5/python/trimul_native/launch.py`, `_driver.py`, `manifest.py`, `kernel.py` | modified: adapted to the engine's build cache, prebuilt bundle and launch paths (header line in each file) |
| `triangle_attention/cuda/fa3_utils.h` | `triattn/triattn_native/pkg/v11/triattn_pkg/cuda/csrc/fa3_utils.h` | none (byte-identical; itself trimmed FlashAttention-3 source, see below) |

Further MiniWorld sources build on these device primitives (they include `tmn_ptx.cuh` / `tmn_kernels.cuh` / `tmn_math.cuh`,
or restate parts of them) and carry the `SPDX-License-Identifier: Apache-2.0` line: the H100 TriMul kernels under
`trimul_inproj/cuda/h100_sources/`, the sm_90a Transition kernels under `transition/cuda/`, and other kernel families that
include the `anthropic_v5` headers. The Anthropic v5 research capsules that once vendored the upstream snapshot were removed
from the tree in 2.2.0 and remain at tag `archive/experiments-20260928`. Original notices in source files remain in place.

## Code from team-gm (MIT, root licence)

Several Triton kernels and the SWA atom-transformer CUDA/Triton path were moved from the team-gm repository
(github.com/CSSB-SNU/team-gm, not public); each such file names its team-gm source path and commit in its header
(`vendored from team-gm ...` / `Moved from team-gm ...`). They are distributed under the root MIT licence.

## Compiled kernels in the `miniworld-engine-kernels-*` wheels

The kernels wheel (`miniworld_engine/_prebuilt/`; docs/kernels.md) carries torch extensions and cubins
compiled by nvcc from this repository's CUDA sources. No library is bundled. Besides the MIT and Anthropic Apache-2.0 code
above, the binaries contain object code compiled from:

- **NVIDIA CUTLASS / CuTe 4.2** headers (BSD-3-Clause, Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES): the H100
  triangle-attention front kernel (`triangle_attention/cuda/h100_sources/front_fused3_sm90.cu`, via `fa3_utils.h`; prebuilt
  `triattn_h100_front3_sm90_v1`) includes `cute/` / `cutlass/` (built with `CUTLASS_PATH`); the A100 outer-product-mean GEMM
  (`outer_product_mean/cuda/sm80/ogemm_cutlass.cu`) does as well when built. Licence: [`CUTLASS-LICENSE.txt`](CUTLASS-LICENSE.txt).
  The CUTLASS headers themselves are not shipped. CUTLASS pulls in NVIDIA CCCL (libcu++) headers from the CUDA toolkit
  (Apache-2.0 WITH LLVM-exception, whose exception covers portions embedded in object code).
- **FlashAttention-3** helpers (`hopper/utils.h`, `hopper/softmax.h`; BSD-3-Clause, Copyright (c) 2022, the respective
  contributors, as shown by the AUTHORS file; the carried portions Copyright (c) 2024, Tri Dao), trimmed in
  `triangle_attention/cuda/fa3_utils.h` (taken from the Anthropic release, the BSD-3-Clause notice kept at the top of the
  file) and compiled into `triattn_h100_front3_sm90_v1`. Licence: [`FlashAttention-LICENSE.txt`](FlashAttention-LICENSE.txt).
- **NVIDIA CUDA Toolkit** headers (`cuda.h`, `cuda_runtime.h`, `cuda_bf16.h`, `cuda_fp16.h`, cooperative groups,
  `cuda_pipeline.h`, `curand_kernel.h`, ...) and the device code nvcc generates from them, which the CUDA Toolkit EULA
  (https://docs.nvidia.com/cuda/eula/) allows to be distributed as part of an application. The extensions link the CUDA
  runtime, cuBLAS / cuBLASLt and libtorch dynamically (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`,
  `libtorch*.so`, `libc10*.so`); the wheel carries none of those libraries (they come with torch).
- **PyTorch** C++ extension headers (BSD-3-Clause, https://github.com/pytorch/pytorch/blob/main/LICENSE): every torch
  extension is compiled against them; libtorch is not bundled.

No shipped artifact compiles NVIDIA MathDx / cuBLASDx: no source in the tree includes it. The build helpers that locate a
MathDx install (`kernels/_nvcc.py`) remain; a binary built with it would fall under NVIDIA's MathDx SDK licence agreement.

## Runtime dependencies and optional extras (installed, not redistributed)

These are installed by pip from their own distributions under their own terms; no file of them is copied into this
package. Required: torch (BSD-3-Clause and others), triton (MIT), einops (MIT), jaxtyping (MIT), numpy (BSD-3-Clause and
others), scipy (BSD-3-Clause), pydantic (MIT), pyyaml (MIT), and `cuda-bindings` (NVIDIA Software License, not an
open-source licence; used to call the CUDA driver API). Optional extras: `baselines` installs `cuequivariance-torch`
(Apache-2.0), which needs NVIDIA's `cuequivariance-ops-torch-*` binary packages (proprietary: NVIDIA's licence agreement for its math-library SDKs) and, in a
repository checkout, a fork of Hugging Face transformers (Apache-2.0) for the `anthropic` baseline; `flash` installs
`flash-attn-4` (BSD-3-Clause). The `anthropic` and `cueq` backends only connect to those external packages / the local
Anthropic clone.

## Model weights and data

No model weights, checkpoints, structures (PDB / mmCIF), alignments (MSA / A3M) or other datasets are distributed in the
repository, the sdist or the wheels. Benchmark tables and plots under `benchmarks/` (repository only) are MiniWorld
measurements.
