# miniworld-engine 3.0.4

Base: 3.0.3 (`b63ba5a9c`, tag `v3.0.3` -> `edc8deca4`). Integration branch `rc/3.0.4`, released from `rc/3.0.4-release` (= `rc/3.0.4` + the version bump).

**What this release is.** A release for the sm_80-family cards (A100 and RTX 5090) plus two infrastructure fixes that affect every card. The H100 numerics are untouched:
no H100 `backend_versions` entry and no module `version` changed, and the installed H100 inference / training default paths are bitwise identical to the installed 3.0.3 (see
"Verification"). The headline is the RTX 5090 **real Protenix v1 training step on the cuda backend**, which used to fall back to Triton / PyTorch or crash.

## Fixes that apply to every card

1. **nvcc of torch's CUDA major (the 5090 segfault).** A JIT build used the first nvcc whose release matched torch's CUDA, else *any* nvcc. On the RTX 5090 server
   (torch 2.13+cu129, `/usr/local/cuda` = 13.1) that built extensions with CUDA 13.1; they linked `libcudart.so.13` / `libcublasLt.so.13` next to torch's `.so.12`, and a cuBLASLt handle made by one library
   crashed the other (a call through a NULL function pointer in `cublasLtMatmulAlgoGetHeuristic`, TriangleAttention `out_gemm_residual`: segmentation fault in the real training step).
   `_nvcc.pick_nvcc` now prefers the same release, then the **same major** (closest minor), and only then any other nvcc, with a warning; a locally built artifact records its `toolkit` and is rebuilt
   when its major is not torch's. Tests: `tests/cpu/test_nvcc_pick_same_major.py`. The shipped kernels wheels are unaffected by the bug (they link CUDA 12 only; checked, see "Prebuild").
2. **SM count by `cudaDeviceGetAttribute` (`kernels/_sm_count.cuh`).** `at::cuda::getCurrentDeviceProperties()->multiProcessorCount` read 1 in extensions compiled by a toolkit whose `cudaDeviceProp`
   layout differs from the CUDA libtorch is built with (nvcc 13.1 against torch cu129). 65 sm_80 / sm_120a sources (and the H100 sources that include the header) now use `mw::sm_count()` /
   `max_threads_per_sm()`. Values on the H100 are unchanged; the include re-keys 40 H100 artifacts (same outputs). A follow-up fix includes the header outside the `MINIWORLD_OPM_FP8` block (the non-FP8
   native OPM builds failed with `mw::` undeclared at two launch sites).

## A100 and RTX 5090: real fp32 training step and memory

- **IEEE fp32 CUDA chain (`kernels/ieee_f32`).** The real fp32 Protenix v1 training step (TF32 off) used to be refused by the sm_80 / sm_120a kernels or forced TF32. The engine now has its own IEEE fp32 GEMM
  (one tile template 128x128 by 256 threads, 64x128 / 64x64 by 128; a cost-model chooser of tile and split-K depth fitted to timings on the 170-SM card), AdaLN of the augmented-attention block on `adaln_sm80`,
  `ieee_f32.linear` for the atom DiT modulation tables, fp32 `PoolAtoms` (`sampler_ops_f32_sm80.cu`, forward and backward). The real step GEMM group went 714 -> 687 ms (57.7 -> 60.0 TFLOP/s) in the last tuning round.
  TF32 and bf16 paths are unchanged.
- **Training holes closed on the sm_80 family.** `TriangleAttention.forward_train` passed no `key_mask` to the sm_80 op (every cuda training call raised); opaque launchers run with CUDA autocast off on non-H100 cards
  (a backward replayed under checkpoint autocast rounded the fp32 chain to bf16); no `record_stream` while a CUDA graph is captured on these cards (the capture pool kept every side-stream temporary alive:
  a 48-block step graph ran out of memory on 32 GB; pool 15.3 GiB at chunk 8). The H100 path keeps the shipped behaviour (a test of 144 H100 cases failed with the unconditional skip).
- **5090 training memory.** TriangleAttention `xh` and Transition `xn` are rebuilt in the backward instead of saved (-113 MiB per Pairformer block at L 384; gradients bitwise unchanged); TriMul training saves `p'`
  only (`g'` is recomputed: -36 MiB per TriMul at L 384, max 26 -> 28 Pairformer blocks).
- **5090 launch-config keys** for the shapes the real step still missed (sampler conditioning Transition, 8192-row no-grad forward, augment-chunk train rows): previously a missing key fell to the kernel default
  (131072 B shared memory > 101376 B, `OutOfResources`). Pretuned in situ; the table commits only add keys for shapes absent from the requirements rows, so the `triton:5090_sm120` versions stay at the default.
- **5090 TriMul inference.** D64 hidden-128 small L (k1w64 front up to L 256, k3w ZPRE output stage at L 128: module -10..-19 % at L 128, -6 % at L 256), D64 outgoing at L 256 from 0.965 / 1.00 to 0.89 / 0.92
  of Anthropic, and two CTAs an SM in the D64 / D128 output stages with register-resident `k1t` statistics (the rows 87 / 88 that were below Anthropic). On the 5090 the largest TriMul
  cuda / anthropic time ratio is 0.980.

### Measured: real Protenix v1 training step, RTX 5090 (cssb-rtx, torch 2.13+cu129)

48 trunk Pairformer blocks, 8tlg (384 tokens), augment chunk 8, activation checkpointing, step CUDA graph, effective batch 1.

| | engine cuda (installed 3.0.4 wheels, empty JIT cache) | PyTorch |
|---|---|---|
| step time | **2.0 s** | 5.0 s |
| loss / gnorm at step 0 | 0.3764 / 1.242 | 0.3768 / 1.261 |
| peak memory | 16.93 GiB allocated, 25.34 GiB reserved | not measured here |
| engine modules on Triton | 0 (BackendModule calls: cuda 9021, pytorch 774 = the `PairformerBlock` wrapper, triton 0) | n/a |

The loss and gradient norm agree with PyTorch at the fp32 / bf16 mixed-precision tolerance of the engine standard (step-0 loss 0.1 % apart, gnorm 1.5 % apart). Before the nvcc-major fix this run segfaulted;
before the other fixes it ran on Triton at 2.97 s / step. The only Triton launches in the run (`_gather_grad_kernel`, 117) are FoldForge's own, not an engine module's.
(The 5090 PyTorch time and the PyTorch loss / gnorm are the figures given with the release request; the engine numbers were re-measured on the installed wheels for this release.)

## Versions (`backend_versions`, human-bumped)

No module `version` was bumped. No H100 entry changed.

| class | card | 3.0.3 | 3.0.4 | cause |
|---|---|---|---|---|
| AtomDiT* (Protenix, Boltz2, Chai1, OpenFold3, RF3), `_LocalAtomDiT` | a100_sm80 | 4 | 5 | fp32 TF32-off is IEEE (kernels/ieee_f32) |
| AtomDiT* (same 6 classes) | 5090_sm120 | default (1) | 2 | the same |
| TokenDiTProtenix, Boltz2, Chai1, RF3 | a100_sm80 | 4 | 5 | AdaLN / SwiGLU GEMMs follow the caller's TF32 setting |
| TokenDiTMiniWorld | a100_sm80 | 5 | 6 | inherits |
| TokenDiT* (same classes + MiniWorld) | 5090_sm120 | default (1) | 2 | TF32 no longer forced |
| AttentionPairBias | a100_sm80 / 5090_sm120 | 8 / 3 | 9 / 4 | fp32 GEMMs of the fused update follow TF32 setting (IEEE fp32) |
| Transition | a100_sm80 / 5090_sm120 | 6 / 4 | 7 / 6 | IEEE fp32 chain (a100: was refused to Triton / PyTorch); 5090 also does not save `xn` |
| TriangleAttention / TriangleAttentionAF3 | a100_sm80 | 7 / 6 | 8 / 7 | `forward_train` key_mask fix (every cuda training call raised before) |
| PairformerBlock | a100_sm80 | 9 | 10 | its TriangleAttention training |
| PoolAtoms | a100_sm80 / 5090_sm120 | 2 / 1 | 3 / 2 | fp32 atom features (`sampler_ops_f32_sm80.cu`) |
| TriangleMultiplication, TriangleMultiplicationRF3, BidirectionalTriangleMultiplication | 5090_sm120 | 2 | 5 | D64 / D128 inference fronts, `p'`-only training |

## Honest caveats

- **A100 and 5090 judge tokens are owed.** The judge, run on this tree, owes on `a100_sm80` **2140 accuracy + 2140 speed** cells and on `5090_sm120` **2107 + 2107** (96 of each card are the old
  `PairformerBlock` chai1 rows without a token since 3.0.1). They are not re-measured for this release; the A100 is waiting for its queue. `h100_sm90` owes **0 / 0**. What was measured instead is stated in
  "Verification" (installed-wheel accuracy runs, the real 5090 step).
- **FoldForge's policy limits `cuda` to the H100.** Unless the caller sets the backend explicitly (or, as in the benchmark above, patches the card the policy treats as cuda-capable), FoldForge on an A100 or
  a 5090 still picks Triton. The 5090 step above used `PF_CC=12,0`; this release does not change FoldForge, and FoldForge was not notified.
- The 5090 real-model numbers are one model (Protenix v1, 48 blocks, 384 tokens), one seed, 4 steps.
- Nothing else about the H100 changed: no new FP8 route, no re-measure.

## Prebuild regeneration (release build)

Kernels wheels (sm90 / sm80 / sm120, cu129, cp312) rebuilt incrementally from the 3.0.3 release cache with GCC 12.4 and the pixi CUDA 12.9 toolkit (torch 2.13+cu129; CUTLASS 4.2, MathDx), GPU-less on the cluster
(`MINIWORLD_ENGINE_TARGET_ARCH` set): sm90a **40 built** (579 artifacts, 0 absent; the 40 are the artifacts that include `_sm_count.cuh`, re-keyed, outputs unchanged), sm80 **29 built** (74 artifacts),
sm120a **29 built** (66 artifacts). The carried names (`triattn_h100_core_h16_v2`, `miniworld_pwa_dgv_bwd`, `_glue3`, `_pair3`, byte-for-byte from the 2.5.5 / 3.0.0 wheels) are carried again (8 files).
**No name of 3.0.3 is dropped:** manifest names sm90 364 -> 364 (0 dropped / 0 added), sm80 72 -> 74 (0 / 2: `ieee_f32_sm80`, `sampler_ops_f32_sm80`), sm120 64 -> 66 (0 / 2: `ieee_f32_sm120`, `sampler_ops_f32_sm80_sm120`).
**`ldd` on every `.so` of every arch (staged tree and installed wheels, 377 distinct files): 0 link a CUDA-13 library, 0 unresolved**; the only CUDA libraries are `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`.
Highest symbol versions needed: GLIBC 2.14, GLIBCXX 3.4.21, CXXABI 1.3.11 (manylinux_2_28 safe).

## Verification (rel/3.0.4)

Installed-wheel runs use the five release artifacts extracted without the repo on the path (`PYTHONPATH` = the extracted wheels, no `python -I`), an empty JIT cache, no CUTLASS / MathDx and no `nvcc` on `PATH`.

| item | result |
|---|---|
| sha256 of the five artifacts | `dist/SHA256SUMS`, checked before every run |
| H100 installed-wheel load of every indexed sm90 artifact (579 = 163 + 190 + 226) | 579 present, 579 resolved, **0 compiled**; also with the system libstdc++ preloaded (RHEL 8) |
| H100 installed accuracy (12 modules, pytorch / cuda / triton, minimum lengths) | RED 0, 0 builds |
| H100 installed `test_restored_triton_backends.py` | 63 passed |
| H100 default routes, installed 3.0.3 vs installed 3.0.4 (separate processes) | inference outputs (29 cases) and the default fp16 h16 TriAttn core (6) bitwise identical; default TRAINING paths: token DiT (Protenix x4, MiniWorld), OPM x2, PWA, Transition D128 bitwise identical; Boltz-2 / RF3 token DiT, TriMul D128 / D256 and TriAttn differ between two runs of 3.0.4 itself, exactly as in 3.0.3 (nondeterministic backward) |
| CPU gate (`tools/check.sh cpu`) | ruff check clean, pyright 0 errors, tests/cpu 3414 passed, 2 skipped (`ruff format --check` lists files that are not mass-formatted: not enforced) |
| RTX 5090 installed-wheel load of every indexed sm120 artifact (66) | 66 present, 66 resolved, **0 compiled** |
| RTX 5090 installed accuracy (12 modules, pytorch / cuda / triton) | RED 0, 0 builds |
| RTX 5090 installed `test_restored_triton_backends.py` | 59 passed, 4 skipped (the H100 / A100 bias-only path) |
| RTX 5090 real Protenix v1 step from the installed wheels | see the table above; 0 engine kernels built into the empty JIT cache, no segfault |
| A100 installed wheels | **not run on an A100**: the A100 partition of cssb3 is full (the job is queued, estimated start 2026-11-22). Stand-in: GPU-less, installed wheels, empty cache, `MINIWORLD_ENGINE_TARGET_ARCH=sm80`: all 74 indexed sm80 artifacts present and resolved by the runtime loaders, 0 compiled. No A100 accuracy, restored-Triton or real-step run exists for 3.0.4 yet |
| judge owed (versions only) | `h100_sm90` 0 / 0; `a100_sm80` 2140 / 2140 and `5090_sm120` 2107 / 2107 owed (not re-measured, see caveats) |
