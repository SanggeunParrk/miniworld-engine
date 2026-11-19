# miniworld-engine 3.0.4

Base: 3.0.3 (`b63ba5a9c`, tag `v3.0.3` -> `edc8deca4`). Source: `rc/3.0.4-final` (+ the one-line sm120 prebuild-plan fix), released from `rc/3.0.4-release2`.

**What this release is.**

1. **sm_80-family (A100 and RTX 5090) speed and correctness fixes**, headlined by the RTX 5090 **real Protenix v1 training step on the cuda backend** (it used to fall back to Triton / PyTorch or crash).
2. **The user rule "bias-only token DiT pair-bias logits and softmax in fp32, in training and inference, on every backend and card"** (`TokenDiTMiniWorldBiasOnly`: PyTorch reference, Triton, H100 / A100 / 5090 cuda). bf16 logits (values 1e3..2e5, key spread ~500, bf16 resolution 32..1024) change which tokens attend; the peptide C-N bond grew and FoldBench lDDT halved. This changes `TokenDiTMiniWorldBiasOnly` outputs on all cards (see "Versions"); every other H100 module is bitwise unchanged (see "Verification").
3. Fixes that apply to every card (nvcc major, SM count).

## Fixes that apply to every card

1. **nvcc of torch's CUDA major (the 5090 segfault; JIT picked CUDA 13.1 against torch cu129).** A JIT build used the first nvcc whose release matched torch's CUDA, else *any* nvcc. On the RTX 5090 server (torch 2.13+cu129, `/usr/local/cuda` = 13.1) extensions were built with CUDA 13.1, linked `libcudart.so.13` / `libcublasLt.so.13` next to torch's `.so.12`, and a cuBLASLt handle made by one library crashed the other (segmentation fault in the real training step; `multiProcessorCount` read 1). `_nvcc.pick_nvcc` now prefers the same release, then the **same major** (closest minor), and only then any other nvcc with a warning; a locally built artifact records its `toolkit` and is rebuilt when its major is not torch's. Test: `tests/cpu/test_nvcc_pick_same_major.py`. The shipped kernels wheels link CUDA 12 only (checked: see "Prebuild").
2. **SM count by `cudaDeviceGetAttribute` (`kernels/_sm_count.cuh`).** 65 sm_80 / sm_120a sources (and H100 sources that include the header) use `mw::sm_count()` / `max_threads_per_sm()` instead of `getCurrentDeviceProperties()->multiProcessorCount`. Values on the H100 are unchanged; outputs of the H100 modules that include the header are bitwise unchanged (see "Verification").
3. **sm120 prebuild plan completeness.** The sm120 plan lacked the extension `swa_wgrad5_sm120` (the 5090 atom-training weight-gradient GEMM): an installed sm120 wheel with an empty JIT cache failed 7 InputFeatureEmbedder / SWAAtomTransformer train cells with "Ninja is required". Found by the installed-wheel verification, fixed in the plan, artifact now shipped.

## Bias-only token DiT: fp32 pair-bias logits (user rule)

`to_bias(ln_pair(pair))` and its softmax are fp32 in training and inference on every backend and card; TF32 is not allowed for the logits' products. New kernels in the H100 hoist extension (`ln0_fwd_rows`, `logits_fused`, `dln_bwd`, `dwf`, `dbias_rows`), fp32 paths on sm_80 / sm_120a, a Triton pair-bias table for the 5090. Tests: `tests/gpu/test_bias_only_fp32_logits.py` (H100) and `test_bias_only_fp32_logits_sm80.py` (A100, 5090) against fp64 on huge-logit pairs. Kernel-level check (huge-logit repro): max-row total variation of P vs fp64 about 1.2e-3 for every backend (the PyTorch fp32 reference itself 1.25e-3); the previous bf16 logits gave 0.75. Gradients (pair, `to_bias.weight`, `ln_pair.weight`) within 5e-2 relative in the tests.
On the registry accuracy cases the module stays at the same error vs fp64 as 3.0.3 (e.g. fp32 L128 rel-rms 2.66e-4 in both; master 3.08e-3 in both), since the registry pairs are not huge-logit.

Note for the owner of the B200 path: the fp32 pair-bias path files are shared with the B200 sources; B200 was **not** validated in this release.

## A100 and RTX 5090: speed, memory, training

- **IEEE fp32 CUDA chain (`kernels/ieee_f32`)** (fp32 with TF32 off on cuda: Transition fp32, PoolAtoms fp32, AdaLN / SwiGLU GEMMs, atom DiT modulation tables). Previously refused to Triton / PyTorch or TF32 was forced. TF32 and bf16 paths unchanged.
- **Training holes closed on the sm_80 family.** `TriangleAttention.forward_train` passed no `key_mask` to the sm_80 op: every cuda training call raised (47 error tokens of 3.0.0..3.0.3 are gone); opaque launchers run with CUDA autocast off on non-H100 cards; no `record_stream` while a CUDA graph is captured on these cards; the `torch.compile` guard fix.
- **5090 training memory.** TriangleAttention `xh` and Transition `xn` rebuilt in the backward; TriMul training saves `p'` only. Measured (Protenix trunk Pairformer blocks, L 384, no checkpointing, step CUDA graph, one RTX 5090): max Pairformer blocks **31 (default allocator; 32 with `expandable_segments`)** (3.0.3: 26).
- **5090 TriMul inference:** 120 cells all < 1.00 vs Anthropic (max 0.983). **5090 Transition** bf16 eval: 135 of 137 cells <= 1.00 vs Triton. **Atom-level training** (window 128 + global): 29 cells <= 1.00 vs Triton. TokenPairInit, TriAttn family (cueq / Triton), RF3, OPM fixes.
- **A100:** `ParallelResidual` plain loads / stores (21 cells 1.06-1.09 -> 1.000 vs Triton); OPM Chai groups = 8 per-group table plus an all-live probe (60 unpadded cells, max 1.025 vs Triton); `key_mask` training fix; `TokenDiTChai1` identity AdaLN on cuda (sm_80 family); 5090 launch-config keys for shapes the real step missed; 5090 Triton bias-only pair-bias table.

### Measured: real Protenix v1 training step, RTX 5090 (cssb-rtx, torch 2.13+cu129)

48 trunk Pairformer blocks, 8tlg (384 tokens), augment chunk 8, activation checkpointing, step CUDA graph, effective batch 1, installed wheels, empty JIT cache.

| | engine cuda | PyTorch |
|---|---|---|
| step time | **2.0 s** | 5.0 s |
| loss / gnorm at step 0 | 0.3765 / 1.242 (this run: 0.3764) | 0.3768 / 1.261 |
| engine modules on Triton | 0 (cuda 9021 calls, pytorch 774 = the `PairformerBlock` wrapper, triton 0) | n/a |

Only Triton launches in the run (`_gather_grad_kernel`) are FoldForge's own. One model, one seed, 4 steps. (PyTorch figures are the ones given with the release request.)

## Versions (`backend_versions` / module `version`, human-bumped)

`TokenDiTMiniWorldBiasOnly` module `version` 2 -> 3 (the fp32 logits rule). All other module versions untouched.

| class | card | 3.0.3 | 3.0.4 | cause |
|---|---|---|---|---|
| TokenDiTMiniWorldBiasOnly | h100_sm90 / a100_sm80 / 5090_sm120 | 4 / 4 / 5 | 6 / 5 / 6 | fp32 pair-bias logits |
| TokenDiTMiniWorldBiasOnly | triton:5090_sm120 | default | 2 | fp32 logits, 5090 Triton table |
| AtomDiT* (Protenix, Boltz2, Chai1, OpenFold3, RF3), `_LocalAtomDiT` | a100_sm80 | 4 | 5 | IEEE fp32 chain |
| AtomDiT* (same), TokenDiTProtenix / Boltz2 / RF3 / MiniWorld | 5090_sm120 | default | 2 | IEEE fp32 / TF32 no longer forced |
| AtomDiTESMFold2, AtomDiTMiniWorld, SWAAtomTransformer | 5090_sm120 | 2 | 6 | window training in L2-sized chunks, streaming backward |
| InputFeatureEmbedder | 5090_sm120 | 2 | 5 | same family |
| TokenDiTProtenix, Boltz2, Chai1, RF3 | a100_sm80 | 4 | 5 | AdaLN / SwiGLU GEMMs follow the caller's TF32 setting |
| TokenDiTMiniWorld | a100_sm80 | 5 | 6 | inherits |
| TokenDiTChai1 | a100_sm80 / 5090_sm120 | 4 / default | 7 / 4 | identity AdaLN and transition on the sm_80 row kernels |
| AttentionPairBias | a100_sm80 / 5090_sm120 | 8 / 3 | 9 / 4 | IEEE fp32 GEMMs of the fused update |
| OuterProductMean | a100_sm80 / 5090_sm120 | 3 / 3 | 5 / 4 | Chai-1 per-group products, all-live probe, cuBLAS at deep MSA |
| ParallelResidual | a100_sm80 | default | 2 | plain loads / stores |
| Transition | a100_sm80 / 5090_sm120 | 6 / 4 | 7 / 8 | IEEE fp32 chain; 5090: LN folded into the dual GEMM at D >= 256, `xn` rebuild |
| TriangleAttention (+AF3) | a100_sm80 | 7 (AF3 6) | 8 (AF3 7) | `forward_train` key_mask fix |
| TriangleAttention family, TriangleAttentionRF3 / Chai1 / Chai1Confidence / Fp32Softmax | 5090_sm120 | 4 | 5 | D256 rows engine without constant columns |
| PairformerBlock | a100_sm80 | 9 | 10 | its TriangleAttention training |
| PoolAtoms | a100_sm80 / 5090_sm120 | 2 / 1 | 3 / 2 | fp32 atom features |
| TriangleMultiplication, RF3, Bidirectional | 5090_sm120 | 2 | 5 | D64 / D128 fronts, `p'`-only training |
| TokenPairInit | 5090_sm120 | 1 | 3 | 5090 fixes |
| LayerNorm | 5090_sm120 | 3 | 3 | unchanged (table keys only) |

No other H100 entry changed.

## Judge

`judge owed` is 0 on `h100_sm90`, `a100_sm80` and `5090_sm120`, re-measured on the final source (carried 17815 per kind). The tokens are not packaged in the wheels (pyproject excludes `judge/tokens`).

## KNOWN LIMITS

(a) **The exact fp32 logits rule costs speed.** H100 bias-only token DiT cuda: eval up to 2x (master eval d_pair 256 S5: +62 % median, +114 % at L768: 0.425 -> 0.906 ms); train +4..+55 % by cell (bf16 non-hoist L >= 512 +12..14 %, fp32 non-hoist +4..11 % in the module-level measurements of the bias/speed branch). A100 train up to +6 %, 5090 up to +2 %. A speed recovery of the H100 eval hoist exists as a branch (`bias/eval`, unverified at the time of this draft; planned next release).
(b) 5090 Transition train residual cells 1.00-1.02 and OPM Chai train L128 / L256 1.0075-1.018 vs Triton; A100 OPM non-Chai residual cells <= 1.025; LayerNorm exceptions.
(c) **FoldForge's policy `choose_backend` allows cuda only on the H100**: FoldForge on an A100 / 5090 selects Triton unless overridden (the 5090 benchmark above used `PF_CC=12,0`). FoldForge was not changed or notified.
(d) **The peptide-bond geometry (C-N length) end-to-end check of the fp32 logits fix was NOT run in this release**: kernel-level checks only (see above). The check list (FoldBench targets 7xi5 7yhr 8t93 8pak) is in the MiniWorld session: `~/MiniWorld/scripts/analysis/fp32_inference/` (`fused_logit.sbatch`, `fusedlogit_geometry.sbatch`, `x0hat_bonds` / `cif_geometry.py`).
(e) A100 judge cells were re-measured on a shared cluster: some cuda cells are 3-5 % noise versus the previous tokens.
(f) The fp32 pair-bias path files are shared with B200; B200 was not validated.

## Prebuild regeneration (release build)

Kernels wheels (sm90 / sm80 / sm120, cu129, cp312) built incrementally from the 3.0.3 release cache (via the earlier 3.0.4 candidate cache) with GCC 12.4 and the pixi CUDA 12.9 toolkit (torch 2.13+cu129; CUTLASS 4.2, MathDx), GPU-less on the cluster. The final plan: sm90a 579 artifacts (4 newly built on the final source: `token_pair_init_sm90`, `parallel_residual_sm90`, hoist-rows extension and one more), sm80 74 (8 built), sm120a 67 (9 built + `swa_wgrad5_sm120`). Carried names (`triattn_h100_core_h16_v2`, `miniworld_pwa_dgv_bwd`, `_glue3`, `_pair3`) are carried byte-for-byte (8 files).
**No name of 3.0.3 is dropped:** manifest names sm90 364 -> 364 (0 dropped / 0 added), sm80 72 -> 74 (0 / 2: `ieee_f32_sm80`, `sampler_ops_f32_sm80`), sm120 64 -> 67 (0 / 3: `ieee_f32_sm120`, `sampler_ops_f32_sm80_sm120`, `swa_wgrad5_sm120`).
**`ldd` on every `.so` of every arch (staged trees and installed wheels, 377 files at the baseline build): 0 link a CUDA-13 library, 0 unresolved**; the CUDA libraries are `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` only. Highest symbol versions: GLIBC 2.14, GLIBCXX 3.4.21, CXXABI 1.3.11. The sm90 kernels wheel is 100.74 MB (PyPI limit 100 MiB; 3.0.3 was 100.25 MB).

## Verification

Installed-wheel runs use the release artifacts extracted without the repo on the path, an empty JIT cache, no CUTLASS / MathDx and no `nvcc` on `PATH`.

| item | result |
|---|---|
| ldd of every `.so` (staged trees + installed wheels) | 0 link CUDA 13, 0 unresolved |
| H100 installed-wheel load of every indexed sm90 artifact (579 = 163 + 190 + 226) | 579 present, 579 resolved, **0 compiled**; also with the system libstdc++ preloaded |
| H100 installed accuracy (13 modules, pytorch / cuda / triton) | RED 0, error 0, 0 builds (cuda 172 GREEN / 98 YELLOW) |
| H100 `test_restored_triton_backends.py` / `test_bias_only_fp32_logits.py` (installed wheel, no JIT) | 63 passed / 46 passed, 8 skipped (Triton has no hoisted-bias path of its own) |
| H100 outputs vs the installed 3.0.3 wheel (`_sm_count.cuh` modules: OPM, TokenDiT rows, bias-only DiT, PairEmbed*, TriAttn, AugmentNoise / AlignEuler / PoolAtoms, TokenPairInit; 108 registry eval cases, min length) | 100 cases bitwise identical, 0 different, 0 error; the 8 `TokenDiTMiniWorldBiasOnly` cases (fp32, master) changed as intended by the fp32-logits rule, error vs fp64 unchanged (e.g. rel-rms 2.66e-4 / 3.08e-3 in both versions) |
| H100 default routes vs 3.0.3 (inference 29 cases, h16 TriAttn core 6) | bitwise identical; default training paths: as 3.0.3 (token DiT Boltz-2 / RF3, TriMul D128 / D256, TriAttn differ between two runs of the same wheel: nondeterministic backward, same as 3.0.3) |
| A100 installed wheels (cssb3) | 74 indexed artifacts resolved, 0 compiled; accuracy RED 0 / error 0 (cuda 181 GREEN / 103 YELLOW); `test_restored_triton_backends.py` + `test_bias_only_fp32_logits_sm80.py`: 69 passed; 0 kernels built (baseline build, before the sm120 plan fix) |
| RTX 5090 installed wheels (final build) | 67 indexed sm120 artifacts resolved, **0 compiled**; accuracy RED 0, error 0 (cuda 188 GREEN / 96 YELLOW; the 7 `swa_wgrad5_sm120` errors of the first build are gone); restored Triton + sm80 bias-only fp32 tests: 65 passed, 4 skipped; 0 kernels built |
| RTX 5090 real Protenix v1 step (installed wheels) | 2.0 s / step, step-0 loss / gnorm 0.3764 / 1.242, 0 engine Triton, no segfault, 0 kernels built, peak 16.93 / 25.32 GiB (alloc / reserved) |
| RTX 5090 max Pairformer blocks L 384 (`pf_train_bench.py max`, installed wheels, default allocator) | **31** (0.264 s / step, 27.99 GiB alloc); 32 blocks fit only with `expandable_segments:True` (28.81 GiB alloc) |
| CPU gate (source branch) | ruff, pyright 0 errors, pytest 3414 |
