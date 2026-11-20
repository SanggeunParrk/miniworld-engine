# miniworld-engine 3.0.6 (NOT RELEASED: installed-wheel verification found two failures, see "Open issues")

Base: 3.0.5 (`bef4085c2`). Integration branch `rc-3.0.6` = 3.0.5 + `fix-tr-sm80-resid-overflow` + `pwa-long-h100` + `tokendit-long-h100` + `pwa-long-l` (A100 PWA) + `tokendit-bugs-h100` (tests only) + `pwa-long-l-5090` (5090 PWA, cherry-picked `7992df822`). The package version is NOT bumped in this candidate.

**What this release is.** A 3.0.5 repair (the 5090 / A100 Transition overflow that broke 3.0.4 / 3.0.5 AF3 on the 5090) plus long-sequence inference for two modules (MSA pair-weighted averaging, bias-only token DiT). No module `version` changed; only `backend_versions` entries.

## Changes

1. **Transition, bf16 one-kernel forward (sm_80 family: A100 and RTX 5090).** FoldForge AF3 5I28 n=1 on the 5090 (3.0.5) gave `-inf` coordinates ("non-finite AF3 coordinates"): the `msa_transition` input had |x| = 65536. Cause: the D 64 / 128 one-kernel forward (`fused_fwd_sm80`; `fused_sm80` on the A100) starts its f16 accumulator at the residual x, so any |x| >= 65504 (f16 max) overflowed to inf. The accumulator now carries (x + Ws h) / 16 (`TR_RESID_SCALE`; pack flag bit 30 scales the packed Ws) and the bf16 store multiplies back. Result stays finite up to |x| of about 1e6 (the limit of this fix; above it the f16 accumulator overflows again). Outputs are otherwise within 1 bf16 ulp of the previous version. Backward is unchanged. **3.0.4 and 3.0.5 on the 5090 produced broken AF3 results for such inputs.**
2. **MSAPairWeightedAveraging, inference past the old length limits (A100, H100, RTX 5090).** The kernels stage the bf16 logits in the output rows when the shared-memory row does not fit: `pair_fwd` (sm_80 family), `pair_fwd3` (H100: `pair3.cu` fp32-row path, `pair_wide.cu` bf16-row path). A100 and H100: L up to 8192 (A100 `MAX_LEN_INFER` 8192). RTX 5090: the limit is now the card's memory (L^2 (2 d_pair + 16) bytes <= total memory - 2 GiB, L a multiple of 16, cap 8192): d_pair 128 / 256 / 384 = 8192 / 7712 / 6336 on the 32 GB card, was 4080 / 1840 / 2672 (L 2048 at d_pair 256 is newly served). L within the previous limit is unchanged (5090: 15 cells at L <= the old limit are bitwise identical to 3.0.5). A call past what fits is refused cleanly (`BackendUnsupported` / a refusal reason), never an out-of-memory.
3. **Bias-only token DiT, inference past L 3072 (H100).** Two-pass softmax rows (bf16 rows and fp32 `rows32`) past the 3072 keys a warp holds; `pvx` / `pvx32` bound 3072 -> 8192; a call that cannot fit the device memory is refused (`BackendUnsupported`). L <= 3072 outputs are identical; the pair-bias logits and the softmax stay fp32 (user rule 2026-11-19).
4. **Bug fix (H100): `softmax_t` held 1024.** In 3.0.5, bf16 / master-weights bias-only token DiT inference on the H100 with 1024 < L <= 3072 (standalone `TokenDiTMiniWorldBiasOnly`, e.g. L 1152 and 3072; L 3200 raised `BackendUnsupported`; float32 passed) raised `RuntimeError: softmax_t: L ... at most 1024`. The hoist now takes `softmax_rows` past L 1024 and the case is served. FoldForge AF3 uses `TokenDiTProtenix` and does not go through the bias-only block, so FoldForge AF3 n = 9 / 16 / 24 had no error in 3.0.5 either: the RuntimeError reproduces only with the engine called standalone. (`tokendit-bugs-h100` adds `tests/gpu/test_bias_only_mid_inference.py` for L 1152 .. 3072 on the H100 and 3200 / 4096 on A100 / 5090.)
5. **Token DiT bug investigation (`tokendit-bugs-h100`), results.** (a) The larger error of a "huge" input with master weights is not a defect: it is an artifact of the artificial weights in the AdaLN table fold; the pair-bias logits stay fp32. (b) In 3.0.5, n >= 25 (L > 3072) was unsupported on the H100 for every dtype, and on the A100 / RTX 5090 raised `softmax_rows: L ... at most 3072`; both now serve up to 8192. (c) On the A100 and RTX 5090, bias-only L 1152..3072 was already fine in 3.0.5.

## Versions (`backend_versions`)

| class | card | 3.0.5 | 3.0.6 | cause |
|---|---|---|---|---|
| Transition | a100_sm80 | 7 | 8 | f16 residual scaling |
| Transition | 5090_sm120 | 9 | 10 | f16 residual scaling |
| MSAPairWeightedAveraging | h100_sm90 | 2 | 3 | inference L > 2048 |
| MSAPairWeightedAveraging | a100_sm80 | 2 | 3 | inference L > 2048 |
| MSAPairWeightedAveraging | 5090_sm120 | 3 | 4 | inference up to the card's memory |
| TokenDiTMiniWorldBiasOnly | h100_sm90 | 8 | 9 | inference L > 3072, softmax_t fix |

**Not versioned:** the A100 / RTX 5090 token DiT bias-only support for L > 3072 came from a change in a shared `.cu` source and has no `backend_versions` entry. The judge cells (the requirements stop at L <= 768 for this module) do not cover it; it is verified by GPU tests only (`test_bias_only_mid_inference.py` on A100 and 5090: L 3200 / 4096).

`judge owed` on the merged candidate: 0 / 0 on `h100_sm90`, `a100_sm80`, `5090_sm120` (carried 17815 per kind). The 5090 PWA tokens (392 records) were issued on the 5090 branch; no token had to be re-issued in the merged tree.

## Results

Reported by the branch agents (not re-measured in the integration step):

- PWA cuda vs triton, speed: A100 1.9-2.4x faster, H100 1.4-2.8x faster, RTX 5090 1.5-2x faster.
- RTX 5090 PWA accuracy: rms vs float64 1.72e-3 .. 1.74e-3.
- Bias-only token DiT cuda vs triton at long L (H100): about 5x faster.
- Accuracy: engine standard (rms / max vs float64), 0 RED in every re-issued token set.

Integration-step GPU tests: see "Verification".

## Known limits

- Transition fix: |x| up to about 1e6 (measured limit); beyond that the f16 accumulator overflows.
- PWA lengths are limited by pair-tensor memory (5090: 8192 / 7712 / 6336 at d_pair 128 / 256 / 384). The requirements rows stay at L <= 2048 (token DiT bias-only: L <= 768), so the judge cells do not cover long L; the GPU tests (`test_pwa_long_inference_sm80/sm90`, `test_bias_only_long_inference_sm90`, `test_bias_only_mid_inference`) do.
- Every known limit of 3.0.5 remains (LayerNorm exceptions, bias-only fp32 logits cost on the H100, OpenFold3 L128 fp32 / ParallelResidual L128 vs Triton on the 5090, B200 not validated, peptide-bond C-N end-to-end check not run): see `docs/RELEASE_NOTES_3.0.5.md` and `3.0.4`.

## Verification (integration step, rc-3.0.6)

| item | result |
|---|---|
| judge owed | 0 / 0 on `h100_sm90`, `a100_sm80`, `5090_sm120`, each evaluated in that card's own environment (H100 env, cssb3 A100 env, cssb-rtx 5090 env) |
| A100 (cssb3) `test_transition_sm80_resid_overflow`, `test_pwa_long_inference_sm80` | 31 passed, 3 skipped (5090-only refusal tests); with `test_bias_only_mid_inference` (L 3200 / 4096 and the pair path): 49 passed, 16 skipped in the earlier run |
| RTX 5090 (cssb-rtx) `test_transition_sm80_resid_overflow` | 4 passed |
| RTX 5090 `test_pwa_long_inference_sm80` | 25 passed, 5 skipped (A100-only gate tests): every width at L 2176 / 3072 / 4096, the card maximum ("max": 8192 / 7712 / 6336 at d_pair 128 / 256 / 384), BIG width 2176 / 3072, refusal past the limit |
| RTX 5090 `test_bias_only_mid_inference` | 18 passed (pair path 1152 .. 3072 and 3200 / 4096), H100-only cases skipped |
| H100 `test_bias_only_mid_inference`, `test_bias_only_long_inference_sm90`, `test_pwa_long_inference_sm90`, `test_transition_sm80_resid_overflow` in one process | 72 passed, 10 skipped (the Transition test and the A100 / 5090 cases skip on the H100). Before the test fix the long bias-only cases ran out of memory in one process: the cached bias-only runners kept the hoisted P and the (pair, mask) key tensors alive; the tests now clear `bias_only_dit._RUNNERS` between cases |
| H100 Transition spike check (`chk1.py`: D 64 / 128, n 2 / 4, master and bf16 params, rows 16384 / 65536, spikes 65536 and 1e6, eval and train) | nonfinite 0, Triton launches 0 in all 64 cases, relrms 5e-6 .. 8e-5 vs float64 (D 128 rows seen) |
| CPU gate (`rc-3.0.6`) | pyright 0 errors; ruff check `src` clean; ruff format: no new unformatted file versus 3.0.5 (`ruff format --check .` already reports 175 files on 3.0.5, 174 here); pytest cpu (judge, modules_cpu, requirements, prebuild_plan, supports_signature, imports, init, capture_api) 3231 passed, 2 skipped |

## Release build (`release306`, from `rc-3.0.6` 46303e5a3 = version bump; the notes commits after it do not change `src/`)

- Prebuild (incremental, warm release cache, GCC 12.4 host): sm90 8 artifacts built (`miniworld_pwa_pair3_native_v1`, `miniworld_pwa_pair_wide_{128_8_2,256_4_3,384_6_1,512_4_1}`, `bias_only_dit_sm90a`, `bias_only_dit_rows_cuda_sm90a`, `bias_only_dit_f32_rows_cuda`); sm80 5 (`transition_fwd_sm80`, `transition_bwd_sm80`, `pwa_sm80`, `bias_only_dit_rows_cuda`, `bias_only_dit_f32_rows_cuda`); sm120 5 (`transition_fwd_sm120`, `transition_bwd_sm120`, `pwa_sm80_sm120`, `bias_only_dit_rows_cuda`, `bias_only_dit_f32_rows_cuda`). 0 failed. Plans: sm90 579, sm80 74, sm120 67 artifacts.
- Manifest names against the 3.0.5 wheels: sm90 364 -> 364, sm80 74 -> 74, sm120 67 -> 67; dropped 0, added 0 (the 8 carried older-release sm90 names are byte-identical to their source wheels).
- `ldd` of every `.so` staged (sm90 258, sm80 65, sm120 58) and of the 378 installed `.so`: 0 link a CUDA 13 library, 0 unresolved; no RPATH; max symbol versions GLIBC 2.14 / GLIBCXX 3.4.21 / CXXABI 1.3.11.
- `twine check`: 5 of 5 PASSED; engine METADATA pins the three kernels wheels `==3.0.6`, no git URLs.
- sha256 (`release306/dist/SHA256SUMS`):

```
6a98fd4dadbabb7e1491a4fb2befc9d0f57e7a896f9da61555a23d36209d342a  miniworld_engine-3.0.6-py3-none-any.whl
3265b7f7533410719bc4757106160a699eff4387451f37ff266e714f871d61e4  miniworld_engine_kernels_sm120_cu129-3.0.6-cp312-cp312-manylinux_2_28_x86_64.whl
d12e4cb55230b6be8b5615abcbfeef9aebc745a39c36d710c40aac2c143f3bd2  miniworld_engine_kernels_sm80_cu129-3.0.6-cp312-cp312-manylinux_2_28_x86_64.whl
99bb3b0beaf3654186683eb41982e853864110dfabba5c1d93fdbdbc6b6a1664  miniworld_engine_kernels_sm90_cu129-3.0.6-cp312-cp312-manylinux_2_28_x86_64.whl
b78d7f77f435f7b8e12aeb11f2ef41fd9e7802a43884f7490e6103244805bce7  miniworld_engine-3.0.6.tar.gz
```

- `benchmarks/paper/`: paper experiment harness (E1-E6) for re-measuring on one release; not part of the wheel.

## Installed-wheel verification (extracted wheels, empty JIT cache, no nvcc, no repo on the path)

| item | H100 | A100 (cssb3) | RTX 5090 (cssb-rtx) |
|---|---|---|---|
| import, versions | 3.0.6 engine + kernels | 3.0.6 / 3.0.6 | 3.0.6 / 3.0.6 |
| every indexed artifact loads, 0 builds | sm90 579 of 579 | sm80 74 of 74 | sm120 67 of 67 |
| accuracy per module (pytorch / cuda / triton), RED | 0, builds 0 | 0, builds 0 | 0, builds 0 |
| `judge owed` (repo tree; the wheel carries no tokens) | 0 / 0 on all three cards | | |
| `test_pwa_long_inference_sm90` | 26 passed | | |
| `test_bias_only_long_inference_sm90` | 21 passed | | |
| `test_bias_only_mid_inference` | 25 passed, 6 skipped | 18 passed, 13 skipped | 18 passed, 13 skipped |
| `test_large_magnitude_inference_sm90` | 98 passed, 1 xfailed (documented TriAttn limit) | | |
| `test_transition_sm80_resid_overflow` | 4 skipped (sm_80 family only) | **1 failed**, 3 passed | 4 passed |
| `test_pwa_long_inference_sm80` | | 27 passed, 3 skipped | **fails** (d_pair 512 cases, see below) |
| `test_large_magnitude_inference_sm80` | | **2 failed**, 280 passed, 7 skipped | **245 failed**, 42 passed (one sticky CUDA error, see below) |
| `test_restored_triton_backends` | | 63 passed | 59 passed, 4 skipped |
| AF3 5I28 n=1 through FoldForge, installed 3.0.6 | | finite, 18026 hook calls, 0 non-finite; all 9 engine ops cuda (install and runtime); `msa_stack.2/3.msa_transition` max abs 64768 | finite; all op plans cuda except conditioning (Triton / PyTorch by plan, not used by the AF3 classes listed); the 3.0.2 failing `msa_transition` input (max abs 65536) replayed finite on cuda |

## Open issues found by the installed-wheel verification

1. **A100: `transition_fused_sm80` is not in the sm80 kernels wheel.** `Transition` D 128 n 4 at the row counts routed through `fused_sm80` (the A100 one-kernel path that carries the 3.0.6 residual-scale fix) requests `transition_fused_sm80 [fc09f682]`; with no nvcc / ninja the load raises `KernelBuildError`. The same request exists in the 3.0.5 wheel (`[da43d32e]`) and the name was never in the 3.0.0 / 3.0.4 / 3.0.5 sm80 manifests, so this gap is not new, but it means the A100 part of the Transition fix is only reachable with a toolchain. The failing cases are `test_transition_sm80_resid_overflow[128-4]` and the two `Transition(d_hidden=128,n=4)` / `PairformerBlock` cases of the large-magnitude net.
2. **RTX 5090: shipped `pwa_gemm_sm120` artifact faults for d_pair 512.** `MSAPairWeightedAveraging` d_pair 512 (the BIG width) raises `illegal memory access` / `CUBLAS_STATUS_INTERNAL_ERROR` with the wheel's `pwa_gemm_sm120.so` (built with nvcc 12.9, 2026-11-15 release cache); the same source key built by the 5090 box's JIT (nvcc 12.8) passes (`test_pwa_long_inference_sm80[256-512-32-2176]` passed after swapping only that file). The 3.0.5 wheel fails the same case, so it predates 3.0.6; d_pair 128 / 256 / 384 and AF3 are not affected. The error is sticky, which is why the large-magnitude net on the 5090 reports 245 failures after its first d_pair 512 case.

## AF3 end-to-end facts (FoldForge, from the e2e agent; not re-measured here)

- The `softmax_t` RuntimeError for 1024 < L <= 3072 reproduces only with the engine standalone (`TokenDiTMiniWorldBiasOnly`, bf16 / master; L 3200: `BackendUnsupported`; float32 passes); FoldForge AF3 n = 9 / 16 / 24 never reached it (token DiT is `TokenDiTProtenix`).
- 3.0.5 on H100 / A100: AF3 n >= 17 (more than 2048 tokens) had `MSAPairWeightedAveraging` refused on cuda and served by Triton (a CUDA hole; `_pwa_*` Triton launches seen at n = 24 / 25 / 32). In 3.0.6 every row is cuda, Triton launches 0.
- 5090 AF3 n = 1: 3.0.5 non-finite (reproduced); 3.0.6 finite. A100 AF3 n = 1 / 9 / 16 / 24 / 32 / 37 were finite on 3.0.5 as well: the `msa_transition` input max abs 65024 is just below the f16 limit 65504, only the 5090 input exceeded 65536, so the A100 Transition fix is defensive.
- 3.0.6 limits: H100 80 GB n = 36 OK, n = 37 / 38 OOM in trunk TriAttn `_generic_rows`; A100 80 GB n = 37 OK (78.2 GiB); 5090 32 GB n = 24 OK, n = 25 / 26 OOM in trunk TriMul incoming `trimul_sm80_wide` (card memory, not an engine bug). H100 n = 1 / 9 / 16 coordinates identical to 3.0.5; n = 24 / 25 / 32 chain RMSD 0.28-0.31 A (PWA Triton -> CUDA), below the 0.5 A spread between 3.0.5 samples.
- 5090 n = 19..22: default OOM in FoldForge conditioning `features_2d` (FoldForge side, `PAIR_BAND_THRESHOLD_BYTES`); with a temporary override (0.25 GiB, no repo change) all finite, peak 20.4 / 22.3 / 24.3 / 26.9 GiB.

## Known limits added in 3.0.6

- Judge requirements rows still cover L <= 2048 (PWA) and L <= 768 (token DiT): long L is GPU-test-verified only.
- Real-weight survey margins (guards not implemented): TriAttn fp16 score margin IntelliFold2 2.5x, Chai1 8.7x; Transition fp16 `h` margin ESMFold2 `msa_stack.2.msa_transition` 1.8x. Chai1 token DiT gives inf with simultaneous cond + single spikes of 1e5 (unrealistic). LayerNorm known gaps unchanged.

## Release steps

NOT DONE: tag, PyPI upload, `pixi.lock`, push. Held by the two open issues above.
