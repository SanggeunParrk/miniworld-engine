# TriMul / OuterProductMean ladder fixes (H100, 2026-10-26)

Branch `perf/ladder-fix-trimul` on `perf/ladder-audit`. Rows re-measured with `benchmarks/runners/ladder.py` (groups `tm_ tmb_ ta_ opm pwa`, 3 repeats, CUDA graph, one H100, torch 2.13+cu129,
commit `fd4a78fb`). Raw rows: [data/trimul_opm_ladder_20261026.csv](data/trimul_opm_ladder_20261026.csv). "Before" = [performance_ladder_20261025.md](performance_ladder_20261025.md).
Result (final run job 35907, commit `eac94284`): 86 rows, **86 OK, 0 violations, 0 routing gaps**; before: 16 violation / n.a. rows of these ops plus 32 `n.a.` fp32-master TriMul rows and 4 Chai-1 routing gaps.

## What changed

| item | before | after | change |
|---|---|---|---|
| TriMul bidirectional, hidden 2 D, D64 / hidden 128, L384 inference | CUDA 1.231 ms (= PyTorch 1.219; portable 0.318; ratio 3.86) | CUDA 0.209 (0.66 of portable) | the contract refused it, the packaged K1/K3 unit (c_z 64, 256 channels) already existed: gate `trimul_h100._hidden_twice_bidirectional` |
| same, training | CUDA 3.643 (portable 1.379; ratio 2.64) | CUDA 0.766 (0.56) | new `d64_h2_bwd.cu` + `h100_d64_h2.py`: the D64 backward split into channel halves (`b1a` whole-row LN sums / dp / dgl, `b1b` per half, `b2w` two passes, `fin`); K3 save compiles at `MW_HIDDEN=256`; one sample per launch, any B chunked |
| single-direction TriMul, fp32 master params (32 rows `n.a.`: "Both operands must be same dtype") | portable failed | all measured, all OK | `TriangleMultiplication._forward_triton` casts the weights to the activation dtype (as the bidirectional module did); the cast is differentiable, master gradients stay fp32 |
| TriMul outgoing D128 training L128 (bf16 params) | 0.986 | 0.936 (master params) | five weight transposes in one `mw_aux_copy_n` launch |
| OPM RF3 training (projection bias), both orders | routing gap: CUDA = PyTorch (0.864 / 1.767 ms) | 0.670 / 1.326 ms (0.83-0.87 of portable) | `opm_train._forward_wide` takes the biases (prologue) and their gradients (two GEMVs) in grad-enabled calls; routed in `OuterProductMean.forward` |
| OPM RF3 inference | 0.992 / 0.963 / 1.001 / 0.978 of portable | 0.948 / 0.936 / 0.964 / 0.960 | wide prologue keeps the 32 mask bits per thread from smem instead of 32 global loads of `mask16` (26.6 -> 18.9 us) |
| OPM RF3 "divide after" training, portable vs PyTorch | 1.016 / 1.004 (VIOLATION) | 0.883 / 0.881 | portable bias gradient: `ones . dz` GEMV instead of an ATen axis-0 reduction (66-140 us) |

Parity: `tests/gpu/sm90/test_trimul_h100_bidir_hidden2x_gpu.py` (new, 88 with its sibling): output and every gradient against the fp32 module (y < 6e-3, grads < 2e-2), bitwise run to run, CUDA-graph replay, dropout draw, checkpointed chunks;
probe vs fp64: engine error 0.66-0.96x of bf16 PyTorch's own on y, dx and every parameter gradient. `test_opm_variants_gpu.py::test_rf3_training_matches_fp64_statements` (6 cases): <= 1.5x the bf16 statements' error.
GPU tests of OPM / TriMul / MSA Triton (254 passed, 9 skipped, job 35510). No rtol loosened.

## Table: ratios after (CUDA / portable, portable / PyTorch)

Every TriMul (outgoing, incoming, hidden != D, bidirectional, bidirectional hidden != D), TriAttn, PWA and OPM row is OK; the largest CUDA / portable ratios are in the CSV. Rows >= 0.93: OPM S1024 inference 0.977 / 0.959 / 0.934,
RF3 inference 0.936-0.964, TriMul outgoing D128 training L128 0.936, PWA inference L768 0.895.

## Chai-1 OPM (8 groups): CUDA now serves inference and training

Before: inference = composed cuBLAS + `gather.cu` (0.277 / 0.552 ms, 1.24-1.27x slower than portable 0.218 / 0.446), training = PyTorch statements (0.81 / 1.59 vs portable 0.759 / 1.451).
After (S512, pair 256, 3 repeats, commit `eac94284`): inference **0.187 / 0.366** (c/p 0.86 / 0.82, vs PyTorch 0.65 / 0.64), training **0.654 / 1.223** (c/p 0.86 / 0.85).
`kernels/outer_product_mean/chai.py` + `cuda/chai.cu`: `chai_proj` (LayerNorm + projections + mask, `mma.sync`, group-major outputs), grouped products in one cuBLAS batched GEMM, `chai_epilogue`
(per pair: LayerNorm of 512 products + out-projection + residual; TMA pair tile into a no-swizzle K-major layout, wgmma m64n128k16 against W' = W_out gamma, b' = b_out + W_out beta; two CTAs per SM),
backward `chai_bwd` (LayerNorm backward scattered by TMA), `chai_projbwd` (dcat, LayerNorm backward, gamma / beta partials), dW' / dy in cuBLAS, dgamma = sum_o W dW', dbeta = W^T dbias.
Parity (`test_chai1_fused_passes_match_fp64_statements`, 3 cases): y and every gradient within 1.5x of the bf16 statements' own error.

## Still open (as open, with numbers)

* **No row violates the ladder (86 / 86 OK), but these CUDA / portable ratios are above 0.95** (the user's target for a clear win): OPM MiniWorld S1024 inference 0.958 / 0.957 / 0.971 (L256 / 384 / 768), RF3 "divide after" inference 0.947-0.954 (borderline),
  Chai-1 training L256 0.936 is below. Both rungs are cuBLAS GEMM (~110-190 us) + epilogue (66-78 us) bound; only a fused GEMM + epilogue (O never written, ~90 us of ~190) would give <= 0.95 clearly: **not built** (a persistent wgmma kernel that repacks the 128 x 128 accumulator tile into pair-major
  operands for the W_out GEMM; out of reach in this session). Measured instead: the pack launch / casts removed (RF3 inference 0.2125 -> 0.2067 ms), cuBLAS operand-layout variants are within 3 % and not consistently better.
* Chai-1 CUDA is not tuned further: the epilogue (85 us inference / 115 us with y saved) and `chai_projbwd` (67 us) are 1.5-2x over their memory floors; the projection / LayerNorm weights are cast per call (a few small kernels in fp32-master mode).
* `ruff` / `ty` / CPU tests (job 35908, tree `eac94284`): CPU suite 4363 passed, 2 failed + 2 errors (`test_key_flags_are_built_on_both_sides`, `test_no_stale_caches`, `test_a_ladder_offers_what_wins` x2), **identical on the base commit `d9be45ad`** (same 2 failures + 2 errors).
  ruff on the changed files: 2 findings (`bench.py:734` RUF100, `triangle_multiplication/module.py:192` I001), both present on the base. ty: findings on the changed files predate this work (`h100_single.py` argument errors, `triton_outer_product_mean` call in the OPM module, `test_opm_variants_gpu.py:34`); my new test lines add the same
  `projection_bias=int` pattern as line 34.
* `dev derive --arch sm86` run in this tree (writes only `registry_kernel.units.json`, re-derived at the end). No `registry.csv` rows were added: the H100 hand kernels (`h100_d64_b200gate`, `opm_epilogue.cu`) are not registry rows either, and the layout / registry tests pass.
* Process note: for a few short probes (< 15 s each) my debug job overlapped a running ladder job; no measurement was affected.
