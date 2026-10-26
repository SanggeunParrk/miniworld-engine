# TriMul / OuterProductMean ladder fixes (H100, 2026-10-26)

Branch `perf/ladder-fix-trimul` on `perf/ladder-audit`. Rows re-measured with `benchmarks/runners/ladder.py` (groups `tm_ tmb_ ta_ opm pwa`, 3 repeats, CUDA graph, one H100, torch 2.13+cu129,
commit `fd4a78fb`). Raw rows: [data/trimul_opm_ladder_20261026.csv](data/trimul_opm_ladder_20261026.csv). "Before" = [performance_ladder_20261025.md](performance_ladder_20261025.md).
Result: 86 rows, **82 OK, 0 violations, 4 routing gaps (Chai-1 OPM, not fixed)**; before: 16 violation / n.a. rows of these ops plus 32 `n.a.` fp32-master TriMul rows.

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

## Not fixed (truthfully)

* **Chai-1 OPM (8 groups), 4 rows**: still a routing gap, the CUDA rung (the composed cuBLAS + `gather.cu` inference path; training = PyTorch) is 1.07-1.27x *slower* than the portable Triton kernels. A fused TMA + wgmma epilogue
  (gather + product LayerNorm + out-projection + residual, saves ~120 us of ~210) was started; the first build showed that TMA SWIZZLE_128B pads a 16-byte inner box line to a 128-byte row, so the tile has to be loaded with a no-swizzle K-major descriptor
  (box (l, j, k), LBO 1024 B, SBO 128 B). The WIP (`chai.cu`, `chai_helpers.cuh`, probes) is outside the tree in `~/miniworld-engine-scratch/lf-trimul/chai_wip/`; nothing of it is shipped. Not routed to portable (policy: no Triton on H100).
* **OPM near-misses** (MiniWorld S1024 inference 0.934-0.977, RF3 inference 0.94-0.96): CUDA wins by 2-7 %, not <= 0.95 everywhere. Both rungs are cuBLAS-GEMM (~110-190 us) + epilogue (~66-78 us) bound; the CUDA epilogue is already faster than the Triton one,
  cuBLAS algorithm choices (mm vs bmm vs transposed operands) differ by <= 3 % and not consistently. Only a fused GEMM + epilogue (never writing O, ~90 us of ~190) would give a clear win.
* TriAttn and PWA had no row >= 0.95; unchanged.
* Process note: for a few short probes (< 15 s each) my debug job overlapped a running ladder job; no measurement was affected.
