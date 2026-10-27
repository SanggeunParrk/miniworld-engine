# H100 bias-only DiT, per-sample pair (`bodit_psb`): the hoist in one pass over the pair, 2026-10-27

Branch `perf/final-psb-cuda` (from `perf/final-psb` `8d074bc9`), code commit `384ef5db`. H100 80GB HBM3, torch 2.13.0+cu129, `benchmarks/runners/ladder.py`
groups `bodit_psb` / `bodit_psb_fp32` (3 repeats, compile + CUDA-graph and CUDA-graph rungs, the faster regime per rung), training,
`single` [1, 4, L, 768], `pair` [4, L, L, 128], 1 block, 16 heads x 48. Times in ms. Before: `docs/records/portable_bodit_psb_20261027.md`
(base `8d074bc9`, same harness); after: this branch, runs `out_bf2` / `out_f322` (Slurm jobs 36747 / 36748).

## Result

| row | L | CUDA before | CUDA after | portable | CUDA / portable before | CUDA / portable after | CUDA / PyTorch after |
|---|---|---|---|---|---|---|---|
| training bf16 | 256 | 0.473 | 0.373 | 0.427 | 1.113 | **0.874** | 0.608 |
| training bf16 | 512 | 1.154 | 0.751 | 0.880 | 1.310 | **0.853** | 0.497 |
| training fp32 | 256 | 0.981 | 0.697 | 0.785 | 1.249 | **0.889** | 0.637 |
| training fp32 | 512 | 2.610 | 1.507 | 1.704 | 1.529 | **0.885** | 0.522 |

PyTorch = the fastest of compiled + graph / eager + graph / eager (`pt_c` 0.613 / 1.509 / 1.095 / 2.889 ms). Inference rows are unchanged in kind: `miniworld`
has no hand-CUDA inference path for a per-sample pair and runs the PyTorch composition (routing gap, not a violation); that composition takes the new
hoist forward, which moved it from 1.38 / 1.65 / 1.38 / 1.62 x the portable time to 1.01 / 0.99 / 0.96 / 0.93 (bf16 L256 / L512, fp32 L256 / L512).

## What the step was spending

`nsys` of the ladder's own `bench.py` row (CUDA rung, CUDA-graph replay, kernel averages per step, bf16 | fp32, L = 256 / 512):

| piece | before | after |
|---|---|---|
| hoist forward: LayerNorm kernel + cuBLAS GEMM (reads the pair twice, writes it once) | 51 + 31 us \| 90 + (GEMM) us ; 195 + 105 us \| 353 + (GEMM) us | `hoist_fwd`: 38 \| 64 us ; 120 \| 225 us |
| hoist backward: `hoist_bwd_k` (bf16, 1 CTA / SM, block-phased) / `ln0_bwd_rows` + 3 cuBLAS GEMMs (fp32) | 109 \| 181 + GEMMs us ; 413 \| 713 + GEMMs us | `hoist_bwd1`: 56 \| 105 us ; 205 \| 413 us |

Isolated (CUDA-graph, `R = 4 L^2` pair rows, N = 16; `~/miniworld-engine-scratch/psbcuda/probe/hk.py`, `sweep.py`), us, with the DRAM floor of the traffic at the
2.88 TB/s a read + write copy of the same size reaches on this card (`copy_` 268 MB: 186 us):

| op | dtype | L | before | after | bytes | floor |
|---|---|---|---|---|---|---|
| forward | bf16 | 256 / 512 | 80 / 299 | 30 / 108 | 75 / 302 MB | 26 / 105 |
| forward | fp32 | 256 / 512 | 147 / 560 | 55 / 207 | 138 / 553 MB | 48 / 192 |
| backward | bf16 | 256 / 512 | 112 / 416 | 58 / 208 | 142 / 570 MB | 49 / 198 |
| backward | fp32 | 256 / 512 | 311 / 1175 | 107 / 410 | 274 / 1.09 GB | 95 / 380 |

(The old bf16 backward was the single-CTA-per-SM `hoist_bwd_k`: 1.27 TB/s, its 64-row items were loaded, computed and stored in barrier-separated phases, so one 16 KB item
per SM was in flight. fp32 had no fused kernel: d LN0 (fp32, 4 B per pair element) and Y went through HBM between three cuBLAS calls and `ln0_bwd_rows`.)

## The kernels (`kernels/bias_only_dit/cuda/bo_hoist_pass.cu`, sm_90a)

* **One warp is a whole pipeline.** Each warp owns a ring of 2-4 shared-memory stages (cp.async, 16-byte chunks) holding 16 pair rows (4 KB bf16 / 8 KB fp32), takes
  tiles `warp + k * all_warps` of a persistent grid (one CTA per SM, 8 warps; fp32 backward 4) and never waits for another warp: no `__syncthreads` in the loop. 100-160 KB per
  SM of loads are in flight whatever the compute does.
* **The tile is the M = 16 side of the MMA.** A row's 128 channels sit in the 4 lanes of a quad: LayerNorm mean / variance are in-register sums + two xor shuffles; the pair
  fragments come from `ldmatrix`; `d LN0 = dbias^T W'` (K = 16 heads: one k16 / two k8 steps) accumulates in the layout of the tile it is applied to, so the LayerNorm
  backward runs on the accumulators with no data movement; `d pair` goes back into the same shared tile and leaves as 16-byte coalesced stores.
* **dW' without a shared-memory round trip (bf16)**: `y` is split into `hi = bf16(y)`, `lo = bf16(y - hi)` in registers and transposed into the B-operand layout with
  `movmatrix.trans`: two MMAs per 8-channel tile against `dbias` (A operand by `ldmatrix`). fp32: `y` (TF32) through a per-warp 8 KB tile; the TF32 d LN0 MMA permutes the
  channels of each n8 tile (`8 j + n/2 + 4 (n & 1)`) so that its accumulator holds the channels the A-layout fragment of x delivered.
* **Forward**: LayerNorm and `W' y^T` in the same tile; `y` rounded once to bf16 (the old LayerNorm kernel's output) / TF32; N a multiple of 16 up to 64 rows (loop over
  16-head groups, weights in shared memory); output `[N, R]` head-major as before.
* **Determinism**: no atomics; per-warp dW' partials summed warp 0..7 into one `[16, 128]` fp32 partial per CTA, the 132 CTA partials summed by the caller in torch: a fixed
  function of the inputs (the grid is the SM count). Bitwise repeatable (tests).
* Scope: `hoist_fwd` N <= 64 (any dtype, R a multiple of 16); `hoist_bwd1` N = 16 (one block). More blocks keep `bo_hoist_bwd.cu` (bf16, N <= 192) and, for fp32, the cuBLAS
  composition; B200 untouched (gated on `(9, 0)`). Switches: `settings.bias_only_dit_hoist_fwd_fused` (new) and `bias_only_dit_hoist_bwd_fused`.

Pipeline depth / warps swept (`probe/sweep.py`): forward flat over 3-5 stages (bf16: 3 best, -2 %) and 2-3 (fp32: 2), backward flat over 3-5 stages; 6, 10 or 12 warps are
worse (register file: 10 warps spill the backward, 120 -> 460 us) -- kept 8 / 8 warps (bf16), 8 / 4 (fp32).

## Numerics

Same recipes as before, accumulation orders differ. Against an fp64 oracle (random pair, scale 2): bf16 forward max |err| 2.1e-2 on logits of scale 6 (the old LayerNorm + cuBLAS path: the same
2.1e-2: bf16 output rounding), dpair 3.9e-3 (identical to the old kernel), dW' 4.6e-3 on scale 1854 (old 8.8e-3: the one-pass kernel is closer, hi + lo accumulate in
fp32 registers); fp32 (TF32): forward 1.9e-3, dpair 6.3e-4, dW' 0.53 on scale 1854 (old 0.54): equal to the TF32 GEMMs they replace.

## Tests

* `tests/gpu/sm90/test_bias_only_dit_psb_gpu.py`: 101 passed (26 new: `test_one_pass_hoist_forward` bf16 / fp32 x N 16 / 32 / 64 x R 16 / 4112 / 64000,
  `test_one_pass_hoist_backward` bf16 / fp32 x R 16 / 4112 / 64000 / 262144; each against fp64, against the old route and bitwise repeatability; the existing
  `test_fused_hoist_backward_*` cover N = 16 through the new route and N = 48..192 through the old one).
* `tests/gpu/sm90/test_h100_bias_only_dit_gpu.py`: 84 passed, 36 skipped (as before).
* `tests/gpu/any/test_bias_only_dit_portable_gpu.py`: 40 passed, 1 failed (`test_block_matches_pytorch[1024h16dNone-fp32-infer]`) in the full-file run while 4 other
  jobs shared the node; the test passes alone on this branch and on the base (portable kernels only: unrelated to this change).
* `tests/cpu`: 6 failed + 2 errors (autotune sweep-page counts, stale committed caches, key-flag coverage), the same 8 as on the base `8d074bc9`.

## Left

* The inference rows of `bodit_psb` stay routing gaps: there is no hand-CUDA per-sample inference block (`miniworld` runs the PyTorch composition, now 0.93-1.01 x the portable time
  through the new hoist forward); training is where the invariant now holds.
* Several blocks (`nb > 1`): the backward keeps the old paths (bf16 `bo_hoist_bwd.cu` at ~1.3 TB/s, fp32 cuBLAS); the one-pass backward would need the `[16 nb, 128]` dW'
  accumulators in shared memory instead of registers.
* The rest of the step is outside this change: `pack90` weight packing (~40 us a bf16 L512 step), the attention core (`psb_fwd` / `psb_bwd`, 46 + 39 us) and the cuBLAS GEMMs.
