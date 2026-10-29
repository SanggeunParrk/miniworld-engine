# Transition on H100 (sm90)

Kernel-level status of the Transition (`modules.Transition`, `y = x + W_s(silu(W_a·LN(x)) · (W_b·LN(x)))`, hidden n·D) on H100; the
module-level summary is in [h100.md](../h100.md), the dispatch contract in [dispatch.md](../dispatch.md). H100 = hand-written CUDA
(wgmma + TMA + mbarrier, accumulators in registers; sm_90a has no TMEM, so nothing here is a port of the sm_100 kernels); a shape without
a CUDA path is 미구현 and runs the PyTorch reference (no portable-kernel fallback on H100, see [dispatch.md](../dispatch.md)).

Coverage (`Transition(d, n)` through the module, `ops.transition`, the whole-op wrapper; training = forward saving what the backward needs +
backward; inference = forward saving nothing):

| | bf16, whole 128-row tiles | bf16, any row count | fp32 (TF32 tensor cores), any row count |
|---|---|---|---|
| n = 4, D = 64 / 128 / 256 / 384 / 512 | tuned (`fused_sm90a`, `fused_wide_sm90a`) | generic | generic |
| n = 2, D = 64 / 128 / 256 | tuned (the same kernels, built for hidden 2 D) | generic | generic |
| n = 2, D = 384 / 512 / 768 and n = 4, D = 768 | generic | generic | generic |
| any D = 64 k ≤ 1024, hidden = 128 j (n = 3, ...) | generic | generic | generic |

"Whole 128-row tiles" holds for every pair stream (`L²` is a multiple of 128 for L = 128 ... 768), the MSA stream and the single stream at
the registry lengths. A ragged row count (e.g. a 100-token pair, 10 000 rows) is not a fallback: the generic kernels run it natively (TMA
zero-fills the tail tile on the way in, clips it on the way out; the row kernels and fp32 stores guard the row index).

Dispatch order (`modules/transition/module.py::_residual_forward`, `kernels/transition/whole_op.py`), the first `available()` wins:
`fused_sm90a` (D = 128, hidden 512 or 256, bf16, rows % 128 = 0) → `fused_wide_sm90a` (D = 64 / 256 with hidden 4 D or 2 D, D = 384 / 512 with
hidden 4 D; bf16, rows % 128 = 0) → `fused_generic_sm90a` (everything else the table lists) → sm_100 / sm_80 paths (they reject sm_90) → on H100
the PyTorch reference (the Triton residual path on other cards or on request). fp32 enters `Transition.forward` through the same gate (`_generic_sm90a_serves`) instead of the bf16-only
`guard_dtype` PyTorch fallback, and only when TF32 is allowed (`torch.set_float32_matmul_precision("high")` or
`torch.backends.cuda.matmul.allow_tf32`, the switch cuBLAS follows); with exact fp32 the PyTorch reference keeps running. No
opt-out switch: only `force_portable_kernels` sends Transition to Triton. `transition_wide_repl` overrides the D = 64, n = 2
backward's weight-role replicas.
Every launch is an opaque op with a fake, so all of it traces under `torch.compile(fullgraph=True)` and captures in a CUDA graph.

### fp32 at few rows (the token-stream conditioning transitions, 2026-10-29)

At 128 - 768 rows (the n = 2 conditioning transitions of the token stream, `[1, L, d]`) one 128-row tile per CTA leaves most SMs idle, so
`fused_generic_sm90a` changes plan by shape (no switch):

| call | plan |
|---|---|
| fp32 training, rows <= d, d >= 512 (`_f32_cublas`) | hand `ln_fwd` -> cuBLAS TF32 a \| b over one stacked [Wa; Wb] per step -> `swiglu_f32` -> cuBLAS addmm (residual as C); backward cuBLAS dh -> `swiglu_bwd` (fp32) -> dW GEMMs on the side stream -> cuBLAS dxn over the same [Wa; Wb] -> hand `ln_bwd` |
| fp32 training, otherwise | the hand kernels; the step's TF32 weight copies (rounded Wa / Wb / Ws, Ws^T, [Wa; Wb]^T) in ONE `prep` launch in the forward, reused by the backward |
| fp32 inference | the hand kernels on TF32 weight copies made once per parameter version inside the launch (a compiled call and a capture reuse them); the squeeze as a cuBLAS addmm at <= 512 rows; the fp16 path's scales (>= 16 384 rows) cached the same way |

Ladder (H100, CUDA graph, `trc_fp32_single_D768`, ms, CUDA / portable / PyTorch compiled): training L128 0.075 / 0.085 / 0.087, L384 0.109 /
0.133 / 0.138, L768 0.165 / 0.178 / 0.190; before: CUDA 0.119 / 0.139 / 0.195 and portable 0.115 / 0.152 / 0.199 (portable > PyTorch on every row).
The portable rung at these shapes: the squeeze splits K (`_split_k_residual_kernel` sums the splits in order), the saved halves stay fp32 below 4096
rows, dx is two GEMMs over the dAB halves, and the H100 autotune caches carry the fp32 token-stream keys.

## Tuned kernels at n = 2 (D = 64 / 128 / 256)

The kernels of the n = 4 builds (D = 128: `transition_fused_{fwd,bwd}_sm90a_kernel.cu`; D = 64 / 256: `wide/kernels/`), compiled for hidden 2 D
(`-DHMULT=2`, `-DWIDE_HM=2`): the forward streams four (D = 128), two (D = 64) or sixteen 32-unit (D = 256) hidden chunks per row tile instead of
eight / four / thirty-two; the D = 128 / D = 64 backward weight role owns four / two hidden slices instead of eight / four. What had to change besides
the literal: the weight-role replica count (the weight role is `slices x replicas` CTAs and the input role has half the chunks to stream, so n = 2
wants ~64 weight CTAs in total, not `8 x 8`):

| D128 n = 2, M = 147 456 / 589 824 | repl 4 | 6 | 8 | 10 | 12 | **16** | 20 |
|---|---|---|---|---|---|---|---|
| backward µs | 779 / 3028 | 534 / 2046 | 404 / 1537 | 331 / 1240 | 279 / 1034 | **255 / 952** | 335 / 1261 |

| D64 n = 2, M = 147 456 / 589 824 | repl 16 | 24 | 32 | 40 | 48 | 56 |
|---|---|---|---|---|---|---|
| backward µs | 176 / 626 | 126 / 398 | **123** / 437 | 166 / 568 | 225 / 808 | 385 / 1384 |

Defaults: D128 `transition_candidates(slices=4)` = 16 replicas (the autotune grid keeps 2 / 4 / 8 / 12 / 16 / 20; its cache keys carry the weight
shapes, and `dev build` now drives both hidden widths: `kernels/drivers/transition.py`), D64 `WIDE_REPL` = 28 (between the two measured best).
The D = 256 backward is the gate kernel → fp32 dW GEMMs (cuBLAS) → d_xn + LayerNorm-backward kernel of the n = 4 path with `K2 = 2·hidden`.
`dgamma` / `dbeta` are summed in a fixed CTA order (no atomics), so a replay is bit-identical.

## Generic kernels (`fused_generic_sm90a`, sources in `kernels/transition/cuda/generic_sm90a/`)

Any D that is a multiple of 64 up to 1024, any hidden width that is a multiple of 128, bf16 (wgmma k16) or fp32 on TF32 (wgmma k8), any rows.
One persistent GEMM skeleton (`tr90.cuh`): a producer warpgroup issues TMA loads of 128 B-swizzled K-major slabs through an mbarrier ring, two
consumer warpgroups (64 rows each) run `wgmma` SS from shared memory with fp32 accumulators in registers (`setmaxnreg` 40 / 232); every operand is
K-major, which TF32 wgmma requires, so the backward gets transposed weight copies from a one-launch tiled transpose kernel (`prep`). The output
tile is the widest that still gives every SM a tile (256 / 192 / 128 / 64 columns; 128 / 64 hidden units for the SwiGLU GEMM), so a few rows
(single stream) still fill the GPU. bf16 outputs are staged through shared memory and written by TMA (a thread's fragment holds 4 contiguous
bytes: direct stores were 5x slower on the gate stage, 448 → 117 µs); fp32 outputs write 8 B per thread = a whole sector and are stored directly.
fp32 operands the MMA reads next (xn, h, dA / dB, the weights) are rounded to nearest TF32 where they are produced: the tensor core would
truncate them, a one-sided error that made y 4.5x less accurate than cuBLAS TF32 (1.3e-3 → 2.9e-4 against float64).

##### Forward (inference: stats not stored; training: xn, rstd, c1 = mean·rstd saved)

| kernel | what | µs, M = 147 456, D128, n = 2, bf16 |
|---|---|---|
| F1 `ln_fwd_kernel` | LayerNorm, warp per row, two-pass in registers → xn (T), rstd, c1 | 32 |
| F2 `gemm_kernel<SWIGLU>` | h = silu(xn Wa^T) · (xn Wb^T): Wa / Wb tiles side by side (two TMA boxes), SwiGLU in the epilogue | 57 |
| F3 `gemm_kernel<RES>` | y = h Ws^T + x: the residual tile is TMA-loaded into the staging buffer during the mainloop, added in place | 57 |

In the bf16 training forward F2 is `gemm_kernel<SWIGLU_SAVE>`: it also stores `[a | b]` (bf16, `[M, 2 HID]`) and keeps `h` (from the fp32
accumulators) for the backward, as TE does. fp32 / TF32 keeps the recompute backward below
(for bf16 it was ~10 % slower at large M, with dx / dW errors 10-25 % lower).

##### Backward (bf16, a | b and h saved)

| kernel | what |
|---|---|
| B1 `gemm_kernel<GATE>` | dh = dy Ws (the weight is read **as stored**: MN-major wgmma operand, `tnspB = 1`, LBO 8192 / SBO 1024 - no transposed copy, no prep launch); the saved a, b tiles are TMA-loaded during the mainloop and the SwiGLU backward in the epilogue stores [dA \| dB] |
| B2 `gemm_kernel<OUT, BT>` | d_xn = [dA \| dB] [Wa; Wb]: Wa then Wb read as stored (MN-major); below 12 288 rows a cuBLAS `mm` over a concatenated [Wa; Wb] is 3-12 % faster (same rounding) and is used |
| B3 `ln_bwd_kernel` + `reduce_cols_kernel` | as below (grid = 2 x SMs; `reduce_cols` is a PDL dependent) |
| dW | two cuBLAS GEMMs (dWs = dy^T h, [dWa; dWb] = [dA \| dB]^T xn) |

##### Backward (fp32 / TF32, or `SAVE_AB=0`): recompute

| kernel | what | µs, same shape |
|---|---|---|
| B0 `prep_bwd_kernel` | Ws^T, [Wa; Wb]^T (+ TF32-rounded Wa / Wb) in one launch | 3 (13 at 768 rows, D768) |
| B1 `gate_kernel` | dh = dy Ws, a / b = xn Wa^T / Wb^T over K = D, SwiGLU backward → h, [dA \| dB] (three m64n64 chains per k step) | 117 |
| B2 `gemm_kernel<OUT>` | d_xn = [dA \| dB] [Wa; Wb] (activation dtype) | 64 |
| B3 `ln_bwd_kernel` + `reduce_cols_kernel` | LayerNorm backward from the saved stats + residual dy, per-CTA dgamma / dbeta partials summed in fixed order | 63 + 6 |
| dW | two cuBLAS GEMMs (dWs = dy^T h, [dWa; dWb] = [dA \| dB]^T xn), fp32 accumulation, one rounding to the weight dtype | 127 (incl. split-K reduce) |

The dW GEMMs are the only non-hand-written step (as on B200, where the D ≥ 256 dW GEMMs are cuBLAS too): TF32 wgmma cannot read an M-major operand,
so a hand-written dW would need transposed copies of dy / h / dA / dB / xn.

### Choices that were measured and kept or dropped (2026-10-14)

| feature | result |
|---|---|
| a / b and h saved by the forward, backward gate = dh GEMM + epilogue | kept. D768 n = 2, 147 456 rows training 6.59 -> 5.60 ms; h from the fp32 accumulators keeps dWs at the Triton error (3.35e-3; recomputing h from bf16 a, b gave 4.25e-3) |
| MN-major B operands (no `prep_bwd`) | kept. -11 µs of 94 at 768 rows |
| tile width = argmin(waves x (width + 32)) | kept (`pick_width`). 144 tiles on 132 SMs (two waves) no longer chosen at 3072 rows, D768: -7 % |
| cuBLAS `mm` for d_xn below 12 288 rows | kept (`_cublas_plan`) |
| PDL (`ln_fwd` -> F2 -> F3, `ln_bwd` -> `reduce_cols`; `griddepcontrol.wait` at kernel entry) | kept, 2-4 % at small rows |
| `ln_bwd` grid 2 x SMs (was 8 x) | kept, 2-3 % at 768 / 3072 rows |
| 2 CTAs / SM (epilogue of one overlaps the mainloop of the other) | **dropped** (removed after measurement): no gain; the n128 SwiGLU accumulator does not fit 80 registers |
| 2-CTA cluster, B tile TMA multicast | **dropped** (removed after measurement): no gain. A first version was 2x slower: `mbarrier.arrive.release.cluster` on the remote empty barrier costs a `membar` per stage (ncu: membar stalls); with the plain arrive it is level with the non-clustered kernel |
| `torch.addmm` for F3 (residual as beta C) | **dropped**: faster below 12 k rows but y error 2.3e-3 -> 2.6e-3 (two roundings) |
| cuBLAS dh + elementwise SwiGLU backward | **dropped**: dx / dW errors +7-10 % (dh rounded to bf16) |

ncu on the old gate kernel at 147 456 rows (D768): 1.17 ms, DRAM 52 %, tensor pipe 39 % - the two alternate (epilogue and mainloop do not overlap),
floor ~0.6 ms.

## Verification (GPU, node01, H100 SXM)

`tests/gpu/any/test_transition_generic_sm90a_gpu.py` (2026-10-14: `-k transition` over the GPU test directories of that time, 429 passed; the 6 failures are `test_conditioned_transition_precision_gpu` and fail identically on origin/main):

* output and all six gradients against a float64 PyTorch reference: bf16 relative error 2.3e-3 (out), 2.6e-3 (dx), 2.5 - 3.5e-3 (dgamma, dbeta, dW) on
  every shape - the same as the portable path's own error; TF32 2.9e-4 / 4.4e-4 / 5 - 7e-4 (torch's own TF32 reference: 2.9e-4 / 3.7e-4 / 5 - 5.5e-4);
  tolerances in the tests: 8e-3 bf16, 4e-3 fp32;
* shapes: M = 1 ... 1000 ragged, D = 64 ... 1024 (incl. 192 / 320 / 576), n = 2 / 3 / 4, 4-D activations, the module path with its parameters;
* bitwise: replay of forward + backward identical (tuned D64 / 128 / 256 and generic at every width), the no-grad forward equals the training forward,
  CUDA-graph capture + replay identical to eager;
* module: `Transition(d, n)` enters the expected path (spies on the three entry points), is no less accurate than the portable path (≤ 1.6x its error against float64),
  fp32 enters the generic kernels only with TF32 on, `torch.compile(fullgraph=True)` forward + backward matches eager.

## Measurements (2026-10-15)

Module path `Transition(...)`, CUDA-graph replay, H100 SXM, torch 2.13 / cu129. Raw anthropic rows and the stack limits:
[docs/records/2026-10-15-h100-transition-vs-anthropic.md](../../../records/2026-10-15-h100-transition-vs-anthropic.md). The 2026-10-14
tables (pair / single / ragged rows and fp32 against PyTorch, Triton and Transformer Engine):
[docs/records/h100_transition_20261014.md](../../../records/h100_transition_20261014.md).

### Against anthropic and cuequivariance (inference) and PyTorch (training)

cuequivariance has no Transition kernel: `implementation="cuequivariance"` runs the PyTorch reference, so that column is `torch.compile` of the
reference. anthropic (`integrations/anthropic.py`; rows `v2` at D128 n = 4, `pf` at D64 / 128 / 256 / 384; inference only) refuses D512, n = 4 at D64
and n = 2 at D384 / 512; its D256 specialised row `esm_t16` needs the CUDA 13 driver stack (this cluster has 575 / CUDA 12.9: the cubin load
crashes), so D256 is `pf`. The official harness (`bench.py target=transition level=module`, compiled + CUDA graph, `L x L` pair rows >= 16 384) measured D128
1.18 - 1.42x anthropic on 2026-10-14 with the fused kernel that was corrected on 2026-10-15; the module-path table below is the current one.

Inference, µs (ours / best loadable anthropic row / PyTorch compile), re-measured 2026-10-15 on the final kernels (single-call CUDA graph; the
anthropic column is the fastest row that loads on this stack: D64 `v2` / `lnl` / `af3_fused`, D128 `lnl` (n = 2) and `v2` (n = 4), D256 `torch_swiglu` /
`lnl` / `pf`, D384 `pf`; `-` = every row refuses the cell). Every cell with an anthropic number is the fastest:

| n, D | 768 rows | 3072 | 12 288 | 49 152 | 147 456 |
|---|---|---|---|---|---|
| 2, 64 | 4.4 / 5.3 / 13.5 | 4.6 / 5.4 / 14.8 | 6.2 / 7.0 / 18.8 | 11.8 / 14.2 / 41.5 | 27.8 / 35.6 / 146 |
| 2, 128 | 8.6 / 9.4 / 14.2 | 8.9 / 9.7 / 16.6 | 9.7 / 12.7 / 26.4 | 22.9 / 34.4 / 91.7 | 69.1 / 102 / 280 |
| 2, 256 | 10.4 / 26.0 / 16.6 | 13.8 / 32.7 / 21.3 | 25.7 / 43.8 / 48.5 | 80.4 / 136 / 198 | 222 / 396 / 545 |
| 2, 384 | 11.8 / - / 18.6 | 20.7 / - / 28.0 | 58.6 / - / 78.8 | 190 / - / 305 | 550 / - / 864 |
| 2, 512 | 13.6 / - / 20.3 | 24.8 / - / 34.9 | 80.9 / - / 126 | 294 / - / 447 | 892 / - / 1281 |
| 4, 64 | 5.6 / 6.9 / 13.9 | 5.8 / 7.0 / 15.5 | 8.0 / 9.4 / 23.1 | 16.6 / 20.1 / 74.6 | 42.2 / 54.0 / 227 |
| 4, 128 | 9.4 / 13.9 / 15.2 | 12.0 / 14.2 / 18.9 | 13.8 / 19.2 / 37.4 | 35.3 / 54.3 / 157 | 103 / 157 / 437 |
| 4, 256 | 12.0 / 28.2 / 18.5 | 18.0 / 46.6 / 28.4 | 41.7 / 77.7 / 90.9 | 130 / 234 / 317 | 374 / 683 / 892 |
| 4, 384 | 17.1 / 77.4 / 21.9 | 29.5 / 159 / 40.7 | 95.8 / 460 / 150 | 306 / 1839 / 521 | 906 / 5407 / 1487 |
| 4, 512 | 20.4 / - / 25.4 | 40.0 / - / 56.9 | 142 / - / 211 | 486 / - / 788 | 1505 / - / 2257 |

Not measurable on this machine: the D256 specialised row `esm_t16` and `flash_sm90a` (see the record). The D256 anthropic column is
therefore the best row that loads, not the best row that exists.

Thinnest margins against anthropic: n = 2 D128 at 3 072 rows 7.2 %, at 768 rows 7.4 %; n = 2 D64 at 12 288 rows 10 %; n = 4 D64 at 12 288 rows 13 %,
n = 4 D128 at 3 072 rows 15 %.

Training (fwd + bwd, re-measured 2026-10-15 on the final build; µs, ours / PyTorch `torch.compile`; anthropic has no backward), rows 768 / 3 072 /
12 288 / 49 152 / 147 456, every cell 1.32 - 3.0x:

| n = 4 | 768 | 3 072 | 12 288 | 49 152 | 147 456 |
|---|---|---|---|---|---|
| D64 | 22.6 / 39 | 29.7 / 58 | 47.9 / 95 | 98.8 / 306 | 227 / 646 |
| D128 | 24.1 / 46 | 39.6 / 71 | 71.0 / 170 | 190 / 483 | 516 / 1282 |
| D256 | 34.8 / 58 | 65.9 / 117 | 198 / 349 | 632 / 860 | 1767 / 2326 |
| D384 | 47.7 / 67 | 108 / 148 | 324 / 509 | 1227 / 2597 | 3568 / 7040 |
| D512 | 61.6 / 106 | 143 / 254 | 462 / 956 | 1934 / 3560 | 5600 / 9984 |

n = 2 (ours only): D64 21.9 / 27.9 / 41.9 / 75.2 / 160; D128 22.6 / 31.9 / 50.5 / 125 / 326; D256 29.4 / 47.6 / 126 / 395 / 1012;
D384 33.1 / 66.0 / 203 / 643 / 1858; D512 43.6 / 90.9 / 274 / 963 / 2839.

Found by this re-measurement: the first version of the ping-pong switches (`DECOUPLE`, `SKEW`) was compiled into the **training** forward too and made
D128 n = 4 training at 147 456 rows 639 µs instead of 520 (+23 %). They are now inference-only, and at run time only while a CTA has at most two
tiles (`tiles <= 2 x CTAs`): beyond that the coupling-free refill is 1 - 4 % slower (n = 4 49 152 rows 32.6 -> 33.8 µs), so large inference runs the
blocking hand-shake of the base kernel.

Error against float64 (y, n = 4, D128 .. 384, 3072 / 49 152 rows): ours 2.3e-3, anthropic 2.9 - 3.2e-3, PyTorch bf16 3.1e-3 (h goes from the fp32
accumulators into the squeeze). Against the PyTorch bf16 module anthropic `v2` is bit-identical (0.0) and ours differs by 3.8e-3 -- the rounding points.

What produced the small-row numbers (all measured):

| change | effect |
|---|---|
| D64 forward in 64-row tiles (`NWG = 1`, one consumer warpgroup, 2 CTAs / SM, `wide/d64_fwd_small.cu`, since replaced by `d64_fwd_small_f2.cu` / `_f3.cu` and removed), inference up to 20 480 rows | removes a +2.5 µs step at 96+ tiles: n = 2 12 288 rows 9.9 -> 7.3 µs, 3072 rows 7.1 -> 5.7; bit-identical output |
| squeeze-weight transpose cached by storage / version (`cuda/_weights.py`) | -1.7 µs kernel + gap per call (D64 768 rows 8.6 -> 6.8) |
| generic kernels below 3 072 .. 16 384 rows (`prefers_generic`, per D, mode and n) | D256 3072 rows 44 -> 18 µs, D384 77 -> 30, D512 117 -> 40 |
| weight-gradient GEMMs on a second stream in the training backward (`_weight_grads_aside`, rows <= 12 288 (D64) / 3 072 (D128 .. 384) / 768 (D512)) | 1.25 - 1.28x at D64 / 128 768 - 3072 rows, 1.03 - 1.13x D256 / 384; 1 - 3 % loss at 49 152+ rows so it is off there |

Tried and not kept: LayerNorm formed inside the expand GEMM (correct, 5 - 25 % slower: the per-CTA normalization is on the critical path;
removed after measurement); 2 CTAs / SM and 2-CTA clusters with B-tile multicast (no gain; the first cluster version was 2x slower from
`mbarrier.arrive.release.cluster` fences; removed); staggering the D64 weight-chunk order across CTAs (no change); `addmm` for the residual
(y error 2.3e-3 -> 2.6e-3); cuBLAS dh + elementwise SwiGLU backward (dx / dW error +7 - 10 %).

### D128 fused forward, 2026-10-15 (inference build: fragment LayerNorm, x tile first)

The 2026-10-14 D128 numbers of the fragment-LayerNorm fused build were measured on a kernel that was **wrong** (fused vs float64 7.9e-3 /
1.2e-2 at n = 2 / 4): ptxas allocated the loop-carried RS `wgmma` A operand to the swiglu temporaries of the rolled hidden-chunk loop.
Unrolling the chunk loop (`RSMASK` bit 8 -> `#pragma unroll`) fixes it; regression test `test_inference_matches_fp64_on_whole_persistent_tiles`
(n = 2 / 4, 128 / 2048 / 16 896 rows, fp64 reference, < 3e-3). Details: the record above.

Final numbers (µs, module path, CUDA-graph replay, default dispatch; anthropic best row: `lnl` n = 2, `v2` n = 4):

| n | rows | 768 | 3072 | 6144 | 9216 | 12 288 | 20 480 | 49 152 | 147 456 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | ours | 8.8 | 9.5 | 9.8 | 9.8 | 10.2 | 16.1 | 22.6 | 66 |
| 2 | anthropic | 9.4 | 9.5 | 10.0 | 12.5 | 12.8 | 20.7 | 33.9 | 100 |
| 4 | ours | 9.6 | 12.2 | 13.7 | 13.7 | 14.2 | 24.0 | 34.1 | 102 |
| 4 | anthropic | 14.1 | 14.3 | 15.0 | 18.9 | 19.4 | 34.4 | 53.7 | 156 |

n = 2 at 3072 rows is level with anthropic (9.5 / 9.5, run-to-run noise is +-0.1). Everything else is faster. float64 error 1.66e-3 at every
row count (generic 1.66e-3).

What moved: the x tile's TMA is issued before the weight chunks' (`XFIRST`): x arrives at 1 201 instead of 1 612 cycles and the LayerNorm ends at
2 393 instead of 3 181 (the weight loads no longer queue ahead of it), -0.3 µs at 768 .. 12 288 rows at both n. The grid launches only the CTAs that
have a tile (idle CTAs cost +1.3 - 2.5 µs); the 5-tensor-map prefetch; the unrolled loop also took 49 152 rows 27.1 -> 23.0.

**Warpgroup ping-pong (`DECOUPLE` + `SKEW`, default on, later the same day).** The two warpgroups were coupled by the weight-slot hand-shake
(thread 0 blocked until both had retired a slot), so they re-synchronised every chunk and both ran swiglu (MUFU-bound, tensor core idle) at the same
time. Now the second warpgroup to retire a slot issues the refill (shared-memory counter, nobody blocks) and warpgroup 1 starts its first GEMM after
warpgroup 0 has issued its own; the tensor core then runs one warpgroup's GEMM while the other does swiglu. n = 2, 3072 rows 9.8 -> 8.8 - 9.1 µs,
6144 10.2 -> 9.2, 12 288 10.7 -> 9.6; 49 152 rows unchanged. float64 error 1.66e-3 everywhere, 0 / 200 non-identical replays at n = 2 / 4 and
768 .. 49 152 rows. Per-call kernel-to-kernel time (50 calls per graph, so the ~1.1 µs graph-launch cost of the single-call harness is amortised),
ours / anthropic best row, µs:

| n | rows | 768 | 1 536 | 3 072 | 4 608 | 6 144 | 9 216 | 12 288 | 20 480 | 49 152 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | ours | 7.56 | 7.57 | 7.72 | 7.82 | 7.81 | 7.88 | 8.15 | 14.8 | 21.6 |
| 2 | anthropic `lnl` | 8.31 | 8.34 | 8.62 | 8.78 | 8.93 | 11.2 | 11.5 | 19.8 | 32.4 |
| 4 | ours | 8.10 | 8.53 | 10.86 | 11.94 | 12.04 | 11.92 | 12.30 | 22.7 | 34.0 |
| 4 | anthropic `v2` | 12.9 | 12.9 | 13.2 | 13.3 | 13.7 | 17.3 | 17.9 | 32.8 | 52.1 |

Margins: n = 2 7.4 % (1 536 rows) .. 29 %, n = 4 10 % .. 37 %. In the single-call-graph harness (graph launch included on both sides) n = 2 at
3 072 rows is 8.8 - 9.1 vs 9.5 µs, i.e. 4 - 7 %. The official `bench.py` sweep is L x L pair rows (>= 16 384), where the margins are 18 - 35 %.
Also measured and not kept this round: a CTA-rotated chunk order (`STAGGER`: slower at >= 4 608 rows), `cp.async` for tile 0's x rows (`CPX`: x
arrives 400 cycles earlier but the per-call time does not move), three weight slots with the decoupled refill (+0.1 .. +1 µs).

Timeline of one 128-row tile (n = 2, cycles, `TRACE` build): x 1 201 | LN 1 190 | per chunk: first GEMM ~1 000 .. 1 200, swiglu ~1 050, second
GEMM (hidden) | 4 chunks ~ 9 000 | epilogue + store ~1 400. The swiglu is issue / MUFU bound (8 192 sigmoids = 16 384 MUFU ops per chunk per CTA)
and runs with the tensor core idle; the two warpgroups run in lockstep.

Build switches of the inference forward are the constants of `_FWD_TUNING` in `kernels/transition/cuda/fused_sm90a.py` (the saving
training build turns the inference-only ones off, `_fwd_tuning_flags`): `LNFRAG` 1 fragment LayerNorm, `RSMASK` 256 (bit 8 = unrolled chunk
loop, required for correctness), `XFIRST` 1, `DECOUPLE` 1, `SKEW` 1; experiments, all 0: `RESIDENT`, `SPLIT2`, `PDL`, `NEWTON`, `WSLOTS` 2,
`GBSMEM`, `SUBCH`, `STAGGER`, `CPX`, `ABL` (timing ablation, wrong results), `TRACE` (clock64 timeline of CTA 0).

Tried for this chain and not kept (each measured, correct, same or slower; the switches stay in `_FWD_TUNING`, default off): all four weight chunks
resident (+1.0 µs: the four fetches share the bandwidth and chunk 0 lands late); 64-row tile with the hidden split over the two warpgroups and a
shared-memory reduction (`SPLIT2`: +0.8 µs at 3072, +2.2 at 6144 - 96 CTAs x 192 KB of weights is L2-bandwidth bound); PDL on the fused kernel
(no change); warpgroup skew so one swiglu overlaps the other's GEMM (no change at n = 2, +1 µs at large rows); the sigmoid reciprocal by three
Newton steps on the FMA pipe (+0.9 µs, 240 registers); a three-slot weight ring (+0.7); gamma / beta staged in shared memory (mixed, within 0.3);
software-pipelined 32-unit sub-chunks with `m64n64` RS GEMMs (`SUBCH`: +0.6 .. +2.4 µs - the n64 GEMMs run at about 65 % of the n128 rate).
Dispatch (`prefers_generic`, D128 inference): generic up to 512 rows at n = 2 (the ping-pong fused build is level with it at 768 rows and 0.3 µs
ahead per call in the single-call graph) and up to 4 096 rows at n = 4 (generic 10.9 / fused 11.8 µs at 3 072 rows, fused 11.8 / generic 12.2 at 4 608).

## Left

* The weight-gradient GEMMs are cuBLAS. A hand-written bf16 dW (MN-major wgmma operands) would let the gate kernel feed it without the dA / dB round
  trip; TF32 needs a different formulation.
* The generic path does not fuse the LayerNorm backward into the d_xn GEMM (the tuned D <= 256 kernels do). The forward fusion (LN in the expand GEMM)
  was built, measured slower below 12 288 rows (see above) and removed; at large row counts it was not better either (D384 49 152 rows training 1.095 vs 1.131 ms).
* Small rows are latency bound: D128 .. 512 at 3072 rows run 12 - 40 µs against a GEMM-time floor of about 6 - 25 µs (three launches with PDL; the D128
  fused kernel is now covered in the section above, its remaining cost is the swiglu not overlapping the tensor core). The
  D64 forward fixed the same shape with 64-row tiles; for D >= 128 each tile streams 393 KB+ of weights, so splitting the hidden dimension across a
  CTA cluster (DSMEM reduction of the squeeze partials) is the next structure to try.
* Training at 49 152+ rows is at 41 - 61 % of the GEMM time floor (epilogue not overlapped with the mainloop; ping-pong consumers untried).
* n = 2 at D = 384 / 512 is served by the generic kernels (the n = 2 builds of the two-kernel wide path exist - `WIDE_HM=2` - but measured equal to
  the generic path within 4 %, with non-deterministic dgamma / dbeta from the engine LayerNorm backward and slower for few rows, so `supported()` rejects them).
* No figures (`figures/*.svg`) yet and no autotune cache for the n = 2 schedules (`dev build` now measures them; the defaults above are measured by hand).
