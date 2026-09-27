# The Transition forward beyond D = 128 (2026-09-22)

H100, bf16, n = 4 (H = 4D), M = L^2 rows (pair-shaped), inference forward, µs, CUDA-graph replay medians.
Scratch: `MiniWorld/runs/transition_widths_20260922/`. Survey of every existing path: `survey_widths.py` (`survey-D*.json`).

| D | design | L384: engine -> ours | L768: engine -> ours | fp32 rel (ours / engine) |
|---|---|---|---|---|
| 64 | one fused kernel (`src/transition_fwd_d64.cu`, 2 CTAs/SM) | 122 -> 55 (2.2x) | 441 -> 176 (2.5x) | 2.27e-3 / 2.51e-3 |
| 128 | one fused kernel (shipped) | 271 -> 117 (2.3x) | 1027 -> 422 (2.4x) | 2.25e-3 / ~2.5e-3 |
| 256 | one fused kernel (`src/transition_fwd_d256.cu`) | 713 -> 382 (1.87x) | 2987 -> 1543 (1.94x) | 2.32e-3 / 2.58e-3 |
| 384 | Triton LN + `swiglu_gemm` + `squeeze_gemm` | 1476 -> 1014 (1.46x) | 5819 -> 4207 (1.38x) | 2.24e-3 / 2.48e-3 |
| 512 | Triton LN + `swiglu_gemm` + `squeeze_gemm` | 2371 -> 1641 (1.45x) | 9830 -> 6695 (1.47x) | 2.24e-3 / 2.48e-3 |

D = 256 also beats Anthropic's fused esm_t16 (397 / 1624 µs) by 4-5 %.

**Why the multiple falls with width.** Per row the op is 24 D^2 FLOP against ~16 D bytes of SwiGLU intermediate, so the
unfused path's intensity is ~1.5 D FLOP/B: memory-bound at D = 64 (engine 12 % of the tensor floor), compute-bound at
D = 512 (engine 40 %). Fusion wins big only where the intermediate's round trip matters.

**Why D >= 384 is two kernels.** A 64-row warpgroup holds its output accumulator: D/2 registers (192 at D = 384, 256 at
D = 512) -- over the register file. Splitting output columns across warpgroups halves the rows per CTA and doubles the weight
re-reads (~82 % of the measured ~11 TB/s L2 peak at D = 512). A 2-CTA cluster multicasting the weight slabs was built and is
18-30 % SLOWER: NCU shows the single-CTA GEMM already 81 % tensor-active with L2 at 64 %, and the cluster-scope release on
the cross-CTA stage arrive adds membar stalls (9 %). L2 was never the limit; that assumption came from arithmetic, not
measurement.

**swiglu_gemm** (`src/swiglu_gemm.cu`): 128x256 tile over the packed [Wa 128 | Wb 128] weight so a and b of a hidden block
share a tile; producer warpgroup (setmaxnreg 40 / 232), 4-stage TMA ring, SwiGLU from the fp32 accumulator in the epilogue.
The epilogue's transcendentals were 12-23 % of the kernel (ablation); `tanh.approx` sigmoid removes that and leaves rel vs
fp32 unchanged to four figures (1.659e-3), so it is the default for these new widths. 61-71 % of tensor peak; cuBLAS
writing the un-SwiGLU'd [M, 8D] reaches 56-62 %; the engine's Triton expand 35-41 %.
**squeeze_gemm** (`src/squeeze_gemm.cu`): out = h Ws^T + x, residual tile TMA-loaded into the output staging tile and added
in fragment layout. 41-62 % of peak vs cuBLAS addmm 37-58 %; tile width and ring depth move it by a few % only.

Not done: backward at any new width; wiring into the engine; LN fused into swiglu_gemm (the LN kernel is ~6 % at D = 512).

## Backward for D >= 256 (2026-09-22)

The D=128 two-role recompute design does not fit Hopper at D >= 256 (dW accumulators alone are 192 registers). The wide backward
is therefore unfused-intermediate: gate kernel -> dW GEMMs -> d_xn (+ LayerNorm backward). Engine = `_fused_bwd` through the
module (cuBLAS dh + Triton `_transition_expand_gatebwd` + cuBLAS dWs/dWab/d_xn + LN bwd + add).

**Gate stage, `src/gate_gemm2.cu`** (one kernel for dh = dy Ws, ab = xn [Wa;Wb]^T and the SwiGLU backward; writes h and
[dA|dB]). Best build `gg2_k32s4h_w0` = 128 rows x 128 hidden tile, TBK 32, 4 stages, epilogue in two 64-column passes,
early slab release (`-DTBK=32 -DNSTAGE=4 -DSTG_HALF=1 -DWAIT0=1`). Bench `bench_gate.py --hb 128 --tbk 32`.

| D (L384) | engine dh + gate | ours | x | cuBLAS same GEMMs + bytes, no gate math |
|---|---|---|---|---|
| 256 | 905 | 543 | 1.67 | 427 |
| 384 | 1640 | 1018 | 1.61 | 898 |
| 512 | 2647 | 1664 | 1.59 | 1511 |

What lost (all measured, D256/384/512 L384 unless noted): first version 128x64 tile 608/1134/1945; deeper ring at TBK 32
(6 stages) -5 % at D256 but +5-8 % at 384/512 -- 64-B swizzle (TBK 32) is consistently slower than 128-B at equal stage count;
2-CTA cluster multicast of xn/dy (rank 0 xn, rank 1 dy) 799/1600/2678 -- +30-40 %, same verdict as the forward; warpgroup
stagger 1-4 slabs +-3 %; 128x128 tile with 3 stages (full staging) 632/1196/1903, with TBK 64 x 2 stages 614/1273/2145.
Ablations on the 128x64 tile: no gate math 573, no stores 474, neither 451 -- the mainloop alone is ~52 % of peak.
NCU (128x64): top stall = consumer full-barrier wait 25 %, DRAM 45 %, L2 49 %, tensor 46 %. Depth of the ring (bytes in flight)
is the lever that moved it: 3 -> 4 stages of the 128x128 tile = -12 %.

**d_xn + LayerNorm backward, `src/dxn_lnbwd.cu`**: d_xn GEMM (K = 2H) with the LN backward, dgamma/dbeta partials and the dy
residual in the epilogue; d_xn never leaves the accumulator (fp32), so dgamma/dbeta error drops from ~2.3e-3 to ~2e-6 and dx
from 1.85e-3 to 1.66e-3 (vs fp32). COLS=2 (both warpgroups on 64 rows, D/2 columns each, row sums exchanged in smem).
D256 `dl_d256c2_w0` (`-DDW=256 -DCOLS=2 -DTBK=64 -DNSTAGE=4 -DWAIT0=1`) 386 us vs engine cuBLAS + LN + add 434 (1.12x).
D384 782 vs 761, D512 1702 (NW=256 spills: x prefetch 64 regs on 128 acc) vs 1141 -> NOT used there; at D512 cuBLAS runs
d_xn at 73 % of peak and a 64-row column-split tile (57 FLOP/B) cannot match it. COLS=1 (128 x 256) at D256 476 (spills).
NCU D256: DRAM 767 MB read (dAB alone 604) at 61 % of peak, tensor 53 % -> near the practical HBM limit.
Bench `bench_dxln.py`. Trap: `drv.Kernel` passes a Python int as int32 -- pass the TENSOR for pointer args, never data_ptr().

**Assembled backward** (`bench_bwd_wide.py`; gate_gemm2 -> cuBLAS dWs, dWab -> dxn_lnbwd at D256 / cuBLAS + engine LN
elsewhere; xn/rstd/c1 as the forward saves them). All six gradients within the engine's error vs fp32 (dx/dgamma/dbeta better
at D256):

| D | L384 engine bwd | ours | x | L768 engine | ours | x |
|---|---|---|---|---|---|---|
| 256 | 1813 | 1379 | 1.31 | 7194 | 5444 | 1.32 |
| 384 | 3445 | 2767 | 1.24 | 13650 | 10797 | 1.26 |
| 512 | 5498 | 4216 | 1.30 | 22200 | 18507 | 1.20 |

Where the rest is (D256 L384): gate 563, d_xn+LN 398, dWab 243, dWs 136. The floor is HBM traffic of the intermediates: h, dA,
dB = 906 MB written and ~1.5 GB read back (dWs reads h, dWab and d_xn each read dAB) ~= 720 us at 3.35 TB/s, against a FLOP
floor of ~630 us for the whole backward. Removing it needs dW accumulated inside the gate kernel (gate 96 + dWs 64 + dWab 128
accumulator registers per thread) -- does not fit sm_90 without TMEM. Nothing here is wired into the engine.

D128 on the same module-level method (`bench_mod_bwd.py`, fwd+bwd minus fwd; baseline = `transition_fused_sm90a=False`,
i.e. the Triton residual path the fused kernel replaced): L384 852.8 -> 482.0 us (1.77x), L768 3135.0 -> 1722.4 (1.82x).

**D64, re-measured at module level (2026-09-22)** (`bench_mod_d64.py`: our fwd `transition_fwd_d64x2` + bwd `tbwd_d64_r16` in an
autograd.Function vs `modules.Transition(64)`, both captured in CUDA graphs -- eager timing at L384 is CPU-launch-bound on
both sides and misleading): bwd L384 387.5 -> 170.2 us (2.28x), L768 1429.4 -> 617.5 (2.31x); fwd+bwd 2.25x / 2.35x. All six
grads equal the engine's error vs fp32. The earlier "5.5x" compared against a standalone `_fused_bwd(has_xn=True)` call, which
is NOT the path the module runs at D64 -- retracted. Harness trap found on the way: the module keeps the LayerNorm gamma/beta
fp32; rounding them to bf16 before the kernels shifts 29 % of xn by an ulp and makes every gradient ~20 % less accurate
(located with a one-hot dy, which makes a dWs row equal to one row of h).

## Backward, all widths, one method (2026-09-22): module bwd = (fwd+bwd) - fwd, both sides in CUDA graphs

| D | L384 engine -> ours (us) | x | L768 engine -> ours | x | ours = |
|---|---|---|---|---|---|
| 64 | 387.5 -> 170.2 | 2.28 | 1429.4 -> 617.5 | 2.31 | fused 1 kernel (+reduce), autograd.Function (bench_mod_d64.py) |
| 128 | 776.4 -> 414.1 | 1.88 | 2956.8 -> 1621.3 | 1.82 | fused 1 kernel, wired (engine with transition_fused_sm90a off vs on) |
| 256 | 1680.0 -> 1308.6 | 1.28 | 6734.7 -> 5463.2 | 1.23 | gate_gemm2 + fp32-out addmm dW + dxn_lnbwd (bench_bwd_chunk.py, chunk 0) |
| 384 | 3407.8 -> 2699.4 | 1.26 | 13373.5 -> 10894.6 | 1.23 | gate_gemm2 + fp32-out dW + cuBLAS d_xn + engine LN |
| 512 | 5243.4 -> 4296.5 | 1.22 | 20715.9 -> 17285.4 | 1.20 | same |
D>=256 rows use fp32-output dW GEMMs (`torch.mm(..., out_dtype=torch.float32)`): dWa/dWb/dWs error vs fp32 3.49/3.36/2.93e-3
against the engine's 3.86/3.75/3.36e-3, at no cost. For D>=256 "ours" is the op sequence on saved tensors, not an
autograd.Function (no engine wiring yet).

### D >= 256, round 2 (all measured, D256/384/512 L384 unless noted)
* Ping-pong gate (`gate_pp.cu`, each warpgroup its own 64 x 128 tile, order barrier): 691/1482/2436 vs 543/1028/1651 -- halves
  the weight-slab reuse; the unordered version crashes (slab order vs ring), same as the forward.
* 2-CTA multicast, found why it lost before: the peer release `mbarrier.arrive.release.cluster.shared::cluster` (no MEMBAR in the
  SASS, still) -> CUTLASS's default-scope `mbarrier.arrive.shared::cluster` took the old xn/dy multicast from 797 to 631 us
  (base 606). With that fix, multicasting the WEIGHT slabs of the 128 x 128 kernel (60 % of its L2 traffic) = parity (549/1010/
  1684, TBK 32; TBK 64 x 2 stages worse) -> L2 bytes are not the binding limit either (NCU: L2 64 %, DRAM 51 %).
* Cross-tile software-pipelined epilogue (`gate_sw.cu`: 2 x 96 accumulators on the 128 x 64 tile, the previous tile's gate
  epilogue a quarter per slab under the next tile's MMAs, D compiled in): 559/1055/1862 -- helps its own tile (606 -> 559) but
  stays behind the 128 x 128 kernel.
* L2-resident row chunking (`bench_bwd_chunk.py`: chunk the rows so one chunk's h + dAB, reused buffers, fit the 50 MB L2;
  dW accumulated by fp32 addmm): monotonically SLOWER as chunks shrink, e.g. D256 1309 (1 chunk) -> 1507 (4) -> 1733 (8) ->
  1995 (16, 57 MB) -> 2315 (32, 28 MB: fits L2). Wave tails and per-kernel prologues cost more than the HBM traffic saved.
* NCU of gg2_k32s4h_w0 (D256): stalls = wgmma completion wait 30 %, MUFU.TANH 10 %, full-barrier spin 9 %; tensor 51 %.
* The one lever that pays: NOT storing h (`-DNO_H=1`, h taken from the forward, which at D384/512 already writes it to HBM
  between swiglu_gemm and squeeze_gemm): gate 547 -> 468, 1045 -> 907, 1788 -> 1569 (-12..-15 %), ~-5 % of the backward.
  Costs activation memory M x H bf16 per layer (L384: 302 / 453 / 604 MB at D256/384/512); at D256 the fused forward would
  have to write h (+302 MB), so it only makes sense at D384/512. NOT adopted by default -- a memory/speed trade to decide.

## Wired into the engine (2026-09-22, uncommitted)

`kernels/transition/cuda/fused_wide_sm90a.py` + `kernels/transition/cuda/wide/` (see its README), dispatched from
`_residual_forward` after the D128 path, same `transition_fused_sm90a` setting, opt-out `MINIWORLD_TRANSITION_WIDE_SM90A=0`.
Forward: D64/D256 one fused kernel, D384/512 ln_swiglu_gemm + squeeze_gemm (path A). Backward: D64 fused; D>=256 gate kernel,
fp32-output dW GEMMs, then dxn_lnbwd (D256) or cuBLAS d_xn + engine LN bwd (D384/512).

`check_wide_wiring.py` through `modules.Transition(D, 4, implementation="triton")`, default settings, CUDA graphs (sustained
back-to-back replays), engine = same module with the wide path switched off:

| D | L | inference fwd | train fwd | fwd+bwd | bwd |
|---|---|---|---|---|---|
| 64 | 384 | 125.9 -> 54.9 (2.30x) | 126.6 -> 61.3 (2.07x) | 510.5 -> 231.5 (2.21x) | 383.9 -> 170.2 (2.26x) |
| 64 | 768 | 442.6 -> 170.6 (2.59x) | 442.7 -> 183.2 (2.42x) | 1867.8 -> 791.3 (2.36x) | 1425.1 -> 608.1 (2.34x) |
| 256 | 384 | 706.6 -> 429.3 (1.65x) | 706.6 -> 420.5 (1.68x) | 2447.8 -> 1804.2 (1.36x) | 1741.2 -> 1383.7 (1.26x) |
| 256 | 768 | 2901.8 -> 1590.2 (1.82x) | 2974.4 -> 1674.2 (1.78x) | 9776.2 -> 7664.4 (1.28x) | 6801.8 -> 5990.2 (1.14x) |
| 384 | 384 | 1403.3 -> 937.6 (1.50x) | 1457.8 -> 1002.8 (1.45x) | 4874.9 -> 3818.4 (1.28x) | 3417.2 -> 2815.7 (1.21x) |
| 384 | 768 | 5861.0 -> 3893.8 (1.51x) | 5929.3 -> 3943.2 (1.50x) | 19695.8 -> 15184.7 (1.30x) | 13766.5 -> 11241.5 (1.22x) |
| 512 | 384 | 2240.9 -> 1550.3 (1.45x) | 2360.4 -> 1617.6 (1.46x) | 7505.1 -> 6137.1 (1.22x) | 5144.7 -> 4519.5 (1.14x) |
| 512 | 768 | 9438.9 -> 6295.9 (1.50x) | 9400.4 -> 6485.2 (1.45x) | 31057.5 -> 23944.8 (1.30x) | 21657.1 -> 17459.6 (1.24x) |

Output and all six grads at or below the engine's error vs fp32 at every width (D256: dx/dgamma/dbeta better; output better
at every width); no_grad output bit-identical to the grad path; D64/D256 bit-reproducible (D384/512: the engine LN backward's
dgamma/dbeta atomics, same as the Triton path); torch.compile fwd+bwd bit-identical to eager (D64/256/512).
D256 L768 bwd: ONE replay after idle = 6530 us span with no gaps (kernels 6523); the sustained median is ~1 ms higher --
clocks under sustained load, not the wiring. Tests: `tests/numerics/test_transition_wide_sm90a_gpu.py` (29, + the 8 D128
ones: 37 passed); tests/compile: only the two known pre-existing failures, neither mentions the new ops.
Bugs found on the way: (1) the D64 forward's default FWD_SAVE=1 writes xn/rstd/c1 unconditionally -> the no_grad call wrote
through 1-element placeholders (illegal address at the next graph capture); (2) squeeze_gemm token-pastes SQ_BN, so it must
be a literal; (3) a default `Transition(D, 4)` resolves to the PyTorch backend -- a wiring check without
implementation="triton" compares PyTorch with PyTorch and "passes".

## Forward, round 2 for D >= 256 (2026-09-22) -- adopted in the engine build

A/B at FIXED clocks (`ncu_one.sh` = NCU `--clock-control base`, median of 3 launches, spread 2-12 us; graph wall-clock on
this node moved the SAME binary by +-10 %, e.g. ln_swiglu_gemm 571 vs 646 us in two runs -- useless for 3 % effects).
* ln_swiglu_gemm, SWP (software-pipelined epilogue: two 64-register accumulator sets, tile n's SwiGLU epilogue a quarter per
  slab under tile n+1's MMAs): D384 721 -> 678 us (-6 %), D512 1207 -> 1105 (-8.4 %); output unchanged (rel 2.252e-3).
  Early slab release alone: +4 % / -1.6 % (not adopted without SWP).
* squeeze_gemm, WAIT0: 340 -> 336 (D384), 543 -> 528 (D512). NCU D512: tensor 80 % active, 54 % of stalls on the MMA
  completion -> compute-bound, at its ceiling.
* transition_fwd_d256, PP (FA3-style ping-pong: each warpgroup issues [G2(j-1), G1(j)] as one batch, hands the pipe over by
  named barrier, computes SwiGLU(j) under the other's batch): 482.6 -> 469.6 (-2.7 %), output unchanged. NCU before: tensor
  65 % active, 25 % of stalls on the MMA wait of the G1 -> SwiGLU -> G2 chain.
Module after adoption (check_wide_wiring.py, graphs, L384 / L768): D256 inference fwd 1.77x / 1.94x (was 1.65 / 1.82), train
fwd 1.65 / 1.73; D384 train fwd 1.56 / 1.57 (was 1.45 / 1.50); D512 1.52 / 1.52 (was 1.46 / 1.45). fwd+bwd 1.38 / 1.26 (D256),
1.29 / 1.29 (D384), 1.21 / 1.28 (D512): the backward is ~3/4 of the step and is unchanged (its 1.10-1.30x rows move by run
noise). 37 GPU tests pass.

## Backward, round 3 for D >= 256 (2026-09-22) -- both negative, nothing adopted

Fixed clocks (NCU `--clock-control base`), L384.
* gate_gemm2 `EPI_DIRECT` (h / dA / dB stored straight from the fragments with st.global -- no staging, so the 48 KB go to the
  ring: 5 stages): D256 605.5 -> 1629-1656 us, D512 1744 -> 3463-3512, i.e. 2.0-2.7x SLOWER at every ring depth and TBK. Each
  warp store is 8 rows x 16 B; the write path does not absorb that. TMA-store staging is essential.
* dxn_lnbwd `XSM` (x TMA-loaded into shared memory once per tile, mid-mainloop, instead of the 64-register prefetch that made
  the 128-row tile spill): COLS=2 403.7 -> 437.0 us, COLS=1 (128 x 256, 3 stages) 444.4 -- +8-10 %. Output identical.
So the gate kernel's half-staged 4-stage ring and the 64-row dxn tile stay. Every structural backward lever in reach on sm_90
is now measured: what remains is the HBM traffic of h / dA / dB.

## Backward, round 4 (2026-09-22): what actually bounds dxn_lnbwd, and what did not help

Fixed clocks, D256 L384.
* Premise check for a co-scheduled dWab + d_xn kernel that would read dAB from HBM once (l2_premise.py): with dAB fully
  L2-resident (just rewritten) the per-row cost of dxn_lnbwd drops only 1.12-1.15x and of cuBLAS dWab 1.13-1.17x on small
  chunks, and not at all at full size (390.8 vs 391.1 / 264.3 vs 262.6 us). dAB's HBM traffic is not what bounds them -> not
  built (<= ~6 % of the backward).
* dxn_lnbwd 128-row tile via batched global x loads (XB, 16-register double buffer, no spill): 405.7 = the 64-row 406.0; with
  wait<1> 407.7; TBK 32 x 8 stages 428-429. Shared-memory operand bandwidth is not the limit either.
* Ablations: dx stores removed 401 -> 358; mainloop alone (epilogue removed) 240-257. The LN-backward epilogue is ~150 us =
  37 % of the kernel, latency-bound (its instruction count is ~2 us per tile, measured 8.6).
* dx staged with stmatrix + TMA store (STGDX) instead of st.global: 407.4 -> 401.3 (-1.5 %), identical output -> ADOPTED.
* Epilogue software-pipelined under the next tile's mainloop (`src/dxn_sw.cu`: two 64-register accumulator sets, ten
  epilogue steps, one per two slabs, x / dy prefetched a step ahead): correct, 398.4 -> 456.0 (+14 %). A step (~0.9 us) is
  longer than a slab's MMAs (~0.56 us for both warpgroups) and both warpgroups take theirs at the same slab, so the queue
  drains; finer steps need more prefetch registers than the 8-byte-spilling build has.

## Backward, round 5 (2026-09-22)

* dxn_lnbwd with dy also prefetched into registers at the tile start (like x, DYPRE): 396.9 -> 438.6 (+10 %) -- this mainloop
  is sensitive to extra live state; not adopted.
* gate_sw (128 x 64, pipelined epilogue) with TBK 32 and 5-6 stages: D256 629 -> 683-694, D512 1929 -> 2336-2370 -- the
  64-B-swizzle penalty again; gate_gemm2 (608 / 1707 at base clocks) stays.
* **Opt-in `MINIWORLD_TRANSITION_WIDE_SAVE_H=1` (D384/512, wired, default OFF):** the forward's h (materialized between
  ln_swiglu_gemm and squeeze_gemm anyway) is kept for the backward and the gate kernel skips its h store (`wide/gate_noh.cu`).
  Gate kernel at fixed clocks: D384 1141.8 -> 1007.2 (L384), 4507.7 -> 3943.6 (L768), -12 %; D512 1707.1 -> 1643.5, 6838.4
  -> 6448.3, -4..-6 %. About -4 % of the backward at D384, -1.5 % at D512. Costs M x 4D bf16 activation memory per layer
  (453 / 604 MB at L384). Module wall-clock A/B (graphs) moved -3..+6 % run to run -- below its noise, hence the kernel A/B.
  All outputs/grads identical to the default path except dWs in the last bits (h from a different GEMM tiling); tested.
  Trap for the next person: NCU `--nvtx-include` misses the backward (autograd runs it on its own thread), and counting
  kernels with torch.profiler to compute --launch-skip does not line up with NCU's launch count.
