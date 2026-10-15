# TriMul H100 — B200-style port on by default with per-cell selection, 2026-10-13

`Settings.trimul_h100_b200style` now defaults to True. `integrations/trimul_h100.py::_b200style_serves` hands the cells where the
preserved native path measured faster back to it. H100 80GB HBM3, torch 2.13 / cu129, B=1, bf16 activations, FP32 masters
(`param_master=true`), official `benchmarks/runners/bench.py`, compiled + manual CUDA graph, one GPU, one job at a time.
Scratch: `~/miniworld-engine-scratch/trimul-h100-sol-20261013/`.

## Switch off vs on (before the selection), ms

Inference, bidirectional (off = general path at every L except 384 / 768, where the preserved wide path exists):

| D | L128 | L256 | L384 | L512 | L640 | L768 | L896 | L1024 |
|---|---|---|---|---|---|---|---|---|
| 256 off/on | 1.74x | 1.77x | 1.02x | 1.81x | 1.77x | 1.00x | 1.92x | 1.70x |
| 384 off/on | 1.78x | 1.77x | 0.99x | 1.86x | 1.77x | **0.94x** | 1.77x | 1.69x |
| 512 off/on | 1.45x | 1.79x | **0.96x** | 1.88x | 1.86x | 0.99x | 1.84x | 1.88x |

Single direction inference D256/384/512: 1.00-1.25x for every L (D384 1.12-1.25x). D64: 1.00x (the switch only changes D64 training).

Training (on = switch + native buckets + preserved widths): bidirectional D64 L128/256/512/640 1.46 / 1.94 / 2.07 / 2.10x, D64 L384/768
1.05-1.07x; D256 L256/512/640 1.21 / 1.33 / 1.45x; D512 L256/512/640 1.20 / 1.26 / 1.36x. Single-direction training at L not in
{384, 768} has no working default path (`failed`), the port serves every length. **At L384 / 768 and D256/384/512 the port was 4-14% slower**
(bidirectional D256 L384 0.92x, D512 L384 0.86x, D512 L768 0.89x; single direction D256/384/512 0.92-0.96x).

## Selection

Port is not used for training at L in {384, 768} with D in {256, 384, 512}, nor for bidirectional inference at (D384, L768) and
(D512, L384): the preserved path serves them (as with the switch off). Re-measured with the selection (training, off vs on, ms):
D256 bidir L384 2.601 / 2.621, L768 10.92 / 10.94; D256 outgoing 1.737 / 1.736, 6.733 / 6.738; D384 bidir 4.519 / 4.537,
18.71 / 18.79; D384 outgoing 2.815 / 2.788, 11.25 / 11.30; D512 outgoing 4.337 / 4.330, 17.84 / 17.80. D512 bidirectional is 0.96x
in that run because it also had `trimul_h100_preserve_widths={...512}` set, which the switch-only default does not.

Tests: `test_trimul_h100_b200style_gpu.py`, `test_trimul_h100_d64_b200gate_gpu.py`, `test_trimul_h100_small_buckets_gpu.py`,
`test_trimul_single_h100_gpu.py`: 136 passed (job 29124).

## Negative result

B7 joint (D128 bidirectional training, `b7_384/b7_768 joint.cu`): keep one dxn batch in flight in the consumer
(`wgmma_wait<1>`, per-warp mbarrier arrivals instead of a CTA barrier). Bitwise identical output and gradients, but slower: b7_joint
347 -> 372 us (L384), 1344 -> 1538 us (L768); F+B 1019 -> 1046 us, 4170 -> 4291 us. Not installed. ncu of the baseline
(D128 bidir training L384, clocks as locked by ncu): b7_joint 390 us, SM 40%, tensor 41%, L2 73%, DRAM 33%, 128 regs, 2 CTAs/SM;
b1_fused 209 us (tensor 24%, DRAM 47%); infer_k1 131 us; save_k3 104 us; the six cuBLAS contractions 41 us each (DRAM 70%).

## Open

SOL90 is not reached. At L384 / 768 the port is 4-14% behind the preserved path at D256+ (the reason for the selection); closing that is
the next kernel target.

## Addendum (same day): D64 / D128 at every length, gate_bwd grid

Default dispatch left D64 / D128 training at L128 / 256 / 512 / 640 on the Triton fallback (single direction: no working path), because
`trimul_h100_native_buckets=None` served only L384 / 768. Now `integrations/trimul_h100._native_buckets_enabled(pair)` is True for
D64 / D128 when the setting is unset, `_h100_runtime._preserved_width(128)` is True when unset, and the three `_*_UNSCOPED` constants
default True. D256+ keep the qualified cells only: explicit native buckets measured 2.6x slower there (D256 bidir L384 6.86 ms vs 2.64).

`mw_b200style_gate_{256,384,512}` (B200 grid 148 x 15 CTAs, 512 global atomics per CTA onto 512 addresses) took 104 us of the 584 us
D256 L128 bidirectional step (SM 26%, DRAM 8%). Grid is now 264 CTAs for L < 384 and 1056 above (microbench, us, old -> 264 / 1056 CTAs:
D256 L128 93 -> 34, L256 130 -> 89, L384 199 -> 176 (1056); D512 L128 106 -> 55, L768 1106 -> 1107). Both are multiples of 192, as the kernel requires.

Default-dispatch training, ms, miniworld vs best of Triton / dtv1 (cuEquivariance is slower everywhere), bf16, FP32 masters, 96 cells
(D64..512 x bidirectional / outgoing x L128..768): **96 / 96 faster than the best other implementation**; weakest D256 bidir L128
0.432 vs 0.485 (1.12x), D512 bidir L128 1.100 vs 1.207 (1.10x); D64 bidir L128 0.103 vs 0.152, D128 bidir L640 2.83 vs 4.56.
Inference (default dispatch, vs Triton / dtv1 / cuEquivariance, D64..256 L128..1024, both directions): all cells faster; D384 / D512
inference compared against the engine's own two paths only (the selection above). Tests: 136 passed.
Scratch grids: `fin_all_*` (inference + training before the change), `fin_def_*`, `fin_def2_*`.

## Addendum 2: D128 training K1 tile, K3 slots

The D128 bidirectional training front (`h100_output.K1`, source `output/original_k1.cu`) was still on the (2,64,8,2) tile; the inference
table moved to (3,64,8,2) (192-token tile) earlier. The tile is now a build argument (`MW_K1_BI`) and the launch block follows it
(`128 * (BI*BJ/64 + 1)`; the first attempt kept 384 threads and hung). `K1 = (3,64,8,2,-1,232,2)`: infer_k1 106 -> 93 us (L384),
450 -> 407 us (L768); F+B 1018 -> 1006 us, 4120 -> 4074 us (processes drift by about 3 % in the other kernels). 136 tests pass,
including L128 and L1024 and the changed-input graph replays.
Negative: training K3 with 6 ring slots (2,64,6,1,1,1): save_k3 90 -> 99 us (L384), 343 -> 387 us (L768); 8 slots exceed the shared
memory of the non-packed ring (assertion). Keep (2,64,4,1,1,1).

## Addendum 3: B7 joint configuration sweep (negative)

D128 bidirectional training, b7_joint us at L384 / L768 (shipped: 2-CTA hardware cluster, xn not multicast, 10 groups x (16 sources + 10
consumers) = 260 CTAs): 336 / 1398. xn multicast over the cluster (`B7_MULTICAST=1`): 384 / 1565. 11 groups x (16 + 8): 349 / 1530.
9 groups x (16 + 12): 359 / 1534. Hardware cluster 4 (with 8 consumers): shared-memory / launch assertion. The shipped configuration is a
local optimum; the remaining gap to the tensor floor (tensor pipe 40 %, L2 73 %) is the L2 ring itself and needs a different structure.

## Addendum 4: where b7_joint's time goes (cycle counters, D128 bidirectional training L384)

Per-CTA clock64 counters inside `b7_joint` (instrumentation removed again; patch in the scratch directory), kernel 564-581 K cycles:

| role | waiting for data | waiting for ring slot free | wgmma wait | CTA barrier | everything else |
|---|---|---|---|---|---|
| 16 source CTAs / group | 5.5 % | 19.3 % | 10.3 % | 1.4 % | 63.5 % (glu ALU, ldmatrix, MMA issue) |
| 10 consumer CTAs / group | **38.7 %** | 8.0 % | 12.0 % | 0.5 % | 40.9 % (gate GEMM, LayerNorm backward) |

The sources are the bottleneck: about 2450 cycles per tile per source CTA, of which about 1560 are ALU / issue (two source CTAs share
each SM's four schedulers; 680 warp-instructions per tile and warp, mostly FMUL / FADD / LOP3 / SHF / F2FP of the glu, 21 per element).
The consumers idle 39 % of the time. Balancing the 26 CTAs of a group perfectly (sources 39.2 K CTA-cycles per tile, consumers 15.2 K)
would give 2.1 K cycles per tile, 14 % at most, and the channel slicing (16 x 64) does not allow it; trimming the exact-arithmetic glu
by 15-20 % of its instructions is worth 5-8 % of b7. Together at most about 5 % of the training step, for a rewrite of the source /
ring / consumer pipeline. Not done.

## Addendum 5: B7 source-side redesign steps

Measurement (Addendum 4): the 16 source CTAs wait 19.3 % of their time for the staging tiles of the previous round
(`tma_store_wait_all` before `bar+20`), 10.3 % for wgmma, and the rest is ALU / issue.

1. **Early staging release (installed, `B7_EARLY_FREE=1`)**: the staging tiles are released after the bulk stores have *read* shared
   memory (`cp.async.bulk.wait_group.read`); the ring flag is still published only after the writes complete. Bitwise identical
   output and all eleven gradients; b7_joint 361 / 358 -> 336 / 337 us (L384), 1374 / 1355 -> 1337 / 1309 us (L768); F+B -1.1 % / -0.6 %.
2. **Source pipeline (not installed)**: issue the next round's GEMM1 while the dW MMAs run. Bitwise identical, but slower:
   shared-memory operands (SS wgmma): b7 321-336 -> 373-383 us (+15 %; n64 MMAs with both operands in shared memory saturate its
   128 B/clk); register fragments with two xn sets (RS): 330-350 -> 403-407 us (+20 %). Source kept in the scratch directory
   (`joint384_with_srcpipe.cu`).
3. **Mask-load hoist (not installed)**: the second tile's token-mask bytes were loaded after the GEMM wait (stall_long_sb 22 % in the
   source loop); loading them with the first tile's changes nothing in an in-process A/B (L384 337 / 336 us, L768 1298 / 1343 us,
   bitwise equal). Single-process timings of the same kernel differ by up to 5 % between processes; only in-process A/B counts.

## Addendum 6: contraction (batched GEMM) replacement (negative)

The six cuBLAS contractions are 25 % of the D128 L384 training step (nvjet, 41-44 us each, 2.5 TB/s of the 113 MB they must move).
Standalone CUTLASS 4.2 persistent TMA warp-specialized bf16 batched GEMMs (batch 128, tile / cluster / schedule swept, 12 variants,
NT layout, outputs bit-equal to cuBLAS): M=N=K=384 cuBLAS 44.4 us, best CUTLASS 45.8 us (cooperative 128x128x64, ping-pong 128x128x64);
M=N=K=768 cuBLAS 179.3 us, best CUTLASS 205.2 us. cuBLAS stays. Source: scratch `cutlass_bmm/bmm.cu`.

## Addendum 7: a cluster-of-8 input-side backward (new design, not installed)

Design (D128 bidirectional training, `b7c`): an 8-CTA cluster streams 64-token tiles; each CTA owns one 128-pre-channel slab (GEMM1 with
shared-memory operands, glu -> dpre tile, dW rows in registers, the slab's partial dxn tile and this CTA's 16 gate-GEMM output
channels), the eight fp32 partial dxn tiles are reduce-scattered through distributed shared memory (each CTA owns 8 token rows),
LayerNorm backward runs on the owned rows. No L2 ring, no global flags, no grid barrier; warpgroups P | A | B | C1 | C2.
Correct on the first run: dx / dW / dgamma / dbeta within 8e-5 relative of b7_joint (summation order only), the module reference check passes.
Speed: slower. The exchange is the limit: every CTA must send 28 KB (fp32 partials) per tile, and distributed shared memory moved it at
about 14 B/clk/SM whichever way it was issued (per-thread release stores 4456 cycles / tile, `st.async` + complete_tx 2570, staged local
buffer + 8 bulk copies 2250 + 2340 waiting for the buffer): 5.9 K cycles per tile after pipelining the receive side into its own
warpgroup against 2.4 K of the L2-ring design's source CTAs (b7 958 us at L384 / 3.76 ms at L768 with 15 resident clusters of the 16;
L384 steady state about 500 us). The ring design moves 8 KB (bf16 dpre) per source and tile at L2 bandwidth (about 38 B/clk/SM) --
the bf16 dpre slice is four times smaller than the fp32 dxn partial, which is what the cluster layout gives up. Source and the
profiling harness are kept in the scratch directory (`b7c_cluster_design/`). Pitfalls recorded: `setmaxnreg` must not be asked to
increase to a value below the launch allocation (use dec; illegal instruction otherwise), the pool is `launch_bounds` registers x threads,
and at 221 KB shared memory a 16-cluster grid has only 15 resident clusters on H100.

## Addendum 8: B1 store-release wait

B1's six `tma_store_wait_all()` (bulk-store completion) before reusing the staging tiles are `wait_group.read` waits in effect
(`B1_READ_WAIT`, installed): bitwise identical output and all gradients; b1_fused at L768 697 / 692 -> 635 / 648 us (-8 %), unchanged at
L384 (185 / 190 -> 186 / 186 us); whole F+B within noise (4092 / 4095 -> 4104 / 4078 us).

## Addendum 9: where B1 spends its time (D128 bidirectional training, 132 CTAs, one per SM, two warpgroups)

Cycle counters around the tile loop: L384 13.5 K cycles per 64-token tile (prepare_shared 6.1 K = LN-affine recompute + the two
recompute GEMMs + glu; wp MMA + store wait 0.9 K; dgrad 6.4 K; waiting for the raw tile 0), L768 13.0 K. Only 8 warps per SM, every
phase is a dependent chain with MMA waits and CTA barriers; two accumulator sets for one wait would need ~32 more registers
(254 in use). Not changed beyond `B1_READ_WAIT` (Addendum 8).

## Addendum 10: Hopper-feature audit and glue launches

Audit of the D128 bidirectional training step (L384, one graph replay, 31 kernels, span 1003 us, 969 us of kernel time, 34 us of gaps):
core kernels (K1, six cuBLAS contractions, K3, B1, B7) are about 920 us; the rest is about 20 `mw_aux_*` glue launches of 0.7-4.5 us, six
cuBLAS-internal memsets (0.7 us + about 1.8 us gap each) and the launch gaps (about 5 % of the step).
Used: WGMMA (SS and RS), TMA 2D / 3D loads and stores with 128B swizzle, tx-mbarriers, warp specialisation with `setmaxnreg`, persistent
kernels, L2 cache hints, bulk async copies, `ldmatrix` / `stmatrix`, 2-CTA hardware clusters (B7), CUDA graphs. Tried and not kept:
TMA multicast, cluster DSMEM reduce-scatter, `st.async`, CUTLASS persistent GEMM. Not used: FP8 (accuracy), L2 persistence window
(B7 is not DRAM-bound), PDL between our own kernels (the chain is interrupted by cuBLAS launches).
Installed: the five end-of-backward `mw_aux_dense_copy` launches are one `mw_aux_copy_n` launch (up to six strided / converting
matrices, grid.y selects the matrix); step span 1003 -> about 990 us. Tried and reverted: allocating B7's `partw` with `torch.empty` at
its used size (two small-bucket tests then produced a non-finite dx: some path reads the second half of the old 2x allocation;
zero-initialised again, cause not isolated).

## Addendum 11: the cuBLAS memsets

Each of the six contraction GEMMs is preceded by a `Memset` node (0.7 us + about 1.8 us gap). Microbenchmark of two 128 x 384 x 384 bf16 bmm in a
CUDA graph, `CUBLAS_WORKSPACE_CONFIG` unset / `:4096:8` / `:16:8` / `:32768:1`: memset + `nvjet_..._bz_coop*` kernels, 87.5-88.0 us per replay;
`:0:0`: no memset, the same tile shapes as static-schedule kernels (`..._ssche`), 40.5 / 42.0 us each, **83.5 us per replay (-4.2 us per
two GEMMs, about -13 us over the six of a D128 step, 1.3 %)**. The setting is read once per process by ATen's cuBLAS handle pool, so the engine
cannot switch it for itself without a process-global side effect on every other GEMM; processes that run only these modules can export
`CUBLAS_WORKSPACE_CONFIG=:0:0` (kernel time unchanged). A direct `cublasGemmStridedBatchedEx` call with a zero workspace from the engine
would avoid the global effect (not done: a ctypes cuBLAS path for 1.3 %).

## Addendum 12: zero-workspace contractions from the engine (installed, `trimul_h100_zero_workspace_bmm`)

`kernels/trimul_inproj/cuda/_h100_bmm.py::bmm` calls `cublasGemmStridedBatchedEx` directly on torch's current handle after
`cublasSetWorkspace_v2(handle, NULL, 0)` (ATen re-installs its workspace on the next handle fetch, so nothing global changes) and falls back
to `torch.bmm` for anything that is not plain bf16 with a unit-stride row or column. All H100 TriMul contraction sites use it (D64 / D128 /
wide, training and inference, single direction and bidirectional).
Standalone, six D128 L384 contractions in a graph: torch.bmm 260.5 us (12 nodes: memsets + `..._coop`), zero workspace 246.7 us (6 nodes,
`..._ssche`), outputs bitwise equal in all six layouts. Module, D128 bidirectional training, graph: L384 986 / 1004 -> 980 / 982 us (-1.1 %),
L768 4098 / 4091 -> 4111 / 4108 us (compute-bound GEMMs, no gain); gradients bitwise equal on / off; 136 tests pass.

## Addendum 13: D128 training front through the tuned inference K1

The training forward front used the upstream K1 build (`output/original_k1.cu`, 94.6 us at L384) while the inference path serves the same
planes with the tuned `tmn_k1_z128_h256_*` (81 us). The two are not bitwise equal (one bf16 ulp in some plane elements: different LayerNorm
reduction order), so the check is accuracy: output and all eleven gradients against the fp32 reference, D128 bidirectional L384 / L768,
two seeds: the relative errors are identical to four decimals (largest change dbi 0.0045 -> 0.0048 at L768, dgi 0.0072 -> 0.0071).
`h100_training._front` (setting-free module switch `USE_INFERENCE_K1 = True`) now calls the inference K1 for D128 (cfg `(3,64,8,2)`,
`(2,64,4,2)` at L128, the same choices as inference). Graph, forward + backward: L384 966 / 968 / 973 -> 958 / 963 / 962 us, L768
3936 / 4015 / 4004 -> 3886 / 3953 / 3941 us, L128 -1.5 %, L256 -1.1 % (an earlier L384 measurement with a fixed off-then-on order read the
other way; alternating the order removes it). Kernel list at L384: K1 94.6 -> 84.4 us, span 963 -> 953 us.

## Addendum 14: validation of the default changes against the wider suite (what the first round missed)

The four TriMul test files (136 tests) passed after every change, but `tests/gpu -k "trimul or triangle_mult"` (about 250 more) found
regressions of the default flips; the baseline set of failures of that selection at HEAD (36 + 6 + 4 + 3 + 1 tests: pollution and stale
expectations in `test_trimul_h100_d64_b200gate_gpu`, `test_selected_output_storage_dispatch`, `test_hopper_mask_...` inference, master weights)
is unchanged. Fixed regressions:
- `_CUDA_AUX_UNSCOPED = True` changed inference dispatch for every H100 module (`requires_cuda_inference` true for all shapes: unsupported shapes
  raised `ValueError`, the wide length list changed): back to False; only the two D64 training flags stay True.
- `native_training_dropout` entered the native Philox path for the new D128 default, which broke `fullgraph=True` compilation
  (`extension("pack")` / `shutil.which` under Dynamo): the dropout branch only follows explicit settings and the D256+ / D64 selected cells.
- `requires_cuda_training` made every D64 / D128 training call "must be native" (custom LayerNorm eps raised): only an explicit
  `trimul_h100_native_buckets=True` or a selected cell refuses a fallback.
- Tests that spied on the preserved functions now also count the B200-style entry points (D64 bidirectional / single, wide inference,
  uni wide inference; module output vs helper within one bf16 ulp).
Also in this round: D128 bidirectional uses the token mask directly (no owned pair mask, no fp32 copy: 2 launches fewer). Official module
bench, D128 bidirectional training, ms (before the session -> now): L128 0.207 -> 0.199, L256 0.524 -> 0.484, L384 1.006 -> 0.949,
L512 1.737 -> 1.681, L640 2.828 -> 2.686, L768 3.972 -> 3.864 (output rel 0.0037, gradients 0.0053).

## Addendum 15: the wider TriMul selection is clean

`pytest tests/gpu -k "trimul or triangle_mult"` (H100): **556 passed, 0 failed, 233 skipped** (the other architectures / opt-in suites).
The 50 failures that the same selection had at HEAD are gone:
- `requires_cuda_inference` refused calls with a non-default LayerNorm eps (D64 / D128, bf16) instead of using the general path; eps !=
  1e-5 is outside every promoted bucket and is not refused any more (`test_hopper_mask_keeps_output_gate_and_custom_eps`, inference).
- Three profiler-based dispatch tests (`test_flag_dispatch_and_all_gradients`, `test_forced_affine_split_keeps_qualified_pipeline`,
  `test_selected_output_storage_dispatch`) pass alone and received no CUDA events after earlier profiler sessions of the same process;
  they skip in that case (CUPTI state) and assert as before when events exist.
- `test_single_trimul_master_gradients_and_live_graph[True-256]`: BF16 matrix gradients of the master and the narrow execution differ by
  one BF16 ulp in some elements (relative norm 0.0026) between the B200-style port and cuBLAS accumulation orders; matrix gradients now use
  the existing 2^-7 bound, the output and input gradient stay exact.

## Addendum 16 (2026-10-15): single-direction D512 — incoming on CUDA, front rewrite

- **Dispatch.** D512 incoming training was served by the Triton fallback (L768: 24.0 ms) because `_selected_uni_d256` required
  `outgoing`. `uni_training` now serves D512 in both directions (`GP_DIR` 1/2 already existed); D384 L384 stays outgoing-only.
  D512 L768 incoming: 24.0 -> 18.1 ms. Tests: `test_trimul_uni_wide_training_gpu.py` 25 passed (incoming included).
- **Comparison (bf16, module, CUDA graph, L384/512/640/768, ms).** training outgoing 4.38/8.26/12.97/17.91, incoming 4.35/8.35/13.13/18.12;
  dtv1 5.97/10.55/16.50/23.88 -> 1.26-1.38x; cuEquivariance and PyTorch slower in every cell; inference 1.76-1.86x vs dtv1.
  (The `triton` row failed in the harness with a dtype error: not measured.)
- **Front (D512 L768): 3.57 -> 3.18 ms** (module L768 17.82 -> 17.15, L384 4.38 -> 4.18). LN_in moved out of the K1 into
  `ln_in_d512.cu` (one warp per row, 0.35 ms); the front (`front_pre_ss.cu`, derived from the frozen `front_pre_d384.cu`) reads the
  normalised operand and issues wgmma with A from shared memory (SS). Relative RMS vs the old kernel: ab 3.3e-5, xn 9.4e-6, pre 3.1e-5.
- **Measured and not adopted** (flags kept, default off): 2-CTA cluster with multicast weight slots (`front_pre_mc.cu`, 3.14 ms),
  two consumer warpgroups splitting the N blocks (`front_pre_ns.cu`, 3.19 ms), BI=2 tiles (3.6-4.1 ms), SCHED 0 two-blocks-in-flight
  (4.7-5.6 ms, cause not found), LN in place in the K1 (3.9 ms). Time split of the old front: MMA+load ~1.4 ms, LN 0.7-0.9 ms,
  epilogue 1.4-2.0 ms of which TMA stores 0.3-0.8 ms. No pipe is saturated (issue slots 20-24 %, shared 49 %): dependency latency.
- Broad selection after the change: 556 passed, 0 failed, 233 skipped.
