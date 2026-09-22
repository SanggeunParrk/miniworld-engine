You are a senior CUDA performance engineer.

Your task is to analyze, correct, and optimize the fused B1–B4 backward kernel
(`dual_b1b4`) of the MiniWorld bidirectional TriMul training path on H100, and to
beat the existing Triton/cuBLAS baseline by at least 1.7×.

## Project context (read before touching code)
- We adopted Anthropic's TriMul inference CUDA kernels as the base
  (csrc/tmn_kernels.cuh, tmn_ptx.cuh, common/tmn_math.cuh under
  third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5).
  Our contribution is the *training* kernels. Reuse their primitives (TMA
  descriptors, 128B swizzle, WGMMA m64n64k16 bf16→f32, mbarrier, named barriers,
  ldmatrix/stmatrix helpers, setmaxnreg) rather than reinventing them.
- The fusion boundary and the saved-tensor policy are FIXED: forward saves xn,
  norm (LN output), tri, mean, rs (rstd), gate, proj. B1–B4 consume those and
  produce exactly the outputs listed below. B5+ (B6, B9, …) stay in the Triton
  path and consume dg and dtri.
- All experiments live in
  /home/psk6950/MiniWorld/runs/anthropic_b1b4_pipeline_20260919/.
  Latest kernel: dual_dspref.cu (+ dual_primitives.cuh). Harness: dual.py,
  core.py. Numbers: dspref.log, dual.log. NCU: dual-profile.ncu-rep.
  Do not create `cutlass*` files. Use node02 only and only the GPUs already
  assigned to this task.

## Target environment
- GPU: NVIDIA H100 80GB HBM3, sm_90a, 132 SMs, 228 KB smem/SM, 3.35 TB/s HBM
  (≈3.0 TB/s sustained in practice).
- Toolchain: nvcc 12.8 (pixi env, /home/psk6950/miniworld-engine/.pixi/envs/default/bin/nvcc,
  needs `-ccbin` of a compatible host gcc or the activated env) or nvcc 12.9
  (/usr/local/cuda-12.9). Driver 575 ⇒ CUDA 12.9 max at runtime.
  Anthropic's shipped cubins were built with CUDA 13.0 and require an R580+
  driver, so we rebuild their csrc from source with 12.x. Every instruction
  used (TMA cp.async.bulk.tensor, WGMMA, setmaxnreg, mbarrier) is sm_90a
  CUDA 12.x; nvcc 12.8 and 12.9 produce byte-identical SASS for dual_dspref.cu
  (4,320 instructions, 255 registers, 0 spills). Do not spend time on toolkit
  upgrades. The gap is structural.
- Compiler command (as in dual.py):
    nvcc -std=c++17 -O3 -arch=sm_90a --cubin -lineinfo -Xptxas=-v
         -I<…/pkg/v5/csrc> -DROLE=0 -DUCOUNT=132 -DPART_ONLY=2
         dual.cu -o build/<sha>.cubin
  Loaded through the engine's BlockDriver; launched with
  cuLaunchCooperativeKernel, grid = UCOUNT (≤ 132), block = 256 (2 warpgroups),
  dynamic smem 231,424 B ⇒ 1 CTA/SM, 8 warps/SM. PART_ONLY=1 launches a
  separate `unified_reduce` (grid 194, block 256) instead of the in-kernel
  grid barrier + reduction.
- Input shapes: M = L×L rows, C = 128 gate/proj channels, H = 256 LN channels.
  L ∈ {384, 768} are the shapes that matter (M = 147,456 / 589,824).
  L = 64 (M = 4,096) is the small correctness case. Ignore L = 128.
  Row tile = 64; M % 64 == 0 is guaranteed. Tiles are distributed persistently
  over UCOUNT CTAs, so tiles-per-CTA is ragged (2,304/132 = 17.45,
  9,216/132 = 69.8): handle the tail and account for the imbalance.
- Data types: BF16 for dy, gate, proj, xn, norm, tri, ds, dg, dtri, dWg, dWp.
  FP32 for mean, rs, gamma, dγ, dβ, all dW partial accumulators and LN row
  statistics. WGMMA is bf16×bf16→f32.
- Layouts (row-major, contiguous):
    dy, gate, proj, xn, dg : [M, 128]
    ds                     : [L, 128]  dropout mask already scaled by 1/(1-p)
                                       or 0; row r uses ds[r mod L]
    norm                   : [M, 256]
    tri, dtri              : [256, M]  (channel-major)
    Wp                     : [128, 256]; kernel receives Wpᵀ contiguous [256, 128]
    mean, rs               : [M] fp32;  gamma, dγ, dβ : [256] fp32
    dWg : [128, 128]  dWp : [128, 256]
  TMA maps: 64×64 boxes, 128B swizzle, L2 promotion 128B; dtri is stored via a
  3-D map [M, 256, 1] with box [64, 16, 1].
- Numerical tolerance (relative L2 vs the Triton/cuBLAS baseline, dropout on):
  dg bit-exact (0.0); dtri ≤ 2e-5; dγ, dβ ≤ 5e-6; dWg, dWp ≤ 5e-4, at
  L = 384 and 768. Current: dg 0.0, dtri 7e-6–1.3e-5, dγ/dβ < 1e-6,
  dWg 0.9–2.7e-4, dWp 1.6–2.9e-4. Intermediates dp, dg, dnorm are rounded to
  BF16 at the same points as the Triton path (that is why dg is exact). Keep it.
- Current latency (H100, dropout p = 0.25, CUDA events, median):
    L = 384 : baseline 295 µs → dual_dspref 250.7 µs (1.18×)
    L = 768 : baseline 1,179 µs → dual_dspref 869 µs (1.36×)
  Whole backward gains only ~4 % / ~7 % because B5+ are unchanged.
- Reference implementation: core.baseline() =
  triton.bidirectional.gate_elem_bwd_ew (B1) + torch.mm(xnᵀ, dg) (B2) +
  triton.bidirectional._te_backward (B3a/B3b via cuBLAS, B4 via Triton LN bwd).
  Always compare against this exact call in the same process.
- Optimization objective: latency of B1–B4 as one launch.
  Target ≤ 173 µs at L = 384 and ≤ 693 µs at L = 768 (≥ 1.7× vs baseline).
  Secondary: whole-backward time. Never trade correctness or the save policy.

## The operation (B1–B4)
Per row r (0 ≤ r < M), C = 128, H = 256:
  B1  y  = dy[r] ⊙ ds[r mod L]
      dp = bf16( y ⊙ gate )                         → feeds B3a, B3b
      dg = bf16( y ⊙ proj ⊙ gate ⊙ (1 − gate) )     → OUTPUT (feeds B9)
  B2  dWg += xnᵀ · dg                (128×128, FP32 accumulate, BF16 out)
  B3a dnorm = bf16( dp · Wp )        ([128] × [128,256] → [256], stays on-chip)
  B3b dWp += dpᵀ · norm              (128×256, FP32 accumulate, BF16 out)
  B4  LayerNorm backward over H = 256 with saved mean/rs and gamma:
      xhat = (tri[:, r] − mean[r]) · rs[r];  h = dnorm ⊙ gamma
      dtri[:, r] = rs[r] · ( h − mean(h) − xhat · mean(h ⊙ xhat) ) → OUTPUT
      dγ += dnorm ⊙ xhat,  dβ += dnorm                              → OUTPUT
Outputs: dg [M,128] bf16, dtri [256,M] bf16, dWg, dWp bf16, dγ, dβ fp32.

## What is already known — start here, do not rediscover it
Roofline (compute it again and confirm): unavoidable HBM traffic per row is
≈ 2,056 B read (dy, gate, proj, xn 256 B each; norm, tri 512 B each; ds is
L2-resident) + 768 B written (dg 256 B, dtri 512 B) ≈ 2.8 KB/row.
  L = 384 : 416 MB → 124 µs at 3.35 TB/s (≈139 µs at 3.0 TB/s)
  L = 768 : 1.67 GB → 497 µs at 3.35 TB/s (≈555 µs at 3.0 TB/s)
FLOPs per row ≈ 2·128·128 + 2·2·128·256 ≈ 164 kFLOP → ~24 GFLOP at L = 384,
trivial for Tensor Cores. The kernel is memory-bound; the 1.7× target
(173 / 693 µs) requires ~72 % of peak HBM bandwidth end-to-end. Anthropic's
forward kernels reach that class of efficiency, so it is feasible but tight.

History of this kernel:
- dW was the first bottleneck: ~264 µs fused vs ~81 µs for the two cuBLAS
  GEMMs at L = 384. It is now computed by a warpgroup that shares the
  TMA-loaded tile with the dX/LN warpgroup.
- Register spills were eliminated (an earlier prototype spilled ~282 MB of local
  traffic). ptxas now reports 255 registers, 0 spills. Check `-Xptxas=-v` after
  every change; any spill is a regression.
- Last NCU: DRAM throughput ~32 % of peak, Tensor Core active ~6 %, one
  256-thread CTA per SM, WGMMA serialization and named-barrier waits visible.
  Bank conflicts were reduced but not eliminated.

SASS evidence (cuobjdump/nvdisasm with -lineinfo on dual_dspref.cu, sm_90a):
  4,320 instructions. HGMMA 28, UTMALDG 31, UTMASTG 4, LDS 63, STS 26,
  LDG 532, STG 202, LDL/STL 0.
  The streaming path is fine: ds reads are LDG.128 (3 instr, line 20) and dg
  writes are STG.128 (1 instr, tmn_ptx.cuh:42). The scalar global traffic is
  entirely the dW/LN partial machinery:
    - dual_dspref.cu:92  → 192 scalar 4-byte STG: each thread dumps its WGMMA
      accumulators (3 tiles × 16 × 4 floats) into partw[12, UCOUNT, 4096] fp32
      with scattered addresses (c, c+1 adjacent but not even vectorized to
      STG.64). Total 25.9 MB written per launch.
    - dual_dspref.cu:96  → grid-wide spin barrier (atomicAdd + __nanosleep) that
      makes every CTA wait for the slowest one (ragged tail: 18 vs 17 tiles at
      L = 384).
    - dual_dspref.cu:99–100 (and 109–110 in unified_reduce) → 528 scalar
      volatile LDG: the final reduction is unrolled UCOUNT = 132 times per
      output element, reading the 25.9 MB back with 4-byte volatile loads
      (volatile blocks vectorization and L1). Then 49,664 scalar STG for the
      bf16/fp32 outputs.
  That is ≈ 52 MB of extra HBM traffic plus a full grid barrier and a serial
  132-deep reduction: ≈ 15–20 µs, ~7 % of the L = 384 time and pure overhead
  compared with the roofline. It is a concrete, measurable target; fix it
  before anything speculative. Options to evaluate: (a) fewer, larger
  partials (e.g. reduce across a 2-CTA cluster via DSMEM before writing, or
  accumulate over more tiles per CTA so the partial is written once);
  (b) vectorized float4 partial stores/loads and a tree reduction instead of a
  132-deep serial sum; (c) atomicAdd-free two-level reduction owned by the
  last-arriving CTA per output tile (the pattern dual_primitives.cuh already
  documents) so no CTA spins; (d) keep the PART_ONLY=1 split-launch as the
  control and measure whether the fused reduction is even a win.

## Non-negotiable correctness requirements
- Preserve the exact operation semantics and the BF16 rounding points above.
- Support M = L² for any L ≥ 64 with M % 64 == 0, including the ragged
  tiles-per-CTA tail and UCOUNT not dividing the tile count.
- No out-of-bounds access on any TMA box, on the ds[r mod L] wrap, or on the
  channel-major dtri store.
- No data races or invalid synchronization: every mbarrier phase, named
  barrier, fence.proxy.async and wgmma fence/commit/wait must be justified in a
  comment. Counters must be back to zero after every launch so CUDA-Graph
  replay stays correct.
- Do not specialize for the benchmark values (L = 384/768, p = 0.25) and do
  not skip work when dropout = 0 unless the generic path is also kept.
- Keep the PART_ONLY=1 variant (kernel + separate reduce) compiling and correct
  as the simple, obviously-correct fallback.

## Optimization workflow
1. Restate the B1–B4 math, the tile/warpgroup ownership, the full smem map
   (offsets 0 … 231,424) and every assumption you rely on.
2. Inspect dual.py / core.py (tensor maps, workspace, cooperative launch) as
   well as the device code; confirm the host side is not on the critical path.
3. Re-profile dual_dspref.cu once (ncu, metrics in step 10) and split the
   measured time into: streaming B1–B4 body, partial dump, grid barrier wait,
   final reduction. Report each in µs at L = 384 and 768 before changing code.
4. Review, in this order, because this is where the time is:
   a. The dW partial/reduction path described above (52 MB, barrier, serial
      132-deep sum). Remove or shrink it first.
   b. Overlap: are TMA loads of tile t+1 fully hidden behind WGMMA + LN of
      tile t? The current design is single-buffer + prefetch; evaluate a 2-stage
      ring. Smem budget allows two stages of the 192 KB input set only if the
      64 KB Wp tiles move (re-fetched from L2 or shared across the CTA); quantify.
   c. Warp specialization: a producer warp for TMA plus consumer warpgroups
      for dX (B3a+B4) and dW (B2+B3b), with setmaxnreg rebalancing, instead of
      all-256-thread named barriers.
   d. dW accumulation: 128×128 + 128×256 FP32 per CTA is 192 KB of registers
      across 256 threads; check whether the accumulator tiling forces the
      WGMMA serialization NCU shows.
   e. B1 recomputation in the dW warpgroup vs writing dp/dg once to smem.
   f. dtri epilogue: channel-major [256, M] through the 3-D TMA map; measure
      the stmatrix transposition and bank conflicts.
   g. L2 policy: evict_first for streamed rows, evict_last for Wp and ds.
5. Propose the three most promising optimizations in priority order with
   expected µs saved at L = 384 and 768, risk, and which shapes they help.
6. Implement the best candidate as complete compilable code: a new
   dual_<name>.cu + dual_<name>.py in the run directory. Leave the previous
   version untouched for A/B.
7. Correctness tests (extend the check_*.py pattern): L = 64, 384, 768;
   dropout p = 0 and 0.25; randomized inputs with fixed seeds;
   UCOUNT ∈ {66, 132} to exercise different tail remainders; compare all six
   outputs against core.baseline() with the tolerances above; run twice under
   CUDA-Graph capture/replay with inputs changed between replays.
8. Benchmark: 20 warm-up iterations, CUDA events, sync, 200 timed iterations;
   report median and p90 for the kernel and for the baseline at L = 384 and
   768 with dropout on. Also report full-backward time via the engine path.
9. Exact commands for compilation (the nvcc line above with your -D flags),
   compute-sanitizer --tool memcheck / racecheck / synccheck at L = 64 and
   384, and `ncu --set full --import-source yes -k regex:dual_b1b4`.
10. NCU metrics that confirm or reject the hypothesis:
    dram__throughput.avg.pct_of_peak_sustained_elapsed (target > 70 %),
    dram__bytes_read.sum / dram__bytes_write.sum (compare with the 416 MB /
    1.67 GB roofline; anything above is the partial traffic),
    sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,
    smsp__average_warps_issue_stalled_barrier / _long_scoreboard / _membar /
    _wait / _sleeping (the last one is the spin barrier),
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,
    launch__registers_per_thread, local load/store bytes (must be 0),
    lts__t_bytes.sum for L2 traffic.
11. After each measurement change ONE major variable (reduction scheme, stage
    count, warp roles, accumulator tiling). Log each variant to <name>.log with
    CHECK and RESULT lines like the existing logs, and update b1b4-shared.svg
    and the trimul.html status page.

## Decision rules
- Correctness first; dg bit-exactness is a hard gate.
- No speedup claim without CUDA-event medians against core.baseline() in the
  same process, dropout on.
- Occupancy is 1 CTA/SM by design; do not chase occupancy by shrinking smem
  unless the roofline split says latency, not bandwidth, is the wall.
- Do not add registers or smem without stating the trade and showing ptxas
  output with zero spills.
- Keep the PART_ONLY=1 generic path whenever a specialized fast path is added.
- Prefer the Anthropic tmn_* primitives and plain CUDA C++; inline PTX only
  where they do not already provide the instruction and the gain is measured.
- The cuBLAS GEMMs (B2, B3a, B3b) at ~81 µs / L = 384 are the bar the fused dW
  path must clear; if a variant cannot, say so and evaluate un-fusing dW as a
  measured alternative.
- If the measured roofline says 1.7× is unreachable at L = 384, report the
  bound with numbers instead of forcing it, then maximize L = 768.
- If information is missing, list it before assuming.

## Required response format
1. Operation, smem/warpgroup ownership, assumptions
2. Correctness issues found in dual_dspref.cu
3. Time split and bottleneck hypothesis with roofline numbers (L = 384 / 768)
4. Ranked optimization plan (3 items, expected µs each)
5. Optimized complete code (new .cu + .py)
6. Correctness tests and results
7. Benchmark and profiler commands and measured results
8. Profiler signals observed vs expected
9. Remaining gap to 1.7×, risks, and the single next experiment

## Current implementation
Read directly from disk; do not paste from memory:
- /home/psk6950/MiniWorld/runs/anthropic_b1b4_pipeline_20260919/dual_dspref.cu
- /home/psk6950/MiniWorld/runs/anthropic_b1b4_pipeline_20260919/dual_primitives.cuh
- /home/psk6950/MiniWorld/runs/anthropic_b1b4_pipeline_20260919/dual.py, core.py
- Anthropic base headers: <…>/pkg/v5/csrc/tmn_kernels.cuh, tmn_ptx.cuh,
  common/tmn_math.cuh
