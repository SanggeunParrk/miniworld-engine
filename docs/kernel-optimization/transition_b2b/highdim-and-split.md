# Transition b2b — high-dimension (d=256/512) + split-vs-fused investigation

Branch `b2b-cutlass-opt` (uncommitted). Continues the d=128 b2b win (v11–v14, 2.07x vs cute).
Goal: extend the hand-CUDA fused b2b forward past d=128, and evaluate a two-version dispatch
(**b2b fused** | **split** = expand+gate kernel → h, then cuBLAS squeeze) per d_hidden.
n=4 throughout, so K = d_hidden, ND = 4·d_hidden, D = d_hidden. bf16 in/out, fp32 accum.
All timings: H100, M = 131072 rows, CUDA-event, no L2 flush.

## Result summary (measured)

| d_hidden | b2b (fused) | cute expand | cuBLAS squeeze | cute-split (exp+sq) | hand-CUDA expand | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 128 | **112 µs** (2.07x vs cute) | 149 µs | 62 µs | 211 µs | 152 µs | **b2b wins** |
| 256 | **409 µs** (1.21x vs cute) | 341 µs | 120 µs | 461 µs | 430 µs | **b2b wins** |
| 512 | n/a (can't fit) | 929 µs | 382 µs | **1311 µs** | 1698 µs | **cute-split wins** |

All paths cos ≥ 0.99999 vs fp32 torch reference.

## b2b generalization: d=128 → d=256 (SHIPPED-QUALITY, wired to inference)
- **Key lever = D-inner tiling.** Naïve generalization used D-OUTER (loop D-tiles outside the ND
  reduction, recompute expand a,b,h per D-tile) → 683 µs = **0.72x, a LOSS** (expand is ~2/3 of FLOPs,
  done 2× → 1.67× work). Swapping to **D-inner** (compute expand ONCE per ND-chunk, feed G live
  `out_acc[WG_M, DN=128]` accumulators, one per output D-tile) → **406 µs = 1.21x**.
- d=256 config: 2 warpgroups, CTA_M=128, WG_M=64, BN=64 (expand atom `SM90_64x64x16_SS`), DN=128,
  2 accumulators, STAGES=1. 242 regs, no spill. d=128 unchanged (single D-tile ≡ old kernel).
- ncu d=256: compute 48%, DRAM 4%, top stall short_scoreboard 31% (STAGES=1, no TMA/MMA overlap).
  Pipelining (STAGES=2) doesn't fit smem at BN=64 D-inner (263KB > 227KiB); a partial-pipe attempt
  (double-buffer Wa/Wb, single Ws via cp.async) hit a cute-copy type error + TMA-swizzle mismatch and
  was dropped. 2WG already overlaps gate/MMA across warpgroups, so pipelining ROI was marginal — kept
  the clean 406 µs.
- Wired: `modules/transition/module.py::_inference_forward` routes `d_hidden in (128,256)` (+ n==4,
  bf16, M%128==0) → `cuda_transition_b2b`; verified e2e (module cos 1.0, d=128 129µs / d=256 442µs
  incl. stats+reshape overhead; both beat cute). d≥512 → cute.

## d=512 b2b: VERDICT = route to cute (fusion is hardware-limited; confirmed empirically)
Three structures tried, all lose:
1. **single-warpgroup CTA_M=64** (xn[128,512]=131KB too big for 2WG) — 3209 µs = 0.44x (half
   throughput, 6.25% occupancy).
2. **2WG BN=32** — won't compile: `Layout_K_SW128_Atom` swizzle needs the swizzled dim ≥64 bf16 elems.
3. **2WG BN=64 K-tiled** (xn resident in smem + Wa/Wb streamed in KT=128 K-slices, pipelined STAGES=2,
   G=2 D-grouping) — 3066 µs = 0.46x, cos 1.0. ncu: compute 24%, DRAM 1%, **registers 255 (spilling)**,
   occ 12.5%.
Ceiling analysis: even spill-free (≈48% compute like d=256) ≈1533 µs — STILL loses cute 1409 µs, because
d=512 FLOPs = 4× d=256 and cute's larger non-fused tiles amortize better, PLUS fusion forces G=2 expand
recompute (`out_acc[64,512]` = 256 fp32/thread spills otherwise). Fundamental: at K=D=512, smem cannot
co-hold xn + weights + accumulator. cute's split (non-fused tiled GEMM, streams h, never co-resident)
wins. Archived: `scratchpad_ncu/b2b_d512_ktile_slow_046x.cu`.

## Split path optimization attempt (per user: "optimize the two split kernels")
The split = `cute_transition_fused` forward = `transition_expand_swiglu_cute` (LN-folded gated
dual-GEMM → h) + `torch.matmul(h, Ws^T)` (cuBLAS squeeze).
- Per-kernel roofline (cute expand): d=128 23% SM / 34% HBM (latency/overhead-bound), d=256 41% SM,
  d=512 60% SM. **cuBLAS squeeze is already ~roofline**: d=128 81% HBM (memory-bound reading h — the
  irreducible floor; not fusing means you must read h), d=512 71% SM. → squeeze not worth touching.
- **Wrote a hand-CUDA `transition_expand_gate_fwd`** (= b2b kernel minus squeeze; writes h[M,ND] via the
  vectorized STG.128 swizzle epilogue; no squeeze accumulator → no d=512 spill). Configs: d128
  BN128/STAGES3, d256 BN128/STAGES2, d512 BN64/KT64/STAGES2. Correct (cos 1.0) but **SLOWER than cute at
  every d** (0.98x / 0.79x / 0.55x) — same occupancy wall (255/173 regs, 1 block/SM, 12.5% occ);
  compute only 33% at d=512 vs cute 60%. Beating cute's mature CUTLASS expand GEMM with a hand kernel
  hit the same limit as b2b d=512.

## Dispatch decision
- **b2b for d_hidden ≤ 256** (fusion beats BOTH the cute-split and the hand-CUDA-split — it avoids the h
  HBM round-trip; wins 113µs@128 / 409µs@256).
- **cute-split for d_hidden = 512** (1311 µs; best available). The hand-CUDA expand does not beat cute.
- To make the d=512 split faster, the target is the **cute expand kernel** (tile/pipeline tuning), not a
  hand-CUDA rewrite. Open (options: (A) ship as-is, (B) tune cute expand, (C) hunt a lever in the hand
  expand).

## Methodology lessons (this session's real time-sinks)
- **The repeated "30-min hangs" were NOT slow compilation.** ninja log shows the expand_gate `.cu`
  compiles in **46 s**. The hangs were (a) a genuine **runtime pipeline deadlock** in the new kernel
  (codex reset the TMA `PipelineState` per-nd while reusing the same mbarriers → phase mismatch → infinite
  wait; fixed by hoisting PipelineState out of the nd loop + flattened stage mapping), and (b) **stale
  torch build locks** left by `scancel`'d jobs, which make the next import block forever on the lock.
  Both present as "job runs to the -t limit with 0 output." Mis-diagnosing this as compile time wasted
  the most wall-clock.
- **Fast kernel-debug loop:** clear stale locks (`rm ~/.cache/torch_extensions/*/…/lock`) after any
  scancel; run a bounded smoke (`timeout 120 python -u eg_smoke.py`) under a short `srun -t` so a hang
  surfaces in minutes, not at the 50-min cap. Never wrap the first (uncached) build in a `timeout`
  smaller than the build, and never let `srun -t` cancel mid-build (that discards the cache → death
  spiral of rebuilds). Drop the cute JIT from micro-benches (it recompiles per shape, minutes each).

## Artifacts (scratchpad_ncu/)
- Kernels: `b2b_d256_win_406us.cu` (d128+d256 winner), `b2b_d512_ktile_slow_046x.cu` (archived loss).
- New: `src/.../cuda/transition_expand_gate_kernel.cu` (+ bindings) — correct, slower than cute.
- Verify/bench: `verify_b2b_highdim.py`, `verify_module_infer.py`, `split_profile.py`,
  `verify_expand_gate.py`, `eg_smoke.py`, `ncu_b2b_d256.py`, `ncu_b2b_d512.py`.
