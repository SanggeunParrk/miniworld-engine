# B7-B12 beyond C = 128

Scope: the fused TriMul input-side backward (B7-B12) developed here for c_pair = 128, carried to 256, 384 and 512.
Source: `front_wideC.cu` (one source, `MW_C` selects the width; `MW_C = 128` compiles to the qualified C = 128 kernel
byte for byte in structure and register count). Harness: `wide_plan.py`, `bench_front_wide.py`, `sweep_wide.py`.
L = 384 (M = 147456 rows) throughout, H100 80GB HBM3, bf16, 25% pair mask.

## What the width changes

At C = 128 one CTA holds the whole channel axis of a 64-row tile in registers (acc[32] per warpgroup), which is what lets
the LayerNorm backward -- whose row sums run over ALL channels -- stay fused into the same kernel. Wider C keeps that
property only by holding C/2 channels per warpgroup: acc[CNW][32], 64 registers at C = 256 and 128 at C = 512. That
crosses the 128-register line of `__launch_bounds__(256, 2)`, so the wide dx role runs one CTA per SM.

The dW role does not: its accumulator is a fixed 64 hidden x 128 channels, so it still fits two CTAs per SM. Because of
that the two roles are launched as separate kernels from C = 256 (`ROLE_ONLY`), each owning the whole grid, instead of
the single cooperative launch the C = 128 path uses.

## Accuracy

Both paths are bf16, so the tolerance is the fp32 result, not the Triton output (`reference_fp32`, L = 128 to keep the
fp32 intermediates in memory). Relative L2 against fp32, ours vs the Triton path being replaced:

| C | dx | dWL | dgamma | dbeta |
|---|----|-----|--------|-------|
| 128 | 2.4899e-3 / 2.4899e-3 | 2.3090e-3 / 2.3090e-3 | 2.7030e-3 / 2.7029e-3 | 2.6076e-3 / 2.6076e-3 |
| 256 | 2.5187e-3 / 2.5187e-3 | 2.3627e-3 / 2.3633e-3 | 2.4004e-3 / 2.3971e-3 | 2.2439e-3 / 2.2424e-3 |
| 384 | 2.5236e-3 / 2.5236e-3 | 2.4060e-3 / 2.4058e-3 | 2.4151e-3 / 2.4130e-3 | 2.2188e-3 / 2.2173e-3 |
| 512 | (same pattern) | | | |

The two implementations are indistinguishable at every width. The C = 128 fixed tolerances (dx 2e-5, dgamma 5e-6) do NOT
carry: they describe how close two bf16 paths sit at C = 128, and that distance grows with the number of accumulated
terms. Judging a wide width by them fails a correct kernel.

## Timing (L = 384, CUDA-graph replay, median of 600 samples)

| C | Triton/cuBLAS path | first generalisation | final | speedup |
|---|--------------------|----------------------|-------|---------|
| 128 | 617 us | 366 us (unchanged path) | - | **1.69x** |
| 256 | 1255 us | 1883 us | 1287 us | 0.98x |
| 384 | 2136 us | 3427 us | 2961 us | 0.72x |
| 512 | 3199 us | 7427 us | 4842 us | 0.66x |

Per role at C = 256 (each timed in its own process; capturing several graphs from one plan is unreliable here):
dW 563 us, dx 690 us.

Per role, split launches: C = 256 dW 567 us / dx 798 us; C = 512 dW 2122 us / dx 2470 us.

The C = 512 baseline moved from 3889 us to 3195 us partway through this work: the autotune cache build for the wide
shapes (job 16084) landed in between. The last two rows compare against the tuned baseline, which is the fair one.

## Why C = 256 stops at parity

Both kernels move about as many bytes as the path they replace, at about the same rate, so the result is parity:

| | bytes moved | achieved rate |
|---|---|---|
| this kernel (dW 3.0 GB + dx 2.4 GB) | 5.4 GB | 4.4 TB/s |
| Triton/cuBLAS (d_concat + two GEMMs + LN) | ~5.6 GB | 4.5 TB/s |

The dW kernel alone runs at **5.3 TB/s** -- the L2 roofline on this machine -- so it cannot be made faster without
moving less. Every way of moving less that fits the accumulator budget was measured and lost (below). A 1.68x result
would need ~1.7x fewer bytes than Triton, and the fused-LayerNorm decomposition cannot get there: fusing the LN forces
N = C in the dx GEMM, which caps the row block at M = 128, which fixes the weight re-reads at 8 KB per row.

The one lever that would change the byte count is **cluster multicast of the weight tiles** (4 CTAs sharing one L2 read
of the same tile): dx traffic 2.4 -> ~1.5 GB, which models to ~1.4x overall. It needs a multicast TMA helper and a
cluster launch, neither of which exists in the vendored launch module.

## What was tried, and what it did

| change | C = 256 | C = 512 |
|--------|---------|---------|
| straight generalisation, one cooperative kernel | 1883 us | 7427 us |
| four TMA issuers per tile group instead of one | 1917 us (no effect) | 7539 us (no effect) |
| **roles split into two kernels, dW at two CTAs per SM** | **1390 us** | **4728 us** |
| third weight ring slot + one wgmma group in flight | 1393 us (no effect) | (no shared memory) |
| gate contraction split into 64-wide slices, double buffered | 1643 us (worse) | 5683 us (worse) |
| dW job splitting chosen to minimise ceil(jobs/ctas)/splits | - | C = 384: 3186 -> 2974 us |
| **128-row blocks in the dx role** (one weight tile feeds 128 rows) | **1410 -> 1300 us** | spills at C >= 384 |
| epilogue in the h ring (gate double buffered, x and res in one load) + per-warpgroup weight rings | **1300 -> 1269 us** | - |
| dW channel span 2 (256 threads / 512 threads) | 617 / 608 us against 563 | - |
| GLU derivative in its own buffer (one barrier instead of two) | 692 vs 694 us: nothing | - |
| 128-row blocks at C = 384 (304 B of spill) | - | 2980 vs 2955 us: nothing |

Two negatives worth keeping: more TMA issuers do nothing here (the barrier counts NISSUE arrivals and each warp leader
issues its own tiles -- the 27 GB/s per SM was occupancy, not issue rate), and halving the gate contraction's K per
wgmma group doubles the number of `wgmma_wait<0>` stalls, which costs more than the smaller loads save.

## Two barrier bugs, and how they presented

Both cost hours, both were mine, and both were reported correctly by the tools before I believed them.

1. **A trailing comment swallowed the `allsync()` after `mbar_init`.** An edit appended `// ...` to the end of the init
   line, and the `allsync();` that followed on the same line became part of the comment. Warps then entered the ring
   before thread 0 had initialised the barriers. It showed up as an *intermittent* "unspecified launch failure" about
   one replay in a hundred, only under CUDA-graph replay, never in eager launches -- and it moved with any change that
   altered scheduling, which is what made the bisect lie: disabling the LayerNorm "fixed" it because the kernel got
   shorter, and "dx alone fails, dx + dW passes" was the same artefact.
2. **The per-warpgroup single-issuer barrier rule leaked into the dW kernel.** `mbar_init` gave barriers 2..GBAR an
   arrival count of 1 (right for the dx role's per-warpgroup rings), but the dW ring at depth 3 uses barrier 2 as well,
   where four issuers arrive. The barrier opened after the first of four TMA loads. Deterministic, and it made every
   channel-span experiment fail.

`compute-sanitizer --tool synccheck` said "Barrier error detected. Missing init" both times, and memcheck pointed at
`mbarrier.arrive.expect_tx`; racecheck's report against the in-place GLU was the false lead. The lesson: when a bisect
implicates a stage whose only distinction is that it makes the kernel longer, suspect initialisation, not that stage.

## NCU, split kernels, C = 512 (base clocks)

| kernel | duration | tensor active | SM throughput | L2 | DRAM | warps active |
|--------|----------|---------------|---------------|-----|------|--------------|
| dW (256 CTAs) | 2.60 ms | 31.2% | 58.0% | 39.9% | 25.6% | 23.3% |
| dx (132 CTAs) | 2.86 ms | 33.7% | 32.4% | 27.7% | 26.1% | 12.6% |

The dW kernel is now SM-pipe bound (the GLU derivative is recomputed once per channel chunk, and its shared-memory
traffic matches the TMA traffic). The dx kernel is still latency bound at one CTA per SM -- and it cannot have two,
because 206-214 KB of shared memory per CTA caps residency before the register file does.

## What the measurements said, in order

1. **Not a bandwidth or a compute wall to begin with.** NCU on the fused wide kernel (C = 256): DRAM 26.7%, L2 30.7%,
   compute 32.0%, achieved occupancy **12.5%** -- one CTA of 256 threads per SM, and NCU's own verdict "latency issues".
2. **Role ablation** (`ROLE_ONLY`, outputs partial, timing real) put 98% of the fused time in the dW role at C = 512:
   both 7133 us, dW alone 7003 us, dx alone 4542 us. Per dW CTA that is 189 MB of TMA traffic in 7.0 ms = 27 GB/s.
3. **Four TMA issuers instead of one made no difference** (1917 vs 1883 us at C = 256): the barrier counts NISSUE
   arrivals and each warp leader issues its own tile, and the number did not move. The 27 GB/s was not an issue-rate cap.
4. **Occupancy was the cap.** Splitting the roles into two kernels -- dW at 128 registers and two CTAs per SM over all
   132 SMs, dx at one CTA per SM over all 132 -- took C = 256 from 1883 to 1390 us and C = 512 from 7427 to 4728 us,
   with no change to the arithmetic.

## Why the C = 128 composition does not win at these widths

The fused LayerNorm forces the dx GEMM to carry all C channels in one CTA, so the row block stays at M = 64 and the
weight tiles (4 matrices, HS x C) are re-read for every 64 rows: 4.2 MB per row tile at C = 512, 9.7 GB over the tensor.
A standard GEMM tiling (M = 128/256 per CTA, N = 128 channels) reads them 2-4x less, which is exactly what the Triton and
cuBLAS path does -- and why it scales from 14% of peak at C = 128 to 34% at C = 512 while this kernel holds ~18%
(now ~28% after the split). At C = 128 the fusion wins because the GEMMs are small and that path is launch bound.

Current NCU at C = 256: dx 791 us with DRAM 46%, L2 48%, SM 35%, tensor 29%, warps active 12.5% (one CTA per SM),
barrier stalls 23%; dW 664 us with SM 57%, L2 53%, warps 23%. The dx role is still latency bound and the dW role is at
its memory roofline.

Remaining levers, in order of expected effect:
- **Cluster multicast of the weight tiles** -- the only one that changes the byte count enough to matter (~1.4x).
- **Warp specialisation in the dx role** (three warpgroups: one TMA producer at a low `setmaxnreg`, two consumers).
  This is the only way left to overlap loads with wgmma at one CTA per SM, and it is what the PWA forward kernels in
  this family needed for the same reason. Biggest expected win.
- **A bigger dW group tile** (64 hidden x 256 channels, acc[128] per warpgroup): halves the group count, so it halves
  both the redundant GLU work and the preactivation re-reads. Costs the dW role its second CTA per SM, so it has to be
  measured, not assumed.
- **Cluster multicast of the weight tiles** (four CTAs sharing one L2 read of the same weight tile). Needs a multicast
  TMA helper and a cluster launch, neither of which exists in the vendored launch module.
- Materialising d_concat once (what the Triton path does) instead of re-reading the preactivations per dW group.
