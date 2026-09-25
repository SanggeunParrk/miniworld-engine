# Token DiT: what is left beyond v6, measured

`probe.py` re-assembles the v6 step of `../token_dit_fused` (frozen at cf70909b, imported, not edited) and times
variants of it. H100, 24 blocks, S = 5, bf16, CUDA graph, per block:

| variant | L768 | L384 | what it tells |
|---|---:|---:|---|
| v6 step as packaged | 170.9 us | 86.8 us | |
| re-assembled here | 174.3 us | 88.4 us | the baseline below |
| both `resgate_adaln_rows` passes skipped | 148.8 us | 75.1 us | the most that moving AdaLN into a GEMM can save: 25.5 / 13.3 us, 15 % (numbers wrong) |
| attention skipped | 124.0 us | 73.7 us | the core is 50 / 15 us of the block (numbers wrong) |
| samples 3 + 2 on two streams | 177.1 us | 93.9 us | slower |
| samples 2 + 2 + 1 on three streams | 180.4 us | 99.8 us | slower |
| samples 1 x 5 on five streams | 202.3 us | 105.0 us | slower |

**Stream-level overlap loses.** The samples are fully independent through the token DiT, so splitting them over
streams lets one group's attention (softmax, ALU-bound) run beside another group's GEMMs (tensor-bound). It does run
concurrently, but every GEMM gets smaller (M = 1920 or less instead of 3840) and loses more efficiency than the overlap
wins back. This also lowers confidence in the "persistent per-block kernel" bound (~100-110 us): the overlap that
bound assumes is not free at these sizes.

**AdaLN-into-GEMM is worth at most 15 %.** Skipping both residual + AdaLN passes is the ceiling. A real version keeps
the residual update (fp32 read-modify-write of x in the Wo / squeeze epilogue) and the AdaLN transform (in the next
GEMM's A-operand load), so it saves the xa and y round trips -- 23.6 MB per half-block at L768, ~16 us a block -- not
the whole 25.5 us. It needs custom CUDA GEMMs that match cuBLAS / quack: Triton lost 3.6x on exactly this (v1).

## The residual GEMM with the AdaLN in its epilogue (`gra/`): built, correct, slower

`gra/gemm_resgate_adaln.cu` replaces `torch.mm` + `resgate_adaln_rows` for Wo and squeeze:
`x += sigmoid(gl) (A W^T)` in place (fp32), then `xa = LN(x) sigmoid(ms) + mb` (bf16). A row's 768 columns are split over a
cluster of 4 CTAs (192 each). Each CTA reduces its columns to (mean, M2), and the 4 partials are merged over DSMEM (Chan).
`y` never exists and the row pass's launch is gone. It is exact: x agrees with fp32 to 1e-6, against 1.7e-3 for the
row path, which rounds y to bf16. `gra/test_gra.py` covers L 128 / 384 / 768, K 768 / 1536, 1 or 2 warpgroups, and
the last half-block without AdaLN.

Kernel latency, H100, CUDA graph (`gra/bench_gra.py`):

| op | L768 Wo | L768 squeeze | L384 Wo | L384 squeeze |
|---|---:|---:|---:|---:|
| cuBLAS mm | 9.7 us | 15.1 us | 6.0 us | 8.9 us |
| mm + resgate_adaln_rows | **20.1** | **25.7** | **11.9** | **15.0** |
| gra v1 (plain loads after the mainloop) | 28.0 | 37.8 | 20.1 | 28.3 |
| gra v3 (b3b56d37), best of 1/2 warpgroups | 25.4 | 33.1 | 17.4 | 21.9 |
| gra v4 (st.async stats, 16-B AdaLN chunks) | 22.3 | 31.0 | 14.1 | 18.3 |

Per-CTA `%globaltimer` stamps (`gra/times.py`, build with `GRA_DEFS=GRA_TIMES`) show where v3's time goes. At L768 Wo:
mainloop 9.5 us (cuBLAS parity), x in smem +1 us, resgate + x store 3.1, row stats + cluster barrier 5.3, AdaLN + xa
store 3.6. How the mainloop got to parity:
- 2 stages were TMA-latency-bound: 1.3 us a chunk against 0.42 us of MMA. The x and gl tiles now alias the ring (the
  producer loads them as the last chunks free it), which leaves room for 4 stages.
- `mbarrier.arrive.release.cluster` for the cross-CTA slot release cost 4x. A plain `mbarrier.arrive.shared::cluster`
  fixes it.
- A multicast over the cluster gains nothing measurable, so L2 bandwidth is not the limit.
- Trap: a diagnostic build that drops the epilogue lets ptxas delete every HGMMA whose output is unused. Keep a fake use.

**Why it cannot win in this form.** The whole grid is one wave: 120 CTAs of 128 x 192 at L768. So the epilogue has
nothing to overlap with. It writes x + xa (17.7 MB, ~6 us of HBM) and runs latency-bound math on 8 warps, after the
mainloop. The baseline pays for y, but y (5.9 MB) sits in L2 between the mm and the row pass, so fusing saves little
more than a launch and a tail. The probe's "-rows" ceiling (25.5 us a block) also removes the x traffic, which no fused
version can. Reaching the ~11-14 us memory floor needs a persistent kernel over tiles of 64 rows. There, two consumer
warpgroups ping-pong: one tile's mainloop runs under the previous tile's epilogue, and each tile's row stats are
exchanged with per-tile DSMEM mbarriers. Estimated at best 15 us a block at L768 (9 %), less at L384, where there is
only one 64-row tile per CTA.

## Residual stream in bf16 (`residual_dtype.py`): rejected

Accuracy is measured against the IEEE fp32 PyTorch reference, with bench.py's seed and init. x is the dominant traffic of the row passes.

| residual | L768 | L384 | rel_rms |
|---|---:|---:|---:|
| fp32 (v7) | 178.3 us | 90.8 us | 4.1e-3 |
| bf16 | 170.0 us | 87.8 us | 1.1e-2 (the engine bf16 path's error) |

It gains 3-5 % and gives back the 2.5x accuracy margin over the engine that the fp32 residual buys.

### v4: every in-kernel stall removed, still behind

Fine stamps (`gra/times2.py`) found three more stalls, all fixed in v4:
- The `barrier.cluster.arrive.release` for the row stats took 2 us: a release waits for the thread's outstanding
  memory operations. The stats are now pushed into the four CTAs with `st.async ... mbarrier::complete_tx`, and the merge
  reads local shared memory. The only cluster barrier left is a relaxed teardown barrier.
- Issuing the ms / mb loads in the wgmma fragment's layout took 3 us: 4-B loads over 8 rows, 8 sectors per instruction,
  L1-wavefront bound. The AdaLN phase reads xn from shared memory, so its thread <-> column map is free; it now takes
  8-column chunks with 16-B loads (1.5 us, issued before the x store stream).
- Shared-memory loads serialized behind stores the compiler could not prove disjoint. They are now batched per 64-column block.

In-kernel timeline, L768 Wo: mainloop 8.7, x ready 10.0, resgate 12.5, stats 13.1, ms/mb issued 14.6, AdaLN 16.9,
end 17.9 us. The op still measures 22.3 against 20.1 for mm + rows. The rest is the tail: launch, plus the
17.7 MB of x + xa stores draining after the last CTA. In the 24-block step (`probe.py`) it is 185.2 vs 178.9 us per
block at L768, and 101.6 vs 89.4 at L384.

**Verdict.** In a one-wave kernel, memory time and compute time add. The mainloop is ~9 us of tensor work that touches
no HBM. The ~30 MB of epilogue traffic (x read + write, xa write) needs ~9 us at HBM speed, so the floor is ~18 us
against the baseline's 20. The baseline's two kernels have the same total traffic, less y, which is L2-resident. Only
overlapping one tile's epilogue with another's mainloop can beat that: a persistent ping-pong over 64-row tiles. It
re-reads W per tile, and L384 has one tile per CTA, so the estimate is ~3-4 % per block at L768 and nothing at L384.
Not pursued.

## Where the block's time actually goes (L768, in-step, per block)

`probe.py` variants, measured against the re-assembled step (175 us/block):

| part | in-step | note |
|---|---:|---|
| 4 GEMMs | 98.2 us | 507 TFLOP/s; standalone sum of the same kernels is ~83 |
| attention core | 50.5 us | 183 TFLOP/s |
| 2 row passes | 23.6 us | 71 MB, at the HBM floor |
| conditioning | ~3 us | amortised over the 24 blocks |

The GEMM gap to their standalone sum is NOT weight streaming (forcing all 24 blocks to share one L2-resident weight set
changes nothing: 99.6 vs 98.2) and NOT launch overhead alone (`boundary.py`: cycling the four GEMMs in block order costs
4 us more than timing each alone, ~1 us a kernel). It is the dependency chain: every kernel waits for the previous one,
so prologues and tails never overlap. (An earlier version of this line said quack's sm90 GEMMs have no PDL. That
is WRONG: `quack/gemm_sm90.py` takes `use_pdl=True` by default, waits before its TMA loads and triggers in its
epilogue. The chain is broken by OUR kernels between them -- see the PDL section at the end.)

## What helped: cluster_N in the GEMM configs (kept)

`gemm_sweep.py` sweeps quack's (tile_N, cluster_M, cluster_N, pingpong) against cuBLAS for the block's four GEMMs.
The packaged `_quack_mm` always passes cluster_N = 1, so A multicast over the cluster is never tried. At M = 3840 it wins:

| GEMM (L768) | cuBLAS | v7's candidates | best with cluster_N |
|---|---:|---:|---:|
| Wo | 9.37 us | 9.11 | **9.03** (tN192, cl 1x4, pp0) |
| squeeze | 15.14 us | 14.47 | **14.35** (tN192, cl 1x4, pp0) |
| qkvg | 32.59 us | 26.46 | 26.38 (tN192, cl 1x1, pp1) |
| expand+swiglu | - | 26.29 | 26.13 (tN192, cl 1x1, pp1) |

A wider sweep (`--wide`: tile_M 64 / 128 / 256) confirms tile_M 128 and adds cluster_M: q|k|v|g 25.84 us at
(192, cl 2x1, pp1) and Wo 8.91 at (192, cl 1x2, pp0). The candidate list carries the union, and `_pick_mm` races it
against cuBLAS per (M, N, K) on the first call, so nothing is hard-coded per shape.

Absolute step numbers move by +-5 us between runs when the node is shared, so the effect is measured A/B in one process
with the two paths interleaved (`ab_mm.py`, which flips `_mm_cfg` between the picked configs and cuBLAS):

| | cuBLAS for the plain GEMMs | picker | saving |
|---|---:|---:|---:|
| L768 | 178.66 us/block | 172.89 | **5.77 (3.2 %)** |
| L384 | 86.29 | 83.71 | **2.58 (3.0 %)** |

rel_rms is unchanged at 4.40e-3.

Two things had to be right before the picker could see this. Its candidate list has to contain tile_M 64 (Wo and squeeze
want it at both M), and **it has to time a graph replay, not eager launches**: quack's python wrapper costs more per call
than these 9-15 us kernels differ by, so eager timing picked cuBLAS for Wo and squeeze every time while the captured step
lost the difference. The change is in `tdit/runner.py`: `_pick_mm` races cuBLAS against PLAIN_CFGS per (M, N, K) on the
first call, the conditioning GEMMs go through the same picker, and GATED_CFGS carries cluster_N candidates too. At M = 1920 (L384) cuBLAS still wins Wo and squeeze and
v7's picks are already best, so the configs must stay per-M. This is a config-table change in `tdit/runner.py`
(PLAIN_CFGS / GATED_CFGS), which this session does not own.

## What did not help

- **L2 persisting window** (`l2p/`) on the residual x, or on x + xa + y: 176.7 and 182.4 vs 172.3 us/block at L768. The
  window reserves capacity the streaming tensors need; the row passes are at the HBM floor either way.
- **Sample batching in the core** (`core_sb/attn_sb.py`): the S samples share the bias tile exactly, and the ablation
  (`core_sb/bench_ab.py`) prices the bias at **11.3 us of the core's 50.0** at L768 (2.6 of 16.0 at L384; the gate is
  1.4). But a program that owns all S samples needs S accumulators: it spills, and the grid collapses from 960 CTAs to
  192, so it loses badly (88.8 us even with the samples un-batched at one per program). Groups of 2 or 3 lose too.
- **Triton clusters for the bias** (`num_ctas`): 5 is not a power of two and the grid dim must divide it; 2 and 4 fail to
  compile with a device-side TMA descriptor.

## The CUDA core (`core_cu/`): correct, and still 1.46x off Triton

`attn_core.cu` is the core as one sm_90a kernel: the S CTAs that share a bias tile form a cluster and the tile arrives by
TMA **multicast**, read once per block instead of once per sample. It matches the packaged core (2.3e-3 against an fp32
reference in the exp2 domain; 2.3e-5 against the Triton core itself at L768). Head dim 48 needs no padding (K = 48 is
three k-steps of 16), and P converts from the score accumulator to the PV A-operand by packing alone.

| variant (L768) | us |
|---|---:|
| packaged Triton core | **49.4** |
| CUDA, 64 query rows, no overlap | 72.1 |
| CUDA, 128 query rows | 84.9 |
| CUDA, softmax/PV software pipeline | 163.3 |

The pipeline loses because the extra score buffer costs registers, and with a dynamic buffer index ptxas injects a
`warpgroup.wait` that serialises the async matmuls (C7514); naming the buffers statically recovers part of it (113 us)
but not enough. Beating a tuned Triton core needs real warp specialisation and ping-pong scheduling across warpgroups,
which is a much larger build than the ~11 us the bias multicast can return. Mechanics worth keeping (`core_cu/qk_probe.cu`
validates them): wgmma flags (0,0) already read B as [N][K], so QK^T needs no transpose; PV needs the MN-major
descriptor (base + ks*2048) with trans-b = 1; `__nv_bfloat162_raw(...).x` is only the low half, which silently halves P.


## The measurement, corrected (and what it changed)

The engine benchmarks kernels with triton's `do_bench`, which zeroes an L2-sized buffer before every timed iteration.
Everything above this section was first measured by replaying a CUDA graph, which leaves the operands hot in L2 --
the regime that most flatters whichever variant re-reads the most. `bench.py` is now the one timing routine for
kernel-level work here. Re-measured that way, the fused residual GEMM is not as far behind as it looked (L768 Wo 33.2
vs 34.7 us for mm + rows, i.e. ahead; squeeze 41.2 vs 40.2, behind), though the step A/B still says it loses by ~10 us
a block -- in the step the intermediate the fusion removes is L2-resident anyway.

For **choosing** between implementations the regime matters the other way. These GEMMs run captured, back to back,
on data the previous kernel just left in L2, and timing candidates that way beats do_bench at predicting the step:
7.20 vs 5.17 us a block saved at L768, 3.46 vs 1.92 at L384. So `_pick_mm` times a graph replay, and `bench.py` uses
do_bench; both are recorded in the code.

## Softmax without the online max (kept, in the packaged core)

Writing the CUDA core made the redundancy obvious: softmax is shift-invariant, so the online running max exists only
to keep `exp2` in range. One fixed offset per row, taken from the first key block, does the same job, and then no key
block rescales its sums or its accumulator. It is the same number, not an approximation.

| | with the running max | fixed offset |
|---|---:|---:|
| CUDA core, L768 (do_bench) | 79.7 us | **68.4** |
| Triton core, L768 (do_bench) | 58.1 | **52.9** |
| step, L768 (A/B in one process) | 171.4 us/block | **163.1** |
| step, L384 | 81.8 | 82.3 (no change) |

rel_rms against the IEEE fp32 reference is unchanged: 4.40e-3 at L768, 4.42e-3 at L384. The offset is floored at -60
so a row whose first key block is entirely masked cannot leave an offset that overflows later blocks.

## The CUDA core, where it stands

`tdit/cuda_core/attn_core.cu` (moved into the package) is correct at every shape and carries the one thing Triton cannot express here -- the bias tile
TMA-multicast to the S CTAs that share it -- and it is still **68.0 us against the Triton core's 52.8** at L768
(do_bench). The variants, all measured: 128 query rows 84.9; a software pipeline over two score buffers 113 (163 with a
dynamic buffer index, which makes ptxas inject a warpgroup.wait); two blocks in flight 82.2; softmax interleaved with
PV per k-step 70.4; a single producer warp instead of a warpgroup 68.0 (the best). Every attempt at overlap costs
registers, and at 2 CTAs per SM (the only thing keeping this latency-bound kernel fed) there are none to spare. Beating
a tuned Triton core needs the full FA3 arrangement -- two consumer warpgroups ping-ponging phase by phase at 1 CTA per
SM -- which is a larger build than what it would return. The algorithmic half of the work already shipped: the fixed
softmax offset above came out of this kernel and now runs in the Triton core.


## The CUDA core, second pass: it now wins (core_cu/)

Rebuilt against `bench.py` (do_bench) and a measured floor, not a guessed one. `roof/l2_roof.cu` pulls L2-resident
tiles through TMA at this core's own tile shape and gets **6.75 TB/s**; the kernel moves 248 MB a block in 46.8 us,
which is 5.3 TB/s, **78 % of that roof**.

| | Triton core | CUDA core |
|---|---:|---:|
| L768, do_bench | 52.8-59.7 us | **46.7** |
| L384 | 22.0 | **20.7** |
| step, L768 (A/B in one process) | 166.0 us/block | **157.2 (+8.80, 5.3 %)** |
| step, L384 | 83.6 | **81.7 (+1.93, 2.3 %)** |

rel_rms against the fp32 reference: 4.52e-3 for the CUDA core against 4.40e-3 for the Triton one at L768 (the CUDA
core drops the softmax offset entirely rather than taking it from the first key block, so p carries larger magnitudes
into bf16); identical at L384.

What got it from 79.7 to 46.7 us, in the order it happened, each measured:

| change | us |
|---|---:|
| start (running max, 64-row tiles, one warpgroup) | 79.7 |
| softmax without the running max | 68.4 |
| 128 query rows, 2 consumer warpgroups, 2 CTAs a SM (18 warps) | 54.7 |
| q tile in registers (ldmatrix -> wgmma A operand), which buys a third ring stage | 53.2 |
| bias read with ldmatrix, seeded into the score accumulator | 49.1 |
| one producer warp; sums reduced once in the epilogue | 48.6 |
| sample as the fastest grid dimension (the S CTAs sharing a bias tile launch together) | 46.7 |

And what lost, each left in the defines with its number: the bias TMA-multicast over a 5-CTA cluster, with a fixed
issuer (53.5) and with the issuer rotating per key block (53.4) -- it moves 95 MB a block and costs more in placement;
256-row tiles (52.0); a per-slot wgmma descriptor cache (52.2, the array spills); the denominator from the tensor core
via a tile of ones (58.6, four extra wgmma beat 32 FADDs); softmax interleaved with PV per k-step (53.5); warpgroup
ping-pong (89.5 at the time); three TMA issuers rather than one (no change); more stages or more CTAs a SM (no change).

The profile says what is left is instruction issue, not bandwidth: 12.0 M instructions at IPC 1.6, ALU 3.3 M and
FMA 3.4 M against 1.6 M of ex2, and turning the multicast on and off moves L2 traffic 257 <-> 162 MB without moving
the clock. Closing 78 % -> 90 % means either cutting instructions further or cutting traffic without a cluster.


## Wired: the step runs the CUDA core

It lives in the package now (`tdit/cuda_core`) and `step()` picks it wherever the shape fits -- bf16, d 768, 16 heads,
L a multiple of 128 -- falling back to the Triton core otherwise (including a machine with no nvcc). `core="gated2"`
or `"cuda"` pins one for an A/B.

| | Triton core | CUDA core |
|---|---:|---:|
| step, L768 | 166.3 us/block | **158.0 (+8.36, 5.0 %)** |
| step, L384 | 84.7 | **83.0 (+1.73, 2.0 %)** |

rel_rms 4.42e-3 against the Triton core's 4.40e-3.

**The softmax offset is a define (MOFF), and it is off.** Subtracting a per-row offset taken from the first key block
is mathematically a no-op and only buys exp2 headroom, but it costs ~8 us a block in the step whichever way it is
written -- a subtract per element (the obvious way), or folded into the bias seed so the steady-state blocks pay
nothing (the register pressure lands somewhere else instead). Without it rel_rms is 4.42-4.52e-3 rather than 4.40e-3,
still 2.4x better than the engine's bf16 path, and a logit would have to pass ~128 in the exp2 domain to overflow.

## Every kernel in the block, against a measured roof (`sol_table.py`, `ncu_step.py`)

Roofs measured in the same process on the same node: a large bf16 cuBLAS GEMM **716 TFLOP/s**, an HBM read+write
stream **2.97 TB/s**, `roof/l2_roof.cu`'s TMA read of L2-resident tiles at the core's tile shape **6.68 TB/s**.
Latency is `bench.py` (do_bench, L2 evicted) at L768, S = 5, bf16; the floor is the roofline max of the kernel's
compute and memory time. NCU columns are from `run_ncu.sh` (one block, cache-flushed between kernels, so its
durations run ~20 % long against the captured step).

| kernel | us | floor | SoL | bound | NCU SM% / DRAM% / L2% |
|---|---:|---:|---:|---|---|
| q\|k\|v\|g GEMM (quack) | 32.77 | 25.3 | **77 %** | compute | 67 / 13 / 52 |
| attention core (CUDA, sm_90a) | 46.88 | 37.0 | **79 %** | L2 bandwidth | 36 / 30 / 60 |
| Wo GEMM (quack) | 16.03 | 6.3 | 39 % | neither | 40 / 15 / 33 |
| resgate + AdaLN rows (x2) | 20.64 | 11.9 | 58 % standalone, **101 % in step** | memory | 48 / 38 / 64 |
| expand + SwiGLU GEMM (quack `gemm_act`) | 33.25 | 25.3 | **76 %** | compute | 57 / 8 / 37 |
| squeeze GEMM (cuBLAS) | 20.93 | 12.7 | 60 % | compute | 58 / 20 / 30 |

The core's floor is its own traffic, not a FLOP count: NCU measures **247 MB of L2 reads a block**, which at 46.88 us
is 5.27 TB/s against the 6.68 TB/s TMA roof. Its tensor side is only 193 TFLOP/s (27 % of the GEMM roof), so the
tensor pipe is not what holds it.

The row passes are the one place where the standalone number misleads. In the captured step the two passes move
70.8 MB in 23.6 us = 3.00 TB/s, **at the HBM roof**; do_bench evicts `y`, which in the step is L2-resident between the
GEMM that writes it and the pass that reads it.

Wo is the only kernel far from both roofs: M = 3840, N = K = 768 is one wave of 120 CTAs with a short k-loop, so
prologue and epilogue are most of it. It is 16.03 us standalone against a 6.3 us floor, but ~9 us in the captured
step, where its operands are already hot.

## Where the block's 155 us go now (`token_dit_fused/prof.py`, L768, 24 blocks, bf16)

Re-measured with the GEMM picker, the fixed-offset softmax and the CUDA core all in. `bench.py` in the same session:
**557.6 us/block for the engine, 233.7 for Anthropic's parts with the pair bias hoisted, 155.3 here** (3.59x / 1.50x);
at L384, 272.7 / 133.2 / 81.2 (3.36x / 1.64x). Per-sample hoist 263.1 us against Anthropic's 1524.3.

| | us/block | share |
|---|---:|---:|
| plain GEMMs: q\|k\|v\|g, Wo, squeeze (+ the 2 conditioning GEMMs, amortised) | 63.4 | 39 % |
| attention core | 40.8 | 25 % |
| expand + SwiGLU GEMM | 29.1 | 18 % |
| resgate + AdaLN rows x2 | 25.8 | 16 % |
| the rest | 1.6 | 1 % |
| **GEMMs together** | **92.5** | **58 %** |

The core is 40.8 us in the step against 46.9 standalone: the S samples' bias tiles overlap in L2 across the block.

**The bottleneck is the GEMMs, and inside them the dependency chain.** Each is at 60-77 % of the measured 716 TFLOP/s
cuBLAS roof, which is itself 72 % of the theoretical peak, so kernel-for-kernel there is little left. But the four
standalone sum to ~83 us and cost 98.2 in the step (`probe.py`), and that gap is neither weight streaming (pinning one
L2-resident weight set: 99.6 vs 98.2) nor launch overhead alone (`boundary.py`: ~1 us a kernel). It is that every
kernel waits for the previous one, so prologues and tails never overlap. That
~15 us a block is the largest single lever left, and it needs either PDL between the GEMMs or one persistent kernel
per block. After it, the core's instruction count (79 % of its traffic roof, issue-bound at IPC 1.6) is worth ~10 us.
The row passes are already at 100 % of the HBM roof in the step and have nothing left.

## PDL through the block, and a race it uncovered in the CUDA core

**PDL.** quack's sm90 GEMMs wait before their TMA loads and trigger in their epilogue (`use_pdl=True`, the default), so
the block's dependency chain was broken only by our kernels between them. The CUDA core (`griddepcontrol.wait` after its
prologue, `launch_dependents` at the end, launched with `ProgrammaticStreamSerialization`) and the resgate + AdaLN row
pass (`gdc_wait` / `gdc_launch_dependents`, `launch_pdl=True`) now close it. Triggering early is safe whatever follows:
a kernel launched without the attribute (cuBLAS, torch) still waits for completion. `ab_pdl.py`, in one process,
interleaved, outputs bit-identical in every pairing:

| | PDL off | PDL on | saving |
|---|---:|---:|---:|
| L768 | 159.13 us/block | 157.37 | +1.76 (1.1 %) |
| L384 | 82.20 | 79.11 | +3.09 (3.8 %) |

`bench.py` with everything in (`results/bf16-L{768,384}-pdl.json`): **153.3 us/block at L768** (engine 557.7, 3.64x;
Anthropic hoisted 234.1, 1.53x) and **78.1 at L384** (272.2, 3.49x; 132.7, 1.70x). rel_rms 4.40e-3 / 4.42e-3.

**The race.** The first PDL A/B showed the baseline differing from itself (1.6e-3), which led to a real bug in the wired
CUDA core, present since the q-in-registers change. `det_core.py` (same input, 600 runs, compared bitwise) and
`det_core2.py` (which CTA, what kind of error, which run is right):

- 7/600 runs at L768 corrupted ONE CTA each; the corrupted run is the wrong one (2.54e-3 vs fp32 against 2.37e-3);
  the error is element-wise (ratio to the right value -4.8 .. 3.9 within one row), not a row scale.
- Only second-wave CTAs (heads >= 9, blockIdx >= 264 = 132 SMs x 2); L384 fits one wave and never failed.
- BACC=0 and PWARP=0 widen the window (95 / 27 of 600), which made every hypothesis testable.

Cause: q is parked in slot 0's bias area and read into registers with three ldmatrix (k-steps 0, 1, 2); the `qdone`
arrive that frees slot 0 was made data-dependent on the first and last destination only, so ptxas could sink the
middle ldmatrix past it, and the producer's first bias TMA overwrote q dims 16-31 for a few rows. A co-resident CTA's
contention (only in the second wave) widens the window. Fix (`QDEP=2`): the arrive depends on all twelve registers.

| 600 runs at L768 | old dependency | all twelve (`QDEP=2`) | q not aliased (`QREG=0`) |
|---|---:|---:|---:|
| BACC=0 | 99 | **0** | 0 |
| defaults | 7 | **0** | 0 |

Ruled out on the way, each with the amplified build: a warp fence before the slot release (`__syncwarp` 61/300,
`+__threadfence_block` 45/300), the setmaxnreg over-ask (67/300 with it fixed), the bias multicast to a one-CTA cluster
(143/600 with a plain TMA), the `mapa` + `shared::cluster` release (89/600 CTA-local). Two of those are real defects
and are fixed anyway at no cost (`core_ab.py`, do_bench, 46.99 vs 46.88 us at L768, 20.70 vs 20.77 at L384):
- `REGFIX`: the setmaxnreg split assumed a 128-thread producer and one CTA a SM, so the consumers asked for 232
  registers out of 112 (59 K asked, 32 K owned). Sized from the real launch now.
- `NODANGLE`: the last STAGES slot releases were never waited for and could be in flight at exit. Not issued now.

After the fix: `test_core.py` passes at L256/384/768; 0/600 kernel runs and 13/13 step runs bit-identical, with both cores.

Lesson: a data-dependent arrive has to depend on *every* load it releases -- ptxas only orders what the dependency
names. And a kernel that runs deterministically in one wave can still race in two; test at a multi-wave shape.

## Where the PDL chain still broke: trigger position (`graph_trace.py`, `ab_trig.py`)

`graph_trace.py` replays the captured step under `nsys --cuda-graph-trace=node` (needs
`NSYS_NVTX_PROFILER_REGISTER_ONLY=0` for the NVTX capture range) and `graph_trace_parse.py` reports per-kernel time
and the gap between consecutive kernels (negative = overlapped). At L768 the wall was 151.5 us/block against 160.1 us
of kernel time, so PDL already hid 8.6 us, but unevenly:

| transition (median gap) | L768 | L384 |
|---|---:|---:|
| GEMM -> attention core | -3.10 us | -2.72 |
| GEMM -> row pass | -0.54 | -1.50 |
| core -> Wo GEMM | -0.70 | -0.51 |
| **row pass -> expand GEMM** | **+0.13** | **+0.13** |
| **row pass -> q\|k\|v\|g GEMM** | **+0.06** | **+0.10** |

quack triggers early (in its epilogue), so whatever follows a GEMM overlaps. Our row pass triggered after its stores,
and a dependent launches only once EVERY program has triggered: with ~1000 programs over several waves that is
completion. Triggering at program start (and in the core right after its own wait) lets the dependent launch once the
last program has started; it still waits for completion before reading, so this is always safe. `ab_trig.py`,
bit-identical outputs:

| per block | base | row pass at start | core at start | both |
|---|---:|---:|---:|---:|
| L384 | 78.48 us | 76.78 (+1.70) | 78.32 (+0.16) | **76.47 (+2.02)** |
| L768 | 155.64 | 154.24 (+1.40) | 154.65 (+1.00) | **153.92 (+1.72)** |

Both are on (`TDIT_TRIG_START`, `CTRIG=1`). `bench.py` (`results/bf16-L{768,384}-trig.json`): **149.5 us/block at
L768** (engine 555.7, 3.72x; Anthropic hoisted 231.8, 1.55x) and **75.6 at L384** (270.7, 3.58x; 132.1, 1.75x).
test_core passes, 0/600 kernel runs and 13/13 step runs bit-identical.

**Measured and not taken:**
- **Loads before the wait** (`ab_early.py`): the row pass reading x and the conditioning rows before `gdc_wait` (only y
  depends on the GEMM it waits on), or the core pulling its bias rows into L2 before its wait. L384: -1.07 and -1.14
  us/block, -2.11 together. Traffic under the previous GEMM's tail is exactly what slows that tail.
  (`TDIT_EARLY`, `BPRE`, both off.)
- **Keeping cuBLAS out of the GEMM race** (`ab_picks.py`, `TDIT_MM_CUBLAS=0`): cuBLAS launches without PDL, so a cuBLAS
  pick would break the chain on both sides. But the race no longer picks it at either length (tile_M 64 quack configs
  take Wo and squeeze at M = 1920 too): identical picks, 78.82 vs 78.79 us/block at L384. The picks do vary between
  processes within noise (an earlier NCU run had squeeze on cuBLAS at L768).

## fp8 for q|k|v|g (opt-in, `TDIT_FP8_QKVG=1`)

The GEMMs are 58 % of the step and at 60-77 % of the bf16 roof, so the one large lever left on them is fp8 (2x the
tensor rate). Accuracy first, by fake-quant emulation (`fp8_emul.py`: e4m3 quantize -> dequantize at a GEMM's inputs,
against the IEEE fp32 reference):

| GEMMs in fp8 | L768 rel_rms | L384 |
|---|---:|---:|
| none (bf16 as shipped) | 4.40e-3 | 4.42e-3 |
| expand / squeeze / both | 3.05e-2 / 2.12e-2 / 3.69e-2 | same |
| **q\|k\|v\|g** | **4.86e-3 (x1.10)** | **5.09e-3 (x1.15)** |
| Wo / q\|k\|v\|g + Wo | 4.51e-3 / 4.95e-3 | 4.58e-3 / 5.23e-3 |

The transition GEMMs cannot take fp8 (7-8x the error, past the engine's 1.1e-2). q|k|v|g can; per-tensor scales cost
the same as per-row.

Kernels at the q|k|v|g shape (`fp8_gemm.py`, graph replay): bf16 quack 25.99 / 14.41 us (M = 3840 / 1920);
`torch._scaled_mm` rowwise fp8 28.71 / 16.16 (slower: short K, the bf16 output write and prologue dominate); quack
fp8 with a per-tensor `alpha` 21.51 / 11.91. quack's sm90 GEMM accepts e4m3 but its torch -> CuTe dtype map has no fp8
entry (registered in `_quack_gemm()`), and only `quack.gemm.gemm` takes `alpha` (`gemm_act` does not).

**Static activation scale.** The q|k|v|g input is an AdaLN output, LN(x) sigmoid(ms) + mb: |LN(x)| <= sqrt(d - 1) =
27.7 by construction, and e4m3 keeps full relative precision over ~2^14.8. One static scale (`TDIT_FP8_XA_BOUND` / 448,
default 128) therefore costs nothing -- rel_rms is identical for bounds 32, 64, 128 and 256 -- so there is no amax pass
and no delayed scaling. The row pass that feeds q|k|v|g writes `xa8` (e4m3, clamped to +-448 so an overflow saturates);
the one that feeds the transition stays bf16. Weights get one scale per block at pack time.

Measured (`ab_fp8.py`, in one process, interleaved; `bench.py` both ways on node02):

| | bf16 | fp8 q\|k\|v\|g | saving | rel_rms |
|---|---:|---:|---:|---:|
| L768, A/B (14 rounds) | 154.14 us/block | 147.57 | +6.57 (4.3 %) | 4.40e-3 -> 4.86e-3 |
| L768, bench.py | 147.4 | **142.9** | +4.5 (3.1 %) | |
| L384, A/B | 76.31 | 75.16 | +1.15 (1.5 %) | 4.42e-3 -> 5.09e-3 |
| L384, bench.py | 76.2 | **73.7** | +2.5 (3.3 %) | |

Deterministic. Off by default: it is an accuracy trade (x1.10-1.15, still 2.2x better than the engine's bf16 path),
which is the caller's to make.
