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
so prologues and tails never overlap, and quack's sm90 GEMMs have no PDL.

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

`core_cu/attn_core.cu` is correct at every shape and carries the one thing Triton cannot express here -- the bias tile
TMA-multicast to the S CTAs that share it -- and it is still **68.0 us against the Triton core's 52.8** at L768
(do_bench). The variants, all measured: 128 query rows 84.9; a software pipeline over two score buffers 113 (163 with a
dynamic buffer index, which makes ptxas inject a warpgroup.wait); two blocks in flight 82.2; softmax interleaved with
PV per k-step 70.4; a single producer warp instead of a warpgroup 68.0 (the best). Every attempt at overlap costs
registers, and at 2 CTAs per SM (the only thing keeping this latency-bound kernel fed) there are none to spare. Beating
a tuned Triton core needs the full FA3 arrangement -- two consumer warpgroups ping-ponging phase by phase at 1 CTA per
SM -- which is a larger build than what it would return. The algorithmic half of the work already shipped: the fixed
softmax offset above came out of this kernel and now runs in the Triton core.
