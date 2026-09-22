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
