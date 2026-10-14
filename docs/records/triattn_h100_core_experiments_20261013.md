# H100 TriangleAttention: attention-core experiments (2026-10-13)

Branch `h100/triattn-sol` (the sm_90a raw-PTX path of `h100_sources/triattn_{sm90,bwd_sm90,mod_sm90}.cu` on top of `h100/cuda-sweep`).
H100 80GB, torch 2.13 / CUDA 12.9, one GPU per job. Kernel time = CUDA events (graph-free, 30 runs, median) or `sm__cycles_elapsed`
under ncu (clocks locked, 1.45 GHz). Unless stated: forward core, B1, H4 x 32, L768, 870 us / 1.566 M cycles.

## What the forward core is bound by

- Not MUFU (ex2 16 / clk / SM, microbenchmarked 0.50 warp-inst / clk / SM): XU 57 %, tensor 31 %, issue 46 %.
- Per sub-step (128 queries x ... per warpgroup) period 1906 cycles; phases (cycles): bias wait + ldmatrix + unpack 180-260, K/V wait
  60-90, QK^T issue 60-90, QK^T retire 160-215, exponentials 850-920 (512 minimum), PV wait + release 170-200.
- wgmma shapes are full rate in isolation (`m64n128k16` 64 clk, `m64n40k16` 20 clk per SM with two warpgroups).

## Ablations (wrong results by design; a removed consumer makes the exponentials dead code, so keep P alive)

| removed | cycles / base |
|---|---|
| bias unpack | 0.94 |
| FFMA scale / shift | 0.86 |
| ex2 | 0.91 |
| QK^T wgmma | 0.88 |
| PV wgmma (P kept alive) | 0.97 |
| PV + QK^T | 0.82 |
| unpack + FFMA + ex2 | 0.86 |
| bwd key side: dK / dV wgmma (P, dS kept alive) | see `TB_ABL` in the patch |

(Early ablations that removed PV / M without keeping P alive read 0.49 / 0.40 and 0.44: those were dead-code elimination, not cost.)

## Variants tried (all correct, none faster than the baseline)

| variant | us |
|---|---|
| baseline (2 consumer warpgroups x 64 queries, 2 rows interleaved) | 870 |
| K/V ring 3 / 6, bias ring 3 | 872-876 |
| consumer ping-pong (`TA_PP`) | 898 |
| PV issued first / after the exponentials / after QK^T retired | 1100 / 940 / 887 |
| FFMA + ex2 fused in one asm (program order interleave) | 870 |
| warpgroup 1 start delayed 250-5000 cycles | 867-878 |
| 3 consumer warpgroups, one row each (`TA_V3`) | 977 |
| P through shared memory, PV as SS wgmma (`TA_PVSS`) | 1067 |
| 2 consumers, one row each, scores double-buffered (`TA_V4`, 128-key tiles) | 1136-1175, ptxas C7515 / C7512 |
| same with 64-key units (`TA_V5`) | 1035 |
| `TA_V5` with 3 consumers | 966, ptxas C7512 |

Notes that carry over: ptxas serializes every wgmma (C7513 / C7515 / C7514) when a CUDA-core instruction (including an empty
`asm volatile("+f")` fence) defines an accumulator while any wgmma stage is open, or a wgmma stays in flight across a call, branch or
loop back edge; wait_group counts must be compile-time. With bias unpacked into the accumulator the double-buffered form needs a
third register file the sm_90 consumers do not have (232 / 152 registers), and moving the bias add after QK^T costs one FADD per score.

The experimental source is kept outside the tree (`~/miniworld-engine-scratch/triattn_cs/core_experiments.patch`).

## Generic path (d_pair 64 / 256 / 384 / 512), inference

- one GEMM for q | k | v | g | bias when L <= 256 (a launch less; beyond L 256 the extra tile column costs more than it saves),
- `x + u Wo^T` as one `addmm` (residual in the GEMM epilogue) when there is no dropout scale.

d_pair 256, L128: 0.085 -> 0.0785 ms (Anthropic 0.0757); d_pair 64, L128: 0.035 -> 0.031 ms; other cells within +-1 %.

## Second round (same day): redesign attempts

Phase trace of the baseline forward (cycles per 128-key sub-step of one warpgroup, two warpgroups, period 1906): bias wait + ldmatrix
+ unpack 180-260, K/V wait 60-90, QK^T issue 60-90, QK^T retire 160-215, exponentials 850-920, PV wait + release 170-200.

| variant | us (baseline 870) | finding |
|---|---|---|
| exponentials one warpgroup at a time (token over named barriers, `TA_XEX`) | 877 | the exponential phase still takes 833 cycles with the MUFU to itself: not contention |
| truncating bf16 pack, one PRMT (`TA_PACKI=2`) | 872 | cvt is not the limiter either (accuracy 8e-3 vs 4e-3) |
| round-half-up pack on the integer pipes (`TA_PACKI=1`) | 1082 | |
| polynomial exp2 for 1 / 2 / 4 / 8 of 16 elements | 1136 / 1101 / 1060 / 1096 | in-order issue behind a dependent chain |
| `TA_V5` per-unit trace | period 1211 per 64-key unit (2422 per 128 keys vs 1906) | 325 cycles to issue PV(u-1) + QK^T(u+1) (6 wgmma, fences) is paid per unit |
| backward key side, valid ablations (P, dS kept alive) | G wgmma -13 %, dK / dV wgmma -9 %, ex2 -2 %, bias ldmatrix -3 % | 12 instructions per score, 2.2 of them descriptor arithmetic |

Reading: the forward core runs at 55 % of the MUFU bound with 31 % tensor and 46 % issue utilisation, close to what the FlashAttention-3
class of designs reaches against the same bound (about 60 %); every remaining term (unpack, scale, exponentials, convert, wgmma issue and
retire, barriers) is 3-12 % and none can be removed without the others growing.

## Third round: the bias on the tensor core

`S = I . bias + Q K^T`: the 64 x 128 bias tile of a warpgroup (as stored by TMA, 128B swizzle) is the B operand of four `m64n128k16`
wgmma against an identity A fragment (nonzero only in each warp's diagonal 16 x 16 block); the first of them overwrites the accumulator.
No CUDA-core instruction defines an accumulator any more, and the bias unpack / ldmatrix leave the sub-step.

| variant | us (baseline 876) | finding |
|---|---|---|
| bias on the tensor core inside the baseline kernel (`TA_BIASMMA`) | 900 | bit-identical output; the four extra wgmma cost what the unpack saved |
| `TA_V6`: 128-key tiles, one row per warpgroup, scores double-buffered | not run | ptxas C7512 (insufficient register resources) at the 168-register launch cap, with or without `setmaxnreg` |
| `TA_V7`: V6 with the loads issued by the consumers (256 threads, 243 registers, no producer warpgroup) | 1660 | output bit-identical to the baseline; ptxas still serializes the wgmma (C7512) |

C7512 appears whenever the PV wgmma is present next to two live score buffers, also with the PV retired before the next QK^T is issued,
with n32 instead of n40, without the loader code and without the bias wgmma; it disappears when the PV wgmma is removed. Free of the
bias unpack the pipelined form is correct and complete, but ptxas will not schedule it.
Sources: `~/miniworld-engine-scratch/triattn_cs/triattn_sm90.biasmma_v6_v7.cu`.

## Fourth round: the backward in one pass (dQ in the key-side kernel)

Measured instead of estimated: the key-side kernel got the dQ partial of every (step, row): dS^T through `stmatrix` into shared memory
(128B swizzle), four SS `m64n32k16` wgmma against the resident K tile (MN-major B), the fp32 result staged and added with one
`cp.reduce.async.bulk.tensor` into an fp32 `[rows, H * 32]` buffer.  K had to stay in shared memory until the end of a task (two K
buffers by task parity: with a single buffer the next task's load overwrote it under the dQ wgmma), and the K A fragments were
re-read from it to make room for the 16 dQ accumulator registers (without that ptxas serialized every wgmma, C7512).

| | us (L768, d_pair 128) |
|---|---|
| key-side kernel (dK, dV) | 1182 |
| key-side kernel + dQ partials, bit-checked against the query-side kernel (rel err 6e-5) | 2609 |
| query-side kernel (dQ, dbias) it would replace | 1170 |

The added 1.43 ms is the reduce engine: a bulk reduce-add into an L2-resident target sustains 3.3 TB/s on this card (microbenchmark,
8 / 16 KiB tiles; 1.1 TB/s into a 300 MB target), and the dQ partials alone are 3.6 GB (2 warpgroups x 6 key tiles x 8 KiB per step and
row), the dbias partials another 3.6 GB.  Summing the two warpgroups in shared memory first halves the dQ term; a 2 x 2 cluster
(DSMEM sum over two key tiles and two row groups) halves both again: about 0.8 ms of reduce time on top of 1.18 ms, plus the dQ buffer's
memset and its conversion to bf16, against 2.35 ms today -- at best -4 % of the training step for a cluster kernel with a
transposed dbias path.  Not built.  Source: `~/miniworld-engine-scratch/triattn_cs/triattn_bwd_sm90.dq_singlepass.cu`.

## Round 5 (2026-10-14): fewer wgrad splits to shrink `tri_grad_reduce`

C128, device time per kernel (us), training; splits = CTA pairs of `tri_wgrad` (default SMs/2 = 66):

| splits | L128 wgrad / reduce / step | L256 | L384 |
|---|---|---|---|
| 66 (default) | 16.0 / 9.8 / 166 | 46.1 / 9.3 / 473 | 91.6 / 10.6 / 1050 |
| 32 | 18.5 / 8.8 / 166 | 62.3 / 7.9 / 485 | 127.6 / 8.2 / 1082 |
| 16 | 30.6 / 8.5 / 177 | 111.9 / 7.5 / 534 | 244.0 / 7.8 / 1197 |
| 8 | 55.6 / 8.0 / 201 | 217.8 / 7.1 / 637 | 482.6 / 7.5 / 1435 |

Negative: the reduce stays at about 7-8 us even with 8 splits (the floor is the gate_bwd dWo partials, `ggrid` x 64 KB), while
wgrad loses far more than the reduce gains. The default stays.

## Round 6 (2026-10-14): dWo partials summed inside a 4-CTA cluster (DSMEM)

`tri_gate_bwd` staged each CTA's fp32 dWo in smem and summed the cluster's shares in rank order through `ld.shared::cluster`, so
`tri_grad_reduce` read a quarter of the slots (grid sized by `cudaOccupancyMaxActiveClusters`). 82 tests pass, deterministic.

| C128 train | L128 | L256 | L384 |
|---|---|---|---|
| reduce 9.8/9.3/10.6 -> | 7.2 | 7.2 | 8.7 us |
| step 166/473/1050 -> | 170 | 478 | 1052 us |

Negative: the reduce shrinks by 2-3 us but the gate_bwd tail (stage, two cluster barriers, DSMEM reads, fewer resident CTAs) costs
more. Reverted; patch kept in the scratch dir (`gate_bwd_cluster_reduce.patch`).
