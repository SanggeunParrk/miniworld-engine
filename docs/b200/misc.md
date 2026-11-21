# Sampler glue, smooth lDDT, La-Proteina pair builders, parallel residual, ProteinMPNN, PairformerBlock on B200 (sm100)

Scope: the `cuda` backend of `AugmentNoise`, `AlignEuler`, `PoolAtoms` (`modules/sampler_ops.py`), `SmoothLddtLoss`
(`modules/smooth_lddt.py`), `PairEmbedAdaLN`, `PairEmbedGeom`, `PairEmbedCA` (`modules/pair_embed.py`), `ParallelResidual`
(`modules/residual.py`), `ProteinMPNN` (`modules/mpnn/`) and the composite `PairformerBlock` (`modules/pairformer.py`) on a capability
(10, 0) device (card key `b200_sm100`). Every class has `backend_versions["cuda:b200_sm100"] = 1` (2026-11-16), except `PairEmbedCA`,
`ParallelResidual` and `SmoothLddtLoss` = 2 (optimization round, 2026-11-17, below) and the round-3 bumps listed in the next section; no class
`version` and no other card's entry changed.
Measured on the B200 box (GPU 1, torch 2.13 cu129, nvcc 12.9).

## Round 3 (2026-11-17): B200 cuda vs H100 cuda, target >= 1.5x on every case

Rule: for every requirement case both cards serve, B200 cuda median <= H100 cuda median / 1.5 (graph timing, L2 flushed, the speed tokens
`judge/tokens/{b200_sm100,h100_sm90}`). Branch `b200r3-misc`. `backend_versions["cuda:b200_sm100"]`: AugmentNoise / AlignEuler / PoolAtoms 2,
PairEmbedGeom 2, PairEmbedAdaLN 2, PairEmbedCA 3 (no class `version`, no other card's entry). Records on the B200 box:
`$W/b2511/agents/r3-misc/out/{spd,acc}_<Class>_{v2,v3}.jsonl` (v3 where present), scripts `$W/b2511/agents/r3-misc/r3m/`.

| class | cases vs H100 | not >= 1.5x: before -> after | worst b200 / h100: before -> after | slower than H100: before -> after | accuracy (cuda) | ladder (cuda vs triton / pytorch tokens) |
|---|---|---|---|---|---|---|
| AugmentNoise | 20 | 20 -> 20 | 1.189 -> 1.130 | 19 -> 2 | 60 GREEN | 0 violations |
| AlignEuler | 10 | 10 -> 10 | 1.137 -> 0.862 | 8 -> 0 | 30 GREEN | 0 |
| PoolAtoms | 18 | 18 -> 10 (training 8 / 8 pass, 0.27 - 0.38) | 1.190 -> 1.118 | 17 -> 7 | 46 GREEN | 0 |
| PairEmbedGeom | 24 | 18 -> 10 -> **5** (second part) | 1.015 -> 0.837 -> **0.735** | 2 -> 0 | 44 GREEN / 4 YELLOW (row 156 master: as before, as triton) | 0 |
| PairEmbedCA | 24 (20 at H100 >= 20 us) | 16 -> 10 -> **2** | 1.082 -> 0.981 -> **0.690** | 4 -> 0 | 48 GREEN | 0 |
| PairEmbedAdaLN | 12 | 12 -> 12 -> **1** | 1.064 -> 0.968 -> **0.735** | 2 -> 0 | 18 GREEN / 6 YELLOW (fp32 training: as before) | 0 |
| ParallelResidual | 45 | 2 -> 2 (not touched) | 0.693 | 0 | - | - |
| PairformerBlock | 30 | 14 (composite: report only) | 1.029 | 1 | - | - |

**The floor of a small launch on this card.** Measured with the measurer's timer (`r3m/floor.py`, `floor2.py`): a CUDA-graph replay of ONE
empty kernel is 6.0 us, every further kernel node +2.05 us (PDL or not: k = 1 / 2 / 3 / 4 / 6 kernels 6.02 / 8.06 / 10.11 / 12.16 / 16.26), and
the medians are quantised to that 2.05 us step: one kernel that spins for 0 - 1.5 us reads 6.02, 2 - 3.5 us 8.06, 4 - 5 us 10.11. The H100 runs
the same sampler ops in 8.5 - 9.1 us (AugmentNoise), 9.0 - 11.2 (PoolAtoms inference), 11.2 - 12.9 (AlignEuler), so the 1.5x targets are 5.7 - 6.1,
6.0 - 7.5 and 7.5 - 8.6 us: at or below the empty-kernel reading. A cluster kernel that only loads its atoms and stores them back (no sums, no
exchange) is 2.3 us of device time = 8.0 us: AugmentNoise cannot read below 8.0, PoolAtoms inference (two dependent cold round trips: index,
then rows) not below 8.0 either, AlignEuler passes only where the H100 needs >= 12.1 us and the whole kernel stays under ~3.8 us.

**Sampler glue (`sampler_ops/cuda/sampler_ops_sm100.cu`, the H100 source `sampler_ops.cu` included unchanged).**

| op | before (H100 source built for sm_100a) | now | S1, us: before -> now (H100) |
|---|---|---|---|
| AugmentNoise | 2 kernels (sums, PDL, augment) | 1 thread-block-cluster kernel a call: the CTAs of a sample exchange their (w, w p) sums through distributed shared memory in one cluster barrier, every CTA adds the partials in rank order (same bits in every CTA) | 10.1 -> 8.0 (8.5 - 9.1); N 12288 / 16384 at the 8 / 10 step (3.4 us device) |
| AlignEuler | 3 kernels (centres, covariance, rotation + update) | 1 cluster kernel: one-pass moments about atom 0 (one exchange instead of two), Horn's rotation in thread 0's registers on the 10 distinct entries with a power-of-two rescale from the trace every third squaring (~530 instructions; the 16-entry version with a division per squaring 1400 = 1.6 us) | 12.1 - 14.2 -> 10.0, N 1024 8.1 (11.2 - 12.9) |
| PoolAtoms inference | 1 kernel (SUB 8 / 4) | up to 1536 CTAs the same sums with every atom row of a token in flight (SUB 32), above that the H100 kernel | N 8192 / 12288 12.2 -> 10.0; else unchanged 10.0 / 12.1 (9.0 - 11.2) |
| PoolAtoms training | forward + prep + `torch.sort` (4 - 11 kernels) + backward | forward (zeroes the row counters) + prep (buckets every slot under its row by an atomic counter) + backward (orders a row's <= 16 slots by slot index in registers, adds in that order: the sorted kernel's sums, bitwise; a row with more slots scans the keys) | 39.8 - 80.0 -> 16.3 - 22.4 (43.0 - 70.5) |

Cluster shape (measured S 1 / 5 / 48 x N 1024 - 16384): the smallest K (atoms a thread) in 1, 2, 4 whose cluster fits 16 CTAs within 4 CTAs
per SM for the launch; K 8 only when K 4 does not fit. Rejected (numbers in the source comments): one CTA of K 4 / 8 for N <= 2048 without any
exchange (align 10.0 vs 8.0 at N 1024); Horn's rotation in every thread (12.1 us at N >= 8192) or in warp 0 with shuffles (the H100 form, 2 us
of the critical path); a shuffle-free pooling forward with 2 channels a thread (16.2 vs 10.0 us); the SUB 32 pooling forward at every size
(S48 N 8192 116.5 vs 73.5 us: now gated at 1536 CTAs).

**Pair builders (`modules/pair_embed.py` lean preparation, B200 cuda only).** A graph node costs ~2 us here, and the preparation kernels of
`_prepared` (bin limits by `torch.linspace`, mask and index casts, the transposed bf16 table and its autograd backward) were 6 - 10 of the 8 - 15
kernels of a step. On a (10, 0) device the cuda route now: makes the bin limits once per device; views contiguous bool masks as bytes; reads
the int64 indices as they are in the encoder pair's training forward (the one-launch kernels of the build, gt 2, write the same codes);
makes the differentiable table and its gradient with one kernel each (`_TableGrad`); serves inference from the bf16 table cached per weight
version (`kernels/_capture.param_forms`, `capture_reads_eager`: a replay runs the forward kernel alone; an in-place weight update misses the
cache). Outputs and weight gradients are bitwise those of the full preparation (`r3m/check_pe.py`, 60 configurations). `reduce_parts` of the
tcgen05 table gradient requests 16 partials at once (the 74 strided loads of a thread were 19 dependent round trips, ~11 us whatever the size).

| class | kernels a step: before -> now | S1 L128 / L256 / L384, us: before -> now (H100) |
|---|---|---|
| PairEmbedGeom inference | 2 -> 1 | 24.5 / 40.8 / 59.3 -> 22.4 / 36.7 / 57.2 (35.2 / 56.1 / 92.2) |
| PairEmbedGeom training | 14 -> 5 (LayerNorm: 15 -> 6) | 63.5 / 90.0 / 122.8 -> 51.1 / 69.6 / 98.2 (65.4 / 112.6 / 189.5) |
| PairEmbedCA inference | 2 -> 1 | 12.2 / 16.3 / 20.4 -> 10.0 / 14.1 / 20.3 (16.8 / 26.3 / 40.0) |
| PairEmbedCA training | 9 -> 6 | 41.0 / 53.1 / 71.6 -> 36.6 / 45.0 / 65.3 (37.9 / 62.3 / 98.4) |
| PairEmbedAdaLN inference | 8 -> 4 (fp32: 6 -> 2) | fp32 26.5 / 49.0 / 77.7 -> 18.5 / 42.9 / 69.6 (24.9 / 53.4 / 82.7) |
| PairEmbedAdaLN training | 15 -> 12 | fp32 210.8 / 410.5 / 618.5 -> 204.3 / 399.2 / 614.7 (214.1 / 452.2 / 708.1) |

**Still not 1.5x, and what each would take.**

| cases | now | why | what it would take |
|---|---|---|---|
| AugmentNoise 20 / 20 | 8.0 us vs H100 8.5 - 9.1 (0.88 - 0.94; two cases at 10.0) | target 5.7 - 6.1 us is the reading of an EMPTY kernel on this card (6.0); load + store alone reads 8.0 | nothing kernel-side; a timer without the 2 us step would read ~7 us (still not 1.5x) |
| AlignEuler 10 / 10 | 10.0 us (N 1024: 8.1) vs 11.2 - 12.9 (0.72 - 0.86) | target 7.5 - 8.6: only the N >= 5120 cases can pass, at 8.0 (device <= ~3.8 us; now 3.8 - 4.7, the kernel without any rotation 3.1 - 3.8) | the rotation's 12 squarings spread over 16 lanes through shared memory (~0.3 us less) and the block sums without their second barrier: N 5120 - 8192 may then read 8.0 |
| PoolAtoms inference 10 / 10 | 10.0 us (N 16384: 12.1) vs 9.0 - 11.2; 7 cases (N 1024 - 6144, 16384) still slower than the H100 (1.03 - 1.12) | target 6.0 - 7.5; two dependent cold loads (index, rows) + the launch read 10.0 (device 4.1 - 4.6 us; 8.0 needs <= 3.8) | a layout whose token -> row map is known without the index load (consecutive rows per token), i.e. a caller-side contract; not a kernel change |
| PairEmbedGeom inference L256 (2), with LayerNorm L256 (2), with LayerNorm L384 fp32 (1) | 38.9 vs target 37.4 / 37.5 (sweeps 36.8 - 38.8); 45.0 vs 40.9 / 41.1; 71.71 vs 71.70 | one forward kernel (pe_geom_sm100.cuh) bound by the eleven L1 table-row gathers and the binning a pair, not by HBM (33 MB in ~35 us at L256); more warps help only to 24 a SM (register cap), and every load-batching / lane / chunk variant measured (second part, below) was slower or equal | a forward that keeps the table in shared memory with more warps in flight, or partial row sums precomputed per feature group (the four distance rows / five angle rows as tables of their own) |
| PairEmbedCA training with LayerNorm L128 (2) | 32.7 (sweeps 30.6 - 32.7) vs target 31.6 / 31.9 | four launches (table 2.3, forward 6.1, LayerNorm backward 6.6, table gradient + reduce 11.6 us device) + the 2 us a graph node | the LayerNorm backward inside the table gradient's scatter warps (one launch less) |
| PairEmbedAdaLN inference fp32 L128 (1) | 18.3 vs target 16.6 (sweeps 16.2 - 16.4: the measurer's 2.05 us step) | one launch (fwd_s); four dependent memory round trips a chunk (inputs, Gram entries, U rows, store) at one chunk a warp | nothing that moves it by a whole timer step short of a different launch floor |
| ParallelResidual inference L128 d256 (2) | 12.2 / 14.2 vs target 12.0 / 13.7 | one kernel at the HBM roofline (34 - 42 MB in 6 - 7 us); the 2 us step | nothing kernel-side |
| PairformerBlock (14, composite) | 0.67 - 1.03 | its children (TriMul / TriAttn / Transition / AttentionPairBias: other agents) | follows their branches |

### Round 3 restart (2026-11-17 evening): true state, then fixes

Branch `b200r3-misc` on `b200-2.5.x` 83e4c1ddf. True state first: `measure.speed` of cuda / triton / pytorch on every requirement case of the
ten classes (samples min), each (class, length) in its own measurer run in an exclusive `dev.sh -x` slot on GPU 1; records
`$W/b2511/agents/r3-misc/out/t6a/` (before), `t6b` / `t6c` (after, the changed classes), scripts `r3m6/t6.sh`, `r3m6/t6sum.py`.
`cuda:b200_sm100`: PairEmbedGeom 20, PairEmbedCA 20, PairEmbedAdaLN 20; `triton:b200_sm100`: PairEmbedAdaLN 20.

| class | cases (H100 >= 20 us) | not 1.5x: before -> after | worst b200 / h100: before -> after | ladder: before -> after |
|---|---|---|---|---|
| PairEmbedGeom | 24 (24) | 6 -> 0 | 0.733 -> 0.634 | 0 -> 0 |
| PairEmbedCA | 24 (20) | 1 -> 0 | 0.688 -> 0.687 | 0 -> 0 |
| PairEmbedAdaLN | 12 (12) | 1 -> 0 | 0.735 -> 0.653 | triton > pytorch 2 (L128 train, 1.13 / 1.10) -> 0 |
| ParallelResidual | 45 (44) | 2 -> 2 (not changed) | 0.693 | 0 |
| PairformerBlock | 21 (21) | 5 -> 5 (children only) | 0.847 | 0 |
| SmoothLddtLoss | 32 (32) | 32 (17 slower than the H100; exception, report only) | 1.097 | 0 |
| AugmentNoise / AlignEuler / PoolAtoms | 20 / 10 / 18 (0 / 0 / 8) | 0 (the training PoolAtoms cases pass; the rest exempt) | 1.148 / 0.871 / 1.129 (exempt cases) | 0 |
| ProteinMPNN | 8 (H100 cuda unsupported) | - | - | 0 |

| change | what | effect (measurer timing, S1, us) |
|---|---|---|
| encoder forward with the table in shared memory (`pe_geom_sm100.cuh` `fwd_smt`, above 128^2 pairs, eval and training) | the bf16 table (317 x 512 B = 162 KB) in shared memory, one CTA of 16 warps per SM, loaded by four 1-D bulk copies (TMA, mbarrier) while the first chunk is binned; 32-pair chunks, 16 lanes a pair; the rows added in the same order (bitwise the L1 forward: eval + training, all cases, `r3m6/chk_smt.py`) | eval L256 38.8 -> 30.6, with LayerNorm 44.9 -> 34.7; L384 59.4 -> 51.1 / 73.6 -> 61.3; training L384 96.1 -> 87.7, with LayerNorm 178.3 -> 165.9 |
| decoder / encoder LayerNorm backward with 8 lanes a pair at <= 128^2 pairs | was 16 (its sums in another fixed order; accuracy 48 / 48 GREEN) | CA training with LayerNorm L128 32.6 -> 30.6 (L256 / L384 kept at 16: 8 lanes cost 2 - 6 us there) |
| PairEmbedAdaLN inference `fwd_s` at 2 CTAs per SM | was 4: one chunk a warp at L128, nothing in flight | fp32 eval L128 / 256 / 384 18.3 / 34.7 / 51.1 -> 16.3 / 28.6 / 49.1 (CPS 1 / 3 slower) |
| PairEmbedAdaLN triton backward rows a program on the B200 (`RB_B200`) | `_adaln_bwd` ran 8 pair rows a program: 16 programs at L128 S1 (406 us of a 512 us step); now 1 / 2 / 4 rows by B * n | fp32 training L128 / 256 / 384 S1 511 / 1003 / 1560 -> 162 / 332 / 604, S5 616 / 1492 / 2499 -> 376 / 1116 / 2078; partials bitwise repeatable, accuracy as before (fp32 training YELLOW = cuda) |

Rejected: `fwd_smt` at 8 / 24 / 32 warps (24 / 32: register cap; L256 49.1 / 47.0 vs 38.8 at 16 before the bulk copy) and with 8- / 16-pair
chunks (L256 42.9 / 34.7 vs 30.6 at 32), `fwd_smt` at L128 (30.6 vs 20.4: the table load is a fixed ~2 us a CTA); ParallelResidual with 2 / 4
vectors a thread and tensor at 2 - 3 branches (a build-time hook in the shared source; L128 x 3 branches 14.21 / 14.21 vs 14.21, slower at
L256 / L2048 x 2 branches; the hook reverted, diff in the scratchpad `r3m6/pr_unroll_hook_rejected.diff`).

Remaining: ParallelResidual L128 d256 x 3 branches eval 14.21 vs 13.66 and training 30.62 vs 30.59 (one timer step / at the target: one
HBM-bound launch, 42 MB in ~8 us under the L2 flush); PairformerBlock L128 / L256 (0.67 - 0.85): its 13 / 19 kernels are all its children's
(TriMul k1w / k3g / ctr, TriAttn front / core / tail, Transition, AttentionPairBias; profiled, no block glue), which other agents own;
SmoothLddtLoss (user exception).


### Round 3, part 7 (2026-11-17 night): ParallelResidual to the HBM peak (rejected), PairformerBlock child timeline

No source change (`parallel_residual_sm100.cu` restored; the experiment source is kept in the scratchpad as `r3m6/pr_sm100_forms_rejected.cu`).
Records: `$W/b2511/agents/r3-misc/out/t7/` (true state, the same `t6.sh` runs), job logs `$W/b2511/jobs/r3m-pr7.log`, `r3m-pr8.log`, `r3m-pf7.log`.

**ParallelResidual L128 d256 x 3 (bf16, 41.9 MB), measurer timer, 200 - 600 replays x 3 - 4 interleaved rounds.** The event timer's readings
are quantised (12.16 / 14.21 / 16.26 us), so the mean of the readings estimates the true time: shipped 14.22 (the median flips to 12.16
only below ~13.2 us). The kernel's device time (profiler, after the flush) is 8.4 us = 5.0 TB/s; the 2-branch case (33.5 MB) 6.8 us: time
is linear in the bytes with ~0.1 us fixed, i.e. throughput-bound, while the same kernel reaches 7.2 TB/s at L2048. The difference is the
L2 the measurer's flush (`zero_` of 256 MB) leaves full of dirty lines: every line the kernel brings in evicts a dirty one.

| form (all bitwise equal to the shipped kernel) | mean of readings, us |
|---|---|
| shipped (256 threads, 1 vector a tensor, ld.global.cs / st.global.cs) | 14.22 - 14.24 |
| load policy: ld.global.nc, nc + L1::no_allocate + L2 evict_first (createpolicy), + L2::256B prefetch | 14.20 - 14.24 |
| store policy: plain, L2 evict_first | 14.20 - 14.24 |
| store with L2 evict_last (the output kept in the L2 through the flush) | 13.6 - 14.2, drifting back to 14.2 within 4 rounds; rejected (it lives off L2 state that a replay leaves for the next one, and it pins lines the rest of the model would want) |
| 128 / 512 / 1024 threads, 2 / 4 vectors a thread | 14.2 (4 vectors: 16.3 - 22.4) |
| bulk copies (TMA): 1 or 2 CTAs a SM, every CTA's balanced share loaded at once in 1 / 4 / 8 / 16 pieces on mbarriers, stored from registers or by bulk store | 14.2 at best (1 piece), 15 - 35 with more pieces / 1024 threads |

Training L128 x 3 (30.6 vs target 30.59; 28.7 in another run): the step is the forward, the `(1 - n) g` kernel and three `Memcpy DtoD` that
autograd makes because one `g` is the gradient of three leaf inputs. Removing them would mean the backward writing three copies of `g`,
which a real caller (branches that are not leaves) would pay for: not done.

**PairformerBlock (5 misses).** The block launches nothing of its own (profiler: every kernel of the block step is one a child launches by
itself; L128 eval 13 / 19 kernels, train 50). Child times (each child's own graph step; the H100 token of the same child shape where one
exists):

| case (b200 / H100 block, target) | child kernels (device us) | child vs its H100 token |
|---|---|---|
| r140 L128 eval (112.5 / 132.9, 88.6) | TriMul x2: k1w 8.7, ctr 5.3, k3g 7.9; TriAttn x2: tri_front_sm100<128,128,4> 10.2, triattn_fwd_sm100 9.3 - 9.5, tri_tail_sm100 6.5 - 7.1; Transition: transition_fwd2_sm100 11.6 | TriAttn d128 x 4 heads 34.7 vs 36.6 (0.95: the main miss); Transition d128 n4 16.3 vs 23.5 (0.69) |
| r141 L128 eval (139.2 / 171.7, 114.4) | as r140 + AttentionPairBias inf_front_k<16,128,3> 6.8, augattn_inf_sm100 6.7, nvjet 3.1; single Transition sx_ln 1.9, sx_expand 4.3, sx_squeeze 6.0 | as r140 |
| r140 L128 train (460.6 / 674.8, 449.8) | TriAttn x2 (122 each: front 11.0, fwd 9.3, tail 8.2, gate_bwd 12.0, bwd_kv 10.9, bwd_q 10.9, head_bwd 11.0, wgrad 15.8, slot_sum 6.9, pack 3.4, param_finish 3.2, bwd_pre 2.1, a DtoD copy 1.4); TriMul x2 (94: b7g 28.3, b1w 17.0, k3g 10.4, k1w 8.8, cgb 8.4, grad_reduce 6.0, ctr 5.3, philox + drop_scale 3.6); Transition 61 (bwd 35.7, bwd_reduce 15.9, fwd2 12.1, cast3 2.0) | TriAttn 122 vs 157 (0.78); Transition 61 vs 86 (0.71) |
| r140 / r141 L256 eval (262.0 / 386.5, 257.7; 297.0 / 432.7, 288.5) | TriAttn x2: triattn_fwd_sm100 33.4 - 34.1, tri_front_sm100<128,128,3> 19.3 - 19.6, tri_tailg_sm100 17.6 - 19.2; TriMul x2: k3g 19.2 - 19.4, k1w 17.2 - 17.5, nvjet 7.9 - 8.2; Transition fwd2 28.0 | TriAttn 77.7 vs 107 - 109 (0.72); Transition 34.6 vs 57.3 (0.60) |

### Round 3, part 8 (2026-11-18): true state on engine 2.5.14 (b200-2.5.x 05fe8a72b), PairformerBlock child shares at large L

No source change. True state with `r3m6/t6.sh` (cuda / triton / pytorch, each (class, length) its own measurer run, `dev.sh -x`, GPU 3)
against the 2.5.14 H100 tokens; records `$W/b2511/agents/r3-misc/out/t8/` (all classes, on f5f97915f) and `out/t9/` (PairformerBlock again
after merging 05fe8a72b, whose new TriAttn / TriMul / Transition sources the block's children use; no source of these classes changed).

| class | cases (H100 >= 20 us) | not 1.5x | worst b200 / h100 | slower than H100 | ladder |
|---|---|---|---|---|---|
| PairEmbedGeom / PairEmbedCA / PairEmbedAdaLN | 24 (24) / 24 (20) / 12 (12) | 0 / 0 / 0 | 0.633 / 0.682 / 0.649 | 0 | 0 |
| ParallelResidual | 45 (44) | 1 (r161 d256 x 3 L128 eval: 14.11 vs target 13.66, 0.689) | 0.689 | 0 | 0 |
| PairformerBlock | 21 (21) | 6 -> 5 (after 05fe8a72b: L1024 0.669 -> pass) | 0.728 | 0 | 0 |
| AugmentNoise / AlignEuler / PoolAtoms | 20 / 10 / 18 (0 / 0 / 8) | 0 | exempt cases only above 0.667 | (exempt) | 0 |
| ProteinMPNN | 8 (no H100 cuda token) | - | - | - | 0 |
| SmoothLddtLoss (exception) | 32 (32) | 32 | 1.096 | 17 | 0 |
| MSAPairWeightedAveraging / OuterProductMean (re-check) | 98 / 222 | 0 / 0 | 0.641 / 0.622 | 0 | 0 |

PairformerBlock remaining misses (r140 L128 0.724; r141 L128 0.728, L256 0.681, L1536 0.677, L2048 0.668): the block launches nothing of
its own (`r3m6/prof_pf8.py`: L256 19 kernels, L1024 22, every one a child's). r141 child steps vs the H100 child tokens:
L256 TriAttn d128 x 4 heads 73.7 / 75.6 us vs 107 - 109 (0.68 - 0.70; core `triattn_fwd_sm100` 33 us, front 20, tail 21), TriMul 50.9 / 50.3,
Transition 32.8, AttentionPairBias 30.5, single Transition 16.2; L1024 TriAttn 1799 / 1819 vs 2560 - 2564 (0.70 - 0.71; the core 1484 / 1501 of it),
TriMul 688 / 680 (k1w 264 - 270, k3g 248 - 250, cuBLAS 200), Transition 319 vs 665 (0.48). The block's miss at L >= 256 is the TriAttn
child (owned by the TriAttn agent); at L128 TriAttn, TriMul and the launch floor of 13 - 19 kernels.

ParallelResidual r161: unchanged from part 7 (one HBM-bound launch under the flushed, dirty L2; the event timer's 2.05 us step).

### Round 3, second part (2026-11-17 afternoon): pair builders

Scope: the pair-builder cases with an H100 cuda median >= 20 us (the sampler ops are exempt). `cuda:b200_sm100`: PairEmbedAdaLN 3,
PairEmbedGeom 3, PairEmbedCA 4. Records `$W/b2511/agents/r3-misc/out/{acc,spd}_<Class>_v4.jsonl` (AdaLN: `_v5`), measurer timing on GPU 0.

| change | what | effect (S1, measurer / sweep timing, us) |
|---|---|---|
| table gradient + reduce in one launch (`pe_tbl_sm100.cuh` FR, `RED_V = 2`) | the tcgen05 kernel launched cooperatively (2 x nblk <= SMs CTAs); after a grid barrier (two words per device, generation counter, reused by every launch) every CTA sums its columns over the slices in slice order, two columns a thread, all 74 partials in flight: the adds and the bf16 rounding of `reduce_parts_t<true>` (weight gradient bitwise equal) | CA training L128 30.5 -> 26.6; the reduce launch (7.8 us device + a 2 us node) gone at every size |
| training table by `pg120_table` behind the forward (`TBL_IN_LAUNCH`) | a training step passes the module's weight to the forward launch; the A100 source's `table_of` makes the bf16 table and the forward follows it by PDL (`griddepcontrol.wait` before its first table read; the binning overlaps the transpose); the encoder forward got the same hook | the `copy_(weight.t())` elementwise kernel (2.8 - 3.8 us) replaced by a 2.3 us PDL predecessor: CA training L128 26.6 vs 28.6 |
| PDL from the LayerNorm backward into the table gradient | `ln_bwd` triggers its dependents at its start; the fused table gradient waits after its setup (barriers, TMEM, zeroed one-hot buffers) | CA training with LayerNorm L128 34.7 -> 32.6 (sweeps 30.6) |
| encoder forward register cap above 128^2 pairs | `pe100g::fwd<..., MB>`: `__launch_bounds__(128, 6)` (24 warps a SM) at CPS 6 above 128^2 pairs, the uncapped build at CPS 4 below; lanes a pair 32 without the LayerNorm above 128^2 pairs, else 16; LayerNorm backward CPS 8 | Geom eval L256 / L384 40.9 / 63.4 -> 36.8 - 38.8 / 59.3; with LayerNorm L384 77.7 -> 71.6; training L384 105.4 -> 96.1 |
| PairEmbedAdaLN inference from LayerNorm statistics tables (`fwd_s`) | both LayerNorms act on a sum of <= 3 table rows: mean / variance are sums of per-row and per-row-pair numbers of the weights (centred rows T~, U = lw T~ - mean, Gram matrices T~ T~^T and U U^T, V = U . b~), made by two small kernels once per weight version (fp64 sums); a lane gets (r1 r2, r2) from six Gram entries and three V, the pair's lanes need no reduction | fp32 eval L384 53.1 -> 49.0 (FWD_G 16 path), L128 / L256 equal |
| PairEmbedAdaLN binning in fp64 | the distances of `code_of` in fp64 (the gold's bin at every limit) | accuracy fix of the round-3 forward: S1 L384 master eval / train was YELLOW (an sc distance 1.5500000416 read 1.5499999523 = the limit 1.55: err 0.385 vs the gold, the H100 build 0.0037); now GREEN as the H100 build |
| decoder WT forward removed | the training forward making the table in every CTA's shared memory (first part, wip) | 25.5 us forward vs 4.6 + 2.3 (pg120_table): 46.9 vs 30.5 a step; rejected |

Rejected (measured, reverted): encoder pair rows requested together before any is added (`pair_x`: G16 2x slower, registers), evict-last
L1 hints for the table rows (equal), chunks of 8 / 16 / 32 pairs fixed (L128 with LayerNorm 22.4 / 26.5 / 36.7; L256 49.0 / 44.9 / 44.9),
CPS 3 / 8 for the encoder forward (L256 44.9 / 42.9 vs 38.8), fewer table-gradient slices at <= 128^2 pairs (37: 26.5 vs 24.5, 48: equal),
`fwd_s` with 16 lanes a pair (L384 57.1 vs 49.0).

Checks: `r3m/check_g.py quick` / `check_ad.py quick` (new vs the earlier route: no LayerNorm bitwise incl. the weight gradient, with it to
bf16 rounding; repeat bitwise; captured == eager): ALL OK. GPU tests 91 / 91 (`r3m/run_tests.py all`; new:
`test_pair_embed_fused_reduce_graph_b200` -- fused vs separate reduce bitwise, a captured training step replayed twice to the eager bits --
and `test_pair_embed_adaln_stats_inference_b200`).

**H100 technique audit (docs/h100/pair_embed.md, `cuda/sm90/pair_embed_geom.cu`, `cuda/sm90/pair_embed_adaln.cu`).**

| class | H100 technique | on the B200 |
|---|---|---|
| all three | table gradient on the tensor cores (wgmma, scatter warps writing the swizzled one-hot, fixed slices, fixed-order reduce) | used: tcgen05 / TMEM / TMA (`pe_tbl_sm100.cuh`); the reduce now inside the launch (FR) for Geom / CA |
| Geom / CA | 16-byte codes from the training forward, replayed by the LayerNorm backward and the table gradient | used (same codes) |
| Geom / CA | binning by binary search over the limits in shared memory | used (the sm90 `bucket` is included unchanged) |
| Geom / CA | persistent CTAs, a warp per row / 32-pair item; LayerNorm backward with the pair recomputed, loads two steps deep | used in the B200 form (persistent 4-warp CTAs over 8 / 16 / 32-pair chunks, G lanes a pair, next chunk in flight); the 16-warp CTA of the H100 measured slower here (CPS 4 vs 6 / 8 above) |
| Geom / CA | bf16 table by one transpose-and-cast copy (untracked) | improved: inference reads a table cached per weight version (no copy); training: `pg120_table` behind the forward by PDL |
| Geom / CA | launches a training step: table copy, forward, (LayerNorm backward), table gradient, reduce = 4 - 5 | 3 - 4 (table, forward, (LayerNorm backward), table gradient + reduce) |
| AdaLN | forward: lane bins, both LayerNorms by warp shuffles, 32-bit bin word saved | training: used (pe_adaln_sm100.cuh forward, same bin word); inference: replaced by the statistics-table forward (no reductions); binning moved to fp64 (accuracy) |
| AdaLN | backward: rows kernel (dx + per-CTA partials), wgmma table gradient (`pe_bwd_tbl4`), two fixed-order reduces, batch in chunks | used (B200 rows kernel, tcgen05 table gradient, chunks); the fused FR reduce not applied (the chunked launches accumulate partials across chunks; training already at 0.25 - 0.29 of the H100) |
| all | fp16 / TF32 forms | not applicable: gathers and adds, no GEMM besides the one-hot product (bf16 operands, fp32 accumulators on both cards) |

Outside sm100-only code (second part): none. `pe_tbl_sm100.cuh` (shared by PairEmbedGeom / PairEmbedCA / PairEmbedAdaLN, sm_100a only)
gets the FR variant as a template default `false` (AdaLN's launches unchanged); `pair_embed_geom/cuda/function.py` changes only the
B200 autograd function.

Outside sm100-only code (round 3): `modules/pair_embed.py` (helpers `_b200`, `_linspace`, `_mask8`, `_idx_pair`, `_TableGrad`; `_prepared(...,
lean=False)`: the default is the unchanged preparation, which the triton backend and every other card keep; only `_forward_cuda` on a (10, 0)
device passes `lean=True`), `kernels/sampler_ops/cuda/function.py` (`_PoolAtomsCuda`: a (10, 0) branch for the sort-free backward),
`kernels/sampler_ops/cuda/__init__.py` (`ext_sm100` builds `sampler_ops_sm100.cu`; the extension name `sampler_ops_sm100a` is unchanged, so the
sm100 prebuild plan needs no edit). No shared `.cu` / `.cuh` changed.

GPU tests (`tests/gpu/test_misc_b200.py`, run by `r3m/run_tests.py`: the B200 env has no pytest): 35 / 35 pass (the 22 earlier ones + the cluster
kernels vs PyTorch, the sort-free pooling backward vs the sorted kernel bitwise, the cached inference table after a weight update).

## Optimization round (2026-11-17)

Ladder on the B200 tokens before the round (`docs/ladder.md` rule, graph timing, L2 flushed): `SmoothLddtLoss` cuda < triton 1 / 32 REAL
(worst 1.045, master L1024 eval), `ParallelResidual` cuda < triton 8 / 45 (exception PR, worst 1.040), `PairEmbedAdaLN` triton < pytorch 2 / 12
(exception small L, 1.13); `PairEmbedCA` 16 of 48 cases refused by the cuda route; ProteinMPNN without a B200 prebuild entry. After (records
`$W/b2511/agents/opt-misc/out/{acc,spd}_<Class>.jsonl`): no ladder violation in `SmoothLddtLoss`, `ParallelResidual`, `PairEmbedCA`; nothing refused.

| change | before | after |
|---|---|---|
| `PairEmbedCA` inference: a B200 decoder-pair forward (`pair_embed_geom/cuda/sm100/pe_ca_sm100.cuh`, `forward_ca_b200`) replaces the RTX 5090 one-launch kernel and the H100 launch at every size; `ca_eval_slower` removed | 16 / 48 refused; served eval cases 1.0-1.5x triton | 48 / 48 served, all GREEN; eval 1.29-1.50x triton without the LayerNorm, 1.34-1.56x with it (S5 L384 63.4 / 67.5 us vs 81.8 / 100.3); train unchanged 1.07-1.85x |
| `ParallelResidual`: `parallel_residual_sm100.cu` = the shared source with the inputs of tensors past the L2 (>= 160 MB) read through the default cache path instead of evict-first loads | cuda / triton up to 1.040 (8 cases > 1.02) | max 1.012 (0 cases > 1.02); L2048 x 3 branches 1547 -> 1486 us (7.2 TB/s, triton 1489) |
| `SmoothLddtLoss`: `slddt_prep4` (four threads a row; B200-only translation unit `smooth_lddt_sm100.cu`, which includes the shared `smooth_lddt.cu`) for prep grids of <= 4 CTAs per SM | master L1024 eval 59.8 vs triton 57.2 us | 53.1 vs 57.2 (triton / cuda 1.14-9.96 over 64 cases); prep outputs bitwise equal |
| ProteinMPNN: the sm100 prebuild target yields the four MPNN sm_100a builds (`kernels/integrations/__init__.py`; `_prebuild.py` sets `MINIWORLD_ENGINE_BUILD_ARCH=sm100` for the sm100 target, as it does sm120) | JIT build on first use | `mpnn_{message,node_message,edge,relpos}_sm80_sm100` in the sm100 plan |

**The decoder-pair forward.** The op writes B n^2 x 256 bf16 (S5 L384: 377 MB). The RTX 5090 kernel (non-persistent CTAs of 32 pairs, each
half-warp binning its pair from three dependent global loads before its first store) reached ~3 TB/s on the B200, and with the LayerNorm it
was instruction bound (~10 shuffles + ~10 scalar instructions a channel and pair). The B200 kernel: persistent warps (at most 32 CTAs of 128
threads per SM) over chunks of 32 consecutive pairs; lane l bins pair 32 q + l (coalesced j-side loads) and the next chunk's inputs are loaded
before the current chunk is written; a group of 8 lanes owns a pair (four pairs per warp step, 128 contiguous bytes per store and pair);
`bf16(T[sep] + T[dist])` as one bf16x2 add per two channels (the correctly rounded sum, what the triton kernel's fp32 sum rounded to bf16 gives:
bitwise equal output without the LayerNorm), the LayerNorm in fp32x2 (`__fadd2_rn` / `__ffma2_rn` / `__fmul2_rn`, sm_100) with 3 shuffles per
reduction; streaming stores; four warp steps interleaved up to 128^2 pairs, one above. Launch sweep (measurer timing, us; cps 8 / 16 / 32 x
lanes 8 / 16 / 32 x steps 1 / 2 / 4 x streaming on / off):

| case | triton | RTX 5090 kernel | H100 launch (jp 16) | first B200 version (32 lanes a pair) | this kernel |
|---|---|---|---|---|---|
| S1 L128 | 18.3 | 12.2 | 22.5 | 18.4 | 12.2 |
| S5 L256 | 43.0 | 61.3 | 45.1 | 36.9 | 34.8 |
| S5 L384 | 79.8 | 122.9 | 77.8 | 69.4 | 63.4 |
| S5 L256 + LayerNorm | 55.3 | 81.9 | 65.5 | 60.4 | 36.7 |
| S5 L384 + LayerNorm | 100.3 | 168.0 | 116.7 | 118.8 | 67.5 |

32 lanes a pair: 1.1-1.5x slower with the LayerNorm (shuffle bound); S5 L384 without it runs at 6.0 TB/s. The training forward (with codes)
keeps the H100 launch (served and faster than triton before this round).

**ParallelResidual** load / store variants (text copies of `parallel_residual.cu`, measurer timing, us): at L2048 x 3 branches shipped
(`__ldcs` / `__stcs`) 1541, plain loads + plain stores 1486, `__ldcs` + plain stores 1533, `__ldg` + `__stcs` 1486, two vectors a thread 1543,
512 / 128 threads 1551 / 1537, triton 1490. Below the L2 size the evict-first loads were one timer step (2 us) faster (L256 / L384 x 3 branches),
hence the per-tensor size switch (uniform per launch).

**SmoothLddtLoss** (L1024 forward, profiler): 14 kernels, 51.7 us device: `slddt_fwd_split` 12.3, `slddt_prep` 10.1 (64 CTAs of one thread
a row looping over 128 columns), the rest PyTorch glue (casts, sums, the loss) shared with triton (12 kernels, 53.9 us: its forward 24.6, prep
3.4). `slddt_prep4` (module, eval / train us, prep vs prep4): N1024 57.2 / 53.1, 83.8 / 79.7; N2048 53.1 / 51.1, 77.7 / 74.8; N3072-4096 equal;
N6144 96.1 / 104.4; N8192 129.0 / 137.1, hence the grid-size gate.

**Branch features**: `origin/b200/ladder-final` has no B200 kernel of its own for these classes beyond what main already carried (smooth lDDT:
the branch's `smooth_lddt.cu`, launch geometry and the split already in; pair builders: `pe_tbl_sm100.cuh` already adopted). Nothing else to
take; the decoder-pair forward above is new.

GPU test `tests/gpu/test_misc_b200.py` (+ `test_pair_embed_ca_inference_b200`: every size served, bitwise repeatable, CUDA graph = eager, equal
to triton without the LayerNorm, within one bf16 step with it): 22 / 22 pass on the B200. CPU gate (`gate.sh`): ruff clean, CPU pytest 3361
passed; pyright 1 error and `ruff format --check` in files outside this round's diff (base branch; fixed there since).

## Routes

| class | B200 route | kernels | source |
|---|---|---|---|
| `AugmentNoise`, `AlignEuler`, `PoolAtoms` | `_cuda_h100` admits (10, 0); `sampler_ops/cuda/sm80.device_ext` -> `ext_sm100` | `sampler_ops_sm100.cu` built as `sampler_ops_sm100a`: `sampler_ops.cu` (the H100 source, included unchanged) + the round-3 single-launch kernels (cluster augment / align, SUB 32 pooling forward up to 1536 CTAs, sort-free pooling backward); sizes outside their contracts take the H100 kernels | main + round 3 |
| `SmoothLddtLoss` | already admitted (10, 0) | `smooth_lddt/cuda/sm100/smooth_lddt.cu` cubin for sm_100a (main kept it); the B200 now takes the split tile launches (`_split`, the A100's fill of 32 CTAs per SM); `slddt_prep4` for small prep grids (2026-11-17) | main + this round |
| `PairEmbedGeom`, `PairEmbedCA` | `_cuda_arch_b200`; round 3: lean preparation, inference on the cached bf16 table (`sm100._eval_table`); `pair_embed_geom/cuda/sm80.device_build` -> `cuda/sm100`; decoder-pair inference: `sm100.forward_ca_eval` (`forward_ca_b200`, 2026-11-17) | `sm100/pair_embed_geom_sm100.cu` = the A100 source `sm80/pair_embed_geom_sm80.cu` included unchanged (`-DMWE_SM120` on, so its one-launch inference forwards are compiled too) + the tcgen05 table gradient `pe_tbl_sm100.cuh` (`backward_tc`) + the decoder-pair inference forward `pe_ca_sm100.cuh` | branch `pe_tbl_sm100.cuh` + main A100 source + this round |
| `PairEmbedAdaLN` | `_cuda_arch_b200`; `pair_embed_adaln/cuda/sm80.device_build` -> `cuda/sm100` | `sm100/pair_embed_adaln_sm100.cu` = `sm80/pair_embed_adaln_sm80.cu` unchanged + `backward_tc` (rows kernel of the H100 source, table gradient on `pe_tbl_sm100.cuh`) | as above |
| `ParallelResidual` | `_cuda_arch_b200` | `parallel_residual_sm100.cu` (includes the shared `parallel_residual.cu`; cached input loads past the L2) built as `parallel_residual_sm100` | main + this round |
| `ProteinMPNN` | `mpnn_sm80_unsupported` admits (10, 0) | the sm_80 MPNN sources (`mpnn_message`, `mpnn_node_message`, `mpnn_relative_position`, `mpnn_edge_tail`) built for sm_100a as `<name>_sm100` through the base hook `kernels/_sm120.sm80_load_extension` (its sm100 target; mma.sync / ldmatrix / cp.async: legal on sm_100a; the 162 KB fused message backward fits the B200) | main |
| `PairformerBlock` | no kernel of its own: routes when every child routes | TriMul / TriAttn / Transition / LayerNorm / AttentionPairBias B200 routes (other agents' branches) | - |

The MPNN loaders keep calling `kernels/_sm120.sm80_load_extension` (unchanged lines); on a B200 the base's `building_sm100()` hook builds
the same sources as `<name>_sm100` with `-DMWE_SM100`.

### Launch choices measured on the B200

All timings: CUDA-graph replay (`triton.testing.do_bench_cudagraph`), GPU 1 of the B200 box, us.

**SmoothLddtLoss** (`smooth_lddt/cuda/sm100/__init__.py` `_split`). Before: the B200 ran the whole-structure kernels (N <= 3520, 512
threads, A * B * N / 512 CTAs: 2 CTAs at A1 x N1024) or the unsplit tile kernels (A * B * N / 128 CTAs), and lost to Triton up to N 3072.
Now it takes the split tile launches with the A100's fill (KS = min(NB, ceil(32 * 148 / (A * B * NB)))):

| A x N | eval before / split / Triton | train before / split / Triton |
|---|---|---|
| 1 x 1024 | 122.3 / 47.3 / 72.8 | 230.7 / 69.5 / 183.2 |
| 1 x 2048 | 153.4 / 42.7 / 129.4 | 299.3 / 63.4 / 344.4 |
| 1 x 4096 | 187.5 / 60.0 / 244.3 | 359.6 / 85.3 / 676.7 |
| 1 x 8192 | 286.1 / 116.9 / 516.3 | 515.4 / 153.4 / 1361.4 |
| 5 x 1024 | 121.6 / 47.3 / 76.2 | 229.2 / 70.2 / 188.9 |
| 5 x 4096 | 192.7 / 85.1 / 459.5 | 372.2 / 143.4 / 1528.4 |
| 5 x 8192 | 299.3 / 181.4 (KS 16) / 1401.3 | 547.6 / 301.3 / 4828.0 |

(The rule's KS is the best measured or within 3 % at every point of the A1 / A5 x N1024-8192 grid.)

**PairEmbedGeom / PairEmbedCA training** (table gradient, `sm100.BWD_IMPL`, `SMALL_PAIRS`): tcgen05 (`backward_tc`) vs the A100
source's mma.sync impls 1 / 2 / 3, module forward + backward under bf16 autocast:

| case | Triton | tcgen05 | impl 1 | impl 2 | impl 3 |
|---|---|---|---|---|---|
| Geom S1 L128 | 591.0 | 83.9 | **80.7** | 90.0 | 98.7 |
| Geom S1 L384 | 1132.1 | **292.9** | 363.8 | 454.2 | 462.9 |
| Geom S5 L384 | 4419.5 | **1173.6** | 1566.6 | 2030.2 | 2036.6 |
| CA S1 L128 | 125.5 | 62.2 | **57.8** | 59.9 | 61.6 |
| CA S1 L384 | 335.0 | **245.7** | 284.1 | 304.2 | 286.4 |
| CA S5 L384 | 1320.5 | **1000.4** | 1229.8 | 1331.6 | 1216.0 |

so pair counts <= 128^2 take impl 1, larger tcgen05. The LayerNorm backward (with `ln_eps`): the H100 kernel (table in shared
memory) and the RTX 5090's (table through the L1) measure the same within 3 % (`LN_BWD_L1 = False` kept).

**PairEmbedAdaLN training**: tcgen05 1.03-1.11x the A100 source's backward from L256 / S5 on (S1 L384 734.1 vs 763.2; S5 L384 1713.0 vs
1901.4), equal at L128 x S1 (194.8 vs 190.0: the A100 backward kept there). Triton 1.5-2.7x slower than either.

**PairEmbedGeom inference**: the RTX 5090's one-launch forward from the module's inputs (`forward_eval`, gt 2) vs the H100 launch:
S1 L128 16.6 vs 27.2 (Triton 27.1), S5 L384 189.5 vs 226.0 (Triton 281.3); with the LayerNorm S5 L384 245.0 vs 283.9 (Triton 332.3):
`EVAL_ONE_LAUNCH["geom"] = True`.

**PairEmbedCA inference** (2026-11-16, superseded 2026-11-17 by the decoder-pair forward above): the RTX 5090 one-launch forward won up to
4 x 128^2 pairs, the H100 launch with jp 16 above; neither beat Triton with the LayerNorm above 128^2 pairs nor without it between 4 x 128^2
and 400k pairs (16 cases refused then).

**ParallelResidual inference** (2026-11-16, before `parallel_residual_sm100.cu`): at the HBM roofline (2048^2 x 256 bf16, 3 branches: 1.47 ms = 7.3
TB/s); without an L2 flush the shipped kernel, an `__ldg` / plain-store variant and Triton are equal within 0.3 %. The measurer (L2
flushed) puts the CUDA kernel 0-4 % behind Triton on the large inference cases (see the gaps).

## Accuracy and speed (B200)

Records on the B200 box: `$W/b2511/agents/misc/out/{acc,spd}_<Class>.jsonl` (2026-11-16), `$W/b2511/agents/opt-misc/out/{acc,spd}_<Class>.jsonl`
(2026-11-17: PairEmbedCA, ParallelResidual, SmoothLddtLoss; the last record per case is the current one) and, for the
composite, `$W/b2511/agents/misc-pf/out/{acc,spd}_PairformerBlock.jsonl`. Accuracy: every requirement row and length, all samples
(`b200cfg/acc_<Class>.yaml`); speed: CUDA-graph timing, L2 flushed, the same cases. Ratios are other / cuda over the cases cuda serves.

| class | cuda GREEN / YELLOW / RED / unsupported | triton / cuda | pytorch / cuda | anthropic / cuda |
|---|---|---|---|---|
| SmoothLddtLoss | 57 / 3 / 4 / 0 (triton identical) | 1.14 - 9.96, geomean 3.19 (64 cases, after the B200 pretune of triton) | 41 - 1250 (geomean 656) | - |
| AugmentNoise | 60 / 0 / 0 / 0 | 1.00 - 1.51 (60) | 7.3 - 11.0 | - |
| AlignEuler | 30 / 0 / 0 / 0 | 1.25 - 1.67 (30) | 21.9 - 34.6 | 5.5 - 19.4 |
| PoolAtoms | 46 / 0 / 0 / 0 | 1.08 - 1.64 (46, eval + train) | 1.5 - 27.5 | 1.7 - 3.6 |
| PairEmbedAdaLN | 18 / 6 / 0 / 0 (triton identical) | 1.13 - 3.48 (24) | 2.1 - 25.0 | - |
| PairEmbedGeom | 44 / 4 / 0 / 0 (triton identical) | 1.19 - 14.5 (48) | 6.3 - 23.6 | - |
| PairEmbedCA | 48 / 0 / 0 / 0 | 1.07 - 1.85, geomean 1.34 (48; eval 1.29 - 1.56) | 3.9 - 33.7 (geomean 15.6) | - |
| ParallelResidual | 45 / 0 / 0 / 0 | 0.99 - 2.36, geomean 1.33 (45; no case past 1.02) | 4.6 - 14.9 (geomean 10.0) | - |
| ProteinMPNN | 3 / 5 / 0 / 0 (triton 2 / 6, pytorch 7 / 1) | 1.19 - 1.36 (8) | 1.33 - 1.47 | - |
| PairformerBlock (children from the other agents' branches, scratch merge) | 20 / 6 / 0 / 0 (triton 20 / 6); 4 skipped (fp64 gold OOM at L1536 / L2048) | 2.87 - 4.57 (30) | 5.1 - 20.3 | 1.48 - 2.25 (eval) |

The non-GREEN lights equal Triton's and the other cards' (judge tokens h100_sm90 / a100_sm80 / 5090_sm120): SmoothLddtLoss fp32 train
L5120 / L7168 RED (err 1.3e-3 / 1.1e-3 for cuda and triton alike vs pytorch's 1e-7; the same RED on H100 and A100) and L6144 / L8192 YELLOW;
PairEmbedAdaLN fp32 train YELLOW (the bf16 pair gradient by design); PairEmbedGeom row 156 L384 master YELLOW (err 0.226, pytorch 0.2257:
a distance at a bin edge); ProteinMPNN YELLOW (mixed precision; A100 / 5090 the same); PairformerBlock master train YELLOW (H100 the same).

GPU test `tests/gpu/test_misc_b200.py` (routes; the tcgen05 table gradient bitwise reproducible, within 2e-2 of Triton, CUDA-graph
captured = eager; the decoder-pair inference forward at every size): 22 / 22 pass on the B200 (run without pytest, which the B200 env lacks: `b200cfg/run_tests_nopytest.py`).

## Gap analysis

| class | cases the B200 cuda route does not serve or does not win | why | what it would take |
|---|---|---|---|
| PairEmbedCA | none (2026-11-17: the 16 refused inference cases are served by `forward_ca_b200`, 1.29-1.56x triton) | - | - |
| ParallelResidual | none past the 2 % rule (2026-11-17); the largest inference cases tie triton (cuda / triton 0.99-1.012) | both kernels at the HBM roofline (7.2 TB/s at L2048 x 3 branches) | nothing kernel-side: the remaining traffic is the n + 2 tensor passes |
| SmoothLddtLoss | fp32 training L5120 / L7168 RED, L6144 / L8192 YELLOW (served) | the same values for Triton and for the H100 / A100 tokens: a ground-truth pair at the 15 A inclusion radius classified differently by the fp64 gold | none on the B200 side; a measurement-harness / input question shared by all cards |
| AugmentNoise | none (several cases a tie, 1.00x) | 8-16 us kernels at launch latency | - |
| ProteinMPNN | none refused (cuda 1.19-1.36x triton); the B200 prebuild entry added 2026-11-17 | the cuda route is the sm_80 MPNN sources built for sm_100a (mma.sync); no tcgen05 MPNN kernel exists on the branch | a tcgen05 message / edge GEMM path if MPNN speed on the B200 matters |
| PairformerBlock | none of its own; it routes exactly when TriMul / TriAttn / Transition / LayerNorm / AttentionPairBias route on the B200 (measured with b200-trimul, b200-triattn, b200-transition, b200-token-dit merged in the scratch worktree misc-pf) | composite | merging those branches |
| PairEmbedAdaLN | none on the cuda side; triton slower than pytorch at L128 S1 training (fp32 / master, 1.12-1.13x; ladder exception small L) | the portable table-gradient kernel (fixed NBLK slices) at 128^2 pairs: launch-bound | a triton launch-config / B200-gated variant (would bump `triton:b200_sm100`); left as the small-L exception |
| PairEmbedGeom, AlignEuler, PoolAtoms | none | - | - |

No requirement row of these classes is outside the kernels' contracts, and since 2026-11-17 the B200 cuda route serves every requirement case.

## Changes outside sm100-only code

No shared `.cu` / `.cuh` source changed (2026-11-16 and 2026-11-17): `smooth_lddt.cu` and `parallel_residual.cu` are byte-identical to the
base; the B200 builds include them from B200-only translation units (`smooth_lddt_sm100.cu` adds `slddt_prep4`, `parallel_residual_sm100.cu`
switches the input loads), so the H100 / A100 / RTX 5090 cubins and extensions keep their keys; `kernels/_prebuild.py` sets `MINIWORLD_ENGINE_BUILD_ARCH=sm100` for the sm100
target (on a B200 build host the same artifacts as before; elsewhere the sm_80 builders now target sm_100a as they should);
`kernels/integrations/__init__.py` sm100 branch yields `_prebuild_mpnn()`. Python-side, outside sm100-only files (2026-11-16):

* `modules/sampler_ops.py` `_cuda_h100`, `modules/smooth_lddt.py` (already admitted 10.0), `modules/mpnn/_functional.py` `mpnn_sm80_unsupported`,
  `kernels/integrations/mpnn_{msg,edge}_sm80.py` `_a100()`: the capability lists gain (10, 0).
* `modules/pair_embed.py`: new `_cuda_arch_b200` (the three pair builders and `ParallelResidual` use it; `_cuda_arch` itself, shared with
  `TokenPairInit`, is unchanged); `_eval_sm120(weight, x, kind)` gained the `kind` argument and a B200 branch; `PairEmbedCA._supports_cuda`
  called `ca_eval_slower` on a B200 (removed 2026-11-17: no refusal).
* `kernels/pair_embed_geom/cuda/function.py`: the RTX 5090 inference ops call `sm80.device_build().forward_eval` (on an RTX 5090 that is the
  same function as before).
* `kernels/{sampler_ops,pair_embed_geom,pair_embed_adaln}/cuda/sm80*`: `device_ext` / `device_build` return the sm100 builds on (10, 0).
* `kernels/smooth_lddt/cuda/sm100/__init__.py` `_split`: `sm_100a` joins the split launches (H100 / A100 / 5090 choices unchanged).
* `kernels/parallel_residual/cuda/__init__.py` `_ARCH` gains `sm100`; prebuild hooks of sampler_ops / pair_embed_* gain the sm100 target.
* `backend_versions["cuda:b200_sm100"] = 1` on all ten classes (2 on PairEmbedCA, ParallelResidual, SmoothLddtLoss since 2026-11-17); no
  `version` and no other card entry changed.
