# How the kernel got to 406 µs, and what did not work

Every number below is the same benchmark (`bench.py`, CUDA-graph replay median) on one node02 H100 80 GB, CUDA 12.9,
PyTorch 2.10 cu128, at L384 with `DW_REPL=8` unless stated. The development tree is
`MiniWorld/runs/transition_bwd_fused_20260921/` (variant sources `src/tbwd_*.cu`, full log in its `RESULTS.md`).

## First design: split the hidden axis over a cluster (rejected)

16 clusters × 8 CTAs, one CTA per 64-unit hidden slice with its weights resident, 64-row tiles, and the `d_xn` partial sums
reduce-scattered over distributed shared memory. It needs **no** recomputation, so its floor is the absolute 16·M·D·H one
(163 µs at L384) rather than 224 µs — and it still lost, because the cross-CTA reduction cost 572 µs of its 1017:

| step | µs | |
|---|---:|---|
| first working version, 16 clusters | 2855 | `cuOccupancyMaxActiveClusters` is 15 for a size-8 cluster at 1 CTA/SM, so the 16th cluster ran as a second wave |
| drop `fence.acq_rel.cluster` | 2787 | it lowers to `MEMBAR.ALL.GPU` + **`CCTL.IVALL`** (a full L1 invalidate) — 15 % of all stall samples. `mbarrier.arrive.release.cluster` already releases |
| 15 clusters | 1564 | SM active/elapsed 0.48 → 0.95 |
| 528-byte landing-row stride | 1568 | no change: the cost was not bank conflicts |
| `d_xn` issued before the weight-gradient GEMMs | 1560 | |
| per-peer mbarriers → hardware cluster barrier | **1017** | sixteen `mbarrier.arrive.release.cluster` per tile, issued one after another from a single thread, each draining that CTA's outstanding DSMEM stores: **761 µs** |

Ablations at that point: the two cluster barriers 359 µs, the remote DSMEM stores 213 µs, everything else 445 µs against a
163 µs floor. Larger tiles do not help (the landing buffer is already 33 KB for 64 rows) and one barrier per tile would need
bf16 partials, which changes the rounding contract. Abandoned.

## Second design: two CTA roles (this kernel)

| step | µs | |
|---|---:|---|
| first working version | 466 | Triton's `div.full` sigmoid form |
| kit `rcp.approx` sigmoid | 438 | `rcp.approx.ftz(1 + ex2.approx.ftz(-a log2 e))`; output identical |
| Wa and Wb packed into one `[128 n][128 d]` operand | 420 | `a` and `b` become a single m64n128 chain instead of two m64n64 chains: 24 wgmma per chunk instead of 32 |
| the same packing for `d_xn` | 419 | `[dA|dB]` is the m64k128 A fragment of one K=128 RS chain |
| dgamma / dbeta to a private global row per (CTA, warp) | **406** | shared-memory atomics were 77 µs; a private row needs none, and all six outputs became bit-reproducible |

Role-ratio sweep at L384 (`records/ratio-r*-L384.json`): R=6 555 · R=7 469 · **R=8 406** · R=9 460 · R=10 519 µs.
Above R=8 the input role is the bottleneck at a constant ~23 µs per 128-row tile, below it the weight role at ~2.9 µs;
R=8 also divides 1152 tiles exactly. Both roles are within 5 % of each other there, which is why every later change to one
role alone measured as no change at all.

## Rejected, with numbers

| | µs | why |
|---|---:|---|
| `tanh.approx` sigmoid | 426 | faster, but dx moves from 5.4e-5 to 2.9e-4 against the engine — a different tolerance class |
| weight-gradient GEMMs left in flight across the tile boundary | 527, then 501 | tried twice; the second attempt removed an illegal operand fence that could have explained the first. The next tile's stage-1 chains queue behind them and the `wait<2>/<1>/<0>` ladder gets longer |
| per-warpgroup one-way hand-off instead of the stage barrier | 493 | stage 2 breaks into two K=64 chains with an mbarrier spin between them |
| interleaving the epilogue's two rows | 414 | no change: the weight role was the wall |
| single-buffered xn + x read from global + warp-private shared dgamma | 392–396 | **the fastest thing measured**, and dropped: dx is not reproducible run to run, intermittently (2 of 3 replays differ). Swapping the `bar.sync` hand-off for an mbarrier did not fix it and both `racecheck` and `synccheck` are clean, so the cause is still unknown |

## Where it stands (NCU `--set full`, L384, R=8)

Tensor pipe 58.1 % active, SM active/elapsed 0.966, issue active 41.8 %, no excessive shared-memory wavefronts, and a flat
stall profile whose largest single SASS line is 4.5 % (`WARPGROUP.DEPBAR` 13 % and `HGMMA` 9.5 % in total). Since this design
must execute 22·M·D·H FLOP, 58 % tensor *is* 406 µs; 330 µs would need 70 %, spread over wgmma dependency waits rather than
any one hotspot.

The two levers left are both blocked by a hard limit. Software-pipelining the input role's chunk loop needs a second `a|b`
accumulator (64 more registers; the kernel uses 255). Making the weight role's two warpgroups independent needs a 32-wide
slice, which fits the registers only with transposed accumulators and triples the stage-2 wgmma instruction count. Splitting
the hidden chunk in half relieves both at once — 24 KB ring slots and half-size accumulators — at the cost of doubling the
wgmma instruction count for `dh` and `a|b`, which is exactly the trade the packing step above won 18 µs on.

## Recomputing xn in the backward instead of saving it: measured, and it loses

The forward saves `xn` (37.7 MB at L384, 151 MB at L768) for the backward. Dropping that and recomputing
`xn = (x·rstd − c1)·gamma + beta` in the backward looks attractive on paper, and it is not:

| | |
|---|---|
| what it saves in the forward | 9 µs at L384, 31 µs at L768 — the measured cost of the `xn` store (`-DFWD_SAVE=0`) |
| what it saves in the backward | **nothing.** Removing the `xn` read outright (timing-only probes) measured 414 µs for the input role and 415 µs for the weight role against a 409 µs baseline — inside the noise, i.e. zero. The backward moves ~226 MB in 406 µs = 557 GB/s against a ~3 TB/s peak and is nowhere near bandwidth-bound; the same probe on the 442 MB weight stream also measured nothing |
| what it costs | both roles would have to read `x` instead (the same bytes: no read is saved) and then run a LayerNorm-apply pass over 128 × 128 elements into shared memory before the GEMMs. In the weight role that is roughly 500 of its 5100 cycles a tile, and that role is the binding one at R = 8 — about 40 µs over its 144 tiles |

So it is a net loss for speed. It is still the right trade if activation memory is the constraint rather than time: 37.7 MB per
Transition layer at L384 and 151 MB at L768 is not nothing when the trunk has many blocks.

## Tuning the forward

Ablations at L384 against the 145 µs first version (timing-only variants, `src/transition_fwd_p*.cu`). **Read these with
care**: a probe that stops consuming `acc` lets the compiler delete the whole GEMM chain, and `p4` (drop the output stores)
measured 85 µs for exactly that reason — the real cost of those stores turned out to be about 6 µs.

| probe | µs | |
|---|---:|---|
| baseline | 144.8 | |
| `p1` no LayerNorm reductions | 111.2 | the two five-deep shuffle chains a row needs, done one row at a time |
| `p2` no SwiGLU arithmetic | 129.2 | |
| `p3` tanh.approx sigmoid | 144.1 | the transcendental is not the cost, so the tolerance-class change buys nothing |
| `p5` ring always fetches chunk 0 | 145.3 | the weight stream is free: not L2-bound |
| `p6` one TMA box instead of six | 138.9 | |
| `p7` no ring mbarrier handshake | 132.5 | |
| `p8` `p7` without the squeeze chain | 100.6 | |

Two changes came out of it, both keeping the numerics bit-identical:

| | L384 (training build) |
|---|---:|
| first working version | 155 µs |
| output staged in the x tile and handed to a TMA store, instead of 4-byte global stores from the fragment layout | 155 µs (and 145.9 → 140.4 on the inference build) |
| **LayerNorm reductions restructured to eight rows at a time** | **129.5 µs** |

The LayerNorm was the real one. Each row needs two five-deep shuffle chains, and doing one row at a time left that latency
fully exposed; eight independent chains per step hide it, and since the reduction order *within* a row is unchanged the
statistics are bit-identical. After it the kernel is at 47-51 % of its tensor floor, up from 39-43 %.

One side effect: `-DFWD_SAVE=0`, which drops the `xn` / `rstd` / `c1` stores for inference, is now *slower* than the full
build (138.8 vs 129.5 µs at L384) — a scheduling artefact, not a traffic one. The same build serves both cases.

## Traps worth remembering

- The effective dynamic shared-memory ceiling on sm_90 is **231424 B**, not the 232448 opt-in: 1 KB per block is reserved.
  Asking for more launches fine and then dies with an illegal instruction on the first out-of-range access.
- Re-applying `wgmma.fence` or an operand fence to an accumulator that still has an outstanding group **hangs**. Fences
  bracket a group's issue region.
- `bar.sync` orders threads, not the asynchronous proxy: a TMA write against wgmma operand reads of the same buffer needs an
  mbarrier hand-off.
- A launcher that packs arguments individually cannot take a single `__grid_constant__` struct parameter; pass each
  `CUtensorMap` as its own parameter and keep only addresses in the struct.
- A timing probe that stops consuming an accumulator lets the compiler delete the GEMM chain that fills it; the number it
  prints then measures nothing. Check that the probe still stores something derived from the accumulator.
- Keeping dgamma / dbeta in registers for the whole kernel (64 of them) spills 652 B. Reduce them per tile.

## Forward, round 2 (2026-09-22): what the time actually is, and the inference fix

Development scratch: `MiniWorld/runs/transition_fwd_opt_20260922/` (`ab.sbatch` A/B, `ncu.sbatch`, `trace.sbatch`); variant
sources `src/transition_fwd_p9.cu` … `p18a.cu`, `src/transition_fwd_trace.cu`, `trace_fwd.py`, `bench_infer.py`.
Every variant below was checked against the base output (`bench_fwd.py --ref-cubin`) and for replay reproducibility; all
are bit-identical to the base unless stated. Medians of three interleaved rounds, same job, L384 / L768.

**Where a tile's time goes** (the `%clock`-stamped trace build, calibrated against `%globaltimer`): about 14.5 µs per
128-row tile — LayerNorm ~1.9 µs (13 %), the eight chunks ~10.4 µs (72 %; the tensor floor of a tile is 7.0 µs), the
epilogue ~2.2 µs (15 %). The two warpgroups run in lockstep (G1-issue skew median −32 cycles), so LayerNorm and epilogue
are tensor-idle time and both warpgroups' SwiGLU share the MUFU at once.

**Hypotheses that measured wrong** (so the README's forward headroom paragraph was wrong too):

| variant | idea | L384 | L768 |
|---|---|---:|---:|
| p9 | software-pipelined chunk loop: G1(j+1) issued before SwiGLU(j), second [a\|b] accumulator | +38.6 % | +26.8 % |
| p10 | p9 with G1 descriptors as (lo, hi) + immediate offsets (255 → 255 regs, spills 408/620 B → 72/72 B) | +10.2 % | +9.4 % |
| p11 | p10 with wait<1> every chunk and a [Wa;Wb] × 3 + Ws^T × 2 ring (no spills) | +4.5 % | +1.4 % |
| p12 | next tile's LayerNorm in NSLICE = 2 / 4 / 8 slices under this tile's G1s | +19.5 / +10.9 / +18.1 % | +12.4 / +8.2 / +17.3 % |
| p13b | weight ring split by operand, refills at the base kernel's point: (2,2) control / (3,2) / (2,3) | −0.5 / +1.6 / +3.2 % | +0.4 / +2.0 / +4.7 % |
| p14 | FA3 ping-pong: one G2(j-1)+G1(j) batch per chunk, alternating warpgroups on two named barriers | +13.7 % | +12.1 % |
| p15 | output store waits for `.read` instead of global completion | 0.0 % | +0.4 % |
| p16 | two dedicated producer warps (320 threads) | +22.7 % | +21.5 % |
| p16ctl | the base kernel with a 320-thread launch bound only (183 → 168 registers, two idle warps) | +16.5 % | +15.6 % |
| p17a | `tanh.approx` sigmoid, one MUFU instead of two — **not bit-identical**: 0.6 % of outputs 1 bf16 ulp off, rel 2.0e-4 | **−3.0 %** | **−3.8 %** |
| p17b | ex2 on the MUFU, reciprocal as three Newton steps on the FMA pipe (218 registers) | +29.4 % | +19.5 % |

What the controls establish: (1) the ~11 % of NCU samples spinning on `w_full` before G1 is **not** load latency — deeper
rings with more slack are slower, not faster (p13b); (2) the kernel is **extremely sensitive to its register allocation**:
the same code at 168 instead of 183 registers is 16 % slower (p16ctl), which is also why every restructuring that raised
or lowered the live set lost; (3) halving the MUFU work buys only 3-4 % (p17a), so SwiGLU is latency-, not MUFU-throughput-
bound. The one gain on the training path (p17a) changes the tolerance class for 3.5 % and was not adopted.

**Inference: two fixes, both shipped, output bit-identical to the training build.**

1. `-DFWD_SAVE=0` compiled the xn / rstd / c1 stores out, and ptxas then scheduled the kernel into 155 registers that ran
   7-8 % *slower* than the training build. The inference build now keeps the stores in the code behind a runtime `save`
   argument, launched with 0 (`src/transition_fwd_p18.cu`): kernel 138.3 → 116.9 µs (L384), 516.3 → 432.5 µs (L768),
   −15.5 % / −16.2 %. The training build is unchanged (the runtime guard costs it 1.5 %, so it keeps the compiled form).
2. **No real inference call ever used the inference build.** The build was chosen inside the autograd Function from
   `ctx.needs_input_grad`, which reflects only `requires_grad` — and module parameters keep that under `torch.no_grad()` —
   while the Function's `forward` always runs with grad mode off. Every no_grad call therefore ran the training build and
   wrote an xn nobody reads. The choice is now made in `transition_fused_sm90a()` before `apply`, from grad mode AND
   `requires_grad`, and a non-grad call skips the Function entirely.

Module level (`bench_infer.py`, `modules.Transition(128, 4)` under `torch.no_grad()`, same run): L384 129.6 → 119.5 µs
(−7.8 %), L768 505.8 → 470.6 µs (−7.0 %), 2.25× / 2.22× the Triton path. `tests/numerics/test_transition_fused_sm90a_gpu.py`
8 passed.

### LayerNorm and epilogue, the two pieces outside the chunk loop (2026-09-22)

| variant | idea | L384 | L768 | output |
|---|---|---:|---:|---|
| p19 | LayerNorm with two threads per row (64 columns each), the base reduction tree reproduced exactly: xor 16 as one shuffle with the partner thread, xor 8/4/2/1 in registers; 32 shuffles per warp per tile instead of 160 | +12.7 % | +9.2 % | bit-identical |
| p19b | p19 re-reading its granules per pass (201 → 175 registers) and storing the saved xn from the shared tile in whole rows | −0.8 % | −0.5 % | bit-identical |
| **p20** | **epilogue through `ldmatrix.x4` / `stmatrix.x4`: the x tile read in exactly the accumulator's fragment layout, 16 instructions instead of 64** | **−3.3 … −4.2 %** | **−3.3 … −3.6 %** | **bit-identical** |
| p22 | p19b + p20 | −3.9 % | −3.1 % | bit-identical |

The LayerNorm is not bound by its shuffles (cutting them 5× buys nothing once the register count is back down); p20 is
shipped in both builds. Module level after p20 (`bench_infer.py`, same run): no_grad forward 116.7 µs L384 / 422.6 µs L768
(2.30× / 2.48× the Triton path); `bench_wired.py` forward + backward L384 554.5 µs (1.95× Triton, 3.29× eager; the L768
run of that job was noisy, median 2205 vs min 2042 µs, and is not used). GPU numerics tests 8 passed.
