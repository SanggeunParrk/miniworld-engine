# Bias-only token DiT on H100: the B200 three-kernel step and fused backward, ported and measured (2026-10-19)

H100 80GB HBM3 (cssb node01 / node02, 700 W), the repo pixi env (torch 2.13.0+cu129), branch `h100/bias-only-dit-inf3` off main
ea7e8717. bf16, 16 heads x 48, d768 / cond 384 / pair 128, key mask 20 %. All timings are CUDA-graph replays (median of rounds);
per-kernel numbers are graphs of 10-20 repeats of one launch; "step" numbers are one block's whole step. Probes (outside the
checkout): `~/miniworld-engine-scratch/wt-bodit-logs/probe_*.py`.

## 1. Where today's H100 step spends its time (before this work)

Per node, one block, us (`probe_brk.py`: each launch of the 12-launch step as its own graph; "step" the whole step):

| (L, A) | step | cond LN | cond g1 | cond g2 | adaln_in | v\|g GEMM | pvx | out GEMM | resgate_adaln | expand+SwiGLU | squeeze GEMM | resgate_out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128, 1 | 40.5 | 1.5 | 3.7 | 3.3 | 2.1 | 3.7 | 3.5 | 3.4 | 2.6 | 10.2 | 4.4 | 1.9 |
| 768, 1 | 72.0 | 1.7 | 6.2 | 4.9 | 2.7 | 6.0 | 8.8 | 4.9 | 3.5 | 10.3 | 7.6 | 2.4 |
| 128, 5 shared | 47.8 | 1.5 | 3.8 | 3.3 | 2.6 | 4.8 | 4.5 | 4.0 | 3.4 | 10.3 | 5.3 | 2.3 |
| 384, 5 shared | 84.7 | 1.6 | 4.7 | 3.9 | 4.2 | 9.3 | 8.9 | 5.9 | 5.7 | 19.0 | 8.6 | 3.6 |
| 768, 5 shared | 140.8 | 1.7 | 6.3 | 4.8 | 7.0 | 16.1 | 20.0 | 9.2 | 9.4 | 27.1 | 14.5 | 5.6 |
| 1536, 5 shared | 287.4 | 1.9 | 9.3 | 6.2 | 18.9 | 31.7 | 67.7 | 15.6 | 28.6 | 52.3 | 26.1 | 17.2 |
| 2560, 5 shared | 528.1 | 2.1 | 15.5 | 8.3 | 32.5 | 49.2 | 160.5 | 26.0 | 52.8 | 84.8 | 42.4 | 28.2 |

(L1536 / L2560 with the length bound lifted, section 4; before it those lengths ran the PyTorch module.) The node sum matches the
step within 2-19 us: the 12 launches are serialized, each node at its own rate.

## 2. The B200 three-kernel step on wgmma (`bo_inf3_sm90.cu`, `inf3_h100.py`)

Built as B200's `bo_front_bf16` / `pv_gate_inf -DPDL_INF` / `bo_tail_bf16`: a cluster of CL CTAs (8, or 16 non-portable) per
128-row tile, CTA c owning 768 / CL output columns and 1536 / CL hidden units; LayerNorm statistics as per-CTA (mean, M2) pushed to
every peer through DSMEM (`cp.async.bulk.shared::cluster`) and merged in rank order; xa / xt / h through L2 scratch with one release
per exchange; the activation k-blocks multicast across the cluster by TMA (issuer = position % CL); weights streamed by a
producer that starts before the PDL wait; front -> `pvx` (new `pv_cta_pdl`) -> tail chained by programmatic dependent launch.
Correct (vs the fp32 module chain 2.96e-3 against the 12-launch step's 3.06e-3; vs the 12-launch step 1.7e-3) and bitwise
repeatable at every shape (`test_inf3_h100_matches_the_oracle_and_is_bitwise_repeatable`).

First build: 330 us per cluster round. The trace (`probe_tr.py`, `%globaltimer` per k-block) showed every k-block 4.0 us apart:
the slot releases were `mbarrier.arrive.release.cluster` to 8 CTAs, ~0.5 us each. With the default `.release.cta` remote arrive
(CUTLASS's `ClusterBarrier::arrive(cta)`; the reads it orders are wgmma's, complete at `wgmma.wait_group`) -- the kept form:

Per kernel, us (`probe_k.py`; multicast on / off made <= 5 % difference, the better one shown), against the 12-launch nodes it
replaces (front = adaln_in + v|g GEMM; tail = out GEMM + resgate_adaln + expand + squeeze + resgate_out):

| (L, A) | 12-launch front / core / tail | three-kernel CL 8: front / core / tail / chain | CL 16: front / core / tail / chain |
|---|---|---|---|
| 128, 1 | 5.8 / 3.5 / 22.5 | 19.2 / 3.4 / 44.6 / 67.5 | 13.2 / 3.4 / 32.3 / 49.3 |
| 128, 5 shared | 7.4 / 4.5 / 25.3 | 20.7 / 4.5 / 46.1 / 73.5 | 15.1 / 4.5 / 35.4 / 57.3 |
| 384, 5 shared | 13.5 / 8.9 / 42.8 | 25.1 / 9.1 / 51.8 / 91.6 | 41.6 / 9.0 / 103.0 / 159.8 |
| 768, 5 shared | 23.1 / 20.0 / 65.8 | 50.2 / 20.6 / 103.9 / 177.4 | 72.4 / 20.2 / 172.2 / 275.7 |
| 2560, 5 shared | 81.7 / 160.5 / 234.2 | 175.4 / 161.7 / 370.1 / 721.6 | 233.7 / 161.0 / 543.5 / 940.9 |

PDL on / off changed the chain by <= 5 us. Max resident clusters (`cudaOccupancyMaxActiveClusters`): 15 of 8, 7 of 16.

**Why it loses on H100** (tail trace, L128 A1, CL 8, one cluster, us from kernel start): P1 (y, 12 k-blocks) 4.5 -> 9.1, 0.38 us per
k-block; P2-P4 (residual, statistics exchange 0.8 us, xt stores + release 2.5 us) -> 14.0; P5 (a|b, 2 passes x 12 k-blocks) -> 28.1,
0.55 us per k-block plus ~2 us of h stores between the passes; h release + first z block -> 31.4; P7 (z, 24 k-blocks) -> 41.9, 0.44
us per k-block; end 42.2. Per k-block a CTA takes in A 16 KB + W 12 KB (P1 / P7) or 16 + 24 KB (P5): 64-73 GB/s per SM, the same
TMA intake B200's traces show (~70 GB/s per SM), so the n96 / n192 per-CTA tiles of a column-split cluster run at 55 / 77 FLOP per
byte -- 47-76 % of the tensor rate -- and multicast does not help (it saves L2 reads, not the SM's intake). The exchanges and
epilogues (~13 us of 42) run with the tensor core idle, as on B200, but B200 recovers it with M 256 2-CTA MMAs and TMEM accumulators;
on H100 the accumulators are registers (n <= 256 per warpgroup, x1 kept live), so the per-CTA tile cannot grow. The front's serial
prologue (x loads + statistics 3.5 us, exchange 0.9, xa stores + release 2.6) is longer than adaln_in + cuBLAS v|g (5.8 us) at L128
before its GEMM starts. Larger L adds rounds (30 tiles = 2 rounds of 15 clusters at L768 A5; 100 tiles = 7 rounds at L2560) at a
per-round cost above the cuBLAS tiles' (128 x 192-256, 77-85 FLOP / byte). Kept opt-in (`inf3_h100._ON = False`).

## 3. GEMMs with the row pass in their epilogue (`bo_gemm_sm90.cu`)

The fewer-launch alternative: our own wgmma GEMM (128 x BN tiles, two consumer warpgroups, TMA ring, persistent) with the following
row pass fused. The mainloop alone is at cuBLAS's rate (`probe_ml.py`, us, TF/s in brackets):

| GEMM | cuBLAS | ours, best tile |
|---|---|---|
| M3840 K1536 N768 (squeeze) | 15.3 (593) | 13.9 (654), BN 192 |
| M3840 K768 N768, B MN-major (dog) | 9.8 (464) | 8.3 (549), BN 192 |
| M12800 K1536 N768 | 44.6 (678) | 48.9 (617), BN 256 |
| M36864 K1536 N768 | 120.3 (723) | 131.7 (661), BN 256 |
| M36864 K768 N768, MN-major (dog, training) | 82.6 (526) | 71.1 (611), BN 256 |
| M36864 K768 N3072 | 280.6 (620) | 286.3 (608), BN 192 |

With the epilogues:

| fused kernel | shape | cuBLAS + rows | fused |
|---|---|---|---|
| squeeze + resgate_out (`sq_resout`) | M 128 / 640 / 1920 / 3840 / 12800 / 36864 | 6.4 / 7.8 / 12.3 / 22.4 / 73.3 / 215.9 | 9.6 / 10.0 / 16.0 / 26.2 / 89.0 / 237.7 |
| dog + gate backward, register epilogue | A48 L384 / L768 | 90.0 / 170.3 | 172.8 / 304.8 |
| the same, TMA epilogue (g / ao in, do / dg out by TMA, own loader thread) | A48 L384 / L768 | 90.0 / 170.0 | 104.0-110.2 / 184.7-197.5 |

The fused kernels keep the mainloop's rate but add their epilogue serially (both consumer warpgroups stop the tensor core; the TMA
epilogue's 96 KB buffer also leaves a 3-stage ring). Hiding it needs ping-pong consumers (one warpgroup's epilogue under the other's
mainloop), which on H100 halves the per-warpgroup tile (64 x BN: ~50 FLOP / byte) -- not attempted. Kept as probes only (not dispatched).

## 4. What ships (12-launch step on H100)

- **Long token rungs**: the Hopper inference path served L <= 768 only (the B200 TMEM bound copied into `core_supported`); `pvx` /
  `pvx32` stream P through a TMA ring and have no length bound, and the softmax rows now take 3072 keys (bf16 CPL 12, fp32 CPL 24).
  H100 inference now serves L <= 3072 (`H100_MAX_L`); before, L1536 / L2560 ran the PyTorch module (pair bias recomputed per block).
- **No fp32 copy of the step input**: `adaln_in_rows` skips the copy, the first residual pass reads the input itself
  (`resgate_adaln_rows_in`): 6 B per element less traffic (`runner._H100_NOCOPY`).
- **Expand at few rows**: cuBLAS + the SwiGLU rows instead of `gemm_swiglu_sm90` where its 128 x 256 tiles fill at most half a wave
  or just over one (`runner._H100_EXPAND_ROWS`; us: M128 10.6 -> 6.0, M640 10.6 -> 9.7, M1536 19.2 -> 17.6).
- **One-launch conditioning tables** up to 384 table rows (`bo_gemm_sm90.cu cond_tables`: [LN(c) | c] [W1; W2]^T + bias, weights
  prefetched before the PDL wait; `runner._H100_COND1`, `_H100_COND1_ROWS`).

A/B in one process (`probe_ab.py`, runner level, one block, us; "base" = main):

| (L, A) | base | + no copy | + expand rows | + tables (shipped) | rel vs fp32 module: base / shipped |
|---|---|---|---|---|---|
| 128, 1 | 41.8 | 41.6 | 37.0 | **35.4** | 3.08e-3 / 3.30e-3 |
| 384, 1 | 51.1 | 51.0 | 48.6 | **47.9** | 3.05e-3 / 3.26e-3 |
| 768, 1 | 72.4 | 71.8 | 72.0 | **72.0** (one launch: 78.4, so off above 384 rows) | 3.06e-3 / 3.06e-3 |
| 1536, 1 | 127.2 | 125.8 | 126.2 | **126.2** | 3.06e-3 / 3.27e-3 |
| 2560, 1 | 210.3 | 206.2 | 208.1 | **208.1** | 3.06e-3 / 3.06e-3 |
| 128, 5 shared | 48.7 | 48.1 | 48.7 | **47.4** | 3.06e-3 / 3.28e-3 |
| 384, 5 shared | 85.6 | 84.0 | 84.0 | **84.4** | 3.06e-3 / 3.06e-3 |
| 768, 5 shared | 141.9 | 136.9 | 136.9 | **136.9** | 3.06e-3 / 3.06e-3 |
| 1536, 5 shared | 288.7 | 278.2 | 277.9 | **277.9** | 3.06e-3 / 3.06e-3 |
| 2560, 5 shared | 531.9 | 511.9 | 512.0 | **512.0** | 3.06e-3 / 3.06e-3 |
| 384, 5 per sample | 100.2 | 97.9 | 98.0 | **98.0** | 3.06e-3 / 3.06e-3 |

(The expand-rows rows differ in error because cuBLAS rounds a | b to bf16 before the SwiGLU, as the PyTorch bf16 block does; the
one-launch tables measured slower from 768 rows -- L768 A1 72.0 -> 78.4, L384 A5 per sample (1920 rows) 98.0 -> 115.5 -- hence the
row bound.) Every variant bitwise repeatable.

Module level (`probe_ba.py`: eight `BiasOnlyDiTBlock` calls chained in one CUDA graph, static weights, the per-call conditioning
tables included; main's switches and this branch's alternating in one process, median of three; us per block). Before this work
L1536 / L2560 ran the PyTorch module (pair bias recomputed per call; `probe_step.py` graph, us): 6300 / 6816 / 17318 / 18249 at
(1536, 1) / (1536, 5) / (2560, 1) / (2560, 5).

| (L, A) | main (L <= 768: 12-launch step; above: PyTorch module) | this branch | x |
|---|---|---|---|
| 128, 1 | 45.1 | **39.4** | 1.14 |
| 384, 1 | 52.2 | **51.2** | 1.02 |
| 768, 1 | 71.7 | **70.9** | 1.01 |
| 1536, 1 | 6300 | **124.0** | 51 |
| 2560, 1 | 17318 | **206.8** | 84 |
| 128, 5 shared | 50.9 | **49.2** | 1.03 |
| 384, 5 shared | 84.3 | **82.6** | 1.02 |
| 768, 5 shared | 139.4 | **134.5** | 1.04 |
| 1536, 5 shared | 6816 | **278.6** | 24 |
| 2560, 5 shared | 18249 | **520.9** | 35 |

(At L1536 / L2560 the branch's 12-launch step with main's switches measured 125.5 / 210.6 / 289.1 / 535.8 us: the long-rung gain
is the length bound; the tweaks add 1-4 %.) Against PyTorch compiled (`bench.py target=bias_only_dit level=module mode=inference
precision=bf16-mixed cudagraph=manual mask_prob=0.2`, ms): L1536 A5 shared 1.153 vs **0.283** (4.1x), L2560 A5 shared 2.701 vs
**0.510** (5.3x), L1536 A1 0.922 vs **0.130** (7.1x), L2560 A1 2.282 vs **0.212** (10.8x).

## 5. The fused bf16 backward (B200 T6) on H100

Training step, bf16, A = 48, kernels per step (`probe_trainbrk.py`, torch.profiler, eager; the autograd accumulation kernels of the
probe's repeated backward excluded): L384 2.06 ms, L768 4.09 ms over 43 launches. L768: cuBLAS GEMMs 2.22 ms (54 %; the weight
gradients ~0.72 ms at ~72 % of the tensor peak), row passes 1.23 ms (30 %; res_adaln_b_bwd 224, swiglu_bwd 186, adaln_a_bwd 138,
res_adaln_b 118, res_c 115, swiglu_rows 113, gate_bwd 100, res_c_bwd 94, adaln_a 81 us), attention side 0.62 ms (pvx 161 + 122, dpb2
115, pair_bias_bwd 121, pair_bias 75, softmax_t 24 us).

B200's fused backward fuses the row passes into its GEMMs' epilogues (`bo_bwd_tail`: res_c_bwd, dh + SwiGLU backward, dxt,
res_adaln_b_bwd, dog + gate backward, on CTA pairs with TMEM accumulators) and runs dV + dbias in one launch (`bo_pvdpb`), the six
weight gradients in one (`bo_wgrad`) and the bias gradient on CTA pairs at L768 (`dpbx2_sm100`); it measured 1.6-3.4 % of the step on
B200. On H100 the G4 piece (dog = dy Wo with the gate backward in the epilogue) was built twice (section 3) and is 1.1-2x slower
than cuBLAS + `gate_bwd_rows`; the dh + SwiGLU-backward piece was already measured slower in 2026-10 (509 vs 318 us,
docs/gpus/h100/bias_only_dit/bias_only_dit.md). The CTA-pair bias gradient has no H100 form (no 2-CTA MMA: a cluster pair could
only multicast the key tile, which saves L2 reads, not the per-SM intake that bounds `dpb2`). `bo_wgrad`'s gain on B200 came from one
tile per CTA over all rows sharing L2; H100's cuBLAS weight gradients already run at ~72 % of the tensor peak. Not shipped: the
H100 backward keeps its 23-launch composition.
