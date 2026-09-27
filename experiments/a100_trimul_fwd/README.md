# A100 TriMul forward + training (CUDA, sm_80) — development log

Goal: TriMul inference forward on A100 in CUDA, ideas carried over from the H100 kernels (`experiments/trimul_k1k3_inference`), measured
against Anthropic's published kernels ([baseline](../a100_anthropic_baseline/README.md)) and a speed-of-light composite.

- **Structure** (same decomposition as sm_90 / Anthropic): K1 (LN_in + gated projections + pair mask → channel-major planes) →
  strided-batched contraction (cuBLAS `bmm`; bidirectional = outgoing NT half + incoming TN half) → K3 (LN_out + projection, LN_in + gate,
  residual).
- **Card**: A100 80GB PCIe, 300 W cap. Every stage runs at the power cap (K1 ≈ 1245 MHz, contraction ≈ 1050 MHz, K3 ≈ 1275–1305 MHz;
  `clock_probe.py`), so dropped instructions also buy clock.
- **Harness**: `bench.py` (fixture = the baseline's: B1, BF16, 10% masked tokens, fp32 reference module), CUDA-graph replay median of
  7 × 50, per-kernel CUPTI times. `ncu_stalls.sh` / `sass_top.py` for Nsight Compute stall and SASS breakdowns.

## SoL definition

Composite floor of the three-kernel decomposition (the H100 records' convention): per kernel `max(essential bytes / BW, FLOP / tensor)`,
summed. Ceilings **measured on this card** (`probe_peaks.py`, `records/peaks-gpu08.json`): streaming copy **1.60 TB/s** (83% of the
1.935 TB/s spec) and cuBLAS large-GEMM **240 TFLOP/s** under the power cap (spec 312; an ALU-only HMMA microbenchmark reaches ~300,
`micro_hmma.py`, but only without memory traffic).

| | bytes / token | FLOP / token |
|---|---|---|
| K1 | 256 + 4·CH | 2·128·4·CH |
| contraction | 6·CH | 2·CH·L |
| K3 | 2·CH + 512 | 2·(CH·128 + 128·128) |

(CH = 256 bidirectional, 128 one direction.) Floors: bidirectional L384 397 µs (K1 161 / C 141 / K3 94), L768 1988 µs (644 / 967 / 377);
one direction L384 222 µs, L768 1088 µs.

## Current (records/current.json, gpu08)

| variant | L | op µs | vs Anthropic best | % of SoL | K1 µs (% floor) | contraction | K3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bidirectional | 384 | 600 | 790 → **1.32×** | 66% | 225 (72%) | 183 (77%) | 156 (60%) |
| bidirectional | 768 | 2780 | 3417 → **1.23×** | 72% | 960 (67%) | 1142 (85%) | 643 (59%) |
| one direction | 384 | 365 | 407 → **1.12×** | 61% | 143 (56%) | 93 (76%) | 124 (57%) |
| one direction | 768 | 1616 | 1776 → **1.10×** | 67% | 559 (58%) | 591 (82%) | 457 (62%) |

Run-to-run / node-to-node spread is about ±2% (gpu01 vs gpu08).

Accuracy: rel-RMS vs the fp32 module 2.70e-3 (bidirectional) / 2.69e-3, better than the bf16 PyTorch module itself (3.1e-3) and equal to
Anthropic's rows (2.65–2.87e-3).

## What each step bought (bidirectional L768 K1 / K3, µs)

| step | K1 | K3 | note |
|---|---:|---:|---|
| v1: 8 consumer warps, 1 CTA/SM, CTA barrier per weight block; K3 1 CTA, LN folded into weights | 1550 | 744 | instructions 7.5× the HMMAs |
| v2: producer-free "last warp refills" ring, granule-major weights, tanh-FFMA epilogue | 1806 | | weight waits (spin 117×/step): slowest warp refills |
| v3: two independent 4-warp CTAs per SM, 2-slot ring | 1653 | | 16-way smem conflicts in the weight fill |
| v3b: weights packed granule-major on the host (straight copy), LN pass 128 B per 8 lanes | 1144 | | tensor 51% |
| v5: in-warp software pipeline across channel groups (MMA of one ∥ epilogue of the other, no branch inside) | 1048 | | tensor 57% |
| v6: next tile's LN spread over the current tile's steps, mask table, immediate-offset addressing | 1004 | | |
| K3 v2: two independent 4-warp groups sharing resident weights (32-token tiles) | | 698 | |
| K3 v3: row statistics on the tensor cores (Σx = x·1, Σx² = diag of the Gram block, B operands = the A registers) | | 641 | bit-identical |
| v11: token-pair row order in the A fragments (no shuffle / prmt in the epilogue) | 1013 | | |
| v12–13: z-tile swizzle for the permuted rows, conflict-free LN affine | 981 | 654 | |
| v15: masked tokens as zero LayerNorm rows (no epilogue mask) | ≈ | | neutral, simpler |
| K3: 3 groups for CH = 128 | | 490 → 467 (one dir.) | kept |


More rejected (measured): direct register → plane stores instead of staging (K1 +6%: 64-bit address math, 4× STG, predicate
branches); K1 tile 96 tokens (tail 11% → 1.6% at L384, but +33% weight streaming: K1 +12%); K3 one statistics exchange + early z release
(±, reverted); K3 without any group barrier, statistics on the tensor cores inside the MMA loops and warp 0 refilling (K3 +25%: the refill
warp becomes the straggler, X wait 4% → 20%); one TN contraction over 256 channels (−6% at L384, −2% at L768 of the contraction, needs a
transposed out-half plane store that conflicts with K1's token-pair order). K3 per-phase cycle profile (`k3_prof.py`, `-DK3_PROF=1`):
statistics + exchanges ~40% of a tile, memory waits ~4%. K1 tail split (full waves of 128-token tiles + one wave of 64-token tiles
for the ragged remainder, `-DK1_TAIL=1`): K1 +6% at L384 -- a tile's cost is dominated by the weight-block stream and per-step barriers,
which a 64-token tile does not halve.
K1 per-phase profile (`k1_prof.py`, `-DK1_PROF=1`, bidirectional L768): 3166 cycles per warp-step (tensor minimum 1024) = MMA +
epilogue chunks 66%, weight-block wait 10%, staged plane stores 10%, spread LayerNorm 13%, step barrier 0.6%. Two follow-ups, both
slower and reverted: (a) weights and z filled by disjoint warps (a cp.async mbarrier arrival tracks all earlier copies of its thread) with
the plane stores and the LayerNorm pass moved inside the MMA group loops -- 3450 cycles/step, the weight wait stayed (~256) and the step
barrier grew to ~250; (b) one 8-warp CTA per SM with 256-token tiles and a 3- or 4-slot weight ring (half the weight streaming, deeper
look-ahead) -- K1 +4% (L768) / +10% (L384): eight warps meeting at every step barrier cost more than the ring saves. K3 hybrid (per-warp full-K statistics on the tensor cores inside the MMA loops, no
exchange, but the group barrier + all-thread refills kept): K3 +11% -- the extra ~50% HMMAs lengthen the projection / gate phases by more
than the removed exchange phases. The quarter-K statistics + exchange (current K3) remain the best measured.

Also rejected: producer warp (9 warps cap registers at 168), last-warp refill, 3-stage epilogue split (no change), granule-major K3
weights (slower). (3 K3 groups only fit CH = 128: CH = 256 has neither the registers nor the shared memory.)

## Is SoL90 reachable? (2026-09-25 analysis)

Not proven impossible -- the floor's denominators are measured achievable rates, not hardware bounds (read-heavy streams reach 1.75 TB/s,
an HMMA-only loop ~300 TFLOP/s) -- but it requires every kernel at the edge of this card's power-capped limits:

| | bidirectional L768 | bidirectional L384 |
|---|---:|---:|
| SoL90 target | 2209 µs | 441 µs |
| contraction at 90% of its floor | 1074 µs | 157 µs |
| left for K1 + K3 (after ~30 / 10 µs gaps) | ~1105 µs | ~274 µs |
| K1 + K3 floors | 1021 µs | 255 µs |
| **required K1 / K3 efficiency** | **~92%** | **~93%** |

- K1 at 92% (L768, compute-bound) means ~225 TFLOP/s: cuBLAS's best pure GEMM under the 300 W cap is 240 (~96% tensor utilisation), and K1
  must also move 755 MB, run a MUFU per output, the LayerNorm and the transposing store within the same power.
- K3 needs ~51-60% tensor occupancy at the floor pace while streaming DRAM near its limit.
- `mma.sync` issues in order and the registers (A fragments) / 164 KB of shared memory cap the warps per SM; measured structures plateau
  at ~60% (K1) / ~40% (K3) tensor activity.
- The H100 kernels, with TMA and asynchronous WGMMA, stopped at 81-86% of the same kind of composite floor.
- No algorithmic change lowers the floor (the contraction is a global dependency; LN_out needs all channels; channel-chunked L2 residency
  re-reads z and loses at L768).

Realistic ceiling estimate: K1 ~80%, K3 and contraction ~88% → op **~83-85%** of SoL. Next step taken: a custom contraction (both
directions in one launch), then K1 / K3 restructuring.

## Custom contraction (csrc/contract_sm80.cuh)

128 x 128 x 32 tiles, cp.async 4-stage ring, 4 warps (64 x 64), 2 CTAs/SM, NT and TN in one launch (both bidirectional halves), fragments
double-buffered across k16 steps, lane-constant ldmatrix offsets, staged 16 B coalesced epilogue (the first version's 4 B stores were
`lg_throttle`-bound). Standalone vs cuBLAS (`contract_bench.py`): bidirectional L384 193 vs 200 µs (1.04×), one direction L384 0.94×,
bidirectional L768 1257 vs 1226 µs (0.97×); identical accuracy. `TRIMUL_CONTRACT=auto` (default) uses it for bidirectional L ≤ 512, cuBLAS
elsewhere. It matches the same-tile cuBLAS kernel (`ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5`) but does not beat it
by the ~15% the SoL90 budget would need at L384.

## Where the time is now

- K1: tensor pipe 62% active; per-tile turnover (LayerNorm, fragment load, mask) and the epilogue's MUFU/store chain in the same warp as
  its HMMAs. The CTA-barrier per weight block couples the 4 warps of a CTA.
- K3: 2 warps per SM sub-partition, serial chain X → stats exchange → projection → z stats exchange → gate → store; tensor ~42%.
- Contraction: cuBLAS at the power-capped tensor ceiling for L768 (≈208 TFLOP/s at 1050 MHz), 77% of the memory floor at L384.
- **SoL90 status: not reached.** With the cuBLAS contraction as is, SoL90 at L384 would need K1 + K3 at ~98% of their floors; at L768
  both at ~93%.

## Training (forward + backward) — `trimul_train.py`, `train_bench.py`

Fixture: `train_bench.py`. It uses the baseline's module and shapes, with dropout p = 0.25 (the module's `_make_drop_row_scale` is patched to a
fixed row scale, and the same scale goes to the fp32 reference). The gradients of z and all 10 parameters are compared with the fp32 module.
Timing is CUDA-graph fwd+bwd replay, median of 7 × 20. The correctness graph is released before capture: it pins z's and the weights'
AccumulateGrad nodes with default-stream metadata, and the captured backward then fails with `cudaErrorStreamCaptureImplicit`.

| fwd+bwd, ms | L | **this** | % of SoL (`sol_train_us`) | Anthropic | engine (Triton) | cuEq |
|---|---:|---:|---:|---:|---:|---:|
| bidirectional | 384 | **2.55** | 64% (1.64) | — (no backward) | 3.198 (1.25×) | 4.435 (1.74×) |
| bidirectional | 768 | **11.21** | 69% (7.77) | — (no backward) | 14.154 (1.26×) | 18.270 (1.63×) |
| single | 384 | **1.53** | 65% (1.00) | — (no backward) | 1.976 (1.29×) | 2.614 (1.70×) |
| single | 768 | **6.52** | 71% (4.61) | — (no backward) | 8.170 (1.25×) | 10.609 (1.63×) |

Training SoL (`trimul_a100.sol_train_us`) uses the same per-kernel rule as the forward's. It is the forward composite plus B1, the
weight-gradient GEMMs, the contraction backward, B7src and B8, each bounded by max(bytes / 1.60 TB/s, FLOP / 240 TFLOP/s). At
bidirectional L768 the terms are fwd 1.99 + B1 0.95 + wgrad 0.47 + contraction backward 1.93 + B7src 1.29 + B8 1.14 = 7.77 ms.

Accuracy (rel. L2 vs the fp32 module, every shape): out 2.9e-3, dz 3.4e-3. The worst parameter gradient is 5.3–5.8e-3 (input projections,
ln_pair); ln_out bias is 6e-4. The bf16 PyTorch module's own gradients reach 7.5–7.8e-3 worst on the same fixture.

**sm_90 algorithm on sm_80 (current default).** The backward now follows the H100 training kernels (`src/.../trimul_inproj/cuda/h100_*`,
`h100_sources/b1`, `b7_768/joint.cu`):
- **The forward saves the backward's inputs.** Training K3 (`k3_train`) writes the LayerNorm statistics (μ_o, r_o, μ_i, r_i) and
  x_n = LN_in(z) with its affine. B1 then recomputes neither LayerNorm, and its gate is x_n · W_ogᵀ with no fold.
- **B7 joint** (`csrc/b7j_sm80.cuh`, cooperative launch). Groups of 16 source CTAs (one packed 64-row weight block each; dg / dp; dW in
  registers) and 8 consumer CTAs (the dx_n GEMM, LN_in backward, residual). The two roles meet in an **L2 ring**: sources write each dgp
  tile with an L2 evict_last policy and publish a gpu-scope release flag; consumers acquire the flags, stream the ring slot through
  cp.async.cg, and release the slot. On sm_90 this is TMA bulk stores + cluster barriers. The dgp derivatives no longer make a DRAM round
  trip. Tuned: C = 8 consumers per group at CH = 256 (6 at CH = 128), 8 ring slots (8–9 groups of 128-token tiles).
- **B1 with the W_o gradient on chip** (`csrc/b1g_sm80.cuh`, `A100_B1G`). A_o = d_o r_o goes to the tile, so acc' = A_o · Wo' = r dx̂
  and dX = acc' − mean − x̂ · mean(acc' x̂) needs no r. The two 4-warp groups take consecutive 32-token tiles in lockstep, and all
  8 warps accumulate G = A_oᵀXᵀ over the 64 tokens in registers. This removes the A_o store and the G GEMM. It is the default at CH = 128
  (6.62 vs 6.75 ms at L768). At CH = 256 the 128 accumulator registers spill (224 B; 48 B after moving dy to smem, streaming the
  fragments and granule-major resident weights, `archive/b1g_sm80_v6.cuh`). The latency-bound B1 also absorbs the extra 39 GFLOP almost
  1 : 1, while the split-K GEMM runs at the DRAM floor. So bidirectional keeps B1 + GEMM (11.33 vs 11.49–11.59 ms).
- The W_og gradient is d_gᵀ · x_n (cuBLAS, at the DRAM floor; sm_90 runs this as B1's second "gate phase").

**Earlier pipeline (still selectable: `A100_B7J=0`, `A100_B1G=none`).** Forward: K1 → contraction → K3 (K3 applies the dropout row
scale). Saved: z, the planes, X, ds, mask. Backward:

1. **B1** (`csrc/b1_sm80.cuh`, output side, token tiles of 32, 2 × 4-warp groups). It recomputes the X / z LayerNorm statistics (tensor
   cores, as in K3), o, and the gate. It computes d_o and d_g, then dx̂ = d_o · Wo' (ldmatrix.trans of the resident Wo'), then the LN_out
   backward, giving the dX planes. It also emits A_o = d_o r_o, A_g = d_g r_i, d_g (into the dx operand buffer) and the per-token statistics.
   It writes **x_n = LN_in(z)** once for B7src. It also emits per-group partial sums of A_o μ_o, d_o, A_g μ_i and d_g, the vectors of the
   rank-1 LayerNorm folds.
2. **Weight gradients of W_o / W_og.** Two long-K GEMMs against the raw X / z, plus rank-1 folds: H = A_oᵀXᵀ − (A_oᵀμ)1ᵀ,
   dW_o = H γ + S_o β, dγ = Σ W_o ⊙ H, dβ = Σ W_o S_o. cuBLAS picks a non-split-K kernel for the 128 × 256 × L² GEMM (0.99 ms at L768), so
   the K split is done explicitly: a batched GEMM over 32 token chunks with fp32 outputs, then a sum. That takes 0.32 ms, about the DRAM
   floor (`gemm_tk_bench.py`).
3. **Contraction backward.** Four products per bidirectional half-pair, cuBLAS `bmm`. `contract_multi` (all four in one launch; the tile
   gained an NN mode, `A100_CBWD=custom`) matches but does not beat it: 2.01 vs 1.91 ms at bidirectional L768, where cuBLAS is at the
   tensor ceiling.
4. **B7src** (`csrc/b7_sm80.cuh`, channel-stationary). A CTA owns one 64-row block of the packed input weights for a split of the tokens,
   so its dW block stays in registers for the whole launch. Per 128-token tile it runs (g', p') = MMA(x_n, 0.5·W). It then forms dp = dA·m·s
   and dg = dA·m·p'(1 − th²), with th = tanh g', so one MUFU replaces exp + rcp. The dgp tile is written token-major for B8, and
   dWᵀ += x_nᵀ · dgp is accumulated.
5. **B8** (`csrc/b8_sm80.cuh`). dx_n = [dgp | d_g] · [W_in ; W_og] on the contraction's main loop (A K-major, B MN-major). The epilogue
   stages z / dy in the drained ring, reduces the row sums (quad shuffles + a 2-warp exchange), writes dz = LN_in backward + residual as
   16 B rows, and accumulates per-CTA dγ / dβ partials with smem atomics.

**What each step bought (bidirectional L768 fwd+bwd, ms).**

| step | ms |
|---|---:|
| v1: B1 + B7src + cuBLAS dx GEMM, LN_in backward and every fold in torch | 21.94 |
| B8 (dx GEMM + LN_in backward + residual fused) and B1 fold partials: removes ~15 elementwise / reduce / cast kernels and the fp32 dxn | 14.00 |
| x_n written once by B1 (B7src had re-normalised z in each of its 16 channel-block CTAs); tanh form in B7src | 12.69 |
| B1: dy / ds prefetched to registers at tile start (were loaded in the d_o chain); X fragments streamed instead of held | 12.32 |
| explicit split-K for G = A_oᵀXᵀ | 11.87 |
| B8: no spill (γ from smem, row / column epilogue passes; was 160 B of stack) | 11.74 |
| B7src v4: the next tile's dAB + mask issued under the dgp store and the dW MMA | 11.65 |
| dW_og = d_gᵀ · x_n straight from the stored x_n (no A_g store, no LN_in fold vectors, no Gg over raw z) | 11.48 |
| B7src v5 (`b7pair_kernel`): one 8-warp CTA per SM, two weight blocks sharing a double-buffered x_n tile | 11.49 (≈ v4: 11.59 in the same job) |
| B7 joint (sm_90 algorithm, L2 ring, C = 8, 8 slots) instead of B7src + B8 | 11.55 (≈ the pair: dgp traffic was not the limiter) |
| forward saves the LayerNorm statistics + x_n; B1 drops both LayerNorm recomputations and the x_n write | 11.25 |
| (single) B1 with the on-chip W_o gradient | single 6.75 → 6.62 |
| weight-gradient GEMMs + folds on a side stream under the (tensor-bound) contraction backward (`A100_OVERLAP`) | 11.30 → 11.21 (L384 2.62 → 2.55) |
| K1 saves the LN_in (mean, rstd) of every token; K3 skips its z statistics HMMA + exchange (`TRIMUL_ZST`, also inference) | K3 −3…5%, training −1% |

Tried, no gain: b7j deferred flag publish (the fence was not the limiter); deeper b7j rings (16 / 24 slots: the sources' ~10% slot waits
stay -- the consumer side is ~7% short of throughput and the source count is fixed at one per weight block -- and the L2 footprint
grows); K1 8-warp / 256-token tiles for CH = 128 as well (K1 143 → 162 µs at L384). B1 with 3 groups for CH = 128 (168 registers: 276 B spill, 7.60 vs 7.12 ms). B7src v5's double-buffered x_n (the
load latency was not what bound it: 8 warps/SM of dependent MMA → elementwise → MMA chains). Tanh gate + hoisted `t % L` in B1 (fewer instructions, same time: B1 is latency-bound at 8 warps/SM with 255 registers
and ~7 group barriers per 32-token tile). Smem-staged mask and hoisted addresses in B7src (kept: fewer instructions, same time).
The custom 4-product contraction backward.

**Where the time is (bidirectional L768, per-kernel CUPTI, `prof_train.py`, ≈10.7 ms of kernels).**
- B7src 2.19 ms. Its floor is ~1.3 ms, tensor and DRAM about equal; tensor pipe is 48% active.
- Contraction backward 1.91 ms (at the ceiling). Forward contraction 1.02 ms.
- B1 1.62 ms. Its floor is ~0.85 ms (memory); it is latency-bound.
- B8 1.36 ms. Its floor is ~0.85 ms; its epilogue waits on the z / dy staging.
- K3 1.02 ms, K1 0.81 ms, G + Gg 0.52 ms.

**Power.** The whole step runs at the 300 W cap (throttle reason 0x4, SM clock ≈1170–1185 MHz; `clk_bench.sh`). This is also why the
graph time (11.8 ms) exceeds the sum of kernel times measured with profiler gaps (10.7 ms).

**Not pursued.** B8 recomputing (g, p) from x_n + dA instead of reading the dgp round trip. It saves ~1.6 GB of DRAM traffic at
bidirectional L768, but doubles B8's tensor work (+262 kFLOP/token), and the tensor pipe is the binding resource at these clocks, so the
estimated net is ≤ 0.25 ms. The sm_90 answer, a cluster ring feeding weight-stationary consumers (`h100_sources/b7_768`), has no sm_80
equivalent short of a global-memory ring.

## Code map and defaults (after the 2026-09-26 cleanup)

| stage | file / kernel | default |
|---|---|---|
| forward K1 | `csrc/k1_sm80.cuh` `k1_kernel` (`k1z` binding: also saves the LN_in statistics) | `TRIMUL_ZST=1` |
| contraction | `csrc/contract_sm80.cuh` (bidirectional L ≤ 512) / cuBLAS | `TRIMUL_CONTRACT=auto` |
| forward K3 | `csrc/k3_sm80.cuh` `k3_kernel` (`k3z`: K1's statistics; `k3_train`: also saves the four statistics + x_n) | |
| B1 | `csrc/b1_sm80.cuh` (CH = 256: A_o + split-K G GEMM) / `csrc/b1g_sm80.cuh` (CH = 128: G on chip) | `A100_B1G=128` |
| weight gradients | `trimul_train.py` `weight_grads` on a side stream | `A100_OVERLAP=1` |
| contraction backward | cuBLAS `bmm` | |
| input side | `csrc/b7j_sm80.cuh` `b7j_kernel` (joint, cooperative, L2 ring) | `A100_B7J=1`, C = 8 / 6, 8 ring slots |
| input side, fallback (T % 128 ≠ 0) | `csrc/b7_sm80.cuh` `b7pair_kernel` + `csrc/b8_sm80.cuh` `b8_kernel` | |

Removed in the cleanup (all measured and rejected; the sources stay in `csrc/archive/pre_cleanup/`):
- the K3 LN_out-statistics pre-pass (`TRIMUL_XST`, +3–8% inference);
- the deferred b7j flag publish (`B7J_DEFER`);
- the 4-warp B7src kernel (same speed as the pair kernel);
- the one-launch 4-product contraction backward (`contract_multi`, `A100_CBWD=custom`: 2.01 vs cuBLAS 1.91 ms).

**Why the CUDA path stops here (2026-09-26).** The graph replay has no idle gaps (7 µs per bidirectional L768 step). Its 1.4 ms gap to a
short profiled run is all clock: the sustained step runs at the 300 W cap (≈1185 MHz, `clk_bench.sh`), so it is power-bound. With the
FLOPs fixed, the remaining levers are energy per step (L2 re-reads of x_n by the b7j sources, the consumers' weight streaming, the 7%
consumer shortfall at a fixed 16 sources per group), each worth a few percent.
