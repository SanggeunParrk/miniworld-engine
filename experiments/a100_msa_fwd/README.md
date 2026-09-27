# A100 OPM / PWA forward (CUDA, sm_80) — development log

Goal: inference forward of `OuterProductMean` and `MSAPairWeightedAveraging` on A100 in CUDA, measured against the
[A100 baseline](../a100_anthropic_baseline/README.md) (compiled PyTorch, the current engine, and Anthropic's published kernels).

- **Fixture** (same as the baseline): B=1, BF16, MSA depth S=1024, d_msa 64, d_pair 128. OPM uses hidden 32, `normalize_before_proj`,
  a ~10%-false token mask, and the pair residual. PWA uses 8 heads × 32 and a ~10%-false key mask, and includes the msa residual.
- **Harness**: `bench.py` checks rel-RMS against the fp32 module on the same weights, then times the median of 7 × 50 CUDA-graph
  replays and records per-kernel CUPTI times. `ncu_op.sh` gives Nsight Compute stalls, pipes, and instruction mix.
  `gemm_probe.py` compares the GEMM1 layouts. Runs go through `sbatch job.sbatch <cmd>` (one A100, `env.sh`).
  Compile variants use `bench.py -D NAME=VAL`.
- **Card**: A100 80GB PCIe with a 300 W cap. Ceilings measured on this card are in `records/peaks-gpu08.json`: 1.60 TB/s copy and
  240 TFLOP/s large GEMM. Under load every kernel here runs at 1030–1260 MHz.

## Decomposition

The fusion boundaries are the H100 ones (`src/miniworld_engine/integrations/csrc`). OPM follows `opm_epilogue.cu`: cuBLAS computes
the grouped GEMM `O[(i,c),(j,e)]`, and one fused epilogue does the permute, /n, cast, projection, bias, and residual. H100 does not
fuse GEMM1 into GEMM2 either. PWA follows `pair3.cu`, `ln_vg.cu`, and `pwa_fwd3.cu`: a CTA covers 128 i × an s-pair, one ring
walks (head, j-chunk), and at each head boundary it runs the gate GEMM and sigmoid ⊙ o (as register A fragments), then adds the
out-projection into accumulators kept over all heads. The A100 versions swap wgmma/TMA for mma.sync, ldmatrix, and cp.async.

**OPM** (`msa_a100.opm_forward`)
1. `opm_prologue`: LN(msa) → left/right projection. LN γ is folded into W and β into a bias, the product runs on mma, then the token
   mask is applied. Output is `a, b [S][L][32]` (the GEMM1 operands). `opm_maskbits` packs the mask into bits `[L][S/32]`.
2. GEMM1 in cuBLAS: `O = aᵀ b` → `[(i,d),(j,e)]` bf16.
3. `opm_epilogue`: for each 128-pair × 128-output tile, `W_out · vec(O_ij)` with K=1024. The `[i,j,d,e]` permute is folded into the
   operand loads. Then `/ max(1, popc(bits_i & bits_j))`, + bias, + residual. Dividing after the projection is the same as the
   module's divide-before-projection because the bias is added afterwards.

**PWA** (`msa_a100.pwa_forward`)
1. `pwa_pair`: one CTA per row i. LN_z → proj_z (a single n8 mma tile, γ folded) → key mask (the bf16 minimum, as in the module)
   → softmax over j → `w [8][L][L]` bf16.
2. `pwa_value`: LN_m → value projection, written head-major as `v [8][j][s·32+d]`, the contraction's B layout.
3. `pwa_main`: each CTA is 128 i × 2 s. A single cp.async ring runs over the (head, 64-j chunk) sequence. At each head boundary the
   kernel computes the gate from LN(msa), which it recomputes from the msa tile it already loaded for the residual, so no y buffer
   is needed. It then forms sigmoid(g) ⊙ o directly from the accumulators as A fragments and adds that head's out-projection into
   register accumulators. The residual is added at the end.

## Current (OPM, PWA fused: records/v4.json; PWA split: records/pwa_split_v12.json)

| op | L | µs | vs best existing | % of SoL | rel-RMS vs fp32 (bf16 module) |
|---|---:|---:|---:|---:|---|
| OPM | 384 | 1813 | 2343 (PyTorch compile) → **1.29×** | 80% | 1.67e-3 (1.68e-3) |
| OPM | 768 | 7048 | 8892 (PyTorch compile) → **1.26×** | 82% | 1.67e-3 (1.68e-3) |
| **PWA split** | 384 | **911** | 1919 (Anthropic g_fo2p) → **2.11×** | 53% | 1.68e-3 (1.69e-3) |
| **PWA split** | 768 | **2465** | 5174 (engine main) → **2.10×** | 66% | 1.67e-3 (1.68e-3) |
| PWA fused (H100 boundaries) | 384 | 1137 | → 1.69× | 43% | 1.68e-3 |
| PWA fused (H100 boundaries) | 768 | 3308 | → 1.56× | 49% | 1.67e-3 |

SoL is `max(FLOP / 240 TF, essential bytes / 1.6 TB/s)` for the whole op. For OPM that is GEMM1 + GEMM2; for PWA it is the
contraction plus the projections.

### PWA split path (`pwa_forward(..., split=True)`, bench op `pwa_split`)

This departs from the H100 fusion boundaries. The out-projection, and with it the gate, leave the contraction kernel:

`pwa_compact` (valid keys) → `pwa_pair` ∥ `pwa_value` (v only) → `pwa_ctr` (per head: contraction only, o `[S·L][256]` bf16 in
fragment order) → `pwa_out2` (LN(msa) from the staged tile, gate GEMM, sigmoid ⊙ o, out-projection over heads, residual)

- **Key compaction.** A masked key only contributes exp(min − max) = 0 to its softmax row. `pwa_compact` lists the n valid keys
  (n_pad = round_up(n, 16), both kept on the device, so the path is graph-safe). `pwa_pair` gathers and normalizes only those z rows
  and writes w_c[h][i][k] (zeros for k in [n, n_pad)). `pwa_value` projects only those tokens, and `pwa_ctr`'s K loop runs to n_pad,
  with a partial last chunk on its own code path.
  - If every key is masked, all keys are listed, and the pair kernel's mask test reproduces the module's uniform row.
  - The saving scales with the masked fraction. Fixture (10% masked): −3.3% / −6.9% at L384 / L768. 50% masked: 665 / 1770 µs
    (L768 at 91% of the dense SoL).
  - Checked against fp32: no mask, all masked, 37% (n not a multiple of 16), 50%.
  - `PWA_COMPACT=0` gives the dense path.

- `pwa_ctr` no longer holds the 64 fp32/token out accumulator. It uses a cutlass-shaped tile: one head × 128 i × 4 s, 4 warps of
  64 × 64, 2 CTAs/SM, 8 ldmatrix per 32 mma.
- The gate GEMM moved to `pwa_out2`, a memory-bound kernel with idle tensor time. `pwa_out2` also normalizes the msa tile it
  already reads for the residual, so no y buffer exists anywhere.
- o is written in fragment order: within a head, position 8q + 2dt + e holds d = 8dt + 2q + e. `pwa_out2`'s lane q then reads
  its accumulator-layout values as one 16 B vector per (row, head).
- The cost is the o round trip: 0.8 GB at L768.

| L768 kernel | µs | floor | note |
|---|---:|---:|---|
| pwa_ctr | 1656 | 1290 (FLOP) | 187 TF = 78% of the power-capped 240 TF |
| pwa_out2 | 424 | 376 (602 MB) | 89% |
| pwa_value | 408 | 314 (502 MB) | 77%; the msa rows are read in 128 B pieces |
| pwa_pair | 145 (≈ 60 net) | 94 | side stream, alongside pwa_value |

| step (split path, L768 total) | µs |
|---|---:|
| v1: cuBLAS addmm for the out-projection (plus its 100 MB DtoD copy of msa), LN recomputed in every head's epilogue | 3457 |
| v2: custom out-projection kernel, y written once by pwa_value | 3216 |
| v3: pwa_ctr 64-j stages (half the barriers), u through smem as 16 B stores, 32-bit copy offsets | 2964 |
| v4/v5: pwa_value grid ×8, bias as the accumulator init, j-major steps (512 B msa runs), pwa_pair on a side stream | 2917 |
| gate moved to the out kernel (`pwa_ctr` 1941 → 1566 µs, out 368 → 558) | 2791 |
| o in fragment order (out2 552 → 468) | 2724 |
| no y: out2 normalizes the staged msa tile (value 518 → 408, out2 → 424) | 2629 |
| key compaction, n_pad = round_up(n, 16) (same-job dense: 2649) | 2465 |
| conflict-free o staging in pwa_ctr: u32 slot dt ^ ((row >> 1) & 3), undone with two conditional swaps on the read-out (same job: 2498 → 2465) | **2465** |

Measured and rejected:
- **s-chunk stream pipeline** (value(c+1) ∥ ctr(c) ∥ out(c−1), ring buffers; `pwa_forward_pipelined`): 2921–3538 µs vs 2955.
  `pwa_ctr` fills an SM (2 CTAs, 128 KB smem, 60 K regs), so the streams time-share SMs instead of overlapping, and small chunks
  add fill and tail waste.
- `pwa_ctr` fragment double-buffering: ±0.
- `pwa_out2` with 8 warps sharing a 64-token tile (head halves reduced through smem): 590 vs 558 µs.
- `pwa_out2` with 12 or 16 warps (register-prefetched o): 576 / 699 µs (16 spills).
- Transposing o in registers during `pwa_ctr`'s read-out, to avoid the 4-way conflict on its fragment-order smem stores:
  1860 vs 1650 µs. One row per thread breaks global-store coalescing.
- HMMA microbenchmark (`micro_hmma.py`): with 4 or more independent accumulators, one warp per SMSP already reaches ~300 TF.
  Occupancy is not the limit; barriers, epilogues and memory waits are.

### Per kernel at L768 (µs, floor = max(bytes / 1.6 TB/s, FLOP / 240 TF))

| kernel | µs | floor | % | what bounds it (ncu) |
|---|---:|---:|---:|---|
| OPM GEMM1 (cuBLAS 128x128 nt) | 5737 | 5150 | 90% | tensor under the power cap (~1.1 GHz) |
| opm_epilogue | 1135 | 940 (O 1.2 GB + residual/out) | 83% | barrier / wait / mio; tensor 54% (v2 profile) |
| opm_prologue | 146 | 125 | 86% | memory |
| pwa_main | 2796 | 1510 (FLOP) | 54% | 8 warps/SM (238 regs); tensor 45%, wait / short_scoreboard / math throttle |
| pwa_value | 460 | 313 (v write 402 MB) | 68% | short/long scoreboard |
| pwa_pair | 113 | 94 | 83% | memory |

## What each step bought

| step | effect |
|---|---|
| v1: all five kernels, correct on the first run | OPM 1824 / 7110, PWA 1592 / 5227 µs |
| pwa_main v2: 64-j chunks, precomputed smem offsets (XOR per k16 step), counter-driven chunk walk (no `/`, `%`), Wo as pre-packed per-lane B fragments via 16 B global loads | main 4619 → 2853 µs (L768); instructions 1.03e9 → 3.7e8, HMMA share 9% → 24%, tensor active 26% → 45% |
| pwa_value v2: next tile prefetched into registers, outputs staged through smem for 16 B contiguous stores | 489 → 460 µs |
| opm_epilogue v2: precomputed copy / ldmatrix offsets, stage depth templated | 1257 → 1185 µs |
| pwa_main v3 (H100 fwd3 stage shape): 128 j per stage (2 stages, 160 KB), Wo fragments / bg loaded during a head's last chunk, ldmatrix fragments double-buffered across k16 steps | 2853 → 2796 µs |
| opm_epilogue v3 (H100 accumulator staging): acc/n + bias → fp32 tile in the idle ring, residual / out as 16 B rows (8 KB contiguous per i) | 1189 → 1135 µs |

**Measured and rejected:**
- GEMM1 operand layouts: the four transposes are within 1–3% of each other (`gemm_probe.py`, 207–230 TF). The limit is the power cap.
- `opm_epilogue` with 2 d per stage at NS 2 / 3 / 4: 1182 / 1495 / 1495 µs (the 1-CTA/SM variants lose).
- `pwa_main` NS=4: no change (2841 µs).

## OPM: is a different fusion algorithm better?

- **Fully fusing GEMM1 + GEMM2 in one kernel** would drop O's DRAM round trip (1.2 GB each way at L768). GEMM1's 1.24e12 FLOP
  cannot be reduced: applying W to the right side first costs 4× the FLOP. cuBLAS already runs GEMM1 at 217 TF under the power
  cap, so a fused kernel wins only if its hand-written main loop matches cuBLAS. At 90% of cuBLAS it is slower (≈ 7.25 ms); the
  best case is ~5%. Not pursued.
- **i-chunk stream pipeline** (GEMM1 chunk c+1 ∥ epilogue of chunk c, O ring small enough to stay in L2; `opm_forward_pipelined`):
  slower at every chunk size. At L768: ci = 8 / 16 / 32 / 64 / 96 / 192 → 9576 / 8239 / 7615 / 7401 / 7304 / 7246 µs, vs 7067.
  Small GEMM chunks lose efficiency, and the epilogue's CTAs take SMs from the GEMM.
- Conclusion: the H100 structure (cuBLAS grouped GEMM1 + one fused epilogue) stays. OPM is at 82% of SoL.

## Where the time is now / next

- **OPM is at 81% of SoL**, and GEMM1 alone is 80% of the op, running at the tensor ceiling of a power-capped card. A GEMM1 + GEMM2
  fused kernel would drop the 1.2 GB O write and read. It pays off only if a hand-written 256×128 mma.sync main loop matches
  cuBLAS: the estimate is ~0.5 ms (7%) at L768, at substantial effort.
- **PWA (split) is at 55% of SoL.** `pwa_ctr` is 65% of the time with tensor 65% active. A 256-i tile would halve its L2
  traffic (w and v re-reads, ~5 GB at L768), but it would drop to 1 CTA/SM, so the epilogue could no longer overlap with another
  CTA's main loop. `pwa_value` is at 63% of its floor, limited by its read pattern.
- **PWA (fused, previous) main is at 54% of its floor.** Under the H100 fusion boundaries, the out-projection accumulator stays in registers for all
  heads (64 fp32 per 32-token warp, 238 regs). That caps the CTA at 256 tokens and 8 warps per SM (2 per SMSP) and the warp tile at
  32 × 32. With so few warps, fixed-latency waits leave the tensor pipe idle (45% active). The v3 changes (128-j stages, early Wo
  loads, fragment double-buffering) moved it by only 2%. Getting past this means changing the boundary: for example, writing
  u = sigmoid(g) ⊙ o to memory and running the out-projection as its own pass. That frees registers for 64 × 64 warp tiles, costs
  about 0.8 GB of extra traffic at L768, and is an estimated ~0.6 ms saving.
- pwa_value (68%): the 402 MB head-major v write dominates. Wider per-warp stores, or 2 heads per staging round, could help.

## Training (forward + backward) — in progress

Harness: `bench_train.py --op opm pwa`. Every input gradient and parameter gradient is compared with fp32 autograd of the same
module on the same weights; the bf16 PyTorch module's own error is printed in parentheses for scale. PWA uses dropout p = 0.15:
the reference applies the same row-broadcast keep-mask by hand to a p = 0 module. Timing is forward + backward with a fixed random
upstream gradient, in one CUDA graph. Baselines are from [a100_anthropic_baseline](../a100_anthropic_baseline/README.md):
PyTorch compile, and the current engine for comparison.

| op | L | A100 (µs) | PyTorch compile | engine main | vs best |
|---|---:|---:|---:|---:|---:|
| OPM | 384 | 5400 | 6616 | 6947 | **1.23×** |
| OPM | 768 | 20929 | 25030 | 25999 | **1.20×** |
| PWA | 384 | 3352 | 5628 | 6239 | **1.68×** |
| PWA | 768 | 9053 | 14290 | 15223 | **1.58×** |

All gradients are at or below the bf16 module's error. `ln_pair_b`'s exact gradient is 0 (a softmax's dlogit row sums vanish), so
its error is absolute.

**OPM backward** (records/opm_train_v2.json; this follows the split of the H100 `opm_train.py`)
- `opm_dgrad`: dzn = dz / n, dO = dzn·Wo written in the grouped layout (the forward epilogue's permute, run backwards); dbo via atomics.
- cuBLAS: dA = b·dOᵀ, dB = a·dO.
- `opm_dwo`: dWo = Σ dzn ⊗ O, read from the saved grouped O; split-K with fp32 atomics.
- `opm_pbwd`: mask, dy = [dA|dB]·W, dW, and the LayerNorm backward recomputed from msa.
- Time at L768: the three 1.24e12-FLOP GEMMs (forward GEMM1, dA, dB) are 16.5 ms of 20.9 ms at ~225 TF, so OPM training is bound by
  them. Custom kernels: dgrad 1367 µs (v1: 1728, 1 CTA/SM), dwo 992 µs (64-pair stages; v1: 1100), epilogue 1084, pbwd 374.

**PWA backward** (records/pwa_train_v2.json; split-path forward, dense keys)
- `pwa_bglue`: warp h owns head h. dout' = dres·keep/(1−p); the gate is recomputed; du = dout'·Wo_h; do = du·g (head-major);
  dgp = du·o·g(1−g). dWo accumulates in registers.
- `pwa_ctr` with transposed A: dv = wᵀ·do.
- cuBLAS bmm: dw = do·vᵀ.
- `pwa_bproj`: dy = dgp·Wg + dv·Wv, dWg / dWv, and the LN backward plus residual.
- `pwa_bpair`: softmax backward, dWb, and the LN_z backward.
- Kernel times at L768: dw bmm 1865, ctr fwd 1644, ctr dv 1612, bglue 1273, bproj 1168, out2 452, bpair 441, value 394, pair 146 µs.
  The FLOP floor of fwd + bwd is ~4.6 ms.
- Steps: the out2 keep-mask loaded once per tile instead of per element (training out2 651 → 423 µs); bpair with 4 keys per
  iteration and their z rows requested first (564 → 441 µs).

### PWA push toward SoL 70% (L384 included)

Current (same-run comparisons; run-to-run variation across physical GPUs is ~3–5%):

| | L384 | L768 | SoL (dense FLOP) | % of SoL |
|---|---:|---:|---|---|
| inference (split, compacted keys) | 911–918 µs | 2465–2539 µs | 484 / 1616 µs | 53% / 64–66% |
| training fwd+bwd | 3177–3243 µs | 8587–8765 µs | 1.50 / 4.94 ms | 46–47% / 56–58% |

Ceiling of this decomposition (sum of per-kernel floors, max(FLOP / 240 TF, bytes / 1.6 TB/s)): inference 77% / 88%, training
70% / 78%. At L384 the intermediates (v, o, do, dgp, dv: 512 B / token each) are ~1.9 ms of HBM traffic against a 1.5 ms FLOP
floor, and the contraction (K = L) is L2-bandwidth bound: 1.3 GB of L2 traffic at L384.

Tried in this round:
- **Key compaction in training.** pwa_value, the forward contraction, dv (M = compacted keys, scattered by idx), dw, and bpair all
  run over the n valid keys; `pwa_compact` also emits posinv. Correct for no mask, all masked, 37%, and 10%. Gain at 10% masked:
  L768 9.05 → 8.59 ms.
- **Custom dw kernel** (`pwa_dw`: an NT GEMM with fp32 split-K partials, no atomics, compaction-aware) instead of cuBLAS bmm: L768 1865 → 1445 µs.
- **Bug fixed in pwa_bpair**: a masked key's logit is a constant in the module, so it must pass no gradient to z. This matters when
  every key is masked (compaction then lists all keys, and w is uniform). Checked: the pair / ln_pair / w_bias gradients are now
  exactly 0.
- **Persistent pwa_ctr** (the ring runs through tile boundaries; the epilogue uses the freed slot): ±0. The stalls outside the loop
  were not the prologue bubble.
- **dgp consumed in the glue kernel** (`pwa_bglue2` computes dWg and dy_g = dgp·Wg; `pwa_bproj2` handles the value side only,
  double-buffered): −800 MB of traffic at L768, but the total is unchanged (bglue2 754 vs 625 µs at L384; bproj2 468 vs 593). bglue2
  is at 255 regs with spills and 29% tensor activity. Shared fp32 atomics for the cross-head dy_g sum run as CAS loops on sm_80
  (3× slower) and were replaced by a staged dgp tile. Kept behind `PWA_BWD_V=2`; the default is v1.

### OPM CUDA: last push (after the Triton port matched it)

The CUDA and the Triton OPM are within 1% at inference because both spend 81% of the op in the same cuBLAS GEMM1 (5.7 of 7.1 ms at
L768, 217 TF, 90% of the power-capped ceiling); they differ only in the ~1.3 ms around it. Tried:
- GEMM1 transposed (O^T = bᵀa, epilogue with the pair swapped and W permuted to K order (e, d); inference only, since opm_dwo reads
  O = aᵀb): same run 7168 → 7125 µs (−0.6%). Kept as the inference default (`OPM_T=0` switches it off).
- A 256-pair epilogue tile (8 i × 32 j, 1 CTA/SM, half of W's L2 re-reads, two-pass fp32 staging): 1150 → 1378 µs at L768,
  274 → 342 at L384. Rejected (`OPM_E_BIG`).
OPM inference stays at ~81% of SoL. The remaining lever is data-dependent: compacting fully padded MSA rows (s) or tokens (i/j),
which the benchmark fixture does not have.
