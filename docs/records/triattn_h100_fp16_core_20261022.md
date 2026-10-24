# TriangleAttention on H100: fp16-operand inference core (2026-10-22)

H100 SXM 80 GB, torch 2.13 + cu129 (ours), Anthropic opt_core `triattn_native` (M1) in its cu130 container (on cu129 it has no
prebuilt and falls back to `flash_triattn`, ~1.5x slower: never compare against that). B = 1, H = 4, D = 32, bf16 module, random
weights (`foldforge-scratch/triattn_work/common.py: make()`), no mask unless stated.

## What changed

- `h100_sources/triattn_h16_sm90.cu`: the inference core on fp16 q / k / v / sbias (out bf16). Same schedule as `triattn_sm90.cu`
  (2 consumer warpgroups x 64 queries x 2 pair rows); fp16 QK^T accumulator initialised by the bias fragment (no unpack), offset and
  exponentials on pairs (HFMA2 + MUFU.EX2 .f16x2), P in place as the PV A fragment (no pack). Offset (`TA_SPEC 2`, default): the row
  maximum's shuffles only when a lane of the warp sees a logit past its row's offset + 8 log2 units (per-thread HMNMX2 tree + warp vote):
  p <= 2^9, no fp16 overflow, no fix list.
- `tri_front(..., h16)`: q / k / v and the bias written in fp16 (g stays bf16; finite bias clamped to +-6e4); `h100_sm90.inference`
  uses both (`INFERENCE_FP16 = True`).
- `triattn_sm90.cu` (bf16 core, training and other callers): `TA_DOFF` (wgmma descriptors built once, one add per use).

## Accuracy (core, fp32 reference on fp32 inputs, L = 512 unless stated)

| case | bf16 core rel | fp16 core rel |
|---|---|---|
| random | 7.72e-3 | 2.53e-3 |
| logits x36 (q, k x6, bias x8) | 1.57e-2 | 4.94e-3 |
| +7 log2 units from key 256 | 1.08e-2 | 3.10e-3 |
| +20 log2 units from key 256 | 1.89e-2 | 4.40e-3 |
| +60 log2 units from key 384 (L 768) | 7.20e-2 | 1.61e-2 |
| first 128 keys masked | 7.22e-3 | 2.44e-3 |
| N = 383, L = 384 | 7.24e-3 | 2.44e-3 |

Bitwise deterministic in every case; LSE and the gated output checked. Protenix1 trunk (5I28 x 4 / 8 / 16, every core call re-run):
max rel difference from the bf16 core 1.96e-3, no non-finite value.

## Speed

Core, sustained (8 s of back-to-back CUDA-graph replays; the GPU sits at its 700 W cap, clocks in MHz):

| L | fp16 core | Anthropic M1 | bf16 core | vs Anthropic |
|---|---|---|---|---|
| 1024 | 1.932 ms (1724) | 2.110 (1679) | 2.120 (1713) | 1.09x |
| 2048 | 14.83 ms (1677) | 15.44 (1621) | 16.08 (1671) | 1.04x |
| 3072 | 49.63 ms (1648) | 53.05 (1532) | 53.42 (1650) | 1.07x |

Module, CUDA graph (Anthropic: `anth_stages.py` "module"):

| L | ours fp16 | ours bf16 | Anthropic | vs Anthropic (fp16) | (bf16) |
|---|---|---|---|---|---|
| 256 | 0.109 | 0.121 | 0.165 | 1.51x | 1.36x |
| 512 | 0.497 | 0.535 | 0.718 | 1.44x | 1.34x |
| 768 | 1.297 | 1.404 | 1.754 | 1.35x | 1.25x |
| 1024 | 2.762 | 2.922 | 3.403 | 1.23x | 1.16x |
| 1536 | 8.309 | 9.567 | 9.766 | 1.18x | 1.02x |
| 2048 | 18.173 | 19.516 | 20.564 | 1.13x | 1.05x |
| 2560 | 34.136 | 37.686 | 39.018 | 1.14x | 1.04x |
| 3072 | 57.918 | 63.293 | 65.996 | 1.14x | 1.04x |

Protenix1 trunk inputs (per core call, synchronised, eager): fp16 core 1.20x (512 tokens), 1.25x (1024), 1.27x (2048) the bf16 core.

## Measured and rejected

| attempt | result |
|---|---|
| fp32 bias staged once per call in fragment order (Anthropic's) | fewer instructions, but more traffic: clock -4 %, core 3-6 % slower at L >= 1536 |
| `TA_SPEC 1` on fp16 (no maximum after the first tile, fix list) | ~8 % faster on random inputs; recomputes ~5 % of the tasks on Protenix inputs, level there |
| offset margin 12 for `TA_SPEC 1` | subnormal P: error above the bf16 core |
| QK^T of the next sub-step issued before the exponentials (3 buffers; RS and SS Q) | 0.83-0.87x |
| 3 consumer warpgroups (64 queries x 3 rows) | fewer cycles, lower clock (K / V reuse halved): 0.91-1.08x |
| cluster of 2 CTAs, K / V TMA multicast | 0.55-0.6x |
| 3 rows per CTA (bf16 core) | level |
| chunk stream after Anthropic's M1 (3 warpgroups x 128 queries, 32-key chunks) | 512 threads cap ptxas at 128 registers (setmaxnreg does not raise it); [V \| 1] and branchy accumulator definitions serialise every wgmma (C7511 / C7515); best 1.00-1.12x the bf16 core |
| warpgroup ping-pong on the exponential phase | level sustained |
| row sums by HADD2 instead of the [V \| 1] columns | 4-13 % slower |
| vote after the exponentials (rare miss recomputes the scores with mma.sync) | level with the vote before |

Where the time goes (ncu, bf16 core, L 1024 at 1.42 GHz): issue slots 47 % busy, MUFU (XU) 52 %, tensor 30 %; the stall is the
per-warp chain bias fragment -> QK^T -> maximum -> exponentials -> PV with two warps per scheduler. Removing every K / V and bias
reload (stale tiles) lifts the clock to 1980 MHz at 610 W but saves only ~4 %: data movement is not the limit.
