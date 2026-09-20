# Nsight Compute summary (node02 H100 80 GB, CUDA 12.9, PyTorch 2.10 cu128), 2026-09-20

Raw `.ncu-rep` files are not committed (see the repository `.gitignore`); the numbers below were read with `ncu -i ... --page raw`.

## cuBLAS contraction (`nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN`, `torch.bmm(a, b^T)`, batch 128, M = N = K = L)

| L | duration | SM clock | DRAM read / write | DRAM % peak | tensor pipe active | L2 hit | SM active / elapsed (min / avg / max) |
|---|---:|---:|---:|---:|---:|---:|---|
| 384 | 39.1 us | 1.80 GHz | 75.5 / 21.6 MB | 74.1 % | 42.9 % | 40 % | 0.703 / 0.898 / 0.947 |
| 768 | 182.8 us | 1.58 GHz | 304.4 / 138.6 MB | 72.3 % | 78.1 % | 56 % | 0.922 / 0.955 / 0.988 |

Essential bytes: a and b planes 2 x 128 x L^2 x 2 B read once (302 MB at L768), X = 128 x L^2 x 2 B written once (151 MB; part of the write
stays in L2 at kernel end).  The kernel moves exactly the essential bytes.  At L768 it sits on the DRAM and the tensor ceilings at the same
time, with the SM clock power-throttled to 1.58 GHz; at L384 it is at the memory floor.  `probes/cublaslt-algos-L*.json` times all eight
heuristic algorithms for the same shape: the heuristic's first pick is the fastest (L768 179.1 us vs 182.1 for the next; L384 within 0.1 us).

## K1 / K3 (payload final2, warm, clock-control none)

| kernel | L | duration | SM active / elapsed (min / avg / max) |
|---|---:|---:|---|
| `tmn_k1_z128_h128_b_t6x32_s8k2_m1_l2_v0` | 384 | 52.7 us | 0.780 / 0.887 / 0.956 |
| `tmn_k3_z128_h128_b_t3x64_s8a1_l1` | 384 | 52.4 us | 0.705 / 0.859 / 0.964 |
| `tmn_k1_z128_h128_b_t6x32_s8k2_m1_l2_v0` | 768 | 189.7 us | 0.888 / 0.925 / 0.983 |
| `tmn_k3_z128_h128_b_t3x64_s8a1_l1` | 768 | 177.2 us | 0.909 / 0.945 / 0.987 |

The min/max spread is the ragged tail (CTAs with 5 vs 6 tiles at L384, 23 vs 24 at L768); the gap between the busiest SM and 1.0 is the
CTA startup and drain.  Earlier warm NCU of the same payload (round 3): K1 L768 DRAM 72.3 % of peak, tensor 48.8 %, XU 25.4 %; K3 L768 DRAM
74.5 %, tensor 24.6 %, eligible warps per scheduler 1.00.

## Per-CTA clock probe (`probes/cta-timeline-L*.json`, TMN_CTA_TS build, cycles at ~1.7-1.8 GHz)

| kernel | L | startup (launch -> first tile ready) | steady per 192-token tile | end spread across CTAs |
|---|---:|---:|---:|---:|
| K1 | 384 | 7473 cyc (~4.4 us) | 12581 cyc (~7.4 us) | 8.8 us |
| K3 | 384 | 5670 cyc (~3.3 us) | 12502 cyc | 11.1 us |
| K1 | 768 | 7214 cyc (~4.2 us) | 12700 cyc | 18.1 us |
| K3 | 768 | 8160 cyc (~4.8 us) | 12156 cyc | 16.3 us |

A 192-token tile moves 144 KB (K1: 48 KB z in, 96 KB planes out; K3: 96 KB in, 48 KB out): 12.5-12.7K cycles per tile is 2.65-2.7 TB/s
across 132 SMs, 93-95 % of the 2.85 TB/s streaming pattern floor.  The startup matches 132 CTAs pulling their resident weights (128 KB for K1,
64 KB for K3) plus the first tile through L2 at once.
