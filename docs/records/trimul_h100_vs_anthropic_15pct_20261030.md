# H100 TriMul vs Anthropic `trimul_native`: the 15 % target, 2026-10-30 / 31

Goal (user): H100 single-direction TriMul inference at least 15 % faster than Anthropic's `trimul_native` v5 (ours / Anthropic
<= 0.870) at every shape Anthropic serves: D 64 / 128 / 256 / 384 (hidden = D), L 128-2560 including L % 128 != 0, with and
without a token mask, B = 1, both directions. Numerics unchanged (bf16, same rounding points).

Setup as `trimul_h100_bigl_20261030.md`: H100 80GB HBM3 (power-capped), torch 2.13 + cu129, Anthropic = unmodified v5 1.2.2
rebuilt with nvcc 12.9, one process, same weights / input, CUDA-graph replays alternating, median of 9. Mask = token mask with the
last L/50 tokens padded. Probes: `~/miniworld-engine-scratch/perf1030/probe/` (`grid_probe.py`, `rates_probe.py`,
`rowpad_probe.py`, `ncu_run.sh`).

## 1. What this H100 sustains, and the floor of each cell

| measured (graph, ~1 s windows) | rate |
|---|---|
| bf16 GEMM 8192^3 (cuBLAS) | 667-679 TFLOP/s (same for random and sigmoid-gated data) |
| contraction bmm D256 L1536 / D64 L2048 / D128 L1024 / D384 L768 | 645 / 655 / 578 / 564 TFLOP/s |
| HBM copy (read + write) | 2.45 TB/s |

FLOPs per call: K1 8 L^2 D^2, K3 4 L^2 D^2, contraction 2 L^3 D. Bytes per call (bf16): K1 reads x and writes the planes (6 L^2 D),
K3 reads tri and x and writes y (6 L^2 D). Optimistic floor per cell = K1 max(8 L^2 D^2 / 670 T, 6 L^2 D / 2.6 TB/s) +
2 L^3 D / 700 T + K3 max(4 L^2 D^2 / 670 T, 6 L^2 D / 2.6 TB/s) + 3 launches (9 us). "Need" = Anthropic / 1.15.

| D | L | ours (before) | Anthropic | ours / Anth | need | floor | need / floor |
|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.0134 | 0.0211 | 0.635 | 0.018 | 0.014 | 1.29 |
| 64 | 512 | 0.1297 | 0.1548 | 0.838 | 0.135 | 0.111 | 1.21 |
| 64 | 640 | 0.2050 | 0.2351 | 0.872 | 0.204 | 0.178 | 1.15 |
| 64 | 768 | 0.2894 | 0.3280 | 0.882 | 0.285 | 0.266 | 1.07 |
| 64 | 1024 | 0.5597 | 0.6131 | 0.913 | 0.533 | 0.515 | 1.04 |
| 64 | 1536 | 1.4374 | 1.5556 | 0.924 | 1.353 | 1.369 | **0.99** |
| 64 | 2048 | 2.8764 | 3.1228 | 0.921 | 2.715 | 2.819 | **0.96** |
| 64 | 2560 | 5.0585 | 5.4502 | 0.928 | 4.739 | 5.013 | **0.95** |
| 128 | 128 | 0.0213 | 0.0318 | 0.670 | 0.028 | 0.019 | 1.42 |
| 128 | 512 | 0.2423 | 0.2815 | 0.861 | 0.245 | 0.213 | 1.15 |
| 128 | 768 | 0.5519 | 0.6424 | 0.859 | 0.559 | 0.523 | 1.07 |
| 128 | 1024 | 1.0841 | 1.2228 | 0.887 | 1.063 | 1.021 | 1.04 |
| 128 | 1536 | 2.7941 | 3.1490 | 0.887 | 2.738 | 2.728 | 1.00 |
| 128 | 2048 | 5.8218 | 6.1953 | 0.940 | 5.387 | 5.628 | **0.96** |
| 128 | 2560 | 10.480 | 10.961 | 0.956 | 9.531 | 10.016 | **0.95** |
| 256 | 128 | 0.0405 | 0.0494 | 0.820 | 0.043 | 0.033 | 1.30 |
| 256 | 256 | 0.1595 | 0.1698 | 0.939 | 0.148 | 0.111 | 1.33 |
| 256 | 512 | 0.6085 | 0.6226 | 0.977 | 0.541 | 0.467 | 1.16 |
| 256 | 1024 | 2.6314 | 2.7415 | 0.960 | 2.384 | 2.234 | 1.07 |
| 256 | 1536 | 6.5463 | 6.9022 | 0.948 | 6.002 | 5.900 | 1.02 |
| 256 | 2048 | 13.350 | 13.724 | 0.973 | 11.934 | 12.052 | **0.99** |
| 256 | 2560 | 22.733 | 23.154 | 0.982 | 20.134 | 21.280 | **0.95** |
| 384 | 1536 | 11.611 | 14.042 | 0.827 | 12.210 | 10.230 | 1.19 |

(outgoing, mask; full grids in `~/miniworld-engine-scratch/perf1030/logs/grid_*.json`.) Bold need / floor < 1: 15 % is out of
reach with bf16 math as it is (the contraction alone is 50-55 % of these calls and Anthropic runs the same cuBLAS kernel).
Need / floor 1.0-1.16: every stage at 86-100 % of its own roofline at once; not seen in any kernel here (best: contraction
~95 % of the GEMM rate, K1 ~80 %, K3 ~80 % of the copy bandwidth).

Outputs: ours, the port and Anthropic are bitwise equal at default LayerNorm affines; with non-trivial LN_out affines the
existing paths already differ from each other by 1 bf16 ulp (where LN_out's affine is folded), the any-L port form below is
bitwise equal to the token-flat port.

## 2. Step 1: the port at any L for one-sample D384 (k3.cu ROWPAD)

The packaged K1 / K3 served D384 at L % 128 != 0 at Anthropic's speed (same kernels). The B200-style port took only multiples
of 128 (token-flat 128-token K3 tiles). Now: K1 writes planes padded to 16 (the packaged K1's convention), the contraction runs on
them, K3 walks 64-token chunks of one pair row (`ROWPAD`, 3-D TMA maps that zero-fill / clip the ragged row end).

| D384 | L200 | L500 | L1000 | L1500 |
|---|---|---|---|---|
| packaged K1/K3 / Anthropic (out, mask) | 0.956 | 0.991 | 0.994 | 1.003 |
| port any-L / Anthropic (out, mask) | 0.807 | 0.828 | 0.816 | 0.831 |
| (in, mask) | 0.821 | 0.821 | 0.817 | 0.828 |
| (out / in, no mask) | 0.826 / 0.839 | 0.833 / 0.830 | 0.821 / 0.817 | 0.824 / 0.827 |

D384 default route after the change (out / in, mask): L40 0.908 / 0.915, L128 0.789 / 0.792, L200 0.805 / 0.817, L256 0.798 /
0.797, L384 0.820 / 0.814, L500 0.830 / 0.823, L512 0.822 / 0.818, L640 0.825 / 0.818, L768 0.807 / 0.811, L1000 0.817 / 0.814,
L1024 0.817 / 0.814, L1280 0.829 / 0.815, L1500 0.831 / 0.828, L1536 0.827 / 0.827, L2048 0.849 / 0.840, L2560 0.85 (before).

D256 at odd L: the any-L port and the packaged K1/K3 within 1 % (L200 0.93-1.00, L500 0.97, L1000 0.96, L1500 0.95), so D256
keeps the packaged path there.

## 3. Step 2: D128 / D64 on the port's any-L form

The port's K3 (folded LayerNorms, LN_in statistics from K1) is faster than the packaged K3 at D64 / D128; its K1 is about as fast
as the packaged K1 (it also writes the LN_in statistics). New: BN 64 / 128 K3 builds (`mw_mma64`, wgmma m64n64k16), ROWPAD for B
samples stacked b-major (L % 16 == 0, Params.M = pair rows) so batched and per-sample calls keep one arithmetic, `XBUF = 2` (the
next tile loads x into a second buffer while the previous y is still leaving by TMA store; D64 / D128 only, the D256 / D384 tiles
leave no shared memory for it). Route: `ROWPAD_FROM = {64: 1000, 128: 768}` (one direction, inference, any B).

Per-kernel (ms, D128 L1024 out, profiled graph replay): packaged K1 0.354 / K3 0.317 / bmm 0.400; port K1 0.341 / K3 0.310 /
bmm 0.402 / fold 0.002; Anthropic K1 0.380 / K3 0.386 / bmm 0.401. D64 L1536: packaged K1 0.374 / K3 0.446 / bmm 0.677; port
K1 0.386 / K3 0.376 / bmm 0.628; Anthropic 0.436 / 0.476 / 0.631.

| ours / Anthropic (out, mask) | L768 | L1000 | L1024 | L1536 | L2048 |
|---|---|---|---|---|---|
| D128 packaged K1/K3 | 0.878 | 0.889 | 0.887 | 0.894 | 0.961 |
| D128 port any-L, XBUF 1 | 0.880 | 0.893 | 0.883 | 0.886 | 0.914 |
| D128 port any-L, XBUF 2 | 0.877 | 0.876 | 0.878 | 0.884 | 0.896 |
| D64 packaged K1/K3 | 0.889 | 0.905 | 0.900 | 0.914 | 0.919 |
| D64 port any-L (K1 6x32 / 3x64 beyond L1536, K3 4 stages) | 0.897 | 0.897 | 0.890 | 0.888 | 0.897 |

Batched D128 (rowpad vs packaged, same process): B2 L768 0.992, B4 L768 0.987, B2 L1024 0.987, B3 L1000 0.994; outputs equal.
D64 K1 tile sweep (port, out, L1024 / 1536 / 2048, rowpad/Anthropic): (6,32) 0.890 / 0.888 / 0.913; (3,64) 0.906 / 0.900 /
0.897; (2,64) 0.945 / 0.959 / 0.954. K3 stages 4 vs 6: within noise.

## 4. Where K3's time goes (port K3, graph, ROWPAD maps with a pair-row extent of 1 make TMA zero-fill the loads of a stream
without touching DRAM / drop its stores)

| | full | no x read | no tri read | no y write | no DRAM at all | no weight reads | x / tri / y share | DRAM rate |
|---|---|---|---|---|---|---|---|---|
| D128 L512 | 0.0819 | 0.0743 | 0.0716 | 0.0689 | 0.0643 | 0.0804 | 0.09 / 0.13 / 0.16 | 2.46 TB/s |
| D128 L1024 | 0.3038 | 0.2867 | 0.2609 | 0.2538 | 0.2601 | - | 0.06 / 0.14 / 0.16 | 2.65 TB/s |
| D128 L1536 | 0.6635 | 0.6319 | 0.5990 | 0.5707 | 0.5996 | 0.6600 | 0.05 / 0.10 / 0.14 | 2.73 TB/s |
| D128 L2048 | 1.2047 | 1.1422 | 1.0522 | 0.9815 | 1.0114 | - | 0.05 / 0.13 / 0.19 | 2.67 TB/s |
| D256 L512 | 0.2006 | 0.1887 | 0.1921 | 0.1656 | 0.1556 | 0.2012 | 0.06 / 0.04 / 0.17 | 2.01 TB/s |
| D256 L1024 | 0.7543 | 0.7186 | 0.7399 | 0.6375 | 0.7888 | - | 0.05 / 0.02 / 0.15 | 2.14 TB/s |
| D256 L1536 | 1.6948 | 1.6109 | 1.6635 | 1.4541 | 1.4063 | 1.6547 | 0.05 / 0.02 / 0.14 | 2.14 TB/s |
| D256 L2048 | 2.9727 | 2.7913 | 2.9621 | 2.5889 | 2.5570 | - | 0.06 / 0.00 / 0.13 | 2.17 TB/s |

The x read is 3-9 % of K3, the tri read 0-14 %; with every DRAM stream removed D256 K3 keeps 83-86 % of its time (its own
MMA / epilogue pipeline, ~440 TFLOP/s), D128 K3 ~85 %. Weight reads from L2 cost nothing measurable. ncu (base clock): port K3
D256 L1536 61.5 % memory throughput, tensor pipe 46 %; packaged K3 D64 L1024 57 % / 11 %.

**Fusing K3 into the contraction epilogue (cluster / DSMEM layout: 16 SMs share a 64x64-token tile, each contracts D/16
channels, then exchange so each SM finishes all channels of 256 tokens).** What it removes: the tri write + read (2 L^2 D bytes
each way) and the second x read; measured above that is at most 10-20 % of K3, i.e. 3-5 % of the call (D256 L1536: <= ~0.3 ms of
6.5 ms). What it costs: each channel's output tile is 64 x 64 tokens with K = L, so its operands carry 2 * 64 * L * 2 bytes for
2 * 64 * 64 * L FLOPs = 32 FLOP per byte from L2. At the 650-670 TFLOP/s the contraction sustains here that needs ~21 TB/s of
L2 -> SM operand feed, 3-4x what H100's L2 delivers (~5.5-7 TB/s); cuBLAS's 192 x 192 / 256 x 128 tiles need 3-4x less. The
contraction (35-55 % of the call) would run 2-3x slower. Larger shared tiles do not fit: 128 x 128 tokens x D/16 channels of
fp32 accumulators per SM, then a 2048-token x D exchange per SM (1 MB at D256) against 228 KB of shared memory. Not built.

## 5. Rejected (measured)

- cuBLAS algorithm search (PyTorch TunableOp over the contraction shapes D64-256, L512-2048, NT and TN): within +-5 % run noise of
  the default heuristic, no consistent winner; TunableOp kept "Default".
- Anthropic's incoming form (K1 reads z transposed, NT contraction): K1 +3-11 % at L1536-2560 (record `trimul_h100_bigl_20261030.md`).
- K3 at D256 / D384 with BN 128 (two / three column blocks) + XBUF 2: 1.5-4 % slower than BN 256 / 192 (the A operands stream once
  per column block).
- D128 / D64 port route below L768 / L1000: 1-5 % slower than the packaged K1/K3 (the fold launch and the K1 statistics weigh more).
- D256 at L % 128 != 0 on the port any-L form: within 1 % of the packaged path (L200 0.93-1.00, L500 0.97, L1000 0.96,
  L1500 0.95), kept on the packaged path.
