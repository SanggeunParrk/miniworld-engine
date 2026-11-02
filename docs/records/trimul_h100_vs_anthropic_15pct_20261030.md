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

## 6. Step 3: no-mask inference on the port takes the direct K1

Without a mask the token-flat port packed the four projection weights and built and read a bf16 [L, L] pair mask (`_front_rs`).
Now an inference call without a mask runs the direct K1 on an all-true token mask (same products). No mask, ours / Anthropic
(out / in): D384 L128 1.062 / 1.071 -> 0.842 / 0.837, L256 0.872 / 0.867 -> 0.818 / 0.818, L384 0.857 / 0.850 -> 0.827 / 0.828,
L2048 -> 0.841 / 0.831; D256 L384 1.002 / 1.012 -> 0.969 / 0.969, L128 0.929 -> 0.921. Outputs equal to Anthropic's.

## 7. Final D x L table (branch head; ours / Anthropic; target <= 0.870)

Graph replays, alternating, median of 9; mask = token mask with the last L/50 tokens padded. Cells with a mask: final grid
(2026-10-31); no-mask columns of D256 / D384 at L % 128 == 0 re-measured after step 3. "pass" = every column <= 0.870.
(Anthropic L2560 D384 outgoing: OOM next to our graphs in one process; incoming 0.842.)

| D | L | ours ms (out, mask) | Anthropic ms | out mask | in mask | out no mask | in no mask | worst | pass |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.0139 | 0.0211 | 0.6588 | 0.6571 | 0.8144 | 0.8144 | 0.8144 | yes |
| 64 | 200 | 0.026 | 0.0364 | 0.7143 | 0.7151 | 0.8076 | 0.8025 | 0.8076 | yes |
| 64 | 256 | 0.0333 | 0.0448 | 0.7433 | 0.7415 | 0.8414 | 0.8478 | 0.8478 | yes |
| 64 | 384 | 0.0725 | 0.0888 | 0.8164 | 0.848 | 0.8748 | 0.894 | 0.894 | no |
| 64 | 500 | 0.1294 | 0.1547 | 0.8365 | 0.8467 | 0.8653 | 0.8803 | 0.8803 | no |
| 64 | 512 | 0.1301 | 0.1547 | 0.841 | 0.8447 | 0.8729 | 0.8788 | 0.8788 | no |
| 64 | 640 | 0.2056 | 0.235 | 0.8749 | 0.8702 | 0.8991 | 0.8923 | 0.8991 | no |
| 64 | 768 | 0.2908 | 0.3264 | 0.8909 | 0.909 | 0.9096 | 0.9319 | 0.9319 | no |
| 64 | 1000 | 0.5417 | 0.6077 | 0.8914 | 0.8925 | 0.9056 | 0.914 | 0.914 | no |
| 64 | 1024 | 0.5448 | 0.6168 | 0.8833 | 0.8806 | 0.9074 | 0.9045 | 0.9074 | no |
| 64 | 1280 | 0.9329 | 1.0399 | 0.8971 | 0.887 | 0.9125 | 0.9047 | 0.9125 | no |
| 64 | 1500 | 1.4104 | 1.5453 | 0.9127 | 0.9095 | 0.9246 | 0.9245 | 0.9246 | no |
| 64 | 1536 | 1.4101 | 1.5648 | 0.9011 | 0.9071 | 0.9184 | 0.919 | 0.919 | no |
| 64 | 2048 | 2.8199 | 3.1292 | 0.9012 | 0.902 | 0.9152 | 0.9148 | 0.9152 | no |
| 64 | 2560 | 4.9996 | 5.5302 | 0.9041 | 0.9097 | 0.922 | 0.9155 | 0.922 | no |
| 128 | 128 | 0.0239 | 0.0311 | 0.7685 | 0.766 | 0.8783 | 0.8859 | 0.8859 | no |
| 128 | 200 | 0.0512 | 0.0632 | 0.8101 | 0.8349 | 0.8685 | 0.8841 | 0.8841 | no |
| 128 | 256 | 0.0663 | 0.0824 | 0.8046 | 0.8054 | 0.8338 | 0.8423 | 0.8423 | yes |
| 128 | 384 | 0.1388 | 0.1644 | 0.8443 | 0.8565 | 0.8676 | 0.8718 | 0.8718 | no |
| 128 | 500 | 0.2416 | 0.2812 | 0.8592 | 0.8525 | 0.8684 | 0.864 | 0.8684 | yes |
| 128 | 512 | 0.2424 | 0.281 | 0.8626 | 0.854 | 0.8677 | 0.8655 | 0.8677 | yes |
| 128 | 640 | 0.3864 | 0.448 | 0.8625 | 0.8628 | 0.857 | 0.8538 | 0.8628 | yes |
| 128 | 768 | 0.562 | 0.6487 | 0.8663 | 0.8888 | 0.8848 | 0.9038 | 0.9038 | no |
| 128 | 1000 | 1.0634 | 1.2067 | 0.8812 | 0.8863 | 0.9006 | 0.893 | 0.9006 | no |
| 128 | 1024 | 1.0769 | 1.2256 | 0.8787 | 0.88 | 0.8824 | 0.8935 | 0.8935 | no |
| 128 | 1280 | 1.823 | 2.0471 | 0.8905 | 0.8847 | 0.8973 | 0.893 | 0.8973 | no |
| 128 | 1500 | 2.7684 | 3.0997 | 0.8931 | 0.8949 | 0.9 | 0.8979 | 0.9 | no |
| 128 | 1536 | 2.7859 | 3.1595 | 0.8818 | 0.8984 | 0.8963 | 0.903 | 0.903 | no |
| 128 | 2048 | 5.6873 | 6.3162 | 0.9004 | 0.8901 | 0.9082 | 0.9117 | 0.9117 | no |
| 128 | 2560 | 10.0239 | 11.0295 | 0.9088 | 0.9152 | 0.9065 | 0.9239 | 0.9239 | no |
| 256 | 128 | 0.0406 | 0.0492 | 0.8252 | 0.8222 | 0.9213 | 0.9191 | 0.9213 | no |
| 256 | 200 | 0.1364 | 0.1495 | 0.9124 | 0.9333 | 0.9257 | 0.9463 | 0.9463 | no |
| 256 | 256 | 0.1605 | 0.1707 | 0.9402 | 0.9395 | 0.9438 | 0.9503 | 0.9503 | no |
| 256 | 384 | 0.3421 | 0.3564 | 0.9599 | 0.9578 | 0.9689 | 0.9692 | 0.9692 | no |
| 256 | 500 | 0.6026 | 0.6174 | 0.976 | 0.9546 | 0.9561 | 0.9479 | 0.976 | no |
| 256 | 512 | 0.6084 | 0.6242 | 0.9747 | 0.9717 | 0.967 | 0.957 | 0.9747 | no |
| 256 | 640 | 0.9776 | 0.9924 | 0.9851 | 0.9771 | 0.9736 | 0.9611 | 0.9851 | no |
| 256 | 768 | 1.3978 | 1.4245 | 0.9813 | 0.9935 | 0.9735 | 0.9803 | 0.9935 | no |
| 256 | 1000 | 2.6174 | 2.6792 | 0.9769 | 0.9774 | 0.9509 | 0.9514 | 0.9774 | no |
| 256 | 1024 | 2.6341 | 2.718 | 0.9691 | 0.9647 | 0.9655 | 0.9654 | 0.9691 | no |
| 256 | 1280 | 4.3835 | 4.5476 | 0.9639 | 0.9516 | 0.9736 | 0.9602 | 0.9736 | no |
| 256 | 1500 | 6.659 | 6.7905 | 0.9806 | 0.9687 | 0.9501 | 0.9473 | 0.9806 | no |
| 256 | 1536 | 6.6025 | 6.8139 | 0.969 | 0.9629 | 0.9714 | 0.9721 | 0.9721 | no |
| 256 | 2048 | 13.447 | 13.8705 | 0.9695 | 0.9612 | 0.96 | 0.9518 | 0.9695 | no |
| 256 | 2560 | 22.833 | 23.4719 | 0.9728 | 0.9397 | - | - | 0.9728 | no |
| 384 | 128 | 0.0754 | 0.096 | 0.7854 | 0.7888 | 0.8417 | 0.8371 | 0.8417 | yes |
| 384 | 200 | 0.2483 | 0.3079 | 0.8064 | 0.8185 | 0.8225 | 0.8369 | 0.8369 | yes |
| 384 | 256 | 0.284 | 0.3549 | 0.8002 | 0.7947 | 0.8183 | 0.8176 | 0.8183 | yes |
| 384 | 384 | 0.6184 | 0.754 | 0.8202 | 0.8139 | 0.8273 | 0.8276 | 0.8276 | yes |
| 384 | 500 | 1.0833 | 1.3142 | 0.8243 | 0.8211 | 0.8348 | 0.8235 | 0.8348 | yes |
| 384 | 512 | 1.0869 | 1.3256 | 0.8199 | 0.8125 | 0.8225 | 0.8218 | 0.8225 | yes |
| 384 | 640 | 1.733 | 2.111 | 0.8209 | 0.8178 | 0.8288 | 0.827 | 0.8288 | yes |
| 384 | 768 | 2.4865 | 3.101 | 0.8018 | 0.8129 | 0.8172 | 0.8208 | 0.8208 | yes |
| 384 | 1000 | 4.6012 | 5.6209 | 0.8186 | 0.8192 | 0.8153 | 0.8184 | 0.8192 | yes |
| 384 | 1024 | 4.6836 | 5.708 | 0.8205 | 0.8207 | 0.8191 | 0.8054 | 0.8207 | yes |
| 384 | 1280 | 7.7624 | 9.3525 | 0.83 | 0.82 | 0.8305 | 0.8233 | 0.8305 | yes |
| 384 | 1500 | 11.3535 | 13.7112 | 0.828 | 0.8203 | 0.8227 | 0.8172 | 0.828 | yes |
| 384 | 1536 | 11.6382 | 14.1318 | 0.8235 | 0.8252 | 0.8243 | 0.8261 | 0.8261 | yes |
| 384 | 2048 | 22.8957 | 26.8584 | 0.8525 | 0.8551 | 0.8414 | 0.8311 | 0.8551 | yes |
| 384 | 2560 | 39.8131 | - | - | 0.8423 | - | - | 0.8423 | yes |

22 of 60 (D, L) cells pass in all four columns; 29 with a mask in both directions (the form models call). What bounds the rest:

- **D256, every L >= 200 (0.94-0.99):** the same three stages as Anthropic at near-equal speed: K1 at 540-560 TFLOP/s (~80 % of the
  GEMM rate this card sustains), K3 at ~440 TFLOP/s with its HBM streams removed (its own MMA / epilogue pipeline; 2.0-2.2 TB/s
  with them), the cuBLAS contraction identical to Anthropic's. Need / floor 1.02-1.84 below L2048 (section 1): 15 % needs K3 at its
  memory floor and K1 at ~600 TFLOP/s together; L2048 / 2560 are below the floor (0.99 / 0.95).
- **D64 L >= 384 and D128 L >= 768 (0.87-0.92):** the cuBLAS contraction (Anthropic's kernel) is 35-55 % of the call; K1 / K3
  stream at 2.2-2.7 TB/s, 80-95 % of the 2.45-2.6 TB/s copy rate. D64 L >= 1536 and D128 L >= 2048 are below the floor.
- **No-mask small L (D64 L384-512, D128 L128-200 / 384, D256 L128):** with a mask Anthropic builds an fp32 [L, L] pair mask per
  call, without one it does not; there our margin is 0.87-0.92 at 25-150 us calls (three launches each side).
- D384 passes everywhere (port K3 vs the packaged wide K3: 0.089 vs 0.147 ms at L256).
