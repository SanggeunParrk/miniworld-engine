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
