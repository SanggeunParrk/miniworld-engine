# B200 TriMul: batched samples at every width (2026-10-20)

One B200 (sm_100a, 148 SMs, 192 GB, 1000 W cap), driver 580.95.05, torch 2.13.0+cu129, sm_100a kernels built by nvcc 13.1, GPU 4 of
node 2 with no other process on it during timing. Modules `BidirectionalTriangleMultiplication(D)` / `TriangleMultiplication(D)`
(hidden = D unless stated), fp32 parameters, bf16 pair. The scripts (outside the checkout) build their inputs with fixed generators:
x ~ N(0, 1), a token mask with ~15 % of the tokens dropped per sample, a row-dropout scale (keep 0.75) per sample, dy ~ N(0, 1).
Design and the summary tables: [gpus/b200/trimul/trimul.md](../gpus/b200/trimul/trimul.md#batched-samples-b--48-per-launch-any-b-per-call-every-width).

## Correctness sweep

B 2 / 4 / 8 / 16 / 32 / 48 x L128 / 384 / 768 x D64 / 128 / 256 / 384 / 512 x bidirectional / outgoing / incoming x inference /
training (540 cells), plus L1536 at B 2 / 4 (every width) and B 8 / 16 (D64): 612 cells, with the final launch plan (`_plan`). Per
cell: y (and dx) of one call against B one-sample calls (`torch.equal`, every sample), the parameter gradients against the fp32 sum of
the per-sample gradients, y / dx against the fp32 statement on the first and last sample (L <= 768), and a second run compared exactly
with the first (outputs parked in host memory). Result: 612 / 612 bitwise against the one-sample calls and run to run; per width:

| D | mode | cells | y bitwise | dx bitwise | run to run | weight grads vs per-sample sum (max rel) | vs fp32 (max rel) | chunked (checkpointed) | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| 64 | inference | 66 | 66 | - | 66 | - | y 3.0e-3 | 45 (0) | 11.5 |
| 64 | training | 66 | 66 | 66 | 66 | 5.8e-4 | y 3.1e-3, dx 4.7e-3 | 0 (0) | 82.5 |
| 128 | inference | 60 | 60 | - | 60 | - | y 3.0e-3 | 39 (0) | 15.8 |
| 128 | training | 60 | 60 | 60 | 60 | 7.0e-5 | y 3.1e-3, dx 4.8e-3 | 36 (0) | 81.4 |
| 256 | inference | 60 | 60 | - | 60 | - | y 3.0e-3 | 39 (0) | 31.1 |
| 256 | training | 60 | 60 | 60 | 60 | 7.3e-5 | y 3.1e-3, dx 4.6e-3 | 36 (1) | 133.9 |
| 384 | inference | 60 | 60 | - | 60 | - | y 2.9e-3 | 39 (0) | 46.5 |
| 384 | training | 60 | 60 | 60 | 60 | 1.7e-4 | y 3.1e-3, dx 4.6e-3 | 36 (4) | 135.1 |
| 512 | inference | 60 | 60 | - | 60 | - | y 3.0e-3 | 39 (0) | 62.1 |
| 512 | training | 60 | 60 | 60 | 60 | 2.9e-4 | y 3.2e-3, dx 4.7e-3 | 36 (6) | 155.0 |

The `k1wb` mask hoist came after this sweep; the 81 D256-512 training cells at B 2 / 16 / 48 were re-run on the final sources, all
bitwise (weight gradients <= 2.9e-4, fp32 y <= 3.1e-3, dx <= 4.6e-3). A first sweep before the launch cap (one launch wherever memory
allowed) was bitwise in all 612 cells too; its weight gradients reached 2.3e-3 (D128 L768 B48, the four front matrices: `b7g` keeps
dW1 in TMEM over ~20 k tiles per accumulator; ln_pair / ln_out gradients 1e-6, to_gate / to_out 3.4e-4).

cuBLAS: the contraction products (`bmm`, batch = channels x B up to 48 x 512 matrices, L128-1536) and the D256+ backward's token-flat
GEMMs (`mm`, M = B L^2) return per sample bitwise what the one-sample call computes (checked directly before the kernels were changed).

## Working-set coefficients (bytes per pair token per channel, max over L128 B16 / L384 B4 / L768 B2-4)

| mode | 4 D planes, D <= 128 | 4 D planes, D256+ | 2 D planes, D <= 128 | 2 D planes, D256+ |
|---|---|---|---|---|
| inference peak | 18.0 | 16.1 | 10.0 | 10.0 |
| training peak (x, y, dy, dx included) | 36.2 | 53.2 | 21.6 | 33.2 |
| training, held forward -> backward (y included) | 16.1 | 17.2 | 9.2 | 11.2 |

`_plan` uses these plus 5 % against 0.85 x total memory.
## Launch size (tokens per launch), CUDA graph, launch size forced

| D | module | L | B | mode | ms per samples-per-launch | best | best tokens / launch |
|---|---|---|---|---|---|---|---|
| 64 | bidirectional | 128 | 48 | inference | 48: 0.336, 24: 0.337, 16: 0.351, 12: 0.373, 8: 0.389 | 48 | 786432 |
| 64 | bidirectional | 256 | 48 | inference | 48: 1.559, 24: 1.587, 16: 1.423, 12: 1.320, 8: 1.286, 4: 1.377, 2: 1.527 | 8 | 524288 |
| 64 | bidirectional | 384 | 48 | inference | 48: 3.482, 24: 3.499, 16: 3.521, 12: 3.542, 8: 3.328, 6: 3.209, 4: 2.997, 3: 3.084, 2: 3.172, 1: 3.498 | 4 | 589824 |
| 64 | bidirectional | 768 | 8 | inference | 8: 2.454, 4: 2.461, 2: 2.385, 1: 2.161 | 1 | 589824 |
| 64 | bidirectional | 128 | 48 | training | 48: 1.182, 24: 1.272, 16: 1.380, 12: 1.486, 8: 1.707 | 48 | 786432 |
| 64 | bidirectional | 256 | 48 | training | 48: 4.663, 24: 4.799, 16: 4.737, 12: 4.747, 8: 4.851, 4: 5.522, 2: 6.847 | 48 | 3145728 |
| 64 | bidirectional | 384 | 48 | training | 48: 11.018, 24: 11.141, 16: 11.157, 12: 11.240, 8: 11.102, 6: 11.131, 4: 11.208, 3: 11.511, 2: 12.421, 1: 15.127 | 48 | 7077888 |
| 64 | bidirectional | 768 | 8 | training | 8: 7.892, 4: 8.018, 2: 7.937, 1: 7.998 | 8 | 4718592 |
| 128 | bidirectional | 128 | 48 | inference | 48: 0.684, 24: 0.664, 16: 0.688, 12: 0.711, 8: 0.733 | 24 | 393216 |
| 128 | bidirectional | 256 | 48 | inference | 48: 3.166, 24: 3.194, 16: 2.996, 12: 2.860, 8: 2.751, 4: 2.865, 2: 3.007 | 8 | 524288 |
| 128 | bidirectional | 384 | 48 | inference | 48: 7.482, 24: 7.459, 16: 7.437, 12: 7.505, 8: 7.184, 6: 6.945, 4: 6.347, 3: 6.461, 2: 6.574, 1: 6.983 | 4 | 589824 |
| 128 | bidirectional | 768 | 8 | inference | 8: 5.224, 4: 5.307, 2: 5.107, 1: 4.613 | 1 | 589824 |
| 128 | bidirectional | 128 | 48 | training | 48: 2.602, 24: 2.716, 16: 2.740, 12: 2.842, 8: 3.304 | 48 | 786432 |
| 128 | bidirectional | 256 | 48 | training | 48: 11.147, 24: 11.674, 16: 10.861, 12: 11.358, 8: 10.802, 4: 11.399, 2: 12.537 | 8 | 524288 |
| 128 | bidirectional | 384 | 48 | training | 48: 26.870, 24: 26.756, 16: 26.883, 12: 26.991, 8: 26.442, 6: 26.037, 4: 26.002, 3: 26.212, 2: 27.438, 1: 28.414 | 4 | 589824 |
| 128 | bidirectional | 768 | 8 | training | 8: 18.382, 4: 18.574, 2: 18.285, 1: 17.925 | 1 | 589824 |
| 256 | bidirectional | 128 | 48 | inference | 48: 1.756, 24: 1.785, 16: 1.786, 12: 1.805, 8: 1.839 | 48 | 786432 |
| 256 | bidirectional | 256 | 48 | inference | 48: 8.399, 24: 8.475, 16: 7.860, 12: 7.300, 8: 7.193, 4: 7.357, 2: 7.506 | 8 | 524288 |
| 256 | bidirectional | 384 | 48 | inference | 48: 19.247, 24: 19.168, 16: 19.207, 12: 19.188, 8: 18.416, 6: 17.304, 4: 16.712, 3: 16.757, 2: 16.908, 1: 17.302 | 4 | 589824 |
| 256 | bidirectional | 768 | 8 | inference | 8: 13.460, 4: 13.448, 2: 12.972, 1: 11.816 | 1 | 589824 |
| 256 | bidirectional | 128 | 48 | training | 48: 7.552, 24: 7.792, 16: 7.975, 12: 8.213, 8: 8.461 | 48 | 786432 |
| 256 | bidirectional | 256 | 48 | training | 48: 31.879, 24: 31.818, 16: 31.389, 12: 31.012, 8: 31.055, 4: 32.200, 2: 34.293 | 12 | 786432 |
| 256 | bidirectional | 384 | 16 | training | 48: 24.157, 24: 24.049, 16: 23.971, 12: 23.867, 8: 23.846, 6: 23.439, 4: 23.648, 3: 23.857, 2: 24.517, 1: 25.656 | 6 | 884736 |
| 256 | bidirectional | 768 | 4 | training | 8: 25.320, 4: 25.407, 2: 25.135, 1: 24.613 | 1 | 589824 |
| 512 | bidirectional | 128 | 48 | inference | 48: 4.749, 24: 4.792, 16: 4.849, 12: 4.899, 8: 4.861 | 48 | 786432 |
| 512 | bidirectional | 256 | 48 | inference | 48: 21.742, 24: 21.856, 16: 20.748, 12: 19.572, 8: 19.182, 4: 19.529, 2: 19.737 | 8 | 524288 |
| 512 | bidirectional | 384 | 48 | inference | 48: 49.206, 24: 49.475, 16: 50.025, 12: 50.022, 8: 48.550, 6: 45.308, 4: 44.333, 3: 44.121, 2: 44.339, 1: 44.979 | 3 | 442368 |
| 512 | bidirectional | 768 | 8 | inference | 8: 34.433, 4: 34.676, 2: 33.942, 1: 31.025 | 1 | 589824 |
| 512 | bidirectional | 128 | 48 | training | 48: 19.997, 24: 19.971, 16: 19.972, 12: 20.023, 8: 20.561 | 24 | 393216 |
| 512 | bidirectional | 256 | 48 | training | 48: 83.537, 24: 83.124, 16: 80.004, 12: 79.745, 8: 80.188, 4: 81.224, 2: 82.829 | 12 | 786432 |
| 512 | bidirectional | 384 | 16 | training | 48: 63.213, 24: 62.566, 16: 62.123, 12: 61.359, 8: 61.413, 6: 60.301, 4: 60.556, 3: 60.545, 2: 61.480, 1: 62.495 | 6 | 884736 |
| 512 | bidirectional | 768 | 4 | training | 8: 64.756, 4: 65.926, 2: 64.494, 1: 62.831 | 1 | 589824 |
| 64 | outgoing | 384 | 48 | inference | 48: 2.045, 24: 2.063, 16: 2.082, 12: 2.093, 8: 2.019, 6: 1.875, 4: 1.864, 3: 1.937, 2: 2.032, 1: 2.373 | 4 | 589824 |
| 64 | outgoing | 384 | 48 | training | 48: 7.158, 24: 7.249, 16: 7.345, 12: 7.485, 8: 7.590, 6: 7.612, 4: 7.869, 3: 8.239, 2: 8.718, 1: 11.088 | 48 | 7077888 |
| 512 | outgoing | 384 | 16 | inference | 16: 9.547, 8: 9.509, 4: 9.355, 2: 9.419, 1: 9.561 | 4 | 589824 |
| 512 | outgoing | 384 | 16 | training | 16: 38.743, 8: 37.757, 4: 37.462, 2: 38.054, 1: 39.882 | 4 | 589824 |

## Per-kernel time per pair token (ns), one inference call (torch.profiler)

| D | L | B | M = B L^2 | kernels (ns per pair token) |
|---|---|---|---|---|
| 64 | 768 | 1 | 589824 | k3g_kernel<64, 128, 0> 0.153  k1w_kernel<64, 256> 0.131  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT 0.076  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_NTT 0.073  prep_kernel 0.004 |
| 64 | 768 | 2 | 1179648 | k1w_kernel<64, 256> 0.208  k3g_kernel<64, 128, 0> 0.148  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT 0.069  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_NTT 0.069  prep_kernel 0.002 |
| 64 | 768 | 4 | 2359296 | k1w_kernel<64, 256> 0.242  k3g_kernel<64, 128, 0> 0.147  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_NTT 0.069  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT 0.066  prep_kernel 0.001 |
| 64 | 1536 | 1 | 2359296 | k1w_kernel<64, 256> 0.242  k3g_kernel<64, 128, 0> 0.148  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_NTT 0.122  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT 0.114  prep_kernel 0.001 |
| 64 | 384 | 1 | 147456 | k3g_kernel<64, 128, 0> 0.185  k1w_kernel<64, 256> 0.156  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_NTT 0.084  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_TNT 0.069  prep_kernel 0.016 |
| 64 | 384 | 4 | 589824 | k3g_kernel<64, 128, 0> 0.153  k1w_kernel<64, 256> 0.129  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_TNT 0.063  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_NTT 0.062  prep_kernel 0.004 |
| 64 | 384 | 16 | 2359296 | k1w_kernel<64, 256> 0.243  k3g_kernel<64, 128, 0> 0.142  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_TNT 0.058  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_NTT 0.056  prep_kernel 0.001 |
| 64 | 256 | 9 | 589824 | k3g_kernel<64, 128, 0> 0.154  k1w_kernel<64, 256> 0.128  nvjet_tst_128x128_64x8_2x1_2cta_v_bz_NTT 0.061  nvjet_tst_128x128_64x8_2x1_2cta_v_bz_TNT 0.061  prep_kernel 0.004 |
| 64 | 1024 | 1 | 1048576 | k1w_kernel<64, 256> 0.187  k3g_kernel<64, 128, 0> 0.149  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_NTT 0.084  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT 0.084  prep_kernel 0.002 |
| 64 | 1152 | 1 | 1327104 | k1w_kernel<64, 256> 0.230  k3g_kernel<64, 128, 0> 0.149  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_NTT 0.104  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_TNT 0.102  prep_kernel 0.002 |
| 512 | 384 | 1 | 147456 | k1w_kernel<512, 2048> 2.589  k3w_kernel<512, 1024, 0> 1.531  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_TNT 0.484  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_NTT 0.468  ln_stats_kernel<1024, 8> 0.374 |
| 512 | 384 | 8 | 1179648 | k1w_kernel<512, 2048> 2.901  k3w_kernel<512, 1024, 0> 2.041  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_TNT 0.450  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_NTT 0.443  ln_stats_kernel<1024, 8> 0.300 |
| 512 | 384 | 48 | 7077888 | k1w_kernel<512, 2048> 3.121  k3w_kernel<512, 1024, 0> 2.641  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_TNT 0.451  nvjet_tst_192x128_64x6_2x1_2cta_v_bz_NTT 0.448  ln_stats_kernel<1024, 8> 0.286 |
| 512 | 768 | 1 | 589824 | k1w_kernel<512, 2048> 3.131  k3w_kernel<512, 1024, 0> 1.819  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_NTT 0.575  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT 0.572  ln_stats_kernel<1024, 8> 0.341 |
| 512 | 768 | 4 | 2359296 | k1w_kernel<512, 2048> 3.021  k3w_kernel<512, 1024, 0> 2.466  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_NTT 0.579  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT 0.563  ln_stats_kernel<1024, 8> 0.290 |
| 512 | 1536 | 1 | 2359296 | k1w_kernel<512, 2048> 3.097  k3w_kernel<512, 1024, 0> 2.557  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_NTT 0.981  nvjet_tst_128x256_64x6_2x1_2cta_v_bz_TNT 0.971  ln_stats_kernel<1024, 8> 0.290 |

## B = 1 against main, CUDA graph, six interleaved rounds (median ms; min of rounds in brackets)

| D | module | L | mode | main | branch | branch / main |
|---|---|---|---|---|---|---|
| 64 | bidirectional | 128 | inference | 0.0308 (0.0288) | 0.0308 (0.0308) | 1.000 |
| 64 | bidirectional | 128 | training | 0.1120 (0.1106) | 0.1119 (0.1118) | 1.000 |
| 64 | bidirectional | 384 | inference | 0.0869 (0.0862) | 0.0869 (0.0868) | 0.999 |
| 64 | bidirectional | 384 | training | 0.3110 (0.3077) | 0.3113 (0.3107) | 1.001 |
| 64 | bidirectional | 768 | inference | 0.2606 (0.2562) | 0.2606 (0.2604) | 1.000 |
| 64 | bidirectional | 768 | training | 0.9325 (0.9315) | 0.9352 (0.9309) | 1.003 |
| 64 | outgoing | 128 | inference | 0.0267 (0.0267) | 0.0268 (0.0267) | 1.004 |
| 64 | outgoing | 128 | training | 0.0929 (0.0923) | 0.0941 (0.0923) | 1.013 |
| 64 | outgoing | 384 | inference | 0.0611 (0.0598) | 0.0616 (0.0614) | 1.007 |
| 64 | outgoing | 384 | training | 0.2265 (0.2260) | 0.2268 (0.2264) | 1.001 |
| 64 | outgoing | 768 | inference | 0.1764 (0.1751) | 0.1764 (0.1762) | 1.000 |
| 64 | outgoing | 768 | training | 0.6742 (0.6511) | 0.6753 (0.6748) | 1.002 |
| 128 | bidirectional | 128 | inference | 0.0410 (0.0410) | 0.0411 (0.0410) | 1.001 |
| 128 | bidirectional | 128 | training | 0.1768 (0.1744) | 0.1774 (0.1773) | 1.004 |
| 128 | bidirectional | 384 | inference | 0.1520 (0.1512) | 0.1521 (0.1520) | 1.001 |
| 128 | bidirectional | 384 | training | 0.6112 (0.5827) | 0.6113 (0.6109) | 1.000 |
| 128 | bidirectional | 768 | inference | 0.5188 (0.5182) | 0.5190 (0.5181) | 1.000 |
| 128 | bidirectional | 768 | training | 2.1836 (2.1566) | 2.1780 (2.0584) | 0.997 |
| 128 | outgoing | 128 | inference | 0.0330 (0.0309) | 0.0330 (0.0309) | 0.998 |
| 128 | outgoing | 128 | training | 0.1244 (0.1213) | 0.1258 (0.1215) | 1.011 |
| 128 | outgoing | 384 | inference | 0.1033 (0.1026) | 0.1026 (0.1024) | 0.993 |
| 128 | outgoing | 384 | training | 0.4102 (0.3997) | 0.4110 (0.4107) | 1.002 |
| 128 | outgoing | 768 | inference | 0.3365 (0.3363) | 0.3366 (0.3363) | 1.000 |
| 128 | outgoing | 768 | training | 1.3587 (1.3204) | 1.3783 (1.3596) | 1.014 |
| 256 | bidirectional | 128 | inference | 0.0677 (0.0672) | 0.0673 (0.0665) | 0.994 |
| 256 | bidirectional | 128 | training | 0.3192 (0.3174) | 0.3214 (0.3212) | 1.007 |
| 256 | bidirectional | 384 | inference | 0.3395 (0.3392) | 0.3389 (0.3388) | 0.998 |
| 256 | bidirectional | 384 | training | 1.5326 (1.5205) | 1.5407 (1.5238) | 1.005 |
| 256 | bidirectional | 768 | inference | 1.4331 (1.3787) | 1.4025 (1.3411) | 0.979 |
| 256 | bidirectional | 768 | training | 5.9954 (5.9894) | 6.0145 (6.0065) | 1.003 |
| 256 | outgoing | 128 | inference | 0.0529 (0.0493) | 0.0575 (0.0513) | 1.086 |
| 256 | outgoing | 128 | training | 0.2539 (0.2535) | 0.2555 (0.2548) | 1.006 |
| 256 | outgoing | 384 | inference | 0.2390 (0.2389) | 0.2388 (0.2385) | 0.999 |
| 256 | outgoing | 384 | training | 1.0419 (1.0414) | 1.0486 (1.0481) | 1.006 |
| 256 | outgoing | 768 | inference | 0.8974 (0.8917) | 0.8992 (0.8925) | 1.002 |
| 256 | outgoing | 768 | training | 3.8742 (3.8460) | 3.8925 (3.8512) | 1.005 |
| 384 | bidirectional | 128 | inference | 0.0981 (0.0971) | 0.0970 (0.0969) | 0.989 |
| 384 | bidirectional | 128 | training | 0.4472 (0.4454) | 0.4501 (0.4499) | 1.006 |
| 384 | bidirectional | 384 | inference | 0.5608 (0.5594) | 0.5600 (0.5596) | 0.999 |
| 384 | bidirectional | 384 | training | 2.6301 (2.6083) | 2.6483 (2.5150) | 1.007 |
| 384 | bidirectional | 768 | inference | 2.4986 (2.4210) | 2.5248 (2.4609) | 1.010 |
| 384 | bidirectional | 768 | training | 10.6413 (10.6000) | 10.6448 (10.5829) | 1.000 |
| 384 | outgoing | 128 | inference | 0.0709 (0.0698) | 0.0701 (0.0698) | 0.988 |
| 384 | outgoing | 128 | training | 0.3361 (0.3339) | 0.3373 (0.3372) | 1.004 |
| 384 | outgoing | 384 | inference | 0.3864 (0.3863) | 0.3861 (0.3816) | 0.999 |
| 384 | outgoing | 384 | training | 1.6882 (1.6589) | 1.6886 (1.6679) | 1.000 |
| 384 | outgoing | 768 | inference | 1.6077 (1.5382) | 1.5728 (1.5276) | 0.978 |
| 384 | outgoing | 768 | training | 6.6456 (6.6220) | 6.7544 (6.6380) | 1.016 |
| 512 | bidirectional | 128 | inference | 0.1314 (0.1311) | 0.1312 (0.1311) | 0.998 |
| 512 | bidirectional | 128 | training | 0.5607 (0.5602) | 0.5623 (0.5622) | 1.003 |
| 512 | bidirectional | 384 | inference | 0.8441 (0.8331) | 0.8569 (0.8338) | 1.015 |
| 512 | bidirectional | 384 | training | 3.8150 (3.7772) | 3.8121 (3.7982) | 0.999 |
| 512 | bidirectional | 768 | inference | 3.7752 (3.7460) | 3.7868 (3.7219) | 1.003 |
| 512 | bidirectional | 768 | training | 15.5570 (15.5392) | 15.5634 (15.5083) | 1.000 |
| 512 | outgoing | 128 | inference | 0.0930 (0.0925) | 0.0937 (0.0926) | 1.007 |
| 512 | outgoing | 128 | training | 0.4113 (0.4107) | 0.4121 (0.4121) | 1.002 |
| 512 | outgoing | 384 | inference | 0.5447 (0.5427) | 0.5429 (0.5420) | 0.997 |
| 512 | outgoing | 384 | training | 2.3918 (2.3579) | 2.4099 (2.3880) | 1.008 |
| 512 | outgoing | 768 | inference | 2.3602 (2.3379) | 2.3481 (2.3197) | 0.995 |
| 512 | outgoing | 768 | training | 9.5365 (9.4910) | 9.4983 (9.4869) | 0.996 |

## B = 1 per kernel, final sources (torch.profiler, 20 steps, mean of 2 rounds, us)

| cell | step main | step branch | k1wb main | k1wb branch |
|---|---|---|---|---|
| 384-1-768-training | 6532.9 | 6472.5 | 836.8 | 807.3 |
| 256-0-384-training | 1533.8 | 1519.8 | 246.2 | 236.1 |
| 512-0-384-training | 3667.7 | 3709.9 | 595.2 | 609.3 |
| 256-1-128-training | 255.6 | 254.6 | 18.8 | 17.7 |
| 384-0-128-training | 454.9 | 454.7 | 48.6 | 46.9 |
