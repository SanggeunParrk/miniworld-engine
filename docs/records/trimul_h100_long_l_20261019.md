# TriMul H100 — long and ragged lengths (L > 1024, any L in inference), 2026-10-19

MiniWorld's inference pads tokens to rungs up to 2560 (beyond: multiples of 8); before this change H100 served TriMul only at
L128 ... 1024 (multiples of 128) and every other length ran the module's PyTorch statements. Branch `h100/trimul-long-l` off main
5461755f. H100 80GB HBM3 (132 SMs), torch 2.13.0+cu129, bf16 pair, FP32 parameters, a token mask. Scratch harnesses:
`~/miniworld-engine-scratch/wt-longl-logs/` (`bench_longl.py` CUDA-graph timing, `prof_longl.py` per-kernel torch profiler,
`probe_b1.py` B1 build variants).

## Contract after the change (`integrations.trimul_h100`)

| width | inference | training |
|---|---|---|
| D64 (B <= 48) / D128, both modules, every direction | any L >= 16 | L % 128 == 0, 128 <= L <= 10240 |
| D256 / D384 / D512 | unchanged up to L1024; above: L % 128 == 0 through the B200-style port (single-direction D256 / D384: the packaged K1 / K3 at any L, as before) | unchanged up to L1024; above: L % 128 == 0 (B200-style port) |

Index bound: B L^2 < 2^31 (`_index_ok`, L rounded up to 16). `MAX_TRAINING_LENGTH = 10240` is B200's bound; no H100 kernel has a
smaller one (planes are addressed through TMA coordinates or 64-bit offsets), memory ends first.

## Design (per kernel)

| stage | change |
|---|---|
| inference K1 / K3 (D64 / D128) | none. The tile table has `by_n` buckets to N 4096 and one sample pads the planes to `Np` (L rounded up to 16, zero pad rows from K1). Bug fixed: since the batched-samples change one sample at L % 16 != 0 passed `rows = Np` to K1 / K3, which the batched-geometry check rejects; B = 1 leaves `rows` unset again |
| B samples at L % 16 != 0 | `update_inference` pads each sample to the next multiple of 16 on the host (zero entries, masked pad tokens; K1's `m[i] & m[j]` zeroes the pad tokens' left / right rows and columns, so a real pair contracts the same terms) and slices the output back |
| D128 training (inference K1 front, `save_k3`, B1, B7; single: `width_k1`, `save_k3`, `single_b1`, `single_b7`) | length guards only: `h100_b1.Plan` and `h100_single.Plan` took `n in (128, ..., 1024)`, now `n % 128 == 0`. B1 is built per L (`TRAIN_L`); B7 runs the `b7_768` source at every L but 384; the configs take their "768" entries above L768 |
| D64 training (`save_k3_d64`, `d64_b1w` / `uni64_b1w`, `b2w`, `finw`) | none (the `supports` helpers follow the new rule) |
| B200-style port (D256 / D384 / D512) | the length guard (`LENGTHS` tuple) became `n % 128 == 0` |
| integration | `_training_length` / `_index_ok` / `MIN_INFERENCE_LENGTH` replace the L128 ... 1024 tuples; single-direction D256 / D384 inference beyond L1024 stays on the packaged K1 / K3 it ran before (`_legacy_native_faster`) |

## Correctness

`tests/gpu/sm90/test_trimul_h100_long_l_gpu.py`, 103 / 103: contract; inference (D64 B = 1 and 4, D128; every direction; L 48 /
200 / 520 / 1040 / 1536 / 2048 / 2560; with and without a mask) against the FP32 PyTorch statement (computed in row chunks so L2560
fits), relative error < 1e-2, bitwise run to run, B = 4 equal to four one-sample calls; training (every gradient, row dropout, a
mask) at L 1152 / 1536 / 2048 / 2560 (D64, D128, every direction), D64 B = 4 L1152, D64 L4096 (bidirectional and outgoing), D128
L2816 (bidirectional) and L3584 (incoming), the reference above L2560 on the job's second GPU: y < 5e-3, gradients < 2e-2, bitwise run
to run; D64 B = 4 L2560 against four one-sample calls; the B200-style port at D256 (every direction, L1152 / 1536), D384 (bidirectional,
outgoing) and D512 (bidirectional, incoming) at L1152 against the reference; CUDA-graph capture and replay with live inputs, masks,
dropout scales and weights (inference D64 B4 L520 / 1536, D128 L2048, outgoing D128 L200; training D64 B4 L1536, outgoing D64 L2048,
D128 L1536, incoming D128 L1152).

The B200-style port's gradients are not bitwise run to run (its backward sums LN / gate partials with `atomicAdd`,
`b200_sources/wide_bwd.cu`); y is. Pre-existing: at D256 L512 (the same port, served before) one of the eleven gradients differs
between two runs. The new test asserts < 1e-3 relative for those.

Existing H100 TriMul tests, re-run: `test_trimul_h100_b200style_gpu` 73, `test_trimul_h100_d64_b200gate_gpu` 22,
`test_trimul_h100_small_buckets_gpu` 15, `test_trimul_single_h100_gpu` 8, `test_h100_trimul_batch_gpu` 37,
`test_h100_module_wiring_gpu` 34, `test_trimul_d64_training_gpu` 23 + 2 skipped, `test_trimul_uni_d64_training_gpu` 30 + 4 skipped,
`test_trimul_uni_wide_inference_gpu` 5, `test_trimul_uni_wide_training_gpu` 25, `test_trimul_wide_inference_gpu` 137,
`test_trimul_wide_preparation_gpu` 6, `test_trimul_wide_projection_key` 13, `test_trimul_hot_path_gpu` 3,
`test_trimul_nograd_dropout_gpu` 2: all pass. (A first run with six test processes on one GPU next to the 60 GB module-wiring
process OOMed in four files; rerun one file at a time, all pass.)

## Served lengths unchanged (ms, CUDA graph, median of 5 x 20 replays; two interleaved rounds, main and worktree swapped between the two GPUs)

| case | main | worktree | worktree / main |
|---|---|---|---|
| D64 B4 bidirectional L384 inference | 0.4301-0.4325 | 0.4299-0.4320 | 0.999 |
| D64 B4 bidirectional L384 training | 1.3346-1.3432 | 1.3359-1.3438 | 1.001 |
| D64 B4 bidirectional L768 inference | 1.7826-1.7898 | 1.7830-1.7890 | 1.000 |
| D64 B4 bidirectional L768 training | 5.7202-5.8362 | 5.7412-5.8956 | 1.005 |
| D64 bidirectional L384 inference | 0.1183-0.1185 | 0.1181-0.1185 | 0.999 |
| D64 bidirectional L384 training | 0.3884-0.3940 | 0.3894-0.3950 | 0.999 |
| D64 bidirectional L768 inference | 0.4587-0.4622 | 0.4579-0.4607 | 0.998 |
| D64 bidirectional L768 training | 1.4737-1.4911 | 1.4820-1.4948 | 1.004 |
| D64 outgoing L384 inference | 0.0771-0.0780 | 0.0770-0.0783 | 1.002 |
| D64 outgoing L384 training | 0.2534-0.2549 | 0.2499-0.2552 | 0.997 |
| D64 outgoing L768 inference | 0.2965-0.3009 | 0.2963-0.2998 | 0.998 |
| D64 outgoing L768 training | 0.9067-0.9152 | 0.9061-0.9231 | 1.001 |
| D64 incoming L384 inference | 0.0794-0.0801 | 0.0793-0.0801 | 0.997 |
| D64 incoming L384 training | 0.2553-0.2566 | 0.2514-0.2571 | 0.996 |
| D64 incoming L768 inference | 0.3020-0.3032 | 0.3012-0.3030 | 0.999 |
| D64 incoming L768 training | 0.9017-0.9071 | 0.9013-0.9102 | 1.001 |
| D128 bidirectional L384 inference | 0.2344-0.2389 | 0.2346-0.2390 | 1.000 |
| D128 bidirectional L384 training | 0.9547-0.9600 | 0.9502-0.9608 | 0.997 |
| D128 bidirectional L768 inference | 0.9594-1.0001 | 0.9560-1.0009 | 1.009 |
| D128 bidirectional L768 training | 3.8614-3.9164 | 3.8748-3.9129 | 1.001 |
| D128 outgoing L384 inference | 0.1431-0.1448 | 0.1419-0.1448 | 0.995 |
| D128 outgoing L384 training | 0.7759-0.7799 | 0.7632-0.7780 | 0.994 |
| D128 outgoing L768 inference | 0.5604-0.5669 | 0.5607-0.5658 | 0.999 |
| D128 outgoing L768 training | 2.7794-2.7881 | 2.7698-2.7934 | 1.000 |
| D128 incoming L384 inference | 0.1462-0.1465 | 0.1439-0.1464 | 0.995 |
| D128 incoming L384 training | 0.7737-0.7796 | 0.7731-0.7788 | 0.999 |
| D128 incoming L768 inference | 0.5621-0.5814 | 0.5726-0.5804 | 1.005 |
| D128 incoming L768 training | 2.7716-2.8100 | 2.7776-2.7976 | 1.002 |

(The ratio is of the means of the four runs per side; the code path of these cells did not change.)

## New lengths (ms, CUDA graph; peak GB in parentheses for ours)

"PyTorch" = the module's PyTorch statements (what H100 ran at these lengths before; bf16 autocast, CUDA graph); cuEquivariance =
`implementation="cuequivariance"` under bf16 autocast (its bidirectional composition: vendor input LN / gated projection and output
LN around the two contractions).

| module | L | mode | ours | PyTorch | cuEquivariance | × PyTorch | × cuEquivariance |
|---|---|---|---|---|---|---|---|
| bidirectional D64, B = 4 | 1536 | inference | 9.50 (9.0) | 119.04 | 20.13 | 12.5 | 2.12 |
| bidirectional D64, B = 4 | 1536 | training | 30.47 (19.4) | 248.36 | 97.56 | 8.15 | 3.20 |
| bidirectional D64, B = 4 | 2048 | inference | 20.09 (16.0) | 214.76 | 39.03 | 10.7 | 1.94 |
| bidirectional D64, B = 4 | 2048 | training | 62.27 (34.4) | 468.93 | 181.60 | 7.53 | 2.92 |
| bidirectional D64, B = 4 | 2560 | inference | 36.71 (25.0) | 351.62 | 65.95 | 9.58 | 1.80 |
| bidirectional D64, B = 4 | 2560 | training | 112.54 (53.7) | OOM | OOM | - | - |
| bidirectional D128 | 1536 | inference | 5.19 (4.5) | 60.96 | 14.51 | 11.7 | 2.79 |
| bidirectional D128 | 1536 | training | 19.72 (10.3) | 191.54 | 55.33 | 9.71 | 2.81 |
| bidirectional D128 | 2048 | inference | 10.59 (8.0) | 111.18 | 26.95 | 10.5 | 2.54 |
| bidirectional D128 | 2048 | training | 40.06 (18.1) | 355.27 | 102.82 | 8.87 | 2.57 |
| bidirectional D128 | 2560 | inference | 19.26 (12.5) | 175.94 | 44.57 | 9.14 | 2.31 |
| bidirectional D128 | 2560 | training | 69.47 (28.3) | 563.60 | 159.07 | 8.11 | 2.29 |
| bidirectional D64, B = 1 | 1536 | inference / training | 2.39 / 7.63 | 29.89 / 91.50 | | 12.5 / 12.0 | |
| bidirectional D64, B = 1 | 2048 | inference / training | 4.98 / 15.74 | 53.69 / 169.05 | | 10.8 / 10.7 | |
| bidirectional D64, B = 1 | 2560 | inference / training | 9.14 / 28.19 | 87.81 / 273.29 | | 9.61 / 9.69 | |
| outgoing D128 | 1536 | inference / training | 2.96 / 12.89 | 44.59 / 183.86 | | 15.0 / 14.3 | |
| outgoing D128 | 2048 | inference / training | 6.06 / 25.30 | 80.38 / 337.61 | | 13.3 / 13.3 | |
| outgoing D128 | 2560 | inference / training | 10.61 / 43.77 | 119.37 / 524.41 | | 11.3 / 12.0 | |

## Per-kernel time per pair token (µs per million B L^2 tokens; torch profiler, eager launches, 10 steps)

| case | kernel | L768 | L2048 | L2560 |
|---|---|---|---|---|
| D128 bidirectional inference | K1 `tmn_k1_z128_h256` | 536 | 586 | 602 |
| | K3 `tmn_k3_z128_h256` | 392 | 439 | 440 |
| | two contractions (cuBLAS) | 625 | 1456 | 1790 |
| D128 bidirectional training | `b7_joint` | 2025 | 2309 | 2323 |
| | `b1_fused` | 1067 | 1472 | 1505 |
| | `save_k3` | 521 | 755 | 785 |
| | K1 | 545 | 581 | 612 |
| | six contractions (cuBLAS) | 1919 | 4283 | 5395 |
| | step | 6133 | 9409 | 10626 |
| D128 outgoing training | `single_b1` / `b7_joint` / `save_k3` / `width_k1` | 1548 / 1387 / 397 / 457 | 1581 / 1384 / 515 / 449 | |
| D64 B4 bidirectional inference | K1 / K3 | 265 / 182 | 267 / 212 | |
| D64 B4 bidirectional training | `d64_b2w` / `d64_b1w` / `save_k3` / K1 | 565 / 297 / 204 / 245 | 686 / 299 / 241 / 239 | |
| D64 outgoing training | `uni64_b2w` / `uni64_b1w` / `save_k3` / K1 | 396 / 223 / 202 / 148 | 406 / 216 / 203 / 147 | |

The contractions grow as L^3 / L^2 = L (they run at 730 TFLOP/s at L2048 / 2560: each D128 inference contraction at L2560 is 4.3 TFLOP in 5.9 ms). The
row kernels should be flat per token; K1 / K3 / B7 are within 12-14 %, the D64 B4 `b2w` +21 %, D128 `b1_fused` +38 % and `save_k3`
+50 % (both read and write the channel-strided `[c, L, L]` planes, whose channel stride grows as L^2; the D64 K3 at L2048 also runs
another tile bucket). At L2560 the excess of `b1_fused` + `save_k3` over their L768 rate is 4.6 ms of the 69.6 ms step (6.6 %).

B1 build variants (`probe_b1.py`, D128 bidirectional, bitwise-equal gradients in every variant), `b1_fused` µs / step ms:

| variant | L1536 | L2048 | L2560 |
|---|---|---|---|
| shipped ("768" defines) | 2631 / 19.24 | 6248 / 39.29 | 9619 / 69.56 |
| `B1_EXT_MASK=1` (dropout-fragment cache up to a period of 12) | 2989 / 19.39 | 5790 / 39.43 | 9032 / 68.67 |
| `B1_DTRI_STORE_C=32` | 2860 / 19.10 | 6153 / 39.83 | 9603 / 70.17 |
| both | 3153 / 19.24 | 5676 / 39.22 | 8952 / 68.64 |
| `B1_EXT_MASK=1`, store 128 | 3211 / 19.49 | 5821 / 38.82 | 9046 / 68.84 |
| `B1_EXT_MASK=1`, `B1_GATE_DEMOTE=3` | 3239 / 19.14 | 5594 / 38.95 | 9070 / 69.34 |

B1 moves by up to 10 %, the step by at most 1.3 % and not in one direction across lengths: not adopted.

## Not served / open

- Wide widths (D256 / D384 / D512) beyond L1024 at L % 128 != 0, and with B > 1: the PyTorch statements (the B200-style port tiles
  64 x 2 tokens and takes one sample).
- Memory: D128 bidirectional training ends near L4300 on 80 GB (28.3 GB at L2560), D64 B = 4 training near L3100 (53.7 GB at L2560).
- Long-L row kernels `b1_fused` / `save_k3` (D128) and `b2w` (D64) run 21-50 % slower per token than at L768 (see above).
