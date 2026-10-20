# TriMul H100 D128 / D256 / D384 / D512 — batched samples (B samples in one call), 2026-10-19

Extends the D64 batched-samples path (5461755f) to every H100 TriMul width, so MiniWorld can fold structures / augments (D128 pair
trunk, wider stacks) into one call. Branch `h100/trimul-batched-wide` off main 545b6850. H100 80GB HBM3, torch 2.13.0+cu129, bf16
pair, FP32 parameters, CUDA-graph timing (best of 5 runs of up to 30 replays after warm-up). One GPU per Slurm job. Scratch
harnesses: `~/miniworld-engine-scratch/wt-bwide-logs/` (`sweep_wide.py`, `bench_wide.py`, `b1_compare.py` + `analyze_b1.py`,
`probe_dw.py`, `prof_k.py`); the baseline is a `git archive` of 545b6850 (`wt-bwide-logs/base`).

## Design

As at D64: K1 stacks the samples along its slow token axis (`rows` = B L), the planes are `[c, B L, L]`, the contraction folds the
samples into cuBLAS's batch dim, everything after it is token-flat and reads the row-dropout scale `[B L, D]` at row `s L + j`.
Per kernel: [gpus/h100/trimul/trimul.md](../gpus/h100/trimul/trimul.md#batched-samples-b--48-per-launch-any-b-per-call-every-width). Paths: D128 bidirectional
= inference K1 / K3 (inference), training front K1 + `save_k3` + B1 + B7 (training); D128 single direction = inference K1 / K3,
`h100_native.Front` (upstream header) + `save_k3` + `single_b1` + `single_b7`; D256 / D384 / D512 = the B200-style port, every
direction and mode (B > 1 never takes the preserved single-sample cells).

## Batch-limit sweep (correctness)

B in {2, 4, 8, 16, 32, 48} x L in {128, 384, 768, 1536} x {bidirectional, outgoing, incoming} x {inference, training} x D in {128,
256, 384, 512}, a token mask and a dropout scale drawn per sample, against B calls of one sample on the same path (the D256+ B = 1
calls forced onto the port): every cell that fits in 80 GB passes (D128: 63 / 72 inference, 58 / 72 training; D256: 54 / 72, 47 / 72;
D384: 54 / 72, 42 / 72; D512: 48 / 72, 36 / 72; the rest OOM). In every passing cell y and dx are **bitwise equal** to the B
one-sample calls and y is bitwise run to run; D128 gradients are bitwise run to run, D256+ gradients agree run to run to 7.4e-8
relative (the port's `atomicAdd` row / affine sums, as for one sample). Weight gradients against the sum of the per-sample ones: up
to 1.2e-3 relative (D128 L768 B32), 8.9e-4 (D256+). Against the FP32 PyTorch statement summed sample by sample (exact for the batch):

| cell | parameter group | batch vs FP32 | B one-sample calls vs FP32 | batch vs B calls |
|---|---|---|---|---|
| D128 L768 B16 bidirectional | front projections (B7) | 5.8-6.5e-3 | 5.8-6.5e-3 | 4.5e-4 |
| D128 L768 B16 bidirectional | gate / output (B1) | 5.5-6.2e-3 | 5.5-6.2e-3 | 7.9e-5 |
| D128 L768 B16 bidirectional | LN affines | 3.3-6.0e-3 | identical | <= 7.4e-7 |
| D128 L768 B16 outgoing | front projections | 5.8-6.5e-3 | 5.8-6.4e-3 | 5.6e-4 |
| D256 L384 B8 bidirectional | front projections | 5.5-6.0e-3 | 5.5-6.0e-3 | 9.2e-5 |
| D256 L384 B8 bidirectional | gate / output | 5.0-6.0e-3 | 5.0-6.0e-3 | 1.1e-4 |

The batch is as close to the FP32 statement as the per-sample sum; the difference between them is the FP32 accumulation order of
longer per-CTA chains.

Largest B that fit (bidirectional; single direction where larger), peak GB of the batched call:

| mode | module | D | L | largest B | peak GB |
|---|---|---|---|---|---|
| inference | bidirectional | 128 | 128 | 48 | 1.7 |
| inference | bidirectional | 128 | 384 | 48 | 15.3 |
| inference | bidirectional | 128 | 768 | 48 | 60.9 |
| inference | bidirectional | 128 | 1536 | 8 | 41.1 |
| inference | bidirectional | 256 | 128 | 48 | 3.4 |
| inference | bidirectional | 256 | 384 | 48 | 30.5 |
| inference | bidirectional | 256 | 768 | 16 | 40.9 |
| inference | bidirectional | 256 | 1536 | 4 | 41.7 |
| inference | bidirectional | 384 | 128 | 48 | 5.1 |
| inference | bidirectional | 384 | 384 | 48 | 45.8 |
| inference | bidirectional | 384 | 768 | 16 | 61.3 |
| inference | bidirectional | 384 | 1536 | 4 | 62.6 |
| inference | bidirectional | 512 | 128 | 48 | 6.8 |
| inference | bidirectional | 512 | 384 | 48 | 61.0 |
| inference | bidirectional | 512 | 768 | 8 | 41.2 |
| inference | bidirectional | 512 | 1536 | 2 | 50.2 |
| inference | single direction | 128 | 128 | 48 | 1.2 |
| inference | single direction | 128 | 384 | 48 | 10.2 |
| inference | single direction | 128 | 768 | 48 | 40.7 |
| inference | single direction | 128 | 1536 | 8 | 27.6 |
| inference | single direction | 256 | 128 | 48 | 2.3 |
| inference | single direction | 256 | 384 | 48 | 20.4 |
| inference | single direction | 256 | 768 | 16 | 27.4 |
| inference | single direction | 256 | 1536 | 4 | 28.2 |
| inference | single direction | 384 | 128 | 48 | 3.4 |
| inference | single direction | 384 | 384 | 48 | 30.6 |
| inference | single direction | 384 | 768 | 16 | 41.0 |
| inference | single direction | 384 | 1536 | 4 | 42.3 |
| inference | single direction | 512 | 128 | 48 | 4.6 |
| inference | single direction | 512 | 384 | 48 | 40.8 |
| inference | single direction | 512 | 768 | 8 | 27.7 |
| inference | single direction | 512 | 1536 | 2 | 29.3 |
| training | bidirectional | 128 | 128 | 48 | 3.7 |
| training | bidirectional | 128 | 384 | 48 | 32.2 |
| training | bidirectional | 128 | 768 | 16 | 42.9 |
| training | bidirectional | 128 | 1536 | 4 | 42.9 |
| training | bidirectional | 256 | 128 | 48 | 11.7 |
| training | bidirectional | 256 | 384 | 32 | 70.0 |
| training | bidirectional | 256 | 768 | 8 | 70.0 |
| training | bidirectional | 256 | 1536 | 2 | 70.0 |
| training | bidirectional | 384 | 128 | 48 | 17.6 |
| training | bidirectional | 384 | 384 | 16 | 52.5 |
| training | bidirectional | 384 | 768 | 4 | 52.5 |
| training | bidirectional | 512 | 128 | 48 | 23.4 |
| training | bidirectional | 512 | 384 | 16 | 69.9 |
| training | bidirectional | 512 | 768 | 4 | 69.9 |
| training | single direction | 128 | 128 | 48 | 2.7 |
| training | single direction | 128 | 384 | 48 | 23.7 |
| training | single direction | 128 | 768 | 32 | 63.1 |
| training | single direction | 128 | 1536 | 8 | 63.1 |
| training | single direction | 256 | 128 | 48 | 8.0 |
| training | single direction | 256 | 384 | 48 | 71.2 |
| training | single direction | 256 | 768 | 8 | 47.5 |
| training | single direction | 256 | 1536 | 2 | 47.5 |
| training | single direction | 384 | 128 | 48 | 11.9 |
| training | single direction | 384 | 384 | 32 | 71.1 |
| training | single direction | 384 | 768 | 8 | 71.1 |
| training | single direction | 384 | 1536 | 2 | 71.1 |
| training | single direction | 512 | 128 | 48 | 15.9 |
| training | single direction | 512 | 384 | 16 | 47.4 |
| training | single direction | 512 | 768 | 4 | 47.4 |

The D128 training rows are the sweep re-run with the final `B1_BATCHED` build (58 / 58 cells pass, as with the first build).
Memory per pair token: training D128 bidirectional 4.6 kB, D256 15 kB, D384 22 kB, D512 30 kB (the port saves projection, gate and,
at D512, the normalized input); inference D128 2.2 kB, D512 8.6 kB (bidirectional).

## B = 1 against main (545b6850), two interleaved rounds

y / dx / dW: "bitwise" = equal to main (digest of the whole tensor); a number = relative difference of the weight gradients, only
where main itself is not bitwise run to run (the port's atomics and the preserved D256+ training cells). Times in ms, main / worktree
ranges of the two rounds, ratio of the minima. D128 rows: final build (`B1_BATCHED` split); D256+ rows: the same sources (the B1
change does not touch them).

| cell | y infer | y train | dx | dW | main infer | worktree infer | main train | worktree train | infer ratio | train ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| D128-L384-dir0 | bitwise | bitwise | bitwise | bitwise | 0.2338-0.2343 | 0.2343-0.2345 | 0.9335-0.9388 | 0.9331-0.9358 | 1.002 | 1.000 |
| D128-L384-dir1 | bitwise | bitwise | bitwise | bitwise | 0.1433-0.1447 | 0.1436-0.1446 | 0.7546-0.7557 | 0.7439-0.7449 | 1.002 | 0.986 |
| D128-L384-dir2 | bitwise | bitwise | bitwise | bitwise | 0.1455-0.1464 | 0.1450-0.1454 | 0.7557-0.7569 | 0.7466-0.7516 | 0.997 | 0.988 |
| D128-L768-dir0 | bitwise | bitwise | bitwise | bitwise | 0.9171-0.9249 | 0.9251-0.9297 | 3.7750-3.7867 | 3.7691-3.7758 | 1.009 | 0.998 |
| D128-L768-dir1 | bitwise | bitwise | bitwise | bitwise | 0.5362-0.5373 | 0.5373-0.5382 | 2.7672-2.7701 | 2.7639-2.7703 | 1.002 | 0.999 |
| D128-L768-dir2 | bitwise | bitwise | bitwise | bitwise | 0.5514-0.5527 | 0.5536-0.5544 | 2.7621-2.7689 | 2.7327-2.7404 | 1.004 | 0.989 |
| D256-L384-dir0 | bitwise | bitwise | bitwise | 6.0e-07 | 0.5772-0.5775 | 0.5755-0.5764 | 2.6234-2.6327 | 2.6221-2.6308 | 0.997 | 0.999 |
| D256-L384-dir1 | bitwise | bitwise | bitwise | bitwise | 0.3486-0.3489 | 0.3471-0.3476 | 1.7306-1.7309 | 1.7300-1.7300 | 0.996 | 1.000 |
| D256-L384-dir2 | bitwise | bitwise | bitwise | 6.9e-07 | 0.3463-0.3484 | 0.3474-0.3478 | 1.7501-1.7532 | 1.7534-1.7543 | 1.003 | 1.002 |
| D256-L768-dir0 | bitwise | bitwise | bitwise | bitwise | 2.3994-2.4120 | 2.3879-2.4441 | 10.7858-10.8234 | 10.7267-10.7916 | 0.995 | 0.995 |
| D256-L768-dir1 | bitwise | bitwise | bitwise | 6.3e-07 | 1.3682-1.3729 | 1.3464-1.3672 | 6.7378-6.7507 | 6.7505-6.7583 | 0.984 | 1.002 |
| D256-L768-dir2 | bitwise | bitwise | bitwise | 6.2e-07 | 1.3663-1.3862 | 1.3588-1.3966 | 6.7942-6.7976 | 6.7952-6.7953 | 0.994 | 1.000 |
| D384-L384-dir0 | bitwise | bitwise | bitwise | 5.6e-07 | 1.0841-1.0924 | 1.0866-1.0949 | 4.5951-4.6064 | 4.6122-4.6142 | 1.002 | 1.004 |
| D384-L384-dir1 | bitwise | bitwise | bitwise | bitwise | 0.6305-0.6314 | 0.6267-0.6280 | 2.7866-2.7880 | 2.7968-2.7981 | 0.994 | 1.004 |
| D384-L384-dir2 | bitwise | bitwise | bitwise | 5.5e-07 | 0.6261-0.6275 | 0.6252-0.6293 | 2.8118-2.8209 | 2.8120-2.8249 | 0.999 | 1.000 |
| D384-L768-dir0 | bitwise | bitwise | bitwise | 5.8e-07 | 4.5699-4.5880 | 4.5832-4.5935 | 18.8001-19.0454 | 18.8397-19.1125 | 1.003 | 1.002 |
| D384-L768-dir1 | bitwise | bitwise | bitwise | 5.4e-07 | 2.5591-2.5644 | 2.5526-2.5556 | 11.2646-11.2940 | 11.2655-11.3140 | 0.997 | 1.000 |
| D384-L768-dir2 | bitwise | bitwise | bitwise | 5.8e-07 | 2.5348-2.5652 | 2.5664-2.5919 | 11.3300-11.3472 | 11.3227-11.3599 | 1.012 | 0.999 |
| D512-L384-dir0 | bitwise | bitwise | bitwise | 5.1e-07 | 1.8934-1.9063 | 1.8788-1.9120 | 6.9772-7.0229 | 6.9654-6.9801 | 0.992 | 0.998 |
| D512-L384-dir1 | bitwise | bitwise | bitwise | bitwise | 1.0939-1.0945 | 1.0909-1.0947 | 4.1845-4.2453 | 4.1520-4.1911 | 0.997 | 0.992 |
| D512-L384-dir2 | bitwise | bitwise | bitwise | bitwise | 1.0885-1.0957 | 1.0906-1.1004 | 4.2232-4.2333 | 4.1867-4.2358 | 1.002 | 0.991 |
| D512-L768-dir0 | bitwise | bitwise | bitwise | 5.2e-07 | 7.8071-7.8679 | 7.8115-7.8172 | 28.3355-28.4318 | 28.4933-28.6154 | 1.001 | 1.006 |
| D512-L768-dir1 | bitwise | bitwise | bitwise | bitwise | 4.4283-4.4957 | 4.4627-4.4712 | 17.1867-17.3239 | 17.1603-17.2261 | 1.008 | 0.998 |
| D512-L768-dir2 | bitwise | bitwise | bitwise | bitwise | 4.4514-4.4666 | 4.4611-4.4779 | 17.2295-17.3044 | 17.2251-17.4109 | 1.002 | 1.000 |

The first D128 build computed the per-sample dropout-scale base inside the shared B1 build: B1 went from 180 to 189 us at L384
(per-kernel torch profiler, `prof_k.py`) and the bidirectional L384 training step +1.2 %. The batched code is now a separate build
(`B1_BATCHED=1`, selected when M > L^2): B1 180.10 / 179.89 us (worktree / main), `save_k3` 87.3 / 89.6, B7 319.3 / 317.9.

## Timing (CUDA graph), one call vs B calls vs cuEquivariance vs the PyTorch statements

Module (`BidirectionalTriangleMultiplication(D)` / `TriangleMultiplication(D, outgoing=True)`, hidden D), FP32 parameters, bf16 pair,
a token mask (last L/8 masked), training = forward + backward of x and every parameter (row dropout on). B calls = B one-sample calls
of the same module in one graph; at the preserved single-sample cells (`_legacy_native_faster`: D256+ training at L384 / 768,
bidirectional inference at (D384, L768) and (D512, L384), single-direction D256 inference at L != 384) they take the preserved kernels. cuEquivariance and PyTorch under bf16 autocast. OOM = not in 80 GB (graph pool included).

| D | module | L | B | mode | one call (ms) | B calls (ms) | cuEquivariance (ms) | PyTorch (ms) | × vs B calls | × vs cuEquivariance | × vs PyTorch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | bidirectional | 128 | 4 | inference | 0.121 | 0.160 | 0.326 | 1.125 | 1.33 | 2.70 | 9.31 |
| 128 | bidirectional | 128 | 4 | training | 0.499 | 0.794 | 1.467 | 3.250 | 1.59 | 2.94 | 6.52 |
| 128 | bidirectional | 128 | 48 | inference | 1.245 | 1.901 | 4.440 | 12.847 | 1.53 | 3.57 | 10.32 |
| 128 | bidirectional | 128 | 48 | training | 4.819 | 9.462 | 17.024 | 31.454 | 1.96 | 3.53 | 6.53 |
| 128 | bidirectional | 384 | 4 | inference | 0.877 | 0.948 | 3.042 | 9.920 | 1.08 | 3.47 | 11.31 |
| 128 | bidirectional | 384 | 4 | training | 3.539 | 3.806 | 12.731 | 24.238 | 1.08 | 3.60 | 6.85 |
| 128 | bidirectional | 384 | 48 | inference | 10.591 | 11.301 | 40.099 | 118.813 | 1.07 | 3.79 | 11.22 |
| 128 | bidirectional | 384 | 48 | training | 42.308 | 45.271 | 156.420 | OOM | 1.07 | 3.70 | - |
| 128 | bidirectional | 768 | 4 | inference | 3.688 | 3.738 | 13.523 | 67.234 | 1.01 | 3.67 | 18.23 |
| 128 | bidirectional | 768 | 4 | training | 14.125 | 14.448 | 49.139 | 137.069 | 1.02 | 3.48 | 9.70 |
| 128 | bidirectional | 768 | 48 | inference | 43.028 | 45.213 | OOM | OOM | 1.05 | - | - |
| 128 | bidirectional | 768 | 48 | training | 214.840 | 171.330 | OOM | OOM | 0.80 | - | - |
| 128 | outgoing | 128 | 4 | inference | 0.072 | 0.117 | 0.145 | 0.736 | 1.61 | 2.00 | 10.17 |
| 128 | outgoing | 128 | 4 | training | 0.399 | 0.672 | 0.864 | 2.707 | 1.68 | 2.17 | 6.78 |
| 128 | outgoing | 128 | 48 | inference | 0.673 | 1.384 | 1.417 | 8.513 | 2.06 | 2.11 | 12.66 |
| 128 | outgoing | 128 | 48 | training | 3.533 | 7.946 | 7.796 | 25.623 | 2.25 | 2.21 | 7.25 |
| 128 | outgoing | 384 | 4 | inference | 0.530 | 0.566 | 1.078 | 6.540 | 1.07 | 2.03 | 12.33 |
| 128 | outgoing | 384 | 4 | training | 2.680 | 3.040 | 5.973 | 19.926 | 1.13 | 2.23 | 7.44 |
| 128 | outgoing | 384 | 48 | inference | 6.261 | 6.838 | 12.572 | 77.976 | 1.09 | 2.01 | 12.45 |
| 128 | outgoing | 384 | 48 | training | 31.297 | 36.435 | 68.606 | 235.521 | 1.16 | 2.19 | 7.53 |
| 128 | outgoing | 768 | 4 | inference | 2.044 | 2.179 | 4.268 | 40.081 | 1.07 | 2.09 | 19.61 |
| 128 | outgoing | 768 | 4 | training | 10.549 | 10.939 | 23.504 | 141.556 | 1.04 | 2.23 | 13.42 |
| 128 | outgoing | 768 | 48 | inference | 25.602 | 26.198 | 51.108 | OOM | 1.02 | 2.00 | - |
| 128 | outgoing | 768 | 48 | training | 126.730 | 131.450 | OOM | OOM | 1.04 | - | - |
| 256 | bidirectional | 128 | 4 | inference | 0.272 | 0.315 | 0.791 | 2.051 | 1.16 | 2.91 | 7.54 |
| 256 | bidirectional | 128 | 4 | training | 1.370 | 1.755 | 3.151 | 5.717 | 1.28 | 2.30 | 4.17 |
| 256 | bidirectional | 128 | 48 | inference | 2.838 | 3.761 | 9.279 | 23.522 | 1.33 | 3.27 | 8.29 |
| 256 | bidirectional | 128 | 48 | training | 13.955 | 20.920 | 35.675 | 60.424 | 1.50 | 2.56 | 4.33 |
| 256 | bidirectional | 384 | 4 | inference | 2.144 | 2.289 | 6.976 | 18.415 | 1.07 | 3.25 | 8.59 |
| 256 | bidirectional | 384 | 4 | training | 10.434 | 10.414 | 28.203 | 46.786 | 1.00 | 2.70 | 4.48 |
| 256 | bidirectional | 384 | 48 | inference | 25.307 | 27.218 | 84.739 | OOM | 1.08 | 3.35 | - |
| 256 | bidirectional | 384 | 48 | training | 125.454 | 125.003 | OOM | OOM | 1.00 | - | - |
| 256 | bidirectional | 768 | 4 | inference | 9.158 | 9.352 | 28.276 | 132.712 | 1.02 | 3.09 | 14.49 |
| 256 | bidirectional | 768 | 4 | training | 41.501 | 41.456 | 111.018 | 276.376 | 1.00 | 2.68 | 6.66 |
| 256 | bidirectional | 768 | 48 | inference | 108.135 | 110.995 | OOM | OOM | 1.03 | - | - |
| 256 | bidirectional | 768 | 48 | training | 659.791 | 499.952 | OOM | OOM | 0.76 | - | - |
| 256 | outgoing | 128 | 4 | inference | 0.166 | 0.233 | 0.581 | 1.346 | 1.40 | 3.49 | 8.09 |
| 256 | outgoing | 128 | 4 | training | 0.911 | 1.201 | 2.173 | 4.748 | 1.32 | 2.39 | 5.21 |
| 256 | outgoing | 128 | 48 | inference | 1.705 | 2.767 | 7.348 | 15.183 | 1.62 | 4.31 | 8.91 |
| 256 | outgoing | 128 | 48 | training | 8.899 | 14.300 | 23.482 | 48.752 | 1.61 | 2.64 | 5.48 |
| 256 | outgoing | 384 | 4 | inference | 1.300 | 1.389 | 5.240 | 11.750 | 1.07 | 4.03 | 9.04 |
| 256 | outgoing | 384 | 4 | training | 6.717 | 6.926 | 17.557 | 38.394 | 1.03 | 2.61 | 5.72 |
| 256 | outgoing | 384 | 48 | inference | 15.238 | 16.497 | 62.233 | 141.080 | 1.08 | 4.08 | 9.26 |
| 256 | outgoing | 384 | 48 | training | 79.112 | 83.059 | 209.411 | OOM | 1.05 | 2.65 | - |
| 256 | outgoing | 768 | 4 | inference | 5.359 | 5.462 | 20.854 | 76.236 | 1.02 | 3.89 | 14.22 |
| 256 | outgoing | 768 | 4 | training | 26.924 | 26.932 | 70.697 | 284.983 | 1.00 | 2.63 | 10.58 |
| 256 | outgoing | 768 | 48 | inference | 63.228 | 64.164 | OOM | OOM | 1.01 | - | - |
| 256 | outgoing | 768 | 48 | training | 404.797 | 322.384 | OOM | OOM | 0.80 | - | - |
| 384 | bidirectional | 128 | 4 | inference | 0.497 | 0.582 | 1.629 | 3.072 | 1.17 | 3.28 | 6.18 |
| 384 | bidirectional | 128 | 4 | training | 2.278 | 2.816 | 5.569 | 8.294 | 1.24 | 2.44 | 3.64 |
| 384 | bidirectional | 128 | 48 | inference | 5.484 | 6.963 | 19.203 | 35.411 | 1.27 | 3.50 | 6.46 |
| 384 | bidirectional | 128 | 48 | training | 25.490 | 33.793 | 64.091 | 91.374 | 1.33 | 2.51 | 3.58 |
| 384 | bidirectional | 384 | 4 | inference | 4.118 | 4.306 | 14.418 | 27.799 | 1.05 | 3.50 | 6.75 |
| 384 | bidirectional | 384 | 4 | training | 18.132 | 18.209 | 49.103 | 70.991 | 1.00 | 2.71 | 3.92 |
| 384 | bidirectional | 384 | 48 | inference | 48.893 | 51.730 | OOM | OOM | 1.06 | - | - |
| 384 | bidirectional | 384 | 48 | training | 286.925 | 217.878 | OOM | OOM | 0.76 | - | - |
| 384 | bidirectional | 768 | 4 | inference | 17.439 | 17.311 | 58.946 | 206.705 | 0.99 | 3.38 | 11.85 |
| 384 | bidirectional | 768 | 4 | training | 71.131 | 71.153 | 200.038 | OOM | 1.00 | 2.81 | - |
| 384 | bidirectional | 768 | 48 | inference | 207.065 | 210.314 | OOM | OOM | 1.02 | - | - |
| 384 | bidirectional | 768 | 48 | training | OOM | 863.148 | OOM | OOM | - | - | - |
| 384 | outgoing | 128 | 4 | inference | 0.294 | 0.332 | 1.304 | 1.922 | 1.13 | 4.44 | 6.55 |
| 384 | outgoing | 128 | 4 | training | 1.417 | 1.825 | 3.981 | 6.724 | 1.29 | 2.81 | 4.75 |
| 384 | outgoing | 128 | 48 | inference | 3.140 | 3.925 | 15.434 | 21.933 | 1.25 | 4.92 | 6.99 |
| 384 | outgoing | 128 | 48 | training | 14.809 | 21.903 | 45.074 | 73.017 | 1.48 | 3.04 | 4.93 |
| 384 | outgoing | 384 | 4 | inference | 2.368 | 2.507 | 11.615 | 16.980 | 1.06 | 4.90 | 7.17 |
| 384 | outgoing | 384 | 4 | training | 11.121 | 11.050 | 33.897 | 57.546 | 0.99 | 3.05 | 5.17 |
| 384 | outgoing | 384 | 48 | inference | 27.983 | 29.978 | 139.530 | OOM | 1.07 | 4.99 | - |
| 384 | outgoing | 384 | 48 | training | 132.708 | 132.875 | OOM | OOM | 1.00 | - | - |
| 384 | outgoing | 768 | 4 | inference | 9.772 | 9.917 | 46.583 | 113.581 | 1.01 | 4.77 | 11.62 |
| 384 | outgoing | 768 | 4 | training | 43.661 | 43.658 | 136.007 | 442.309 | 1.00 | 3.12 | 10.13 |
| 384 | outgoing | 768 | 48 | inference | 117.218 | 117.494 | OOM | OOM | 1.00 | - | - |
| 384 | outgoing | 768 | 48 | training | OOM | 525.124 | OOM | OOM | - | - | - |
| 512 | bidirectional | 128 | 4 | inference | 0.885 | 1.167 | 2.723 | 4.220 | 1.32 | 3.08 | 4.77 |
| 512 | bidirectional | 128 | 4 | training | 3.529 | 4.371 | 8.544 | 11.006 | 1.24 | 2.42 | 3.12 |
| 512 | bidirectional | 128 | 48 | inference | 9.350 | 14.103 | 32.231 | 50.637 | 1.51 | 3.45 | 5.42 |
| 512 | bidirectional | 128 | 48 | training | 38.770 | 52.381 | 98.744 | 125.761 | 1.35 | 2.55 | 3.24 |
| 512 | bidirectional | 384 | 4 | inference | 7.075 | 7.390 | 24.191 | 39.047 | 1.04 | 3.42 | 5.52 |
| 512 | bidirectional | 384 | 4 | training | 27.092 | 27.058 | 75.418 | 97.787 | 1.00 | 2.78 | 3.61 |
| 512 | bidirectional | 384 | 48 | inference | 83.573 | 88.792 | OOM | OOM | 1.06 | - | - |
| 512 | bidirectional | 384 | 48 | training | 443.453 | 323.598 | OOM | OOM | 0.73 | - | - |
| 512 | bidirectional | 768 | 4 | inference | 29.400 | 29.536 | 98.713 | 273.413 | 1.00 | 3.36 | 9.30 |
| 512 | bidirectional | 768 | 4 | training | 104.748 | 104.221 | OOM | OOM | 0.99 | - | - |
| 512 | outgoing | 128 | 4 | inference | 0.515 | 0.649 | 2.317 | 2.487 | 1.26 | 4.50 | 4.83 |
| 512 | outgoing | 128 | 4 | training | 2.149 | 2.704 | 6.379 | 8.586 | 1.26 | 2.97 | 4.00 |
| 512 | outgoing | 128 | 48 | inference | 5.331 | 7.826 | 27.556 | 28.965 | 1.47 | 5.17 | 5.43 |
| 512 | outgoing | 128 | 48 | training | 22.736 | 31.751 | 73.332 | 95.408 | 1.40 | 3.23 | 4.20 |
| 512 | outgoing | 384 | 4 | inference | 4.038 | 4.336 | 20.681 | 22.422 | 1.07 | 5.12 | 5.55 |
| 512 | outgoing | 384 | 4 | training | 16.209 | 16.142 | 55.183 | 76.512 | 1.00 | 3.40 | 4.72 |
| 512 | outgoing | 384 | 48 | inference | 47.526 | 52.075 | OOM | OOM | 1.10 | - | - |
| 512 | outgoing | 384 | 48 | training | 258.167 | 193.311 | OOM | OOM | 0.75 | - | - |
| 512 | outgoing | 768 | 4 | inference | 16.423 | 16.708 | 82.860 | 150.889 | 1.02 | 5.05 | 9.19 |
| 512 | outgoing | 768 | 4 | training | 64.828 | 64.706 | 220.907 | OOM | 1.00 | 3.41 | - |

## Large B against the FP32 statement

`sweep_ref.py`: B in {16, 32, 48} x L in {128, 384} x D in {128, 256, 384, 512} x three directions x inference / training (144 cells),
one call against B one-sample calls on the same path and against the FP32 PyTorch statement per sample (weight gradients summed
over the samples, exact for the batch; TF32 off). 144 / 144 pass: y and dx bitwise equal to the one-sample calls in every cell;
against FP32 y <= 3.0e-3, dx <= 4.7e-3, weight gradients <= 7.8e-3 relative; batch vs the sum of one-sample weight gradients
<= 8.6e-4. Training L384 at D256+ with B 32 / 48 ran chunked (and checkpointed) by `_plan`. At L768 (training B 4 / 8 / 16 / 32 / 48,
bidirectional and outgoing; inference B 16 / 48) every cell that fits in 80 GB next to the harness's own x / dy / y / gradients
passes with the same figures, covering one launch, plain chunks and checkpointed chunks:

| cell | chunk | checkpointed | peak GB |
|---|---|---|---|
| infer D128 L768 B16 dir0 | 16 | True | 20.3 |
| infer D128 L768 B16 dir1 | 16 | True | 13.5 |
| infer D128 L768 B48 dir0 | 48 | True | 60.8 |
| infer D128 L768 B48 dir1 | 48 | True | 40.5 |
| infer D256 L768 B16 dir0 | 16 | True | 40.6 |
| infer D256 L768 B16 dir1 | 16 | True | 27.1 |
| infer D256 L768 B48 dir0 | 15 | True | 60.9 |
| infer D256 L768 B48 dir1 | 22 | True | 54.1 |
| infer D384 L768 B16 dir0 | 16 | True | 60.9 |
| infer D384 L768 B16 dir1 | 16 | True | 40.6 |
| infer D512 L768 B16 dir0 | 9 | True | 54.1 |
| infer D512 L768 B16 dir1 | 16 | True | 54.1 |
| train D128 L384 B32 dir0 | 32 | False | 21.5 |
| train D128 L384 B48 dir0 | 48 | False | 32.2 |
| train D128 L768 B16 dir0 | 16 | False | 42.9 |
| train D128 L768 B16 dir1 | 16 | False | 31.6 |
| train D128 L768 B32 dir0 | 11 | False | 61.0 |
| train D128 L768 B48 dir0 | 9 | True | 49.7 |
| train D128 L768 B4 dir0 | 4 | False | 10.8 |
| train D128 L768 B4 dir1 | 4 | False | 8.0 |
| train D128 L768 B8 dir0 | 8 | False | 21.5 |
| train D128 L768 B8 dir1 | 8 | False | 15.9 |
| train D256 L384 B32 dir0 | 27 | False | 48.6 |
| train D256 L384 B48 dir0 | 15 | False | 56.8 |
| train D256 L768 B16 dir0 | 4 | True | 51.9 |
| train D256 L768 B16 dir1 | 3 | False | 47.5 |
| train D256 L768 B32 dir0 | 1 | True | 54.1 |
| train D256 L768 B4 dir0 | 4 | False | 35.0 |
| train D256 L768 B4 dir1 | 4 | False | 23.8 |
| train D256 L768 B8 dir0 | 6 | False | 48.6 |
| train D256 L768 B8 dir1 | 8 | False | 47.5 |
| train D384 L384 B32 dir0 | 10 | False | 56.7 |
| train D384 L384 B48 dir0 | 10 | True | 52.9 |
| train D384 L768 B16 dir0 | 1 | True | 44.8 |
| train D384 L768 B16 dir1 | 2 | True | 47.4 |
| train D384 L768 B4 dir0 | 4 | False | 52.5 |
| train D384 L768 B4 dir1 | 4 | False | 35.6 |
| train D384 L768 B8 dir0 | 1 | False | 48.7 |
| train D384 L768 B8 dir1 | 6 | False | 50.8 |
| train D512 L384 B32 dir0 | 2 | False | 63.8 |
| train D512 L384 B48 dir0 | 5 | True | 52.2 |
| train D512 L768 B16 dir0 | 1 | True | 59.8 |
| train D512 L768 B16 dir1 | 1 | True | 54.1 |
| train D512 L768 B4 dir0 | 3 | False | 49.7 |
| train D512 L768 B4 dir1 | 4 | False | 47.4 |
| train D512 L768 B8 dir0 | 2 | True | 51.9 |
| train D512 L768 B8 dir1 | 1 | False | 51.4 |

- OOM in the harness: train D=384 L=768 B=32 dir=0 OOM
- OOM in the harness: train D=384 L=768 B=48 dir=0 OOM
- OOM in the harness: train D=256 L=768 B=48 dir=0 OOM
- OOM in the harness: infer D=384 L=768 B=48 dir=0 OOM
- OOM in the harness: infer D=384 L=768 B=48 dir=1 OOM
- OOM in the harness: train D=512 L=768 B=32 dir=0 OOM
- OOM in the harness: train D=512 L=768 B=48 dir=0 OOM
- OOM in the harness: infer D=512 L=768 B=48 dir=0 OOM
- OOM in the harness: infer D=512 L=768 B=48 dir=1 OOM

## Chunking (`trimul_h100._plan`, `_Chunked`)

Deterministic from the call's shapes, width, direction, mode and the device's total memory. Coefficients (bytes per pair token per
channel, one launch's peak incl. inputs, from the sweeps; plus 5 %): inference 18.1 bidirectional / 12.1 single; training D64 / D128
40.3 / 32.0, D256+ 62.2 / 42.2. Held from forward to backward (`mem_saved.py`, B = 4 L384, x / dy excluded, y included): D64
14.3 / 8.3, D128 16.1 / 10.0, D256 / 384 18.1 / 12.1, D512 20.1 / 14.0 (the port's backward peak is 56.2 / 36.2: its workspace
dominates). Preserved kernels at B = 1 (`mem_legacy2.py`): held 24.1 (D256, single direction everywhere) / 40.1 (D384 / D512
bidirectional), peak 70-78 bidirectional / 44 single. Budget 0.85 of total memory, after the caller's x / y / dy / dx (8 bytes per
channel).

Training chunks run inside one autograd node (`_Chunked`): y and dx allocated once, each chunk's launch writes its rows in place
through `_h100_runtime.output_buffers` (taken by the path's own final y / dx allocation: `take_output` / `take_input_grad` in the
port, the preserved D256+ paths, D128 `save_k3` / B7 / single output / single B7 and the D64 paths; never while Dynamo traces, where
the path allocates and the rows are copied). Parameter gradients: the chunks' `autograd.grad` results summed in chunk order. Plain
chunks keep their graph to the backward; checkpointed chunks run forward without saves and again with grad in the backward. The
first version (split + `torch.cat`, `torch.utils.checkpoint`) ran out of memory at D256 L768 B48 on the two concatenations.

`big_chunk.py` (harness holds x, dy, y, dx, parameter gradients), against B one-sample calls:

| cell | x / y / dy / dx | plan | peak | y / dx vs one-sample calls | dW vs their sum |
|---|---|---|---|---|---|
| D256 L768 B48 bidirectional | 58.0 GB | chunks of 1, checkpointed | 67.5 GB | bitwise | 0 |
| D384 L768 B26 bidirectional | 47.1 GB | chunks of 1, checkpointed | 63.3 GB | bitwise | 5.3e-7 |
| D512 L768 B20 bidirectional | 48.3 GB | chunks of 1, checkpointed | 70.0 GB | bitwise | 4.9e-7 |
| D256 L768 B48 outgoing (port) | 58.0 GB | chunks of 2, checkpointed | 68.4 GB | bitwise | 6.5e-5 |
| D512 L768 B20 incoming (port) | 48.3 GB | chunks of 2, checkpointed | 69.0 GB | bitwise | 2.8e-4 |
| D128 L768 B48 bidirectional | 29.0 GB | chunks of 16, checkpointed | 65.4 GB | bitwise | 4.5e-4 |

(device 85.0 GB; "(port)": the one-sample calls forced onto the port as well, the bidirectional cells run their chunks of 1 on the
preserved kernels, as the one-sample calls do.) Beyond these sizes the bound is x / y / dy / dx themselves: 8 B L^2 D bytes.

Tests: `test_inference_chunks_a_batch_one_launch_cannot_take` (D512 L768 B48 bidirectional: launches of 2, bitwise vs 48 one-sample
calls, peak under the device; D128 L128 B64 > max_batch), `test_training_chunks_a_batch_one_launch_cannot_take` (D128 L768 B48
bidirectional checkpointed chunks; D256 L768 B24 outgoing checkpointed), `test_chunked_training_graph_replay` (device memory
patched down so B4 L256 splits: checkpointed and plain chunks, D128 / D256 / D512, CUDA-graph capture and replay with live inputs).

## Routing D256+ training at the preserved cells

At `_legacy_native_faster` training cells (D256+, L384 / 768) the call runs, per (D, L, direction) and batch size, whichever was
faster: one launch / chunks of the port, or one launch per sample of the preserved kernels writing into the batch's y / dx
(`PER_SAMPLE_LEGACY_TRAINING`: the largest B routed). CUDA graph, one call / B calls (`bench_route.py`; "B calls" = per-sample
`autograd.grad` without summing the parameter gradients, "accumulating" = per-sample `backward()`):

| D | module | L | B = 4: port | routed | chosen vs B calls | chosen vs accumulating B calls | B = 16 port / routed | B = 48 port / routed |
|---|---|---|---|---|---|---|---|---|
| 256 | bidirectional | 384 | 0.992 | 0.998 | 0.998 | 1.033 | 0.981 / 0.997 | 0.978 / 0.997 |
| 256 | outgoing | 384 | 1.030 | 0.993 | 1.031 | 1.156 | 1.049 / 0.989 | 1.048 / 0.990 |
| 256 | bidirectional | 768 | 0.944 | 0.998 | 0.999 | 1.073 | 1.002 / 0.761 | 0.759 / 0.758 |
| 256 | outgoing | 768 | 0.995 | 1.000 | 1.000 | 1.046 | 0.983 / 0.731 | 0.796 / 0.731 |
| 384 | bidirectional | 384 | 0.955 | 1.000 | 1.004 | 1.032 | 1.029 / 0.996 | OOM / 0.730 |
| 384 | outgoing | 384 | 0.994 | 0.995 | 0.994 | 1.099 | 1.013 / 0.996 | 0.999 / 0.728 |
| 384 | bidirectional | 768 | 0.976 | 0.999 | 1.000 | 1.024 | 0.719 / 0.729 | OOM / OOM |
| 384 | outgoing | 768 | 0.974 | 1.000 | 1.000 | 1.042 | 0.771 / 0.736 | OOM / OOM |
| 512 | bidirectional | 384 | 0.942 | 1.000 | 0.999 | 1.024 | 0.939 / 0.998 | OOM / OOM |
| 512 | outgoing | 384 | 0.945 | 0.997 | 0.996 | 1.092 | 0.962 / 0.998 | OOM / OOM |
| 512 | bidirectional | 768 | 0.877 | 0.996 | 0.995 | 1.019 | 0.731 / 0.731 | OOM / OOM |
| 512 | outgoing | 768 | 0.939 | 0.999 | 0.998 | 1.087 | OOM / 0.724 | OOM / OOM |

The routed call does the B calls' work and sums the gradients, so it sits at 0.994-1.000x of the non-accumulating B calls and
1.02-1.16x of accumulating ones; where the port is faster (D256 outgoing L384, and one launch of the port at B = 16 for D256 L768
and D384) it is kept. With split + `torch.cat` the routed call was 0.93-0.97x: the concatenation of y and of dx was the whole loss.
The port's checkpointed chunks and the routed per-sample calls (checkpointed at large B) are both ~0.72-0.76x of the per-sample
fwd + bwd loop at B 16 / 48, L768 (and D384 / D512 L384 B48): that loop holds one sample's saved tensors at a time, which a model
running B calls and one backward cannot.

Not done: the preserved single-sample D256+ kernels (`h100_wide_training`, `h100_uni_wide_training`) were not batched; routing to
them per sample reaches parity with B calls, not more.

## Tests

`tests/gpu/sm90/test_h100_trimul_batch_wide_gpu.py` 175 / 175 (one call vs B calls and the FP32 statement at B 2 ... 48, every
width and direction, inference and training; bitwise run to run at D128, 1e-4 at D256+; CUDA-graph replay per width and direction;
padded lengths; batch limits; chunked inference and training, chunked training under CUDA graphs), `test_h100_trimul_batch_gpu.py`
37 / 37 (D64; its batch-limit test now expects chunking above `MAX_BATCH`), and the existing H100 TriMul tests: b200style 73,
d64_b200gate 22, small_buckets 15, single_h100 8, long_l 99 + 4 skipped, wide_inference 137, uni_wide_inference 5,
uni_wide_training 25, wide_preparation 6, d64_training 23 + 2 skipped, uni_d64_training 30 + 4 skipped, hot_path 3, nograd_dropout 2,
h100_module_wiring 34, h100_apb_training 45. Run in parallel on one GPU some files OOM next to the 60 GB batch-test process; run one
after another they all pass. CPU suite: the four pre-existing registry failures only (`test_layernorm_is_never_bf16` needed its
allow-list entry for the renamed `h100_single.py` mask line); `dev derive --arch sm86`: 1038 rows; ruff clean on the changed files.
