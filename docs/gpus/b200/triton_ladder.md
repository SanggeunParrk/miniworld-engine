# B200: the Triton ladder (hand CUDA < Triton < torch.compile'd PyTorch)

The engine's rule for every op, shape and mode: **time(hand CUDA) < time(Triton) < time(torch.compile'd PyTorch)**. This page measures it on a B200 after tuning the Triton (portable) kernels for this card. It reports; it changes no kernel.

> **Partial run**: 73 of ~1990 rows measured so far; the page is regenerated when the run finishes.

Provenance: engine commit `0fb41e60` (dirty=False), NVIDIA B200 (sm_100a, 1000 W cap), torch 2.13.0+cu129, triton 3.7.1, run 2026-10-31T06:16:28 UTC on node 2 (`DCTN-0831174147-2`), one `ladder.py run` per GPU on GPUs 0,1,3-7 (GPU 2 hosted a stuck foreign process and was not used for timing; the other GPUs carry a yielding filler process).

## How it was measured

* Harness: `benchmarks/runners/ladder.py` (`miniworld-engine ladder run|report`), which drives `benchmarks/runners/bench.py` (module targets; `ladder_ops` for function-level ops). Three rungs per row: `cuda` = `implementations=miniworld` (the engine's own routing on B200: the hand sm_100a kernels where they exist), `portable` = the same module under `settings.force_portable_kernels` (the Triton family), `pytorch` = the module's PyTorch statements under `torch.compile` (`pt_c`), also timed as eager statements replayed in a CUDA graph (`pt_eg`) and plain eager (`pt_e`).
* Every number is the median of 3 `do_bench` medians of a CUDA-graph replay of one step (inference = forward; training = forward + backward), bf16-mixed (fp32 master parameters, bf16 autocast) unless the dtype column says fp32 (TF32). The engine rungs are timed with and without `torch.compile` around them and the faster one counts.
* Verdicts are the ladder's own: a pair passes when ratio = lower rung / upper rung < 1. The PyTorch rung of a pair is the **fastest** PyTorch form (normally `pt_c`), which is at least as strict as comparing against compiled PyTorch only. `n.a.` = a rung has no measurement (it failed, refused, or launched another rung's kernels: no distinct path); a `routing gap` = `miniworld` on B200 ran the PyTorch reference for that call (no CUDA path).
* Synthetic inputs only (no training data). Times in **microseconds**.

## Triton tuning on B200 (what the Triton rung runs)

Before this page the B200 had autotune caches for 12 Triton ops (4 of them stale). They were built with the engine's own builder,
per-op over every registry Triton kernel that has a driver and is developed (146 ops; the 13 `developed=no` rows are A/B
alternatives with no build unit):

```
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 miniworld-engine build all --kernels portable --gpus all --units-per-gpu 2 --shards <dir>
```

3826 (op, shape, width) units on 7 GPUs, 2 h 10 min: 3768 ok, 26 failed, merged into
`src/miniworld_engine/autotune/data/<op>/NVIDIA B200 (sm100).json` (146 Triton caches + 2 dispatch caches;
`miniworld-engine dev cache-status --gpu B200`: 146 OK, 0 stale). The failed units:

* OuterProductMean Triton at a 16-wide pair (18 units): `ValueError: no config tiles CZ=16, CH=32` -- the default grid has no
  tile for that width (not a MiniWorld / FoldForge width).
* **MSAPairWeightedAveraging Triton backward at d_pair 384** (8 units: `bwd_gate_gemm`, `bwd_layernorm_gemm_dw`,
  `bwd_layernorm_gemm_dx_dlnw`, `bwd_layernorm_gemm_softmax`, L384 / L768): `CUDA error: misaligned address` on sm_100
  (`DZP=512, HP=16`). A portable-kernel bug at that width (OpenDDE's pair width); not fixed here (no kernel changes in this task).

Nine boolean key flags of OPM / PWA / MPNN edge-tail kernels had been driven on one side only (masks, residual, divide-after,
dropout keep); their drivers now call both sides (commit 2024760a) and the other side is built after the ladder run.

## Summary

| rows | OK (ladder verdict) | CUDA >= Triton | Triton >= PyTorch | CUDA >= PyTorch | no CUDA path (routing gap) | other rung missing |
|---|---|---|---|---|---|---|
| 73 | 48 | 2 | 23 | 6 | 1 | 0 |

## Violations (sorted by production weight, then ratio)

weight 3 = a shape MiniWorld / FoldForge runs, 2 = served, 1 = exotic. `ratio` = slower-than-allowed rung / the rung it must beat.

| # | weight | op | mode | dtype | shape | L | pair | ratio | CUDA us | Triton us | PyTorch compiled us | PyTorch best us (form) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | Transition | training | bf16 | D128 | 1024 | Triton >= PyTorch | 1.16 | 1905 | 4249 | 3672 | 3672 (pt_compiled) |  |
| 2 | 3 | Transition | training | bf16 | D128 | 768 | Triton >= PyTorch | 1.15 | 1049 | 2430 | 2118 | 2118 (pt_compiled) |  |
| 3 | 3 | Transition | training | bf16 | D128 | 384 | Triton >= PyTorch | 1.11 | 270.4 | 662.4 | 597.9 | 597.9 (pt_compiled) |  |
| 4 | 2 | TriMul bidirectional hidden!=D | inference | bf16 | D64 hidden128 | 384 | CUDA >= Triton | 2.61 | 603.7 | 231.5 | 661.6 | 661.6 (pt_compiled) |  |
| 5 | 2 | Transition | training | bf16 | D768 | 384 | CUDA >= PyTorch | 1.38 | 8763 | 7929 | 6372 | 6372 (pt_compiled) |  |
| 6 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 128 | Triton >= PyTorch | 1.29 | 81.9 | 92.1 | 71.7 | 71.7 (pt_compiled) |  |
| 7 | 2 | Transition | training | bf16 | D768 | 384 | Triton >= PyTorch | 1.24 | 8763 | 7929 | 6372 | 6372 (pt_compiled) |  |
| 8 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 768 | Triton >= PyTorch | 1.22 | 34.8 | 45.1 | 36.8 | 36.8 (pt_compiled) |  |
| 9 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 768 | Triton >= PyTorch | 1.21 | 104.4 | 118.8 | 98.3 | 98.3 (pt_compiled) |  |
| 10 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 128 | Triton >= PyTorch | 1.20 | 32.7 | 36.8 | 30.7 | 30.7 (pt_compiled) |  |
| 11 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 384 | Triton >= PyTorch | 1.17 | 88.2 | 98.2 | 84.0 | 84.0 (pt_compiled) |  |
| 12 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 128 | CUDA >= PyTorch | 1.14 | 81.9 | 92.1 | 71.7 | 71.7 (pt_compiled) |  |
| 13 | 2 | Transition | inference | bf16 | D768 | 384 | Triton >= PyTorch | 1.14 | 2142 | 2445 | 2153 | 2153 (pt_compiled) | within noise |
| 14 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 384 | Triton >= PyTorch | 1.12 | 32.7 | 36.7 | 32.7 | 32.7 (pt_compiled) |  |
| 15 | 2 | Transition | inference | fp32 | D128 TF32 | 384 | Triton >= PyTorch | 1.11 | 217.0 | 425.5 | 385.0 | 385.0 (pt_compiled) |  |
| 16 | 2 | Transition | training | bf16 | D768 | 384 | CUDA >= Triton | 1.11 | 8763 | 7929 | 6372 | 6372 (pt_compiled) |  |
| 17 | 2 | Transition | inference | fp32 | D128 TF32 | 768 | Triton >= PyTorch | 1.10 | 792.5 | 1602 | 1454 | 1454 (pt_compiled) |  |
| 18 | 2 | OuterProductMean Chai-1 (8 groups) | training | bf16 | S512 pair256 8x64 | 256 | Triton >= PyTorch | 1.09 | 337.9 | 552.8 | 505.9 | 505.9 (pt_compiled) |  |
| 19 | 2 | Transition | training | fp32 | D128 TF32 | 768 | Triton >= PyTorch | 1.07 | 3387 | 4301 | 4003 | 4003 (pt_compiled) |  |
| 20 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 128 | CUDA >= PyTorch | 1.06 | 32.7 | 36.8 | 30.7 | 30.7 (pt_compiled) |  |
| 21 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 768 | CUDA >= PyTorch | 1.06 | 104.4 | 118.8 | 98.3 | 98.3 (pt_compiled) |  |
| 22 | 2 | OuterProductMean Chai-1 (8 groups) | training | bf16 | S512 pair256 8x64 | 384 | Triton >= PyTorch | 1.05 | 610.3 | 960.5 | 914.4 | 914.4 (pt_compiled) |  |
| 23 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 384 | CUDA >= PyTorch | 1.05 | 88.2 | 98.2 | 84.0 | 84.0 (pt_compiled) |  |
| 24 | 2 | Transition | training | fp32 | D128 TF32 | 384 | Triton >= PyTorch | 1.03 | 913.4 | 1155 | 1118 | 1118 (pt_compiled) |  |
| 25 | 2 | Transition conditioning (single) | inference | fp32 | D384 n2 TF32 | 128 | Triton >= PyTorch | 1.00 | 22.4 | 26.5 | 26.5 | 26.5 (pt_compiled) | within noise |
| 26 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 384 | CUDA >= PyTorch | 1.00 | 32.7 | 36.7 | 32.7 | 32.7 (pt_compiled) |  |
| 27 | 1 | Transition | inference | fp32 | D256 TF32 | 384 | Triton >= PyTorch | 1.39 | 565.3 | 1115 | 802.9 | 802.9 (pt_compiled) |  |
| 28 | 1 | Transition | training | bf16 | D512 | 384 | Triton >= PyTorch | 1.30 | 2839 | 4104 | 3149 | 3149 (pt_compiled) |  |
| 29 | 1 | TriMul incoming | inference | bf16 | D512 | 384 | Triton >= PyTorch | 1.19 | 591.9 | 1658 | 1396 | 1396 (pt_compiled) |  |
| 30 | 1 | TriMul outgoing | inference | bf16 | D512 | 384 | Triton >= PyTorch | 1.16 | 602.0 | 1658 | 1425 | 1425 (pt_compiled) |  |
| 31 | 1 | Transition | inference | bf16 | D512 | 384 | Triton >= PyTorch | 1.13 | 792.4 | 1253 | 1108 | 1108 (pt_compiled) |  |

## Ops missing a level on B200

### No hand-CUDA path (`miniworld` on B200 runs the PyTorch reference)

| op | mode | dtype | shape | L | miniworld (= PyTorch) us | Triton us | Triton speed-up lost |
|---|---|---|---|---|---|---|---|
| TriMul bidirectional hidden!=D | training | bf16 | D64 hidden128 | 384 | 1871 | 782.2 | 2.39x |

### Other rows where a rung has no measurement

| op | mode | dtype | shape | L | CUDA us | Triton us | PyTorch compiled us | why |
|---|---|---|---|---|---|---|---|---|

## Per module

`c/t` = CUDA / Triton, `t/p` = Triton / best PyTorch. Bold verdicts are violations.

### TriMul outgoing

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 | 128 | 26.5 | 41.0 | 75.7 | 165.7 | 0.65 | 0.54 | OK |
| inference | bf16 | D128 | 384 | 85.9 | 188.4 | 391.1 | 1048 | 0.46 | 0.48 | OK |
| inference | bf16 | D128 | 768 | 325.8 | 679.8 | 1671 | 4507 | 0.48 | 0.41 | OK |
| inference | bf16 | D128 | 1024 | 637.8 | 1215 | 6331 | 12630 | 0.52 | 0.19 | OK |
| inference | bf16 | D256 | 384 | 236.4 | 491.4 | 737.0 | 1834 | 0.48 | 0.67 | OK |
| inference | bf16 | D256 | 768 | 954.2 | 1822 | 3170 | 8610 | 0.52 | 0.57 | OK |
| inference | bf16 | D512 | 384 | 602.0 | 1658 | 1425 | 3467 | 0.36 | 1.16 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | D512 | 768 | 2509 | 6608 | 7231 | 17281 | 0.38 | 0.91 | OK |
| inference | bf16 | D64 | 128 | 20.4 | 28.6 | - | 126.9 | 0.71 | 0.23 | OK |
| inference | bf16 | D64 | 384 | 51.2 | 79.8 | 208.8 | 676.8 | 0.64 | 0.38 | OK |
| training | bf16 | D128 | 128 | 109.6 | 150.6 | 257.1 | 668.6 | 0.73 | 0.59 | OK |
| training | bf16 | D128 | 384 | 333.8 | 605.2 | 1234 | 3248 | 0.55 | 0.49 | OK |
| training | bf16 | D128 | 768 | 1230 | 2186 | 5061 | 13886 | 0.56 | 0.43 | OK |
| training | bf16 | D128 | 1024 | 2329 | 3957 | 19047 | 50221 | 0.59 | 0.21 | OK |
| training | bf16 | D256 | 384 | 1041 | 1278 | 2264 | 5846 | 0.82 | 0.56 | OK |
| training | bf16 | D256 | 768 | 3941 | 5186 | 10202 | 28996 | 0.76 | 0.51 | OK |
| training | bf16 | D512 | 384 | 2491 | 3415 | 4586 | 11136 | 0.73 | 0.74 | OK |
| training | bf16 | D512 | 768 | 9918 | 14553 | 23304 | 60362 | 0.68 | 0.62 | OK |
| training | bf16 | D64 | 128 | 70.7 | 111.6 | 189.5 | 490.6 | 0.63 | 0.59 | OK |
| training | bf16 | D64 | 384 | 191.6 | 336.9 | 705.5 | 1977 | 0.57 | 0.48 | OK |

### TriMul incoming

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 | 384 | 86.1 | 190.4 | 391.2 | 1049 | 0.45 | 0.49 | OK |
| inference | bf16 | D128 | 768 | 327.6 | 681.8 | 1714 | 4538 | 0.48 | 0.40 | OK |
| inference | bf16 | D512 | 384 | 591.9 | 1658 | 1396 | 3470 | 0.36 | 1.19 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D128 | 384 | 336.7 | 603.1 | 1235 | 3250 | 0.56 | 0.49 | OK |
| training | bf16 | D128 | 768 | 1266 | 2194 | 5039 | 13897 | 0.58 | 0.44 | OK |
| training | bf16 | D512 | 384 | 2427 | 3388 | 4545 | 11135 | 0.72 | 0.75 | OK |

### TriMul outgoing hidden!=D

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 hidden256 | 384 | 126.7 | 325.5 | 678.1 | - | 0.39 | 0.48 | OK |
| inference | bf16 | D64 hidden128 | 128 | 22.5 | 34.7 | 71.6 | 157.7 | 0.65 | 0.48 | OK |
| inference | bf16 | D64 hidden128 | 384 | 69.7 | 137.2 | 325.5 | 965.6 | 0.51 | 0.42 | OK |
| training | bf16 | D64 hidden128 | 128 | 88.1 | 127.1 | 239.6 | 620.5 | 0.69 | 0.53 | OK |
| training | bf16 | D64 hidden128 | 384 | 251.0 | 487.4 | 1105 | 2966 | 0.52 | 0.44 | OK |

### TriMul bidirectional

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 | 128 | 32.9 | 59.3 | 112.6 | 237.8 | 0.55 | 0.53 | OK |
| inference | bf16 | D128 | 384 | 133.0 | 331.6 | 698.4 | 1697 | 0.40 | 0.47 | OK |
| inference | bf16 | D128 | 768 | 565.7 | 1211 | 2983 | 7087 | 0.47 | 0.41 | OK |
| inference | bf16 | D256 | 384 | 362.5 | 882.5 | 1238 | 3099 | 0.41 | 0.71 | OK |
| inference | bf16 | D256 | 768 | 1549 | 3400 | 6442 | 14546 | 0.46 | 0.53 | OK |
| training | bf16 | D128 | 128 | 152.5 | 187.5 | 375.9 | 812.0 | 0.81 | 0.50 | OK |
| training | bf16 | D128 | 384 | 532.6 | 973.9 | 2076 | 4470 | 0.55 | 0.47 | OK |
| training | bf16 | D128 | 768 | 2016 | 3909 | 9184 | 18533 | 0.52 | 0.43 | OK |
| training | bf16 | D256 | 384 | 1587 | 2084 | 3985 | - | 0.76 | 0.52 | OK |
| training | bf16 | D256 | 768 | 6266 | 9190 | 20659 | - | 0.68 | 0.44 | OK |

### TriMul bidirectional hidden!=D

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D64 hidden128 | 384 | 603.7 | 231.5 | 661.6 | 1606 | 2.61 | 0.35 | **VIOLATION(cuda_vs_portable)** |
| training | bf16 | D64 hidden128 | 384 | 1871 | 782.2 | 1914 | 4151 | 2.39 | 0.41 | no CUDA path |

### Transition

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 | 128 | 16.3 | 36.9 | 38.9 | 83.9 | 0.44 | 0.95 | OK |
| inference | bf16 | D128 | 384 | 55.3 | 200.7 | 207.8 | 568.4 | 0.28 | 0.97 | OK |
| inference | bf16 | D128 | 768 | 200.7 | 722.1 | 782.4 | 2143 | 0.28 | 0.92 | OK |
| inference | bf16 | D128 | 1024 | 352.1 | 1259 | 1378 | 3772 | 0.28 | 0.91 | OK |
| inference | bf16 | D512 | 384 | 792.4 | 1253 | 1108 | 2043 | 0.63 | 1.13 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | D768 | 384 | 2142 | 2445 | 2153 | 3475 | 0.88 | 1.14 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D128 TF32 | 384 | 217.0 | 425.5 | 385.0 | 734.2 | 0.51 | 1.11 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D128 TF32 | 768 | 792.5 | 1602 | 1454 | 2832 | 0.49 | 1.10 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D256 TF32 | 384 | 565.3 | 1115 | 802.9 | - | 0.51 | 1.39 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D128 | 128 | 57.3 | 110.5 | 110.5 | 288.8 | 0.52 | 1.00 | OK |
| training | bf16 | D128 | 384 | 270.4 | 662.4 | 597.9 | 1547 | 0.41 | 1.11 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D128 | 768 | 1049 | 2430 | 2118 | 5717 | 0.43 | 1.15 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D128 | 1024 | 1905 | 4249 | 3672 | 10036 | 0.45 | 1.16 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D512 | 384 | 2839 | 4104 | 3149 | 5951 | 0.69 | 1.30 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D768 | 384 | 8763 | 7929 | 6372 | 10004 | 1.11 | 1.24 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D128 TF32 | 384 | 913.4 | 1155 | 1118 | 2194 | 0.79 | 1.03 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D128 TF32 | 768 | 3387 | 4301 | 4003 | 8218 | 0.79 | 1.07 | **VIOLATION(portable_vs_pytorch)** |

### Transition conditioning (single)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 | D384 n2 TF32 | 128 | 22.4 | 26.5 | 26.5 | 28.7 | 0.85 | 1.00 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D384 n2 TF32 | 384 | 24.4 | 28.6 | - | 30.7 | 0.85 | 0.93 | OK |
| inference | fp32 | D384 n2 TF32 | 768 | 24.4 | 28.6 | - | 32.5 | 0.86 | 0.88 | OK |
| inference | fp32 | D768 n2 TF32 | 128 | 32.7 | 36.8 | 30.7 | 34.8 | 0.89 | 1.20 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D768 n2 TF32 | 384 | 32.7 | 36.7 | 32.7 | 36.9 | 0.89 | 1.12 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D768 n2 TF32 | 768 | 34.8 | 45.1 | 36.8 | 43.0 | 0.77 | 1.22 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D768 n2 TF32 | 128 | 81.9 | 92.1 | 71.7 | - | 0.89 | 1.29 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D768 n2 TF32 | 384 | 88.2 | 98.2 | 84.0 | - | 0.90 | 1.17 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D768 n2 TF32 | 768 | 104.4 | 118.8 | 98.3 | - | 0.88 | 1.21 | **VIOLATION(portable_vs_pytorch)** |

### OuterProductMean Chai-1 (8 groups)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S512 pair256 8x64 | 256 | 77.7 | 118.3 | 169.8 | 687.9 | 0.66 | 0.70 | OK |
| inference | bf16 | S512 pair256 8x64 | 384 | 145.3 | 227.1 | 327.5 | 1308 | 0.64 | 0.69 | OK |
| training | bf16 | S512 pair256 8x64 | 256 | 337.9 | 552.8 | 505.9 | - | 0.61 | 1.09 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | S512 pair256 8x64 | 384 | 610.3 | 960.5 | 914.4 | - | 0.64 | 1.05 | **VIOLATION(portable_vs_pytorch)** |

## Reproduce

```
# tune (GPU 2 excluded); then one ladder process per GPU over a partition of the groups
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 miniworld-engine build all --kernels portable --gpus all --units-per-gpu 2 --shards <dir>
CUDA_VISIBLE_DEVICES=<g> python benchmarks/runners/ladder.py run --group '<regex>' --repeats 3 --out <dir>/gpu<g>
python benchmarks/runners/ladder.py report --out <dir>/gpu0 <dir>/gpu1 ... --csv ladder.csv --md ladder.md
```

