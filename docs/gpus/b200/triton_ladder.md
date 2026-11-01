# B200: the Triton ladder (hand CUDA < Triton < torch.compile'd PyTorch)

The engine's rule for every op, shape and mode: **time(hand CUDA) < time(Triton) < time(torch.compile'd PyTorch)**. This page measures it on a B200 with the Triton (portable) kernels tuned for this card. It reports; it changes no kernel.

Provenance: engine commit `24f896295` (dirty=False), NVIDIA B200 (sm_100a, 1000 W cap), torch 2.13.0+cu129, triton 3.7.1, run 2026-11-01T12:09:57 UTC on node 2 (`DCTN-0831174147-2`), one `ladder.py run` per GPU on GPUs 1-7 with the 3ec96849 run's group partition (GPU 0 was busy with a correctness test; every GPU carries a yielding filler process).

## Findings

* **432 rows** (the same 113 groups x mode x length as the 3ec96849 page). **356 OK, 74 break the chain** somewhere (before: 239 OK,
  158 breaking). By kind: CUDA >= Triton 33 (39 before), Triton >= PyTorch 43 (132), CUDA >= PyTorch 14 (48; this page also checks
  that pair for three-rung rows, the ladder's own verdict only for ops without a Triton rung, where it is 0), no CUDA path 2 (43).
  122 rows that broke before pass now, 6 pass-before rows break now (four standalone LayerNorm pair rows and AdaLN atom bf16
  training L1024 at 1.02, all within a few us; Transition conditioning fp32 D384 inference L768: Triton 28.6 vs compiled 26.7 us).
* What moved them: the six merged B200 work trees (TriMul bidirectional D64 hidden 128 on CUDA; OPM / PWA at MiniWorld's 64-wide pair
  on the sm_100a kernels; token DiT / bias-only DiT at a 256-wide pair, APB 8-head pair256 and the smooth-lDDT launch geometry on CUDA;
  LayerNorm / RMSNorm sub-warp row kernels, token_pair_init, AugmentedAttention atom / fp32 pair256; Transition tg_gemm 2-CTA, row-fused
  Triton and fused fp32 tcgen05; AdaLN / ConditionedTransition tcgen05 and cuBLAS routes) and the full rebuild of every B200 Triton
  cache on this tree (24f896295: 3906 units, 3874 ok, 32 skipped by design).
* The remaining CUDA >= Triton rows are dominated by **AdaLN and ConditionedTransition** (36 of the 45 rows that miss the 0.95 bar):
  on B200 their portable rung and their CUDA rung share the cuBLAS TF32 / bf16 GEMMs and differ only in the row passes, so many rows
  tie at 1.000 (same kernels, same time). The rest are small memory-bound calls: pair_embed_ca B16 / B48, pair_embed_geom B16 L128,
  token_pair_init B16 L128, sampler augment_noise, OPM pair64 inference L768 (0.955) and token DiT with an extra pair bias training L256.
* No CUDA path on B200 is left only for the Protenix-v2 token DiT at a 256-wide pair in bf16 training (the `miniworld` rung runs the
  module composition, which is faster than the Triton rung there).

## Summary

| rows | OK (ladder verdict) | rows breaking the chain | CUDA >= Triton | Triton >= PyTorch | CUDA >= PyTorch | no CUDA path (routing gap) | other rung missing |
|---|---|---|---|---|---|---|---|
| 432 | 356 | 74 | 33 | 43 | 14 | 2 | 0 |

## B200 policy (2026-11-01)

The engine-wide rule stays **time(hand CUDA) < time(Triton) < time(torch.compile'd PyTorch)**, and the tables below report every break of it. On B200 the maintainer applies it as follows:

* **Triton >= PyTorch is tolerated on B200**: it is a Triton tuning / code-generation issue of the portable family on this card, not a routing defect, and does not block a B200 change.
* **The B200 bar is CUDA <= 0.95 x Triton** for every op except standalone LayerNorm / RMSNorm (memory-bound row kernels whose CUDA and Triton forms both run at bandwidth).
* Rows missing the 0.95 bar on this run: **45** (24 of them with CUDA >= Triton). They are the work list:

| # | weight | op | mode | dtype | shape | L | CUDA / Triton | CUDA us | Triton us | PyTorch compiled us |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | pair_embed_ca | inference | bf16 out, fp32 geometry | B16 | 128 | 1.060 | 36.8 | 34.7 | 959.4 |
| 2 | 3 | sampler augment_noise | inference | fp32 | B5 | 12288 | 0.981 | 10.0 | 10.1 | 24.5 |
| 3 | 3 | pair_embed_ca | inference | bf16 out, fp32 geometry | B16 | 256 | 0.980 | 100.2 | 102.3 | 909.2 |
| 4 | 3 | pair_embed_geom | inference | bf16 out, fp32 geometry | B16 | 128 | 0.967 | 118.6 | 122.7 | 3890 |
| 5 | 3 | token_pair_init | inference | fp32 in, fp32 out | B16 | 128 | 0.960 | 49.0 | 51.1 | 442.3 |
| 6 | 3 | OuterProductMean | inference | bf16 | S1024 msa64 pair64 | 768 | 0.955 | 1111 | 1163 | 1560 |
| 7 | 2 | ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 1.082 | 456.5 | 421.9 | 444.4 |
| 8 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | 1.062 | 628.7 | 591.9 | 602.1 |
| 9 | 2 | ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 128 | 1.057 | 497.7 | 471.1 | 473.1 |
| 10 | 2 | AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 1.056 | 38.9 | 36.8 | 39.0 |
| 11 | 2 | ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 1.055 | 553.0 | 524.4 | 534.0 |
| 12 | 2 | AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 1.050 | 172.0 | 163.8 | 167.9 |
| 13 | 2 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 128 | 1.046 | 132.8 | 127.0 | 124.9 |
| 14 | 2 | AdaLN | training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 1.029 | 145.3 | 141.2 | 149.6 |
| 15 | 2 | AdaLN | training | bf16 | d128 cond128 A5/A48 (atoms) | 1024 | 1.021 | 102.3 | 100.2 | 137.0 |
| 16 | 2 | ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 1.018 | 118.8 | 116.6 | 126.9 |
| 17 | 2 | ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 1.017 | 2482 | 2440 | 2538 |
| 18 | 2 | ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 384 | 1.011 | 1208 | 1195 | 1234 |
| 19 | 2 | ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 1.009 | 1330 | 1319 | 1361 |
| 20 | 2 | ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 1.005 | 1460 | 1452 | 1565 |
| 21 | 2 | AdaLN | inference | fp32 | d768 cond384 A5/A48 | 768 | 1.004 | 32.7 | 32.6 | 36.8 |
| 22 | 2 | ConditionedTransition | inference | bf16 | d768 A5/A48 | 384 | 1.003 | 45.1 | 45.0 | 61.4 |
| 23 | 2 | AdaLN | inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 128 | 1.002 | 18.3 | 18.3 | 22.5 |
| 24 | 2 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 384 | 1.001 | 298.9 | 298.8 | 301.0 |
| 25 | 2 | ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 768 | 1.000 | 106.5 | 106.4 | 114.7 |
| 26 | 2 | AdaLN | inference | bf16 | d768 cond384 A5/A48 | 384 | 1.000 | 18.2 | 18.2 | 24.4 |
| 27 | 2 | AdaLN | inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 384 | 1.000 | 22.4 | 22.4 | 26.6 |
| 28 | 2 | AdaLN | inference | bf16 | d128 cond128 A5/A48 (atoms) | 1024 | 1.000 | 14.1 | 14.1 | 20.5 |
| 29 | 2 | pair_embed_ca | inference | bf16 out, fp32 geometry | B48 | 128 | 1.000 | 79.7 | 79.7 | 706.6 |
| 30 | 2 | AdaLN | training | bf16 | d768 cond768 A5/A48 (ESMFold2) | 128 | 0.999 | 102.4 | 102.4 | 125.1 |
| 31 | 2 | AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 0.999 | 26.6 | 26.6 | 28.7 |
| 32 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 768 | 0.995 | 1126 | 1132 | 1139 |
| 33 | 2 | ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 0.994 | 2687 | 2702 | 2898 |
| 34 | 2 | ConditionedTransition | inference | bf16 | d768 A5/A48 | 768 | 0.994 | 63.1 | 63.4 | 75.7 |
| 35 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 1024 | 0.993 | 1455 | 1465 | 1415 |
| 36 | 2 | ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 768 | 0.988 | 2196 | 2223 | 2235 |
| 37 | 2 | AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 0.984 | 382.9 | 389.1 | 393.3 |
| 38 | 2 | pair_embed_ca | inference | bf16 out, fp32 geometry | B48 | 256 | 0.983 | 277.7 | 282.5 | 2637 |
| 39 | 2 | ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 0.975 | 79.8 | 81.8 | 87.9 |
| 40 | 2 | ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 384 | 0.973 | 73.7 | 75.7 | 81.8 |
| 41 | 2 | AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 0.961 | 691.2 | 718.9 | 736.2 |
| 42 | 2 | token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | 0.958 | 304.3 | 317.5 | 364.5 |
| 43 | 2 | ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 128 | 0.958 | 47.0 | 49.1 | 55.2 |
| 44 | 2 | AdaLN | training | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 0.958 | 447.1 | 466.9 | 505.8 |
| 45 | 2 | AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 0.952 | 39.0 | 40.9 | 45.0 |

## Violations (sorted by production weight, then ratio)

weight 3 = a shape MiniWorld / FoldForge runs, 2 = served, 1 = exotic. `ratio` = slower-than-allowed rung / the rung it must beat. A row can break two pairs, so this table has more lines than violating rows.

| # | weight | op | mode | dtype | shape | L | pair | ratio | CUDA us | Triton us | PyTorch compiled us | PyTorch best us (form) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 256 | Triton >= PyTorch | 1.24 | 1080 | 5048 | 4060 | 4060 (pt_compiled) |  |
| 2 | 3 | pair_embed_adaln | training | bf16 out, fp32 geometry | B16 | 384 | Triton >= PyTorch | 1.12 | 3200 | 8104 | 7214 | 7214 (pt_compiled) |  |
| 3 | 3 | LayerNorm (pair) | training | bf16 | x[1,L,L,128] | 768 | CUDA >= Triton | 1.12 | 147.3 | 131.2 | 190.5 | 190.5 (pt_compiled) |  |
| 4 | 3 | LayerNorm (pair) | inference | bf16 | x[1,L,L,128] | 384 | CUDA >= Triton | 1.11 | 20.4 | 18.3 | 22.5 | 22.5 (pt_compiled) |  |
| 5 | 3 | pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 384 | Triton >= PyTorch | 1.08 | 2353 | 9664 | 8959 | 8959 (pt_compiled) |  |
| 6 | 3 | LayerNorm (pair) | inference | bf16 | x[1,L,L,128] | 768 | CUDA >= Triton | 1.08 | 57.2 | 53.1 | 65.5 | 65.5 (pt_compiled) |  |
| 7 | 3 | pair_embed_ca | inference | bf16 out, fp32 geometry | B16 | 128 | CUDA >= Triton | 1.06 | 36.8 | 34.7 | 959.4 | 898.9 (pt_eager_graph) |  |
| 8 | 3 | LayerNorm (pair) | training | bf16 | x[1,L,L,128] | 384 | CUDA >= Triton | 1.04 | 51.0 | 49.0 | 71.7 | 71.7 (pt_compiled) |  |
| 9 | 2 | Transition conditioning (pair) | inference | fp32 | D256 n2 TF32 | 768 | Triton >= PyTorch | 1.33 | 1216 | 2464 | 1851 | 1851 (pt_compiled) |  |
| 10 | 2 | Transition conditioning (pair) | inference | fp32 | D256 n2 TF32 | 384 | Triton >= PyTorch | 1.33 | 321.4 | 638.9 | 482.2 | 482.2 (pt_compiled) |  |
| 11 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 128 | Triton >= PyTorch | 1.29 | 77.7 | 92.0 | 71.6 | 71.6 (pt_compiled) | within noise |
| 12 | 2 | Transition | training | bf16 | D768 | 384 | Triton >= PyTorch | 1.28 | 5994 | 7871 | 6154 | 6154 (pt_compiled) |  |
| 13 | 2 | Transition | training | bf16 | D256 | 768 | Triton >= PyTorch | 1.27 | 3878 | 5687 | 4470 | 4470 (pt_compiled) |  |
| 14 | 2 | Transition conditioning (pair) | inference | fp32 | D256 n2 TF32 | 128 | Triton >= PyTorch | 1.27 | 53.1 | 85.9 | 67.5 | 67.5 (pt_compiled) |  |
| 15 | 2 | Transition | training | bf16 | D384 | 768 | Triton >= PyTorch | 1.27 | 6820 | 10661 | 8406 | 8406 (pt_compiled) |  |
| 16 | 2 | Transition | training | bf16 | D384 | 384 | Triton >= PyTorch | 1.26 | 1738 | 2685 | 2125 | 2125 (pt_compiled) |  |
| 17 | 2 | Transition conditioning (pair) | training | fp32 | D256 n2 TF32 | 384 | Triton >= PyTorch | 1.26 | 1505 | 1756 | 1391 | 1391 (pt_compiled) |  |
| 18 | 2 | Transition conditioning (pair) | training | fp32 | D256 n2 TF32 | 768 | Triton >= PyTorch | 1.24 | 5280 | 6227 | 5035 | 5035 (pt_compiled) |  |
| 19 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 768 | Triton >= PyTorch | 1.22 | 32.9 | 45.1 | 36.8 | 36.8 (pt_compiled) |  |
| 20 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 768 | Triton >= PyTorch | 1.20 | 96.2 | 120.6 | 100.2 | 100.2 (pt_compiled) |  |
| 21 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 128 | Triton >= PyTorch | 1.20 | 30.7 | 36.9 | 30.7 | 30.7 (pt_compiled) |  |
| 22 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 384 | Triton >= PyTorch | 1.17 | 84.0 | 98.1 | 84.0 | 84.0 (pt_compiled) |  |
| 23 | 2 | Transition | training | bf16 | D256 | 384 | Triton >= PyTorch | 1.17 | 995.4 | 1431 | 1227 | 1227 (pt_compiled) |  |
| 24 | 2 | Transition | inference | bf16 | D768 | 384 | Triton >= PyTorch | 1.15 | 1835 | 2442 | 2120 | 2120 (pt_compiled) |  |
| 25 | 2 | Transition conditioning (single) | training | fp32 | D384 n2 TF32 | 128 | Triton >= PyTorch | 1.14 | 57.2 | 65.4 | 57.3 | 57.3 (pt_compiled) | within noise |
| 26 | 2 | Transition conditioning (pair) | training | fp32 | D256 n2 TF32 | 128 | Triton >= PyTorch | 1.13 | 223.1 | 258.0 | 229.2 | 229.2 (pt_compiled) |  |
| 27 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 384 | Triton >= PyTorch | 1.12 | 30.8 | 36.8 | 32.8 | 32.8 (pt_compiled) |  |
| 28 | 2 | LayerNorm (pair) | training | bf16 | x[1,L,L,256] | 384 | CUDA >= Triton | 1.11 | 83.8 | 75.6 | 102.4 | 102.4 (pt_compiled) |  |
| 29 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 128 | CUDA >= PyTorch | 1.09 | 77.7 | 92.0 | 71.6 | 71.6 (pt_compiled) | within noise |
| 30 | 2 | Transition conditioning (pair) | training | fp32 | D256 n2 TF32 | 384 | CUDA >= PyTorch | 1.08 | 1505 | 1756 | 1391 | 1391 (pt_compiled) |  |
| 31 | 2 | ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | CUDA >= Triton | 1.08 | 456.5 | 421.9 | 444.4 | 444.4 (pt_compiled) |  |
| 32 | 2 | Transition | training | fp32 | D128 TF32 | 768 | Triton >= PyTorch | 1.08 | 3335 | 4304 | 3980 | 3980 (pt_compiled) |  |
| 33 | 2 | pair_embed_geom | training | bf16 out, fp32 geometry | B48 | 256 | Triton >= PyTorch | 1.08 | 2955 | 12814 | 11874 | 11874 (pt_compiled) |  |
| 34 | 2 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 768 | Triton >= PyTorch | 1.08 | 485.3 | 565.1 | 524.1 | 524.1 (pt_compiled) |  |
| 35 | 2 | Transition conditioning (single) | inference | fp32 | D384 n2 TF32 | 768 | Triton >= PyTorch | 1.07 | 24.5 | 28.6 | 26.7 | 26.7 (pt_compiled) |  |
| 36 | 2 | pair_embed_geom | training | bf16 out, fp32 geometry | B48 | 128 | Triton >= PyTorch | 1.07 | 825.2 | 3308 | 3089 | 3089 (pt_compiled) |  |
| 37 | 2 | LayerNorm (pair) | inference | bf16 | x[1,L,L,256] | 384 | CUDA >= Triton | 1.07 | 32.7 | 30.7 | 30.7 | 30.7 (pt_compiled) | within noise |
| 38 | 2 | LayerNorm (pair) | inference | bf16 | x[1,L,L,256] | 384 | CUDA >= PyTorch | 1.07 | 32.7 | 30.7 | 30.7 | 30.7 (pt_compiled) | within noise |
| 39 | 2 | AdaLN | inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 768 | Triton >= PyTorch | 1.06 | 28.5 | 32.6 | 30.7 | 30.7 (pt_compiled) |  |
| 40 | 2 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 128 | CUDA >= PyTorch | 1.06 | 132.8 | 127.0 | 124.9 | 124.9 (pt_compiled) | within noise |
| 41 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | CUDA >= Triton | 1.06 | 628.7 | 591.9 | 602.1 | 602.1 (pt_compiled) | within noise |
| 42 | 2 | ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 128 | CUDA >= Triton | 1.06 | 497.7 | 471.1 | 473.1 | 473.1 (pt_compiled) | within noise |
| 43 | 2 | AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | CUDA >= Triton | 1.06 | 38.9 | 36.8 | 39.0 | 39.0 (pt_compiled) |  |
| 44 | 2 | RMSNorm | training | bf16 | x[48,L,768] | 384 | CUDA >= Triton | 1.06 | 38.9 | 36.8 | 59.4 | 59.4 (pt_compiled) |  |
| 45 | 2 | ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA >= Triton | 1.05 | 553.0 | 524.4 | 534.0 | 534.0 (pt_compiled) | within noise |
| 46 | 2 | ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 128 | CUDA >= PyTorch | 1.05 | 497.7 | 471.1 | 473.1 | 473.1 (pt_compiled) | within noise |
| 47 | 2 | AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA >= Triton | 1.05 | 172.0 | 163.8 | 167.9 | 167.9 (pt_compiled) |  |
| 48 | 2 | Transition conditioning (pair) | training | fp32 | D256 n2 TF32 | 768 | CUDA >= PyTorch | 1.05 | 5280 | 6227 | 5035 | 5035 (pt_compiled) |  |
| 49 | 2 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 128 | CUDA >= Triton | 1.05 | 132.8 | 127.0 | 124.9 | 124.9 (pt_compiled) | within noise |
| 50 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | CUDA >= PyTorch | 1.04 | 628.7 | 591.9 | 602.1 | 602.1 (pt_compiled) | within noise |
| 51 | 2 | ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA >= PyTorch | 1.04 | 553.0 | 524.4 | 534.0 | 534.0 (pt_compiled) | within noise |
| 52 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 1024 | Triton >= PyTorch | 1.04 | 1455 | 1465 | 1415 | 1415 (pt_compiled) | within noise |
| 53 | 2 | Transition | training | fp32 | D128 TF32 | 384 | Triton >= PyTorch | 1.03 | 897.1 | 1155 | 1119 | 1119 (pt_compiled) |  |
| 54 | 2 | AdaLN | training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | CUDA >= Triton | 1.03 | 145.3 | 141.2 | 149.6 | 149.6 (pt_compiled) |  |
| 55 | 2 | RMSNorm | inference | bf16 | x[48,L,768] | 384 | CUDA >= Triton | 1.03 | 16.3 | 15.8 | 16.4 | 16.4 (pt_compiled) | within noise |
| 56 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 1024 | CUDA >= PyTorch | 1.03 | 1455 | 1465 | 1415 | 1415 (pt_compiled) | within noise |
| 57 | 2 | ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | CUDA >= PyTorch | 1.03 | 456.5 | 421.9 | 444.4 | 444.4 (pt_compiled) |  |
| 58 | 2 | AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA >= PyTorch | 1.02 | 172.0 | 163.8 | 167.9 | 167.9 (pt_compiled) |  |
| 59 | 2 | AdaLN | training | bf16 | d128 cond128 A5/A48 (atoms) | 1024 | CUDA >= Triton | 1.02 | 102.3 | 100.2 | 137.0 | 137.0 (pt_compiled) |  |
| 60 | 2 | ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | CUDA >= Triton | 1.02 | 118.8 | 116.6 | 126.9 | 126.9 (pt_compiled) | within noise |
| 61 | 2 | Transition conditioning (pair) | training | fp32 | D128 n2 TF32 | 128 | Triton >= PyTorch | 1.02 | 110.5 | 122.9 | 120.8 | 120.8 (pt_compiled) | within noise |
| 62 | 2 | Transition conditioning (pair) | training | fp32 | D128 n2 TF32 | 768 | Triton >= PyTorch | 1.02 | 2032 | 2524 | 2481 | 2481 (pt_compiled) | within noise |
| 63 | 2 | ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | CUDA >= Triton | 1.02 | 2482 | 2440 | 2538 | 2538 (pt_compiled) | within noise |
| 64 | 2 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 128 | Triton >= PyTorch | 1.02 | 132.8 | 127.0 | 124.9 | 124.9 (pt_compiled) | within noise |
| 65 | 2 | ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 384 | CUDA >= Triton | 1.01 | 1208 | 1195 | 1234 | 1234 (pt_compiled) | within noise |
| 66 | 2 | ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | CUDA >= Triton | 1.01 | 1330 | 1319 | 1361 | 1361 (pt_compiled) | within noise |
| 67 | 2 | bias-only DiT | training | bf16 | d768 16x48 per-sample pair256 A4 | 256 | Triton >= PyTorch | 1.01 | 346.1 | 507.9 | 503.7 | 503.7 (pt_compiled) | within noise |
| 68 | 2 | ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | CUDA >= Triton | 1.01 | 1460 | 1452 | 1565 | 1565 (pt_compiled) | within noise |
| 69 | 2 | AdaLN | inference | fp32 | d768 cond384 A5/A48 | 768 | CUDA >= Triton | 1.00 | 32.7 | 32.6 | 36.8 | 36.8 (pt_compiled) | within noise |
| 70 | 2 | ConditionedTransition | inference | bf16 | d768 A5/A48 | 384 | CUDA >= Triton | 1.00 | 45.1 | 45.0 | 61.4 | 61.4 (pt_compiled) | within noise |
| 71 | 2 | AdaLN | inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA >= Triton | 1.00 | 18.3 | 18.3 | 22.5 | 22.5 (pt_compiled) | within noise |
| 72 | 2 | Transition conditioning (pair) | training | fp32 | D128 n2 TF32 | 384 | Triton >= PyTorch | 1.00 | 582.7 | 714.7 | 713.7 | 713.7 (pt_compiled) | within noise |
| 73 | 2 | Transition conditioning (single) | inference | fp32 | D384 n2 TF32 | 128 | Triton >= PyTorch | 1.00 | 22.6 | 26.6 | 26.6 | 26.6 (pt_compiled) | within noise |
| 74 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 128 | CUDA >= PyTorch | 1.00 | 30.7 | 36.9 | 30.7 | 30.7 (pt_compiled) |  |
| 75 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 384 | CUDA >= PyTorch | 1.00 | 84.0 | 98.1 | 84.0 | 84.0 (pt_compiled) |  |
| 76 | 2 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 384 | CUDA >= Triton | 1.00 | 298.9 | 298.8 | 301.0 | 301.0 (pt_compiled) | within noise |
| 77 | 2 | Transition conditioning (single) | training | fp32 | D384 n2 TF32 | 384 | Triton >= PyTorch | 1.00 | 61.3 | 67.5 | 67.5 | 67.5 (pt_compiled) | within noise |
| 78 | 2 | ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 768 | CUDA >= Triton | 1.00 | 106.5 | 106.4 | 114.7 | 114.7 (pt_compiled) | within noise |
| 79 | 2 | AdaLN | inference | bf16 | d768 cond384 A5/A48 | 384 | CUDA >= Triton | 1.00 | 18.2 | 18.2 | 24.4 | 24.4 (pt_compiled) | within noise |
| 80 | 2 | AdaLN | inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 384 | CUDA >= Triton | 1.00 | 22.4 | 22.4 | 26.6 | 26.6 (pt_compiled) | within noise |
| 81 | 2 | AdaLN | inference | bf16 | d128 cond128 A5/A48 (atoms) | 1024 | CUDA >= Triton | 1.00 | 14.1 | 14.1 | 20.5 | 20.5 (pt_compiled) | within noise |
| 82 | 2 | LayerNorm (pair) | inference | bf16 | x[1,L,L,256] | 384 | Triton >= PyTorch | 1.00 | 32.7 | 30.7 | 30.7 | 30.7 (pt_compiled) | within noise |
| 83 | 2 | pair_embed_ca | inference | bf16 out, fp32 geometry | B48 | 128 | CUDA >= Triton | 1.00 | 79.7 | 79.7 | 706.6 | 706.6 (pt_compiled) | within noise |
| 84 | 1 | Transition | inference | fp32 | D256 TF32 | 384 | Triton >= PyTorch | 1.42 | 520.0 | 1115 | 784.3 | 784.3 (pt_compiled) |  |
| 85 | 1 | Transition | training | bf16 | D512 | 384 | Triton >= PyTorch | 1.31 | 2844 | 4149 | 3166 | 3166 (pt_compiled) |  |
| 86 | 1 | Transition | training | fp32 | D256 TF32 | 384 | Triton >= PyTorch | 1.18 | 2420 | 2962 | 2505 | 2505 (pt_compiled) |  |
| 87 | 1 | bias-only DiT | training | fp32 | d768 16x48 per-sample pair256 A4 | 256 | Triton >= PyTorch | 1.12 | 605.2 | 852.0 | 763.9 | 763.9 (pt_compiled) |  |
| 88 | 1 | LayerNorm (pair) | inference | fp32 | x[1,L,L,128] | 384 | CUDA >= Triton | 1.07 | 30.7 | 28.7 | 30.6 | 30.6 (pt_compiled) |  |
| 89 | 1 | bias-only DiT | inference | fp32 | d768 16x48 per-sample pair256 A4 | 512 | Triton >= PyTorch | 1.03 | 448.5 | 729.9 | 708.6 | 708.6 (pt_compiled) |  |
| 90 | 1 | LayerNorm (pair) | inference | fp32 | x[1,L,L,128] | 384 | CUDA >= PyTorch | 1.00 | 30.7 | 28.7 | 30.6 | 30.6 (pt_compiled) |  |

## Before / after (the `0fb41e60` ladder of 3ec96849 vs this run)

Same harness, groups, repeats and GPUs; the earlier page's CSV is the baseline. Counts by kind of violation:

| | rows | OK | rows breaking the chain | CUDA >= Triton | Triton >= PyTorch | CUDA >= PyTorch | routing gap | other n.a. |
|---|---|---|---|---|---|---|---|---|
| 3ec96849 (before) | 432 | 239 | 158 | 39 | 132 | 48 | 43 | 1 |
| this tree (after) | 432 | 356 | 74 | 33 | 43 | 14 | 2 | 0 |

Rows that broke the chain (or had no CUDA path) before and pass now: **122**; rows that passed before and break now: **6**; rows whose kind changed: 28. Rows present only before / only now: 0 / 0.

### Fixed (violating or no CUDA path before, OK now)

| op | mode | dtype | shape | L | before | after | CUDA us before -> after | Triton us before -> after | PyTorch compiled us now |
|---|---|---|---|---|---|---|---|---|---|
| AttentionPairBias | inference | bf16 | 8x48 pair128 | 384 | Triton>=PyTorch | OK | 32.9 -> 34.5 | 75.6 -> 65.4 | 71.5 |
| AttentionPairBias | inference | bf16 | 8x48 pair128 | 768 | Triton>=PyTorch | OK | 57.3 -> 57.3 | 151.4 -> 112.5 | 145.4 |
| MSAPairWeightedAveraging | inference | bf16 | S1024 8x32 pair64 | 384 | no CUDA path | OK | 483.4 -> 215.1 | 344.0 -> 344.0 | 475.0 |
| MSAPairWeightedAveraging | inference | bf16 | S1024 8x32 pair64 | 768 | no CUDA path | OK | 1055 -> 495.6 | 729.1 -> 743.3 | 1047 |
| MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair64 | 384 | Triton>=PyTorch, no CUDA path | OK | 1368 -> 658.6 | 1379 -> 1215 | 1348 |
| MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair64 | 768 | no CUDA path | OK | 2915 -> 1473 | 2846 -> 2544 | 2859 |
| MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair128 | 256 | Triton>=PyTorch | OK | 438.4 -> 438.8 | 976.0 -> 867.3 | 912.2 |
| MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair128 | 384 | Triton>=PyTorch | OK | 662.3 -> 650.1 | 1408 -> 1243 | 1367 |
| MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair128 | 768 | Triton>=PyTorch | OK | 1470 -> 1454 | 2951 -> 2658 | 2931 |
| OuterProductMean | inference | bf16 | S1024 msa64 pair64 | 384 | no CUDA path | OK | 673.7 -> 300.9 | 319.4 -> 317.5 | 442.2 |
| OuterProductMean | inference | bf16 | S1024 msa64 pair64 | 768 | no CUDA path | OK | 2499 -> 1111 | 1159 -> 1163 | 1560 |
| OuterProductMean | training | bf16 | S1024 msa64 pair64 | 384 | no CUDA path | OK | 1251 -> 921.7 | 1159 -> 1120 | 1253 |
| OuterProductMean | training | bf16 | S1024 msa64 pair64 | 768 | no CUDA path | OK | 4482 -> 3458 | 4006 -> 4087 | 4476 |
| OuterProductMean | training | bf16 | S1024 msa64 pair128 | 256 | Triton>=PyTorch | OK | 444.4 -> 450.3 | 670.6 -> 647.0 | 661.5 |
| Transition | training | bf16 | D64 | 768 | Triton>=PyTorch | OK | 396.1 -> 395.4 | 1132 -> 780.2 | 1073 |
| Transition | training | bf16 | D128 | 384 | Triton>=PyTorch | OK | 270.4 -> 271.2 | 662.4 -> 548.8 | 597.0 |
| Transition | training | bf16 | D128 | 768 | Triton>=PyTorch | OK | 1049 -> 1033 | 2430 -> 2006 | 2111 |
| Transition | training | bf16 | D128 | 1024 | Triton>=PyTorch | OK | 1905 -> 1922 | 4249 -> 3476 | 3675 |
| smooth_lddt | inference | fp32 | A48 B1 | 1024 | CUDA>=Triton | OK | 172.0 -> 133.2 | 165.9 -> 167.8 | 60503 |
| token DiT (fused+composed) | inference | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | no CUDA path | OK | 141.3 -> 75.6 | 202.6 -> 192.6 | 221.2 |
| token DiT (fused+composed) | inference | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 768 | no CUDA path | OK | 235.6 -> 120.7 | 407.4 -> 389.6 | 477.2 |
| token_pair_init | inference | fp32 in, fp32 out | B16 | 128 | CUDA>=Triton | OK | 71.5 -> 49.0 | 51.0 -> 51.1 | 442.3 |
| token_pair_init | inference | fp32 in, fp32 out | B16 | 256 | CUDA>=Triton | OK | 203.6 -> 135.0 | 169.9 -> 169.9 | 1348 |
| token_pair_init | inference | fp32 in, fp32 out | B16 | 384 | CUDA>=Triton | OK | 372.4 -> 290.8 | 370.5 -> 370.6 | 3056 |
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 768 | CUDA>=Triton | OK | 26.5 -> 22.3 | 26.5 -> 24.6 | 28.4 |
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 1024 | CUDA>=Triton | OK | 32.6 -> 26.4 | 30.6 -> 30.5 | 32.8 |
| AdaLN | inference | bf16 | d128 cond128 A5/A48 (atoms) | 8192 | Triton>=PyTorch | OK | 30.8 -> 26.4 | 34.6 -> 34.6 | 34.7 |
| AdaLN | inference | fp32 | d768 cond384 A5/A48 | 128 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | OK | 28.6 -> 18.4 | 26.6 -> 20.4 | 22.5 |
| AdaLN | inference | fp32 | d768 cond384 A5/A48 | 384 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | OK | 33.1 -> 24.5 | 30.7 -> 26.6 | 28.6 |
| AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | CUDA>=Triton | OK | 18.5 -> 16.3 | 18.4 -> 18.4 | 20.4 |
| AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | CUDA>=PyTorch, Triton>=PyTorch | OK | 30.7 -> 26.6 | 38.8 -> 26.6 | 28.7 |
| AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA>=PyTorch, Triton>=PyTorch | OK | 26.6 -> 18.4 | 38.9 -> 22.5 | 24.6 |
| AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | CUDA>=PyTorch, Triton>=PyTorch | OK | 34.8 -> 28.4 | 42.9 -> 30.7 | 32.8 |
| AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | CUDA>=PyTorch, Triton>=PyTorch | OK | 46.8 -> 39.0 | 75.7 -> 40.9 | 45.0 |
| AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | Triton>=PyTorch | OK | 382.8 -> 382.9 | 530.3 -> 389.1 | 393.3 |
| AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | Triton>=PyTorch | OK | 692.1 -> 691.2 | 989.1 -> 718.9 | 736.2 |
| AttentionPairBias | inference | bf16 | 16x24 pair128 | 384 | Triton>=PyTorch | OK | 34.8 -> 34.8 | 75.7 -> 65.4 | 71.5 |
| AttentionPairBias Chai-1 (gate bias 1.0) | inference | bf16 | 16x24 pair256 | 384 | Triton>=PyTorch | OK | 43.0 -> 43.0 | 102.2 -> 79.9 | 86.0 |
| AttentionPairBias Chai-1 (gate bias 1.0) | inference | bf16 | 16x24 pair256 | 768 | Triton>=PyTorch | OK | 90.2 -> 90.2 | 251.7 -> 168.0 | 219.0 |
| AugmentedAttention atom | inference | fp32 | d128 pair16 4 heads A5/A48 | 128 | no CUDA path | OK | 239.6 -> 102.2 | 214.9 -> 214.8 | 237.3 |
| AugmentedAttention atom | inference | fp32 | d128 pair16 4 heads A5/A48 | 384 | no CUDA path | OK | 1691 -> 391.0 | 1427 -> 1429 | 1677 |
| AugmentedAttention atom | inference | fp32 | d128 pair16 4 heads A5/A48 | 1024 | no CUDA path | OK | 11228 -> 2401 | 9282 -> 9263 | 11138 |
| AugmentedAttention atom | training | fp32 | d128 pair16 4 heads A5/A48 | 128 | no CUDA path | OK | 3727 -> 1627 | 2575 -> 2563 | 3736 |
| AugmentedAttention atom | training | fp32 | d128 pair16 4 heads A5/A48 | 384 | no CUDA path | OK | 29392 -> 10249 | 17555 -> 17509 | 29489 |
| AugmentedAttention token | training | fp32 | d768 cond384 pair128 A5/A48 | 128 | Triton>=PyTorch | OK | 667.7 -> 667.7 | 782.3 -> 759.7 | 781.3 |
| AugmentedAttention token | inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | CUDA>=PyTorch, Triton>=PyTorch | OK | 104.5 -> 75.7 | 110.5 -> 94.1 | 100.3 |
| AugmentedAttention token | inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | CUDA>=Triton, CUDA>=PyTorch | OK | 257.0 -> 147.4 | 233.3 -> 223.2 | 247.7 |
| AugmentedAttention token | inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | CUDA>=Triton, CUDA>=PyTorch | OK | 697.4 -> 333.9 | 616.3 -> 581.5 | 655.2 |
| AugmentedAttention token | training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | Triton>=PyTorch, no CUDA path | OK | 882.7 -> 781.3 | 872.6 -> 824.4 | 861.0 |
| AugmentedAttention token | training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | no CUDA path | OK | 3488 -> 2160 | 3191 -> 3025 | 3440 |
| AugmentedAttention token | training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | no CUDA path | OK | 10552 -> 5757 | 9076 -> 8700 | 10446 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 768 | Triton>=PyTorch | OK | 69.7 -> 63.1 | 83.9 -> 63.4 | 75.7 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 1024 | Triton>=PyTorch | OK | 79.9 -> 73.7 | 100.3 -> 81.9 | 96.1 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 768 | CUDA>=PyTorch, Triton>=PyTorch | OK | 1124 -> 1126 | 1253 -> 1132 | 1139 |
| ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 128 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | OK | 69.7 -> 47.0 | 55.3 -> 49.1 | 55.2 |
| ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 384 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | OK | 98.3 -> 73.7 | 79.8 -> 75.7 | 81.8 |
| ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 768 | Triton>=PyTorch | OK | 2193 -> 2196 | 2337 -> 2223 | 2235 |
| ConditionedTransition | inference | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | Triton>=PyTorch | OK | 38.9 -> 34.8 | 51.2 -> 41.0 | 47.1 |
| ConditionedTransition | inference | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | Triton>=PyTorch | OK | 59.4 -> 55.4 | 128.9 -> 65.6 | 73.7 |
| ConditionedTransition | inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | Triton>=PyTorch | OK | 84.0 -> 81.7 | 213.0 -> 108.5 | 110.6 |
| ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | Triton>=PyTorch | OK | 2686 -> 2687 | 3723 -> 2702 | 2898 |
| ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | OK | 71.7 -> 49.1 | 67.7 -> 53.2 | 59.3 |
| ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | OK | 104.5 -> 79.8 | 93.9 -> 81.8 | 87.9 |
| LocalDiT ESMFold2 atom block | inference | bf16 | d128 192 keys rotary (atoms=8L) | 512 | no CUDA path | OK | 67.7 -> 67.7 | 268.2 -> 268.1 | 375.0 |
| LocalDiT ESMFold2 atom block | inference | bf16 | d128 192 keys rotary (atoms=8L) | 1024 | no CUDA path | OK | 108.7 -> 108.6 | 452.6 -> 450.6 | 657.4 |
| LocalDiT ESMFold2 atom block | training | bf16 | d128 192 keys rotary (atoms=8L) | 512 | no CUDA path | OK | 1396 -> 1395 | 3729 -> 3741 | 7554 |
| LocalDiT ESMFold2 atom block | training | bf16 | d128 192 keys rotary (atoms=8L) | 1024 | no CUDA path | OK | 2562 -> 2561 | 7269 -> 7270 | 14768 |
| LocalDiT atom block | inference | fp32 | d128 32x128 (atoms=8L) | 512 | Triton>=PyTorch | OK | 118.7 -> 118.7 | 339.9 -> 262.2 | 304.0 |
| MSAPairWeightedAveraging value_dim 8 | inference | bf16 | S512 8x8 pair128 | 256 | Triton>=PyTorch | OK | 59.4 -> 59.4 | 98.3 -> 81.8 | 94.2 |
| MSAPairWeightedAveraging value_dim 8 | inference | bf16 | S512 8x8 pair128 | 384 | Triton>=PyTorch | OK | 81.8 -> 80.9 | 139.2 -> 120.8 | 129.1 |
| MSAPairWeightedAveraging value_dim 8 | training | bf16 | S512 8x8 pair128 | 256 | Triton>=PyTorch | OK | 214.0 -> 214.0 | 404.3 -> 293.9 | 330.7 |
| MSAPairWeightedAveraging value_dim 8 | training | bf16 | S512 8x8 pair128 | 384 | Triton>=PyTorch | OK | 289.7 -> 289.4 | 578.6 -> 406.4 | 461.6 |
| OuterProductMean Chai-1 (8 groups) | training | bf16 | S512 pair256 8x64 | 256 | Triton>=PyTorch | OK | 331.7 -> 331.8 | 552.8 -> 474.9 | 505.9 |
| OuterProductMean Chai-1 (8 groups) | training | bf16 | S512 pair256 8x64 | 384 | Triton>=PyTorch | OK | 584.9 -> 591.7 | 960.5 -> 861.6 | 913.3 |
| OuterProductMean RF3 (bias, divide before) | training | bf16 | S512 msa64 pair128 | 256 | Triton>=PyTorch | OK | 315.0 -> 315.4 | 489.5 -> 481.3 | 483.2 |
| RMSNorm | inference | bf16 | x[48,L,768] | 1024 | CUDA>=Triton, CUDA>=PyTorch | OK | 53.2 -> 30.8 | 32.6 -> 32.6 | 32.7 |
| RMSNorm | training | bf16 | x[48,L,768] | 1024 | CUDA>=Triton, CUDA>=PyTorch | OK | 120.8 -> 79.8 | 75.9 -> 81.9 | 126.9 |
| Transition | inference | bf16 | D256 | 384 | Triton>=PyTorch | OK | 190.2 -> 194.4 | 440.5 -> 372.8 | 423.7 |
| Transition | inference | bf16 | D384 | 384 | Triton>=PyTorch | OK | 407.5 -> 407.4 | 830.0 -> 696.4 | 756.6 |
| Transition | inference | bf16 | D384 | 768 | Triton>=PyTorch | OK | 1613 -> 1596 | 3339 -> 2736 | 3061 |
| Transition | inference | fp32 | D128 TF32 | 384 | Triton>=PyTorch | OK | 217.0 -> 188.1 | 425.5 -> 227.4 | 384.9 |
| Transition | inference | fp32 | D128 TF32 | 768 | Triton>=PyTorch | OK | 792.5 -> 697.2 | 1602 -> 843.6 | 1450 |
| Transition conditioning (pair) | inference | fp32 | D128 n2 TF32 | 384 | Triton>=PyTorch | OK | 147.3 -> 116.7 | 257.9 -> 136.0 | 243.8 |
| Transition conditioning (pair) | inference | fp32 | D128 n2 TF32 | 768 | Triton>=PyTorch | OK | 520.1 -> 421.8 | 952.3 -> 489.3 | 913.2 |
| TriMul bidirectional hidden!=D | inference | bf16 | D64 hidden128 | 384 | CUDA>=Triton | OK | 603.7 -> 116.8 | 231.5 -> 231.5 | 661.5 |
| TriMul bidirectional hidden!=D | training | bf16 | D64 hidden128 | 384 | no CUDA path | OK | 1871 -> 680.6 | 782.2 -> 777.2 | 1916 |
| bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 512 | Triton>=PyTorch | OK | 2729 -> 2756 | 3272 -> 3123 | 3271 |
| bias-only DiT | inference | bf16 | d768 16x48 pair256 A5/A48 | 256 | no CUDA path | OK | 106.5 -> 49.0 | 102.3 -> 98.2 | 135.2 |
| bias-only DiT | inference | bf16 | d768 16x48 pair256 A5/A48 | 512 | no CUDA path | OK | 143.4 -> 63.3 | 137.2 -> 131.1 | 217.1 |
| bias-only DiT | inference | bf16 | d768 16x48 pair256 A5/A48 | 768 | no CUDA path | OK | 172.0 -> 83.8 | 184.4 -> 167.8 | 325.7 |
| bias-only DiT | training | bf16 | d768 16x48 pair256 A5/A48 | 256 | Triton>=PyTorch, no CUDA path | OK | 909.2 -> 796.6 | 961.6 -> 878.6 | 951.2 |
| bias-only DiT | training | bf16 | d768 16x48 pair256 A5/A48 | 512 | Triton>=PyTorch, no CUDA path | OK | 1760 -> 1495 | 1850 -> 1742 | 1817 |
| bias-only DiT | training | bf16 | d768 16x48 pair256 A5/A48 | 768 | Triton>=PyTorch, no CUDA path | OK | 2738 -> 2257 | 2899 -> 2612 | 2734 |
| bias-only DiT | inference | fp32 | d768 16x48 pair256 A5/A48 | 256 | no CUDA path | OK | 165.7 -> 75.7 | 137.2 -> 126.9 | 165.8 |
| bias-only DiT | inference | fp32 | d768 16x48 pair256 A5/A48 | 512 | no CUDA path | OK | 206.9 -> 106.4 | 182.2 -> 176.1 | 317.3 |
| bias-only DiT | training | fp32 | d768 16x48 pair256 A5/A48 | 256 | Triton>=PyTorch, no CUDA path | OK | 1678 -> 1458 | 1768 -> 1691 | 1736 |
| bias-only DiT | training | fp32 | d768 16x48 pair256 A5/A48 | 512 | Triton>=PyTorch, no CUDA path | OK | 3364 -> 2863 | 3518 -> 3340 | 3434 |
| bias-only DiT | inference | bf16 | d768 16x48 per-sample pair256 A4 | 256 | no CUDA path | OK | 155.5 -> 106.3 | 157.5 -> 155.6 | 188.5 |
| bias-only DiT | inference | bf16 | d768 16x48 per-sample pair256 A4 | 512 | no CUDA path | OK | 315.4 -> 221.0 | 323.5 -> 317.2 | 424.0 |
| bias-only DiT | training | bf16 | d768 16x48 per-sample pair256 A4 | 512 | Triton>=PyTorch | OK | 1305 -> 757.8 | 1330 -> 852.0 | 1183 |
| token DiT (fused+composed) | inference | fp32 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | no CUDA path | OK | 362.4 -> 124.8 | 297.0 -> 290.8 | 315.4 |
| token DiT (fused+composed) | training | fp32 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | no CUDA path | OK | 4657 -> 2839 | 4296 -> 4143 | 4633 |
| token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | OK | 415.7 -> 304.3 | 374.5 -> 317.5 | 364.5 |
| token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 512 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | OK | 591.8 -> 421.0 | 550.9 -> 473.1 | 518.1 |
| token_pair_init | inference | fp32 in, fp32 out | B48 | 128 | CUDA>=Triton | OK | 176.0 -> 110.5 | 132.9 -> 133.0 | 1024 |
| token_pair_init | inference | fp32 in, fp32 out | B48 | 256 | CUDA>=Triton | OK | 559.1 -> 403.4 | 491.3 -> 491.4 | 3966 |
| AttentionPairBias | inference | bf16 | 8x48 pair256 | 384 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | OK | 100.1 -> 41.1 | 100.1 -> 79.7 | 85.9 |
| AttentionPairBias | training | bf16 | 8x48 pair256 | 384 | CUDA>=Triton, CUDA>=PyTorch | OK | 317.6 -> 134.9 | 286.7 -> 247.8 | 298.0 |
| AttentionPairBias gate bias 1.0 | inference | bf16 | 16x24 pair128 | 384 | Triton>=PyTorch | OK | 34.7 -> 34.8 | 75.7 -> 65.5 | 71.7 |
| AugmentedAttention atom | inference | bf16 | d128 A5/A48 | 128 | CUDA>=Triton, CUDA>=PyTorch | OK | 194.4 -> 67.5 | 159.7 -> 159.7 | 184.2 |
| AugmentedAttention atom | inference | bf16 | d128 A5/A48 | 384 | CUDA>=Triton, CUDA>=PyTorch | OK | 1261 -> 237.4 | 906.2 -> 906.1 | 1156 |
| AugmentedAttention atom | training | bf16 | d128 A5/A48 | 128 | no CUDA path | OK | 2787 -> 968.7 | 1607 -> 1590 | 2789 |
| AugmentedAttention atom | training | bf16 | d128 A5/A48 | 384 | no CUDA path | OK | 25084 -> 5643 | 10288 -> 10278 | 25010 |
| OuterProductMean RF3 (bias, divide after) | training | bf16 | S512 msa64 pair128 | 256 | Triton>=PyTorch | OK | 317.3 -> 317.2 | 495.5 -> 485.4 | 489.4 |
| Transition | inference | bf16 | D512 | 384 | Triton>=PyTorch | OK | 792.4 -> 784.2 | 1253 -> 1057 | 1111 |
| TriMul incoming | inference | bf16 | D512 | 384 | Triton>=PyTorch | OK | 591.9 -> 591.7 | 1658 -> 1128 | 1399 |
| TriMul outgoing | inference | bf16 | D512 | 384 | Triton>=PyTorch | OK | 602.0 -> 610.2 | 1658 -> 1151 | 1424 |
| bias-only DiT | training | bf16 | 12x64 | 512 | Triton>=PyTorch | OK | 1457 -> 1470 | 1759 -> 1668 | 1755 |
| bias-only DiT | inference | fp32 | d768 16x48 per-sample pair256 A4 | 256 | Triton>=PyTorch, no CUDA path | OK | 305.1 -> 184.2 | 272.3 -> 266.1 | 266.2 |
| bias-only DiT | training | fp32 | d768 16x48 per-sample pair256 A4 | 512 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | OK | 2390 -> 1364 | 2380 -> 1819 | 2094 |
| token DiT + extra_pair_bias | inference | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | Triton>=PyTorch | OK | 81.7 -> 81.7 | 159.7 -> 147.4 | 147.5 |
| token DiT + extra_pair_bias | training | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | Triton>=PyTorch | OK | 417.8 -> 405.5 | 489.5 -> 449.2 | 458.7 |

### New violations (OK before)

| op | mode | dtype | shape | L | before | after | CUDA us before -> after | Triton us before -> after | PyTorch compiled us now |
|---|---|---|---|---|---|---|---|---|---|
| LayerNorm (pair) | inference | bf16 | x[1,L,L,128] | 384 | OK | CUDA>=Triton | 20.4 -> 20.4 | 18.3 -> 18.3 | 22.5 |
| LayerNorm (pair) | inference | bf16 | x[1,L,L,128] | 768 | OK | CUDA>=Triton | 53.1 -> 57.2 | 51.3 -> 53.1 | 65.5 |
| LayerNorm (pair) | training | bf16 | x[1,L,L,128] | 384 | OK | CUDA>=Triton | 112.4 -> 51.0 | 51.2 -> 49.0 | 71.7 |
| AdaLN | training | bf16 | d128 cond128 A5/A48 (atoms) | 1024 | OK | CUDA>=Triton | 96.2 -> 102.3 | 106.3 -> 100.2 | 137.0 |
| Transition conditioning (single) | inference | fp32 | D384 n2 TF32 | 768 | OK | Triton>=PyTorch | 24.4 -> 24.5 | 28.6 -> 28.6 | 26.7 |
| LayerNorm (pair) | inference | fp32 | x[1,L,L,128] | 384 | OK | CUDA>=Triton, CUDA>=PyTorch | 30.5 -> 30.7 | 28.5 -> 28.7 | 30.6 |

### Changed kind

| op | mode | dtype | shape | L | before | after | CUDA us before -> after | Triton us before -> after | PyTorch compiled us now |
|---|---|---|---|---|---|---|---|---|---|
| LayerNorm (pair) | training | bf16 | x[1,L,L,128] | 768 | Triton>=PyTorch | CUDA>=Triton | 236.3 -> 147.3 | 194.5 -> 131.2 | 190.5 |
| AdaLN | inference | fp32 | d768 cond384 A5/A48 | 768 | CUDA>=PyTorch, Triton>=PyTorch | CUDA>=Triton | 41.0 -> 32.7 | 50.7 -> 32.6 | 36.8 |
| AdaLN | training | fp32 | d768 cond384 A5/A48 | 128 | CUDA>=PyTorch, Triton>=PyTorch | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | 133.1 -> 132.8 | 153.5 -> 127.0 | 124.9 |
| AdaLN | training | fp32 | d768 cond384 A5/A48 | 384 | Triton>=PyTorch | CUDA>=Triton | 298.9 -> 298.9 | 365.7 -> 298.8 | 301.0 |
| AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | CUDA>=PyTorch, Triton>=PyTorch | CUDA>=Triton | 41.0 -> 38.9 | 55.3 -> 36.8 | 39.0 |
| AdaLN | training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | Triton>=PyTorch | CUDA>=Triton | 147.3 -> 145.3 | 155.5 -> 141.2 | 149.6 |
| AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA>=PyTorch, Triton>=PyTorch | CUDA>=Triton, CUDA>=PyTorch | 171.9 -> 172.0 | 214.9 -> 163.8 | 167.9 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | CUDA>=PyTorch, Triton>=PyTorch | CUDA>=Triton, CUDA>=PyTorch | 628.7 -> 628.7 | 685.1 -> 591.9 | 602.1 |
| ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 768 | CUDA>=PyTorch, Triton>=PyTorch | CUDA>=Triton | 120.9 -> 106.5 | 125.0 -> 106.4 | 114.7 |
| ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 128 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | CUDA>=Triton, CUDA>=PyTorch | 497.7 -> 497.7 | 497.6 -> 471.1 | 473.1 |
| ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 384 | Triton>=PyTorch | CUDA>=Triton | 1207 -> 1208 | 1263 -> 1195 | 1234 |
| ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | CUDA>=PyTorch, Triton>=PyTorch | CUDA>=Triton, CUDA>=PyTorch | 456.0 -> 456.5 | 568.4 -> 421.9 | 444.4 |
| ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | Triton>=PyTorch | CUDA>=Triton | 1461 -> 1460 | 1981 -> 1452 | 1565 |
| ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | CUDA>=PyTorch, Triton>=PyTorch | CUDA>=Triton | 133.1 -> 118.8 | 155.7 -> 116.6 | 126.9 |
| ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA>=PyTorch, Triton>=PyTorch | CUDA>=Triton, CUDA>=PyTorch | 552.7 -> 553.0 | 571.3 -> 524.4 | 534.0 |
| ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | Triton>=PyTorch | CUDA>=Triton | 1332 -> 1330 | 1459 -> 1319 | 1361 |
| ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | Triton>=PyTorch | CUDA>=Triton | 2481 -> 2482 | 2696 -> 2440 | 2538 |
| LayerNorm (pair) | inference | bf16 | x[1,L,L,256] | 384 | Triton>=PyTorch | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | 30.6 -> 32.7 | 28.7 -> 30.7 | 30.7 |
| LayerNorm (pair) | training | bf16 | x[1,L,L,256] | 384 | CUDA>=PyTorch, Triton>=PyTorch | CUDA>=Triton | 108.4 -> 83.8 | 116.6 -> 75.6 | 102.4 |
| RMSNorm | inference | bf16 | x[48,L,768] | 384 | CUDA>=Triton, CUDA>=PyTorch | CUDA>=Triton | 26.5 -> 16.3 | 16.1 -> 15.8 | 16.4 |
| RMSNorm | training | bf16 | x[48,L,768] | 384 | CUDA>=Triton, CUDA>=PyTorch | CUDA>=Triton | 59.4 -> 38.9 | 38.9 -> 36.8 | 59.4 |
| Transition | training | bf16 | D768 | 384 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | Triton>=PyTorch | 8763 -> 5994 | 7929 -> 7871 | 6154 |
| Transition conditioning (single) | training | fp32 | D384 n2 TF32 | 128 | CUDA>=PyTorch, Triton>=PyTorch | Triton>=PyTorch | 59.3 -> 57.2 | 65.4 -> 65.4 | 57.3 |
| Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 384 | CUDA>=PyTorch, Triton>=PyTorch | Triton>=PyTorch | 32.7 -> 30.8 | 36.7 -> 36.8 | 32.8 |
| Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 768 | CUDA>=PyTorch, Triton>=PyTorch | Triton>=PyTorch | 102.5 -> 96.2 | 118.8 -> 120.6 | 100.2 |
| bias-only DiT | training | bf16 | d768 16x48 per-sample pair256 A4 | 256 | CUDA>=PyTorch, Triton>=PyTorch | Triton>=PyTorch | 519.0 -> 346.1 | 536.6 -> 507.9 | 503.7 |
| bias-only DiT | inference | fp32 | d768 16x48 per-sample pair256 A4 | 512 | Triton>=PyTorch, no CUDA path | Triton>=PyTorch | 758.8 -> 448.5 | 735.2 -> 729.9 | 708.6 |
| bias-only DiT | training | fp32 | d768 16x48 per-sample pair256 A4 | 256 | CUDA>=Triton, CUDA>=PyTorch, Triton>=PyTorch | Triton>=PyTorch | 876.5 -> 605.2 | 848.0 -> 852.0 | 763.9 |

## Ops missing a level on B200

### No hand-CUDA path (`miniworld` on B200 runs the PyTorch reference)

| op | mode | dtype | shape | L | miniworld (= PyTorch) us | Triton us | Triton speed-up lost |
|---|---|---|---|---|---|---|---|
| token DiT (fused+composed) | training | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 1690 | 2272 | 0.74x |
| token DiT (fused+composed) | training | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 768 | 3799 | 6026 | 0.63x |

### Other rows where a rung has no measurement

| op | mode | dtype | shape | L | CUDA us | Triton us | PyTorch compiled us | why |
|---|---|---|---|---|---|---|---|---|

## Per module

`c/t` = CUDA / Triton, `t/p` = Triton / best PyTorch. Bold verdicts are violations.

### TriMul outgoing

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 | 128 | 26.4 | 42.7 | 75.7 | 165.8 | 0.62 | 0.56 | OK |
| inference | bf16 | D128 | 384 | 85.8 | 190.0 | 391.0 | 1049 | 0.45 | 0.49 | OK |
| inference | bf16 | D128 | 768 | 329.6 | 681.4 | 1673 | 4514 | 0.48 | 0.41 | OK |
| inference | bf16 | D128 | 1024 | 647.0 | 1223 | 6374 | 12696 | 0.53 | 0.19 | OK |
| inference | bf16 | D256 | 384 | 229.3 | 491.5 | 737.1 | 1834 | 0.47 | 0.67 | OK |
| inference | bf16 | D256 | 768 | 983.1 | 1811 | 3167 | 8600 | 0.54 | 0.57 | OK |
| inference | bf16 | D512 | 384 | 610.2 | 1151 | 1424 | 3467 | 0.53 | 0.81 | OK |
| inference | bf16 | D512 | 768 | 2492 | 4621 | 7304 | 17263 | 0.54 | 0.63 | OK |
| inference | bf16 | D64 | 128 | 20.4 | 28.6 | 57.3 | 126.8 | 0.71 | 0.50 | OK |
| inference | bf16 | D64 | 384 | 51.2 | 79.9 | 208.9 | 677.2 | 0.64 | 0.38 | OK |
| inference | bf16 | D64 | 768 | 167.8 | 274.2 | 851.8 | 2756 | 0.61 | 0.32 | OK |
| training | bf16 | D128 | 128 | 109.4 | 150.5 | 257.1 | 667.6 | 0.73 | 0.59 | OK |
| training | bf16 | D128 | 384 | 334.8 | 611.4 | 1233 | 3247 | 0.55 | 0.50 | OK |
| training | bf16 | D128 | 768 | 1244 | 2217 | 5048 | 13876 | 0.56 | 0.44 | OK |
| training | bf16 | D128 | 1024 | 2398 | 3981 | 19193 | 50483 | 0.60 | 0.21 | OK |
| training | bf16 | D256 | 384 | 1035 | 1272 | 2265 | 5845 | 0.81 | 0.56 | OK |
| training | bf16 | D256 | 768 | 3965 | 5260 | 10163 | 28986 | 0.75 | 0.52 | OK |
| training | bf16 | D512 | 384 | 2491 | 3438 | 4579 | 11134 | 0.72 | 0.75 | OK |
| training | bf16 | D512 | 768 | 9876 | 14480 | 23656 | 60376 | 0.68 | 0.61 | OK |
| training | bf16 | D64 | 128 | 70.6 | 112.6 | 191.5 | 491.5 | 0.63 | 0.59 | OK |
| training | bf16 | D64 | 384 | 191.5 | 336.9 | 705.7 | 1977 | 0.57 | 0.48 | OK |
| training | bf16 | D64 | 768 | 621.6 | 1111 | 2565 | 7795 | 0.56 | 0.43 | OK |

### TriMul incoming

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 | 384 | 85.9 | 190.3 | 391.2 | 1048 | 0.45 | 0.49 | OK |
| inference | bf16 | D128 | 768 | 323.4 | 681.8 | 1716 | 4508 | 0.47 | 0.40 | OK |
| inference | bf16 | D512 | 384 | 591.7 | 1128 | 1399 | 3469 | 0.52 | 0.81 | OK |
| inference | bf16 | D64 | 384 | 53.2 | 79.9 | 208.7 | 677.6 | 0.67 | 0.38 | OK |
| inference | bf16 | D64 | 768 | 167.3 | 274.3 | 849.9 | 2758 | 0.61 | 0.32 | OK |
| training | bf16 | D128 | 384 | 335.1 | 613.3 | 1232 | 3249 | 0.55 | 0.50 | OK |
| training | bf16 | D128 | 768 | 1243 | 2215 | 5057 | 13892 | 0.56 | 0.44 | OK |
| training | bf16 | D512 | 384 | 2495 | 3402 | 4545 | 11135 | 0.73 | 0.75 | OK |
| training | bf16 | D64 | 384 | 189.5 | 336.8 | 705.6 | 1979 | 0.56 | 0.48 | OK |
| training | bf16 | D64 | 768 | 616.9 | 1115 | 2537 | 7788 | 0.55 | 0.44 | OK |

### TriMul outgoing hidden!=D

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 hidden256 | 384 | 126.8 | 325.5 | 678.0 | 1655 | 0.39 | 0.48 | OK |
| inference | bf16 | D64 hidden128 | 128 | 22.4 | 34.8 | 71.7 | 159.2 | 0.64 | 0.49 | OK |
| inference | bf16 | D64 hidden128 | 384 | 69.6 | 137.2 | 325.6 | 966.6 | 0.51 | 0.42 | OK |
| training | bf16 | D128 hidden256 | 384 | 498.8 | 963.7 | 2018 | 5226 | 0.52 | 0.48 | OK |
| training | bf16 | D64 hidden128 | 128 | 87.1 | 128.0 | 238.6 | 620.5 | 0.68 | 0.54 | OK |
| training | bf16 | D64 hidden128 | 384 | 255.0 | 479.4 | 1106 | 2964 | 0.53 | 0.43 | OK |

### TriMul bidirectional

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 | 128 | 32.7 | 61.2 | 112.4 | 238.6 | 0.53 | 0.54 | OK |
| inference | bf16 | D128 | 384 | 132.1 | 331.6 | 699.3 | 1697 | 0.40 | 0.47 | OK |
| inference | bf16 | D128 | 768 | 557.0 | 1212 | 2976 | 7085 | 0.46 | 0.41 | OK |
| inference | bf16 | D256 | 384 | 362.4 | 741.3 | 1237 | 3099 | 0.49 | 0.60 | OK |
| inference | bf16 | D256 | 768 | 1548 | 2966 | 6457 | 14542 | 0.52 | 0.46 | OK |
| inference | bf16 | D512 | 384 | 977.6 | 2001 | 2867 | 6317 | 0.49 | 0.70 | OK |
| inference | bf16 | D64 | 384 | 75.7 | 141.2 | 338.0 | 991.2 | 0.54 | 0.42 | OK |
| inference | bf16 | D64 | 768 | 262.2 | 491.4 | 1630 | 3986 | 0.53 | 0.30 | OK |
| training | bf16 | D128 | 128 | 153.6 | 191.3 | 376.8 | 812.0 | 0.80 | 0.51 | OK |
| training | bf16 | D128 | 384 | 525.4 | 975.7 | 2075 | 4471 | 0.54 | 0.47 | OK |
| training | bf16 | D128 | 768 | 2014 | 3948 | 9213 | 18540 | 0.51 | 0.43 | OK |
| training | bf16 | D256 | 384 | 1599 | 2143 | 3985 | 8210 | 0.75 | 0.54 | OK |
| training | bf16 | D256 | 768 | 6275 | 9215 | 20706 | 39378 | 0.68 | 0.45 | OK |
| training | bf16 | D512 | 384 | 3964 | 5898 | 9309 | 16628 | 0.67 | 0.63 | OK |
| training | bf16 | D64 | 384 | 267.3 | 490.5 | 1112 | 2584 | 0.54 | 0.44 | OK |
| training | bf16 | D64 | 768 | 902.2 | 1725 | 4189 | 9989 | 0.52 | 0.41 | OK |

### TriMul bidirectional hidden!=D

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D64 hidden128 | 384 | 116.8 | 231.5 | 661.5 | 1607 | 0.50 | 0.35 | OK |
| training | bf16 | D64 hidden128 | 384 | 680.6 | 777.2 | 1916 | 4153 | 0.88 | 0.41 | OK |

### TriAttn

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 4x32 | 128 | 32.5 | 57.2 | 88.0 | 178.1 | 0.57 | 0.65 | OK |
| inference | bf16 | D128 4x32 | 384 | 175.9 | 352.3 | 1323 | 2406 | 0.50 | 0.27 | OK |
| inference | bf16 | D128 4x32 | 768 | 917.9 | 1879 | 9967 | 16617 | 0.49 | 0.19 | OK |
| inference | bf16 | D256 8x32 | 384 | 329.6 | 604.1 | 2597 | 4615 | 0.55 | 0.23 | OK |
| inference | bf16 | D64 2x32 | 384 | 102.3 | 190.3 | 591.8 | 1308 | 0.54 | 0.32 | OK |
| inference | bf16 | D64 2x32 | 768 | 501.6 | 947.0 | 5022 | 8685 | 0.53 | 0.19 | OK |
| training | bf16 | D128 4x32 | 128 | 113.9 | 205.9 | 271.5 | 547.7 | 0.55 | 0.76 | OK |
| training | bf16 | D128 4x32 | 384 | 646.0 | 1383 | 2198 | 5374 | 0.47 | 0.63 | OK |
| training | bf16 | D128 4x32 | 768 | 3401 | 7722 | 13788 | 35602 | 0.44 | 0.56 | OK |
| training | bf16 | D256 8x32 | 384 | 1334 | 2522 | 6269 | 10309 | 0.53 | 0.40 | OK |
| training | bf16 | D64 2x32 | 384 | 414.9 | 781.3 | 1554 | 2928 | 0.53 | 0.50 | OK |
| training | bf16 | D64 2x32 | 768 | 1925 | 3957 | 7074 | 18591 | 0.49 | 0.56 | OK |

### Transition

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 | 128 | 16.3 | 30.6 | 38.8 | 83.9 | 0.53 | 0.79 | OK |
| inference | bf16 | D128 | 384 | 55.2 | 96.2 | 208.8 | 568.4 | 0.57 | 0.46 | OK |
| inference | bf16 | D128 | 768 | 198.5 | 331.6 | 770.0 | 2143 | 0.60 | 0.43 | OK |
| inference | bf16 | D128 | 1024 | 348.1 | 565.1 | 1377 | 3773 | 0.62 | 0.41 | OK |
| inference | bf16 | D256 | 384 | 194.4 | 372.8 | 423.7 | 997.2 | 0.52 | 0.88 | OK |
| inference | bf16 | D256 | 768 | 774.2 | 1398 | 1730 | 3877 | 0.55 | 0.81 | OK |
| inference | bf16 | D384 | 384 | 407.4 | 696.4 | 756.6 | 1498 | 0.59 | 0.92 | OK |
| inference | bf16 | D384 | 768 | 1596 | 2736 | 3061 | 6036 | 0.58 | 0.89 | OK |
| inference | bf16 | D512 | 384 | 784.2 | 1057 | 1111 | 2071 | 0.74 | 0.95 | OK |
| inference | bf16 | D64 | 384 | 28.5 | 47.0 | 137.1 | 364.4 | 0.61 | 0.34 | OK |
| inference | bf16 | D64 | 768 | 83.9 | 145.3 | 382.8 | 1398 | 0.58 | 0.38 | OK |
| inference | bf16 | D768 | 384 | 1835 | 2442 | 2120 | 3474 | 0.75 | 1.15 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D128 TF32 | 384 | 188.1 | 227.4 | 384.9 | 735.0 | 0.83 | 0.59 | OK |
| inference | fp32 | D128 TF32 | 768 | 697.2 | 843.6 | 1450 | 2832 | 0.83 | 0.58 | OK |
| inference | fp32 | D256 TF32 | 384 | 520.0 | 1115 | 784.3 | 1359 | 0.47 | 1.42 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D128 | 128 | 57.8 | 98.2 | 108.6 | 288.7 | 0.59 | 0.90 | OK |
| training | bf16 | D128 | 384 | 271.2 | 548.8 | 597.0 | 1547 | 0.49 | 0.92 | OK |
| training | bf16 | D128 | 768 | 1033 | 2006 | 2111 | 5717 | 0.51 | 0.95 | OK |
| training | bf16 | D128 | 1024 | 1922 | 3476 | 3675 | 10035 | 0.55 | 0.95 | OK |
| training | bf16 | D256 | 384 | 995.4 | 1431 | 1227 | 2775 | 0.70 | 1.17 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D256 | 768 | 3878 | 5687 | 4470 | 10623 | 0.68 | 1.27 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D384 | 384 | 1738 | 2685 | 2125 | 4256 | 0.65 | 1.26 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D384 | 768 | 6820 | 10661 | 8406 | 16824 | 0.64 | 1.27 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D512 | 384 | 2844 | 4149 | 3166 | 5958 | 0.69 | 1.31 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D64 | 384 | 112.9 | 235.0 | 344.1 | 936.1 | 0.48 | 0.68 | OK |
| training | bf16 | D64 | 768 | 395.4 | 780.2 | 1073 | 3482 | 0.51 | 0.73 | OK |
| training | bf16 | D768 | 384 | 5994 | 7871 | 6154 | 9968 | 0.76 | 1.28 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D128 TF32 | 384 | 897.1 | 1155 | 1119 | 2194 | 0.78 | 1.03 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D128 TF32 | 768 | 3335 | 4304 | 3980 | 8219 | 0.77 | 1.08 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D256 TF32 | 384 | 2420 | 2962 | 2505 | 4358 | 0.82 | 1.18 | **VIOLATION(portable_vs_pytorch)** |

### Transition conditioning (pair)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 | D128 n2 TF32 | 128 | 24.5 | 38.7 | 40.8 | 63.4 | 0.63 | 0.95 | OK |
| inference | fp32 | D128 n2 TF32 | 384 | 116.7 | 136.0 | 243.8 | 488.6 | 0.86 | 0.56 | OK |
| inference | fp32 | D128 n2 TF32 | 768 | 421.8 | 489.3 | 913.2 | 1871 | 0.86 | 0.54 | OK |
| inference | fp32 | D256 n2 TF32 | 128 | 53.1 | 85.9 | 67.5 | 102.3 | 0.62 | 1.27 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D256 n2 TF32 | 384 | 321.4 | 638.9 | 482.2 | 835.5 | 0.50 | 1.33 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D256 n2 TF32 | 768 | 1216 | 2464 | 1851 | 3277 | 0.49 | 1.33 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D128 n2 TF32 | 128 | 110.5 | 122.9 | 120.8 | 247.8 | 0.90 | 1.02 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D128 n2 TF32 | 384 | 582.7 | 714.7 | 713.7 | 1406 | 0.82 | 1.00 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D128 n2 TF32 | 768 | 2032 | 2524 | 2481 | 5185 | 0.80 | 1.02 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D256 n2 TF32 | 128 | 223.1 | 258.0 | 229.2 | 405.5 | 0.86 | 1.13 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D256 n2 TF32 | 384 | 1505 | 1756 | 1391 | 2485 | 0.86 | 1.26 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D256 n2 TF32 | 768 | 5280 | 6227 | 5035 | 9448 | 0.85 | 1.24 | **VIOLATION(portable_vs_pytorch)** |

### Transition conditioning (single)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 | D384 n2 TF32 | 128 | 22.6 | 26.6 | 26.6 | 28.6 | 0.85 | 1.00 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D384 n2 TF32 | 384 | 24.6 | 26.6 | 26.6 | 30.6 | 0.92 | 1.00 | OK |
| inference | fp32 | D384 n2 TF32 | 768 | 24.5 | 28.6 | 26.7 | 30.8 | 0.86 | 1.07 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D768 n2 TF32 | 128 | 30.7 | 36.9 | 30.7 | 34.7 | 0.83 | 1.20 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D768 n2 TF32 | 384 | 30.8 | 36.8 | 32.8 | 36.8 | 0.84 | 1.12 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D768 n2 TF32 | 768 | 32.9 | 45.1 | 36.8 | 42.9 | 0.73 | 1.22 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D384 n2 TF32 | 128 | 57.2 | 65.4 | 57.3 | 69.7 | 0.87 | 1.14 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D384 n2 TF32 | 384 | 61.3 | 67.5 | 67.5 | 78.8 | 0.91 | 1.00 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D384 n2 TF32 | 768 | 63.5 | 73.7 | 75.5 | 86.0 | 0.86 | 0.98 | OK |
| training | fp32 | D768 n2 TF32 | 128 | 77.7 | 92.0 | 71.6 | 86.0 | 0.84 | 1.29 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D768 n2 TF32 | 384 | 84.0 | 98.1 | 84.0 | 98.3 | 0.86 | 1.17 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D768 n2 TF32 | 768 | 96.2 | 120.6 | 100.2 | 114.7 | 0.80 | 1.20 | **VIOLATION(portable_vs_pytorch)** |

### OuterProductMean

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S1024 msa64 pair128 | 256 | 149.4 | 171.9 | 223.1 | 903.2 | 0.87 | 0.77 | OK |
| inference | bf16 | S1024 msa64 pair128 | 384 | 305.2 | 343.9 | 462.8 | 1670 | 0.89 | 0.74 | OK |
| inference | bf16 | S1024 msa64 pair128 | 768 | 1126 | 1298 | 1667 | 5309 | 0.87 | 0.78 | OK |
| inference | bf16 | S1024 msa64 pair64 | 384 | 300.9 | 317.5 | 442.2 | 1660 | 0.95 | 0.72 | OK |
| inference | bf16 | S1024 msa64 pair64 | 768 | 1111 | 1163 | 1560 | 5272 | 0.96 | 0.75 | OK |
| training | bf16 | S1024 msa64 pair128 | 256 | 450.3 | 647.0 | 661.5 | 2098 | 0.70 | 0.98 | OK |
| training | bf16 | S1024 msa64 pair128 | 384 | 933.3 | 1290 | 1333 | 3899 | 0.72 | 0.97 | OK |
| training | bf16 | S1024 msa64 pair128 | 768 | 3490 | 4629 | 4679 | 12001 | 0.75 | 0.99 | OK |
| training | bf16 | S1024 msa64 pair64 | 384 | 921.7 | 1120 | 1253 | 3746 | 0.82 | 0.89 | OK |
| training | bf16 | S1024 msa64 pair64 | 768 | 3458 | 4087 | 4476 | 11754 | 0.85 | 0.91 | OK |

### MSAPairWeightedAveraging

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S1024 8x32 pair128 | 256 | 149.5 | 239.5 | 311.3 | 1044 | 0.62 | 0.77 | OK |
| inference | bf16 | S1024 8x32 pair128 | 384 | 217.0 | 358.2 | 483.3 | 1643 | 0.61 | 0.74 | OK |
| inference | bf16 | S1024 8x32 pair128 | 768 | 493.4 | 780.4 | 1067 | 3799 | 0.63 | 0.73 | OK |
| inference | bf16 | S1024 8x32 pair64 | 384 | 215.1 | 344.0 | 475.0 | 1604 | 0.63 | 0.72 | OK |
| inference | bf16 | S1024 8x32 pair64 | 768 | 495.6 | 743.3 | 1047 | 3674 | 0.67 | 0.71 | OK |
| training | bf16 | S1024 8x32 pair128 | 256 | 438.8 | 867.3 | 912.2 | 2894 | 0.51 | 0.95 | OK |
| training | bf16 | S1024 8x32 pair128 | 384 | 650.1 | 1243 | 1367 | 3875 | 0.52 | 0.91 | OK |
| training | bf16 | S1024 8x32 pair128 | 768 | 1454 | 2658 | 2931 | 8726 | 0.55 | 0.91 | OK |
| training | bf16 | S1024 8x32 pair64 | 384 | 658.6 | 1215 | 1348 | 3787 | 0.54 | 0.90 | OK |
| training | bf16 | S1024 8x32 pair64 | 768 | 1473 | 2544 | 2859 | 8393 | 0.58 | 0.89 | OK |

### OuterProductMean RF3 (bias, divide before)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S512 msa64 pair128 | 256 | 104.5 | 124.8 | 169.9 | 655.2 | 0.84 | 0.73 | OK |
| inference | bf16 | S512 msa64 pair128 | 384 | 202.8 | 237.3 | 335.7 | 1273 | 0.85 | 0.71 | OK |
| training | bf16 | S512 msa64 pair128 | 256 | 315.4 | 481.3 | 483.2 | 1568 | 0.66 | 1.00 | OK |
| training | bf16 | S512 msa64 pair128 | 384 | 615.5 | 905.3 | 966.7 | 2992 | 0.68 | 0.94 | OK |

### OuterProductMean RF3 (bias, divide after)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S512 msa64 pair128 | 256 | 106.4 | 126.9 | 170.0 | 511.8 | 0.84 | 0.75 | OK |
| inference | bf16 | S512 msa64 pair128 | 384 | 210.9 | 237.5 | 339.5 | 953.2 | 0.89 | 0.70 | OK |
| training | bf16 | S512 msa64 pair128 | 256 | 317.2 | 485.4 | 489.4 | 1283 | 0.65 | 0.99 | OK |
| training | bf16 | S512 msa64 pair128 | 384 | 639.9 | 911.3 | 956.5 | 2352 | 0.70 | 0.95 | OK |

### OuterProductMean Chai-1 (8 groups)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S512 pair256 8x64 | 256 | 77.8 | 118.6 | 169.8 | 687.9 | 0.66 | 0.70 | OK |
| inference | bf16 | S512 pair256 8x64 | 384 | 145.2 | 227.2 | 326.6 | 1308 | 0.64 | 0.70 | OK |
| training | bf16 | S512 pair256 8x64 | 256 | 331.8 | 474.9 | 505.9 | 2106 | 0.70 | 0.94 | OK |
| training | bf16 | S512 pair256 8x64 | 384 | 591.7 | 861.6 | 913.3 | 2901 | 0.69 | 0.94 | OK |

### MSAPairWeightedAveraging value_dim 8

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S512 8x8 pair128 | 256 | 59.4 | 81.8 | 94.2 | 377.0 | 0.73 | 0.87 | OK |
| inference | bf16 | S512 8x8 pair128 | 384 | 80.9 | 120.8 | 129.1 | 630.9 | 0.67 | 0.94 | OK |
| training | bf16 | S512 8x8 pair128 | 256 | 214.0 | 293.9 | 330.7 | 1341 | 0.73 | 0.89 | OK |
| training | bf16 | S512 8x8 pair128 | 384 | 289.4 | 406.4 | 461.6 | 1508 | 0.71 | 0.88 | OK |

### AttentionPairBias

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | 16x24 pair128 | 384 | 34.8 | 65.4 | 71.5 | 301.1 | 0.53 | 0.91 | OK |
| inference | bf16 | 8x48 pair128 | 384 | 34.5 | 65.4 | 71.5 | 296.8 | 0.53 | 0.92 | OK |
| inference | bf16 | 8x48 pair128 | 768 | 57.3 | 112.5 | 145.4 | 1003 | 0.51 | 0.77 | OK |
| inference | bf16 | 8x48 pair256 | 384 | 41.1 | 79.7 | 85.9 | 400.5 | 0.51 | 0.93 | OK |
| training | bf16 | 16x24 pair128 | 384 | 110.5 | 208.9 | 252.8 | 715.8 | 0.53 | 0.83 | OK |
| training | bf16 | 8x48 pair128 | 384 | 100.2 | 198.7 | 251.7 | 707.6 | 0.50 | 0.79 | OK |
| training | bf16 | 8x48 pair128 | 768 | 178.2 | 395.2 | 518.0 | 2203 | 0.45 | 0.76 | OK |
| training | bf16 | 8x48 pair256 | 384 | 134.9 | 247.8 | 298.0 | 929.8 | 0.54 | 0.83 | OK |

### AttentionPairBias Chai-1 (gate bias 1.0)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | 16x24 pair256 | 384 | 43.0 | 79.9 | 86.0 | 407.6 | 0.54 | 0.93 | OK |
| inference | bf16 | 16x24 pair256 | 768 | 90.2 | 168.0 | 219.0 | 1430 | 0.54 | 0.77 | OK |

### AttentionPairBias gate bias 1.0

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | 16x24 pair128 | 384 | 34.8 | 65.5 | 71.7 | 303.1 | 0.53 | 0.91 | OK |

### AugmentedAttention token

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d768 A5/A48 | 384 | 69.5 | 116.8 | 145.3 | 497.5 | 0.60 | 0.80 | OK |
| inference | bf16 | d768 A5/A48 | 768 | 126.8 | 243.7 | 337.9 | 1658 | 0.52 | 0.72 | OK |
| inference | bf16 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | 53.2 | 81.9 | 88.1 | 188.3 | 0.65 | 0.93 | OK |
| inference | bf16 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | 83.9 | 143.4 | 168.0 | 608.4 | 0.59 | 0.85 | OK |
| inference | bf16 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | 163.9 | 319.5 | 411.7 | 2076 | 0.51 | 0.78 | OK |
| inference | fp32 | d768 cond384 pair128 A5/A48 | 128 | 67.4 | 85.9 | 92.1 | 139.1 | 0.78 | 0.93 | OK |
| inference | fp32 | d768 cond384 pair128 A5/A48 | 384 | 118.6 | 178.1 | 202.8 | 428.0 | 0.67 | 0.88 | OK |
| inference | fp32 | d768 cond384 pair128 A5/A48 | 768 | 231.5 | 436.1 | 513.9 | 1371 | 0.53 | 0.85 | OK |
| inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | 75.7 | 94.1 | 100.3 | 147.5 | 0.80 | 0.94 | OK |
| inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | 147.4 | 223.2 | 247.7 | 480.2 | 0.66 | 0.90 | OK |
| inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | 333.9 | 581.5 | 655.2 | 1552 | 0.57 | 0.89 | OK |
| training | bf16 | d768 A5/A48 | 384 | 899.1 | 1530 | 2825 | 4950 | 0.59 | 0.54 | OK |
| training | bf16 | d768 A5/A48 | 768 | 2145 | 4391 | 9388 | 15675 | 0.49 | 0.47 | OK |
| training | bf16 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | 434.1 | 489.4 | 630.8 | 1236 | 0.89 | 0.78 | OK |
| training | bf16 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | 993.3 | 1701 | 2941 | 5345 | 0.58 | 0.58 | OK |
| training | bf16 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | 2394 | 4907 | 9698 | 16796 | 0.49 | 0.51 | OK |
| training | fp32 | d768 cond384 pair128 A5/A48 | 128 | 667.7 | 759.7 | 781.3 | 1193 | 0.88 | 0.97 | OK |
| training | fp32 | d768 cond384 pair128 A5/A48 | 384 | 1736 | 2790 | 3214 | 4837 | 0.62 | 0.87 | OK |
| training | fp32 | d768 cond384 pair128 A5/A48 | 768 | 4336 | 8092 | 9782 | 16126 | 0.54 | 0.83 | OK |
| training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | 781.3 | 824.4 | 861.0 | 1265 | 0.95 | 0.96 | OK |
| training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | 2160 | 3025 | 3440 | 5141 | 0.71 | 0.88 | OK |
| training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | 5757 | 8700 | 10446 | 16853 | 0.66 | 0.83 | OK |

### AugmentedAttention atom

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d128 A5/A48 | 128 | 67.5 | 159.7 | 184.2 | 1592 | 0.42 | 0.87 | OK |
| inference | bf16 | d128 A5/A48 | 384 | 237.4 | 906.1 | 1156 | 14075 | 0.26 | 0.78 | OK |
| inference | fp32 | d128 pair16 4 heads A5/A48 | 128 | 102.2 | 214.8 | 237.3 | 1474 | 0.48 | 0.90 | OK |
| inference | fp32 | d128 pair16 4 heads A5/A48 | 384 | 391.0 | 1429 | 1677 | 12804 | 0.27 | 0.85 | OK |
| inference | fp32 | d128 pair16 4 heads A5/A48 | 1024 | 2401 | 9263 | 11138 | 92158 | 0.26 | 0.83 | OK |
| training | bf16 | d128 A5/A48 | 128 | 968.7 | 1590 | 2789 | 8251 | 0.61 | 0.57 | OK |
| training | bf16 | d128 A5/A48 | 384 | 5643 | 10278 | 25010 | 71088 | 0.55 | 0.41 | OK |
| training | fp32 | d128 pair16 4 heads A5/A48 | 128 | 1627 | 2563 | 3736 | 7578 | 0.63 | 0.69 | OK |
| training | fp32 | d128 pair16 4 heads A5/A48 | 384 | 10249 | 17509 | 29489 | 68172 | 0.59 | 0.59 | OK |
| training | fp32 | d128 pair16 4 heads A5/A48 | 1024 | 65224 | 121004 | - | - | 0.54 | - | OK |

### token DiT (fused+composed)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d768 16x48 A5/A48 shared-cond | 384 | 75.6 | 174.1 | 200.7 | 607.2 | 0.43 | 0.87 | OK |
| inference | bf16 | d768 16x48 A5/A48 shared-cond | 768 | 120.7 | 323.6 | 409.6 | 1814 | 0.37 | 0.79 | OK |
| inference | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 75.6 | 192.6 | 221.2 | 709.5 | 0.39 | 0.87 | OK |
| inference | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 768 | 120.7 | 389.6 | 477.2 | 2217 | 0.31 | 0.82 | OK |
| inference | bf16 | d768 qk-norm shared-cond | 384 | 79.7 | 178.2 | 202.8 | 662.4 | 0.45 | 0.88 | OK |
| inference | fp32 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 124.8 | 290.8 | 315.4 | 585.9 | 0.43 | 0.92 | OK |
| inference | fp32 | d768 A5/A48 shared-cond | 384 | 124.8 | 256.0 | 280.6 | 540.6 | 0.49 | 0.91 | OK |
| training | bf16 | d768 16x48 A5/A48 shared-cond | 384 | 1676 | 2170 | 3513 | 6391 | 0.77 | 0.62 | OK |
| training | bf16 | d768 16x48 A5/A48 shared-cond | 768 | 3603 | 5616 | 10649 | 18593 | 0.64 | 0.53 | OK |
| training | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 1690 | 2272 | 3555 | 6612 | 0.74 | 0.64 | no CUDA path |
| training | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 768 | 3799 | 6026 | 10835 | 19422 | 0.63 | 0.56 | no CUDA path |
| training | bf16 | d768 qk-norm shared-cond | 384 | 1768 | 2252 | 3643 | 7612 | 0.79 | 0.62 | OK |
| training | fp32 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 2839 | 4143 | 4633 | 6852 | 0.69 | 0.89 | OK |
| training | fp32 | d768 A5/A48 shared-cond | 384 | 2875 | 4046 | 4583 | 6756 | 0.71 | 0.88 | OK |

### token DiT + extra_pair_bias

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | 54.8 | 129.1 | 131.1 | 335.8 | 0.42 | 0.98 | OK |
| inference | bf16 | d768 A2 shared-cond +[L,L] bias | 512 | 65.5 | 178.2 | 192.6 | 720.9 | 0.37 | 0.93 | OK |
| inference | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | 81.7 | 147.4 | 147.5 | 274.4 | 0.55 | 1.00 | OK |
| training | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | 304.3 | 317.5 | 364.5 | 1241 | 0.96 | 0.87 | OK |
| training | bf16 | d768 A2 shared-cond +[L,L] bias | 512 | 421.0 | 473.1 | 518.1 | 1653 | 0.89 | 0.91 | OK |
| training | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | 405.5 | 449.2 | 458.7 | 1158 | 0.90 | 0.98 | OK |

### DiT variant stack

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | Boltz-2 token | 384 | 81.8 | 168.0 | 207.1 | 636.9 | 0.49 | 0.81 | OK |
| inference | bf16 | Chai-1 token | 384 | 81.7 | 153.4 | 194.7 | 696.3 | 0.53 | 0.79 | OK |
| inference | bf16 | RoseTTAFold3 token | 384 | 83.9 | 155.7 | 194.6 | 645.1 | 0.54 | 0.80 | OK |
| inference | fp32 | Boltz-2 token | 384 | 143.1 | 252.1 | 268.3 | 546.8 | 0.57 | 0.94 | OK |
| inference | fp32 | Chai-1 token | 384 | 137.0 | 256.0 | 270.2 | 546.8 | 0.54 | 0.95 | OK |
| inference | fp32 | RoseTTAFold3 token | 384 | 135.1 | 237.6 | 257.8 | 536.4 | 0.57 | 0.92 | OK |
| training | bf16 | Boltz-2 token | 384 | 1764 | 2287 | 3672 | 6327 | 0.77 | 0.62 | OK |
| training | bf16 | Chai-1 token | 384 | 1491 | 2055 | 3398 | 6162 | 0.73 | 0.60 | OK |
| training | bf16 | RoseTTAFold3 token | 384 | 1670 | 2325 | 3672 | 6816 | 0.72 | 0.63 | OK |
| training | fp32 | Boltz-2 token | 384 | 2983 | 3867 | 4352 | 6561 | 0.77 | 0.89 | OK |
| training | fp32 | Chai-1 token | 384 | 2556 | 3410 | 3855 | 5963 | 0.75 | 0.88 | OK |
| training | fp32 | RoseTTAFold3 token | 384 | 2691 | 3617 | 4075 | 6640 | 0.74 | 0.89 | OK |

### LocalDiT atom block

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d128 32x128 (atoms=8L) | 512 | 59.3 | 239.7 | 258.1 | 1396 | 0.25 | 0.93 | OK |
| inference | bf16 | d128 32x128 (atoms=8L) | 1024 | 94.1 | 387.1 | 417.9 | 2663 | 0.24 | 0.93 | OK |
| inference | fp32 | d128 32x128 (atoms=8L) | 512 | 118.7 | 262.2 | 304.0 | 1203 | 0.45 | 0.86 | OK |
| training | bf16 | d128 32x128 (atoms=8L) | 512 | 1582 | 3159 | 4727 | 17318 | 0.50 | 0.67 | OK |
| training | bf16 | d128 32x128 (atoms=8L) | 1024 | 2962 | 5958 | 9104 | 33096 | 0.50 | 0.65 | OK |
| training | fp32 | d128 32x128 (atoms=8L) | 512 | 3896 | 5402 | 6415 | 16366 | 0.72 | 0.84 | OK |

### LocalDiT ESMFold2 atom block

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d128 192 keys rotary (atoms=8L) | 512 | 67.7 | 268.1 | 375.0 | 2209 | 0.25 | 0.71 | OK |
| inference | bf16 | d128 192 keys rotary (atoms=8L) | 1024 | 108.6 | 450.6 | 657.4 | 4188 | 0.24 | 0.69 | OK |
| training | bf16 | d128 192 keys rotary (atoms=8L) | 512 | 1395 | 3741 | 7554 | 25303 | 0.37 | 0.50 | OK |
| training | bf16 | d128 192 keys rotary (atoms=8L) | 1024 | 2561 | 7270 | 14768 | 50565 | 0.35 | 0.49 | OK |

### DiT variant stack (atom)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | Boltz-2 atom | 512 | 118.6 | 213.0 | 259.1 | 1387 | 0.56 | 0.82 | OK |
| inference | bf16 | Chai-1 atom | 512 | 85.8 | 188.3 | 326.7 | 1344 | 0.46 | 0.58 | OK |
| inference | bf16 | RoseTTAFold3 atom | 512 | 118.7 | 207.9 | 264.2 | 1439 | 0.57 | 0.79 | OK |
| inference | fp32 | Boltz-2 atom | 512 | 153.4 | 243.5 | 315.2 | 1241 | 0.63 | 0.77 | OK |
| inference | fp32 | Chai-1 atom | 512 | 118.7 | 215.1 | 284.7 | 1209 | 0.55 | 0.76 | OK |
| inference | fp32 | RoseTTAFold3 atom | 512 | 147.4 | 237.6 | 307.2 | 1276 | 0.62 | 0.77 | OK |
| training | bf16 | Boltz-2 atom | 512 | 2082 | 2804 | 6536 | 14618 | 0.74 | 0.43 | OK |
| training | bf16 | Chai-1 atom | 512 | 1601 | 2656 | 5640 | 14001 | 0.60 | 0.47 | OK |
| training | bf16 | RoseTTAFold3 atom | 512 | 2103 | 2727 | 6839 | 15600 | 0.77 | 0.40 | OK |
| training | fp32 | Boltz-2 atom | 512 | 2970 | 4483 | 6320 | 13753 | 0.66 | 0.71 | OK |
| training | fp32 | Chai-1 atom | 512 | 2344 | 3903 | 5343 | 12790 | 0.60 | 0.73 | OK |
| training | fp32 | RoseTTAFold3 atom | 512 | 2862 | 4291 | 6106 | 14192 | 0.67 | 0.70 | OK |

### bias-only DiT

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | 12x64 | 512 | 63.4 | 130.0 | 186.4 | 700.5 | 0.49 | 0.70 | OK |
| inference | bf16 | 24x32 | 512 | 65.4 | 133.1 | 188.3 | 702.5 | 0.49 | 0.71 | OK |
| inference | bf16 | d768 16x48 A5/A48 | 256 | 49.0 | 98.2 | 127.0 | 299.0 | 0.50 | 0.77 | OK |
| inference | bf16 | d768 16x48 A5/A48 | 512 | 63.3 | 132.9 | 184.4 | 684.1 | 0.48 | 0.72 | OK |
| inference | bf16 | d768 16x48 A5/A48 | 768 | 83.8 | 167.8 | 258.1 | 1279 | 0.50 | 0.65 | OK |
| inference | bf16 | d768 16x48 pair256 A5/A48 | 256 | 49.0 | 98.2 | 135.2 | 344.1 | 0.50 | 0.73 | OK |
| inference | bf16 | d768 16x48 pair256 A5/A48 | 512 | 63.3 | 131.1 | 217.1 | 867.4 | 0.48 | 0.60 | OK |
| inference | bf16 | d768 16x48 pair256 A5/A48 | 768 | 83.8 | 167.8 | 325.7 | 1681 | 0.50 | 0.52 | OK |
| inference | bf16 | d768 16x48 per-sample pair A4 | 256 | 86.0 | 133.1 | 153.6 | 620.5 | 0.65 | 0.87 | OK |
| inference | bf16 | d768 16x48 per-sample pair A4 | 512 | 155.7 | 233.5 | 307.1 | 1969 | 0.67 | 0.76 | OK |
| inference | bf16 | d768 16x48 per-sample pair256 A4 | 256 | 106.3 | 155.6 | 188.5 | 803.8 | 0.68 | 0.83 | OK |
| inference | bf16 | d768 16x48 per-sample pair256 A4 | 512 | 221.0 | 317.2 | 424.0 | 2681 | 0.70 | 0.75 | OK |
| inference | fp32 | d768 16x48 A5/A48 | 256 | 75.6 | 127.0 | 147.5 | 246.8 | 0.60 | 0.86 | OK |
| inference | fp32 | d768 16x48 A5/A48 | 512 | 104.3 | 174.1 | 253.9 | 554.0 | 0.60 | 0.69 | OK |
| inference | fp32 | d768 16x48 pair256 A5/A48 | 256 | 75.7 | 126.9 | 165.8 | 268.3 | 0.60 | 0.77 | OK |
| inference | fp32 | d768 16x48 pair256 A5/A48 | 512 | 106.4 | 176.1 | 317.3 | 630.8 | 0.60 | 0.56 | OK |
| inference | fp32 | d768 16x48 per-sample pair A4 | 256 | 131.1 | 178.1 | 204.8 | 487.3 | 0.74 | 0.87 | OK |
| inference | fp32 | d768 16x48 per-sample pair A4 | 512 | 275.1 | 386.9 | 478.2 | 1517 | 0.71 | 0.81 | OK |
| inference | fp32 | d768 16x48 per-sample pair256 A4 | 256 | 184.2 | 266.1 | 266.2 | 563.2 | 0.69 | 1.00 | OK |
| inference | fp32 | d768 16x48 per-sample pair256 A4 | 512 | 448.5 | 729.9 | 708.6 | 1807 | 0.61 | 1.03 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | 12x64 | 512 | 1470 | 1668 | 1755 | 4640 | 0.88 | 0.95 | OK |
| training | bf16 | 24x32 | 512 | 1509 | 1674 | 1783 | 4663 | 0.90 | 0.94 | OK |
| training | bf16 | d768 16x48 A5/A48 | 256 | 757.9 | 848.9 | 938.4 | 2545 | 0.89 | 0.90 | OK |
| training | bf16 | d768 16x48 A5/A48 | 512 | 1454 | 1657 | 1755 | 4638 | 0.88 | 0.94 | OK |
| training | bf16 | d768 16x48 A5/A48 | 768 | 2143 | 2475 | 2681 | 7506 | 0.87 | 0.92 | OK |
| training | bf16 | d768 16x48 pair256 A5/A48 | 256 | 796.6 | 878.6 | 951.2 | 2752 | 0.91 | 0.92 | OK |
| training | bf16 | d768 16x48 pair256 A5/A48 | 512 | 1495 | 1742 | 1817 | 4983 | 0.86 | 0.96 | OK |
| training | bf16 | d768 16x48 pair256 A5/A48 | 768 | 2257 | 2612 | 2734 | 8313 | 0.86 | 0.96 | OK |
| training | bf16 | d768 16x48 per-sample pair A4 | 256 | 266.3 | 363.4 | 413.7 | 1366 | 0.73 | 0.88 | OK |
| training | bf16 | d768 16x48 per-sample pair A4 | 512 | 460.9 | 656.4 | 868.4 | 4045 | 0.70 | 0.76 | OK |
| training | bf16 | d768 16x48 per-sample pair256 A4 | 256 | 346.1 | 507.9 | 503.7 | 1711 | 0.68 | 1.01 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | d768 16x48 per-sample pair256 A4 | 512 | 757.8 | 852.0 | 1183 | 5399 | 0.89 | 0.72 | OK |
| training | fp32 | d768 16x48 A5/A48 | 256 | 1434 | 1647 | 1711 | 2918 | 0.87 | 0.96 | OK |
| training | fp32 | d768 16x48 A5/A48 | 512 | 2756 | 3123 | 3271 | 5183 | 0.88 | 0.95 | OK |
| training | fp32 | d768 16x48 pair256 A5/A48 | 256 | 1458 | 1691 | 1736 | 3032 | 0.86 | 0.97 | OK |
| training | fp32 | d768 16x48 pair256 A5/A48 | 512 | 2863 | 3340 | 3434 | 5365 | 0.86 | 0.97 | OK |
| training | fp32 | d768 16x48 per-sample pair A4 | 256 | 479.1 | 545.9 | 594.0 | 1151 | 0.88 | 0.92 | OK |
| training | fp32 | d768 16x48 per-sample pair A4 | 512 | 894.9 | 1085 | 1350 | 3275 | 0.82 | 0.80 | OK |
| training | fp32 | d768 16x48 per-sample pair256 A4 | 256 | 605.2 | 852.0 | 763.9 | 1310 | 0.71 | 1.12 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 16x48 per-sample pair256 A4 | 512 | 1364 | 1819 | 2094 | 3928 | 0.75 | 0.87 | OK |

### SWA DiT (atom)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d128 atoms=8L shared-cond | 512 | 67.5 | 89.9 | 580.4 | 2139 | 0.75 | 0.15 | OK |
| inference | bf16 | d128 atoms=8L shared-cond | 1024 | 106.4 | 137.1 | 1824 | 7143 | 0.78 | 0.08 | OK |
| inference | fp32 | atoms=8L shared-cond | 512 | 94.1 | 206.5 | 2108 | 3586 | 0.46 | 0.10 | OK |
| training | bf16 | d128 atoms=8L shared-cond | 512 | 1204 | 2287 | 13500 | 30573 | 0.53 | 0.17 | OK |
| training | bf16 | d128 atoms=8L shared-cond | 1024 | 2244 | 4445 | 52309 | 109541 | 0.50 | 0.08 | OK |
| training | fp32 | atoms=8L shared-cond | 512 | 1961 | 5810 | 70914 | 87667 | 0.34 | 0.08 | OK |

### SWA atom attention (global)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | atoms=8L | 512 | 67.5 | - | 532.3 | 1951 | - | - | OK |
| inference | bf16 | atoms=8L | 1024 | 108.3 | - | 1747 | 6817 | - | - | OK |
| training | bf16 | atoms=8L | 512 | 1408 | - | 11770 | 27136 | - | - | OK |
| training | bf16 | atoms=8L | 1024 | 2739 | - | 49875 | 102819 | - | - | OK |

### AdaLN

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d128 cond128 A5/A48 (atoms) | 1024 | 14.1 | 14.1 | 20.5 | 49.1 | 1.00 | 0.69 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d128 cond128 A5/A48 (atoms) | 4096 | 20.3 | 24.4 | 26.5 | 104.2 | 0.83 | 0.92 | OK |
| inference | bf16 | d128 cond128 A5/A48 (atoms) | 8192 | 26.4 | 34.6 | 34.7 | 178.0 | 0.76 | 1.00 | OK |
| inference | bf16 | d768 cond384 A5/A48 | 384 | 18.2 | 18.2 | 24.4 | 57.1 | 1.00 | 0.75 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d768 cond384 A5/A48 | 768 | 22.3 | 24.6 | 28.4 | 77.6 | 0.91 | 0.86 | OK |
| inference | bf16 | d768 cond384 A5/A48 | 1024 | 26.4 | 30.5 | 32.8 | 96.1 | 0.87 | 0.93 | OK |
| inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 128 | 18.3 | 18.3 | 22.5 | 45.0 | 1.00 | 0.81 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 384 | 22.4 | 22.4 | 26.6 | 63.4 | 1.00 | 0.84 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 768 | 28.5 | 32.6 | 30.7 | 86.1 | 0.87 | 1.06 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 16.3 | 18.4 | 20.4 | 36.7 | 0.89 | 0.90 | OK |
| inference | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 26.6 | 26.6 | 28.7 | 75.7 | 1.00 | 0.93 | OK |
| inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 38.9 | 36.8 | 39.0 | 133.1 | 1.06 | 0.94 | **VIOLATION(cuda_vs_portable)** |
| inference | fp32 | d768 cond384 A5/A48 | 128 | 18.4 | 20.4 | 22.5 | 28.5 | 0.90 | 0.91 | OK |
| inference | fp32 | d768 cond384 A5/A48 | 384 | 24.5 | 26.6 | 28.6 | 36.9 | 0.92 | 0.93 | OK |
| inference | fp32 | d768 cond384 A5/A48 | 768 | 32.7 | 32.6 | 36.8 | 54.3 | 1.00 | 0.89 | **VIOLATION(cuda_vs_portable)** |
| inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 18.4 | 22.5 | 24.6 | 30.7 | 0.82 | 0.92 | OK |
| inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 28.4 | 30.7 | 32.8 | 43.0 | 0.93 | 0.94 | OK |
| inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 39.0 | 40.9 | 45.0 | 65.6 | 0.95 | 0.91 | OK |
| training | bf16 | d128 cond128 A5/A48 (atoms) | 1024 | 102.3 | 100.2 | 137.0 | 734.2 | 1.02 | 0.73 | **VIOLATION(cuda_vs_portable)** |
| training | bf16 | d128 cond128 A5/A48 (atoms) | 4096 | 290.7 | 317.3 | 358.4 | 1844 | 0.92 | 0.89 | OK |
| training | bf16 | d128 cond128 A5/A48 (atoms) | 8192 | 522.3 | 581.4 | 692.3 | 3471 | 0.90 | 0.84 | OK |
| training | bf16 | d768 cond384 A5/A48 | 384 | 178.2 | 190.4 | 219.0 | 700.5 | 0.94 | 0.87 | OK |
| training | bf16 | d768 cond384 A5/A48 | 768 | 306.9 | 347.9 | 378.8 | 1452 | 0.88 | 0.92 | OK |
| training | bf16 | d768 cond384 A5/A48 | 1024 | 388.1 | 446.5 | 493.4 | 1897 | 0.87 | 0.90 | OK |
| training | bf16 | d768 cond768 A5/A48 (ESMFold2) | 128 | 102.4 | 102.4 | 125.1 | 309.1 | 1.00 | 0.82 | OK |
| training | bf16 | d768 cond768 A5/A48 (ESMFold2) | 384 | 219.0 | 237.7 | 273.4 | 868.3 | 0.92 | 0.87 | OK |
| training | bf16 | d768 cond768 A5/A48 (ESMFold2) | 768 | 378.9 | 440.3 | 491.4 | 1684 | 0.86 | 0.90 | OK |
| training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 145.3 | 141.2 | 149.6 | 664.4 | 1.03 | 0.94 | **VIOLATION(cuda_vs_portable)** |
| training | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 447.1 | 466.9 | 505.8 | 1527 | 0.96 | 0.92 | OK |
| training | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 810.8 | 869.9 | 947.0 | 2813 | 0.93 | 0.92 | OK |
| training | fp32 | d768 cond384 A5/A48 | 128 | 132.8 | 127.0 | 124.9 | 221.2 | 1.05 | 1.02 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond384 A5/A48 | 384 | 298.9 | 298.8 | 301.0 | 586.8 | 1.00 | 0.99 | **VIOLATION(cuda_vs_portable)** |
| training | fp32 | d768 cond384 A5/A48 | 768 | 485.3 | 565.1 | 524.1 | 1203 | 0.86 | 1.08 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 172.0 | 163.8 | 167.9 | 264.2 | 1.05 | 0.98 | **VIOLATION(cuda_vs_portable)** |
| training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 382.9 | 389.1 | 393.3 | 749.5 | 0.98 | 0.99 | OK |
| training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 691.2 | 718.9 | 736.2 | 1416 | 0.96 | 0.98 | OK |

### ConditionedTransition

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d768 A5/A48 | 384 | 45.1 | 45.0 | 61.4 | 104.5 | 1.00 | 0.73 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d768 A5/A48 | 768 | 63.1 | 63.4 | 75.7 | 143.3 | 0.99 | 0.84 | OK |
| inference | bf16 | d768 A5/A48 | 1024 | 73.7 | 81.9 | 96.1 | 178.3 | 0.90 | 0.85 | OK |
| inference | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 34.8 | 41.0 | 47.1 | 67.5 | 0.85 | 0.87 | OK |
| inference | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 55.4 | 65.6 | 73.7 | 131.0 | 0.84 | 0.89 | OK |
| inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 81.7 | 108.5 | 110.6 | 227.4 | 0.75 | 0.98 | OK |
| inference | fp32 | d768 cond384 A5/A48 | 128 | 47.0 | 49.1 | 55.2 | 65.4 | 0.96 | 0.89 | OK |
| inference | fp32 | d768 cond384 A5/A48 | 384 | 73.7 | 75.7 | 81.8 | 100.2 | 0.97 | 0.93 | OK |
| inference | fp32 | d768 cond384 A5/A48 | 768 | 106.5 | 106.4 | 114.7 | 147.3 | 1.00 | 0.93 | **VIOLATION(cuda_vs_portable)** |
| inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 49.1 | 53.2 | 59.3 | 69.6 | 0.92 | 0.90 | OK |
| inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 79.8 | 81.8 | 87.9 | 106.4 | 0.98 | 0.93 | OK |
| inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 118.8 | 116.6 | 126.9 | 161.7 | 1.02 | 0.92 | **VIOLATION(cuda_vs_portable)** |
| training | bf16 | d768 A5/A48 | 384 | 628.7 | 591.9 | 602.1 | 1351 | 1.06 | 0.98 | **VIOLATION(cuda_vs_portable)** |
| training | bf16 | d768 A5/A48 | 768 | 1126 | 1132 | 1139 | 2680 | 1.00 | 0.99 | OK |
| training | bf16 | d768 A5/A48 | 1024 | 1455 | 1465 | 1415 | 3497 | 0.99 | 1.04 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 456.5 | 421.9 | 444.4 | 1065 | 1.08 | 0.95 | **VIOLATION(cuda_vs_portable)** |
| training | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 1460 | 1452 | 1565 | 3114 | 1.01 | 0.93 | **VIOLATION(cuda_vs_portable)** |
| training | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 2687 | 2702 | 2898 | 5699 | 0.99 | 0.93 | OK |
| training | fp32 | d768 cond384 A5/A48 | 128 | 497.7 | 471.1 | 473.1 | 638.9 | 1.06 | 1.00 | **VIOLATION(cuda_vs_portable)** |
| training | fp32 | d768 cond384 A5/A48 | 384 | 1208 | 1195 | 1234 | 1746 | 1.01 | 0.97 | **VIOLATION(cuda_vs_portable)** |
| training | fp32 | d768 cond384 A5/A48 | 768 | 2196 | 2223 | 2235 | 3350 | 0.99 | 0.99 | OK |
| training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 553.0 | 524.4 | 534.0 | 693.1 | 1.05 | 0.98 | **VIOLATION(cuda_vs_portable)** |
| training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 1330 | 1319 | 1361 | 1939 | 1.01 | 0.97 | **VIOLATION(cuda_vs_portable)** |
| training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 2482 | 2440 | 2538 | 3675 | 1.02 | 0.96 | **VIOLATION(cuda_vs_portable)** |

### LayerNorm (pair)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | x[1,L,L,128] | 384 | 20.4 | 18.3 | 22.5 | 227.2 | 1.11 | 0.81 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | x[1,L,L,128] | 768 | 57.2 | 53.1 | 65.5 | 886.7 | 1.08 | 0.81 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | x[1,L,L,256] | 384 | 32.7 | 30.7 | 30.7 | 325.5 | 1.07 | 1.00 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | x[1,L,L,128] | 384 | 30.7 | 28.7 | 30.6 | 161.7 | 1.07 | 0.94 | **VIOLATION(cuda_vs_portable)** |
| training | bf16 | x[1,L,L,128] | 384 | 51.0 | 49.0 | 71.7 | 487.3 | 1.04 | 0.68 | **VIOLATION(cuda_vs_portable)** |
| training | bf16 | x[1,L,L,128] | 768 | 147.3 | 131.2 | 190.5 | 1801 | 1.12 | 0.69 | **VIOLATION(cuda_vs_portable)** |
| training | bf16 | x[1,L,L,256] | 384 | 83.8 | 75.6 | 102.4 | 684.0 | 1.11 | 0.74 | **VIOLATION(cuda_vs_portable)** |
| training | fp32 | x[1,L,L,128] | 384 | 73.5 | 73.6 | 96.2 | 354.2 | 1.00 | 0.76 | OK |

### RMSNorm

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | x[48,L,768] | 384 | 16.3 | 15.8 | 16.4 | 79.7 | 1.03 | 0.97 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | x[48,L,768] | 1024 | 30.8 | 32.6 | 32.7 | 208.8 | 0.94 | 1.00 | OK |
| training | bf16 | x[48,L,768] | 384 | 38.9 | 36.8 | 59.4 | 257.9 | 1.06 | 0.62 | **VIOLATION(cuda_vs_portable)** |
| training | bf16 | x[48,L,768] | 1024 | 79.8 | 81.9 | 126.9 | 661.4 | 0.97 | 0.65 | OK |

### pair_embed_geom

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 out, fp32 geometry | B16 | 128 | 118.6 | 122.7 | 3890 | 1963 | 0.97 | 0.06 | OK |
| inference | bf16 out, fp32 geometry | B16 | 256 | 364.5 | 386.9 | 3078 | 7547 | 0.94 | 0.13 | OK |
| inference | bf16 out, fp32 geometry | B16 | 384 | 760.7 | 831.4 | 7204 | 16843 | 0.91 | 0.12 | OK |
| inference | bf16 out, fp32 geometry | B48 | 128 | 278.3 | 296.8 | 2474 | 5688 | 0.94 | 0.12 | OK |
| inference | bf16 out, fp32 geometry | B48 | 256 | 932.9 | 1079 | 8932 | 22495 | 0.86 | 0.12 | OK |
| training | bf16 out, fp32 geometry | B16 | 128 | 333.6 | 1417 | 4118 | 3271 | 0.24 | 0.43 | OK |
| training | bf16 out, fp32 geometry | B16 | 256 | 1080 | 5048 | 4060 | 12703 | 0.21 | 1.24 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 out, fp32 geometry | B16 | 384 | 2353 | 9664 | 8959 | 28420 | 0.24 | 1.08 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 out, fp32 geometry | B48 | 128 | 825.2 | 3308 | 3089 | 9570 | 0.25 | 1.07 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 out, fp32 geometry | B48 | 256 | 2955 | 12814 | 11874 | 37934 | 0.23 | 1.08 | **VIOLATION(portable_vs_pytorch)** |

### pair_embed_ca

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 out, fp32 geometry | B16 | 128 | 36.8 | 34.7 | 959.4 | 898.9 | 1.06 | 0.04 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 out, fp32 geometry | B16 | 256 | 100.2 | 102.3 | 909.2 | 3532 | 0.98 | 0.11 | OK |
| inference | bf16 out, fp32 geometry | B16 | 384 | 200.6 | 214.9 | 2045 | 7924 | 0.93 | 0.11 | OK |
| inference | bf16 out, fp32 geometry | B48 | 128 | 79.7 | 79.7 | 706.6 | 2652 | 1.00 | 0.11 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 out, fp32 geometry | B48 | 256 | 277.7 | 282.5 | 2637 | 10564 | 0.98 | 0.11 | OK |
| training | bf16 out, fp32 geometry | B16 | 128 | 180.1 | 341.8 | 1156 | 2082 | 0.53 | 0.30 | OK |
| training | bf16 out, fp32 geometry | B16 | 256 | 557.2 | 1113 | 1597 | 8122 | 0.50 | 0.70 | OK |
| training | bf16 out, fp32 geometry | B16 | 384 | 1198 | 1934 | 3715 | 18189 | 0.62 | 0.52 | OK |
| training | bf16 out, fp32 geometry | B48 | 128 | 448.4 | 706.5 | 1217 | 6110 | 0.63 | 0.58 | OK |
| training | bf16 out, fp32 geometry | B48 | 256 | 1539 | 2522 | 5452 | 24233 | 0.61 | 0.46 | OK |

### pair_embed_adaln

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 out, fp32 geometry | B16 | 128 | 131.1 | 194.5 | 1369 | 2235 | 0.67 | 0.14 | OK |
| inference | bf16 out, fp32 geometry | B16 | 256 | 340.0 | 718.8 | 1187 | 8890 | 0.47 | 0.61 | OK |
| inference | bf16 out, fp32 geometry | B16 | 384 | 839.5 | 1590 | 2647 | 19958 | 0.53 | 0.60 | OK |
| inference | bf16 out, fp32 geometry | B48 | 128 | 284.7 | 540.5 | 911.4 | 6671 | 0.53 | 0.59 | OK |
| inference | bf16 out, fp32 geometry | B48 | 256 | 871.8 | 2106 | 3434 | 26589 | 0.41 | 0.61 | OK |
| training | bf16 out, fp32 geometry | B16 | 128 | 395.2 | 1006 | 1983 | 4082 | 0.39 | 0.51 | OK |
| training | bf16 out, fp32 geometry | B16 | 256 | 1261 | 3155 | 3212 | 15926 | 0.40 | 0.98 | OK |
| training | bf16 out, fp32 geometry | B16 | 384 | 3200 | 8104 | 7214 | 35672 | 0.39 | 1.12 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 out, fp32 geometry | B48 | 128 | 931.9 | 2522 | 2569 | 11946 | 0.37 | 0.98 | OK |
| training | bf16 out, fp32 geometry | B48 | 256 | 3547 | 9337 | 10177 | 47406 | 0.38 | 0.92 | OK |

### token_pair_init

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 in, fp32 out | B16 | 128 | 49.0 | 51.1 | 442.3 | 790.4 | 0.96 | 0.12 | OK |
| inference | fp32 in, fp32 out | B16 | 256 | 135.0 | 169.9 | 1348 | 2888 | 0.79 | 0.13 | OK |
| inference | fp32 in, fp32 out | B16 | 384 | 290.8 | 370.6 | 3056 | 6403 | 0.78 | 0.12 | OK |
| inference | fp32 in, fp32 out | B48 | 128 | 110.5 | 133.0 | 1024 | 2183 | 0.83 | 0.13 | OK |
| inference | fp32 in, fp32 out | B48 | 256 | 403.4 | 491.4 | 3966 | 8522 | 0.82 | 0.12 | OK |
| training | fp32 in, fp32 out | B16 | 128 | 192.3 | 483.2 | 663.4 | 1010 | 0.40 | 0.73 | OK |
| training | fp32 in, fp32 out | B16 | 256 | 569.1 | 1462 | 2112 | 3666 | 0.39 | 0.69 | OK |
| training | fp32 in, fp32 out | B16 | 384 | 1186 | 3128 | 4725 | 8072 | 0.38 | 0.66 | OK |
| training | fp32 in, fp32 out | B48 | 128 | 501.8 | 1127 | 1526 | 2774 | 0.45 | 0.74 | OK |
| training | fp32 in, fp32 out | B48 | 256 | 1653 | 3892 | 6184 | 10756 | 0.42 | 0.63 | OK |

### smooth_lddt

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 | A16 B1 | 4096 | 610.3 | 751.6 | 335149 | 341315 | 0.81 | 0.00 | OK |
| inference | fp32 | A48 B1 | 1024 | 133.2 | 167.8 | 60503 | 61652 | 0.79 | 0.00 | OK |
| inference | fp32 | A48 B1 | 2048 | 382.9 | 546.8 | 241630 | 246117 | 0.70 | 0.00 | OK |
| training | fp32 | A16 B1 | 4096 | 1337 | 1999 | 527892 | 538492 | 0.67 | 0.00 | OK |
| training | fp32 | A48 B1 | 1024 | 282.6 | 411.7 | 95631 | 97934 | 0.69 | 0.00 | OK |
| training | fp32 | A48 B1 | 2048 | 849.8 | 1450 | 383910 | 392338 | 0.59 | 0.00 | OK |

### sampler augment_noise

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 | B5 | 4096 | 8.1 | 10.1 | 22.5 | 71.7 | 0.80 | 0.45 | OK |
| inference | fp32 | B5 | 12288 | 10.0 | 10.1 | 24.5 | 90.1 | 0.98 | 0.41 | OK |
| inference | fp32 | B5 | 24576 | 10.1 | 12.2 | 32.7 | 77.9 | 0.83 | 0.37 | OK |

### sampler align_euler

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 | B5 | 4096 | 12.2 | 18.4 | 106.5 | 307.2 | 0.67 | 0.17 | OK |
| inference | fp32 | B5 | 12288 | 16.3 | 18.4 | 102.3 | 493.6 | 0.89 | 0.18 | OK |
| inference | fp32 | B5 | 24576 | 18.4 | 22.5 | 106.5 | 665.5 | 0.82 | 0.21 | OK |

### sampler pool_atoms

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | B5 c768 | 4096 | 14.2 | 16.3 | 73.6 | 210.9 | 0.88 | 0.22 | OK |
| inference | bf16 | B5 c768 | 12288 | 22.4 | 26.5 | 206.7 | 624.7 | 0.85 | 0.13 | OK |
| inference | bf16 | B5 c768 | 24576 | 34.7 | 43.0 | 405.4 | 1218 | 0.81 | 0.11 | OK |

## How it was measured

* Harness: `benchmarks/runners/ladder.py` (`miniworld-engine ladder run|report`), which drives `benchmarks/runners/bench.py` (module targets; `ladder_ops` for function-level ops). Three rungs per row: `cuda` = `implementations=miniworld` (the engine's own routing on B200: the hand sm_100a kernels where they exist), `portable` = the same module under `settings.force_portable_kernels` (the Triton family), `pytorch` = the module's PyTorch statements under `torch.compile` (`pt_c`), also timed as eager statements replayed in a CUDA graph (`pt_eg`) and plain eager (`pt_e`).
* Every number is the median of 3 `do_bench` medians of a CUDA-graph replay of one step (inference = forward; training = forward + backward), bf16-mixed (fp32 master parameters, bf16 autocast) unless the dtype column says fp32 (TF32). The engine rungs are timed with and without `torch.compile` around them and the faster one counts.
* Verdicts are the ladder's own: a pair passes when ratio = lower rung / upper rung < 1. The PyTorch rung of a pair is the **fastest** PyTorch form (normally `pt_c`), which is at least as strict as comparing against compiled PyTorch only. `n.a.` = a rung has no measurement (it failed, refused, or launched another rung's kernels: no distinct path); a `routing gap` = `miniworld` on B200 ran the PyTorch reference for that call (no CUDA path). CUDA >= PyTorch is also checked for three-rung rows (implied by the chain).
* Synthetic inputs only (no training data). Times in **microseconds**.

## Triton tuning on B200 (what the Triton rung runs)

Every B200 Triton autotune cache was rebuilt from scratch on this tree before the run (engine commit 24f896295,
`miniworld-engine build all --kernels portable --gpus all` on the eight node-2 GPUs): 3906 (op, shape, width) units, 3874 ok, 32 skipped
(out of memory / shared memory by design), 0 failed; 152 B200 cache files, 4993 buckets. The five kernels registered by the record gate
(`gated_projection_{fwd,bwd}_residual_triton`, `bias_only_attention_{,bwd_}layernorm_gemm_triton`,
`mpnn_message_fwd_gemm_gelu_reduce_triton`) were tuned over their `grid` config set (`build <ops> grid --per-op`), as on H100: their
default set holds the single launch they had before registration, which the builder records without a timing.
`miniworld-engine dev cache-status --gpu B200`: no stale B200 cache.
CUDA kernels came from the prebuilt sm100 bundle of this line (953 artifacts), so no rung JIT-compiled during timing.

## Reproduce

```
# tune (GPU 2 excluded); then one ladder process per GPU over a partition of the groups
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 miniworld-engine build all --kernels portable --gpus all --units-per-gpu 2 --shards <dir>
CUDA_VISIBLE_DEVICES=<g> python benchmarks/runners/ladder.py run --group '<regex>' --repeats 3 --out <dir>/gpu<g>
python benchmarks/runners/ladder.py report --out <dir>/gpu0 <dir>/gpu1 ... --csv ladder.csv --md ladder.md
```

