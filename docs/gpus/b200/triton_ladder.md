# B200: the Triton ladder (hand CUDA < Triton < torch.compile'd PyTorch)

The engine's rule for every op, shape and mode: **time(hand CUDA) < time(Triton) < time(torch.compile'd PyTorch)**. This page measures it on a B200 after tuning the Triton (portable) kernels for this card. It reports; it changes no kernel.

Provenance: engine commit `0fb41e60` (dirty=False), NVIDIA B200 (sm_100a, 1000 W cap), torch 2.13.0+cu129, triton 3.7.1, run 2026-10-31T06:16:28 UTC on node 2 (`DCTN-0831174147-2`), one `ladder.py run` per GPU on GPUs 0,1,3-7 (GPU 2 hosted a stuck foreign process and was not used for timing; the other GPUs carry a yielding filler process).

## Findings

* **432 rows** (113 groups x mode x length: the 106 ladder groups plus 7 MiniWorld pair-D64 groups). 158 rows break the chain
  somewhere; a row can break two pairs, so the violation table has more lines than that. Several "violations" are ties
  (`within noise`; identical CUDA and Triton times mean both rungs ran the same path).
* **Hand CUDA vs Triton** holds on the big modules: median CUDA / Triton is 0.53 TriAttn, 0.56 TriMul, 0.59 PWA, 0.60 local
  DiT, 0.53 SWA DiT, 0.66 token DiT, 0.70 OPM, 0.70 bias-only DiT. The 39 CUDA >= Triton rows are small memory-bound calls --
  standalone RMSNorm x[48,L,768] (1.53-1.64x), token_pair_init B16 / B48 L128-256 (1.14-1.40x), AdaLN and
  ConditionedTransition inference (bf16 and fp32, 1.06-1.26x), pair_embed_ca B16 L128, smooth_lddt A48 N1024 inference
  (1.04x) -- plus TriMul bidirectional D64 hidden128 inference (CUDA 604 us vs Triton 232 us, 2.61x), AugmentedAttention
  atom bf16 inference (1.22-1.39x), AugmentedAttention token fp32 at the ESMFold2 width (1.10-1.13x), AttentionPairBias
  pair256 training (1.11x), token DiT with an extra pair bias (training 1.07-1.11x) and Transition D768 training (1.11x).
* **Triton vs torch.compile fails most** (132 rows). On MiniWorld training shapes: Transition D128 training at L384 / 768 / 1024
  (Triton 1.11-1.16x slower than compiled PyTorch) and D64 L768 (1.05x); MSAPairWeightedAveraging training S1024 pair128
  L256-768 (1.01-1.07x) and pair64 L384 (1.02x); OPM training L256 (1.02x, noise); AttentionPairBias inference 8x48 pair128
  L384 / 768 (1.03-1.06x); LayerNorm pair D128 training L768 (1.06x); La-Proteina pair_embed_geom / adaln training B16
  (1.08-1.24x). Off those shapes the fp32 (TF32) Triton paths lose broadly: AdaLN, ConditionedTransition, Transition
  conditioning (n=2) and Transition fp32 at 1.1-1.9x of compiled PyTorch, and the wide bf16 Transitions (D256-D768 training).
* **CUDA vs PyTorch** (implied by the chain; checked here for every three-rung row): 48 rows, mostly the fp32 AdaLN /
  ConditionedTransition / Transition-conditioning rows at small L, RMSNorm, token DiT with an extra pair bias (training,
  1.14-1.16x) and Transition D768 training L384 (CUDA 8.76 ms vs compiled 6.37 ms).
* **Ops missing a level on B200** (the `miniworld` rung ran the PyTorch statements): MiniWorld's **pair-D64 OuterProductMean and
  MSAPairWeightedAveraging** (inference and training; the Triton kernels would be 1.4-2.2x faster in inference),
  AugmentedAttention atom (fp32, and bf16 training), AugmentedAttention token fp32 training at the ESMFold2 width, TriMul
  bidirectional hidden != D training, bias-only DiT with a 256-wide pair, token DiT with a 256-wide pair (Protenix-v2), and
  ESMFold2's LocalDiT atom block. The ladder decides this from the kernels the `miniworld` rung launched; where that rung is
  then far below the Triton time (token DiT pair256 bf16, LocalDiT ESMFold2) it is compiled PyTorch beating Triton.
  SWA atom attention (global) has no Triton rung by design. AugmentedAttention atom fp32 training at L1024 ran out of memory
  (a 48 GiB allocation) on the CUDA and PyTorch rungs.
* **Triton bug found while tuning**: the MSAPairWeightedAveraging Triton backward raises `CUDA error: misaligned address` at
  d_pair 384 on sm_100 (see the tuning section). No kernel was changed for this page.

## Summary

| rows | OK (ladder verdict) | CUDA >= Triton | Triton >= PyTorch | CUDA >= PyTorch | no CUDA path (routing gap) | other rung missing |
|---|---|---|---|---|---|---|
| 432 | 239 | 39 | 132 | 48 | 43 | 1 |

## Violations (sorted by production weight, then ratio)

weight 3 = a shape MiniWorld / FoldForge runs, 2 = served, 1 = exotic. `ratio` = slower-than-allowed rung / the rung it must beat.

| # | weight | op | mode | dtype | shape | L | pair | ratio | CUDA us | Triton us | PyTorch compiled us | PyTorch best us (form) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | token_pair_init | inference | fp32 in, fp32 out | B16 | 128 | CUDA >= Triton | 1.40 | 71.5 | 51.0 | 444.5 | 444.5 (pt_compiled) |  |
| 2 | 3 | pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 256 | Triton >= PyTorch | 1.24 | 1079 | 5042 | 4057 | 4057 (pt_compiled) |  |
| 3 | 3 | token_pair_init | inference | fp32 in, fp32 out | B16 | 256 | CUDA >= Triton | 1.20 | 203.6 | 169.9 | 1355 | 1355 (pt_compiled) |  |
| 4 | 3 | Transition | training | bf16 | D128 | 1024 | Triton >= PyTorch | 1.16 | 1905 | 4249 | 3672 | 3672 (pt_compiled) |  |
| 5 | 3 | Transition | training | bf16 | D128 | 768 | Triton >= PyTorch | 1.15 | 1049 | 2430 | 2118 | 2118 (pt_compiled) |  |
| 6 | 3 | pair_embed_adaln | training | bf16 out, fp32 geometry | B16 | 384 | Triton >= PyTorch | 1.12 | 3198 | 8108 | 7212 | 7212 (pt_compiled) |  |
| 7 | 3 | Transition | training | bf16 | D128 | 384 | Triton >= PyTorch | 1.11 | 270.4 | 662.4 | 597.9 | 597.9 (pt_compiled) |  |
| 8 | 3 | pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 384 | Triton >= PyTorch | 1.08 | 2352 | 9667 | 8959 | 8959 (pt_compiled) |  |
| 9 | 3 | MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair128 | 256 | Triton >= PyTorch | 1.07 | 438.4 | 976.0 | 913.2 | 913.2 (pt_compiled) |  |
| 10 | 3 | pair_embed_ca | inference | bf16 out, fp32 geometry | B16 | 128 | CUDA >= Triton | 1.06 | 36.7 | 34.6 | 959.6 | 899.0 (pt_eager_graph) |  |
| 11 | 3 | AttentionPairBias | inference | bf16 | 8x48 pair128 | 384 | Triton >= PyTorch | 1.06 | 32.9 | 75.6 | 71.5 | 71.5 (pt_compiled) |  |
| 12 | 3 | LayerNorm (pair) | training | bf16 | x[1,L,L,128] | 768 | Triton >= PyTorch | 1.06 | 236.3 | 194.5 | 184.3 | 184.3 (pt_compiled) |  |
| 13 | 3 | Transition | training | bf16 | D64 | 768 | Triton >= PyTorch | 1.05 | 396.1 | 1132 | 1074 | 1074 (pt_compiled) |  |
| 14 | 3 | smooth_lddt | inference | fp32 | A48 B1 | 1024 | CUDA >= Triton | 1.04 | 172.0 | 165.9 | 60503 | 60503 (pt_compiled) |  |
| 15 | 3 | AttentionPairBias | inference | bf16 | 8x48 pair128 | 768 | Triton >= PyTorch | 1.03 | 57.3 | 151.4 | 147.1 | 147.1 (pt_compiled) |  |
| 16 | 3 | MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair128 | 384 | Triton >= PyTorch | 1.03 | 662.3 | 1408 | 1371 | 1371 (pt_compiled) |  |
| 17 | 3 | MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair64 | 384 | Triton >= PyTorch | 1.02 | 1368 | 1379 | 1347 | 1347 (pt_compiled) |  |
| 18 | 3 | OuterProductMean | training | bf16 | S1024 msa64 pair128 | 256 | Triton >= PyTorch | 1.02 | 444.4 | 670.6 | 659.5 | 659.5 (pt_compiled) | within noise |
| 19 | 3 | MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair128 | 768 | Triton >= PyTorch | 1.01 | 1470 | 2951 | 2926 | 2926 (pt_compiled) | within noise |
| 20 | 3 | token_pair_init | inference | fp32 in, fp32 out | B16 | 384 | CUDA >= Triton | 1.01 | 372.4 | 370.5 | 3061 | 3061 (pt_compiled) | within noise |
| 21 | 2 | TriMul bidirectional hidden!=D | inference | bf16 | D64 hidden128 | 384 | CUDA >= Triton | 2.61 | 603.7 | 231.5 | 661.6 | 661.6 (pt_compiled) |  |
| 22 | 2 | ConditionedTransition | inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | Triton >= PyTorch | 1.93 | 84.0 | 213.0 | 110.6 | 110.6 (pt_compiled) |  |
| 23 | 2 | ConditionedTransition | inference | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | Triton >= PyTorch | 1.75 | 59.4 | 128.9 | 73.7 | 73.7 (pt_compiled) |  |
| 24 | 2 | AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | Triton >= PyTorch | 1.68 | 46.8 | 75.7 | 45.0 | 45.0 (pt_compiled) |  |
| 25 | 2 | RMSNorm | inference | bf16 | x[48,L,768] | 384 | CUDA >= Triton | 1.64 | 26.5 | 16.1 | 16.3 | 16.3 (pt_compiled) | within noise |
| 26 | 2 | RMSNorm | inference | bf16 | x[48,L,768] | 1024 | CUDA >= Triton | 1.63 | 53.2 | 32.6 | 32.7 | 32.7 (pt_compiled) | within noise |
| 27 | 2 | RMSNorm | inference | bf16 | x[48,L,768] | 1024 | CUDA >= PyTorch | 1.63 | 53.2 | 32.6 | 32.7 | 32.7 (pt_compiled) | within noise |
| 28 | 2 | RMSNorm | inference | bf16 | x[48,L,768] | 384 | CUDA >= PyTorch | 1.63 | 26.5 | 16.1 | 16.3 | 16.3 (pt_compiled) | within noise |
| 29 | 2 | RMSNorm | training | bf16 | x[48,L,768] | 1024 | CUDA >= Triton | 1.59 | 120.8 | 75.9 | 100.3 | 100.3 (pt_compiled) |  |
| 30 | 2 | AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | Triton >= PyTorch | 1.58 | 26.6 | 38.9 | 24.6 | 24.6 (pt_compiled) |  |
| 31 | 2 | RMSNorm | training | bf16 | x[48,L,768] | 384 | CUDA >= Triton | 1.53 | 59.4 | 38.9 | 49.1 | 49.1 (pt_compiled) |  |
| 32 | 2 | AdaLN | inference | fp32 | d768 cond384 A5/A48 | 768 | Triton >= PyTorch | 1.45 | 41.0 | 50.7 | 34.9 | 34.9 (pt_compiled) |  |
| 33 | 2 | AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | Triton >= PyTorch | 1.42 | 41.0 | 55.3 | 38.9 | 38.9 (pt_compiled) |  |
| 34 | 2 | Transition | training | bf16 | D768 | 384 | CUDA >= PyTorch | 1.38 | 8763 | 7929 | 6372 | 6372 (pt_compiled) |  |
| 35 | 2 | AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | Triton >= PyTorch | 1.35 | 30.7 | 38.8 | 28.7 | 28.7 (pt_compiled) |  |
| 36 | 2 | AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | Triton >= PyTorch | 1.34 | 382.8 | 530.3 | 395.1 | 395.1 (pt_compiled) |  |
| 37 | 2 | AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | Triton >= PyTorch | 1.34 | 692.1 | 989.1 | 739.2 | 739.2 (pt_compiled) |  |
| 38 | 2 | Transition conditioning (pair) | inference | fp32 | D256 n2 TF32 | 768 | Triton >= PyTorch | 1.32 | 1326 | 2464 | 1861 | 1861 (pt_compiled) |  |
| 39 | 2 | token_pair_init | inference | fp32 in, fp32 out | B48 | 128 | CUDA >= Triton | 1.32 | 176.0 | 132.9 | 1022 | 1022 (pt_compiled) |  |
| 40 | 2 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 768 | Triton >= PyTorch | 1.32 | 485.3 | 690.1 | 524.2 | 524.2 (pt_compiled) |  |
| 41 | 2 | Transition conditioning (pair) | inference | fp32 | D256 n2 TF32 | 384 | Triton >= PyTorch | 1.32 | 339.8 | 638.9 | 485.3 | 485.3 (pt_compiled) |  |
| 42 | 2 | AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | Triton >= PyTorch | 1.31 | 34.8 | 42.9 | 32.7 | 32.7 (pt_compiled) |  |
| 43 | 2 | Transition conditioning (pair) | inference | fp32 | D256 n2 TF32 | 128 | Triton >= PyTorch | 1.30 | 55.2 | 87.9 | 67.6 | 67.6 (pt_compiled) |  |
| 44 | 2 | Transition | training | bf16 | D384 | 768 | Triton >= PyTorch | 1.29 | 6871 | 10665 | 8263 | 8263 (pt_compiled) |  |
| 45 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 128 | Triton >= PyTorch | 1.29 | 81.9 | 92.1 | 71.7 | 71.7 (pt_compiled) |  |
| 46 | 2 | ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | Triton >= PyTorch | 1.28 | 2686 | 3723 | 2899 | 2899 (pt_compiled) |  |
| 47 | 2 | AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | Triton >= PyTorch | 1.28 | 171.9 | 214.9 | 167.8 | 167.8 (pt_compiled) |  |
| 48 | 2 | ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | Triton >= PyTorch | 1.28 | 456.0 | 568.4 | 444.4 | 444.4 (pt_compiled) |  |
| 49 | 2 | Transition | training | bf16 | D384 | 384 | Triton >= PyTorch | 1.28 | 1739 | 2708 | 2122 | 2122 (pt_compiled) |  |
| 50 | 2 | AdaLN | inference | fp32 | d768 cond384 A5/A48 | 128 | CUDA >= PyTorch | 1.27 | 28.6 | 26.6 | 22.5 | 22.5 (pt_compiled) | within noise |
| 51 | 2 | Transition conditioning (pair) | training | fp32 | D256 n2 TF32 | 384 | Triton >= PyTorch | 1.26 | 1514 | 1756 | 1389 | 1389 (pt_compiled) |  |
| 52 | 2 | ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 128 | CUDA >= PyTorch | 1.26 | 69.7 | 55.3 | 55.1 | 55.1 (pt_compiled) | within noise |
| 53 | 2 | ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | Triton >= PyTorch | 1.26 | 1461 | 1981 | 1568 | 1568 (pt_compiled) |  |
| 54 | 2 | ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 128 | CUDA >= Triton | 1.26 | 69.7 | 55.3 | 55.1 | 55.1 (pt_compiled) | within noise |
| 55 | 2 | MSAPairWeightedAveraging value_dim 8 | training | bf16 | S512 8x8 pair128 | 384 | Triton >= PyTorch | 1.25 | 289.7 | 578.6 | 461.9 | 461.9 (pt_compiled) |  |
| 56 | 2 | Transition | training | bf16 | D768 | 384 | Triton >= PyTorch | 1.24 | 8763 | 7929 | 6372 | 6372 (pt_compiled) |  |
| 57 | 2 | Transition | training | bf16 | D256 | 768 | Triton >= PyTorch | 1.24 | 3767 | 5574 | 4487 | 4487 (pt_compiled) |  |
| 58 | 2 | ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 384 | CUDA >= PyTorch | 1.23 | 98.3 | 79.8 | 79.7 | 79.7 (pt_compiled) | within noise |
| 59 | 2 | ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 384 | CUDA >= Triton | 1.23 | 98.3 | 79.8 | 79.7 | 79.7 (pt_compiled) | within noise |
| 60 | 2 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 128 | Triton >= PyTorch | 1.23 | 133.1 | 153.5 | 124.9 | 124.9 (pt_compiled) |  |
| 61 | 2 | Transition conditioning (pair) | training | fp32 | D256 n2 TF32 | 768 | Triton >= PyTorch | 1.23 | 5375 | 6198 | 5042 | 5042 (pt_compiled) |  |
| 62 | 2 | ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | Triton >= PyTorch | 1.23 | 133.1 | 155.7 | 127.0 | 127.0 (pt_compiled) |  |
| 63 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 768 | Triton >= PyTorch | 1.22 | 34.8 | 45.1 | 36.8 | 36.8 (pt_compiled) |  |
| 64 | 2 | MSAPairWeightedAveraging value_dim 8 | training | bf16 | S512 8x8 pair128 | 256 | Triton >= PyTorch | 1.22 | 214.0 | 404.3 | 330.8 | 330.8 (pt_compiled) |  |
| 65 | 2 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 384 | Triton >= PyTorch | 1.21 | 298.9 | 365.7 | 301.1 | 301.1 (pt_compiled) |  |
| 66 | 2 | RMSNorm | training | bf16 | x[48,L,768] | 384 | CUDA >= PyTorch | 1.21 | 59.4 | 38.9 | 49.1 | 49.1 (pt_compiled) |  |
| 67 | 2 | ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA >= PyTorch | 1.21 | 71.7 | 67.7 | 59.2 | 59.2 (pt_compiled) |  |
| 68 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 768 | Triton >= PyTorch | 1.21 | 102.5 | 118.8 | 98.3 | 98.3 (pt_compiled) |  |
| 69 | 2 | RMSNorm | training | bf16 | x[48,L,768] | 1024 | CUDA >= PyTorch | 1.20 | 120.8 | 75.9 | 100.3 | 100.3 (pt_compiled) |  |
| 70 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 128 | Triton >= PyTorch | 1.20 | 32.7 | 36.8 | 30.7 | 30.7 (pt_compiled) |  |
| 71 | 2 | AttentionPairBias Chai-1 (gate bias 1.0) | inference | bf16 | 16x24 pair256 | 384 | Triton >= PyTorch | 1.19 | 43.0 | 102.2 | 86.0 | 86.0 (pt_compiled) |  |
| 72 | 2 | ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | CUDA >= PyTorch | 1.19 | 104.5 | 93.9 | 88.0 | 88.0 (pt_compiled) |  |
| 73 | 2 | Transition | training | bf16 | D256 | 384 | Triton >= PyTorch | 1.18 | 945.8 | 1432 | 1211 | 1211 (pt_compiled) |  |
| 74 | 2 | AdaLN | inference | fp32 | d768 cond384 A5/A48 | 128 | Triton >= PyTorch | 1.18 | 28.6 | 26.6 | 22.5 | 22.5 (pt_compiled) | within noise |
| 75 | 2 | AdaLN | inference | fp32 | d768 cond384 A5/A48 | 768 | CUDA >= PyTorch | 1.17 | 41.0 | 50.7 | 34.9 | 34.9 (pt_compiled) |  |
| 76 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 384 | Triton >= PyTorch | 1.17 | 88.2 | 98.2 | 84.0 | 84.0 (pt_compiled) |  |
| 77 | 2 | LayerNorm (pair) | training | bf16 | x[1,L,L,256] | 384 | Triton >= PyTorch | 1.16 | 108.4 | 116.6 | 100.3 | 100.3 (pt_compiled) |  |
| 78 | 2 | token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 512 | CUDA >= PyTorch | 1.16 | 591.8 | 550.9 | 512.0 | 512.0 (pt_compiled) |  |
| 79 | 2 | AdaLN | inference | fp32 | d768 cond384 A5/A48 | 384 | CUDA >= PyTorch | 1.15 | 33.1 | 30.7 | 28.7 | 28.7 (pt_compiled) |  |
| 80 | 2 | AttentionPairBias Chai-1 (gate bias 1.0) | inference | bf16 | 16x24 pair256 | 768 | Triton >= PyTorch | 1.15 | 90.2 | 251.7 | 219.0 | 219.0 (pt_compiled) |  |
| 81 | 2 | ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | Triton >= PyTorch | 1.14 | 71.7 | 67.7 | 59.2 | 59.2 (pt_compiled) |  |
| 82 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 128 | CUDA >= PyTorch | 1.14 | 81.9 | 92.1 | 71.7 | 71.7 (pt_compiled) |  |
| 83 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | Triton >= PyTorch | 1.14 | 628.7 | 685.1 | 600.1 | 600.1 (pt_compiled) |  |
| 84 | 2 | Transition conditioning (single) | training | fp32 | D384 n2 TF32 | 128 | Triton >= PyTorch | 1.14 | 59.3 | 65.4 | 57.3 | 57.3 (pt_compiled) | within noise |
| 85 | 2 | token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | CUDA >= PyTorch | 1.14 | 415.7 | 374.5 | 364.6 | 364.6 (pt_compiled) | within noise |
| 86 | 2 | token_pair_init | inference | fp32 in, fp32 out | B48 | 256 | CUDA >= Triton | 1.14 | 559.1 | 491.3 | 3967 | 3967 (pt_compiled) |  |
| 87 | 2 | ConditionedTransition | inference | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | Triton >= PyTorch | 1.14 | 38.9 | 51.2 | 45.0 | 45.0 (pt_compiled) |  |
| 88 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 1024 | Triton >= PyTorch | 1.14 | 1452 | 1638 | 1442 | 1442 (pt_compiled) |  |
| 89 | 2 | Transition | inference | bf16 | D768 | 384 | Triton >= PyTorch | 1.14 | 2142 | 2445 | 2153 | 2153 (pt_compiled) | within noise |
| 90 | 2 | AugmentedAttention token | inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | CUDA >= Triton | 1.13 | 697.4 | 616.3 | 654.4 | 654.4 (pt_compiled) |  |
| 91 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 384 | Triton >= PyTorch | 1.12 | 32.7 | 36.7 | 32.7 | 32.7 (pt_compiled) |  |
| 92 | 2 | bias-only DiT | training | bf16 | d768 16x48 per-sample pair256 A4 | 512 | Triton >= PyTorch | 1.12 | 1305 | 1330 | 1183 | 1183 (pt_compiled) |  |
| 93 | 2 | LocalDiT atom block | inference | fp32 | d128 32x128 (atoms=8L) | 512 | Triton >= PyTorch | 1.12 | 118.7 | 339.9 | 303.1 | 303.1 (pt_compiled) |  |
| 94 | 2 | Transition conditioning (pair) | training | fp32 | D256 n2 TF32 | 128 | Triton >= PyTorch | 1.12 | 227.1 | 257.0 | 229.4 | 229.4 (pt_compiled) |  |
| 95 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 768 | Triton >= PyTorch | 1.12 | 1124 | 1253 | 1122 | 1122 (pt_compiled) |  |
| 96 | 2 | AdaLN | inference | bf16 | d768 cond384 A5/A48 | 384 | CUDA >= Triton | 1.12 | 20.4 | 18.3 | 24.4 | 24.4 (pt_compiled) |  |
| 97 | 2 | ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | CUDA >= Triton | 1.11 | 104.5 | 93.9 | 88.0 | 88.0 (pt_compiled) |  |
| 98 | 2 | AdaLN | inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA >= Triton | 1.11 | 20.4 | 18.3 | 22.5 | 22.5 (pt_compiled) |  |
| 99 | 2 | token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | CUDA >= Triton | 1.11 | 415.7 | 374.5 | 364.6 | 364.6 (pt_compiled) | within noise |
| 100 | 2 | ConditionedTransition | inference | bf16 | d768 A5/A48 | 768 | Triton >= PyTorch | 1.11 | 69.7 | 83.9 | 75.7 | 75.7 (pt_compiled) |  |
| 101 | 2 | Transition | inference | fp32 | D128 TF32 | 384 | Triton >= PyTorch | 1.11 | 217.0 | 425.5 | 385.0 | 385.0 (pt_compiled) |  |
| 102 | 2 | Transition | training | bf16 | D768 | 384 | CUDA >= Triton | 1.11 | 8763 | 7929 | 6372 | 6372 (pt_compiled) |  |
| 103 | 2 | AugmentedAttention token | inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | Triton >= PyTorch | 1.10 | 104.5 | 110.5 | 100.2 | 100.2 (pt_compiled) |  |
| 104 | 2 | Transition | inference | fp32 | D128 TF32 | 768 | Triton >= PyTorch | 1.10 | 792.5 | 1602 | 1454 | 1454 (pt_compiled) |  |
| 105 | 2 | AugmentedAttention token | inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | CUDA >= Triton | 1.10 | 257.0 | 233.3 | 245.7 | 245.7 (pt_compiled) |  |
| 106 | 2 | OuterProductMean Chai-1 (8 groups) | training | bf16 | S512 pair256 8x64 | 256 | Triton >= PyTorch | 1.09 | 331.7 | 552.8 | 505.9 | 505.9 (pt_compiled) |  |
| 107 | 2 | AdaLN | inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 384 | CUDA >= Triton | 1.09 | 24.4 | 22.4 | 26.6 | 26.6 (pt_compiled) |  |
| 108 | 2 | Transition conditioning (pair) | training | fp32 | D256 n2 TF32 | 384 | CUDA >= PyTorch | 1.09 | 1514 | 1756 | 1389 | 1389 (pt_compiled) |  |
| 109 | 2 | Transition | inference | bf16 | D384 | 768 | Triton >= PyTorch | 1.09 | 1613 | 3339 | 3063 | 3063 (pt_compiled) |  |
| 110 | 2 | ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 768 | Triton >= PyTorch | 1.09 | 120.9 | 125.0 | 114.7 | 114.7 (pt_compiled) |  |
| 111 | 2 | AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA >= PyTorch | 1.08 | 26.6 | 38.9 | 24.6 | 24.6 (pt_compiled) |  |
| 112 | 2 | LayerNorm (pair) | training | bf16 | x[1,L,L,256] | 384 | CUDA >= PyTorch | 1.08 | 108.4 | 116.6 | 100.3 | 100.3 (pt_compiled) |  |
| 113 | 2 | Transition | inference | bf16 | D384 | 384 | Triton >= PyTorch | 1.08 | 407.5 | 830.0 | 768.1 | 768.1 (pt_compiled) |  |
| 114 | 2 | pair_embed_geom | training | bf16 out, fp32 geometry | B48 | 256 | Triton >= PyTorch | 1.08 | 2954 | 12816 | 11868 | 11868 (pt_compiled) |  |
| 115 | 2 | AdaLN | inference | fp32 | d768 cond384 A5/A48 | 128 | CUDA >= Triton | 1.08 | 28.6 | 26.6 | 22.5 | 22.5 (pt_compiled) | within noise |
| 116 | 2 | AdaLN | inference | fp32 | d768 cond384 A5/A48 | 384 | CUDA >= Triton | 1.08 | 33.1 | 30.7 | 28.7 | 28.7 (pt_compiled) |  |
| 117 | 2 | token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 512 | Triton >= PyTorch | 1.08 | 591.8 | 550.9 | 512.0 | 512.0 (pt_compiled) |  |
| 118 | 2 | ConditionedTransition | inference | bf16 | d768 A5/A48 | 384 | CUDA >= Triton | 1.08 | 57.3 | 53.3 | 61.4 | 61.4 (pt_compiled) |  |
| 119 | 2 | Transition | training | fp32 | D128 TF32 | 768 | Triton >= PyTorch | 1.07 | 3387 | 4301 | 4003 | 4003 (pt_compiled) |  |
| 120 | 2 | token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 512 | CUDA >= Triton | 1.07 | 591.8 | 550.9 | 512.0 | 512.0 (pt_compiled) |  |
| 121 | 2 | ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | Triton >= PyTorch | 1.07 | 552.7 | 571.3 | 532.5 | 532.5 (pt_compiled) |  |
| 122 | 2 | ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | Triton >= PyTorch | 1.07 | 1332 | 1459 | 1361 | 1361 (pt_compiled) |  |
| 123 | 2 | MSAPairWeightedAveraging value_dim 8 | inference | bf16 | S512 8x8 pair128 | 384 | Triton >= PyTorch | 1.07 | 81.8 | 139.2 | 130.0 | 130.0 (pt_compiled) |  |
| 124 | 2 | pair_embed_geom | training | bf16 out, fp32 geometry | B48 | 128 | Triton >= PyTorch | 1.07 | 825.2 | 3309 | 3089 | 3089 (pt_compiled) |  |
| 125 | 2 | AdaLN | inference | fp32 | d768 cond384 A5/A48 | 384 | Triton >= PyTorch | 1.07 | 33.1 | 30.7 | 28.7 | 28.7 (pt_compiled) |  |
| 126 | 2 | bias-only DiT | training | bf16 | d768 16x48 per-sample pair256 A4 | 256 | Triton >= PyTorch | 1.07 | 519.0 | 536.6 | 501.9 | 501.9 (pt_compiled) |  |
| 127 | 2 | AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | CUDA >= PyTorch | 1.07 | 30.7 | 38.8 | 28.7 | 28.7 (pt_compiled) |  |
| 128 | 2 | ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | Triton >= PyTorch | 1.07 | 104.5 | 93.9 | 88.0 | 88.0 (pt_compiled) |  |
| 129 | 2 | ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | Triton >= PyTorch | 1.07 | 2481 | 2696 | 2528 | 2528 (pt_compiled) |  |
| 130 | 2 | Transition conditioning (pair) | training | fp32 | D256 n2 TF32 | 768 | CUDA >= PyTorch | 1.07 | 5375 | 6198 | 5042 | 5042 (pt_compiled) |  |
| 131 | 2 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 128 | CUDA >= PyTorch | 1.07 | 133.1 | 153.5 | 124.9 | 124.9 (pt_compiled) |  |
| 132 | 2 | AugmentedAttention token | inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | CUDA >= PyTorch | 1.07 | 697.4 | 616.3 | 654.4 | 654.4 (pt_compiled) |  |
| 133 | 2 | AdaLN | inference | bf16 | d768 cond384 A5/A48 | 1024 | CUDA >= Triton | 1.06 | 32.6 | 30.6 | 32.8 | 32.8 (pt_compiled) |  |
| 134 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 128 | CUDA >= PyTorch | 1.06 | 32.7 | 36.8 | 30.7 | 30.7 (pt_compiled) |  |
| 135 | 2 | AdaLN | inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 768 | Triton >= PyTorch | 1.06 | 30.6 | 32.6 | 30.7 | 30.7 (pt_compiled) |  |
| 136 | 2 | ConditionedTransition | inference | bf16 | d768 A5/A48 | 1024 | Triton >= PyTorch | 1.06 | 79.9 | 100.3 | 94.4 | 94.4 (pt_compiled) |  |
| 137 | 2 | AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | CUDA >= PyTorch | 1.06 | 34.8 | 42.9 | 32.7 | 32.7 (pt_compiled) |  |
| 138 | 2 | ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA >= Triton | 1.06 | 71.7 | 67.7 | 59.2 | 59.2 (pt_compiled) |  |
| 139 | 2 | Transition conditioning (pair) | inference | fp32 | D128 n2 TF32 | 384 | Triton >= PyTorch | 1.06 | 147.3 | 257.9 | 243.8 | 243.8 (pt_compiled) |  |
| 140 | 2 | AdaLN | inference | bf16 | d128 cond128 A5/A48 (atoms) | 8192 | Triton >= PyTorch | 1.06 | 30.8 | 34.6 | 32.7 | 32.7 (pt_compiled) |  |
| 141 | 2 | AttentionPairBias | inference | bf16 | 16x24 pair128 | 384 | Triton >= PyTorch | 1.06 | 34.8 | 75.7 | 71.7 | 71.7 (pt_compiled) |  |
| 142 | 2 | ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 768 | CUDA >= PyTorch | 1.05 | 120.9 | 125.0 | 114.7 | 114.7 (pt_compiled) |  |
| 143 | 2 | AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | CUDA >= PyTorch | 1.05 | 41.0 | 55.3 | 38.9 | 38.9 (pt_compiled) |  |
| 144 | 2 | ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 128 | CUDA >= PyTorch | 1.05 | 497.7 | 497.6 | 473.2 | 473.2 (pt_compiled) | within noise |
| 145 | 2 | ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 128 | Triton >= PyTorch | 1.05 | 497.7 | 497.6 | 473.2 | 473.2 (pt_compiled) | within noise |
| 146 | 2 | OuterProductMean Chai-1 (8 groups) | training | bf16 | S512 pair256 8x64 | 384 | Triton >= PyTorch | 1.05 | 584.9 | 960.5 | 914.4 | 914.4 (pt_compiled) |  |
| 147 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 384 | CUDA >= PyTorch | 1.05 | 88.2 | 98.2 | 84.0 | 84.0 (pt_compiled) |  |
| 148 | 2 | ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | CUDA >= PyTorch | 1.05 | 133.1 | 155.7 | 127.0 | 127.0 (pt_compiled) |  |
| 149 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | CUDA >= PyTorch | 1.05 | 628.7 | 685.1 | 600.1 | 600.1 (pt_compiled) |  |
| 150 | 2 | AugmentedAttention token | inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | CUDA >= PyTorch | 1.05 | 257.0 | 233.3 | 245.7 | 245.7 (pt_compiled) |  |
| 151 | 2 | MSAPairWeightedAveraging value_dim 8 | inference | bf16 | S512 8x8 pair128 | 256 | Triton >= PyTorch | 1.04 | 59.4 | 98.3 | 94.1 | 94.1 (pt_compiled) |  |
| 152 | 2 | ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 768 | Triton >= PyTorch | 1.04 | 2193 | 2337 | 2241 | 2241 (pt_compiled) |  |
| 153 | 2 | AugmentedAttention token | inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | CUDA >= PyTorch | 1.04 | 104.5 | 110.5 | 100.2 | 100.2 (pt_compiled) |  |
| 154 | 2 | Transition conditioning (single) | training | fp32 | D768 n2 TF32 | 768 | CUDA >= PyTorch | 1.04 | 102.5 | 118.8 | 98.3 | 98.3 (pt_compiled) |  |
| 155 | 2 | AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | CUDA >= PyTorch | 1.04 | 46.8 | 75.7 | 45.0 | 45.0 (pt_compiled) |  |
| 156 | 2 | AdaLN | training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | Triton >= PyTorch | 1.04 | 147.3 | 155.5 | 149.5 | 149.5 (pt_compiled) |  |
| 157 | 2 | Transition conditioning (pair) | inference | fp32 | D128 n2 TF32 | 768 | Triton >= PyTorch | 1.04 | 520.1 | 952.3 | 916.4 | 916.4 (pt_compiled) |  |
| 158 | 2 | ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA >= PyTorch | 1.04 | 552.7 | 571.3 | 532.5 | 532.5 (pt_compiled) |  |
| 159 | 2 | bias-only DiT | training | bf16 | d768 16x48 pair256 A5/A48 | 768 | Triton >= PyTorch | 1.03 | 2738 | 2899 | 2801 | 2801 (pt_compiled) |  |
| 160 | 2 | bias-only DiT | training | bf16 | d768 16x48 per-sample pair256 A4 | 256 | CUDA >= PyTorch | 1.03 | 519.0 | 536.6 | 501.9 | 501.9 (pt_compiled) |  |
| 161 | 2 | Transition conditioning (single) | training | fp32 | D384 n2 TF32 | 128 | CUDA >= PyTorch | 1.03 | 59.3 | 65.4 | 57.3 | 57.3 (pt_compiled) | within noise |
| 162 | 2 | Transition | training | fp32 | D128 TF32 | 384 | Triton >= PyTorch | 1.03 | 913.4 | 1155 | 1118 | 1118 (pt_compiled) |  |
| 163 | 2 | bias-only DiT | training | fp32 | d768 16x48 pair256 A5/A48 | 512 | Triton >= PyTorch | 1.03 | 3364 | 3518 | 3415 | 3415 (pt_compiled) |  |
| 164 | 2 | Transition | inference | bf16 | D256 | 384 | Triton >= PyTorch | 1.03 | 190.2 | 440.5 | 428.0 | 428.0 (pt_compiled) | within noise |
| 165 | 2 | Transition conditioning (single) | training | fp32 | D384 n2 TF32 | 384 | Triton >= PyTorch | 1.03 | 63.4 | 69.5 | 67.5 | 67.5 (pt_compiled) |  |
| 166 | 2 | token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | Triton >= PyTorch | 1.03 | 415.7 | 374.5 | 364.6 | 364.6 (pt_compiled) | within noise |
| 167 | 2 | ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | CUDA >= PyTorch | 1.03 | 456.0 | 568.4 | 444.4 | 444.4 (pt_compiled) |  |
| 168 | 2 | ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 384 | Triton >= PyTorch | 1.02 | 1207 | 1263 | 1232 | 1232 (pt_compiled) |  |
| 169 | 2 | AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | CUDA >= PyTorch | 1.02 | 171.9 | 214.9 | 167.8 | 167.8 (pt_compiled) |  |
| 170 | 2 | bias-only DiT | training | fp32 | d768 16x48 pair256 A5/A48 | 256 | Triton >= PyTorch | 1.02 | 1678 | 1768 | 1728 | 1728 (pt_compiled) |  |
| 171 | 2 | OuterProductMean RF3 (bias, divide before) | training | bf16 | S512 msa64 pair128 | 256 | Triton >= PyTorch | 1.02 | 315.0 | 489.5 | 481.4 | 481.4 (pt_compiled) | within noise |
| 172 | 2 | Transition conditioning (pair) | training | fp32 | D128 n2 TF32 | 768 | Triton >= PyTorch | 1.01 | 2092 | 2520 | 2485 | 2485 (pt_compiled) | within noise |
| 173 | 2 | bias-only DiT | training | bf16 | d768 16x48 pair256 A5/A48 | 256 | Triton >= PyTorch | 1.01 | 909.2 | 961.6 | 949.3 | 949.3 (pt_compiled) | within noise |
| 174 | 2 | bias-only DiT | training | bf16 | d768 16x48 pair256 A5/A48 | 512 | Triton >= PyTorch | 1.01 | 1760 | 1850 | 1828 | 1828 (pt_compiled) | within noise |
| 175 | 2 | AugmentedAttention token | training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | Triton >= PyTorch | 1.01 | 882.7 | 872.6 | 862.4 | 862.4 (pt_compiled) | within noise |
| 176 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 1024 | CUDA >= PyTorch | 1.01 | 1452 | 1638 | 1442 | 1442 (pt_compiled) |  |
| 177 | 2 | AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | CUDA >= Triton | 1.01 | 18.5 | 18.4 | 20.4 | 20.4 (pt_compiled) | within noise |
| 178 | 2 | LayerNorm (pair) | inference | bf16 | x[1,L,L,256] | 384 | Triton >= PyTorch | 1.00 | 30.6 | 28.7 | 28.6 | 28.6 (pt_compiled) | within noise |
| 179 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 768 | CUDA >= PyTorch | 1.00 | 1124 | 1253 | 1122 | 1122 (pt_compiled) |  |
| 180 | 2 | bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 512 | Triton >= PyTorch | 1.00 | 2729 | 3272 | 3264 | 3264 (pt_compiled) | within noise |
| 181 | 2 | ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 128 | Triton >= PyTorch | 1.00 | 69.7 | 55.3 | 55.1 | 55.1 (pt_compiled) | within noise |
| 182 | 2 | ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 384 | Triton >= PyTorch | 1.00 | 98.3 | 79.8 | 79.7 | 79.7 (pt_compiled) | within noise |
| 183 | 2 | AugmentedAttention token | training | fp32 | d768 cond384 pair128 A5/A48 | 128 | Triton >= PyTorch | 1.00 | 667.7 | 782.3 | 781.3 | 781.3 (pt_compiled) | within noise |
| 184 | 2 | Transition conditioning (pair) | training | fp32 | D128 n2 TF32 | 384 | Triton >= PyTorch | 1.00 | 597.9 | 715.0 | 714.8 | 714.8 (pt_compiled) | within noise |
| 185 | 2 | ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 128 | CUDA >= Triton | 1.00 | 497.7 | 497.6 | 473.2 | 473.2 (pt_compiled) | within noise |
| 186 | 2 | Transition conditioning (pair) | training | fp32 | D128 n2 TF32 | 128 | Triton >= PyTorch | 1.00 | 112.5 | 122.8 | 122.8 | 122.8 (pt_compiled) | within noise |
| 187 | 2 | Transition conditioning (single) | inference | fp32 | D384 n2 TF32 | 128 | Triton >= PyTorch | 1.00 | 22.4 | 26.5 | 26.5 | 26.5 (pt_compiled) | within noise |
| 188 | 2 | AdaLN | inference | bf16 | d768 cond384 A5/A48 | 768 | CUDA >= Triton | 1.00 | 26.5 | 26.5 | 28.5 | 28.5 (pt_compiled) | within noise |
| 189 | 2 | AdaLN | inference | bf16 | d128 cond128 A5/A48 (atoms) | 1024 | CUDA >= Triton | 1.00 | 14.1 | 14.1 | 20.4 | 20.4 (pt_compiled) | within noise |
| 190 | 2 | pair_embed_ca | inference | bf16 out, fp32 geometry | B48 | 128 | CUDA >= Triton | 1.00 | 79.7 | 79.7 | 706.4 | 706.4 (pt_compiled) | within noise |
| 191 | 2 | Transition conditioning (single) | inference | fp32 | D768 n2 TF32 | 384 | CUDA >= PyTorch | 1.00 | 32.7 | 36.7 | 32.7 | 32.7 (pt_compiled) |  |
| 192 | 1 | AugmentedAttention atom | inference | bf16 | d128 A5/A48 | 384 | CUDA >= Triton | 1.39 | 1261 | 906.2 | 1157 | 1157 (pt_compiled) |  |
| 193 | 1 | Transition | inference | fp32 | D256 TF32 | 384 | Triton >= PyTorch | 1.39 | 565.2 | 1115 | 802.9 | 802.9 (pt_compiled) |  |
| 194 | 1 | Transition | training | bf16 | D512 | 384 | Triton >= PyTorch | 1.30 | 2839 | 4104 | 3149 | 3149 (pt_compiled) |  |
| 195 | 1 | AugmentedAttention atom | inference | bf16 | d128 A5/A48 | 128 | CUDA >= Triton | 1.22 | 194.4 | 159.7 | 182.3 | 182.3 (pt_compiled) |  |
| 196 | 1 | TriMul incoming | inference | bf16 | D512 | 384 | Triton >= PyTorch | 1.19 | 591.9 | 1658 | 1396 | 1396 (pt_compiled) |  |
| 197 | 1 | Transition | training | fp32 | D256 TF32 | 384 | Triton >= PyTorch | 1.17 | 2451 | 2966 | 2540 | 2540 (pt_compiled) |  |
| 198 | 1 | AttentionPairBias | inference | bf16 | 8x48 pair256 | 384 | Triton >= PyTorch | 1.17 | 100.1 | 100.1 | 85.8 | 85.8 (pt_compiled) | within noise |
| 199 | 1 | AttentionPairBias | inference | bf16 | 8x48 pair256 | 384 | CUDA >= PyTorch | 1.17 | 100.1 | 100.1 | 85.8 | 85.8 (pt_compiled) | within noise |
| 200 | 1 | TriMul outgoing | inference | bf16 | D512 | 384 | Triton >= PyTorch | 1.16 | 602.0 | 1658 | 1425 | 1425 (pt_compiled) |  |
| 201 | 1 | bias-only DiT | training | fp32 | d768 16x48 per-sample pair256 A4 | 256 | CUDA >= PyTorch | 1.15 | 876.5 | 848.0 | 763.8 | 763.8 (pt_compiled) | within noise |
| 202 | 1 | bias-only DiT | training | fp32 | d768 16x48 per-sample pair256 A4 | 512 | CUDA >= PyTorch | 1.15 | 2390 | 2380 | 2084 | 2084 (pt_compiled) | within noise |
| 203 | 1 | bias-only DiT | training | fp32 | d768 16x48 per-sample pair256 A4 | 512 | Triton >= PyTorch | 1.14 | 2390 | 2380 | 2084 | 2084 (pt_compiled) | within noise |
| 204 | 1 | Transition | inference | bf16 | D512 | 384 | Triton >= PyTorch | 1.13 | 792.4 | 1253 | 1108 | 1108 (pt_compiled) |  |
| 205 | 1 | bias-only DiT | training | fp32 | d768 16x48 per-sample pair256 A4 | 256 | Triton >= PyTorch | 1.11 | 876.5 | 848.0 | 763.8 | 763.8 (pt_compiled) | within noise |
| 206 | 1 | AttentionPairBias | training | bf16 | 8x48 pair256 | 384 | CUDA >= Triton | 1.11 | 317.6 | 286.7 | 298.8 | 298.8 (pt_compiled) |  |
| 207 | 1 | AugmentedAttention atom | inference | bf16 | d128 A5/A48 | 384 | CUDA >= PyTorch | 1.09 | 1261 | 906.2 | 1157 | 1157 (pt_compiled) |  |
| 208 | 1 | token DiT + extra_pair_bias | inference | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | Triton >= PyTorch | 1.08 | 81.7 | 159.7 | 147.5 | 147.5 (pt_compiled) |  |
| 209 | 1 | token DiT + extra_pair_bias | training | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | Triton >= PyTorch | 1.07 | 417.8 | 489.5 | 458.8 | 458.8 (pt_compiled) | within noise |
| 210 | 1 | AugmentedAttention atom | inference | bf16 | d128 A5/A48 | 128 | CUDA >= PyTorch | 1.07 | 194.4 | 159.7 | 182.3 | 182.3 (pt_compiled) |  |
| 211 | 1 | AttentionPairBias | training | bf16 | 8x48 pair256 | 384 | CUDA >= PyTorch | 1.06 | 317.6 | 286.7 | 298.8 | 298.8 (pt_compiled) |  |
| 212 | 1 | AttentionPairBias gate bias 1.0 | inference | bf16 | 16x24 pair128 | 384 | Triton >= PyTorch | 1.06 | 34.7 | 75.7 | 71.5 | 71.5 (pt_compiled) |  |
| 213 | 1 | bias-only DiT | inference | fp32 | d768 16x48 per-sample pair256 A4 | 512 | Triton >= PyTorch | 1.04 | 758.8 | 735.2 | 708.7 | 708.7 (pt_compiled) |  |
| 214 | 1 | bias-only DiT | training | fp32 | d768 16x48 per-sample pair256 A4 | 256 | CUDA >= Triton | 1.03 | 876.5 | 848.0 | 763.8 | 763.8 (pt_compiled) | within noise |
| 215 | 1 | bias-only DiT | inference | fp32 | d768 16x48 per-sample pair256 A4 | 256 | Triton >= PyTorch | 1.03 | 305.1 | 272.3 | 264.3 | 264.3 (pt_compiled) |  |
| 216 | 1 | OuterProductMean RF3 (bias, divide after) | training | bf16 | S512 msa64 pair128 | 256 | Triton >= PyTorch | 1.01 | 317.3 | 495.5 | 489.5 | 489.5 (pt_compiled) | within noise |
| 217 | 1 | bias-only DiT | training | bf16 | 12x64 | 512 | Triton >= PyTorch | 1.01 | 1457 | 1759 | 1745 | 1745 (pt_compiled) | within noise |
| 218 | 1 | bias-only DiT | training | fp32 | d768 16x48 per-sample pair256 A4 | 512 | CUDA >= Triton | 1.00 | 2390 | 2380 | 2084 | 2084 (pt_compiled) | within noise |
| 219 | 1 | AttentionPairBias | inference | bf16 | 8x48 pair256 | 384 | CUDA >= Triton | 1.00 | 100.1 | 100.1 | 85.8 | 85.8 (pt_compiled) | within noise |

## Ops missing a level on B200

### No hand-CUDA path (`miniworld` on B200 runs the PyTorch reference)

| op | mode | dtype | shape | L | miniworld (= PyTorch) us | Triton us | Triton speed-up lost |
|---|---|---|---|---|---|---|---|
| AugmentedAttention atom | inference | fp32 | d128 pair16 4 heads A5/A48 | 128 | 239.6 | 214.9 | 1.11x |
| AugmentedAttention atom | inference | fp32 | d128 pair16 4 heads A5/A48 | 384 | 1691 | 1427 | 1.18x |
| AugmentedAttention atom | inference | fp32 | d128 pair16 4 heads A5/A48 | 1024 | 11228 | 9282 | 1.21x |
| AugmentedAttention atom | training | fp32 | d128 pair16 4 heads A5/A48 | 128 | 3727 | 2575 | 1.45x |
| AugmentedAttention atom | training | bf16 | d128 A5/A48 | 128 | 2787 | 1607 | 1.73x |
| AugmentedAttention atom | training | fp32 | d128 pair16 4 heads A5/A48 | 384 | 29392 | 17555 | 1.67x |
| AugmentedAttention atom | training | bf16 | d128 A5/A48 | 384 | 25084 | 10288 | 2.44x |
| AugmentedAttention token | training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | 882.7 | 872.6 | 1.01x |
| AugmentedAttention token | training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | 3488 | 3191 | 1.09x |
| AugmentedAttention token | training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | 10552 | 9076 | 1.16x |
| LocalDiT ESMFold2 atom block | inference | bf16 | d128 192 keys rotary (atoms=8L) | 512 | 67.7 | 268.2 | 0.25x |
| LocalDiT ESMFold2 atom block | inference | bf16 | d128 192 keys rotary (atoms=8L) | 1024 | 108.7 | 452.6 | 0.24x |
| LocalDiT ESMFold2 atom block | training | bf16 | d128 192 keys rotary (atoms=8L) | 512 | 1396 | 3729 | 0.37x |
| LocalDiT ESMFold2 atom block | training | bf16 | d128 192 keys rotary (atoms=8L) | 1024 | 2562 | 7269 | 0.35x |
| MSAPairWeightedAveraging | inference | bf16 | S1024 8x32 pair64 | 384 | 483.4 | 344.0 | 1.41x |
| MSAPairWeightedAveraging | inference | bf16 | S1024 8x32 pair64 | 768 | 1055 | 729.1 | 1.45x |
| MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair64 | 384 | 1368 | 1379 | 0.99x |
| MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair64 | 768 | 2915 | 2846 | 1.02x |
| OuterProductMean | inference | bf16 | S1024 msa64 pair64 | 384 | 673.7 | 319.4 | 2.11x |
| OuterProductMean | inference | bf16 | S1024 msa64 pair64 | 768 | 2499 | 1159 | 2.16x |
| OuterProductMean | training | bf16 | S1024 msa64 pair64 | 384 | 1251 | 1159 | 1.08x |
| OuterProductMean | training | bf16 | S1024 msa64 pair64 | 768 | 4482 | 4006 | 1.12x |
| TriMul bidirectional hidden!=D | training | bf16 | D64 hidden128 | 384 | 1871 | 782.2 | 2.39x |
| bias-only DiT | inference | bf16 | d768 16x48 pair256 A5/A48 | 256 | 106.5 | 102.3 | 1.04x |
| bias-only DiT | inference | fp32 | d768 16x48 pair256 A5/A48 | 256 | 165.7 | 137.2 | 1.21x |
| bias-only DiT | inference | bf16 | d768 16x48 per-sample pair256 A4 | 256 | 155.5 | 157.5 | 0.99x |
| bias-only DiT | inference | fp32 | d768 16x48 per-sample pair256 A4 | 256 | 305.1 | 272.3 | 1.12x |
| bias-only DiT | inference | bf16 | d768 16x48 pair256 A5/A48 | 512 | 143.4 | 137.2 | 1.05x |
| bias-only DiT | inference | fp32 | d768 16x48 pair256 A5/A48 | 512 | 206.9 | 182.2 | 1.14x |
| bias-only DiT | inference | bf16 | d768 16x48 per-sample pair256 A4 | 512 | 315.4 | 323.5 | 0.98x |
| bias-only DiT | inference | fp32 | d768 16x48 per-sample pair256 A4 | 512 | 758.8 | 735.2 | 1.03x |
| bias-only DiT | inference | bf16 | d768 16x48 pair256 A5/A48 | 768 | 172.0 | 184.4 | 0.93x |
| bias-only DiT | training | bf16 | d768 16x48 pair256 A5/A48 | 256 | 909.2 | 961.6 | 0.95x |
| bias-only DiT | training | fp32 | d768 16x48 pair256 A5/A48 | 256 | 1678 | 1768 | 0.95x |
| bias-only DiT | training | bf16 | d768 16x48 pair256 A5/A48 | 512 | 1760 | 1850 | 0.95x |
| bias-only DiT | training | fp32 | d768 16x48 pair256 A5/A48 | 512 | 3364 | 3518 | 0.96x |
| bias-only DiT | training | bf16 | d768 16x48 pair256 A5/A48 | 768 | 2738 | 2899 | 0.94x |
| token DiT (fused+composed) | inference | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 141.3 | 202.6 | 0.70x |
| token DiT (fused+composed) | inference | fp32 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 362.4 | 297.0 | 1.22x |
| token DiT (fused+composed) | inference | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 768 | 235.6 | 407.4 | 0.58x |
| token DiT (fused+composed) | training | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 1691 | 2394 | 0.71x |
| token DiT (fused+composed) | training | fp32 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 4657 | 4296 | 1.08x |
| token DiT (fused+composed) | training | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 768 | 3779 | 6211 | 0.61x |

### Other rows where a rung has no measurement

| op | mode | dtype | shape | L | CUDA us | Triton us | PyTorch compiled us | why |
|---|---|---|---|---|---|---|---|---|
| AugmentedAttention atom | training | fp32 | d128 pair16 4 heads A5/A48 | 1024 | - | 121055 | - | cuda_vs_portable: cuda: OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 GiB. GPU 0 has a total capac / portable_vs_pytorch: pytorch: OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 GiB. GPU 0  |

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
| inference | bf16 | D64 | 768 | 167.7 | 274.2 | 851.7 | 2758 | 0.61 | 0.32 | OK |
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
| training | bf16 | D64 | 768 | 622.6 | 1112 | 2563 | 7793 | 0.56 | 0.43 | OK |

### TriMul incoming

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 | 384 | 86.1 | 190.4 | 391.2 | 1049 | 0.45 | 0.49 | OK |
| inference | bf16 | D128 | 768 | 327.6 | 681.8 | 1714 | 4538 | 0.48 | 0.40 | OK |
| inference | bf16 | D512 | 384 | 591.9 | 1658 | 1396 | 3470 | 0.36 | 1.19 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | D64 | 384 | 53.1 | 79.9 | 207.9 | 675.9 | 0.66 | 0.38 | OK |
| inference | bf16 | D64 | 768 | 167.8 | 274.4 | 849.8 | 2758 | 0.61 | 0.32 | OK |
| training | bf16 | D128 | 384 | 336.7 | 603.1 | 1235 | 3250 | 0.56 | 0.49 | OK |
| training | bf16 | D128 | 768 | 1266 | 2194 | 5039 | 13897 | 0.58 | 0.44 | OK |
| training | bf16 | D512 | 384 | 2427 | 3388 | 4545 | 11135 | 0.72 | 0.75 | OK |
| training | bf16 | D64 | 384 | 190.5 | 338.1 | 706.6 | 1979 | 0.56 | 0.48 | OK |
| training | bf16 | D64 | 768 | 619.5 | 1115 | 2542 | 7787 | 0.56 | 0.44 | OK |

### TriMul outgoing hidden!=D

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 hidden256 | 384 | 126.7 | 325.5 | 678.1 | 1655 | 0.39 | 0.48 | OK |
| inference | bf16 | D64 hidden128 | 128 | 22.5 | 34.7 | 71.6 | 157.7 | 0.65 | 0.48 | OK |
| inference | bf16 | D64 hidden128 | 384 | 69.7 | 137.2 | 325.5 | 965.6 | 0.51 | 0.42 | OK |
| training | bf16 | D128 hidden256 | 384 | 493.7 | 964.6 | 2018 | 5225 | 0.51 | 0.48 | OK |
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
| inference | bf16 | D512 | 384 | 967.1 | 2703 | 2867 | 6317 | 0.36 | 0.94 | OK |
| inference | bf16 | D64 | 384 | 75.7 | 141.3 | 337.9 | 991.1 | 0.54 | 0.42 | OK |
| inference | bf16 | D64 | 768 | 257.9 | 491.1 | 1630 | 3986 | 0.53 | 0.30 | OK |
| training | bf16 | D128 | 128 | 152.5 | 187.5 | 375.9 | 812.0 | 0.81 | 0.50 | OK |
| training | bf16 | D128 | 384 | 532.6 | 973.9 | 2076 | 4470 | 0.55 | 0.47 | OK |
| training | bf16 | D128 | 768 | 2016 | 3909 | 9184 | 18533 | 0.52 | 0.43 | OK |
| training | bf16 | D256 | 384 | 1587 | 2084 | 3985 | 8210 | 0.76 | 0.52 | OK |
| training | bf16 | D256 | 768 | 6266 | 9190 | 20659 | 39332 | 0.68 | 0.44 | OK |
| training | bf16 | D512 | 384 | 3929 | 5934 | 9256 | 16626 | 0.66 | 0.64 | OK |
| training | bf16 | D64 | 384 | 268.3 | 491.5 | 1113 | 2583 | 0.55 | 0.44 | OK |
| training | bf16 | D64 | 768 | 924.6 | 1719 | 4213 | 9989 | 0.54 | 0.41 | OK |

### TriMul bidirectional hidden!=D

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D64 hidden128 | 384 | 603.7 | 231.5 | 661.6 | 1606 | 2.61 | 0.35 | **VIOLATION(cuda_vs_portable)** |
| training | bf16 | D64 hidden128 | 384 | 1871 | 782.2 | 1914 | 4151 | 2.39 | 0.41 | no CUDA path |

### TriAttn

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 4x32 | 128 | 32.5 | 57.1 | 87.9 | 178.2 | 0.57 | 0.65 | OK |
| inference | bf16 | D128 4x32 | 384 | 175.8 | 337.9 | 1317 | 2406 | 0.52 | 0.26 | OK |
| inference | bf16 | D128 4x32 | 768 | 917.4 | 1879 | 9967 | 16617 | 0.49 | 0.19 | OK |
| inference | bf16 | D256 8x32 | 384 | 327.6 | 604.0 | 2597 | 4614 | 0.54 | 0.23 | OK |
| inference | bf16 | D64 2x32 | 384 | 102.3 | 190.5 | 591.9 | 1308 | 0.54 | 0.32 | OK |
| inference | bf16 | D64 2x32 | 768 | 501.6 | 946.9 | 5023 | 8684 | 0.53 | 0.19 | OK |
| training | bf16 | D128 4x32 | 128 | 113.7 | 205.9 | 270.5 | 547.8 | 0.55 | 0.76 | OK |
| training | bf16 | D128 4x32 | 384 | 646.2 | 1397 | 2195 | 5375 | 0.46 | 0.64 | OK |
| training | bf16 | D128 4x32 | 768 | 3382 | 7797 | 13787 | 35604 | 0.43 | 0.57 | OK |
| training | bf16 | D256 8x32 | 384 | 1326 | 2561 | 6270 | 10309 | 0.52 | 0.41 | OK |
| training | bf16 | D64 2x32 | 384 | 416.8 | 789.5 | 1554 | 2928 | 0.53 | 0.51 | OK |
| training | bf16 | D64 2x32 | 768 | 1926 | 3976 | 7072 | 18594 | 0.48 | 0.56 | OK |

### Transition

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | D128 | 128 | 16.3 | 36.9 | 38.9 | 83.9 | 0.44 | 0.95 | OK |
| inference | bf16 | D128 | 384 | 55.3 | 200.7 | 207.8 | 568.4 | 0.28 | 0.97 | OK |
| inference | bf16 | D128 | 768 | 200.7 | 722.1 | 782.4 | 2143 | 0.28 | 0.92 | OK |
| inference | bf16 | D128 | 1024 | 352.1 | 1259 | 1378 | 3772 | 0.28 | 0.91 | OK |
| inference | bf16 | D256 | 384 | 190.2 | 440.5 | 428.0 | 996.1 | 0.43 | 1.03 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | D256 | 768 | 787.2 | 1664 | 1725 | 3869 | 0.47 | 0.96 | OK |
| inference | bf16 | D384 | 384 | 407.5 | 830.0 | 768.1 | 1528 | 0.49 | 1.08 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | D384 | 768 | 1613 | 3339 | 3063 | 6062 | 0.48 | 1.09 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | D512 | 384 | 792.4 | 1253 | 1108 | 2043 | 0.63 | 1.13 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | D64 | 384 | 28.5 | 92.0 | 137.1 | 364.3 | 0.31 | 0.67 | OK |
| inference | bf16 | D64 | 768 | 83.8 | 329.7 | 382.6 | 1398 | 0.25 | 0.86 | OK |
| inference | bf16 | D768 | 384 | 2142 | 2445 | 2153 | 3475 | 0.88 | 1.14 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D128 TF32 | 384 | 217.0 | 425.5 | 385.0 | 734.2 | 0.51 | 1.11 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D128 TF32 | 768 | 792.5 | 1602 | 1454 | 2832 | 0.49 | 1.10 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D256 TF32 | 384 | 565.2 | 1115 | 802.9 | 1363 | 0.51 | 1.39 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D128 | 128 | 57.3 | 110.5 | 110.5 | 288.8 | 0.52 | 1.00 | OK |
| training | bf16 | D128 | 384 | 270.4 | 662.4 | 597.9 | 1547 | 0.41 | 1.11 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D128 | 768 | 1049 | 2430 | 2118 | 5717 | 0.43 | 1.15 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D128 | 1024 | 1905 | 4249 | 3672 | 10036 | 0.45 | 1.16 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D256 | 384 | 945.8 | 1432 | 1211 | 2771 | 0.66 | 1.18 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D256 | 768 | 3767 | 5574 | 4487 | 10645 | 0.68 | 1.24 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D384 | 384 | 1739 | 2708 | 2122 | 4252 | 0.64 | 1.28 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D384 | 768 | 6871 | 10665 | 8263 | 16945 | 0.64 | 1.29 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D512 | 384 | 2839 | 4104 | 3149 | 5951 | 0.69 | 1.30 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D64 | 384 | 114.1 | 329.5 | 344.1 | 936.9 | 0.35 | 0.96 | OK |
| training | bf16 | D64 | 768 | 396.1 | 1132 | 1074 | 3482 | 0.35 | 1.05 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | D768 | 384 | 8763 | 7929 | 6372 | 10004 | 1.11 | 1.24 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D128 TF32 | 384 | 913.4 | 1155 | 1118 | 2194 | 0.79 | 1.03 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D128 TF32 | 768 | 3387 | 4301 | 4003 | 8218 | 0.79 | 1.07 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D256 TF32 | 384 | 2451 | 2966 | 2540 | 4355 | 0.83 | 1.17 | **VIOLATION(portable_vs_pytorch)** |

### Transition conditioning (pair)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 | D128 n2 TF32 | 128 | 30.6 | 38.8 | 41.0 | 63.3 | 0.79 | 0.95 | OK |
| inference | fp32 | D128 n2 TF32 | 384 | 147.3 | 257.9 | 243.8 | 489.1 | 0.57 | 1.06 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D128 n2 TF32 | 768 | 520.1 | 952.3 | 916.4 | 1871 | 0.55 | 1.04 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D256 n2 TF32 | 128 | 55.2 | 87.9 | 67.6 | 102.3 | 0.63 | 1.30 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D256 n2 TF32 | 384 | 339.8 | 638.9 | 485.3 | 836.7 | 0.53 | 1.32 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D256 n2 TF32 | 768 | 1326 | 2464 | 1861 | 3273 | 0.54 | 1.32 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D128 n2 TF32 | 128 | 112.5 | 122.8 | 122.8 | 247.8 | 0.92 | 1.00 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D128 n2 TF32 | 384 | 597.9 | 715.0 | 714.8 | 1406 | 0.84 | 1.00 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D128 n2 TF32 | 768 | 2092 | 2520 | 2485 | 5185 | 0.83 | 1.01 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D256 n2 TF32 | 128 | 227.1 | 257.0 | 229.4 | 405.4 | 0.88 | 1.12 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D256 n2 TF32 | 384 | 1514 | 1756 | 1389 | 2477 | 0.86 | 1.26 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D256 n2 TF32 | 768 | 5375 | 6198 | 5042 | 9467 | 0.87 | 1.23 | **VIOLATION(portable_vs_pytorch)** |

### Transition conditioning (single)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 | D384 n2 TF32 | 128 | 22.4 | 26.5 | 26.5 | 28.7 | 0.85 | 1.00 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D384 n2 TF32 | 384 | 24.4 | 28.6 | - | 30.7 | 0.85 | 0.93 | OK |
| inference | fp32 | D384 n2 TF32 | 768 | 24.4 | 28.6 | - | 32.5 | 0.86 | 0.88 | OK |
| inference | fp32 | D768 n2 TF32 | 128 | 32.7 | 36.8 | 30.7 | 34.8 | 0.89 | 1.20 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D768 n2 TF32 | 384 | 32.7 | 36.7 | 32.7 | 36.9 | 0.89 | 1.12 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | D768 n2 TF32 | 768 | 34.8 | 45.1 | 36.8 | 43.0 | 0.77 | 1.22 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D384 n2 TF32 | 128 | 59.3 | 65.4 | 57.3 | 69.7 | 0.91 | 1.14 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D384 n2 TF32 | 384 | 63.4 | 69.5 | 67.5 | 77.9 | 0.91 | 1.03 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D384 n2 TF32 | 768 | 65.4 | 73.6 | 73.8 | 86.0 | 0.89 | 1.00 | OK |
| training | fp32 | D768 n2 TF32 | 128 | 81.9 | 92.1 | 71.7 | 86.0 | 0.89 | 1.29 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D768 n2 TF32 | 384 | 88.2 | 98.2 | 84.0 | 98.3 | 0.90 | 1.17 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | D768 n2 TF32 | 768 | 102.5 | 118.8 | 98.3 | 114.7 | 0.86 | 1.21 | **VIOLATION(portable_vs_pytorch)** |

### OuterProductMean

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S1024 msa64 pair128 | 256 | 149.6 | 172.0 | 221.2 | 903.1 | 0.87 | 0.78 | OK |
| inference | bf16 | S1024 msa64 pair128 | 384 | 305.3 | 347.1 | 454.6 | 1670 | 0.88 | 0.76 | OK |
| inference | bf16 | S1024 msa64 pair128 | 768 | 1159 | 1281 | 1683 | 5311 | 0.91 | 0.76 | OK |
| inference | bf16 | S1024 msa64 pair64 | 384 | 673.7 | 319.4 | 434.1 | 1660 | 2.11 | 0.74 | no CUDA path |
| inference | bf16 | S1024 msa64 pair64 | 768 | 2499 | 1159 | 1569 | 5273 | 2.16 | 0.74 | no CUDA path |
| training | bf16 | S1024 msa64 pair128 | 256 | 444.4 | 670.6 | 659.5 | 2098 | 0.66 | 1.02 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | S1024 msa64 pair128 | 384 | 927.5 | 1320 | 1336 | 3902 | 0.70 | 0.99 | OK |
| training | bf16 | S1024 msa64 pair128 | 768 | 3491 | 4639 | 4725 | 12001 | 0.75 | 0.98 | OK |
| training | bf16 | S1024 msa64 pair64 | 384 | 1251 | 1159 | 1270 | 3746 | 1.08 | 0.91 | no CUDA path |
| training | bf16 | S1024 msa64 pair64 | 768 | 4482 | 4006 | 4434 | 11752 | 1.12 | 0.90 | no CUDA path |

### MSAPairWeightedAveraging

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S1024 8x32 pair128 | 256 | 149.6 | 239.5 | 311.4 | 1044 | 0.62 | 0.77 | OK |
| inference | bf16 | S1024 8x32 pair128 | 384 | 217.1 | 356.3 | 484.9 | 1643 | 0.61 | 0.73 | OK |
| inference | bf16 | S1024 8x32 pair128 | 768 | 475.2 | 763.8 | 1061 | 3800 | 0.62 | 0.72 | OK |
| inference | bf16 | S1024 8x32 pair64 | 384 | 483.4 | 344.0 | 475.1 | 1603 | 1.41 | 0.72 | no CUDA path |
| inference | bf16 | S1024 8x32 pair64 | 768 | 1055 | 729.1 | 1046 | 3673 | 1.45 | 0.70 | no CUDA path |
| training | bf16 | S1024 8x32 pair128 | 256 | 438.4 | 976.0 | 913.2 | 2893 | 0.45 | 1.07 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | S1024 8x32 pair128 | 384 | 662.3 | 1408 | 1371 | 3875 | 0.47 | 1.03 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | S1024 8x32 pair128 | 768 | 1470 | 2951 | 2926 | 8726 | 0.50 | 1.01 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | S1024 8x32 pair64 | 384 | 1368 | 1379 | 1347 | 3786 | 0.99 | 1.02 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | S1024 8x32 pair64 | 768 | 2915 | 2846 | 2881 | 8391 | 1.02 | 0.99 | no CUDA path |

### OuterProductMean RF3 (bias, divide before)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S512 msa64 pair128 | 256 | 104.5 | 124.7 | 169.7 | 655.1 | 0.84 | 0.73 | OK |
| inference | bf16 | S512 msa64 pair128 | 384 | 198.7 | 229.5 | 337.6 | 1273 | 0.87 | 0.68 | OK |
| training | bf16 | S512 msa64 pair128 | 256 | 315.0 | 489.5 | 481.4 | 1568 | 0.64 | 1.02 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | S512 msa64 pair128 | 384 | 628.4 | 928.7 | 944.9 | 2989 | 0.68 | 0.98 | OK |

### OuterProductMean RF3 (bias, divide after)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S512 msa64 pair128 | 256 | 106.5 | 125.9 | 170.0 | 511.7 | 0.85 | 0.74 | OK |
| inference | bf16 | S512 msa64 pair128 | 384 | 208.9 | 239.5 | 339.9 | 953.2 | 0.87 | 0.70 | OK |
| training | bf16 | S512 msa64 pair128 | 256 | 317.3 | 495.5 | 489.5 | 1283 | 0.64 | 1.01 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | S512 msa64 pair128 | 384 | 639.0 | 937.9 | 951.4 | 2352 | 0.68 | 0.99 | OK |

### OuterProductMean Chai-1 (8 groups)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S512 pair256 8x64 | 256 | 77.7 | 118.3 | 169.8 | 687.9 | 0.66 | 0.70 | OK |
| inference | bf16 | S512 pair256 8x64 | 384 | 145.3 | 227.1 | 327.5 | 1308 | 0.64 | 0.69 | OK |
| training | bf16 | S512 pair256 8x64 | 256 | 331.7 | 552.8 | 505.9 | 2106 | 0.60 | 1.09 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | S512 pair256 8x64 | 384 | 584.9 | 960.5 | 914.4 | 2900 | 0.61 | 1.05 | **VIOLATION(portable_vs_pytorch)** |

### MSAPairWeightedAveraging value_dim 8

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | S512 8x8 pair128 | 256 | 59.4 | 98.3 | 94.1 | 376.9 | 0.60 | 1.04 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | S512 8x8 pair128 | 384 | 81.8 | 139.2 | 130.0 | 630.9 | 0.59 | 1.07 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | S512 8x8 pair128 | 256 | 214.0 | 404.3 | 330.8 | 1340 | 0.53 | 1.22 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | S512 8x8 pair128 | 384 | 289.7 | 578.6 | 461.9 | 1508 | 0.50 | 1.25 | **VIOLATION(portable_vs_pytorch)** |

### AttentionPairBias

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | 16x24 pair128 | 384 | 34.8 | 75.7 | 71.7 | 301.1 | 0.46 | 1.06 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | 8x48 pair128 | 384 | 32.9 | 75.6 | 71.5 | 296.9 | 0.44 | 1.06 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | 8x48 pair128 | 768 | 57.3 | 151.4 | 147.1 | 1003 | 0.38 | 1.03 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | 8x48 pair256 | 384 | 100.1 | 100.1 | 85.8 | 401.2 | 1.00 | 1.17 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| training | bf16 | 16x24 pair128 | 384 | 110.5 | 208.9 | 254.0 | 715.8 | 0.53 | 0.82 | OK |
| training | bf16 | 8x48 pair128 | 384 | 100.3 | 198.6 | 249.9 | 708.5 | 0.51 | 0.79 | OK |
| training | bf16 | 8x48 pair128 | 768 | 178.2 | 458.8 | 516.1 | 2203 | 0.39 | 0.89 | OK |
| training | bf16 | 8x48 pair256 | 384 | 317.6 | 286.7 | 298.8 | 929.8 | 1.11 | 0.96 | **VIOLATION(cuda_vs_portable)** |

### AttentionPairBias Chai-1 (gate bias 1.0)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | 16x24 pair256 | 384 | 43.0 | 102.2 | 86.0 | 407.6 | 0.42 | 1.19 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | 16x24 pair256 | 768 | 90.2 | 251.7 | 219.0 | 1429 | 0.36 | 1.15 | **VIOLATION(portable_vs_pytorch)** |

### AttentionPairBias gate bias 1.0

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | 16x24 pair128 | 384 | 34.7 | 75.7 | 71.5 | 303.1 | 0.46 | 1.06 | **VIOLATION(portable_vs_pytorch)** |

### AugmentedAttention token

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d768 A5/A48 | 384 | 71.5 | 116.8 | 145.3 | 497.5 | 0.61 | 0.80 | OK |
| inference | bf16 | d768 A5/A48 | 768 | 128.9 | 245.2 | 336.0 | 1658 | 0.53 | 0.73 | OK |
| inference | bf16 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | 55.2 | 81.9 | 88.1 | 188.3 | 0.67 | 0.93 | OK |
| inference | bf16 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | 83.9 | 143.4 | 169.9 | 608.3 | 0.59 | 0.84 | OK |
| inference | bf16 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | 163.9 | 319.6 | 412.7 | 2076 | 0.51 | 0.77 | OK |
| inference | fp32 | d768 cond384 pair128 A5/A48 | 128 | 75.6 | 91.7 | 92.0 | 139.0 | 0.82 | 1.00 | OK |
| inference | fp32 | d768 cond384 pair128 A5/A48 | 384 | 128.8 | 182.0 | 200.5 | 428.0 | 0.71 | 0.91 | OK |
| inference | fp32 | d768 cond384 pair128 A5/A48 | 768 | 241.4 | 458.6 | 511.7 | 1371 | 0.53 | 0.90 | OK |
| inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | 104.5 | 110.5 | 100.2 | 147.4 | 0.95 | 1.10 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | 257.0 | 233.3 | 245.7 | 480.1 | 1.10 | 0.95 | **VIOLATION(cuda_vs_portable)** |
| inference | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | 697.4 | 616.3 | 654.4 | 1551 | 1.13 | 0.94 | **VIOLATION(cuda_vs_portable)** |
| training | bf16 | d768 A5/A48 | 384 | 899.0 | 1575 | 2829 | 4950 | 0.57 | 0.56 | OK |
| training | bf16 | d768 A5/A48 | 768 | 2145 | 4494 | 9383 | 15674 | 0.48 | 0.48 | OK |
| training | bf16 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | 436.1 | 503.8 | 630.8 | 1236 | 0.87 | 0.80 | OK |
| training | bf16 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | 992.0 | 1736 | 2939 | 5345 | 0.57 | 0.59 | OK |
| training | bf16 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | 2393 | 4977 | 9701 | 16796 | 0.48 | 0.51 | OK |
| training | fp32 | d768 cond384 pair128 A5/A48 | 128 | 667.7 | 782.3 | 781.3 | 1192 | 0.85 | 1.00 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond384 pair128 A5/A48 | 384 | 1736 | 2854 | 3215 | 4837 | 0.61 | 0.89 | OK |
| training | fp32 | d768 cond384 pair128 A5/A48 | 768 | 4337 | 8243 | 9782 | 16130 | 0.53 | 0.84 | OK |
| training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 128 | 882.7 | 872.6 | 862.4 | 1266 | 1.01 | 1.01 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 384 | 3488 | 3191 | 3442 | 5142 | 1.09 | 0.93 | no CUDA path |
| training | fp32 | d768 cond768 pair256 A5/A48 (ESMFold2) | 768 | 10552 | 9076 | 10446 | 16851 | 1.16 | 0.87 | no CUDA path |

### AugmentedAttention atom

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d128 A5/A48 | 128 | 194.4 | 159.7 | 182.3 | 1592 | 1.22 | 0.88 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d128 A5/A48 | 384 | 1261 | 906.2 | 1157 | 14075 | 1.39 | 0.78 | **VIOLATION(cuda_vs_portable)** |
| inference | fp32 | d128 pair16 4 heads A5/A48 | 128 | 239.6 | 214.9 | 237.4 | 1473 | 1.11 | 0.90 | no CUDA path |
| inference | fp32 | d128 pair16 4 heads A5/A48 | 384 | 1691 | 1427 | 1676 | 12804 | 1.18 | 0.85 | no CUDA path |
| inference | fp32 | d128 pair16 4 heads A5/A48 | 1024 | 11228 | 9282 | 11142 | 92144 | 1.21 | 0.83 | no CUDA path |
| training | bf16 | d128 A5/A48 | 128 | 2787 | 1607 | 2789 | 8249 | 1.73 | 0.58 | no CUDA path |
| training | bf16 | d128 A5/A48 | 384 | 25084 | 10288 | 25008 | 71078 | 2.44 | 0.41 | no CUDA path |
| training | fp32 | d128 pair16 4 heads A5/A48 | 128 | 3727 | 2575 | 3736 | 7576 | 1.45 | 0.69 | no CUDA path |
| training | fp32 | d128 pair16 4 heads A5/A48 | 384 | 29392 | 17555 | 29483 | 68170 | 1.67 | 0.60 | no CUDA path |
| training | fp32 | d128 pair16 4 heads A5/A48 | 1024 | - | 121055 | - | - | - | - | n.a. |

### token DiT (fused+composed)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d768 16x48 A5/A48 shared-cond | 384 | 75.6 | 182.3 | 200.8 | 607.2 | 0.41 | 0.91 | OK |
| inference | bf16 | d768 16x48 A5/A48 shared-cond | 768 | 119.9 | 341.1 | 407.6 | 1813 | 0.35 | 0.84 | OK |
| inference | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 141.3 | 202.6 | 223.2 | 709.5 | 0.70 | 0.91 | no CUDA path |
| inference | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 768 | 235.6 | 407.4 | 477.2 | 2217 | 0.58 | 0.85 | no CUDA path |
| inference | bf16 | d768 qk-norm shared-cond | 384 | 79.7 | 186.2 | 204.8 | 662.4 | 0.43 | 0.91 | OK |
| inference | fp32 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 362.4 | 297.0 | 315.4 | 585.9 | 1.22 | 0.94 | no CUDA path |
| inference | fp32 | d768 A5/A48 shared-cond | 384 | 124.8 | 260.1 | 280.5 | 540.6 | 0.48 | 0.93 | OK |
| training | bf16 | d768 16x48 A5/A48 shared-cond | 384 | 1747 | 2301 | 3514 | 6387 | 0.76 | 0.65 | OK |
| training | bf16 | d768 16x48 A5/A48 shared-cond | 768 | 3852 | 5840 | 10657 | 18593 | 0.66 | 0.55 | OK |
| training | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 1691 | 2394 | 3557 | 6612 | 0.71 | 0.67 | no CUDA path |
| training | bf16 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 768 | 3779 | 6211 | 10833 | 19421 | 0.61 | 0.57 | no CUDA path |
| training | bf16 | d768 qk-norm shared-cond | 384 | 1834 | 2405 | 3639 | 7610 | 0.76 | 0.66 | OK |
| training | fp32 | d768 16x48 pair256 A5/A48 shared-cond (Protenix-v2) | 384 | 4657 | 4296 | 4642 | 6854 | 1.08 | 0.93 | no CUDA path |
| training | fp32 | d768 A5/A48 shared-cond | 384 | 2913 | 4176 | 4587 | 6758 | 0.70 | 0.91 | OK |

### token DiT + extra_pair_bias

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | 55.1 | 131.1 | 133.1 | 335.9 | 0.42 | 0.98 | OK |
| inference | bf16 | d768 A2 shared-cond +[L,L] bias | 512 | 65.4 | 180.3 | 194.6 | 721.0 | 0.36 | 0.93 | OK |
| inference | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | 81.7 | 159.7 | 147.5 | 274.4 | 0.51 | 1.08 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | 415.7 | 374.5 | 364.6 | 1241 | 1.11 | 1.03 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| training | bf16 | d768 A2 shared-cond +[L,L] bias | 512 | 591.8 | 550.9 | 512.0 | 1652 | 1.07 | 1.08 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | 417.8 | 489.5 | 458.8 | 1157 | 0.85 | 1.07 | **VIOLATION(portable_vs_pytorch)** |

### DiT variant stack

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | Boltz-2 token | 384 | 81.9 | 178.2 | 207.9 | 636.9 | 0.46 | 0.86 | OK |
| inference | bf16 | Chai-1 token | 384 | 81.8 | 174.1 | 194.6 | 696.3 | 0.47 | 0.89 | OK |
| inference | bf16 | RoseTTAFold3 token | 384 | 83.8 | 165.9 | 194.6 | 645.2 | 0.51 | 0.85 | OK |
| inference | fp32 | Boltz-2 token | 384 | 143.1 | 253.9 | 268.3 | 546.8 | 0.56 | 0.95 | OK |
| inference | fp32 | Chai-1 token | 384 | 137.0 | 256.0 | 270.2 | 546.8 | 0.54 | 0.95 | OK |
| inference | fp32 | RoseTTAFold3 token | 384 | 135.0 | 235.5 | 256.9 | 536.5 | 0.57 | 0.92 | OK |
| training | bf16 | Boltz-2 token | 384 | 1763 | 2291 | 3673 | 6327 | 0.77 | 0.62 | OK |
| training | bf16 | Chai-1 token | 384 | 1492 | 2053 | 3398 | 6161 | 0.73 | 0.60 | OK |
| training | bf16 | RoseTTAFold3 token | 384 | 1672 | 2325 | 3672 | 6815 | 0.72 | 0.63 | OK |
| training | fp32 | Boltz-2 token | 384 | 2984 | 3864 | 4431 | 6559 | 0.77 | 0.87 | OK |
| training | fp32 | Chai-1 token | 384 | 2521 | 3402 | 3850 | 5960 | 0.74 | 0.88 | OK |
| training | fp32 | RoseTTAFold3 token | 384 | 2686 | 3621 | 4079 | 6641 | 0.74 | 0.89 | OK |

### LocalDiT atom block

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d128 32x128 (atoms=8L) | 512 | 61.2 | 239.6 | 258.1 | 1396 | 0.26 | 0.93 | OK |
| inference | bf16 | d128 32x128 (atoms=8L) | 1024 | 94.2 | 385.2 | 416.7 | 2663 | 0.24 | 0.92 | OK |
| inference | fp32 | d128 32x128 (atoms=8L) | 512 | 118.7 | 339.9 | 303.1 | 1203 | 0.35 | 1.12 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | d128 32x128 (atoms=8L) | 512 | 1585 | 3478 | 4727 | 17316 | 0.46 | 0.74 | OK |
| training | bf16 | d128 32x128 (atoms=8L) | 1024 | 2961 | 6573 | 9101 | 33092 | 0.45 | 0.72 | OK |
| training | fp32 | d128 32x128 (atoms=8L) | 512 | 3896 | 5975 | 6411 | 16369 | 0.65 | 0.93 | OK |

### LocalDiT ESMFold2 atom block

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d128 192 keys rotary (atoms=8L) | 512 | 67.7 | 268.2 | 375.7 | 2209 | 0.25 | 0.71 | no CUDA path |
| inference | bf16 | d128 192 keys rotary (atoms=8L) | 1024 | 108.7 | 452.6 | 657.4 | 4189 | 0.24 | 0.69 | no CUDA path |
| training | bf16 | d128 192 keys rotary (atoms=8L) | 512 | 1396 | 3729 | 7561 | 25301 | 0.37 | 0.49 | no CUDA path |
| training | bf16 | d128 192 keys rotary (atoms=8L) | 1024 | 2562 | 7269 | 14761 | 50569 | 0.35 | 0.49 | no CUDA path |

### DiT variant stack (atom)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | Boltz-2 atom | 512 | 118.6 | 215.0 | 258.2 | 1388 | 0.55 | 0.83 | OK |
| inference | bf16 | Chai-1 atom | 512 | 85.9 | 188.0 | 326.5 | 1344 | 0.46 | 0.58 | OK |
| inference | bf16 | RoseTTAFold3 atom | 512 | 118.7 | 208.9 | 262.3 | 1439 | 0.57 | 0.80 | OK |
| inference | fp32 | Boltz-2 atom | 512 | 153.4 | 243.4 | 315.2 | 1240 | 0.63 | 0.77 | OK |
| inference | fp32 | Chai-1 atom | 512 | 118.7 | 216.8 | 284.6 | 1209 | 0.55 | 0.76 | OK |
| inference | fp32 | RoseTTAFold3 atom | 512 | 147.3 | 237.5 | 308.2 | 1276 | 0.62 | 0.77 | OK |
| training | bf16 | Boltz-2 atom | 512 | 2084 | 2803 | 6539 | 14618 | 0.74 | 0.43 | OK |
| training | bf16 | Chai-1 atom | 512 | 1601 | 2651 | 5641 | 14003 | 0.60 | 0.47 | OK |
| training | bf16 | RoseTTAFold3 atom | 512 | 2104 | 2724 | 6840 | 15598 | 0.77 | 0.40 | OK |
| training | fp32 | Boltz-2 atom | 512 | 2966 | 4487 | 6322 | 13757 | 0.66 | 0.71 | OK |
| training | fp32 | Chai-1 atom | 512 | 2344 | 3924 | 5341 | 12790 | 0.60 | 0.73 | OK |
| training | fp32 | RoseTTAFold3 atom | 512 | 2864 | 4291 | 6102 | 14192 | 0.67 | 0.70 | OK |

### bias-only DiT

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | 12x64 | 512 | 63.4 | 137.2 | 184.4 | 701.4 | 0.46 | 0.74 | OK |
| inference | bf16 | 24x32 | 512 | 64.4 | 141.3 | 188.5 | 702.5 | 0.46 | 0.75 | OK |
| inference | bf16 | d768 16x48 A5/A48 | 256 | 49.0 | 102.4 | 129.1 | 299.0 | 0.48 | 0.79 | OK |
| inference | bf16 | d768 16x48 A5/A48 | 512 | 63.4 | 139.2 | 186.3 | 684.1 | 0.46 | 0.75 | OK |
| inference | bf16 | d768 16x48 A5/A48 | 768 | 83.9 | 186.4 | 257.6 | 1280 | 0.45 | 0.72 | OK |
| inference | bf16 | d768 16x48 pair256 A5/A48 | 256 | 106.5 | 102.3 | 133.2 | 344.1 | 1.04 | 0.77 | no CUDA path |
| inference | bf16 | d768 16x48 pair256 A5/A48 | 512 | 143.4 | 137.2 | 219.0 | 867.3 | 1.05 | 0.63 | no CUDA path |
| inference | bf16 | d768 16x48 pair256 A5/A48 | 768 | 172.0 | 184.4 | 324.5 | 1681 | 0.93 | 0.57 | no CUDA path |
| inference | bf16 | d768 16x48 per-sample pair A4 | 256 | 86.0 | 135.1 | 153.6 | 620.6 | 0.64 | 0.88 | OK |
| inference | bf16 | d768 16x48 per-sample pair A4 | 512 | 157.7 | 241.7 | 307.2 | 1969 | 0.65 | 0.79 | OK |
| inference | bf16 | d768 16x48 per-sample pair256 A4 | 256 | 155.5 | 157.5 | 188.4 | 803.8 | 0.99 | 0.84 | no CUDA path |
| inference | bf16 | d768 16x48 per-sample pair256 A4 | 512 | 315.4 | 323.5 | 424.0 | 2681 | 0.98 | 0.76 | no CUDA path |
| inference | fp32 | d768 16x48 A5/A48 | 256 | 75.7 | 137.2 | 147.5 | 246.7 | 0.55 | 0.93 | OK |
| inference | fp32 | d768 16x48 A5/A48 | 512 | 106.5 | 182.2 | 254.0 | 554.8 | 0.58 | 0.72 | OK |
| inference | fp32 | d768 16x48 pair256 A5/A48 | 256 | 165.7 | 137.2 | 163.9 | 268.3 | 1.21 | 0.84 | no CUDA path |
| inference | fp32 | d768 16x48 pair256 A5/A48 | 512 | 206.9 | 182.2 | 317.5 | 630.8 | 1.14 | 0.57 | no CUDA path |
| inference | fp32 | d768 16x48 per-sample pair A4 | 256 | 131.1 | 186.3 | 204.8 | 487.3 | 0.70 | 0.91 | OK |
| inference | fp32 | d768 16x48 per-sample pair A4 | 512 | 272.5 | 395.2 | 477.2 | 1517 | 0.69 | 0.83 | OK |
| inference | fp32 | d768 16x48 per-sample pair256 A4 | 256 | 305.1 | 272.3 | 264.3 | 563.2 | 1.12 | 1.03 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 16x48 per-sample pair256 A4 | 512 | 758.8 | 735.2 | 708.7 | 1807 | 1.03 | 1.04 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | 12x64 | 512 | 1457 | 1759 | 1745 | 4639 | 0.83 | 1.01 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | 24x32 | 512 | 1502 | 1766 | 1779 | 4665 | 0.85 | 0.99 | OK |
| training | bf16 | d768 16x48 A5/A48 | 256 | 760.7 | 928.8 | 936.8 | 2545 | 0.82 | 0.99 | OK |
| training | bf16 | d768 16x48 A5/A48 | 512 | 1462 | 1738 | 1738 | 4637 | 0.84 | 1.00 | OK |
| training | bf16 | d768 16x48 A5/A48 | 768 | 2190 | 2649 | 2677 | 7535 | 0.83 | 0.99 | OK |
| training | bf16 | d768 16x48 pair256 A5/A48 | 256 | 909.2 | 961.6 | 949.3 | 2753 | 0.95 | 1.01 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | d768 16x48 pair256 A5/A48 | 512 | 1760 | 1850 | 1828 | 4981 | 0.95 | 1.01 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | d768 16x48 pair256 A5/A48 | 768 | 2738 | 2899 | 2801 | 8313 | 0.94 | 1.03 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | d768 16x48 per-sample pair A4 | 256 | 267.3 | 399.3 | 413.7 | 1366 | 0.67 | 0.97 | OK |
| training | bf16 | d768 16x48 per-sample pair A4 | 512 | 460.8 | 778.3 | 866.4 | 4045 | 0.59 | 0.90 | OK |
| training | bf16 | d768 16x48 per-sample pair256 A4 | 256 | 519.0 | 536.6 | 501.9 | 1710 | 0.97 | 1.07 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | d768 16x48 per-sample pair256 A4 | 512 | 1305 | 1330 | 1183 | 5400 | 0.98 | 1.12 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 16x48 A5/A48 | 256 | 1433 | 1714 | 1715 | 2919 | 0.84 | 1.00 | OK |
| training | fp32 | d768 16x48 A5/A48 | 512 | 2729 | 3272 | 3264 | 5186 | 0.83 | 1.00 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 16x48 pair256 A5/A48 | 256 | 1678 | 1768 | 1728 | 3031 | 0.95 | 1.02 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 16x48 pair256 A5/A48 | 512 | 3364 | 3518 | 3415 | 5366 | 0.96 | 1.03 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 16x48 per-sample pair A4 | 256 | 477.2 | 563.1 | 593.9 | 1152 | 0.85 | 0.95 | OK |
| training | fp32 | d768 16x48 per-sample pair A4 | 512 | 894.9 | 1258 | 1351 | 3275 | 0.71 | 0.93 | OK |
| training | fp32 | d768 16x48 per-sample pair256 A4 | 256 | 876.5 | 848.0 | 763.8 | 1310 | 1.03 | 1.11 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 16x48 per-sample pair256 A4 | 512 | 2390 | 2380 | 2084 | 3928 | 1.00 | 1.14 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |

### SWA DiT (atom)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d128 atoms=8L shared-cond | 512 | 67.5 | 89.9 | 579.6 | 2139 | 0.75 | 0.16 | OK |
| inference | bf16 | d128 atoms=8L shared-cond | 1024 | 107.4 | 137.2 | 1824 | 7142 | 0.78 | 0.08 | OK |
| inference | fp32 | atoms=8L shared-cond | 512 | 94.2 | 204.9 | 2108 | 3585 | 0.46 | 0.10 | OK |
| training | bf16 | d128 atoms=8L shared-cond | 512 | 1205 | 2287 | 13481 | 30556 | 0.53 | 0.17 | OK |
| training | bf16 | d128 atoms=8L shared-cond | 1024 | 2246 | 4442 | 52319 | 109537 | 0.51 | 0.08 | OK |
| training | fp32 | atoms=8L shared-cond | 512 | 1961 | 5809 | 70898 | 87634 | 0.34 | 0.08 | OK |

### SWA atom attention (global)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | atoms=8L | 512 | 67.7 | - | 531.1 | 1951 | - | - | OK |
| inference | bf16 | atoms=8L | 1024 | 110.4 | - | 1746 | 6817 | - | - | OK |
| training | bf16 | atoms=8L | 512 | 1407 | - | 11758 | 27106 | - | - | OK |
| training | bf16 | atoms=8L | 1024 | 2737 | - | 49899 | 102831 | - | - | OK |

### AdaLN

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d128 cond128 A5/A48 (atoms) | 1024 | 14.1 | 14.1 | 20.4 | 49.1 | 1.00 | 0.69 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d128 cond128 A5/A48 (atoms) | 4096 | 22.6 | 24.4 | 24.6 | 104.3 | 0.93 | 0.99 | OK |
| inference | bf16 | d128 cond128 A5/A48 (atoms) | 8192 | 30.8 | 34.6 | 32.7 | 178.0 | 0.89 | 1.06 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | d768 cond384 A5/A48 | 384 | 20.4 | 18.3 | 24.4 | 57.1 | 1.12 | 0.75 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d768 cond384 A5/A48 | 768 | 26.5 | 26.5 | 28.5 | 77.6 | 1.00 | 0.93 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d768 cond384 A5/A48 | 1024 | 32.6 | 30.6 | 32.8 | 96.1 | 1.06 | 0.93 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 128 | 20.4 | 18.3 | 22.5 | 45.0 | 1.11 | 0.81 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 384 | 24.4 | 22.4 | 26.6 | 63.4 | 1.09 | 0.84 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 768 | 30.6 | 32.6 | 30.7 | 86.1 | 0.94 | 1.06 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 18.5 | 18.4 | 20.4 | 36.7 | 1.01 | 0.90 | **VIOLATION(cuda_vs_portable)** |
| inference | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 30.7 | 38.8 | 28.7 | 75.7 | 0.79 | 1.35 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 41.0 | 55.3 | 38.9 | 133.1 | 0.74 | 1.42 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond384 A5/A48 | 128 | 28.6 | 26.6 | 22.5 | 28.7 | 1.08 | 1.18 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond384 A5/A48 | 384 | 33.1 | 30.7 | 28.7 | 36.9 | 1.08 | 1.07 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond384 A5/A48 | 768 | 41.0 | 50.7 | 34.9 | 53.4 | 0.81 | 1.45 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 26.6 | 38.9 | 24.6 | 30.7 | 0.68 | 1.58 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 34.8 | 42.9 | 32.7 | 42.9 | 0.81 | 1.31 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 46.8 | 75.7 | 45.0 | 65.5 | 0.62 | 1.68 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | d128 cond128 A5/A48 (atoms) | 1024 | 96.2 | 106.3 | 135.0 | 734.3 | 0.90 | 0.79 | OK |
| training | bf16 | d128 cond128 A5/A48 (atoms) | 4096 | 290.7 | 324.4 | 361.2 | 1844 | 0.90 | 0.90 | OK |
| training | bf16 | d128 cond128 A5/A48 (atoms) | 8192 | 524.2 | 587.6 | 693.2 | 3471 | 0.89 | 0.85 | OK |
| training | bf16 | d768 cond384 A5/A48 | 384 | 178.3 | 200.6 | 219.0 | 700.4 | 0.89 | 0.92 | OK |
| training | bf16 | d768 cond384 A5/A48 | 768 | 307.1 | 357.3 | 378.8 | 1452 | 0.86 | 0.94 | OK |
| training | bf16 | d768 cond384 A5/A48 | 1024 | 389.0 | 455.7 | 493.5 | 1898 | 0.85 | 0.92 | OK |
| training | bf16 | d768 cond768 A5/A48 (ESMFold2) | 128 | 102.3 | 112.5 | 125.0 | 309.2 | 0.91 | 0.90 | OK |
| training | bf16 | d768 cond768 A5/A48 (ESMFold2) | 384 | 219.0 | 249.8 | 272.4 | 868.3 | 0.88 | 0.92 | OK |
| training | bf16 | d768 cond768 A5/A48 (ESMFold2) | 768 | 378.8 | 450.5 | 489.3 | 1684 | 0.84 | 0.92 | OK |
| training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 147.3 | 155.5 | 149.5 | 664.4 | 0.95 | 1.04 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 446.5 | 493.5 | 503.7 | 1527 | 0.90 | 0.98 | OK |
| training | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 811.8 | 919.6 | 948.2 | 2814 | 0.88 | 0.97 | OK |
| training | fp32 | d768 cond384 A5/A48 | 128 | 133.1 | 153.5 | 124.9 | 221.1 | 0.87 | 1.23 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond384 A5/A48 | 384 | 298.9 | 365.7 | 301.1 | 586.7 | 0.82 | 1.21 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond384 A5/A48 | 768 | 485.3 | 690.1 | 524.2 | 1203 | 0.70 | 1.32 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 171.9 | 214.9 | 167.8 | 264.1 | 0.80 | 1.28 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 382.8 | 530.3 | 395.1 | 749.5 | 0.72 | 1.34 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 692.1 | 989.1 | 739.2 | 1416 | 0.70 | 1.34 | **VIOLATION(portable_vs_pytorch)** |

### ConditionedTransition

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | d768 A5/A48 | 384 | 57.3 | 53.3 | 61.4 | 104.5 | 1.08 | 0.87 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | d768 A5/A48 | 768 | 69.7 | 83.9 | 75.7 | 143.4 | 0.83 | 1.11 | **VIOLATION(portable_vs_pytorch)** |
| inference | bf16 | d768 A5/A48 | 1024 | 79.9 | 100.3 | 94.4 | 180.1 | 0.80 | 1.06 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 38.9 | 51.2 | 45.0 | 67.5 | 0.76 | 1.14 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 59.4 | 128.9 | 73.7 | 131.0 | 0.46 | 1.75 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 84.0 | 213.0 | 110.6 | 227.4 | 0.39 | 1.93 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond384 A5/A48 | 128 | 69.7 | 55.3 | 55.1 | 65.5 | 1.26 | 1.00 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond384 A5/A48 | 384 | 98.3 | 79.8 | 79.7 | 100.3 | 1.23 | 1.00 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond384 A5/A48 | 768 | 120.9 | 125.0 | 114.7 | 147.4 | 0.97 | 1.09 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 71.7 | 67.7 | 59.2 | 69.5 | 1.06 | 1.14 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 104.5 | 93.9 | 88.0 | 106.4 | 1.11 | 1.07 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 133.1 | 155.7 | 127.0 | 161.6 | 0.85 | 1.23 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | d768 A5/A48 | 384 | 628.7 | 685.1 | 600.1 | 1352 | 0.92 | 1.14 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | d768 A5/A48 | 768 | 1124 | 1253 | 1122 | 2679 | 0.90 | 1.12 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | d768 A5/A48 | 1024 | 1452 | 1638 | 1442 | 3496 | 0.89 | 1.14 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 456.0 | 568.4 | 444.4 | 1065 | 0.80 | 1.28 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 1461 | 1981 | 1568 | 3115 | 0.74 | 1.26 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 2686 | 3723 | 2899 | 5697 | 0.72 | 1.28 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond384 A5/A48 | 128 | 497.7 | 497.6 | 473.2 | 638.8 | 1.00 | 1.05 | **VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond384 A5/A48 | 384 | 1207 | 1263 | 1232 | 1746 | 0.96 | 1.02 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond384 A5/A48 | 768 | 2193 | 2337 | 2241 | 3350 | 0.94 | 1.04 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 552.7 | 571.3 | 532.5 | 693.2 | 0.97 | 1.07 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 1332 | 1459 | 1361 | 1939 | 0.91 | 1.07 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 2481 | 2696 | 2528 | 3676 | 0.92 | 1.07 | **VIOLATION(portable_vs_pytorch)** |

### LayerNorm (pair)

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | x[1,L,L,128] | 384 | 20.4 | 18.3 | 22.4 | 161.7 | 1.11 | 0.82 | OK |
| inference | bf16 | x[1,L,L,128] | 768 | 53.1 | 51.3 | 65.4 | 622.4 | 1.04 | 0.78 | OK |
| inference | bf16 | x[1,L,L,256] | 384 | 30.6 | 28.7 | 28.6 | 192.5 | 1.07 | 1.00 | **VIOLATION(portable_vs_pytorch)** |
| inference | fp32 | x[1,L,L,128] | 384 | 30.5 | 28.5 | 30.5 | 161.7 | 1.07 | 0.94 | OK |
| training | bf16 | x[1,L,L,128] | 384 | 112.4 | 51.2 | 67.5 | 339.8 | 2.20 | 0.76 | OK |
| training | bf16 | x[1,L,L,128] | 768 | 236.3 | 194.5 | 184.3 | 1252 | 1.22 | 1.06 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 | x[1,L,L,256] | 384 | 108.4 | 116.6 | 100.3 | 407.5 | 0.93 | 1.16 | **VIOLATION(portable_vs_pytorch)** |
| training | fp32 | x[1,L,L,128] | 384 | 126.8 | 73.8 | 96.1 | 354.2 | 1.72 | 0.77 | OK |

### RMSNorm

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | x[48,L,768] | 384 | 26.5 | 16.1 | 16.3 | 30.7 | 1.64 | 0.99 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 | x[48,L,768] | 1024 | 53.2 | 32.6 | 32.7 | 71.5 | 1.63 | 1.00 | **VIOLATION(cuda_vs_portable)** |
| training | bf16 | x[48,L,768] | 384 | 59.4 | 38.9 | 49.1 | 126.9 | 1.53 | 0.79 | **VIOLATION(cuda_vs_portable)** |
| training | bf16 | x[48,L,768] | 1024 | 120.8 | 75.9 | 100.3 | 384.9 | 1.59 | 0.76 | **VIOLATION(cuda_vs_portable)** |

### pair_embed_geom

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 out, fp32 geometry | B16 | 128 | 118.7 | 122.8 | 3891 | 1964 | 0.97 | 0.06 | OK |
| inference | bf16 out, fp32 geometry | B16 | 256 | 364.4 | 386.9 | 3078 | 7540 | 0.94 | 0.13 | OK |
| inference | bf16 out, fp32 geometry | B16 | 384 | 759.7 | 831.4 | 7209 | 16837 | 0.91 | 0.12 | OK |
| inference | bf16 out, fp32 geometry | B48 | 128 | 278.3 | 296.8 | 2470 | 5688 | 0.94 | 0.12 | OK |
| inference | bf16 out, fp32 geometry | B48 | 256 | 932.6 | 1079 | 8925 | 22487 | 0.86 | 0.12 | OK |
| training | bf16 out, fp32 geometry | B16 | 128 | 332.0 | 1418 | 4120 | 3272 | 0.23 | 0.43 | OK |
| training | bf16 out, fp32 geometry | B16 | 256 | 1079 | 5042 | 4057 | 12702 | 0.21 | 1.24 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 out, fp32 geometry | B16 | 384 | 2352 | 9667 | 8959 | 28424 | 0.24 | 1.08 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 out, fp32 geometry | B48 | 128 | 825.2 | 3309 | 3089 | 9567 | 0.25 | 1.07 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 out, fp32 geometry | B48 | 256 | 2954 | 12816 | 11868 | 37931 | 0.23 | 1.08 | **VIOLATION(portable_vs_pytorch)** |

### pair_embed_ca

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 out, fp32 geometry | B16 | 128 | 36.7 | 34.6 | 959.6 | 899.0 | 1.06 | 0.04 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 out, fp32 geometry | B16 | 256 | 100.2 | 102.2 | 909.0 | 3533 | 0.98 | 0.11 | OK |
| inference | bf16 out, fp32 geometry | B16 | 384 | 200.5 | 214.9 | 2045 | 7924 | 0.93 | 0.11 | OK |
| inference | bf16 out, fp32 geometry | B48 | 128 | 79.7 | 79.7 | 706.4 | 2652 | 1.00 | 0.11 | **VIOLATION(cuda_vs_portable)** |
| inference | bf16 out, fp32 geometry | B48 | 256 | 278.3 | 282.4 | 2636 | 10562 | 0.99 | 0.11 | OK |
| training | bf16 out, fp32 geometry | B16 | 128 | 178.3 | 342.8 | 1157 | 2082 | 0.52 | 0.30 | OK |
| training | bf16 out, fp32 geometry | B16 | 256 | 557.2 | 1113 | 1596 | 8123 | 0.50 | 0.70 | OK |
| training | bf16 out, fp32 geometry | B16 | 384 | 1198 | 1934 | 3713 | 18188 | 0.62 | 0.52 | OK |
| training | bf16 out, fp32 geometry | B48 | 128 | 448.3 | 706.4 | 1216 | 6110 | 0.63 | 0.58 | OK |
| training | bf16 out, fp32 geometry | B48 | 256 | 1539 | 2522 | 5441 | 24230 | 0.61 | 0.46 | OK |

### pair_embed_adaln

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 out, fp32 geometry | B16 | 128 | 131.2 | 194.5 | 1368 | 2235 | 0.67 | 0.14 | OK |
| inference | bf16 out, fp32 geometry | B16 | 256 | 340.0 | 718.7 | 1184 | 8892 | 0.47 | 0.61 | OK |
| inference | bf16 out, fp32 geometry | B16 | 384 | 839.7 | 1590 | 2642 | 19982 | 0.53 | 0.60 | OK |
| inference | bf16 out, fp32 geometry | B48 | 128 | 284.7 | 540.5 | 909.2 | 6670 | 0.53 | 0.59 | OK |
| inference | bf16 out, fp32 geometry | B48 | 256 | 871.3 | 2106 | 3437 | 26592 | 0.41 | 0.61 | OK |
| training | bf16 out, fp32 geometry | B16 | 128 | 397.1 | 1007 | 1984 | 4082 | 0.39 | 0.51 | OK |
| training | bf16 out, fp32 geometry | B16 | 256 | 1262 | 3155 | 3216 | 15928 | 0.40 | 0.98 | OK |
| training | bf16 out, fp32 geometry | B16 | 384 | 3198 | 8108 | 7212 | 35670 | 0.39 | 1.12 | **VIOLATION(portable_vs_pytorch)** |
| training | bf16 out, fp32 geometry | B48 | 128 | 931.8 | 2521 | 2564 | 11949 | 0.37 | 0.98 | OK |
| training | bf16 out, fp32 geometry | B48 | 256 | 3550 | 9339 | 10176 | 47421 | 0.38 | 0.92 | OK |

### token_pair_init

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 in, fp32 out | B16 | 128 | 71.5 | 51.0 | 444.5 | 790.4 | 1.40 | 0.11 | **VIOLATION(cuda_vs_portable)** |
| inference | fp32 in, fp32 out | B16 | 256 | 203.6 | 169.9 | 1355 | 2889 | 1.20 | 0.13 | **VIOLATION(cuda_vs_portable)** |
| inference | fp32 in, fp32 out | B16 | 384 | 372.4 | 370.5 | 3061 | 6405 | 1.01 | 0.12 | **VIOLATION(cuda_vs_portable)** |
| inference | fp32 in, fp32 out | B48 | 128 | 176.0 | 132.9 | 1022 | 2183 | 1.32 | 0.13 | **VIOLATION(cuda_vs_portable)** |
| inference | fp32 in, fp32 out | B48 | 256 | 559.1 | 491.3 | 3967 | 8524 | 1.14 | 0.12 | **VIOLATION(cuda_vs_portable)** |
| training | fp32 in, fp32 out | B16 | 128 | 215.0 | 483.1 | 663.6 | 1010 | 0.45 | 0.73 | OK |
| training | fp32 in, fp32 out | B16 | 256 | 636.7 | 1462 | 2112 | 3666 | 0.44 | 0.69 | OK |
| training | fp32 in, fp32 out | B16 | 384 | 1268 | 3129 | 4719 | 8073 | 0.41 | 0.66 | OK |
| training | fp32 in, fp32 out | B48 | 128 | 568.2 | 1127 | 1528 | 2773 | 0.50 | 0.74 | OK |
| training | fp32 in, fp32 out | B48 | 256 | 1809 | 3891 | 6181 | 10756 | 0.46 | 0.63 | OK |

### smooth_lddt

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 | A16 B1 | 4096 | 610.3 | 751.6 | 335155 | 341313 | 0.81 | 0.00 | OK |
| inference | fp32 | A48 B1 | 1024 | 172.0 | 165.9 | 60503 | 61648 | 1.04 | 0.00 | **VIOLATION(cuda_vs_portable)** |
| inference | fp32 | A48 B1 | 2048 | 494.6 | 546.9 | 241628 | 246113 | 0.90 | 0.00 | OK |
| training | fp32 | A16 B1 | 4096 | 1337 | 2000 | 527886 | 538467 | 0.67 | 0.00 | OK |
| training | fp32 | A48 B1 | 1024 | 333.8 | 411.7 | 95627 | 97932 | 0.81 | 0.00 | OK |
| training | fp32 | A48 B1 | 2048 | 1066 | 1451 | 383907 | 392346 | 0.73 | 0.00 | OK |

### sampler augment_noise

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 | B5 | 4096 | 8.1 | 10.1 | 22.5 | 71.7 | 0.80 | 0.45 | OK |
| inference | fp32 | B5 | 12288 | 8.2 | 10.1 | 24.5 | 90.1 | 0.81 | 0.41 | OK |
| inference | fp32 | B5 | 24576 | 10.1 | 12.2 | 30.8 | 77.8 | 0.83 | 0.39 | OK |

### sampler align_euler

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | fp32 | B5 | 4096 | 12.3 | 18.4 | 106.5 | 307.2 | 0.67 | 0.17 | OK |
| inference | fp32 | B5 | 12288 | 16.4 | 18.4 | 102.2 | 493.6 | 0.89 | 0.18 | OK |
| inference | fp32 | B5 | 24576 | 18.4 | 22.5 | 106.5 | 665.5 | 0.82 | 0.21 | OK |

### sampler pool_atoms

| mode | dtype | shape | L | CUDA | Triton | PyTorch compiled | PyTorch eager+graph | c/t | t/p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | bf16 | B5 c768 | 4096 | 14.2 | 16.3 | 73.6 | 210.8 | 0.87 | 0.22 | OK |
| inference | bf16 | B5 c768 | 12288 | 22.4 | 26.5 | 206.7 | 624.6 | 0.85 | 0.13 | OK |
| inference | bf16 | B5 c768 | 24576 | 34.7 | 42.9 | 405.4 | 1217 | 0.81 | 0.11 | OK |

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
dropout keep); their drivers now call both sides (commit 2024760a) and the other side was built after the ladder run (7f91dfe1: 128 units,
110 ok, the same 18 failing shapes). The ladder ran before that second build: a key it met that the cache lacked was tuned at
run time over the miss policy's bounded subset (`settings.autotune_miss_cap`, 24 configs) before timing.

## Reproduce

```
# tune (GPU 2 excluded); then one ladder process per GPU over a partition of the groups
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 miniworld-engine build all --kernels portable --gpus all --units-per-gpu 2 --shards <dir>
CUDA_VISIBLE_DEVICES=<g> python benchmarks/runners/ladder.py run --group '<regex>' --repeats 3 --out <dir>/gpu<g>
python benchmarks/runners/ladder.py report --out <dir>/gpu0 <dir>/gpu1 ... --csv ladder.csv --md ladder.md
```

