# miniworld-engine 2.5.13 (draft)

Base: 2.5.12 (7c544cf2f), release candidate `rc/2.5.13`. H100 only, one change to the fp16 TriangleAttention inference core
(`kernels/triangle_attention/cuda/h100_sources/triattn_h16_sm90.cu`): **bit-identical**. No module `version`, no `backend_versions`
bump. A100 / RTX 5090: no source they compile changed.

## Changes

| # | Change | Outputs | `backend_versions["cuda:h100_sm90"]` | Measured (module, CUDA-graph replay, same process) |
|---|---|---|---|---|
| 1 | `TA_RC=1` (new default): the consumer loop of the fp16 core runs key tiles 0 .. nk - 2, the last key tile is peeled after it, and the gate prefetch (`gpre`) is loaded right before that last tile, outside the loop. The 2.5.12 loop's `j + 1 < nk` test kept the other row's retired probabilities live through the exponentials (32 registers) and made `gpre` loop-carried (16); ptxas now uses those registers to hoist the next bias ldmatrix. Same operations in the same order. One kernel body, so every shipped instantiation gets it (32 / 16-channel heads, POLY / POLY16, ORD1, SPEC / LIST, BTR, CL, masked / `key_tiles`, gated / ungated). `TA_RC=0` builds the 2.5.12 loop (used only as the in-process baseline) | bit-identical | - | 64:4:16 cool 0.92-0.97x, other configs 0.97-1.01x |
| 2 | `docs/h100/triangle_attention.md`: "Register / concurrency experiments (2026-11)" (the kept result plus the "tried, no gain" list). The experiment macros (`TA_CREG`, `TA_NCW`, `TA_QSS`, `TA_CTA2`, `TA_K64`) stay on `exp/ta-regs` and are not in the release | - | - | - |

## Bit-identity (H100, RC tree vs 2.5.12 tree 7c544cf2f, separate processes, sha1 per tensor)

326 registry-built cases, cuda backend, dropout off: TriangleAttention rows 71-76 (64:2:32, 64:4:16, 64:4:32, 128:4:32, 256:8:32,
384:12:32), TriangleAttentionAF3 77-80 (64 / 128 / 256 / 512), RF3 81-82, Chai1 83-84, Chai1Confidence 85, ParallelResidual 162, bias-only
TriangleAttention 188. Both nodes, eval at L128 / 256 / 384 / 512 / 768 / 1024 / 1536 / 2048 with B1 / B2 and with the registry key mask or
none; training at L256 (rows 72, 79, 80, 188, 241-252, 283) plus 128:4:32 at L512 B2. **326 / 326 identical**; the RC built with
`TA_RC=0` also equals 2.5.12 on all 291 cases it ran.

* 196 eval cases launch the fp16 core (counted per case): 64:2:32, 64:4:16, 64:4:32, 128:4:32, 256:8:32, 384:12:32 (except L256 B2),
  AF3 64 / 128 / 256, RF3 128, Chai1 64. They do not launch it: AF3 512, RF3 64 (d_hidden 256), Chai1 256, Chai1Confidence 256 (heads of
  64 channels: the `h100_dh` core), the bias-only path, ParallelResidual.
* **Training never launches it** (0 calls in every training case); training outputs and gradients are identical.
* Not run (out of memory in both trees): AF3 512 at L2048 B2 eval; 256:8:32 training at L512 B2.

## Speed (2.5.12 loop vs RC, same process, CUDA-graph replay)

H100 80GB HBM3, **power-capped at 700 W**: under sustained load the SM clock drops 15-18 %, so cool and sustained are reported
separately and cannot be compared with the benchmark pages. All three variants run in one process with the same module and inputs, and
each variant's graph is captured with its own core: "2.5.12" = the RC core built `-DTA_RC=0` (equal to the 2.5.12 tree's bytes, above);
"RC" = the default build; "ctl" = the RC core built a second time with an unused define (same SASS, separate cubin): this is the noise
control. cool = median of 4 rounds (1.5 s idle, about 20 ms of replays), order rotating. sustained = 1.5 s idle, 1 s burn, then the mean
over the next 0.5 s. Registry inputs, starting node, no mask, B1, eval. ms.
Noise control ctl / RC: cool 0.993-1.019, sustained 0.977-1.025.

| Module (row) | L | 2.5.12 cool | RC cool | ratio cool | 2.5.12 sus | RC sus | ratio sus |
|---|---|---|---|---|---|---|---|
| TriAttn 64:4:16 (72) | 256 | 0.0886 | 0.0856 | 0.966 | 0.0889 | 0.0864 | 0.972 |
| | 512 | 0.4072 | 0.3759 | 0.923 | 0.4043 | 0.3779 | 0.935 |
| | 768 | 1.0705 | 0.9952 | 0.930 | 1.0758 | 1.0182 | 0.946 |
| | 1024 | 2.2245 | 2.0649 | 0.928 | 2.2664 | 2.1460 | 0.947 |
| | 1536 | 6.5082 | 6.0380 | 0.928 | 6.6844 | 6.4336 | 0.962 |
| | 2048 | 14.1112 | 13.3818 | 0.948 | 14.8517 | 14.6673 | 0.988 |
| TriAttnAF3 64 (77) | 256 | 0.0882 | 0.0853 | 0.967 | 0.0877 | 0.0848 | 0.967 |
| | 512 | 0.4071 | 0.3775 | 0.927 | 0.4083 | 0.3843 | 0.941 |
| | 768 | 1.0735 | 0.9925 | 0.925 | 1.1114 | 1.0307 | 0.927 |
| | 1024 | 2.2244 | 2.0677 | 0.930 | 2.2836 | 2.1888 | 0.958 |
| | 1536 | 6.4984 | 6.0669 | 0.934 | 6.8040 | 6.4793 | 0.952 |
| | 2048 | 14.0834 | 13.3513 | 0.948 | 15.0251 | 14.4959 | 0.965 |
| TriAttn 64:2:32 (71) | 256 | 0.0653 | 0.0646 | 0.989 | 0.0657 | 0.0653 | 0.994 |
| | 512 | 0.2755 | 0.2677 | 0.972 | 0.2815 | 0.2762 | 0.981 |
| | 1024 | 1.3065 | 1.2919 | 0.989 | 1.4086 | 1.4079 | 1.000 |
| | 2048 | 8.0569 | 7.9369 | 0.985 | 9.0284 | 8.8832 | 0.984 |
| TriAttn 64:4:32 (73) | 256 | 0.1061 | 0.1068 | 1.007 | 0.1083 | 0.1087 | 1.004 |
| | 1024 | 2.4208 | 2.3802 | 0.983 | 2.6793 | 2.6021 | 0.971 |
| | 2048 | 15.5225 | 15.3140 | 0.987 | 17.4000 | 17.1061 | 0.983 |
| TriAttn 128:4:32 (74) | 256 | 0.1038 | 0.1010 | 0.973 | 0.1051 | 0.1012 | 0.963 |
| | 512 | 0.4385 | 0.4332 | 0.988 | 0.4649 | 0.4614 | 0.992 |
| | 768 | 1.1601 | 1.1326 | 0.976 | 1.2437 | 1.2278 | 0.987 |
| | 1024 | 2.4068 | 2.3477 | 0.975 | 2.6034 | 2.5721 | 0.988 |
| | 1536 | 7.0234 | 6.8830 | 0.980 | 7.8288 | 7.6789 | 0.981 |
| | 2048 | 15.3820 | 15.1787 | 0.987 | 17.0700 | 17.0768 | 1.000 |
| TriAttn 256:8:32 (75) | 256 | 0.2324 | 0.2303 | 0.991 | 0.2412 | 0.2394 | 0.993 |
| | 1024 | 5.3380 | 5.2476 | 0.983 | 6.0028 | 5.9253 | 0.987 |
| | 2048 | 33.0469 | 32.4556 | 0.982 | 37.0552 | 36.2168 | 0.977 |
| TriAttn 384:12:32 (76) | 256 | 0.3995 | 0.3999 | 1.001 | 0.4218 | 0.4193 | 0.994 |
| | 1024 | 8.8276 | 8.7217 | 0.988 | 9.8108 | 9.7183 | 0.991 |
| | 2048 | 54.8826 | 54.5040 | 0.993 | 60.7947 | 60.0229 | 0.987 |
| TriAttnAF3 128 (78) | 256 | 0.1037 | 0.1002 | 0.966 | 0.1052 | 0.1018 | 0.968 |
| | 1024 | 2.4127 | 2.3481 | 0.973 | 2.6553 | 2.6301 | 0.991 |
| | 2048 | 15.3553 | 15.1211 | 0.985 | 17.5248 | 17.1571 | 0.979 |
| TriAttnAF3 256 (79) | 256 | 0.2321 | 0.2313 | 0.997 | 0.2436 | 0.2418 | 0.993 |
| | 1024 | 5.3591 | 5.2879 | 0.987 | 6.1694 | 6.0798 | 0.985 |
| | 2048 | 33.2747 | 32.8226 | 0.986 | 37.6940 | 37.4050 | 0.992 |
| TriAttnRF3 128 (82) | 256 | 0.1069 | 0.1040 | 0.973 | 0.1098 | 0.1064 | 0.969 |
| | 1024 | 2.4269 | 2.3641 | 0.974 | 2.7050 | 2.6487 | 0.979 |
| | 2048 | 15.4647 | 15.2535 | 0.986 | 17.4711 | 17.2307 | 0.986 |
| TriAttnChai1 64 (83) | 256 | 0.1067 | 0.1072 | 1.005 | 0.1086 | 0.1090 | 1.004 |
| | 1024 | 2.4355 | 2.3944 | 0.983 | 2.6267 | 2.6142 | 0.995 |
| | 2048 | 15.5953 | 15.4008 | 0.988 | 17.3391 | 17.0754 | 0.985 |

66 cells in all (11 rows x L256 / 512 / 768 / 1024 / 1536 / 2048; table abridged). The full grid is in
`~/miniworld-engine-scratch/rc2513-work/results/speed_table.txt`. Summary: heads of 16 channels (64:4:16, AF3 64) cool 0.92-0.97x,
sustained 0.93-0.99x; 32-channel heads cool 0.97-1.01x, sustained 0.96-1.00x (the 1.00-1.01 cells, all at L256, sit inside the noise
control).

## Judge (version-based tokens)

The outputs are bit-identical, so no version is bumped. `judge owed` after the change: H100, A100 and RTX 5090 each show accuracy 0 / speed 0
owed and 0 pending (17719 carried). The H100 TriAttn speed tokens were measured on the 2.5.12 core: they are stale but conservative
(this change only makes it faster). There are two options, and the user decides: (a) leave them; (b) bump
`TriangleAttention` / `TriangleAttentionAF3` / `TriangleAttentionRF3` / `TriangleAttentionChai1`
`backend_versions["cuda:h100_sm90"]` to re-measure speed. That re-measures accuracy too, on every case of those classes.

## Tests and checks

* GPU, one 4-GPU work-queue job (one file per free GPU), 11 items, all pass: `test_accuracy_smoke.py -k TriangleAttention` (22),
  `test_determinism.py -k TriangleAttention` (2), `test_triattn_h100_variants_infer.py` (17), `test_triattn_h100_tail_gate.py` (15),
  `test_triattn_h100_key_tiles.py` (15), `test_triattn_h16_core_dh16.py` (2), `test_triattn_h100_front_y16.py` (14),
  `test_triattn_h100_front_lnb.py` (6), `test_refiner_sizes.py` (18 passed, 8 skipped), `test_cuda_refuses_not_raises.py` (8),
  `test_anthropic_kit_rows_h100.py` (2 skipped: no kit present).
* CPU: `test_requirements` (1167), `test_prebuild_plan` (4), `test_judge` (20) pass. No Python file changed (ruff / pyright: nothing
  to check).
* Kernels came from a copy of the 2.5.12 RC prebuild cache, which is content-keyed: only the changed core was built.

## Packaging: prebuilt artifacts whose keys change

`_prebuild --list` / `--check` against the 2.5.12 tree (same artifact counts: sm90 513, sm80 72, sm120 64; `--check` shows no new
finding on any arch):

* **sm90**: `triattn_h100_core_h16_v3` re-keyed (a980ab15 -> dc615aed), same name. Nothing else changed.
* **sm80 / sm120**: no change (`triattn_h16_sm90.cu` is compiled only for sm90a).

## A100 (cand/a100-port): the register-resident TriAttn core below L 512

Port check of this week's H100 items to the A100 (sm_80):

* **TA_RC (peeled last key tile)**: the cause does not exist on sm_80. No A100 core has a last-tile branch in its key loop or a
  loop-carried gate prefetch (the gated cores read g after the loop); the rf cores have no spills (`cuobjdump -res-usage`: 199 / 224 /
  166 registers at 64 / 32 / 16-channel heads, stack 0) and their occupancy is bound by shared memory (2 CTAs / SM) as much as by
  registers. Not ported.
* **64-channel fp16 core (H100 cand/ta-d64)**: the A100 rf core already has the main ideas (bias as the score accumulator's C operand,
  no softmax shift, gate after the loop) and runs at parity with SDPA flash *without* bias (L 512-2048, 8 heads: 152-169 TFLOP/s,
  0.95-1.01x flash, 3x SDPA efficient / cuDNN with the bias; ncu at L 1024: tensor pipe 60 %, FMA 31 %, XU 28 %). sm_80 has no
  bf16x2 exponential and its fp16-accumulate mma is not faster than fp32-accumulate; an fp16 front would be the whole H100 rewrite
  for little room. Not ported. **What paid instead**: the generalised core (`attn2_ops.cu`) used the rf core only from L 512; below
  that its attn2 core was 1.4-1.6x slower at 64-channel heads (79 TFLOP/s at L 256 vs 152+ for rf).
* **Transition D 768**: the A100 route has no such entry. D 768 inference runs `fused_wide_sm80` without saves (a / b / h not kept,
  weights cast and cached once): 1.13-1.25x the bare cuBLAS GEMM sum (xn [Wa; Wb]^T + h Ws^T, no LayerNorm / SwiGLU / residual) at
  256-8192 rows, 1.4-1.5x faster than eager torch. Not ported.

Change (sm_80 build only; the sm_120a build of the same source keeps its floors behind `TRIATTN_SM120`): the rf core from L 128
(64-channel heads), L 256 (32-channel heads; 0.92-1.00x at L 128) and L 256 for 16-channel fixed-shift heads (was 1024; 0.88-0.91x at
L 128). Outputs change at those L (within the rf core's error, which the 512+ cases already carry).

* Core alone (A100 80GB PCIe, CUDA graph, interleaved, rf vs attn2 in one process): 64-ch 1.39-1.62x (L 128-384), 32-ch 1.27-1.44x
  (L 256 / 384), 16-ch 1.04-1.12x (L 256-768).
* Module (eval, bf16 pair, master weights, 1/8 masked keys; new rule vs the rc rule in one process, same-graph control 0.99-1.01x):
  AF3 512 x 8 1.05-1.12x, RF3 64 x 4 x 64 1.06-1.19x, RF3 128 x 4 1.03-1.10x (L >= 256), Chai-1 256 x 4 1.05-1.14x, Chai-1
  confidence 1.04-1.12x, Chai-1 64 x 4 x 32 ending node 1.04-1.10x (L >= 256), d_pair 64 x 4 x 16 (TriangleAttention / AF3)
  1.02-1.06x (L 256-768). Unchanged (bitwise): d_pair 64 x 2, 64 x 4 x 32 (starting), 128 x 4, 256 x 8, 384 x 12, AF3 128 / 256.
  Error of the update vs the fp64 module unchanged (rms within 1 %).
* Accuracy (`measure/accuracy.py`, rows 71-85, eval, L 128-768, both trees): 180 / 180 cases same light; the 15 whose judge
  error moved are GREEN (new / old 0.90-1.18, worst err / tol 0.53). r76 L 768: no record in either tree.
* H100: no sm_90 source changed; 66 TriangleAttention outputs (11 configs x L 128 / 256 / 512 x 2 nodes) sha1-identical between
  the rc tree and this one (separate processes).
* Versions: `cuda:a100_sm80` `_TriangleAttentionBase` 5 -> 6 (AF3, RF3, Chai-1, Chai-1 confidence), `TriangleAttention` 6 -> 7.
  Judge owed: a100_sm80 460 accuracy + 460 speed (not re-measured); h100_sm90 0; 5090_sm120 0.
* Prebuilt keys: sm80 `triattn_sm80_core2` (df48da52 -> 4a919b15), sm120 `triattn_sm120_core2` (170a3638 -> e824be37; the same
  behavior); sm90a none. `_prebuild --check` findings unchanged (same as the base tree).
* Tests (A100): `test_triattn_sm80_rf_core.py` (new `test_rf_floor_default_rule`, `test_rf_core_attn2` also at the floor L) and
  `test_determinism.py -k riangle`: 47 passed, 8 skipped (H100-only). CPU: `test_requirements` 1167 passed, ruff / pyright clean.
