# miniworld-engine 2.5.11

Base: 2.5.10 (c96f2fac4) + branch `opm-win` (OuterProductMean H100 inference at large MSA depth). H100 only. Module `version` is not bumped;
only `OuterProductMean.backend_versions["cuda:h100_sm90"]` (new entry: 1 -> 3), so anthropic / cueq / triton / pytorch tokens and the
A100 / RTX 5090 tokens stay valid. The 2.5.10 TriMul / TriAttn changes are untouched by the merge (sources byte-identical to 2.5.10).

## Changes (OuterProductMean, H100 `cuda`)

| # | Change | Outputs | Measured |
|---|---|---|---|
| 1 | AF3-order inference at depth S > 4096 takes the wide TMA prologue (the native prologue streamed at 0.35 TB/s) | changed at S > 4096 (version 2) | native prologue 9.5 ms at S 16384 / L 768 |
| 2 | Inference at S > 4096: cuBLAS product per chunk of i rows + the CUDA epilogue (`opm_train._product_route_deep`, `_product_epilogue`; O workspace bounded by `_O_CHUNK_BYTES` = 2 GiB) instead of the fused kernel; also `composed.biased_wide` (RoseTTAFold3 biased forms) | changed at S > 4096 (O rounded to bf16 as one product): version 3 | 2-6 % faster cool, 3-11 % sustained at S 8192 / 16384 (kernel A/B, L 256-768) |
| 3 | Pair-count divisor table by one fp32-output cuBLAS GEMM of the prologue's bf16 mask (`_count_table`) instead of the epilogue's per-CTA popcount of the mask bit rows | bitwise the count kernel's table | count 0.14 ms -> 44 us; epilogue 0.85 -> 0.47 ms at S 16384 / L 768 |
| 4 | cuBLASLt heuristic algorithm choice for the product at padded N 256 / 384 (`opm_lt_mm`, `_LT_ALGO = {256: 1, 384: 1}`; new extension `miniworld_opm_lt_mm`) | bitwise the same product | 2 % cool, 1-4 % sustained at N 256 |
| 5 | Wide prologue: `setmaxnreg` rebalance (TMA warpgroup 40, LayerNorm / projection warpgroups 152; the 112 B stack frame is gone), tiles in groups of 64 tokens | bitwise the same | 1.53 -> 1.33 ms at S 16384 / L 768 |
| 6 | Fused kernel (S <= 4096) back to the opm-bigs code (STAGES 5 / WST 2 / RASTER 16); the cluster / commit-group template was rejected (2-5 % slower cool at S <= 2048) | unchanged at S <= 4096 | no regression at S <= 2048 |

Details: `docs/h100/outer_product_mean.md`, `benchmarks/opm_win/README.md` (protocol, findings, rejected variants), `benchmarks/opm_bigs`,
`benchmarks/msa_depth`.

## Speed vs Anthropic (fair protocol, `benchmarks/opm_win/results/table.md`)

H100 80GB HBM3, **power-capped at 700 W**: after ~50-80 ms of load the SM clock falls from 1.98 GHz to 1.3-1.5 GHz and job-to-job drift is
+-1-2 %, so every number is a ratio to the PyTorch backend measured in the same process; **cool** = 2 s idle then a short burst (median of 3
reps), **sustained** = 2 s idle, 1 s of back-to-back replays, then timed calls (mean of 3 reps). Lead = (PyTorch_kit / Anthropic) /
(PyTorch_engine / cuda), > 1: the engine is faster. Rows of `requirements_module.csv`, L 256-768, inference, S 8192 / 16384:

| Rows | Form | Lead before (2.5.10) | Lead after (2.5.11) |
|---|---|---|---|
| 57 / 58 | Boltz order (58: + projection biases) | 1.01-1.18 | 1.08-1.22 cool, 1.09-1.26 sustained |
| 55 / 63 | AF3 order (63: + `count_offset`) | 0.95-1.00 cool | 1.00-1.05 cool, 1.01-1.11 sustained; cells >= 1.03: 22 / 32 cool, 29 / 32 sustained |
| 56 | AF3 order + biases (no Anthropic kernel) | - | vs the engine's Triton backend 0.996-1.025 cool, 0.975-1.16 sustained |

S <= 2048 (fused kernel): after / before speed-up over PyTorch mean 1.005 cool (min 0.981), 1.003 sustained (min 0.973): within the
protocol's spread. The remaining gap at S 16384 is the cuBLAS product itself (90-93 % of the call in every competitor, power-bound).

## Judge (version-based tokens)

* H100 after the bump: 222 accuracy + 222 speed cases owed, all `adapter:cuda 1->3` of OuterProductMean. Re-measured with
  `tools/slurm/cycle.sh` (1 cycle, 4 groups, 888 tokens issued). Accuracy lights old -> new: GREEN -> GREEN 214, skipped -> GREEN 6 (L 2048
  cells measured for the first time), YELLOW -> YELLOW 1 (a training gradient, same error 8.04e-3 as before: training is unchanged), skipped ->
  skipped 1 (the PyTorch reference runs out of memory at d_pair 384 L 2048); **RED 0**. Speed (speed-up vs PyTorch, new / old): median 1.001
  over 222 cells, range 0.92-1.15 (the judge's depths are <= 1024, where the code path is unchanged: run-to-run spread).
* The A100 / RTX 5090 OPM sources (`opm_sm80`, `_msa_card` sm_120a builds) are not touched; the sm80 / sm120 prebuild plans are unchanged
  (same 72 / 64 artifacts, same keys). `judge owed` after the merge: H100, A100, RTX 5090 accuracy and speed owed 0, pending 0.

## Checks

* Merge: no conflicts (no file changed on both sides). Byte comparison on an H100 of module outputs, 2.5.10 tree vs this tree (TriMul D64 /
  D128 inference, D128 / D512 single-direction training incl. all gradients, TriAttn 128:4:32 / 64:4:16 / AF3 64 inference, TriAttn 128
  training): 18 / 18 cases bitwise identical; OPM, `opm-win` tree vs this tree (AF3 / Boltz / RF3-biased forms, S 2048 / 8192 / 16384): 9 / 9
  bitwise identical.
* GPU: `tests/gpu/test_opm_deep_product_route.py`, `test_opm_count_offset.py`, `test_i64_opm_pwa.py`, `test_accuracy_smoke.py -k "OuterProduct
  or MSA or PairWeighted"`, and the 2.5.10 TriMul / TriAttn tests (`test_triattn_h16_core_dh16.py`, `test_trimul_single_train_b1.py`,
  `test_trimul_uni_wide_ln_prefetch.py`) as one 2-GPU work-queue job: 7 / 7 files pass (25 + 12 + 3 + 4 + 2 + 24 + 8 tests).
* Installed wheels on an H100 (the five local artifacts unpacked, no repo / CUTLASS / nvcc on the path, empty JIT cache): 513 / 513 sm90
  artifacts resolved, 0 compiled (also with the system libstdc++ preloaded); accuracy OuterProductMean (cuda 32 GREEN), TriangleMultiplication
  (cuda 22 GREEN / 2 YELLOW) and TriangleAttention (cuda 22 GREEN / 6 YELLOW): RED 0, 0 builds, the same counts as the 2.5.10 release check;
  `test_opm_deep_product_route.py -k "8192 or 16384 or 2048"` against the installed wheels: 20 / 20 pass, 0 builds; restored Triton backends
  63 / 63.
* CPU: `test_requirements` (1164), `test_prebuild_plan` (4), `test_judge` (20), `test_modules_cpu` (497), `test_imports` (920) pass. `ruff
  check`, `ruff format --check` and `pyright` clean on the touched Python files. `_prebuild --check`: 0 plan problems on sm90a / sm80 / sm120
  (the other findings -- call sites not reached on a given arch, families compiled by other recipes -- are the same as 2.5.10, plus
  `opm_train._lt_ext` not reached on sm80 / sm120, which is sm90 only).

## Packaging

* The engine pins all three kernel wheels `==2.5.11` (`miniworld-engine-kernels-sm90-cu129`, `-sm80-cu129`, `-sm120-cu129`).
* Prebuild from the 2.5.10 release cache (content-keyed): sm90 513 artifacts (2.5.10: 512), 8 built: the seven re-keyed
  `miniworld_opm_epilogue_native*` extensions (`csrc/opm_epilogue.cu` changed) and the new `miniworld_opm_lt_mm`; sm80 72 and sm120 64,
  0 built. The sm80 / sm120 wheels carry the 2.5.10 payload byte-for-byte, under version 2.5.11.
* Manifest vs the 2.5.10 upload: no name missing in any wheel. sm90 adds the name `miniworld_opm_lt_mm`; the seven
  `miniworld_opm_epilogue_native*` names keep their names under new keys (their 2.5.10 paths are not carried: the 2.5.10 engine pins its
  kernels wheel `==2.5.10`, so no released engine asks for them from this wheel). `triattn_h100_core_h16_v2` stays carried byte-for-byte from
  2.5.5. sm80 / sm120 paths identical.
