# miniworld-engine 2.5.13

Base: 2.5.12 (7c544cf2f), release candidate `rc/2.5.13-int` (= `rc/2.5.13` + `cand/transition-d768` + `cand/ta-d64` +
`cand/trimul-uni-tables` + the bidirectional TriMul exact-key fix). H100 only:

1. TriangleAttention fp16 core (32 / 16-channel heads): `TA_RC` loop, **bit-identical**.
2. TriangleAttention 64-channel heads (AF3 d_pair 512 x 8, RF3 64 x 4 x 64, Chai-1 / confidence 256 x 4): new fp16 wgmma inference core,
   **outputs change, inside the engine accuracy standard** (error vs fp64 lower).
3. Transition D 768 (n 2) inference above 768 rows on the generic kernels, **outputs change, same error vs fp64**.
4. TriMul single direction D64 / D128: measured cuBLASLt contraction table and the no-mask launch-table cells, **bit-identical**; the
   bidirectional no-mask cells now read at their exact key too, **bit-identical**.

No module `version` bump. `backend_versions["cuda:h100_sm90"]`: TriangleAttentionAF3 10 -> 11, `_TriangleAttentionFp32Softmax` (RF3)
11 -> 12, TriangleAttentionChai1 6 -> 7, TriangleAttentionChai1Confidence 6 -> 7, Transition 5 -> 6. TriMul launch table
`kernels/trimul_inproj/configs/h100_sm90.json` version 6 -> 7 (the judge re-measures every class launching a `trimul_inproj` kernel on
the H100). A100 / RTX 5090: no source they compile changed, no version or token change.

## Changes

| # | Change | Outputs | `backend_versions["cuda:h100_sm90"]` | Measured (module, CUDA-graph replay, same process; new / old time) |
|---|---|---|---|---|
| 1 | `triattn_h16_sm90.cu` `TA_RC=1` (new default): the last key tile peeled out of the consumer loop, the gate prefetch loaded right before it (registers freed: ptxas hoists the next bias ldmatrix). Same operations, same order; every shipped instantiation (32 / 16-channel heads). `TA_RC=0` builds the 2.5.12 loop (in-process baseline only) | bit-identical | - | 16-channel heads cool 0.92-0.97x, 32-channel heads 0.97-1.01x |
| 2 | `triattn_d64_sm90.cu` rewritten on the 32-channel fp16 core's design: fp16 q / k / v / g and bias / scale from the glue's fp16 LayerNorm / bias kernels and cuBLAS fp16 GEMMs (`h100_dh._front_fp16`), the bias fragment as the fp16 score accumulator's initial value, f16x2 exponentials (half on the FMA polynomial), lazy offset with a warp vote, `[O \| l] += P [V \| 1]`, Q tiles through the K \| V ring, 128-key tiles. L % 128 == 0, inference only (training keeps the dh training core) | changed, within the standard (Accuracy) | AF3 11, RF3 (Fp32Softmax) 12, Chai1 7, Chai1Confidence 7 | core 1.28-1.50x faster; module 1.09-1.29x (sustained) |
| 3 | Transition D 768 n 2 inference: `_PREFER_GENERIC_ROWS[768]` inference limit 768 -> 16384 rows (the wide D768 build is the training structure: LayerNorm kernel, expand that also writes a / b, per-call `[Wa; Wb]` pack, squeeze). Training unchanged (rows <= 768) | changed above 768 rows (~2.3 % of the elements), same error vs fp64 | Transition 6 | 1024-2048 rows 1.72-1.90x faster, 4096 1.43x, 8192 1.36x; 768 rows 1.00 |
| 4 | TriMul single direction D64 / D128 (hidden = D): the contraction through the measured cuBLASLt table `_h100_bmm_lt_uni.LT_UNI` (284 entries, algorithm 66, each bitwise equal to the default pick), on the packaged route and the port's row-padded form | bit-identical | - (table 6 -> 7) | with row 5: mean 0.991 cool / 0.993 sustained (61 cases), D128 L128 0.90-0.92 |
| 5 | TriMul launch table: 116 single-direction `trimul_fwd_sm90_cuda` cells without a mask (D64 / D128, hidden D / 2 D, DIR 1 / 2, B 1 / 2), every candidate bitwise equal to the kernel default, re-ranked by an in-graph A/B (28 cells change tiles) | bit-identical | - (table 6 -> 7) | see row 4 |
| 6 | `h100_inference._select_config`: a call without a mask reads the launch table at its exact (B, L) only (`near=()`), single direction (row 5) and now also the bidirectional DIR 0 MASK=0 cells of 2.5.12; an off-grid length takes the kernel default instead of the nearest cell's tiles | bit-identical | - | bidirectional off-grid: was 1.08 / 1.09x slower (D64 L912 / 1104), 1.06 / 1.07x (D128 L144 / 176); now 0.99-1.01x of the default at all 22 off-grid shapes |
| 7 | `requirements_kernel.csv`: `triangle_attention_d64_fwd_sm90` also lists `TriangleAttentionAF3` (d_pair 512 x 8 heads of 64 channels launches it; a registry fact, owes nothing by the judge's rule) | - | - | - |

## Bit-identity (H100, the RC tree vs the 2.5.12 tree 7c544cf2f, separate processes, sha1 per tensor)

874 registry-built cases, cuda backend, dropout off (the accuracy measurer's `_run`; training: output, input and every parameter
gradient), registry key mask or none, B1 / B2:

| family | rows | cases | identical |
|---|---|---|---|
| TriangleAttention (32 / 16-channel heads, bias-only), AF3 64 / 128 / 256, RF3 128, Chai1 64; BidirectionalTriangleAttention | 71-79, 82, 83, 188, 142; train 72, 79, 80, 188, 241-252, 142 | 159 | 159 |
| TriangleAttention 64-channel heads: AF3 512, RF3 64 (d_hidden 256), Chai1 256, Chai1Confidence 256, eval | 80, 81, 84, 85 | 40 | 0 (intentional, change 2) |
| TriangleMultiplication D64-D512, RF3, Bidirectional D64-D512 (incl. port K3 L >= 160, off-grid L144 / 176 / 300 / 912 / 1104, B2) | 86-98; train 91, 94-98, 253-259 | 192 | 192 |
| OuterProductMean | 55-70; train 62, 66-70, 234-240, 277-279 | 64 | 64 |
| MSAPairWeightedAveraging | 48-54; train 227-233 | 28 | 26 (see below) |
| Transition (every width, bf16 / master) | 25-40; train 33, 34, 40, 209-221 | 112 | 109 (3 intentional, change 3) |
| Token DiT (Protenix, MiniWorld, bias-only, Boltz-2, RF3, Chai-1; shared cond, folds) | 99-113, 143-145, 163, 179-187; train 101, 103-107, 143-145, 260-266, 286 | 138 | 134 (see below) |
| Atom DiT (Protenix, OpenFold3, ESMFold2 (+pair), Boltz-2, RF3, Chai-1; shared cond, folds) | 114-129, 164-178; train 117, 118, 267-276, 284, 285 | 136 | 136 |
| PairformerBlock | 140, 141 | 5 | 5 |

* The intentional changes equal their candidate branches' outputs (sha1): all 40 TriAttn 64-channel cases equal `cand/ta-d64`, the 3
  Transition D768 cases (L768 B2, L1024, L2048: 1536 / 1024 / 2048 rows) equal `cand/transition-d768`. D768 at 128 / 768 rows is identical
  to 2.5.12.
* 6 training cases differ in LayerNorm weight / bias gradients only (`MSAPairWeightedAveraging` rows 228 / 229 `ln_msa`, `TokenDiTBoltz2`
  263 and `TokenDiTRoseTTAFold3` 264 `ada_ln_in.ln_cond.weight`, float32 and master): these differ between two runs of the 2.5.12 tree
  itself (and of the RC tree): atomic dgamma reduction, not a change. No source they use changed.
* float32 Transition cases are refused by the cuda backend under the measurer's precision (no TF32) in both trees, identically.

## Accuracy (engine standard, `measure/accuracy.py`: per tensor max|x - gold| / max|gold| vs fp64 PyTorch, tol = max(err_pytorch, 8e-3))

* Change 2 (`cand/ta-d64`): rows 80 / 81 / 84 / 85, eval, L128-768 (L >= 1024 the fp64 gold does not fit one H100 for either core): 48 / 48
  GREEN with both cores; fp16 core / 2.5.12 core max-error ratio 0.84-1.32 (mean 0.98: the module output's bf16 rounding dominates that
  metric), worst err / tol 0.63 (Chai-1 confidence L640 starting, 5.04e-3 against 8e-3). Core alone vs fp64 (random q / k / v, bias std 2):
  rms 6.3e-4 / 7.2e-4 against 1.7e-3 / 2.0e-3 of the 2.5.12 core, max 1.1-1.9e-2 against 3.0-3.8e-2 (about 0.37x).
* Change 3 (`cand/transition-d768`): row 37 master eval L1024 / 1536 / 2048: cuda err 3.504e-3 vs pytorch 4.321e-3 on both routes, GREEN.
  Against fp64 at 768-4096 rows, new / old: max 4.53e-3-5.08e-3 / identical, rms 4.36e-3-4.38e-3 / identical (PyTorch bf16: rms 6.15e-3).
* Judge re-measure of every owed cell: see Judge.

## Speed (CUDA-graph replay, one process, interleaved)

H100 80GB HBM3, **power-capped at 700 W**: under sustained load the SM clock drops 15-18 %, so cool and sustained are reported separately and
are not comparable to the benchmark pages. Each candidate measured old and new in one process with a same-route control (a second
capture / cubin of the unchanged route); the merge did not change the code of any measured module (the candidates' files do not overlap;
outputs on the merged tree equal the candidates', above), so their numbers are carried. ms unless noted.

TriangleAttention 64-channel heads, whole module (`h100_dh.forward`, mask with the last L / 8 keys padded; us sustained, ratio old / new
sustained / cool):

| class (d_pair, heads x 64) | L 512 | 768 | 1024 | 1536 | 2048 |
|---|---|---|---|---|---|
| AF3 512, 8, starting | 2983 -> 2683 (1.11 / 1.10) | 7343 -> 6533 (1.12 / 1.12) | 14491 -> 12337 (1.17 / 1.16) | 38127 -> 31380 (1.22 / 1.19) | 75141 -> 64088 (1.17 / 1.27) |
| AF3 512, 8, ending | 3311 -> 3010 (1.10 / 1.10) | 8160 -> 7183 (1.14 / 1.13) | 15444 -> 13268 (1.16 / 1.17) | 41048 -> 35214 (1.17 / 1.21) | 81248 -> 69240 (1.17 / 1.26) |
| RF3 64 (d_hidden 256), 4, starting | 1047 -> 948 (1.11 / 1.11) | 2674 -> 2270 (1.18 / 1.17) | 5246 -> 4185 (1.25 / 1.24) | 14481 -> 12093 (1.20 / 1.17) | 31276 -> 24251 (1.29 / 1.28) |
| RF3 64 (d_hidden 256), 4, ending | 1100 -> 991 (1.11 / 1.12) | 2734 -> 2364 (1.16 / 1.16) | 5413 -> 4373 (1.24 / 1.23) | 15042 -> 12359 (1.22 / 1.16) | 30814 -> 24447 (1.26 / 1.28) |
| Chai-1 256, 4, starting | 1319 -> 1164 (1.13 / 1.13) | 3276 -> 2750 (1.19 / 1.18) | 6423 -> 5046 (1.27 / 1.23) | 17425 -> 14230 (1.22 / 1.22) | 35391 -> 28223 (1.25 / 1.32) |
| Chai-1 256, 4, ending | 1440 -> 1257 (1.15 / 1.14) | 3567 -> 2986 (1.19 / 1.18) | 6821 -> 5426 (1.26 / 1.23) | 18247 -> 14891 (1.23 / 1.21) | 36741 -> 29128 (1.26 / 1.27) |
| Chai-1 confidence 256, 4, starting | 1621 -> 1493 (1.09 / 1.10) | 3963 -> 3452 (1.15 / 1.14) | 7550 -> 6293 (1.20 / 1.19) | 19697 -> 16130 (1.22 / 1.18) | 39934 -> 32886 (1.21 / 1.25) |
| Chai-1 confidence 256, 4, ending | 1788 -> 1616 (1.11 / 1.11) | 4340 -> 3812 (1.14 / 1.14) | 8273 -> 6892 (1.20 / 1.18) | 21332 -> 18027 (1.18 / 1.18) | 42356 -> 35801 (1.18 / 1.23) |

Core alone (512:8:64 / 256:4:64, L512-2048): 1.28-1.50x sustained, 1.25-1.47x cool (`docs/h100/triangle_attention.md`).

Transition D 768 n 2 inference (us; new / old / same-route control; speed-up old / new):

| rows | cool new / old (ctl) | sustained new / old (ctl) | speed-up cool / sustained |
|---|---|---|---|
| 768 | 25.34 / 25.39 (25.42) | 25.38 / 25.38 (25.31) | 1.00 / 1.00 |
| 1024 | 27.50 / 52.19 (26.26) | 27.39 / 51.88 (26.06) | 1.90 / 1.89 |
| 1536 | 35.92 / 62.11 (34.46) | 36.02 / 62.07 (34.44) | 1.73 / 1.72 |
| 2048 | 37.07 / 64.56 (35.78) | 36.82 / 63.25 (35.45) | 1.74 / 1.72 |
| 4096 | 55.92 / 81.86 (55.95) | 56.63 / 81.23 (56.47) | 1.46 / 1.43 |
| 8192 | 107.26 / 148.67 (106.85) | 114.57 / 155.78 (114.34) | 1.39 / 1.36 |

TriangleAttention fp16 core `TA_RC` (change 1; 2.5.12 loop = the RC core built `-DTA_RC=0`, equal to the 2.5.12 tree's bytes; ctl = the
RC core built twice; ctl / RC cool 0.993-1.019, sustained 0.977-1.025), new / old, B1 no mask eval:

| module (row) | L256 | L512 | L1024 | L2048 |
|---|---|---|---|---|
| TriAttn 64:4:16 (72), cool / sus | 0.966 / 0.972 | 0.923 / 0.935 | 0.928 / 0.947 | 0.948 / 0.988 |
| TriAttnAF3 64 (77) | 0.967 / 0.967 | 0.927 / 0.941 | 0.930 / 0.958 | 0.948 / 0.965 |
| TriAttn 64:2:32 (71) | 0.989 / 0.994 | 0.972 / 0.981 | 0.989 / 1.000 | 0.985 / 0.984 |
| TriAttn 128:4:32 (74) | 0.973 / 0.963 | 0.988 / 0.992 | 0.975 / 0.988 | 0.987 / 1.000 |
| TriAttn 256:8:32 (75) | 0.991 / 0.993 | - | 0.983 / 0.987 | 0.982 / 0.977 |
| TriAttn 384:12:32 (76) | 1.001 / 0.994 | - | 0.988 / 0.991 | 0.993 / 0.987 |

(66 cells in all, `~/miniworld-engine-scratch/rc2513-work/results/speed_table.txt`: 16-channel heads cool 0.92-0.97x, sustained
0.93-0.99x; 32-channel heads cool 0.97-1.01x, sustained 0.96-1.00x.)

TriMul single direction (changes 4 + 5; 61 inference cases D64 / D128, hidden D / 2 D, both directions, L128-2048, B1 / 2, masks; the
rc/2.5.13 route by switching both tables back, same-route control captured separately): mean 0.991 cool / 0.993 sustained (control 1.001 /
1.000); D128 L128 0.90-0.92, D64 L768 / 1024 0.97-0.98; elsewhere 0.98-1.01 (`docs/h100/trimul.md`). Bidirectional off-grid (change 6;
22 shapes D64 / D128, L144-1808, B1, module graph A/B on the merged tree, 9 interleaved rounds): the call now picks the kernel default at
every shape (bits equal), table / default 0.992-1.006 (2.5.12: 0.89-1.09, worst D64 L1104 1.094, L912 1.079, D128 L176 1.068, L144 1.057).

## Judge (version-based tokens)

* `judge owed` on the merged tree before the re-measure: H100 1329 accuracy + 1329 speed cases (`adapter:cuda 5 -> 6` Transition 307,
  `10 -> 11` AF3 108, `11 -> 12` RF3 60, `6 -> 7` Chai1 / Chai1Confidence 90, `table:trimul_inproj 6 -> 7` 890: every class launching a
  `trimul_inproj` kernel, cuda and triton backends; a case owed for two reasons counts once); A100 and RTX 5090 0. Adding
  `TriangleAttentionAF3` to the d64 kernel row owes nothing (a kernel entering the list is a correction).
* Re-measured once, on the final merged tree, with `tools/slurm/cycle.sh` (one cycle, 6 groups on 1-GPU 10-minute jobs, plan
  `~/miniworld-engine-scratch/rc2513int-work/judge_plan1`): 4788 tokens issued.
* Accuracy lights old -> new, every re-measured cell: GREEN -> GREEN 2051, YELLOW -> YELLOW 191, skipped -> skipped 152; **no light
  changed, RED 0**. Errors changed on the 64-channel TriAttn eval cuda cells (AF3 512 x 8: new / old 0.88-1.13; RF3 64 x 4 x 64:
  0.85-1.01; Chai1 256: 0.84-1.32; Chai1Confidence 256: 0.84-1.13; all still GREEN / YELLOW as before) and by <= 0.2 % on Triton training
  cells (run-to-run atomics). Transition D768 cells: errors unchanged.
* Speed (speed-up vs PyTorch, new / old, per class: median, range): Transition 1.000, 0.89-1.88 (the D768 rows above 768 rows);
  TriangleAttentionAF3 1.000, 0.97-1.11; TriangleAttentionRF3 1.000, 0.98-1.19; TriangleAttentionChai1 1.000, 0.99-1.19;
  TriangleAttentionChai1Confidence 1.000, 0.96-1.15; TriangleMultiplication 1.000, 0.93-1.07; TriangleMultiplicationRF3 1.000, 0.95-1.04;
  BidirectionalTriangleMultiplication 1.000, 0.98-1.04; PairformerBlock 1.000, 0.97-1.02 (medians over every re-issued backend cell,
  most of which run unchanged code; the judge's TriAttn cases are L <= 768).
* `judge owed` after the cycle: H100, A100, RTX 5090 accuracy and speed owed 0, pending 0 (17719 carried each).


## Tests and checks

* GPU, one 4-GPU work-queue job on the merged tree (one file per free GPU), 19 items, all pass: `test_accuracy_smoke.py -k "Triangle or
  Transition"` (36), `test_determinism.py -k "Triangle or Transition"` (10), `test_triattn_d64_fp16_core.py` (11),
  `test_triattn_h100_variants_infer.py` (17), `test_triattn_dh_query_tile.py` (4), `test_triattn_h100_tail_gate.py` (15),
  `test_triattn_h16_core_dh16.py` (2), `test_triattn_h100_key_tiles.py` (15), `test_triattn_h100_front_y16.py` (14),
  `test_triattn_h100_front_lnb.py` (6), `test_bidir_triattn_h100.py` (254 passed, 24 skipped), `test_refiner_sizes.py` (18 passed, 8
  skipped), `test_cuda_refuses_not_raises.py` (8), `test_anthropic_kit_rows_h100.py` (2), `test_weight_cast_cache_h100.py` (12),
  `test_trimul_inference_workspace.py` (23), `test_trimul_uni_tables.py` (113, incl. the 20 new bidirectional off-grid cases: the
  default is picked, and the output equals the nearest-cell route's), `test_trimul_bidir_contraction_port_k3.py` (23),
  `test_trimul_h100_k3_pipe.py` (12).
* CPU: `tools/check.sh cpu` on the merged tree: ruff check clean, pyright 0 errors, 3329 CPU tests passed (2 skipped); includes
  `test_transition_d768_route.py`, `test_requirements`, `test_prebuild_plan`, `test_judge`.
* Kernels from a copy of the 2.5.13 RC cache (content-keyed; the changed artifacts built once), not the kernels wheel of the shared
  environment.

## Packaging: prebuilt artifacts whose keys change

`_prebuild --list` / `--check` against the 2.5.12 tree: same artifact counts (sm90 513, sm80 72, sm120 64), the same names (none dropped,
none added), `--check` findings identical on every arch, 0 not built in the RC cache.

* **sm90**: `triattn_h100_core_h16_v3` re-keyed (a980ab15 -> dc615aed; change 1) and `triattn_h100_d64` re-keyed (caa5e7aa -> a67fd776;
  change 2), same names. Nothing else changed (the TriMul tables are Python / JSON, not compiled).
* **sm80 / sm120**: no change.

## Release build (2.5.13)

* Incremental prebuild from the 2.5.12 release cache (GCC 12.4 host compiler): sm90 2 built (`triattn_h100_core_h16_v3` [dc615aed],
  `triattn_h100_d64` [a67fd776]), sm80 0, sm120 0; 0 failed, 0 absent. Artifacts: sm90 513, sm80 72, sm120 64 (as 2.5.12).
* Manifest names vs the 2.5.12 upload: identical on sm90 / sm80 / sm120 (no name dropped or added); wheel file lists differ only by the
  two re-keyed sm90 directories. Carried byte-for-byte as before: `triattn_h100_core_h16_v2` (sm90, from 2.5.5).
* Installed-wheel check (the four artifacts unpacked, no repo / CUTLASS / nvcc, empty JIT cache): 513 / 513 sm90 artifacts resolved, 0
  compiled (also with the system libstdc++ preloaded); capture API and init presets OK; `test_triattn_d64_fp16_core.py` 11 passed,
  `test_triattn_h16_core_dh16.py` 2 passed, `test_trimul_uni_tables.py` 113 passed, `test_restored_triton_backends.py` 63 passed, 0
  builds; accuracy smoke (TriangleAttention, TriangleAttentionAF3, TriangleAttentionRF3, TriangleMultiplication, Transition) RED 0, 0
  builds.
* PyPI sha256 (= `dist/SHA256SUMS`): engine wheel `1a51a9e9...`, sdist `4a955474...`, sm90 `b2ff12ec...`, sm80 `e1d83309...`, sm120
  `b4c91628...`.
