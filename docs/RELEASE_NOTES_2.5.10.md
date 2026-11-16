# miniworld-engine 2.5.10

Base: 2.5.9 (ec11768fc). H100 only: (1) TriangleAttention inference on heads of 16 channels (d_pair 64 x 4 heads; TriangleAttention 64:4:16 and
AF3 d_pair 64): half of the fp16 core's exponentials on the FMA polynomial -- **not bit-identical**, inside the engine accuracy standard; (2)
TriangleMultiplication single-direction training (D128, D512): **bit-identical** kernel changes. Module `version` is not bumped anywhere; only
`backend_versions["cuda:h100_sm90"]` of TriangleAttention / TriangleAttentionAF3 (9 -> 10), so anthropic / cueq / triton / pytorch tokens and the
A100 / RTX 5090 tokens stay valid. No FP8, no bidirectional-only work, no OPM.

## Changes

| # | Change | Check | `backend_versions["cuda:h100_sm90"]` | Measured gain |
|---|---|---|---|---|
| 1 | TriAttn fp16 core, 16-channel heads: `TA_POLY16` = 8 (every other exponential pair as `hex2_poly` on the FMA pipes; 32-channel heads keep `TA_POLY`) | 32-channel heads bit-identical (66 / 66 cases vs 2.5.9: TriangleAttention 128:4:32 and AF3 128, both nodes, L128-2048, B1 / 2, masks); 16-channel heads: judge accuracy lights unchanged on every re-measured cell, the 8 cells whose error moved stay GREEN (out err 2.93e-3..3.22e-3 vs tol 8e-3); `tests/gpu/test_triattn_h16_core_dh16.py` (deterministic, err vs fp64 < 8e-3) | TriangleAttention 10, TriangleAttentionAF3 10 | module 0.97x (L512), 0.95x (L768), 0.93x (L1024), 0.91x (L1536), 0.89x (L2048) |
| 2 | TriMul D128 single-direction training B1 (`single_output/kernel.cu`): next tile's xn / tri / dy TMA-prefetched into a second smem set (181248 -> 230400 B), bf16x2 epilogue / dn stores (`B1_VEC`), two interleaved LN rows per warp pass (`B1_LNI` = 2), packed bf16x2 gate rounding and dy x dropout scale (`B1_PK`) | bit-identical y, dx, all parameter grads vs 2.5.9 (D128, outgoing / incoming, L256-768, B1 / 2, masks none / tail / random, fp32 master and bf16 params: 96 / 96); `tests/gpu/test_trimul_single_train_b1.py` | none (outputs unchanged) | module fwd+bwd L512 0.98x, L768 0.91x; L256 / 384 1.00x (host-bound in eager) |
| 3 | TriMul D128 single-direction training B7 (`single_b7/joint.cu`): `B7_EARLY_FREE` (plane staging buffers freed once the bulk stores have read them), dx tile staged in the dg buffer; host: `_build` once per process | bit-identical (same 96 / 96 cases) | none | included in row 2's module numbers (kernel -1.5 %) |
| 4 | TriMul D512 single-direction training: output-LN backward with the next row tile prefetched (`wide_train/output_ln_uni_pf.cu`, `_LN_PF`), same grid and per-CTA order | bit-identical (D512, 96 / 96 cases); `tests/gpu/test_trimul_uni_wide_ln_prefetch.py` | none | module fwd+bwd L384 0.99 / 1.00x, L768 0.98x |
| 5 | Prebuild plan: the D128 single B7 is planned through `h100_single_b7._build` (the cached build otherwise needed a GPU at plan time) | `_prebuild --check`: 512 artifacts planned, 0 absent, no plan problems | - | - |

TriMul inference is unchanged: bytes of the default route compared with 2.5.9 at D64 / 128 / 256 / 384 / 512, outgoing / incoming / bidirectional,
L17 / 255 / 256 / 513 / 1024 / 1536 / 2048, B1 / 2, masks none / ones / tail / random (no B2 at L2048 for D384 / D512): 756 served cases identical
(60 more are outside the H100 kernels on both versions: bidirectional D256-D512 and single D512 at L17 / 255 / 513). TriMul D64 / D256 training: 96 / 96 cases each identical.

## Speed (main = 2.5.9 kernels, measured in the same process as the branch)

H100 80GB HBM3 (700 W cap). One process per group, the 2.5.9 code path and the 2.5.10 one alternating in short bursts (~20 ms, 60 ms pause, 25 rounds,
order alternating; median ms): sustained load throttles the SM clock 15-18 % after ~50-80 ms, so these are not comparable to the benchmark pages'
absolute numbers. TriAttn: CUDA-graph replay of the module; the 2.5.9 path is the same core built with `TA_POLY16=0` (every exponential on the MUFU,
as 2.5.9). TriMul training: eager forward + backward (CUDA-graph capture of the training route faulted in this harness), master weights, dropout on;
the 2.5.9 path is `single_output/kernel.cu` / `single_b7/joint.cu` compiled from the 2.5.9 sources (D128) or `_LN_PF = False` (D512).

| Module | L | 2.5.9 ms | 2.5.10 ms | ratio |
|---|---|---|---|---|
| TriangleAttention 64:4:16 starting | 512 / 768 / 1024 / 1536 / 2048 | 0.417 / 1.131 / 2.385 / 7.137 / 15.872 | 0.404 / 1.072 / 2.217 / 6.472 / 14.058 | 0.968 / 0.948 / 0.930 / 0.907 / 0.886 |
| TriangleAttention 64:4:16 ending | 512 / 768 / 1024 / 1536 / 2048 | 0.420 / 1.148 / 2.412 / 7.210 / 16.012 | 0.407 / 1.088 / 2.244 / 6.546 / 14.197 | 0.968 / 0.948 / 0.930 / 0.908 / 0.887 |
| TriangleAttentionAF3 64 starting | 512 / 768 / 1024 / 1536 / 2048 | 0.418 / 1.131 / 2.386 / 7.145 / 15.879 | 0.404 / 1.072 / 2.218 / 6.479 / 14.058 | 0.968 / 0.948 / 0.930 / 0.907 / 0.885 |
| TriangleAttentionAF3 64 ending | 512 / 768 / 1024 / 1536 / 2048 | 0.424 / 1.151 / 2.414 / 7.183 / 15.875 | 0.401 / 1.073 / 2.224 / 6.549 / 14.516 | 0.946 / 0.932 / 0.921 / 0.912 / 0.914 |
| TriMul D128 out, train fwd+bwd | 256 / 384 / 512 / 768 | 1.868 / 1.871 / 1.947 / 3.112 | 1.886 / 1.880 / 1.910 / 2.835 | 1.009 / 1.004 / 0.981 / 0.911 |
| TriMul D128 in, train fwd+bwd | 256 / 384 / 512 / 768 | 1.835 / 1.841 / 1.893 / 3.120 | 1.842 / 1.841 / 1.867 / 2.845 | 1.004 / 1.000 / 0.986 / 0.912 |
| TriMul D512 out, train fwd+bwd | 384 / 768 | 4.749 / 18.644 | 4.706 / 18.236 | 0.991 / 0.978 |
| TriMul D512 in, train fwd+bwd | 384 / 768 | 4.794 / 18.747 | 4.783 / 18.398 | 0.998 / 0.981 |

Judge speed tokens (re-measured cells vs the 2.5.9 tokens, median new / old): TriangleAttentionAF3 d_pair 64 x 4 eval 0.948 (18 cells), every other
TriangleAttention / AF3 configuration 0.99-1.01 (unchanged code: the ratio is the run-to-run spread).

## Judge (version-based tokens)

* H100 after the bump: 310 accuracy + 310 speed cases owed, all `adapter:cuda 9->10` of TriangleAttention / TriangleAttentionAF3 (every cuda cell of
  the two classes, not only the d_pair 64 ones: the version is per class and card). Re-measured with `tools/slurm/cycle.sh` (1 cycle, 1240 tokens
  issued). Accuracy lights old -> new: GREEN -> GREEN 233, YELLOW -> YELLOW 23 (training gradients, same errors as 2.5.9: the training path is
  unchanged), one cell measured for the first time (GREEN); errors changed on 8 inference cells, all d_pair 64 x 4 heads, all GREEN.
* `judge owed` after the merge: H100, A100, RTX 5090 accuracy and speed owed 0, pending 0.

## Tests

* GPU, one 4-GPU work-queue job (one file per free GPU), 21 files: the three new / changed test files, `test_determinism` and `test_accuracy_smoke`
  (`-k "Triangle or trimul or triattn"`), the TriMul inference / training files and the TriAttn H100 files, the H100 int64 / drop-scale / refiner /
  weight-cast files: 20 files pass; `test_trimul_big_plane.py::test_triton_training_past_2_31_elements[False-14]` (Triton backend, untouched) hit an
  out-of-memory in the queue and passes on its own on both 2.5.9 and this branch.
* CPU: `test_requirements` (1164), `test_prebuild_plan` (4), `test_judge` (20), `test_modules_cpu` (497) pass. `ruff check`, `ruff format --check`
  and `pyright` clean on the touched Python files.

## Not included

* TriMul inference K1 "instruction diet" (k1b: `TMN_K1_NO_VSEL`, `TMN_K1_ST2`, `TMN_K1_HALFA` = 2 as defaults; bit-identical, module 1-3 %): `ST2`
  changes `K1Cfg::SMEM` in `tmn_kernels.cuh`, a header shared by six host-side shared-memory formulas (packaged unit, direct K1, wide / uni-wide
  inference, b200style, uni-wide training); the k1b branch updated two of them (the direct K1 then faulted at D64 L > 1536 until it was switched off
  there), and the defaults would also change the instruction streams of the training families that patch the header. A clean port needs the defaults
  scoped to the packaged unit and a byte check of every family including the header; left for a later release. The stray `#endif` of the
  `TMN_K1_NO_VSEL` define block (only matters when a build passes `-DTMN_K1_NO_VSEL`) is left as it is with it (fixing it alone re-keys every TriMul
  artifact that includes the header for no output change).
* Default-off TriAttn experiments (`TA_PMASK*`, `TA_PD` 2, `TA_VIM`, `TA_DHONLY`, `TA_NOBT`, training-forward `TA_PMASK`): no gain, not shipped.
* Default-off TriMul training variants `B1_TS` (tanh sigmoid; not bit-identical, +0.3 % module), `B7_GATEFUSE`, `B7_TS`, `B7_SRC2`, the
  `contract_gp` 2-CTA cluster / persistent loader variants, the D64 b2 three-stage ring: measured no gain or slower.
* Bidirectional-only work (TriAttn bidirectional, port-K3 for bidirectional TriMul, the bidirectional cuBLASLt table), OPM, FP8 (3.0.0).

## Packaging (for the release build)

Artifacts whose keys change (sm90 wheel; rebuild and diff the manifest against 2.5.9):

* `triattn_h100_core_h16_v3` (torch extension, `triangle_attention/cuda/h100_sources/triattn_h16_sm90.cu`) -- key `a980ab15`
* `trimul_h100/single_output/kernel.cu` (cubin) -- key `6aa34e70`
* `trimul_h100/single_b7/joint.cu` (cubin) -- key `b1d66f0d`
* `trimul_h100/wide_train/output_ln_uni_pf.cu` (cubin, new artifact) -- key `ff1575b0`

No change to `tmn_kernels.cuh`, the packaged TriMul inference unit, or any sm80 / sm120 source: the sm80 and sm120 kernel wheels can stay at 2.5.9 content.
