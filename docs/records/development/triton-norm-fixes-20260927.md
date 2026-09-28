# Triton normalization fixes — 2026-09-27

## Scope and installation

Small, measured fixes to the existing Triton paths. The validated candidate was installed into `.engine-release-2.0.0`; the native `norm_cuda` experiment remains an explicit API and is not selected globally. No commit or push was performed.

- **RMSNorm:** widened tuning grids to permit BLOCK_K=512/1024 (previous maximum 256) and backward warps=8. Added measured H100 cache selections. Kernel arithmetic is unchanged.
- **LayerNorm:** automatic backward routing now consults a cache scoped by activation and affine dtype before applying the H100 heuristic. Measured BF16/FP32-affine D384 and D1024 cells select the existing atomic implementation over persistent. Explicit overrides and dispatch-off behavior are preserved.
- **LayerNormLinear portable training:** save mean/rstd from the existing fused forward, eliminating the separate full FP32 input copy and PyTorch statistics chain. Fix 2D/3D/4D shape handling and flatten/restore around backward. Keep x_normed recomputation in backward. FP32 uses Triton LN plus cuBLAS; CPU, FP64, empty inputs and K>1024 use the composed fallback. The H100 CuTe autograd class is unchanged.
- The cache build driver now exercises both inference and stats-saving Triton forward. SAVE_STATS is part of the autotune key. Projection dot products explicitly request IEEE precision for FP32.

## Paired complete forward + backward timing

H100 80GB on node02, BF16 activation, FP32 normalization affine, epsilon 1e-5. Times are **ms**, including all input/affine/projection gradients. CUDA graph timing uses ten calls per replay and nine timing samples (median). Frozen baseline and candidate ran sequentially on the same allocated GPU in job 19372. M is the flattened row count, D the input width, N the output width. These are module microbenchmarks, not whole MiniWorld training results.

| Path | M | D | N | Before (ms) | After (ms) | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| RMSNorm | 147456 | 128 | 128 | 0.075456 | 0.072406 | 1.04x |
| RMSNorm | 589824 | 128 | 128 | 0.280643 | 0.258678 | 1.08x |
| RMSNorm | 147456 | 384 | 384 | 0.254067 | 0.220176 | 1.15x |
| RMSNorm | 8192 | 1024 | 1024 | 0.052413 | 0.036723 | 1.43x |
| LayerNorm auto | 147456 | 384 | 384 | 0.275853 | 0.246253 | 1.12x |
| LayerNorm auto | 589824 | 384 | 384 | 1.034605 | 0.949216 | 1.09x |
| LayerNorm auto | 8192 | 1024 | 1024 | 0.072486 | 0.044262 | 1.64x |
| LayerNormLinear portable | 147456 | 128 | 16 | 0.350058 | 0.168979 | 2.07x |
| LayerNormLinear portable | 8192 | 384 | 512 | 0.119789 | 0.094858 | 1.26x |
| LayerNormLinear portable | 8192 | 64 | 64 | 0.050499 | 0.035891 | 1.41x |

**LayerNormLinear baseline qualification:** the old public portable wrapper had incompatible flattening/shape plumbing. Its benchmark uses a local PortableShapeAdapter preserving the old kernels, statistics recomputation and backward while repairing only shape plumbing. Thus the 2.07x gain is over that old wrapper algorithm, not an unmodified working public API or the fastest previously available engine path.

The existing composed Triton LN + cuBLAS remains faster than the revised portable wrapper for 128→16 (0.160694 vs 0.168979 ms) and 384→512 (0.087386 vs 0.094858 ms). For 64→64 it measures 0.036928 vs 0.035891 ms. No global replacement of the composed or CuTe paths is justified by these data.

Small-shape regression check: weighted RMSNorm M8192/D64 measured 0.006566 ms before and 0.006525 ms after preserving its earlier fast tile. Unweighted D32/D64 were also checked; differences are sub-microsecond and no speedup claim is made for them. The candidate small-shape results were refreshed in job 19420, so they are less tightly paired than the main table.

## Validation

- Job 19398: **60 tests passed**. Output and all gradients across FP16/BF16/FP32/FP64; K=64,128,137,384,1024,1025; 2D/3D/4D and strided inputs; projection bias present/absent; FP64 statistics oracle with large offsets; empty inputs; fullgraph compile; changed-input/weight/upstream-gradient CUDA graph replay; cache dtype scoping and override/off controls; existing mixed-affine fallback checks.
- All ten large benchmark cells passed output/gradient relative-L2 checks below 0.005.
- Job 19388_0: memcheck, seven selected graph/edge tests, **zero errors**.
- Job 19388_2: synccheck, same selection, **zero errors**.
- Job 19419: direct racecheck of modified LayerNormLinear stats stores (covering and K-tiled paths), plus selected RMS forward/backward schedules at D128/384/1024: **zero hazards/errors/warnings**. This is direct kernel coverage, not a claim that the entire compiled cuBLAS backward was racechecked.
- Earlier sanitizer graph capture failures were resolved by placing first-use warmup in the capture stream; no API-error suppression was used.
- Job 19434: actual installed package path and all 12 file hashes verified; typed LN route resolved to atomic; six installed-path graph/edge/dispatch tests passed. Installed F+B smoke timings: RMS D1024 0.034720 ms, LN D1024 0.043933 ms, portable LNLinear 128→16 0.165766 ms. These separate smoke timings confirm installation and are not substituted into the paired table. Existing missing-cache warnings for some standalone LN/recompute cells remain; they use bounded autotuning.
- Source/grid/cache file hashes were checked against the frozen baseline before installation. Installed hashes are recorded in `installed-manifest.json`. Formatting/static checks and `git diff --check` passed.

## Cache coverage and limits

The H100 RMS cache contains measured BF16 weighted cells at M/D=147456/128,589824/128,147456/384,8192/1024 and8192/64. LN typed routing covers three measured BF16/FP32-affine cells. LNLinear cache covers M/D/N=147456/128/16,8192/384/512,8192/64/64, with bias, both SAVE_STATS values; each used 24 bounded autotune candidates, with the top three retained. This is **partial measured coverage**, not exhaustive tuning of every shape/dtype or every grid configuration. Other shapes retain the normal autotune/fallback behavior.

Standalone normalization forward was already efficient in the preceding profiling audit; no broad rewrite was done. No new SOL90 claim and no end-to-end MiniWorld training speedup claim is made.

## Artifacts

All raw timings, frozen packages, test scripts, sanitizer logs and installation manifest are under `/home/psk6950/MiniWorld/runs/norm_cuda_20260926/triton_fix/`. Prior profiling rationale is in `triton-norm-audit-20260926.md` beside this report.
