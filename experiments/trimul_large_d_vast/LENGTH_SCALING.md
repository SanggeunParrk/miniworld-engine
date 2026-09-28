# Why the Triton speedup changes with length

Measured on the rented H100s on 2026-09-27. This investigation changes no
production dispatch and does not count baseline differences as native speedups.

The main observed cause is an output-LayerNorm backward dispatch cliff in the
Triton baseline at large widths. The earlier explanation based on short-shape
launch overhead was incomplete and is not supported as the main cause.

## Matched full forward + backward

Current CUDA Graph medians, 51 alternating samples. The native wide path is the
frozen D256 pool checkpoint / D512 checkpoint24 (without the split4 candidate).
D128 uses the installed specialized native B1/B7 path. GPU0 ran D128/D256;
GPU1 ran D512. Rows compare the same GPU/runtime and shape.

| D | L | Triton ms | Native ms | Triton / native |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 384 | 1.624 | 1.026 | 1.582 |
| 128 | 768 | 6.417 | 3.894 | 1.648 |
| 256 | 384 | 3.680 | 2.600 | 1.416 |
| 256 | 768 | 18.286 | 10.441 | 1.751 |
| 512 | 384 | 9.819 | 6.989 | 1.405 |
| 512 | 768 | 44.529 | 28.023 | 1.589 |

The native wide workloads scale almost exactly with L squared: 4.016x for D256,
4.010x for D512. Triton grows 4.968x and 4.535x respectively. This reproduces the
historical pattern, although absolute timings and ratios differ with runtime,
autotuning and the measurement. No new native optimization produced these ratios.

## The dispatch boundary and D128

`src/miniworld_engine/kernels/layernorm_linear/triton/mmajor_bwd.py` sets
`_PERSIST_MIN_M = 300_000`. For batch1 pair tensors, M=L*L: L384 has 147,456 rows
and takes the atomic kernel; L768 has 589,824 rows and takes the canonical
persistent kernel. The normalization width is **2D**, not D.

Three-replay backward CUDA activity, averaged per replay (separate from timing):

| D | LN width | L384 atomic ms | L768 persistent ms | Growth |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 256 | 0.0938 | 0.4485 | 4.78x |
| 256 | 512 | 0.2234 | 4.1757 | 18.69x |
| 512 | 1024 | 0.7924 | 6.2465 | 7.88x |

D128 also crosses the dispatch boundary, but its narrower LN does not suffer
the same severe penalty. Its full baseline therefore remains close to quadratic
scaling. In contrast, the long wide baseline pays milliseconds more in this one
step, increasing native's apparent advantage at long lengths. Other components,
including contraction GEMMs, also change with length; this is not an assertion
that output LN explains every microsecond.

The traces confirm `_ln_bwd_kernel` at L384 and `_ln_bwd_persistent` at L768.
Persistent launch grids for D128/D256/D512 are respectively (264,1), (264,1),
(264,4); thread blocks are 256, 64, 128. These launch properties are not hardware
counter measurements. The generic persistent source has different covering-tile
and split-column paths, including repeated full-row statistics for split columns.
Do not infer a specific register-spill or bandwidth limit without additional
evidence. D128's native B1/B7 is additionally a separate specialized implementation;
the same native fusion schedule is not used at all widths.

The intervention records the actual persistent selections:

| D | BLOCK_K | BLOCK_M1 | Warps | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 128 | 256 | 64 | 8 | Covers all 256 LN channels in one column tile |
| 256 | 512 | 8 | 2 | Covers 512 channels but uses much smaller row tiles |
| 512 | 256 | 64 | 4 | Four column tiles; each gathers whole-row statistics |

These are shape-specific tuned/fallback selections in this runtime, not universal
properties of Triton. The persistent kernel's `BLOCK_K < N` branch explicitly
repeats its full-row gather for each column tile. Avoid claiming that repeated
gathers are the D256 explanation: its selected tile covers the full width.

## Controlled intervention

`ablate_output_ln.py` captures default and forced-atomic Triton full workloads in
the same process. Only `settings.layernorm_out_bwd_path` changes. This diagnostic
tests whether the output-LN dispatch causes the timing gap.

75 alternating full-workload graph samples, same process and tensors:

| D | Default Triton F+B ms | Forced atomic F+B ms | Time saved ms |
| ---: | ---: | ---: | ---: |
| 128 | 6.357 | 6.258 | 0.099 |
| 256 | 18.227 | 14.922 | 3.305 |
| 512 | 43.390 | 40.155 | 3.235 |

The corresponding output-LN activity is 0.448/0.368 ms, 4.244/0.887 ms,
and 6.214/3.085 ms (default/atomic). Actual dispatch was checked from the
captured graph kernel names. This intervention confirms that the wide persistent
LN path accounts for a large part of the length-dependent relative speedup.
It does not establish a new native speedup or prove every wide shape is explained;
D384 was not repeated in this diagnostic.

The first run stopped at the strict equivalence check: these existing paths have
different arithmetic/reduction order. Wide dX relative errors were approximately
2.3-2.6e-4 and input affine errors 1.6-2.4e-4. The diagnostic-only second version
retains `strict_equivalent=false` and permits timing below a separately labeled
1e-3 diagnostic bound. This is **not** a qualified equivalent replacement and
must not be installed as a native optimization. Original strict gates remain
unchanged for candidate selection.

## Reproduction and evidence

`profile_length_scaling.py --width D --length L` uses the environment wrapper
and normal source/GPU locks. `ablate_output_ln.py --width D` runs at L768.
Both scripts record their SHA256. Raw results are mirrored under:

- `.bench/vast-20260927/results/trimul-short-large-d/length-profile-v1/`
- `.bench/vast-20260927/results/trimul-short-large-d/ln-ablation-v2/`

The JSON files contain all paired samples, cross-implementation error checks,
kernel names and CUDA activity durations; the length profiles also retain Chrome
traces. Activity profiling can perturb native durations, so full graph timing
must not be replaced by sums of profiler events. NCU was not used.
