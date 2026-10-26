# OuterProductMean: one fused outer product + projection kernel (H100, 2026-10-26/27)

Branch `perf/final-opm-fuse` on `integ/ladder-all` (e6a635d7). Rows measured with `benchmarks/runners/ladder.py` (group `^opm`, 3 repeats, CUDA graph, one H100, torch 2.13+cu129).
Raw rows: [data/opm_fused_ladder_20261026.csv](data/opm_fused_ladder_20261026.csv); "before" = [trimul_opm_ladder_20261026.md](trimul_opm_ladder_20261026.md) ([csv](data/trimul_opm_ladder_20261026.csv)). 28 rows, all OK, no routing gap.

## What changed

* **`opm_fused_infer`** (`integrations/csrc/opm_epilogue.cu`, extension flavour `_fused`, `settings.opm_fused`, default on): the grouped outer product and the c_hidden^2 -> c_z projection in one persistent
  TMA + wgmma (sm90a) kernel; the [N*32, N*32] product (134 MB at L256, 1.2 GB at L768) never reaches HBM. Design in `docs/gpus/h100/opm/opm.md` ("Fused inference kernel"). The decisive step was the work item:
  with 32 pairs per tile the 256 KiB weight was streamed per tile and the projection cost 5 us per tile (L2 / latency bound, 277 us at L256, slower than cuBLAS + epilogue 250 us); splitting the projection over four c-passes
  of a 16 i x 8 j = 128-pair item keeps the 128 x 128 fp32 result in registers and streams the weight once per 128 pairs (212 us).
  Kernel alone, L256 S1024: mainloop 178 us (cuBLAS 190 us incl. its D store), + bf16 conversion / sync 14 us, + projection 20 us (tensor-bound: 12.5 % extra flops), epilogue 2 us (staged, coalesced).
* Serves inference at d_msa 64 / d_pair 128: MiniWorld, RF3 (projection biases), ESMFold2 order (divide after), with or without the pair residual; N % 16, S % 64. Training keeps cuBLAS + epilogue (the backward needs the product).
* The mask count is the popcount of the bit rows inside the kernel (the `opm_count_tiled` launch and its [N, N] table are gone for inference). The packed / permuted weights are cached with `_capture.param_forms`
  (the pack launch and the per-call casts are gone). Chai-1: the weight forms (fp32 LN affine, stacked projections, W' / b') are cached the same way (they were cast per call in fp32-master mode).
* Inference equals the training forward to bf16 noise, not bitwise (same bf16 products, projection summed in another order): two assertions (`test_opm_wide_sm90a_gpu`, `test_opm_train_gpu`) were relaxed accordingly, nothing else.

## Table (ms; ratio = CUDA / portable; target <= 0.95)

| row | before CUDA | before portable | before ratio | after CUDA | after portable | after ratio |
|---|---|---|---|---|---|---|
| OPM MiniWorld S1024 inference L256 | 0.3080 | 0.3213 | 0.958 | 0.2653 | 0.3137 | **0.846** |
| ... L384 | 0.6515 | 0.6809 | 0.957 | 0.5789 | 0.6759 | **0.856** |
| ... L768 | 2.5209 | 2.5952 | 0.971 | 2.1659 | 2.5432 | **0.852** |
| OPM MiniWorld training L256 | 0.9373 | 1.0709 | 0.875 | 0.9327 | 1.0665 | 0.875 |
| ... L384 | 1.9262 | 2.2517 | 0.855 | 1.9379 | 2.2271 | 0.870 |
| ... L768 | 7.1884 | 8.2500 | 0.871 | 7.1320 | 8.2140 | 0.868 |
| RF3 (divide before) inference L256 | 0.2099 | 0.2244 | 0.935 | 0.1602 | 0.2228 | **0.719** |
| ... L384 | 0.4370 | 0.4657 | 0.938 | 0.3362 | 0.4641 | **0.724** |
| RF3 training L256 | 0.6735 | 0.7743 | 0.870 | 0.6673 | 0.7702 | 0.866 |
| ... L384 | 1.3236 | 1.6024 | 0.826 | 1.3133 | 1.5690 | 0.837 |
| RF3 (divide after) inference L256 | 0.2099 | 0.2216 | 0.947 | 0.1598 | 0.2194 | **0.728** |
| ... L384 | 0.4351 | 0.4561 | 0.954 | 0.3438 | 0.4537 | **0.758** |
| RF3 (divide after) training L256 | 0.6708 | 0.7729 | 0.868 | 0.6664 | 0.7700 | 0.865 |
| ... L384 | 1.3223 | 1.5870 | 0.833 | 1.3130 | 1.5974 | 0.822 |
| Chai-1 inference L256 | 0.1917 | 0.2173 | 0.882 | 0.1779 | 0.2173 | 0.819 |
| ... L384 | 0.3671 | 0.4462 | 0.823 | 0.3540 | 0.4444 | 0.797 |
| Chai-1 training L256 | 0.6654 | 0.7598 | 0.876 | 0.6592 | 0.7564 | 0.872 |
| ... L384 | 1.2596 | 1.4504 | 0.868 | 1.2477 | 1.4470 | 0.862 |

Every OPM row is <= 0.875. Module-level timings vs the kernel-alone figure (212 us at L256): the module row adds the prologue (27 us) and the harness's residual / launch overhead.
Noise (spread of the 3 repeats) 0.5-3.8 %.

## Parity and tests

* `tests/gpu/sm90/test_opm_fused_gpu.py` (new, 36 cases): output against the fp64 statements with the engine's bf16 error <= 1.5x the bf16 statements' own (no floor), MiniWorld / ESMFold2 / RF3 x (S, N) x residual, tokens with count 0 (clamp);
  against the unfused composition; bitwise run to run; CUDA-graph replay with in-place refill; `torch.compile`; in-place weight update seen through the cached forms; c_z != 128 keeps the unfused route.
* Related GPU files, final tree: opm_fused 36 passed, opm_variants 18, opm_wide 13, opm_train 8, h100_module_wiring 34 passed; `test_msa_triton_gpu` 48 passed / 9 skipped (first tree, kernel since changed only in the epilogue).
  Base: opm_wide 13 passed, opm_train 8 passed (before the two relaxed assertions the fused tree failed 4 + 1 cases exactly on the bitwise inference-vs-training assertion).
* CPU suite: 4368 passed, 2 failed + 2 errors, **identical to base** (`test_key_flags_are_built_on_both_sides`, `test_no_stale_caches`, `test_a_ladder_offers_what_wins` x2). One allow-list key of `test_layernorm_is_never_bf16` follows the moved Chai-1 weight-cast line.
* `dev derive --arch sm86` run in this tree (`registry_kernel.units.json`), sweep page regenerated (unchanged).

## Open

* `test_op_contracts_gpu.py` was not run to the end (cancelled after ~17 min in the first run); `pwa` rows could not be re-measured in this environment (`CutlassMissing` for the PWA training CUDA rung; PWA code is untouched).
* No registry.csv row: the H100 OPM CUDA kernels are in `integrations/csrc` and have no registry rows (as before); a driver / check row for `opm_fused_infer` was not added.
* The fused kernel is not yet at its ceiling: bf16 conversion + sync (14 us of 212), a BK = 64 ring (does not fit with the 64 KiB weight stage; ~4 % mainloop), TMA multicast across a 2-CTA cluster (cuBLAS' 2x1 trick, ~3-4 %) and overlapping the projection of pass q with the mainloop of q+1 (needs a second accumulator set) are untried.
* Training forward is still cuBLAS + epilogue (0.87 of portable); fusing it needs the product (or its recompute) for dWo.
* The jit extension flavour `_fused` is built on first use (about 100 s); it is not in a prebuilt bundle yet.
