# Triton Transition: shared forward and large-D experiments

2026-09-18, node02, 2 x H100. Local development checkout only. D768 excluded.

## Result and default routing

- **D128**: `transition_triton_b2b=True` (default), SM90/BF16/n4, M>=16384 selects one shared forward for inference and training: stats -> **LN + expand_a/b + SwiGLU + squeeze + residual**. Only training stores xn. The existing split Triton backward and four cuBLAS products are reused. Both `Transition` and `ops.transition` call the same dispatcher. Native auto CUDA/CuTe routing remains unchanged.
- **D256**: the candidate is implemented, tuned and callable, but default stays split. Preserving the actual FP32 LayerNorm parameters makes inference 6-13% slower than split. The previous matched CUDA comparison deliberately cast affine parameters to BF16; its small D256 advantage cannot be transferred to the production FP32-affine path. Full-K candidate spills: inference 122 versus 10 in the earlier BF16-affine control, at the same BM128/BN32/BK256/8-warps/3-stages config. This is observed compiler resource use, not an isolated causal proof.
- **D384/512**: three additional Triton layouts implemented and tested. All retain BF16 h, FP32 squeeze accumulation and BF16 rounding before residual addition. No hidden activation is written to HBM and no expand is duplicated per output tile. D384 output-accumulator segmentation improves on the earlier experimental b2b, but still loses to split. D512 segmentation regresses. Both widths therefore keep split as the default.
- `transition_force_split=True` or `transition_triton_b2b=False` restores split. Other GPU architectures and unsupported shapes retain their existing route.

## New wide layouts

1. Interleaved expand: one dot with alternating Wa/Wb columns, followed by split and SwiGLU.
2. Contiguous expand: one dot with adjacent Wa and Wb column groups, loaded directly through selected pointers, followed by reshape/split and SwiGLU.
3. Segmented output accumulation: independent 128-channel squeeze accumulators inside the **same CTA**, with shared h. D384 allocates three output accumulators instead of a padded 512-channel accumulator; D512 uses four.

The first two increase layout-conversion/scheduling costs; several BM32 variants lower to MMA rather than WGMMA. The segmented BM64 winner uses WGMMA. At D384 the selected segmented kernel has 255 registers/thread, no spill, 120 KiB shared memory; D512 has 255 registers, 18 spill slots and 120 KiB shared memory. Less padding does not imply a faster full module.

Final searches cover **513 layout/width/config attempts** (81+81 interleaved, 81+81 contiguous, 108+81 segmented). All valid configurations checked against a float32 reference; failed compile/resource configurations are retained in JSON. Grid axes are BM, BN, BK, warps and stages; segmented output BO is explicit. There is only a row CTA grid, so GROUP_M has no second grid dimension to reorder.

## Measurement

[Full inference/training table](timings.md), [machine-readable summary](summary.json).

Official `bench_module_transition` fixture, B1 pair L384/768, n4, BF16 activation/weights and **FP32 LN affine**, deterministic nonzero squeeze, static compile (`dynamic=False`, partial allowed, one graph observed), manual CUDA Graph, two captures with reverse arm order. Training = forward+backward, optimizer excluded. General Transition has no dropout. All 80 measured rows pass replay checks. These are same-session split comparisons, not a claim that every pre-existing split cache has been exhaustively retuned; missing legacy entries use the same 24-candidate fallback for all arms.

D128 speedup vs split: inference **1.536x / 1.575x**, training **1.089x / 1.098x**, at L384 / L768.

D384 new segmented vs old b2b: inference **1.073x / 1.082x**. Versus split it remains **11.3% / 2.6% slower**. D512's previous b2b remains faster than the new segmented variant, and split beats both.

## Tuning integration

`transition_b2b_residual_triton` is registered with a driver and checker, config CSVs, resource pruning, shape+SAVE_XN keys and the standard cache reader. The grid has 144 raw combinations, pruned to 72 D128 / 63 D256 full-K candidates per shape/mode. **540 attempts**, 468 valid measurements, cover L384/768 x D128/256 x inference/saved-xn. Failed candidates are shared-memory resource exclusions. Eight measured cache entries were saved; D256 entries support explicit experiments and are not a default dispatch claim. Other shape buckets have not been fully built in this work.

Tiling is selected from CSV/cache, not a hardcoded winning tuple in the production kernel. Cache staging originally mislabeled D128 mixed operands as BF16; `merge_cache.py` corrects that label to the actual BF16+FP32 operands before installing the entries. Both widths are then merged without parallel writers touching the same JSON.

## Verification

- 12 GPU numerical/gradient/fullgraph tests passed under compute-sanitizer; **0 memory errors**. Includes tail rows, FP32/BF16 affine, gamma=0, exact zero-squeeze residual identity, all six gradients, experimental layouts.
- Final D128-b2b / D256-split module+whole-op dispatch and fullgraph checks: 2 passed.
- Config axes, driver imports, tile-order declaration and no-autotuner-bypass checks: 102 passed.
- `git diff --check` passed.

The initial module equality test accidentally instantiated the PyTorch backend; corrected to explicit Triton before the final checks. No kernel tolerance was loosened.

## Files / reproduction

Production: `kernels/transition/triton/b2b_residual.py`. Core/experimental layout launchers: `wide_b2b.py`, `segmented_b2b.py`. Exact source snapshots and SHA-256 are in [sources.json](sources.json) and `sources/`.

Raw run: `/home/psk6950/MiniWorld/runs/transition_triton_next_20260918`. Scripts, configs, measurements and logs are copied here. `measure.py` runs the actual default D128 route and an explicit D256 candidate override after D256 default promotion was rejected; `measure_wide.py` compares split, old b2b and segmented b2b. Run GPU scripts through an allocated **node02** job and the recorded environment, not on the login node.
