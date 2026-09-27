# D128/256 segmented Triton b2b

2026-09-18 · node02 · 2 H100 GPUs · local development checkout.

## Delivered

D128 and D256 now support the segmented output-accumulator algorithm previously implemented at D384/512. Both are connected to the **same default Triton inference/training forward** on SM90/BF16/n4/M>=16384:

```
Triton LayerNorm (FP32 affine) -> xn
    -> [expand_a + expand_b + SwiGLU + segmented squeeze + residual] -> y
```

The second line is one kernel. One CTA owns all output segments and reuses the same hidden activation; expand is not repeated for each output segment. h does not reach HBM. xn is a temporary tensor in inference and saved for the unchanged backward in training. Squeeze rounds to BF16 before adding residual, preserving the existing boundary.

`Transition` and `ops.transition` share the dispatcher. `transition_triton_b2b=False` or `transition_force_split=True` restores split. D384/512 still use split. Native auto CUDA/CuTe dispatch is unchanged. This checkout has not been installed into the running MiniWorld training environment.

## Implementation

- D128 output tiles BO32/64; D256 BO64/128 (four or two accumulators).
- Full-K operand hoisted before the hidden loop: load/normalize once, optionally save xn once, reuse across hidden tiles. The streamed-K path remains available.
- Implemented LN-separate, LN-fused inference and LN-fused saved-xn training variants. **LN-separate wins as a full module at both widths**, so it is the default for both modes. LN-fused remains an explicit callable variant.
- The selected separate config at both lengths: BM128, BN32, BK=D, BO=D/2, 8 warps, 3 stages. These are measured CSV/cache results, not constants in the kernel. D128 compiled with 128 registers/thread; D256 with 210. Both have zero spills.
- Backward reuses existing production classes/helpers. The fused experimental wrapper was aligned with the same saved-xn Triton backward so comparison measures forward scheduling, preserving FP32 affine gradients.

## Performance

[Complete timings](timings.md) · [summary JSON](summary.json).

Actual default module, median of two captures in reverse order; B1 pair L384/768, n4, BF16 activation/weights, FP32 LayerNorm parameters, nonzero squeeze, static compile (one graph observed), manual CUDA Graph. Training means forward+backward, no optimizer. General Transition has no dropout.

| D | L | Split forward | New default forward | Speedup | Whole training speedup |
|---|---|---:|---:|---:|---:|
| 128 | 384 | 0.2763 ms | 0.1654 ms | 1.670x | 1.116x |
| 128 | 768 | 1.0346 ms | 0.5937 ms | 1.743x | 1.128x |
| 256 | 384 | 0.7268 ms | 0.5662 ms | 1.284x | 1.074x |
| 256 | 768 | 2.8957 ms | 2.1512 ms | 1.346x | 1.074x |

The separate comparison also includes the previous full-output b2b and the new fused-LN variant. Compared with the prior full-output b2b, the new separate variant improves forward by about 7-9% at D128 and 43% at D256 in that matched run. It preserves FP32 affine, unlike the earlier matched native-CUDA control which used BF16 affine.

An initial D256/L384 default run measured ~0.709 ms; its selected tile was not recorded. A fresh process selected the measured BM128/BN32/BO128/8-warps/3-stages winner and returned ~0.57 ms. Runtime autotuning is now measured with CUDA Graph, matching the cache and module timing method, and the final run records selected configurations. The initial and repeated measurements are retained; the initial discrepancy alone does not establish its exact cause.

## Configs and cache

`transition_segmented_b2b_triton` has registry/driver/checker, CSV config sets and standard cache integration. Keys include shape, NORMALIZE and SAVE_XN. A row-only grid owns every output column, so there is no second CTA grid axis for GROUP_M to reorder; this is recorded in `tile_order_exempt.csv`.

- Raw grid: BM32/64/128, BN32/64/128, BK=D, BO=D/4 or D/2, warps4/8, stages1/2/3.
- D128: 324 mode/config attempts, 300 valid. D256: 270 attempts, 198 valid. Compile/shared-memory failures are recorded; no illegal access occurred.
- The five best candidates per mode were measured through the actual autotuned wrapper at **both L384 and L768** (60 additional measurements).
- Twelve cache entries cover two widths, two lengths and three forward modes. These are measured shortlists, not a claim of full-grid L768 tuning or coverage of every possible model shape. Other buckets use bounded runtime tuning.
- Cache writing was staged per width and then merged; no simultaneous writers changed the same cache JSON.

## Validation

- 24 tests under compute-sanitizer: numerical parity, all six gradients, gamma=0, exact residual identity with zero squeeze, 129-row tails, two/four segments, streamed/full K, separate/fused LN and fullgraph forward/backward. **Zero memory errors.**
- Four autotuned-wrapper tests repeated under compute-sanitizer after aligning runtime tuning to CUDA Graph: passed, zero errors.
- Actual module/whole-op default dispatch fullgraph tests: both widths passed.
- Existing wide-D suite: 16 passed after the full-K implementation change.
- Config/registry/driver/order/bypass suite: 103 passed.
- All candidate/default module rows passed CUDA Graph replay checks; existing split/backward cache misses use the same bounded candidate budget for all arms.

Source snapshots and SHA-256: [sources.json](sources.json), `sources/`.
Raw reproducible scripts/configs/logs: `/home/psk6950/MiniWorld/runs/transition_segmented_small_20260918` (copied here). GPU scripts must run through an allocated **node02** step. D768 remains excluded.
