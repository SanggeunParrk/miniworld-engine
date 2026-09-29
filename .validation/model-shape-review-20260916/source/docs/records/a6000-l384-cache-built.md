# A6000 L384/A48 cache repair

Recorded 2026-09-10. The four keys identified in [the preceding inspection](a6000-l384-cache-missing.md)
are now stored in the RTX A6000 (sm86) cache. All preexisting entries were preserved.
No kernel implementation or registry tolerance changed. Small-input optimization remains deferred.

## Build and numerical validation

The four grids contained 3,528 configurations. 3,522 compiled; six LayerNorm backward configurations
exceeded the existing 60-second compile budget and were excluded. The five selected configurations
per key were checked on contiguous and column-major inputs with seeds 0 and 1: 80 candidate/input checks.

| Kernel | Input | Max relative difference | Validation policy |
|---|---|---:|---|
| `layernorm_fwd_saveact_strided_triton` | `[18432, 384]` | 0.2586% | Declared registry band against FP32 formula |
| `layernorm_bwd_atomic_strided_triton` | `[18432, 384]` | 0.2784% | Declared registry band against FP32 formula |
| `rmsnorm_fwd_triton` | `[589824, 32]` | 0.3472% | FP64 / BF16 rounding envelope; see qualification below |
| `rmsnorm_bwd_triton` | `[589824, 32]` | 0.4425% | FP64 / BF16 rounding envelope; see qualification below |

**RMSNorm qualification:** the native BF16 reference comparison exceeded the existing maximum-relative
bands (forward 0.32%, backward 0.40%) at rounding boundaries. Changing tiles did not remove this;
all 1,200 forward and 600 backward ranked candidates were rejected by that strict check over the
four large inputs. These failures have not been relabeled as registry-band passes.

The stored top five configurations were instead independently checked against the FP64 normalization
formula and analytic derivative. Each element had to satisfy `abs(error) <= 0.5 * BF16_spacing +
2e-6 * arithmetic_scale`, with no nonfinite values or violating elements. Forward scale is the absolute
FP64 output; backward scale is `(abs(dy) + abs(x)*rstd^2*mean(abs(dy*x)))*rstd`, accounting for
cancellation in the dot product. Saved forward rstd also met a 2e-6 relative bound and the unweighted
backward weight-gradient buffer was zero. All 40 RMSNorm candidate/input checks passed this policy.
This is an explicitly qualified precision validation, not a claim that the original registry bands passed.

## Native benchmark after the repair

The five affected training targets were repeated through the existing native benchmark CLI, each in
three independent processes. PyTorch and MiniWorld comparisons within each target shared one A6000.
BF16, actual torch.compile, L384, augmentation 48, CUDA Graph OFF, atom length 3072, depth 1,
mask probability 0.125, TF32 OFF; training is forward + backward without optimizer.

| Target | PyTorch ms | MiniWorld ms |
|---|---:|---:|
| adaptive_layernorm | 1.2334 | 1.6097 |
| conditioned_transition | 6.0232 | 6.2013 |
| swa_atom_attention | — | 10.0772 |
| dit | 18.0746 | 17.9118 |
| swa_dit | 17.0680 | 16.5007 |

All six previously unmeasured cells now have valid results. The whole L384 table contains 138
accepted timing rows / 46 groups, zero missing cells and zero runtime cache fallbacks. Measurements
did not change the cache. Previously measured unaffected rows were retained; their cache entries
and kernel sources are unchanged. Inference remains A5 / Graph ON.

**Scope:** this closes the inspected L384/A48 workload (all 24 traced keys now have entries). The
general build-case declaration still does not encode mode-dependent augmentation; this repair does
not establish coverage of arbitrary lengths/augmentations. The RMSNorm registry-band discrepancy
also remains documented above.

[Machine-readable build/benchmark record](a6000-l384-cache-built.json). Host-local detailed evidence:
`../team-gm/mw-l384-missing-build/` (raw ranked candidates, rejected candidates, FP64 checks, backups, merge audit);
`../team-gm/mw-l384-modules/results.{md,json}` (full native table and CSV paths).
Source snapshot: `4fa017e5636bb2a393ebb1dfc3a7231de48bfe1644c76b35bb767d375bf2e8b3`.
