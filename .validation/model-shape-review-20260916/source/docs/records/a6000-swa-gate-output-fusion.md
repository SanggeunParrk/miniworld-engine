# A6000 SWA sigmoid-gate/output-projection fusion — 2026-09-11

SWA BF16 width-128 no-grad inference now folds `sigmoid(gate) * attention_output` into the output
projection GEMM. The gate linear remains separate. Training retains the previous path: the
full forward/backward fusion experiment did not demonstrate a substantial module-level gain.

## Experimental comparison

A6000, L384 (3072 atoms), depth1, BF16, TF32 OFF, actual torch.compile ON. Inference A5 with manual
CUDA Graph replay; training A48 with graphs disabled, forward+backward without optimizer. SWA
has no dropout. Native module constructors and timer; median of three samples per process.
Both implementations in each comparison run sequentially on the same physical A6000.

| Module | Mode | Split ms | Full fusion ms | Reduction |
|---|---|---:|---:|---:|
| swa_atom_attention | inference | 0.3657 | 0.3582 | 2.04% |
| swa_atom_attention | training | 9.4563 | 9.4509 | 0.06% |
| swa_dit | inference | 0.5308 | 0.5228 | 1.50% |
| swa_dit | training | 15.8963 | 15.7518 | 0.91% |

The experiment reuses the existing pair-attention fused forward GEMM and fused input-gradient/gate
backward body. A 108-config search at each augmentation chooses separate forward/backward tiles.
Experimental calls use those fixed raw-JIT tiles, not the pair cache. They are not presented as
production-cache benchmarks. All input/parameter gradients passed against BF16 PyTorch for two
nonzero random-weight seeds in both complete modules; maximum gradient relative L2 was 1.1165%
(2.5% bound). The gate linear and output-weight-gradient GEMM were not fused together.

## Production inference comparison

| Module | Previous ms | New cached path ms | Reduction |
|---|---:|---:|---:|
| swa_atom_attention | 0.3638 | 0.3563 | 2.08% |
| swa_dit | 0.5304 | 0.5201 | 1.95% |

Same measurement contract as above, including same-GPU comparison and observed compile evidence.
The old case disables only `_can_fuse_output_projection`; the new case uses normal production
dispatch and published cache candidates. Cache misses abort, tile-cache files stay unchanged,
and each new-kernel lookup must match a stored physical-workload record. Profiling is outside
the timer. Both inference pairs used the same source stamp within their comparison; subsequent
source edits only sorted/wrapped imports. Final training execution confirms the new kernel is
not called; timing fluctuations on that unchanged path are not claimed as a training speedup.

The production guard preserves the ordinary path for gradients-enabled execution, FP32, PyTorch
reference, MPLinear, other projection subclasses, widths other than 128, biased projections and
projection forward/pre hooks. This avoids bypassing MPLinear's weight normalization or hooks.
The fused body is shared with the existing kernel rather than duplicated; the SWA entry owns its
own row-count key, grid, driver, numerical check and workload-attributed cache.

The new grid contains 18 combinations: BM32/64, BN128, BK32, group1, warps2/4/8, stages1/2/3.
It includes the leading configurations from the broader probe. All 18 are searched separately
for each new measured workload; this is not a claim of a universal optimum.
Built BF16 A5 at atom lengths 1024/2048/3072/4096/5120/6144/8192, plus a ragged width127 case.
The published cache has 4 keys and 8 physical profiles.
Selected outputs pass against the rounded-gate BF16 matmul reference (maximum relative L2
0.00000954, bound .013).

56 GPU numerical/dispatch, registry and grid/layout regression cases passed. They include
ordinary inference, training gradients, FP32, PyTorch reference, MPLinear and projection hooks.
Scoped Ruff checks pass after import cleanup. The inference cache was published before its
production benchmark; no full backward-fusion kernel was enabled by default.

## Build-plan status

Final source-stamped sm86 plan: **1008/1008 usable keys**, no missing or invalid cache. The sweep HTML is regenerated from this plan.

[Raw measurements, gradients, cache details and job provenance](a6000-swa-gate-output-fusion.json).
