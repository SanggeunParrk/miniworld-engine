# Triton wide-D Transition b2b experiment — 2026-09-18

## Result and decision

Implemented and measured **D384/512 only**, on **node02**, allocation 13275.
Both complete b2b variants work, but neither beats the current split forward.
They remain explicitly configured experimental APIs; production dispatch is unchanged.
D768 is outside the development plan.

- `transition_wide_b2b`: existing LN saving xn, followed by one Triton
  expand/SwiGLU/squeeze/residual kernel. Existing split backward is reused.
- `transition_wide_b2b_fused_ln`: existing stats followed by one Triton
  normalization/expand/SwiGLU/squeeze/residual kernel. Training also saves xn;
  inference omits the save. Backward uses existing saved-xn stacked gradients.

Implementation: [wide_b2b.py](../../../src/miniworld_engine/kernels/transition/triton/wide_b2b.py).
Both avoid the hidden h tensor's HBM write/read and compute expand only once per
hidden chunk for each row tile. One program owns all output channels, padded to
512 at D384; the contraction K loop still stops at the real width.
BM, BN, BK, warp count and stages are independent explicit configuration inputs.
There is only one output-column tile, so GROUP_M would not change launch order.
Squeeze rounds to BF16 before adding the residual, matching the current split.

## Module measurements

[Full table](timings.md) · [raw samples and summary](summary.json).

Official `bench_module_transition`, B1 pair input `[1,L,L,D]`, n4, one layer,
BF16 with FP32 affine parameters, deterministic nonzero squeeze weights, residual
included. Static compile (`dynamic=False`, one observed graph), manual CUDA graph.
Training includes forward+backward without optimizer. No dropout exists in this
general Transition. Two captures with reversed backend order; table uses medians.

| D | L | Mode | Triton split ms | H100 CuTe ms | New b2b ms | New LN+b2b ms |
|---:|---:|---|---:|---:|---:|---:|
| 384 | 384 | inference | 1.436 | 1.340 | 1.691 | 2.115 |
| 384 | 384 | training | 4.675 | 4.774 | 5.002 | 5.489 |
| 384 | 768 | inference | 5.668 | 5.448 | 6.457 | 7.948 |
| 384 | 768 | training | 19.165 | 19.035 | 19.516 | 21.768 |
| 512 | 384 | inference | 2.413 | 1.827 | 2.773 | 3.500 |
| 512 | 384 | training | 7.804 | 7.254 | 7.940 | 8.666 |
| 512 | 768 | inference | 9.380 | 7.557 | 10.637 | 13.455 |
| 512 | 768 | training | 29.934 | 28.561 | 31.391 | 35.834 |

These compare complete current paths. H100 auto uses CuTe forward and a mixed
Triton/cuBLAS/native backward. The native baseline uses current cache/defaults;
Triton split cache misses search a heuristic 24 candidates. Neither baseline was
exhaustively retuned. Small training differences should not be treated as strong
dispatch evidence from only two captures.

## Search and final implementation

Search records and CSVs are alongside this file: `tune*.json`, `search*.csv`.
Across initial, expanded and refined searches, **158 distinct tile configurations**
were attempted per width. The searches considered BM16/32/64/128, BN16/32/64/128/256,
BK64/128/256/512, 4/8/16 warps, and 1/2/3 stages in recorded subsets, not their entire
Cartesian product. The eight fastest unique normalized-input candidates were also
measured with normalization fused. Selection was done at L384; L768 measures the
selected configuration without claiming an independent exhaustive search.

Both widths selected BM64, BN128, BK64, 8 warps, 3 stages. Configurations live in
the explicit `selected*.json` files; no performance winner is hardcoded in the kernel.

An early BM64/W4 candidate faulted with an illegal memory access. Its output
accumulators alone require 256 FP32 values per thread. The following 19 results
in that process were not independently meaningful. Those were rerun in fresh
processes (`tune-retry*`) after adding a resource guard that rejects configurations
whose output accumulators alone exceed 255 registers/thread. The safe retries did
not improve the selected winner. Resource failures, PTXAS failures and poisoned
initial attempts remain recorded, rather than being mislabeled as numerical passes.
There are 191 attempts per width including these retries and normalization variants;
122 attempts per width completed correctness and timing checks.

The final SAVE_XN branch stores its operand in a short K traversal **before** the
hidden-channel GEMM loops. This avoids a conditional global store in every hot
iteration. It repeats normalization once for the saved operand but emits no extra
kernel. In the earlier implementation, D512/L384 LN+b2b training took 13.677 ms;
after moving the store it took 8.666 ms in the final comparison. These are separate
matched runs, not an isolated kernel-level causal measurement. Previous module
samples are retained in the raw run's `before_store_hoist/` folder.

## Compiled code and NCU

Actual selected PTX, TTGIR, CUBIN and SASS are retained in the raw run directory:
`/home/psk6950/MiniWorld/runs/transition_triton_wide_20260918/best-D{384,512}.*`.
PTX contains WGMMA; SASS contains HGMMA. Input movement uses cp.async, not TMA.
Smaller BM16/32 candidates used mma.sync instead and were slower.

The selected normalized-input kernel uses **255 registers/thread**, four compiler
spill slots and 144 KiB compiler-reported shared memory (NCU allocated 145 KiB).
NCU was collected for the normalized-input specialization, whose arithmetic was
unchanged by the later SAVE_XN-only edit. Its single-kernel profile is separate
from the graph module timings above.

| D | SM throughput | DRAM throughput | L2 throughput | Active warp occupancy |
|---:|---:|---:|---:|---:|
| 384 | 39.45% | 5.35% | 38.55% | 12.46% |
| 512 | 39.35% | 5.17% | 26.63% | 12.45% |

Evidence: [NCU metrics](ncu-summary.json), `ncu-raw-D*.csv`, `kernel-D*.json`.
Low DRAM utilization and high register/shared-memory consumption support the
interpretation that this design is limited by on-chip resource use and scheduling,
not HBM saturation. This does not prove every possible wide-D b2b must lose.
An H100 port would need a different producer/consumer or accumulator distribution
to address these limits; merely translating this kernel to CUDA does not establish
a speedup.

NCU initially collected reports but failed to render metrics due to its embedded
Python loading user-site packages. Importing the reports with
`env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 ncu --import ... --page raw --csv`
resolved the reporting issue. Final raw CSVs contain the selected kernel metrics.

## Verification and reproduction

- [GPU tests](../../../tests/numerics/test_transition_wide_b2b_gpu.py): output and all
  six gradients, zero gamma, row tails, exact zero-projection residual identity,
  resident/tiled operands, selected configuration and static fullgraph compile.
- `memcheck-final.log`: **16 tests passed, compute-sanitizer 0 errors**.
- All 64 final benchmark rows passed finite-value and CUDA-graph replay checks.
  Reference output/input-gradient relative errors are preserved in `summary.json`.
- `sources.json` hashes the final source and tests.

Scripts are archived alongside this record. Run GPU scripts only through a node02
allocation. `env.sh` uses the development checkout and isolated extension cache.
Raw run directory contains exact logs, compiled artifacts, NCU reports and prior
samples. No runtime default, installed training package or remote repository changed.
