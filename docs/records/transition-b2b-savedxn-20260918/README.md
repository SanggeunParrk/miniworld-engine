# H100 b2b: saved normalized activation with residual — 2026-09-18

The **CUDA b2b kernel itself is changed**. During training it now optionally emits
its already computed BF16 normalized operand `xn` via coalesced 128-bit global
stores. Backward consumes that saved operand instead of repeating normalization
and writing another `xn`. Forward still fuses `y=transition(x)+x`; backward still
fuses `dx=dx_LN+dy` in the LN epilogue, without adding identity gradients to affine
parameter gradients.

## Default routing

`transition_h100_save_xn=True` by default. Qualified native b2b inputs with
D128/256, M≥16384 and gradients enabled use the saved path. Inference, smaller M,
and an explicit False setting keep the original recompute behavior. Wide CuTe is
unchanged. Forced Triton remains supported.

The autograd wrapper decides whether gradients are enabled before entering the
custom Function. The CUDA `SAVE_XN` template removes the stores entirely from the
inference specialization. Saving adds **M×D×2 bytes of retained activation**:
36MiB at L384/D128, 144MiB at L768/D128, and 72MiB at L384/D256 (B1).
This is retained tensor size, not a measured whole-model peak-memory increase.

The saved value is the exact BF16 shared-memory operand used by forward WGMMA.
There is no separate normalization/store kernel. Both D128 and D256 saved backward
use the existing **Triton saved-xn stacked** gate backward, followed by four main
cuBLAS GEMMs and LN+residual. D256's older raw-input CUDA gate backward is still
available through the non-saved path. This is not a claim that the entire backward
has become native CUDA.

## Results

[Final module times and matched variants](timings.md) · [machine-readable summary](summary.json).

All measurements use **node02 only**, B1, n4, BF16, nonzero squeeze weights,
static compile (`dynamic=False`, partial allowed, one observed graph), manual
CUDA graph, and two captures with reversed backend order. Training means
forward+backward without optimizer. No dropout exists in this general Transition.

The four-arm comparison uses the official module fixture and forces each candidate.
The final two-arm comparison exercises the **actual default auto dispatch after
cache publication**, with the same fixture. The final native tuning winners match
the declared defaults used in the earlier four-arm run. Small differences between
runs are measurement variation, not additional optimizations.

Forward b2b vs current CuTe expand+squeeze/residual is about 1.93x at L384/D128,
2.05x at L768/D128, and 1.24x at L384/D256. These paths have different fusion
boundaries; this is not a same-algorithm C++ vs Python DSL language comparison.

## Profile explanation

[Full L768/D128 CUDA events](profile-L768-D128.json):

| Kernel | Recompute | Saved xn |
|---|---:|---:|
| CUDA b2b forward | 499.23µs | 504.89µs |
| Triton gate backward | 1480.64µs | 1229.40µs |

The additional forward store costs about 5.7µs in this trace, while gate backward
saves about 251µs. These are individual torch.profiler samples, not NCU roofline
results or the CUDA graph module benchmark itself.

## Configuration and cache

The existing legal b2b grid is retained: four configurations at D128 and two at
D256. Every candidate was compiled and checked for identical output and saved xn.
Inference and saved-training modes have separate native cache buckets and share
the same candidate space. The regular native build driver now exercises both.

Full candidate measurements were recorded for **L384/768 D128 and L384 D256**, each
in saved/non-saved mode: six buckets, 20 configuration/workload measurements,
zero failed candidates. Native tuning used 256MiB cache clearing and 100ms budget.
Final shards were checked against the final native source identity before merging
only `transition_fwd_b2b_sm90_cuda`. This does not complete all native kernels or
all shapes. Conservative native source identity changes can stale other native
entries; they were not rebuilt here.

## Validation

- **52 tests passed**: native residual/saved-xn, prior Triton residual, compile and
  CUDA graph, and native compile/cache checks.
- **2 compute-sanitizer tests passed; 0 memory errors**, covering D128/256 saved
  stores and residual output.
- Zero squeeze weights give exactly `y=x`, `dx=dy`, `dgamma=dbeta=0`.
- Nonzero weights and zero gamma entries exercise all six input/parameter gradients.
- Every final module row passed the original accuracy and graph-replay checks.

All changes are in this local development checkout. Installed MiniWorld training
and remote repositories were not changed. The earlier residual implementation and
recompute-only results remain in [the preceding record](../transition-hopper-residual-20260918/README.md).

## Code

- [Modified CUDA b2b](../../../src/miniworld_engine/kernels/transition/cuda/transition_b2b_kernel.cu)
- [Native launch/cache boundary](../../../src/miniworld_engine/kernels/transition/cuda/__init__.py)
- [Autograd and auto policy](../../../src/miniworld_engine/kernels/transition/hopper.py)
- [Numerical tests](../../../tests/numerics/test_transition_hopper_residual_gpu.py)

Source hashes, native identity, raw measurements, tuning shards, and logs are
retained alongside this record. `env.sh`, `measure.py`, `tune.py`, and
`inspect_kernels.py` reproduce this workspace's setup; GPU work requires a node02
Slurm allocation.
