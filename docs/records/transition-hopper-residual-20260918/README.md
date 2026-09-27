# Hopper Transition with residual fusion — 2026-09-18

[Subsequent saved-xn CUDA b2b development](../transition-b2b-savedxn-20260918/README.md) changes the training activation policy and default dispatch. This record preserves the preceding recompute-only measurements.

Existing H100 forward kernels now connect to residual-fused backward in both
`modules.Transition` and `ops.transition`. This development checkout contains the
implementation; running MiniWorld training jobs and the installed package were
not changed. No remote publication is part of this change.

## What changed

| Path | Forward | Backward |
|---|---|---|
| H100 D128/256 auto | Existing hand-CUDA b2b: LN normalization + dual expand + SwiGLU + squeeze + residual. Stats remain separate. | Existing recompute/stacked-dAB path, four main GEMMs. Identity gradient is added in the LN dx epilogue. |
| H100 wide-D auto / explicit CuTe | Existing CuTe LN-folded expand, followed by CuTe squeeze + residual. | Existing CuTe-family backward selection, with identity gradient in LN dx. Default Triton gate backward retains separate dA/dB and six main GEMMs. |
| Forced Triton / unsupported native shapes | Existing Triton residual split. | Existing Triton residual LN. |

The CUDA LN main kernel now accepts an optional residual gradient. It rounds
dx to the activation dtype **before** adding the residual, and does not add the
identity gradient to dgamma/dbeta. Vector loads, persistent-grid candidates,
parameter-gradient reduction, and the non-residual specialization remain.

The residual backward wrappers also preserve FP32 affine gradients emitted by
the Triton LN reducer. Casting those sums to BF16 and back to FP32 amplified
small atomic-order differences into a full BF16 ULP and failed the unchanged
`1e-4` replay check at L768. The fix removes that intermediate rounding;
it does not relax the benchmark's accuracy gate.

CuTe squeeze now owns a small epilogue subclass of Quack's SM90 TMA/WGMMA GEMM:
FP32 accumulator → BF16 → FP32 + residual → BF16 store. It keeps the separate
residual C operand, so no output staging copy or separate residual add is needed.
It uses the existing plain-SM90 configuration space, including tiles, clusters,
ping-pong, swizzle and static/dynamic scheduling. This is **not** a claim that
the native space matches every Triton configuration axis or has been fully tuned.

## Selection rules

Defaults: `transition_residual_fusion=True`, `transition_h100_residual=True`.
`engine_backend="triton"` or `transition_force_split=True` keeps the Triton path.
Native qualification is exact Hopper, BF16, n=4, D in {128,256,384,512,768},
and flattened row count M divisible by 128.
The hand-CUDA path retains its existing cuBLASDx/CUTLASS header dependency;
the validation launcher sets `MINIWORLD_MATHDX_HOME` to the available mathdx
installation. Native build errors are reported by the new forward wrapper.

- Auto D128/256 uses b2b when `transition_cuda_b2b=True`.
- Auto wide-D uses CuTe for M≥16384. Small wide token workloads keep Triton,
  following the earlier measured split-path advantage.
- Explicit `implementation="cute"` uses CuTe expand/squeeze for qualified shapes,
  including small widths, and now gets fused residual backward too.
- The existing backward backend controls still select the CuTe-family gate path.
- At D128/M≥16384, the current atomic Triton LN/residual epilogue is selected:
  profiling found it faster than the persistent CUDA main+reduce with residual
  traffic. The new native LN/residual implementation remains used by other
  eligible paths and is independently tested.
- Unqualified module shapes retain their existing fallback. Unsupported direct
  `transition_residual_hopper` calls raise a clear error.

```python
from miniworld_engine import settings

# Native Hopper when qualified; residual fusion in both directions.
settings.configure(engine_backend="auto", transition_residual_fusion=True,
                   transition_h100_residual=True)

# Same comparison option as before.
settings.configure(engine_backend="triton")
```

## Validation and measurement

Final Hopper tests run on **node02**, including all six input/parameter gradients,
gamma=0, tail handling, exact CUDA LN residual rounding and affine-gradient
isolation, explicit CuTe, forced Triton, and static fullgraph compile + CUDA graph
backward. The pre-existing Triton residual tests were also run after the shared
kernel edits. CPU checks cover native precompilation, lazy selection and build
width coverage.

Module benchmarks use the official `bench_module_transition` fixture, batch 1,
BF16, n=4, one layer, deterministic **nonzero** squeeze weights shared by all
backends, static compile (`dynamic=False`) and manual CUDA graph. Training is
forward+backward, excluding optimizer. The fixture allows partial compilation;
the observed compiled graph count and execution validation are in every row.
Separate tests use `fullgraph=True`. Each timing is measured in two independent
captures with reversed backend order. No dropout is present in general Transition.

Triton uses the current cache or a bounded 24-candidate miss search; native
implementations use a valid cache or their declared default. **This change does
not include a complete native configuration/cache rebuild.** The `legacy` arm
means residual fusion disabled with current shared kernels, not a historical
checkout of the pre-change source.

Exact final timings, ratios, test outcomes, source hashes and trace evidence are
recorded in [summary.json](summary.json). Individual fixture outputs and kernel
traces are retained alongside it. Kernel traces come from torch.profiler; this
report does not make an NCU roofline or optimality claim.
Some profiler captures returned no CUDA events (including the final wide-D
training captures). Those captures are not evidence for kernel counts; their
separate CUDA-event timing and graph correctness measurements remain recorded.

See [the final node02 timing table](timings.md) for the measured module times.
The reproduction launcher uses a dedicated `TORCH_EXTENSIONS_DIR` so independent
cache builders cannot replace extensions with the same name from another checkout.

## Decisions from the measurements

- Restoring b2b gives a substantial D128 inference improvement. That forward
  already had a residual epilogue; this change restores its default reachability.
- CUDA LN residual fusion alone saved much less time than the forward improvement.
  The faster current Triton LN epilogue is therefore retained for large D128 pairs.
- A trial routing wide-D through raw-input stacked backward did not improve the
  total module time. The final wide-D path preserves its existing backward
  algorithm and adds only the residual epilogue.
- The remaining D128 training cost includes activation recomputation and cuBLAS
  GEMMs. Forward speedup is not the same as whole-training speedup.

## Cache/build integration

Native LN cache buckets distinguish residual present/absent, and its build driver
exercises both. CuTe squeeze precompilation targets the new rounded epilogue;
its build width ladder covers 128/256/384/512/768, including explicit CuTe calls.
Source identity invalidates stale native cache entries. Existing unrelated cache
edits in the checkout were preserved and are not part of this implementation.

## Main code

- [Native Transition wrapper](../../../src/miniworld_engine/kernels/transition/hopper.py)
- [CuTe squeeze epilogue](../../../src/miniworld_engine/kernels/transition/cute/squeeze_residual.py)
- [CUDA LN epilogue](../../../src/miniworld_engine/kernels/layernorm/cuda/layer_norm_cuda_kernel.cu)
- [Numerical and graph tests](../../../tests/numerics/test_transition_hopper_residual_gpu.py)

The initial audit and measurements used node01. On the user's node02-only
instruction, allocation 13258 was canceled and subsequent GPU work used node02
allocation 13264. The final benchmark table uses node02 results only.
