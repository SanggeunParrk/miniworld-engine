# TriMul B4: native TMA implementation and dispatch

## Result

B4 (output LayerNorm backward) now has a native, cache-managed SM90 implementation.
It retains Triton's persistent row grid, feature-tile ownership, FP32 parameter
partials and the two final reductions. There is no new fusion boundary, activation
copy, concatenation or matrix multiplication. TMA loads/stores and mbarrier stage
rings are explicit; WGMMA is not applicable to LayerNorm.

The existing `trimul_sm90_kernels` setting accepts `out_ln_bwd`:

```python
from miniworld_engine import settings
settings.configure(
    engine_backend="triton",
    trimul_sm90_kernels={"front", "f567", "dual_bwd", "out_ln_bwd"},
)
```

Configure before model construction/compilation. The override is inside the
existing canonical persistent output-LN backward path. Small-M atomic and narrow-N
specialized paths keep their existing dispatch. For the benchmark's D128/H128
bidirectional module this means L768 uses TMA B4 and L384 retains Triton atomic B4.
An unsupported dtype, layout, alignment or architecture falls back to Triton
without making an aligned copy. Existing defaults and running training installs
are unchanged.

## Configuration and supported domain

- Reads `layernorm_bwd_split_triton.csv` directly: **1,200 declared configurations**.
- `BLOCK_M1`: 1, 2, 4, 8, 16, 32, 64, 128.
- `BLOCK_K`: 64, 128, 256, 512, 1024.
- Warps: 1, 2, 4, 8, 16, 32. Stages: 1–5, with real independent storage.
- Both covering feature tiles and split feature tiles are implemented. Each split
  CTA gathers whole-row c1/c2 and then reloads only its owned output tile, as Triton does.
- BF16/FP32 activations, FP32 parameters/statistics, matching row/column-major
  strides, logical row/feature tails and aligned physical pitches are supported.
- TMA's 16-byte base/leading-stride/contiguous-box requirements and shared-memory
  capacity are explicit exclusions. Small row-major tiles use exact unswizzled
  layouts, avoiding an artificial WGMMA-layout minimum. No GROUP_M axis is invented
  for this reduction: the Triton source CSV has none.
- Registration includes native build enumeration, driver/checker, source identity,
  exact shape/stride/dtype keys, partial-coverage reporting and runtime selection.
  As with the existing native kernels, an uncached supported shape uses the first
  feasible configuration; only measured cache entries establish performance.

For the hot L768 BF16/column-major/N256 workload, 495 declarations are feasible
under the current resource policy. **24 are measured and cached; 471 remain
unmeasured.** This is not a completed native cache build. The selected
BM32/BK256/8-warps/2-stages config comes from measurements, not a length-specific
constant in the launcher. The build driver covers both layouts at requested
width/dtype; production itself does not pad or copy activations.

## Performance

H100 80GB, alternating CUDA-graph measurements. B4 timings include both final
parameter reductions. These compare against the updated Triton baseline, not the
older pre-optimization LayerNorm.

|Workload|Triton|TMA candidate|Decision|
|---|---:|---:|---|
|B4 L768, BF16 N256|474.91 us|400.14 us|1.187x; 15.7% less time; wire the native path|
|B4 L384, BF16 N256|102.53 us|113.54 us (best measured)|Keep the existing atomic path|

Official bidirectional TriMul fixture: B1, D=hidden=128, dropout=0.25, static
compile and a stochastic CUDA graph, forward+backward without optimizer. Both
arms already use H100 F2/F567/B9+B10; only B4 differs.

|L|Capture|Previous H100 path, ms|With native B4, ms|
|---:|---:|---:|---:|
|768|0|6.002744|5.935768|
|768|1|6.002216|5.931184|
|384|0|1.530096|1.531136|
|384|1|1.532408|1.530200|

L768 improves by about **1.1–1.2%** additionally. L384 uses the same kernel in both
arms; its small timing difference is noise. This does not establish a 15% whole
module improvement. Every replay advances production dropout RNG. Resetting RNG
reproduces outputs/gradients, all gradients are present and buffers are overwritten.
Worst paired gradient relative-L2 is 2.10e-4 at L768 and below 1e-6 at L384.

NCU of the final B4 implementation confirms the prototype's hardware improvements:
main-kernel time 492.70 -> 419.14 us, DRAM 1.897 -> 2.212 TB/s, registers/thread
255 -> 120, active-warp occupancy 12.50% -> 24.47%. These profiler times exclude
the two parameter reductions and are not the graph timings in the table above.

## B11+B12 investigation

Three experimental implementations preserve LN-gradient BF16 rounding before
adding the residual, and relaxed FP32 atomic parameter accumulation:

1. TMA input and output staging: 239.20 -> 231.68 us (1.032x).
2. TMA inputs with direct global output stores: 239.10 -> 226.00 us (1.058x).
3. Removing redundant CTA barriers in the covering-tile probe: about the same as
   direct stores; no clear additional win.

These remain experiments. They cover the hot full-feature BF16 tile, not the full
Triton feature-tile/dtype space, and have not passed whole-module qualification.
They are not wired into production. One oversized original probe exceeded shared
memory; it is not included as a performance result.

NCU full-set measurements of the first TMA candidate (separate from graph timing):

|B11+B12 metric|Triton|TMA staged-output candidate|
|---|---:|---:|
|Duration|252.74 us|240.48 us|
|DRAM throughput|2.447 TB/s|2.561 TB/s|
|DRAM active cycles / peak|73.00%|76.39%|
|Registers/thread|198|40|
|Active-warp occupancy|11.97%|36.64%|
|Shared load bank conflicts|8,888,700|690,510|
|Shared store bank conflicts|9,141,914|224,594|

The improvement removes register/shared-memory pressure but does not reduce the
large activation traffic. This is not proof that 6% is an absolute upper limit.

## Validation and limits

- Expanded standalone matrix: **180 passed, 12 resource exclusions**, BF16/FP32,
  row/column layouts, N128/137/256/384, covering/split BK and multiple row
  tiles/warps/stages; see raw logs/summary.
- Final B4 and native-contract regression run: **74 passed, 9 skipped**. Skips
  correspond to explicit hardware infeasibility. Existing surrounding native
  tests were also run; a stale source-inspection assertion was updated to recognize
  the shared native resolver and its module-level op constant.
- Compute Sanitizer memcheck: zero errors. Racecheck: zero hazards/warnings,
  across covering BF16, split/ragged BF16, and row-major/ragged FP32 probes.
- Native cache selection was checked through the unmodified production resolver
  after merging the measured shard. Full shape/config coverage remains pending.
- No full optimizer/DDP training or convergence claim. No running job was restarted.

Raw scripts, JSON and profiler artifacts are in the workspace's
`runs/trimul_b4_production_20260918/`; selected small evidence is copied alongside
this report. The old normalization report describes the previous prototype.
