# H100 kernels inside the Triton bidirectional TriMul algorithm

`settings.configure(trimul_sm90_kernels={"front", "f567", "dual_bwd"})` selects
individual TMA/WGMMA kernels before model construction/compilation. The default
is an empty set. Explicit PyTorch modules remain references. These overrides
retain the Triton fusion boundaries rather than selecting the legacy CuTe algorithm.
The implementations currently require SM90 and BF16.

## Preserved contracts

- F2: four projections, raw interleaved preactivation saves, FP32 sigmoid/product,
  BF16 product rounding before masking, channel-major left/right outputs in
  disjoint views of one packed allocation. No global intermediate or copy.
- F567: independent projection/gate GEMMs using the materialized affine LayerNorm
  result and original normalized input, original BF16 rounding, saved projection/
  gate, dropout scale and residual. No folded weights or extra bias preparation.
- B9+B10: gate GEMM rounds to BF16 before addition to the FP32 front GEMM.
  Production column-major dconc views are consumed directly.
- F1/F4 LayerNorm, contraction GEMMs and other backward boundaries are unchanged.
  Inference retains its fused F4567 output kernel; F2 supports both save modes.

## H100 implementation

### F2

Both implementations issue explicit TMA loads/stores and WGMMA.

- **Four warps:** one CTA handles one M tile and loops over the channel tiles.
  When K fits the requested stage ring, A remains in shared memory across those
  channel tiles. Larger K refills the same ring. Retired B storage holds disjoint
  raw and gated output tiles. STSM and TMA write the final buffers. Warp shuffles
  convert interleaved gate/projection fragment ownership to the gated store layout.
  Barrier phases account for each stage's use count across channel tiles.
- **Eight warps:** Quack producer/consumer mainloop. M64 uses a 128-register
  consumer budget, allowing two resident CTAs where shared memory also permits.
  The persistent grid derives its residency from register/thread budgets and a
  conservative shared-memory bound. M128 retains the larger register budget.
  Channel grouping derives from the actual projection tile count, including
  non-power-of-two counts. No sequence-length switch or new CSV axis is added.
- Both use full-range FP32 division for sigmoid. The old Quack reciprocal flushed
  very small nonzero gates to zero; explicit extreme-logit tests cover this fix.

### F567

Uses the same `div.full.f32` operation as Triton instead of the costly rounded
reciprocal. Residual fragments use LDSM; aligned dropout scales use paired BF16
loads, with a masked scalar fallback for tails. The two independent reductions
reuse an A/B stage ring after WGMMA completion and a CTA barrier.

At L384, a matched config changed from 162 to128 registers and from74752 to41984
shared bytes. Actual active-warp occupancy increased from18.06% to24.16%.

### B9+B10

Rounds and retains the first GEMM's BF16 result before the second GEMM to shorten
FP32 accumulator lifetimes. STSM gathers the final result into retired A storage;
TMA writes it directly to the output. Storage is sized for the larger of the input
ring and output tile. Widths incompatible with TMA store alignment retain a masked
scalar store. Wait-group overlap and K-loop unrolling experiments that regressed
performance were not retained.

For the actual D128/H256 module this kernel has **KG128/KP1024/N128**. Historical
KG256 standalone measurements describe a different shape and must not be mixed in.

## Configuration and cache

`autotune/trimul_sm90_config.py` reads the corresponding Triton CSV directly:

| Native operation | Shared Triton operation | Declared configurations |
|---|---|---:|
| trimul_inproj_gemm_gate_mmajor_sm90_cute | trimul_gemm_gate_mmajor_triton | 864 |
| trimul_output_f567_train_sm90_cute | trimul_output_f567_train_triton | 3072 |
| trimul_input_dual_bwd_sm90_cute | trimul_input_dual_bwd_triton | 1152 |

All three support M64/M128 with four/eight physical warps. WGMMA requires groups
of four warps; one/two-warp emulation and logical M16/M32 padding are not implemented.
Shared-memory exclusions are explicit. Declared and executable domains therefore
differ. Stages allocate real buffers; GROUP_M retains its meaning where the Triton
CSV includes it. F2's original CSV has no GROUP_M axis.

Native keys include tensor extents, strides, dtypes and source identity, including
`front_single_warpgroup.py`. A cache miss uses the first feasible candidate, not a
hardcoded shape winner. Drivers/checkers/candidate enumeration use the native build
path. These kernels compile on the allocated GPU. Measurements below use explicit
candidate manifests and do not imply that the full native cache has been built.

## Verification and performance

Final validation on H100 80GB:

- 95 integrated GPU regression tests and414 registry/config/launch checks passed.
- Selected Compute Sanitizer memcheck and racecheck: F2 32, F567 26, dual backward32
  cases, all zero errors/hazards/warnings. This is not every declared configuration.
- Whole-module output and all gradients matched Triton at L128 eager/static
  compile and L384 static compile with nonzero weights, holed masks and fixed
  nonzero dropout scales. Worst relative L2:7.98e-7. Zero scale preserved residual
  output/input-gradient identity and zero parameter gradients.
- Four-warp F2's actual cubin contains TMA, HGMMA and STSM,128 registers, no local/
  stack spills. NCU reports50176 dynamic shared bytes and23.52% active-warps occupancy.
- Eight-warp scheduling policy passed16 cold/warm cache and CUDA-graph checks.

Representative Triton binaries already use WGMMA; the inspected configurations
use cp.async rather than TMA. Component times are CUDA-graph microseconds:

| Kernel | L | Triton | CuTe | Triton/CuTe |
|---|---:|---:|---:|---:|
| F2, eight warps |128|26.715|26.385|1.013x|
| F2, four warps |384|221.739|213.979|1.036x|
| F2, four warps |768|875.087|828.302|1.056x|
| F567 |128|9.554|9.729|0.982x|
| F567 |384|107.467|100.188|1.073x|
| F567 |768|417.103|388.902|1.073x|
| B9+B10, KG128 |128|14.362|13.130|1.094x|
| B9+B10, KG128 |384|141.964|135.110|1.051x|
| B9+B10, KG128 |768|535.993|517.291|1.036x|

F2 four-warp L128 is slower at33.551us; eight-warp is214.927/853.572us atL384/768.
The final F2 comparison rotates the same inputs across three implementations for
8rounds, following18 Triton candidate checks atL384,8 four-warp candidates and
bounded eight-warp resource experiments. F567 compares a common13-config grid plus
an existing Triton candidate. Dual backward compares30 common configs atL384 and
5 shared shortlisted configs atL128/768. These are bounded searches, not complete
retuning of the expanded F2 domain or global optima. No per-length winner is hardcoded.

Official module setup B1/L384/D128, BF16, static compile+manual CUDA graph,
FWD+BWD without optimizer, dropout0,12 alternating rounds:

| Replacement | Time ms | Triton/CuTe |
|---|---:|---:|
| Triton baseline |1.615280|1.000x|
| F2 four-warp only |1.610768|1.003x|
| F567 only |1.612624|1.002x|
| B9+B10 only |1.610576|1.003x|
| All, F2 four-warp |1.602736|1.008x|
| All, F2 eight-warp |1.603888|1.007x|

The end-to-end improvement is about0.8%, substantially smaller than component
speedups. Each paired round measured1.0069-1.0096x for the four-warp replacement.
The three compared Triton kernels also use explicit best-measured config manifests;
other shared kernels use the same cache/heuristic path. Full native cache building
and all-domain sanitizer coverage remain open. Keep the paths explicitly selectable.
