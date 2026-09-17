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
  channel tiles. Larger K refills the same ring. Separate operand and output
  storage overlaps the next channel's TMA input with the current epilogue, and
  overlaps the current output TMA with the next GEMM. The row mask is loaded once
  per CTA. Two BF16 values are packed into one 32-bit warp shuffle, then the
  required half is selected without FP32 conversions. STSM and TMA write the
  final buffers. Barrier phases account for each stage's use count across channel
  tiles; waits protect output staging before it is reused.
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

When the retired operand rings can hold three output tiles, projection is rounded
to BF16 and staged first. Residual and dropout values are loaded per matrix-copy
chunk, then sigmoid/output math and saved-gate stores run on that chunk. Residual
storage remains read-only; the three output tiles occupy disjoint retired A/B
regions. This reduces live registers without allocating more shared memory or
adding a global intermediate. Smaller rings retain the original epilogue.

At the measured L384 M64/N64/K64/w4/s2 config, this epilogue reduces registers
128→92 while keeping the same shared allocation; measured active-warp occupancy
rises from23.98% to30.53%. Full-range sigmoid, BF16 saved values, dropout and
residual arithmetic remain unchanged.

Projection TMA prefetch also overlaps in-place FP32 sigmoid(BF16(logit)) in the
retired gate accumulator. The chunked epilogue consumes that FP32 gate without
adding a fragment or changing rounding. The N128 config uses163 registers,
down from168 after chunking, with no spills.

When a fourth output-sized tile fits in the retired operand rings, aligned
broadcast dropout scales use a TMA load into that tile and LDSM fragment reads.
The gate's stage-zero barrier is reused at its next phase after all operand
reads retire. P/G/Y and dropout storage remain disjoint, with no larger shared
allocation. Row-wrap, N-tail and insufficient-capacity cases retain the vector
or scalar load path. The initial projection activation tiles are also prefetched
to L2 during gate work using `cute.prefetch`; this hint does not replace the real
TMA loads or completion barriers. Both policies derive from existing dimensions
and stage storage, with no sequence-length dispatch or new configuration axis.

### B9+B10

Rounds and retains the first GEMM's BF16 result before the second GEMM to shorten
FP32 accumulator lifetimes. STSM gathers the final result into retired A storage;
TMA writes it directly to the output. Storage is sized for the larger of the input
ring and output tile. Widths incompatible with TMA store alignment retain a masked
scalar store. Wait-group overlap and K-loop unrolling experiments that regressed
performance were not retained.

When the short gate reduction fits the stage ring, unused slots receive front
GEMM tiles immediately. A gate slot receives its front tile as soon as its last
WGMMA read retires behind a wait and CTA barrier. The front reduction starts from
the corresponding rotated ring position. Longer gate reductions retain the
ordinary refill path. This overlaps the two reductions without extra shared
memory, global intermediates, cache-retention policy, or config axes.

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
Shared-memory exclusions are explicit. The four-warp F2 budget includes separate
aligned A/B rings and output storage; training saves require more space than
inference. Runtime and native build coverage both use the saved-output flag.
Declared and executable domains therefore
differ. Stages allocate real buffers; GROUP_M retains its meaning where the Triton
CSV includes it. F2's original CSV has no GROUP_M axis.

Native keys include tensor extents, strides, dtypes and source identity, including
`front_single_warpgroup.py`. A cache miss uses the first feasible candidate, not a
hardcoded shape winner. Drivers/checkers/candidate enumeration use the native build
path. These kernels compile on the allocated GPU. Measurements below use explicit
candidate manifests and do not imply that the full native cache has been built.

## Latest F567 follow-up

Dropout TMA and initial projection L2 prefetch improve the measured F567 kernel
further, while retaining the same algorithm/configuration contracts above.
Two independent allocations with five rotated graph rounds each measured:

| L | Previous CuTe us | New CuTe us | Strongest measured Triton us | Triton/new |
|---|---:|---:|---:|---:|
|384|95.236|93.231|105.510|1.132x|
|768|362.116|355.706|409.107|approximately1.150x|

L768 individual ratios span1.1496–1.1506x, so it is not robustly above the1.15x
target. The selected native manifest is M64/N64/K64/G1/4warps/2stages at both
lengths; there is no hardcoded length dispatch. The previous G4 manifest was
also validated across three independent allocations. Full native-cache tuning
is still incomplete.

Selected final memcheck/racecheck suites each pass35 cases with zero errors or
hazards. Compiled whole-module output and all gradients pass L384/L768,
including holed masks, nonzero weights/dropout and zero-scale identity checks;
worst relative L2 is1.415e-6. Twelve alternating official training graph rounds
measure1.608384→1.557472ms atL384 (1.033x) and6.281248→6.106904ms atL768
(1.029x). The whole-module target remains unmet. F2 and dual backward source
are unchanged from0f2d455b; rejected warp-specialized/prefetch dual variants
were not promoted.

NCU of the G4 follow-up eliminates dropout LSU global-load sectors by using
TMA, without materially reducing DRAM bytes. L768 also changes N128→N64,
reducing shared66,560→41,984B and registers163→92; occupancy18.50→30.66%
therefore includes both implementation and config effects. L2 throughput is
92.41% and DRAM86.23% of peak. These counters are not timing medians.

## Previous TMA pipeline checkpoint: 0f2d455b

The per-kernel target is at least1.15x against the strongest measured Triton
configuration; the eventual whole-module target is also1.15x. L128 performance
is outside the requested optimization scope.

| Kernel | L384 Triton / CuTe us | Ratio | L768 Triton / CuTe us | Ratio |
|---|---:|---:|---:|---:|
| F2 |220.238 /183.078|1.203x|876.068 /737.801|1.187x|
| F567 |105.586 /95.373|1.107x|410.061 /360.106|1.139x|
| B9+B10 |141.449 /132.717|1.066x|534.529 /510.572|1.047x|

F2 reaches the component target; the other two do not. A previous paired F567
L768 run measured402.596/362.478us, so its established gain is approximately
1.11–1.14x. The final F2 comparison follows72 additional Triton configurations;
the earlier96-config CuTe search used the pipeline before packed shuffle.
These are bounded searches, not a claim of complete native-cache tuning.

The official static-compile/manual-CUDA-graph training module, B1/D128/BF16,
forward+backward without optimizer, dropout0,12 alternating rounds measured:

| L | Triton ms | All three CuTe kernels ms | Ratio |
|---|---:|---:|---:|
|384|1.608288|1.557488|1.033x|
|768|6.285792|6.113424|1.028x|

The whole-module15% target remains unmet. Other common kernels use the same
cache/heuristic paths on both arms. Source and CSV spaces retain the contracts
above; measured winners are explicit benchmark manifests, not installed native
cache entries or hardcoded sequence-length branches.

Validation: F2 has41 kernel checks plus2 native-storage checks; dual backward36;
the promoted F567 checkpoint27. Selected memcheck/racecheck suites contain
36/36/27 cases respectively and report zero errors/hazards. Compiled whole-module
output and every gradient pass atL384/L768 with nonzero weights, holed masks and
fixed nonzero dropout; worst relative L2 is1.44e-6. Zero-scale residual identity
and zero parameter gradients also pass.

Native/registry checks initially passed1145 cases; two documentation-count
failures were corrected and13 related checks passed on rerun. The unrelated
committed-cache freshness failure for existing LayerNorm/Transition entries was
reproduced unchanged in an isolated600c8c4c checkout; it was not suppressed.

## Historical checkpoint: 600c8c4c

Validation and performance before the additional pipeline work:

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
