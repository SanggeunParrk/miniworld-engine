# SM90 kernels inside the Triton bidirectional TriMul algorithm

`settings.configure(trimul_sm90_kernels={"front", "f567", "dual_bwd"})` selects
individual TMA/WGMMA kernels before model construction/compilation. The default
is an empty set. Explicit PyTorch modules remain references. These overrides
select the existing Triton algorithm, including when the module would otherwise
select the legacy CuTe implementation.

## Preserved contracts

- F2 retains all four projections, raw interleaved BF16 preactivation saves,
  FP32 sigmoid/product, BF16 product rounding before pair masking, and channel-major
  left/right outputs. Left/right are disjoint views of one packed allocation.
- F567 consumes the materialized affine LayerNorm result and the original normalized
  input. It retains independent projection/gate GEMMs, existing BF16 rounding,
  saved projection/gate, dropout scale, and residual. No weight folding or bias
  preparation is added.
- B9+B10 rounds the gate GEMM to BF16 before adding it to the FP32 front GEMM.
  Production column-major dconc views are consumed without transpose copies.
- F4 and all other backward boundaries remain unchanged. Inference keeps its
  existing fused F4567 output kernel; the front replacement also supports inference.

## Reuse decisions

Quack's TMA/WGMMA machinery is reused for F2, with a corrected masking epilogue,
explicit K/stage controls, M128/8-warp mapping, and a single-stage pipeline deadlock
fix. The old F567 TMA plumbing informed the new implementation, but folded-affine
inputs, whole-K staging, and copy preparation were replaced. The projection-aware
legacy output backward is not used because it changes the chosen algorithm.
Existing packed SM90 contractions remain reusable independently; the comparison
here holds the cuBLAS contractions fixed.

## Configuration and cache

`autotune/trimul_sm90_config.py` reads the corresponding Triton CSV directly:

| New native operation | Shared Triton operation | Declared grid size |
|---|---|---:|
| trimul_inproj_gemm_gate_mmajor_sm90_cute | trimul_gemm_gate_mmajor_triton | 864 |
| trimul_output_f567_train_sm90_cute | trimul_output_f567_train_triton | 3072 |
| trimul_input_dual_bwd_sm90_cute | trimul_input_dual_bwd_triton | 1152 |

The declared domains match; the executable domains do not. Rejected configurations
retain explicit reasons. F2 supports M64/M128 with eight actual CTA warps. F567
and dual backward support M64/M128 and four/eight warps independently. Logical
M16/M32 padding and one/two-warp WGMMA emulation are not implemented. Front's
four-warp exclusion is a producer/consumer implementation constraint, not a
universal WGMMA restriction. Shared-memory limits apply separately.

Stages allocate real pipeline slots; K trip counts bound the slots actually used.
GROUP_M controls GEMM tile ordering where present in the original CSV. Native
cache keys include tensor extents/strides/dtypes and implementation identity.
CSV changes invalidate build policy generations. An untuned runtime call uses
the first feasible candidate, not a hardcoded performance winner. Build drivers,
independent reference checkers and native candidate enumeration are registered.
These kernels compile on the allocated compute GPU; the older CPU-only Quack
precompile ABI is deliberately bypassed.

## Validation and measurements (H100 80 GB, BF16)

- 44 integrated GPU regression cases passed, plus the F567 native-selector replay
  regression added after preparing tuning outputs outside the measurement callback.
- Selected-config Compute Sanitizer: front 6, F567 24, dual backward 14 cases;
  all zero errors. This is not a sanitizer sweep of every declared configuration.
- Whole-module output and all input/parameter gradients matched Triton with fixed
  nonzero weights, holed mask and identical dropout scale: L128 eager/static
  fullgraph and L384 static fullgraph. Worst relative L2 was 9.83e-6.
- Zero dropout scale preserved the identity output/gradient and zero parameter gradients.
- 185 registry/naming/launch checks passed. Runtime native candidate reconstruction
  matched the launcher domain for each operation.
- Generated SM90a PTX contains TMA and WGMMA. Representative **Triton kernels
  already contain WGMMA**, but no TMA instructions in the inspected configurations.

| Kernel | L | Triton ms | CuTe ms | Triton/CuTe |
|---|---:|---:|---:|---:|
| F2 | 384 | 0.266530 | 0.283832 | 0.939x |
| F567 | 128 | 0.011343 | 0.015817 | 0.717x |
| F567 | 384 | 0.106290 | 0.151682 | 0.701x |
| F567 | 768 | 0.414277 | 0.542944 | 0.763x |
| B9+B10 | 128 | 0.019580 | 0.023948 | 0.818x |
| B9+B10 | 384 | 0.180554 | 0.182084 | 0.992x |
| B9+B10 | 768 | 0.715165 | 0.652220 | 1.097x |

These are kernel-only graph measurements. F2 compares to a specified fixed
Triton configuration; the other Triton kernels use heuristic candidate tuning.
CuTe timings are selected search results, not full-domain optima. F2 attempted
144 physically supported schedules: 140 correctness/timing passes and four
shared-memory rejections.

The official module harness, B1/L384/D128, static compile plus manual CUDA graph,
12 alternating rounds, FWD+BWD without optimizer, measured 1.648600 ms for Triton
and 1.764840 ms with all three replacements (about 7.1% slower). Individual
replacements measured 1.703424 ms front, 1.702560 ms F567 and 1.653832 ms dual
backward. Dropout is zero for this graph benchmark because the harness rejects
nonzero training dropout with graphs; nonzero dropout correctness is tested
separately. The measured SM90 configurations were explicitly supplied instead
of presenting an incomplete native cache as fully tuned.

**Keep the new implementations opt-in.** L768's dual-backward component gain does
not establish an end-to-end training gain. Full shape/cache tuning remains open.
