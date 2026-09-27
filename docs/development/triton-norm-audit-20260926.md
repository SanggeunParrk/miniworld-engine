# Triton normalization audit — H100, 2026-09-26

## Decision

Keep the existing Triton norm implementations as the primary basis. Stop broad
native-CUDA replacement work. D128 standalone forward is already efficient;
backward/config selection and LayerNormLinear wrapper policy retain measurable
opportunities. This is **not** a claim that all kernels reach SOL90.

## Measurement scope

- H100 80GB HBM3 on node02; BF16 activations and FP32 norm affine parameters;
  epsilon explicitly 1e-5. Dense contiguous input, all input/affine gradients.
- `engine_backend="triton"`; LayerNorm uses `layernorm_kernel` with its
  atomic/persistent Triton routing, RMSNorm uses `triton_rmsnorm`.
- Graph timing: ten complete operations per replay, nine event samples, median.
  Linear inference uses `torch.no_grad()`. Training means complete forward+backward.
- No new CUDA kernel or production dispatch change is made by this audit.
- The former report's Engine column used FP32 affine and therefore already took
  Triton for these LayerNorm cells. Its API permits CUDA for other dtype combinations;
  this audit explicitly disables that possibility. The old 0.08284 ms D128 result
  was not a hidden CUDA comparison.
- Autotune misses were observed: the default search tries 24 candidates, not the full
  grid. This audit does not establish that the full committed cache is complete.
- NCU uses warm caches (`--cache-control none`). Small inputs can remain in L2;
  low DRAM utilization is not evidence of poor performance for such inputs.
  Profiling is separate from graph timing and may reselect an equivalent tuning
  config. Exact configs are recorded separately in bench/profile JSON. NCU times
  must not be added to the graph times as if they came from one run.
- BF16 output/all-gradient relative-L2 observations are in each bench JSON;
  this is an analysis of existing code, not a new promotion/validation gate.

## Standalone norms: actual Triton vs native CUDA candidate

All numbers in milliseconds. M is row count (pair M=L squared). CUDA is the new
explicit norm_cuda candidate, not engine 1.0.0 and not a new production default.

| Op | M | D | Triton FWD | CUDA FWD | Triton F+B | CUDA F+B | CUDA speedup over Triton |
|---|---:|---:|---:|---:|---:|---:|---:|
| LN | 8192 | 64 | 0.00257 | 0.00299 | 0.01269 | 0.00811 | 1.57x |
| LN | 147456 | 128 | 0.02896 | 0.02990 | 0.08276 | 0.07949 | 1.04x |
| LN | 589824 | 128 | 0.10433 | 0.11048 | 0.29264 | 0.30152 | 0.97x |
| LN | 147456 | 384 | 0.07963 | 0.08167 | 0.27548 | 0.22991 | 1.20x |
| LN | 8192 | 1024 | 0.00974 | 0.01757 | 0.07208 | 0.06219 | 1.16x |
| RMS | 8192 | 64 | 0.00215 | 0.00288 | 0.00650 | 0.00780 | 0.83x |
| RMS | 147456 | 128 | 0.02760 | 0.02884 | 0.07562 | 0.08337 | 0.91x |
| RMS | 589824 | 128 | 0.10276 | 0.10731 | 0.28203 | 0.29700 | 0.95x |
| RMS | 147456 | 384 | 0.08706 | 0.08117 | 0.25449 | 0.21033 | 1.21x |
| RMS | 8192 | 1024 | 0.00985 | 0.01364 | 0.05297 | 0.05639 | 0.94x |

### What the existing algorithms already do well

- Covering-row forward retains x in registers, computes centered LN variance / RMS
  sum of squares, applies affine, and stores y plus FP32 statistics in one kernel.
  It does not materialize xhat or separate centering/scaling arrays.
- Covering-row backward reads x and dy once, calculates dx and affine gradients
  together. Atomic variants accumulate tiny FP32 parameter vectors; persistent LN
  keeps local accumulators and emits a bounded `[2*SM,D]` partial buffer.
- Tiled RMS rereads x/dy in backward; this can trade register footprint for more
  cache traffic. Covering tiles are already implemented: they do not require a
  new CUDA kernel.
- These pure normalization bodies are reductions and elementwise arithmetic, not
  matrix products. There is no GEMM to replace with WGMMA. A TMA rewrite is not
  justified merely by naming Hopper features; it must beat the existing register
  reuse and direct loads in a complete benchmark.

## NCU observations

Percentages below are hardware counters for individual kernels, not application
SOL or speedup. Small L2-resident cases are intentionally omitted from this table.

| Operation | M/D | Stage | DRAM % | L2 % | Active warps % | Registers/thread |
|---|---|---|---:|---:|---:|---:|
| LN | 147456/128 | fwd | 65.70 | 81.95 | 44.61 | 64 |
| LN | 147456/128 | bwd | 52.82 | 57.86 | 11.91 | 176 |
| LN | 589824/128 | fwd | 85.33 | 82.08 | 34.96 | 72 |
| LN | 589824/128 | bwd | 65.81 | 67.28 | 12.05 | 176 |
| LN | 147456/384 | fwd | 75.37 | 77.04 | 23.76 | 128 |
| LN | 147456/384 | bwd | 50.39 | 52.08 | 6.21 | 255 |
| RMS | 147456/128 | fwd | 66.61 | 82.19 | 66.10 | 40 |
| RMS | 147456/128 | bwd | 55.83 | 61.15 | 29.52 | 96 |
| RMS | 589824/128 | fwd | 85.26 | 80.66 | 70.78 | 40 |
| RMS | 589824/128 | bwd | 64.11 | 66.08 | 30.01 | 96 |
| RMS | 147456/384 | fwd | 78.41 | 79.49 | 78.22 | 28 |
| RMS | 147456/384 | bwd | 71.47 | 74.68 | 29.21 | 96 |

D128 at M589824 reaches about 85% DRAM throughput in both forwards. The measured
native CUDA forwards and complete F+B are slower at this cell. Further blanket
forward development has low priority.

Backward is not universally saturated. LN D384 persistent uses 255 registers per
thread and around 6.2% active warps in this profile, followed by two separate
PyTorch partial-sum kernels. This combination is a concrete tuning/reduction
investigation target, not proof of register spills. The existing atomic path
already improves full LN D384 F+B from 0.27548 to 0.24766 ms without changing a
kernel. New CUDA reaches 0.22991 ms, leaving a targeted opportunity here.

The D64 LN case also retains a 0.00459 ms absolute F+B gap versus CUDA; tiny
workloads are sensitive to zeroing/reduction launches. RMS D64 already wins with
Triton. These cases should not be generalized into a family-wide CUDA win.

## Same RMSNorm Triton code, different tile configuration

Direct-kernel A/B uses identical preallocated buffers and includes forward,
backward, and weight-gradient zeroing in both arms. It is separate from the
public autograd benchmark above. Eleven choices per stage were tested: the
current config plus ten covering-row schedules, not exhaustive autotuning.
No mathematical/fusion algorithm or kernel source changed.

| M | D | Original settings F+B ms | Covering candidates F+B ms | Speedup | Winning FWD BM/BK/warps | Winning BWD BM/BK/warps |
|---:|---:|---:|---:|---:|---|---|
| 147456 | 128 | 0.07587 | 0.07121 | 1.07x | 16/128/4 | 64/128/4 |
| 589824 | 128 | 0.28147 | 0.25787 | 1.09x | 8/128/4 | 64/128/4 |
| 147456 | 384 | 0.25816 | 0.21981 | 1.17x | 8/512/4 | 16/512/4 |
| 8192 | 1024 | 0.05219 | 0.03760 | 1.39x | 4/1024/4 | 8/1024/4 |

Winner PTX and compiler register/spill counts are saved. All eight selected
forward/backward compilations report zero spills. One random-data full
output/dx/dw check per shape passed with max relative L2 below 3e-5.
These are analysis candidates; configs were not installed into the production
cache and have not undergone new broad dtype/graph/sanitizer promotion gates.

## LayerNormLinear

The fused inference kernel keeps the normalized tile in registers and reuses it
across output-column tiles. This is effective for a narrow projection. Wider
outputs increase accumulator/register cost and serialize more output tiles within
one CTA; existing Triton LN + cuBLAS can be faster.

| M | K→N | Fused Triton inference ms | Triton LN + cuBLAS inference ms | New CUDA inference ms |
|---:|---|---:|---:|---:|
| 147456 | 128→16 | 0.02185 | 0.04488 | 0.03096 |
| 8192 | 384→512 | 0.03124 | 0.01386 | 0.01384 |
| 8192 | 64→64 | 0.00328 | 0.00583 | 0.00604 |

### Training wrapper findings

The public portable `layernorm_linear_triton_fn` has a shape-contract issue:
fused forward refuses a flattened 2-D x via `rows_of`, while its backward expects
2-D x and dY. The audit uses a **local shape-only adapter**: identical fused
forward, identical FP32 stats recomputation, identical `_compose_backward`, with
saved x and dY flattened correctly. Engine files were not patched. Its numbers
below describe that adapted experimental path, not a working production wrapper.

It casts the full BF16 x to FP32, separately computes mean and variance, then
adds epsilon and applies rsqrt after the fused forward has already computed the
same statistics. For M147456/K128/N16, inference is 0.02185 ms but this training
forward is 0.20439 ms. NCU confirms the extra cast/mean/variance/add/rsqrt kernels.
Backward recomputes x_normed and materializes dx_normed. This is a wrapper/saving
policy problem; it does not justify rewriting the fast inference math in CUDA.

| M | K→N | Portable adapter F+B ms | Triton LN + cuBLAS F+B ms | Existing TE-style Triton/cuBLAS F+B ms | New CUDA F+B ms |
|---:|---|---:|---:|---:|---:|
| 147456 | 128→16 | 0.34905 | 0.16206 | 0.17049 | 0.15773 |
| 8192 | 384→512 | 0.11938 | 0.08703 | 0.06122 | 0.06950 |
| 8192 | 64→64 | 0.05071 | 0.03636 | 0.02957 | 0.03231 |

TE-style is an existing alternative that saves x_normed and uses layout-aware
backward. Its timing here is exploratory (no fresh full TE numerical/sanitizer
suite); do not infer automatic promotion. H100's public hardware-dispatched
LayerNormLinear inference can use CuTe, and model-specific fused modules have
their own backward. The portable wrapper issue is not evidence that the whole
current MiniWorld training model incurs this overhead.

## Recommended stopping boundary

1. Stop a broad CUDA rewrite of LayerNorm/RMSNorm and narrow LayerNormLinear
   inference: the existing Triton bodies are a good basis and often faster.
2. Prioritize missing shape/dtype tuning-cache coverage and covering-tile
   selection. The measured RMS gains need no new kernel body.
3. Retain targeted work for LN backward at D384/1024 and tiny LN reductions.
4. For LayerNormLinear training, correct shape plumbing and reuse/save statistics
   or select an existing composition path before considering a new CUDA body.
5. Wider/irregular D, FP16/FP32/FP64, strided input, no-affine RMS, and full-model
   timing are outside this fresh BF16 profile set. This audit does not certify
   all possible shapes/dtypes as optimal.

## Evidence

Jobs: 19284_0..2 (benchmark + 13 NCU cases), 19293 (explicit no-grad linear
recheck), 19294 (paired same-buffer RMS tile sensitivity), all exit 0.
19280 was an input-rank harness failure; corrected before measurements.
Raw CSV/JSON/PTX and scripts: `runs/norm_cuda_20260926/triton_audit/`.
No commit, push, production dispatch or live training change was performed.
