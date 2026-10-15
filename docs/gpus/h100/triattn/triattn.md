# TriangleAttention on H100 (sm_90a)

Kernel-level status of `TriangleAttention` (`use_self_attention=True`) on H100; the module-level summary belongs in
[h100.md](../h100.md). bf16 only; columns are (Length, Dimension). H100 = CUDA where the hand-written sm_90a path serves the
shape; a shape without it is 미구현 (it runs the module's existing path: Triton core + the prebuilt CUDA pieces).

Dispatch: `integrations/triattn_h100_sm90.py`, called in `TriangleAttention.forward` after the B200 checks and ahead of the b200style path of `integrations/triattn_h100.py` (`module._h100_cuda = False` falls back to it). Two paths share the attention
core:

- **fused** (d_pair = d_hidden = 128, 4 heads × 32): `kernels/triangle_attention/cuda/h100_sm90.py` (build + autograd) and
  `h100_sources/` (raw-PTX wgmma / TMA / mbarrier, no CUTLASS or CuTe: `sm90.cuh` = helpers, `triattn_sm90.cu` = attention forward,
  `triattn_bwd_sm90.cu` = attention backward, `triattn_mod_sm90.cu` = the kernels around it);
- **width-generic** (every other served shape): `h100_sm90_generic.py` and `triattn_generic_sm90.cu` — the same attention core
  and cuBLAS projections with hand-written LayerNorm, gate, residual and LayerNorm-backward kernels between them
  ([Other widths](#other-widths--d_pair-64-512-width-generic-path)).

The kernel boundaries are the B200 path's ([b200/triattn/triattn.md](../../b200/triattn/triattn.md)); the kernels are sm_90a designs,
not ports of the tcgen05 ones.

The path is served when all of these hold, otherwise the module runs its existing path unchanged:

- the module's backend is TRITON (`implementation=miniworld`, or an explicit `triton`), `module._h100_cuda` is True (the
  default; False keeps the existing path, e.g. as a benchmark baseline) and `settings.engine_backend != "triton"`;
- `use_self_attention`, no QK-norm; heads of 32 channels (16 is zero-padded to 32), at most 16 heads, d_pair a multiple of 8 up
  to 512 (`d_hidden = heads × head_dim`); `d_pair = d_hidden = 128` with 4 heads takes the fused path;
- `pair` is a bf16 CUDA tensor `[B, L, L, d_pair]` with L a multiple of 16 (any B), `mask` is None or bool `[B, L]`; a length that
  is not a multiple of 128 is zero-padded to one (pair, dropout scale; the pad keys are masked, the pad rows and columns are cut off
  the result), so L = 144 costs an L = 256 call;
- compute capability (9, 0).

Starting and ending node both run it (the ending node transposes in and out). Without grad and without training dropout the
forward kernels run on bf16 weight packs cached on the module (rebuilt when a parameter's data pointer or version changes; scoped to
a CUDA-graph capture by `kernels._capture`); otherwise the autograd function runs, with the module's own broadcast dropout draw
applied inside the tail and its backward.
Parameters may be bf16 or fp32 (master weights): gradients come back in the parameters' dtype. Masked keys get -inf in the bias.
The kernel launches are `opaque` custom ops (`kernels/_compile.py`: `triattn_h100_infer`, `_fwd`, `_bwd`; one autograd function calls one
op forward and one backward), so the module compiles to one graph (`tests/compile/test_compile_regimes_gpu.py`), the weight packs
(cached on the parameters' identity and version, scoped to a CUDA-graph capture by `kernels._capture`) and the launches run eagerly
inside the ops, and the benchmark harness' captured module works (traced in Dynamo, the packs were recompiled inside the capture).
The ops return fresh tensors only (an output of a custom op may not alias an input or another output): the backward's parameter
gradients are cloned out of their shared buffers.

## Starting / ending node · d_pair 128, 4 heads × 32

### Inference

F1 · front → F2 · attention → F3 · tail.

- **F1 `tri_front_sm90`** — 128-row tiles; x is ldmatrix'd straight into the wgmma A-fragment layout and normalised in
  registers (a row's channels sit on a quad: the statistics are two quad sums), so y never touches shared memory; q | k | v | g are
  four m64n128 RS chains over the resident [Wq; Wk; Wv; Wg] (128 KiB), the bias one m64n16 chain over Wb (zero-padded to 16 rows),
  written head-major as bf16(bias / scale), the attention core's operand. Outputs are staged with stmatrix over the tile's own x stage
  and stored by TMA.
- **F2 `triattn_fwd_sm90`** — task = (b, h, 128-query tile) × 2 pair rows, persistent grid ordered query tile, head, row group
  (the CTAs running together share whole K / V rows in L2); a TMA producer warpgroup and two consumer warpgroups (64 queries each).
  The score accumulator starts from the bias (ldmatrix of the bf16 bias / scale tile: no per-logit bias add — the technique of
  Anthropic's opt_core `triattn_m1`), S += Q K^T is an RS wgmma with Q in registers for the whole task, the softmax is max-free
  (the offset is the maximum of the row's first 32 keys, fixed for the task; the B200 kernel's and opt_core's), and one m64n40 RS
  wgmma per 16 keys computes [O | l] += P [V | 1] (the row sums on the tensor core, from the same bf16 P). The sub-steps run in
  FlashAttention-3 order (QK^T of this sub-step, PV of the previous one, the exponentials under the PV) with nothing in flight at a
  loop back edge.
- **F3 `tri_tail_sm90`** — g, o and x are ldmatrix'd in the A-fragment layout, which is also the accumulator layout of the same
  rows and columns: u = sigmoid(g) o o is the A operand of the m64n128 chain over Wo, and drop o acc + x is added in place. A CTA
  keeps one 128-wide j block, so the dropout scale (which depends on (b, j) only) is loaded into registers once.

| (Length, Dimension) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Training

Forward F1, F2 (also writing the base-2 LSE), F3 with the dropout scale; backward:

- **B1 `tri_gate_bwd_sm90`** — dyd = dy o drop, du = dyd Wo (RS wgmma, Wo MN-major), do, dg, the core's Delta = Σ do o per head
  (quad sums), dWo += dyd^T (s o o) (SS wgmma with both operands MN-major from the staged tile, register accumulator for the CTA's
  tiles), and the residual seed of dpair = dy (TMA store straight from the stage).
- **B2 `triattn_bwd_kv_sm90`** — dK, dV: task = (b, h, 128-key tile) × 2 rows, 64-query steps; S^T starts from bias^T (the same
  bias tensor through ldmatrix.trans), P^T recomputed from the LSE, dV += P^T dO and dK += dS^T Q as RS wgmma with the operands
  MN-major from the same shared tiles. The two consumer warpgroups take turns issuing their GEMMs (ping-pong).

- **B3 `triattn_bwd_q_sm90`** — dQ, dbias: task = (b, h, 128-query tile) × 4 rows (the rows innermost), 64-key steps; the step's
  64 × 64 dbias partial is summed over the four rows in registers and added by one fp32 TMA reduce-add into an L2-resident
  `[B, 4, L, L]` buffer (L / 4 partials per element in all). The consumers only stage the partial and arrive on an mbarrier; an
  otherwise idle warp of the producer warpgroup issues the reduce-adds and hands the staging buffer back.
- **B4 `tri_head_bwd_sm90`** — gy = Σ_t d_t W_t (SS wgmma over [128 rows][64] half-tiles of the d_t, W4 MN-major) + the bias
  projection's share in fp32 on the CUDA cores, LN backward from recomputed statistics, dx added into the dpair seed by a bf16 TMA
  reduce-add.
- **B5 `tri_wgrad_sm90` + `tri_param_finish`** — M_t = xhat^T d_t, cs_t = d_t^T 1, B_h = xhat^T db and cdb on the tensor core
  (n8 chains for the vectors), then the B200 path's parameter algebra. H100 has no tensor memory: the four 128 × 128 fp32
  accumulators do not fit one CTA, so a CTA owns a pair of the projections (x is read twice).

| (Length, Dimension) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

"cache build ✓" means only that the path needs no autotune cache (JIT-built CUDA).

## Other widths · d_pair 64-512 (width-generic path)

Registered shapes served (`registry_module.csv`, `triangle_attention_checkpoint` / `_heads`):

| d_pair | d_hidden | heads | model |
|---|---|---|---|
| 64 | 64 | 4 × 16 | AF3 template stack (16-channel heads are zero-padded to 32) |
| 64 | 128 | 4 × 32 | Protenix-v1 template stack |
| 64 | 64 | 2 × 32 | Protenix-v2 / OpenDDE template stack |
| 128 | 128 | 4 × 32 | MiniWorld (the fused path above) |
| 256 | 256 | 8 × 32 | Protenix-v2 / ESMFold2 trunk |
| 384 | 384 | 12 × 32 | OpenDDE trunk |
| 512 | 512 | 16 × 32 | none (benchmark width) |

The fused path keeps LayerNorm and the projections in registers because its weights (128 KiB) stay resident in shared memory; at
d_pair 256-512 the weights are 0.5-2 MiB and do not. The generic path therefore keeps the attention core (heads of 32 channels at any
head count: its head count is a runtime argument) and puts cuBLAS projections between hand-written kernels. Kernels in order
(`triattn_generic_sm90.cu` unless noted); M = B L² rows:

Inference: G1 `ln_fwd` (y = LN(x) and the statistics; C / 8 lanes per row, so a warp takes 4 / 2 rows at C = 64 / 128) → cuBLAS
y [Wq; Wk; Wv; Wg]^T and y Wb^T (two GEMMs, so the 4 H × 32 columns tile cleanly) → G2 `front_bias` (head-major bf16(bias / scale),
masked keys -inf) → **attention core** (`triattn_sm90.cu`, with the gate fused into its epilogue: it writes u = sigmoid(g) ∘ O and not O;
the gate values are prefetched under the last key tile) → cuBLAS u Wo^T → G3 `residual` (x + drop ∘ t).

Training adds the base-2 LSE and O to the forward, and:

- B1 `gate_bwd_in` (dy ∘ drop) → cuBLAS du = ddrop Wo and dWo = ddrop^T u → B2 `gate_bwd_out` (do, dg, and the core's Delta per head from
  the quad of lanes that owns a head);
- B3 / B4 `triattn_bwd_kv` / `triattn_bwd_q` (`triattn_bwd_sm90.cu`, unchanged math) write dk, dv, dq straight into their columns of one
  `dall` buffer ([M, 4 HD + Hp] = dq | dk | dv | dg | dbias, the packed weights' row order); dbias (fp32) is converted into its
  columns by B5 `assemble`;
- cuBLAS dyln = dall Wcat and dW = dall^T y (one GEMM each, K = M for the weights) → B6 `ln_bwd` (dx = dy + LN backward, the
  gamma / beta partial sums per CTA, finished by a torch sum).

Packed weights (`pack_weights`): Wcat = [Wq; Wk; Wv; Wg; Wb (Hp rows)] bf16 and Wo; a 16-channel head is padded with zeros to 32
(the padded q / k / v / g channels are 0 and meet zero columns of Wo; the softmax scale stays 1/√16) and its gradients are cut
back. The LN backward reads `dyln` in bf16 (the reference module's own precision at that point).

The tables mark the shapes this path serves; "cache build ✓" means only that the path needs no autotune cache (JIT-built CUDA).

### Inference

#### d_pair 64 · heads 4 × 16, 4 × 32, 2 × 32

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### d_pair 256 (8 × 32) · 384 (12 × 32) · 512 (16 × 32)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Training

With the module's dropout; bf16 or fp32 parameters. The same shapes, the same tables:

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Measurements, all widths (2026-10-05)

H100 80GB HBM3 (132 SMs), torch 2.13.0+cu129, triton 3.7.1, cuEquivariance 0.12.0, B=1, bf16, starting node, no mask, dropout 0.25 in
training. PyTorch compiled and cuEquivariance: `benchmarks/runners/bench.py target=triangle_attention level=module min_seq_len=128
max_seq_len=768 implementations=[pytorch,cuequivariance]` with `d_pair=<C> +tri_d_hidden=<HD> +tri_n_head=<heads>` (compiled, manual CUDA
graph; "—" = out of memory: PyTorch training at (768, 512)). "Triton path" = the module with `_h100_cuda = False` (the repository's
previous H100 path: Triton core + prebuilt CUDA pieces) and "ours", both timed by one module call (forward, or forward + autograd.grad of
the input and every parameter) replayed from a CUDA graph (median of 5 × 10 replays); the harness gives the same numbers where
spot-checked (`+tri_h100_cuda=true|false`, L384, ms: d_pair 64 / 4 × 32 inference ours 0.298 / Triton path 0.318, training 1.086 / 1.664;
d_pair 256 inference 0.664 / 0.981, training 2.262 / 3.960). Latency in ms, median. × = ours vs the faster of PyTorch compiled /
cuEquivariance; the last column is ours vs the Triton path.
Anthropic's payload is not in this table (its native cell serves 4-head d_pair 128 inference only; see the 2026-09-30 table below).

The d_pair 128 rows are the fused path, all others the width-generic path. The generic path is bound by its GEMMs and HBM passes
at d_pair 64 and by the cuBLAS projections (about 2.7 ms of the 7.7 ms at (768, 512)) above; the attention core is the same kernel
as the fused path's. Where the Triton path is close (L384 and L768 at d_pair 64: 0.98-1.18×) the Triton kernels happen to sit at
a good tile for that shape; the 4 × 16 layout pays for the zero-padded 32-channel heads.

### Inference (CUDA graph)

#### Heads × 32 (d_pair 64: 2 × 32, 128: 4, 256: 8, 384: 12, 512: 16)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Triton path | ours | × | × vs Triton path |
|---|---|---|---|---|---|---|
| (128, 64) | 0.062 | 0.051 | 0.069 | 0.031 | 1.65 | 2.21× |
| (256, 64) | 0.238 | 0.143 | 0.214 | 0.084 | 1.69 | 2.54× |
| (384, 64) | 0.816 | 0.350 | 0.243 | 0.187 | 1.88 | 1.30× |
| (512, 64) | 1.495 | 0.711 | 0.486 | 0.337 | 2.11 | 1.44× |
| (640, 64) | 3.790 | 1.249 | 0.829 | 0.553 | 2.26 | 1.50× |
| (768, 64) | 6.560 | 1.989 | 1.291 | 0.834 | 2.38 | 1.55× |
| (128, 128) | 0.099 | 0.073 | 0.092 | 0.034 | 2.17 | 2.74× |
| (256, 128) | 0.475 | 0.261 | 0.330 | 0.114 | 2.28 | 2.89× |
| (384, 128) | 1.621 | 0.708 | 0.394 | 0.268 | 2.65 | 1.47× |
| (512, 128) | 3.251 | 1.417 | 0.949 | 0.507 | 2.80 | 1.87× |
| (640, 128) | 8.313 | 2.457 | 1.616 | 0.872 | 2.82 | 1.85× |
| (768, 128) | 13.098 | 3.973 | 1.927 | 1.357 | 2.93 | 1.42× |
| (128, 256) | 0.185 | 0.121 | 0.132 | 0.084 | 1.44 | 1.57× |
| (256, 256) | 1.165 | 0.539 | 0.596 | 0.287 | 1.88 | 2.08× |
| (384, 256) | 3.613 | 1.382 | 0.978 | 0.663 | 2.08 | 1.47× |
| (512, 256) | 7.812 | 2.756 | 1.909 | 1.254 | 2.20 | 1.52× |
| (640, 256) | 16.617 | 4.845 | 3.276 | 2.091 | 2.32 | 1.57× |
| (768, 256) | 26.038 | 7.788 | 5.179 | 3.265 | 2.38 | 1.59× |
| (128, 384) | 0.268 | 0.186 | 0.204 | 0.126 | 1.47 | 1.62× |
| (256, 384) | 2.012 | 0.813 | 0.864 | 0.450 | 1.81 | 1.92× |
| (384, 384) | 5.937 | 2.079 | 1.465 | 1.047 | 1.99 | 1.40× |
| (512, 384) | 16.919 | 4.092 | 2.995 | 1.969 | 2.08 | 1.52× |
| (640, 384) | 31.749 | 7.306 | 5.144 | 3.434 | 2.13 | 1.50× |
| (768, 384) | 55.265 | 11.916 | 8.218 | 5.301 | 2.25 | 1.55× |
| (128, 512) | 0.452 | 0.239 | 0.265 | 0.172 | 1.39 | 1.54× |
| (256, 512) | 3.141 | 1.078 | 1.117 | 0.623 | 1.73 | 1.79× |
| (384, 512) | 9.672 | 2.791 | 2.024 | 1.480 | 1.89 | 1.37× |
| (512, 512) | 22.699 | 5.558 | 4.114 | 2.862 | 1.94 | 1.44× |
| (640, 512) | 42.557 | 10.048 | 7.083 | 4.863 | 2.07 | 1.46× |
| (768, 512) | 73.912 | 18.310 | 11.316 | 7.651 | 2.39 | 1.48× |

![Inference (CUDA graph), length sweep at D128](figures/triattn_inference_cuda_graph_length.png) ![Inference (CUDA graph), dimension sweep at L384](figures/triattn_inference_cuda_graph_dimension.png) <!-- measure_bars -->

#### d_pair 64 · 4 × 32 (d_hidden 128)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Triton path | ours | × | × vs Triton path |
|---|---|---|---|---|---|---|
| (128, 64) | 0.097 | 0.067 | 0.084 | 0.041 | 1.63 | 2.06× |
| (256, 64) | 0.444 | 0.235 | 0.274 | 0.126 | 1.86 | 2.17× |
| (384, 64) | 1.550 | 0.628 | 0.314 | 0.299 | 2.10 | 1.05× |
| (512, 64) | 3.100 | 1.271 | 0.811 | 0.551 | 2.31 | 1.47× |
| (640, 64) | 7.409 | 2.260 | 1.417 | 0.929 | 2.43 | 1.53× |
| (768, 64) | 12.756 | 3.669 | 1.687 | 1.434 | 2.56 | 1.18× |

#### d_pair 64 · 4 × 16 (d_hidden 64)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Triton path | ours | × | × vs Triton path |
|---|---|---|---|---|---|---|
| (128, 64) | 0.090 | 0.089 | 0.074 | 0.057 | 1.57 | 1.31× |
| (256, 64) | 0.397 | 0.201 | 0.231 | 0.146 | 1.38 | 1.58× |
| (384, 64) | 1.433 | 0.540 | 0.311 | 0.318 | 1.70 | 0.98× |
| (512, 64) | 2.931 | 1.145 | 0.650 | 0.568 | 2.02 | 1.14× |
| (640, 64) | 7.100 | 2.095 | 1.166 | 0.955 | 2.19 | 1.22× |
| (768, 64) | 12.281 | 3.445 | 1.872 | 1.447 | 2.38 | 1.29× |

### Training, dropout 0.25 (CUDA graph)

#### Heads × 32 (d_pair 64: 2 × 32, 128: 4, 256: 8, 384: 12, 512: 16)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Triton path | ours | × | × vs Triton path |
|---|---|---|---|---|---|---|
| (128, 64) | 0.191 | 0.198 | 0.272 | 0.133 | 1.44 | 2.05× |
| (256, 64) | 0.698 | 0.646 | 0.896 | 0.319 | 2.03 | 2.81× |
| (384, 64) | 2.009 | 1.660 | 1.094 | 0.662 | 2.51 | 1.65× |
| (512, 64) | 3.933 | 3.416 | 2.144 | 1.183 | 2.89 | 1.81× |
| (640, 64) | 8.197 | 6.060 | 3.633 | 1.944 | 3.12 | 1.87× |
| (768, 64) | 13.673 | 9.814 | 5.668 | 2.934 | 3.34 | 1.93× |
| (128, 128) | 0.296 | 0.291 | 0.356 | 0.150 | 1.94 | 2.37× |
| (256, 128) | 1.309 | 1.194 | 1.348 | 0.425 | 2.81 | 3.17× |
| (384, 128) | 3.894 | 3.219 | 1.355 | 0.955 | 3.37 | 1.42× |
| (512, 128) | 7.991 | 6.626 | 4.188 | 1.814 | 3.65 | 2.31× |
| (640, 128) | 16.855 | 11.827 | 7.142 | 3.092 | 3.82 | 2.31× |
| (768, 128) | 27.115 | 19.391 | 6.992 | 4.880 | 3.97 | 1.43× |
| (128, 256) | 0.520 | 0.505 | 0.558 | 0.299 | 1.69 | 1.87× |
| (256, 256) | 2.795 | 2.374 | 2.331 | 0.958 | 2.48 | 2.43× |
| (384, 256) | 8.139 | 6.391 | 4.199 | 2.234 | 2.86 | 1.88× |
| (512, 256) | 17.093 | 13.155 | 8.271 | 4.267 | 3.08 | 1.94× |
| (640, 256) | 33.526 | 23.870 | 14.328 | 7.430 | 3.21 | 1.93× |
| (768, 256) | 53.775 | 40.181 | 22.788 | 11.966 | 3.36 | 1.90× |
| (128, 384) | 0.800 | 0.785 | 0.807 | 0.436 | 1.80 | 1.85× |
| (256, 384) | 4.410 | 3.509 | 3.300 | 1.425 | 2.46 | 2.32× |
| (384, 384) | 12.666 | 9.522 | 6.458 | 3.463 | 2.75 | 1.86× |
| (512, 384) | 31.114 | 19.839 | 12.815 | 6.769 | 2.93 | 1.89× |
| (640, 384) | 57.713 | 37.594 | 22.198 | 12.047 | 3.12 | 1.84× |
| (768, 384) | 97.936 | 64.748 | 35.381 | 19.623 | 3.30 | 1.80× |
| (128, 512) | 1.113 | 0.996 | 1.020 | 0.572 | 1.74 | 1.78× |
| (256, 512) | 6.309 | 4.648 | 4.247 | 1.944 | 2.39 | 2.19× |
| (384, 512) | 18.606 | 12.590 | 8.646 | 4.839 | 2.60 | 1.79× |
| (512, 512) | 41.645 | 27.032 | 17.337 | 9.544 | 2.83 | 1.82× |
| (640, 512) | 77.060 | 52.430 | 30.224 | 17.274 | 3.04 | 1.75× |
| (768, 512) | 130.947 | 88.535 | 48.087 | 27.497 | 3.22 | 1.75× |

![Training, dropout 0.25 (CUDA graph), length sweep at D128](figures/triattn_training_dropout_0_25_cuda_graph_length.png) ![Training, dropout 0.25 (CUDA graph), dimension sweep at L384](figures/triattn_training_dropout_0_25_cuda_graph_dimension.png) <!-- measure_bars -->

#### d_pair 64 · 4 × 32 (d_hidden 128)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Triton path | ours | × | × vs Triton path |
|---|---|---|---|---|---|---|
| (128, 64) | 0.274 | 0.267 | 0.317 | 0.160 | 1.66 | 1.98× |
| (256, 64) | 1.194 | 1.076 | 1.141 | 0.471 | 2.29 | 2.42× |
| (384, 64) | 3.654 | 2.956 | 1.369 | 1.044 | 2.83 | 1.31× |
| (512, 64) | 7.529 | 6.126 | 3.578 | 1.956 | 3.13 | 1.83× |
| (640, 64) | 15.517 | 11.115 | 6.268 | 3.302 | 3.37 | 1.90× |
| (768, 64) | 26.073 | 18.331 | 6.964 | 5.141 | 3.57 | 1.35× |

#### d_pair 64 · 4 × 16 (d_hidden 64)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Triton path | ours | × | × vs Triton path |
|---|---|---|---|---|---|---|
| (128, 64) | 0.251 | 0.251 | 0.290 | 0.191 | 1.31 | 1.52× |
| (256, 64) | 1.073 | 0.928 | 1.014 | 0.500 | 1.86 | 2.03× |
| (384, 64) | 3.348 | 2.597 | 1.522 | 1.060 | 2.45 | 1.44× |
| (512, 64) | 7.079 | 5.593 | 3.095 | 1.960 | 2.85 | 1.58× |
| (640, 64) | 14.711 | 10.260 | 5.451 | 3.344 | 3.07 | 1.63× |
| (768, 64) | 25.007 | 17.081 | 8.780 | 5.128 | 3.33 | 1.71× |

## Measurements, d_pair 128 (2026-09-30)

H100 80GB HBM3 (132 SMs), torch 2.13.0+cu129, B=1, bf16, starting node, mask all-true,
`benchmarks/runners/bench.py target=triangle_attention level=module min_seq_len=128 max_seq_len=768` (compiled, CUDA graph).
Latency in ms, median. "existing" = the module's previous H100 path (`_h100_cuda = False`: Triton core + prebuilt CUDA pieces);
× = ours vs the fastest of the others. Anthropic: `integrations.anthropic` row `block:triattn_native` (its H100 native cell), inference only.

### Inference (CUDA graph)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | existing | ours | × |
|---|---|---|---|---|---|---|
| (128, 128) | 0.100 | 0.073 | 0.052 | 0.056 | 0.032 | 1.63 |
| (256, 128) | 0.475 | 0.259 | 0.184 | 0.183 | 0.108 | 1.69 |
| (384, 128) | 1.631 | 0.711 | 0.445 | 0.393 | 0.253 | 1.55 |
| (512, 128) | 3.232 | 1.417 | 0.863 | 0.950 | 0.490 | 1.76 |
| (640, 128) | 8.305 | 2.439 | 1.467 | 1.593 | 0.837 | 1.75 |
| (768, 128) | 12.948 | 3.950 | 2.326 | 1.923 | 1.298 | 1.48 |

### Training, dropout 0.25 (CUDA graph)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | existing | ours | × |
|---|---|---|---|---|---|---|
| (128, 128) | 0.296 | 0.290 | — | 0.213 | 0.142 | 1.50 |
| (256, 128) | 1.304 | 1.194 | — | 0.789 | 0.418 | 1.89 |
| (384, 128) | 3.888 | 3.197 | — | 1.284 | 0.959 | 1.34 |
| (512, 128) | 7.985 | 6.604 | — | 3.944 | 1.798 | 2.19 |
| (640, 128) | 16.817 | 11.902 | — | 6.751 | 3.043 | 2.22 |
| (768, 128) | 27.103 | 19.496 | — | 6.778 | 4.821 | 1.41 |

### Training, dropout 0.25 (no CUDA graph)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | ours | × |
|---|---|---|---|---|
| (128, 128) | 0.671 | 0.938 | 0.537 | 1.25 |
| (256, 128) | 1.384 | 1.288 | 0.540 | 2.39 |
| (384, 128) | 3.941 | 3.316 | 0.989 | 3.35 |
| (512, 128) | 8.072 | 6.728 | 1.833 | 3.67 |
| (640, 128) | 16.948 | 11.927 | 3.089 | 3.86 |
| (768, 128) | 27.382 | 19.601 | 4.849 | 4.04 |

Without a graph, ours costs ~0.5 ms at L128 / L256 regardless of L: host-side time (tensor-map encoding, the autograd function,
a dozen launches) above the GPU work, as on B200.

Run-to-run determinism: the parameter gradients are reduced from per-CTA partial slots in a fixed order (no float atomics); the only
order-dependent sum left is dbias (fp32 TMA reduce-adds of the query-side kernel), which reaches the parameters at ≤ 1.3e-6
relative (fp32 master parameters; the harness checks 1e-4). The bias projection's B_h reads dbias as a bf16 hi + lo pair so that
order noise cannot flip a bf16 rounding.

## Kernel times (2026-09-30)

One training step (fwd + bwd, dropout 0.25), `torch.profiler` device time, µs:

| kernel | (384, 128) | (768, 128) |
|---|---|---|
| F1 · tri_front | 73.0 | 284.3 |
| F2 · triattn_fwd (+ LSE) | 126.7 | 814.0 |
| F3 · tri_tail | 54.0 | 198.9 |
| B1 · tri_gate_bwd | 96.6 | 338.0 |
| B2 · triattn_bwd_kv | 188.7 | 1145.9 |
| B3 · triattn_bwd_q | 190.9 | 1181.0 |
| B4 · tri_head_bwd | 98.7 | 361.8 |
| B5 · tri_wgrad + finish | 106.9 | 346.5 |
| step | ~938 | ~4673 |

The attention core (F2, B2, B3) is 54 % of the step at L384 and 67 % at L768. B2 runs at ~404 TFLOP/s at L768 (41 % of the
bf16 dense peak, FlashAttention-3 hdim-64 backward class); D = 32 attention is bound by the exponentials and the per-step
issue overhead rather than the tensor core. The module kernels run at 2.2–2.8 TB/s of HBM traffic.

Accuracy (`tests/integrations/test_triattn_h100_gpu.py`, 24 cases: L128–768 inference and training, ending node, B=2, mask
layouts, dropout on both nodes, fp32 master parameters, CUDA-graph capture): against an fp32 autograd reference, the output and
every gradient stay within 1.25× the bf16 PyTorch module's own error (+2e-3). The attention core alone (fp32 reference, same bf16
operands): out 2.2e-3, dK 2.4e-3, dV 2.4e-3, dQ 2.5e-3 relative Frobenius (bf16 autograd: 7–8e-3), dbias 5e-4.

## Kernel times, width-generic path (2026-10-05)

`torch.profiler` device time per module call, µs (mean of 5 calls after warm-up, eager, B=1, no mask). The cuBLAS row sums the GEMM launches
(forward: y [Wq; Wk; Wv; Wg]^T, y Wb^T, u Wo^T; training adds du = ddrop Wo, dWo, dyln = dall Wcat, dW = dall^T y with its split-K reduce).

| kernel | inference (512, 256) | training (512, 256) | training (384, 64 · 4 × 32) |
|---|---|---|---|
| G1 · ln_fwd | 104 | 103 | 14 |
| G2 · front_bias | 7 | 6 | 4 |
| F2 · triattn_fwd (+ gate; LSE and O in training) | 595 | 605 | 155 |
| G3 · residual | 138 | 140 | 19 |
| B1 · gate_bwd_in | | 94 | 15 |
| B2 · gate_bwd_out (do, dg, Delta) | | 275 | 77 |
| B3 · triattn_bwd_kv | | 751 | 179 |
| B4 · triattn_bwd_q | | 790 | 203 |
| B5 · assemble (dbias columns) | | 21 | 11 |
| B6 · ln_bwd (+ gamma / beta partials) | | 232 | 32 |
| cuBLAS GEMMs | 397 | 1110 | 288 |
| torch glue (memset, casts, sums) | | 38 | 32 |
| call | ~1245 | ~4170 | ~1030 |

The attention core is 48 % of the inference call at (512, 256) and 51 % of the training call at (384, 64); the GEMMs run at
550-800 TFLOP/s. What the fused path avoids and this one pays: one write and one read of y (G1), the t tensor between the out-projection
and the residual (G3) and the q | k | v | g round trip through HBM before the core. An LN + projection kernel with the weights
streamed in K (they do not fit shared memory above d_pair 128) is the next step for d_pair 64 / 256-512.

Accuracy (`tests/integrations/test_triattn_h100_generic_gpu.py`, 58 cases): inference and training at the eight (d_pair, d_hidden,
heads) shapes of the table, padded lengths (80, 192, 304 and an unmasked 208), starting and ending node, B=2, dropout on both nodes,
fp32 master parameters, determinism, CUDA-graph capture with and without grad, the compiled module replayed from a CUDA graph. Against an fp32 autograd reference the output and every
gradient stay within 1.25× the bf16 PyTorch module's own error (+2e-3). Run-to-run: the output and the q / k / v / gate / out weight
gradients are bitwise equal; the x, LayerNorm and bias-projection gradients carry the order noise of the core's fp32 dbias reduce-add
(a bf16 ulp at most; the fused path's hi + lo trick for the bias projection is not applied here).

## Development notes

- Measured dead ends (forward core): fp32 bias in fragment order via LDS.128 (shared-memory bandwidth), bias fragments held
  across rows (spills), 3–4 pair rows per task (register pressure), exponentials split onto the FMA pipe (slower: the exp phase is
  issue-bound, not MUFU-bound — as on B200), consumer ping-pong (neutral once the sub-steps are pipelined).
- A wgmma left in flight across a loop back edge makes ptxas serialize every wgmma of the kernel (C7514); every pipelined loop
  here drains at its back edge.
- Stages read with ldmatrix / ld.shared and then released to the TMA producer need `fence.proxy.async` before the release
  (without it the forward was non-deterministic run to run). Every kernel was checked bitwise run to run.
- The head order of the tasks matters: with the head slowest, K / V were re-read from DRAM 2.5×.
- A captured CUDA graph replays the weight packs with every call (so a replay sees the current parameters): their kernel count is
  latency.  The generic path's twelve torch cast / pad / cat kernels were 17 of 74 us at (128, 256) inference with fp32 master
  parameters; `pack_generic_k` and `tri_pack_k` do it in one launch (`7f7851aa`).
- Measured on the backward core (clock-locked `sm__cycles_elapsed`, the only A/B that holds on a shared node): consumer ping-pong
  −8 % (L384) / −2 % (L768) on the key side, +11–13 % on the query side (so: key side only); the dbias reduce-adds on a storer warp
  −1 %; NOT kept: wgmma descriptors built once and moved by offsets (+7 %: ptxas' own uniform-datapath descriptors are
  cheaper), bf16 packing on the integer pipes instead of `cvt.rn.bf16x2` (+9–17 %). The dbias flush costs 12–15 % of the query side,
  mostly as L2 traffic (L / 4 fp32 partials per element), not synchronisation.

## Known limits / next

- The attention core is ~60 % of its MUFU / issue bound; the backward recomputes P twice (as on B200). Measured since: removing the
  `scale * S - lse` FFMA of every score changes neither backward kernel (the sub-step's dependency chain is the limit, not the
  instruction count); the single-pass backward (dQ in shared memory) needs the dbias partials summed over several rows first
  (7.2 GB of reduce-add with one row per CTA instead of 1.8 GB) and two rows' fp32 dQ do not fit; clustering the dWo partials, fewer
  wgrad splits and a chunked weight load of the front were neutral or slower.  All in
  `docs/records/triattn_h100_core_experiments_20261013.md` (rounds 4-8).
- Heads of 64 / 96 / 128 channels (the module's default of 4 heads at d_pair >= 256), QK-norm, the bias-only variant and
  `BidirectionalTriangleAttention` (not served on B200 either) run the Triton path. L must be a multiple of 16; any other length runs
  the Triton path, and a length that is not a multiple of 128 is computed at the next multiple of 128 (L = 144 costs an L = 256 call).
- The width-generic path materialises y and q | k | v | g and runs the projections on cuBLAS; a fused LN + projection kernel
  (weights streamed in K) and a fused gate + out-projection + residual kernel would remove the G1 and G3 passes (19 % of the
  inference call at (512, 256), 242 of 1245 µs) and the HBM round trip of y and t.

## Current measurements (2026-10-15, `c9583b43`)

One H100 80GB HBM3, bf16-mixed, CUDA graph, one run of the grid; ms.  Raw logs and the earlier dated tables (10-13 / 10-14 / 10-15, with
the harness caveats and the weight-pack correction): [docs/records/triattn_h100_remeasure_20261013_15.md](../../../records/triattn_h100_remeasure_20261013_15.md).
Core experiments: [docs/records/triattn_h100_core_experiments_20261013.md](../../../records/triattn_h100_core_experiments_20261013.md).

### Against PyTorch compiled and cuEquivariance (`bench.py`, B = 1, starting node, no mask)

ours, (x = the faster of PyTorch compiled / cuEquivariance ÷ ours); cuEquivariance training with the fp32 execution-consistency limit
relaxed from 1e-4 to 1e-3 (its gradients come from atomics); training `n_augment=48 dropout=0.25`.

| mode, d_pair | L128 | L256 | L384 | L512 | L640 | L768 |
|---|---|---|---|---|---|---|
| inference 64 | 0.025 (2.34x) | 0.064 (2.37x) | 0.152 (2.38x) | 0.282 (2.57x) | 0.480 (2.62x) | 0.735 (2.73x) |
| inference 128 | 0.033 (2.41x) | 0.111 (2.38x) | 0.265 (2.70x) | 0.507 (2.81x) | 0.872 (2.82x) | 1.345 (2.94x) |
| inference 256 | 0.065 (2.00x) | 0.257 (2.14x) | 0.610 (2.30x) | 1.170 (2.35x) | 1.970 (2.47x) | 3.096 (2.53x) |
| inference 384 | 0.104 (1.85x) | 0.391 (2.11x) | 0.972 (2.13x) | 1.853 (2.21x) | 3.270 (2.22x) | 5.103 (2.33x) |
| inference 512 | 0.147 (1.69x) | 0.553 (1.97x) | 1.423 (1.97x) | 2.697 (2.08x) | 4.646 (2.15x) | 7.442 (2.44x) |
| training 64 | 0.122 (1.70x) | 0.310 (2.14x) | 0.656 (2.58x) | 1.166 (2.93x) | 1.924 (3.15x) | 2.909 (3.40x) |
| training 128 | 0.133 (2.29x) | 0.410 (2.98x) | 0.942 (3.43x) | 1.793 (3.70x) | 3.064 (3.89x) | 4.955 (3.94x) |
| training 256 | 0.296 (1.75x) | 0.972 (2.47x) | 2.231 (2.88x) | 4.194 (3.10x) | 7.537 (3.14x) | 11.78 (3.44x) |
| training 384 | 0.428 (1.88x) | 1.431 (2.51x) | 3.438 (2.78x) | 6.759 (2.93x) | 11.94 (3.15x) | 19.40 (3.33x) |
| training 512 | 0.569 (1.81x) | 1.974 (2.40x) | 4.882 (2.60x) | 9.629 (2.81x) | 17.20 (3.04x) | 27.54 (3.23x) |

60 of 60 cells faster; smallest 1.69x.  PyTorch compiled runs out of memory at (768, 256) and (640 / 768, 512) training.

### Ending node, key mask (20 %), B = 2

Scratch script, d_pair 64-512, L 128 / 256 / 512 / 768, inference and training, PyTorch compiled and cuEquivariance; smallest ratio
over the 20 cells of each row:

| case | inference | training |
|---|---|---|
| base (starting, B = 1) | 2.43x | 1.69x |
| ending node | 2.27x | 1.69x |
| key mask | 2.35x | 1.74x |
| B = 2 | 2.30x | 1.88x |

160 of 160 cells faster.  Ending node and mask cost nothing over the base case (ours within +-3 %).

### Against Anthropic's native cell (inference)

d_pair 128 and 256 are the only widths its cell serves; scratch script, ms, Anthropic / ours (Anthropic ÷ ours).  `bench.py`
(B = 1, starting, no mask) agrees: d_pair 128 1.57 / 1.52 / 1.49 / 1.43 / 1.35 / 1.30x, d_pair 256 1.17 / 1.21 / 1.25 / 1.21 / 1.19 / 1.11x
for L128-768.

| case, d_pair | L128 | L256 | L384 | L512 | L640 | L768 |
|---|---|---|---|---|---|---|
| base, 128 | 0.047 / 0.030 (1.57x) | 0.163 / 0.109 (1.50x) | 0.378 / 0.265 (1.43x) | 0.696 / 0.504 (1.38x) | 1.146 / 0.873 (1.31x) | 1.736 / 1.360 (1.28x) |
| base, 256 | 0.070 / 0.061 (1.15x) | 0.302 / 0.258 (1.17x) | 0.745 / 0.606 (1.23x) | 1.387 / 1.176 (1.18x) | 2.286 / 2.016 (1.13x) | 3.467 / 3.233 (1.07x) |
| ending, 128 | 0.051 / 0.030 (1.70x) | 0.177 / 0.110 (1.61x) | 0.406 / 0.264 (1.54x) | 0.728 / 0.507 (1.44x) | 1.205 / 0.878 (1.37x) | 1.786 / 1.384 (1.29x) |
| ending, 256 | 0.076 / 0.061 (1.25x) | 0.320 / 0.260 (1.23x) | 0.783 / 0.615 (1.27x) | 1.463 / 1.178 (1.24x) | 2.418 / 2.066 (1.17x) | 3.608 / 3.185 (1.13x) |
| mask, 128 | 0.048 / 0.030 (1.60x) | 0.166 / 0.109 (1.52x) | 0.386 / 0.262 (1.47x) | 0.711 / 0.500 (1.42x) | 1.170 / 0.875 (1.34x) | 1.766 / 1.375 (1.28x) |
| mask, 256 | 0.072 / 0.060 (1.20x) | 0.303 / 0.255 (1.19x) | 0.764 / 0.604 (1.26x) | 1.403 / 1.179 (1.19x) | 2.321 / 1.985 (1.17x) | 3.488 / 3.183 (1.10x) |
| B = 2, 128 | 0.105 / 0.050 (2.10x) | 0.409 / 0.207 (1.98x) | 0.952 / 0.505 (1.89x) | 1.746 / 1.001 (1.74x) | 2.821 / 1.718 (1.64x) | 4.262 / 2.710 (1.57x) |
| B = 2, 256 | 0.194 / 0.122 (1.59x) | 0.767 / 0.481 (1.59x) | 1.864 / 1.178 (1.58x) | 3.471 / 2.364 (1.47x) | 5.759 / 4.092 (1.41x) | 8.533 / 6.485 (1.32x) |

48 of 48 cells faster; the thinnest margin is (768, 256) at 1.07x (1.11x in `bench.py`), where the cuBLAS projections dominate.
Anthropic's cell does not run d_pair 64 (`no-cell:fpf:prologue:64x2x32`), d_pair 384 (`pow2:c384h12d32`) or d_pair 512
(`Refused: pair_fused:epilogue`, a Triton 3.7 PTXAS error) and ships no training path: there the comparison is PyTorch / cuEquivariance only.

### Not measured

fp32 activations, B > 2, a length that is not a multiple of 128 (computed at the next multiple), heads of 64+ channels (Triton path),
and Anthropic with a mask-free ending node at d_pair 64 / 384 / 512 (its cell refuses these widths).
