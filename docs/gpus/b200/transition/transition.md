# Transition on B200 (sm100)

Kernel-level status of the pair Transition (`modules.Transition`, `y = x + W_s(silu(W_a·LN(x)) · (W_b·LN(x)))`) on B200; the
module-level summary is in [b200.md](../b200.md). The hand-CUDA paths cover **n = 4, bf16, D = 64 / 128 / 256 / 384 / 512, whole
128-row tiles** (every registry length: L² is a multiple of 128 for L = 128 … 768); n = 2, other widths and fp32 keep the Triton
residual path. Columns are (Length, Dimension) because the CUDA paths are bf16 only.

Dispatch: `modules/transition/module.py::_residual_forward` and `kernels/transition/whole_op.py` try
`kernels/transition/cuda/fused_sm100a.available()` (D = 128) and then `fused_wide_sm100a.available()` (D = 64 / 256 / 384 / 512)
before the Triton path: sm_100 with an even SM count, bf16 activations and weights, hidden = 4 D, rows a multiple of 128 (D = 64
also needs 82+ SMs for its backward's two roles). Opt out of both with `MINIWORLD_TRANSITION_FUSED_SM100A=0`. The D128 backward's
weight-role replica count is `DW_REPL = 9` (`MINIWORLD_TRANSITION_SM100_REPL` overrides it for A/B runs); D64's is `D64_REPL = 20`.

**Build.** The kernels are compiled into cubins by the newest nvcc on the machine that knows sm_100a (`/usr/local/cuda*/bin/nvcc`
or the torch-matched one; `MINIWORLD_TRANSITION_SM100_NVCC` overrides), one cubin group per width built on first use and cached
under `MINIWORLD_ENGINE_JIT_ROOT/transition_sm100a/`, and launched through the driver API from the torch extension
(`sm100/transition_sm100.cu`: the D128 entry points, plus a generic cubin / tensor-map / launch surface the other widths are
assembled on in Python). The reason is measured: the same D128 backward through CUDA 12.9's ptxas (the toolkit torch cu129 pins)
runs ~8 % slower than through 13.1's (L384 backward 250 vs 230 µs, L768 990 vs 885 µs). On the B200 box CUDA 13.1 is at
`/usr/local/cuda-13.1` and is picked automatically. The sources under `sm100/` are generated from the research capsule by its
`export_engine.py` (experiment switches resolved to the adopted build; each generated file's SASS equals the capsule cubin's).

Figures: one box per kernel, left to right, HBM reads (blue, left) and writes (red, right); generated from
`figures/transition_pair.json` by `python -m miniworld_engine.viz.kernel_flow`. (No `rsvg-convert` on the B200 box or the cssb login
node, so the pages embed the SVG until the PNGs are rendered.) The dW GEMMs of the D ≥ 256 backward are cuBLAS (fp32 output) and
appear in the figures only.

## Pair Transition, n = 4 (`Transition(d_hidden, n=4)`)

### Inference

#### D64 path · every L

![Transition inference, D64](figures/transition_pair_inference_d64.svg)

##### F1 · LN + expand + SwiGLU + squeeze + residual

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
#### D128 path · every L

![Transition inference, D128](figures/transition_pair_inference_d128.svg)

##### F1 · LN + expand + SwiGLU + squeeze + residual

| (Length, Dimension) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
#### D256 path · every L

![Transition inference, D256](figures/transition_pair_inference_d256.svg)

##### F1 · LN + expand + SwiGLU + squeeze + residual

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
#### D384 / D512 path · every L

![Transition inference, D384 / D512](figures/transition_pair_inference_d384.svg)

##### F1 · LayerNorm

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F2 · expand + SwiGLU

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · squeeze + residual

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Training

#### D64 path · every L

![Transition training, D64](figures/transition_pair_training_d64.svg)

##### F1 · LN + expand + SwiGLU + squeeze + residual (saves x_n, rstd, c1)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · fused backward (weight role + input role)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B2 · partial-sum reduction

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
#### D128 path · every L

![Transition training, D128](figures/transition_pair_training_d128.svg)

##### F1 · LN + expand + SwiGLU + squeeze + residual (saves x_n, rstd, c1)

| (Length, Dimension) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · fused backward (weight role + input role)

| (Length, Dimension) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B2 · partial-sum reduction

| (Length, Dimension) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
#### D256 path · every L

![Transition training, D256](figures/transition_pair_training_d256.svg)

##### F1 · LN + expand + SwiGLU + squeeze + residual (saves x_n, rstd, c1)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · gate (recomputes a, b; writes h, dA | dB)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B3 · d_xn GEMM

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B4 · LayerNorm backward + residual (+ reduction)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
#### D384 / D512 path · every L

![Transition training, D384 / D512](figures/transition_pair_training_d384.svg)

##### F1 · LayerNorm (saves rstd, c1)

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F2 · expand + SwiGLU (saves h, a, b)

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · squeeze + residual

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · gate (dh GEMM + SwiGLU backward from the saved a, b)

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B3 · d_xn GEMM

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B4 · LayerNorm backward + residual (+ reduction)

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

No kernel on these paths is autotuned, so nothing needs a cache (✓).

### How the kernels work

- **D128 F1** (`sm100/tr_fwd_sm100.cu`): persistent, 128-row tiles, the two CTAs of a 2-CTA cluster in lockstep. Per 64-unit hidden
  chunk the leader issues the expand `[a|b] = x_n [W_a; W_b]_j^T` as one M = 256 `cta_group::2` product (the leader holds `W_a,j`,
  the peer `W_b,j`, so each SM streams half of the weights) into tensor memory; two SwiGLU warpgroups form `h = bf16(silu(a) b)` and
  write it back to tensor memory, where the squeeze reads it as the A operand. `h` never reaches shared or global memory. LayerNorm
  of the next tile runs on its own warpgroup. Inference skips the `x_n` / statistics stores at run time.
- **D128 B1** (`sm100/tr_bwd_sm100.cu`): 8 × `DW_REPL` weight-role CTAs keep one hidden slice's `W_s / W_a / W_b` resident, recompute
  `dh / a / b`, run the gate, and accumulate `dW_a | dW_b | dW_s` in tensor memory; the other CTAs stream the weight chunks
  (multicast over the pair), form `d_xn`, and run the LayerNorm backward and the residual. **B2** sums the partials in a fixed order.
- **D64** (`sm100/widths/tfwd_d64.cu`, `tbwd_d64.cu`): D128's design with all weights resident in shared memory (48 KB per CTA in
  the forward, 96 KB in the backward's input role); the backward's weight role is 4 hidden slices × 20 replicas.
- **D256 F1** (`widths/tfwd_d256.cu`): one fused kernel; the output accumulator takes 256 tensor-memory columns and the two `[a|b]`
  buffers the rest, so `h` is written in place over its own `[a|b]` buffer. The backward is split: **B1** (`widths/tgate_w.cu`)
  recomputes `a, b` in fp32 from `x_n`, forms `dh = dy W_s` and the SwiGLU backward, and writes `h` and `[dA | dB]`; cuBLAS forms the
  weight gradients in fp32; **B3** (`widths/tgemm_nd.cu`) `d_xn = [dA | dB][W_a; W_b]`; **B4** (`widths/tlnbwd_w.cu`) the LayerNorm
  backward and the residual.
- **D384 / D512**: the output accumulator alone is 384 / 512 tensor-memory columns, so the forward is three kernels: **F1** LayerNorm
  (`widths/tln_w.cu`), **F2** expand + SwiGLU (`widths/tswiglu_w.cu`, writes `h`, and in training `a` and `b`), **F3** squeeze +
  residual (`widths/tgemm_nd.cu`). F2 deals (tile pair, 128-unit chunk) items over all SMs up to 1152 tiles (L ≤ 384: the forward
  5-15 % faster at L256 / L384, equal from L512) and whole tile pairs above. The backward's **B1** (`widths/tgate_ab.cu`) fuses
  `dh = dy W_s` with the SwiGLU backward from the saved `a, b` (no fp32 recompute; ~2 × M × 4D bf16 more activation memory per
  layer), then B3 / B4 as at D256.
- Numerics are the sm_90a contract (same rounding points; `x_n`, `h`, `dA`, `dB`, `d_xn` in bf16, fp32 accumulation; at D ≥ 384 `a, b`
  saved in bf16); vs the fp32 module every width is at least as close as the Triton path
  (`tests/numerics/test_transition_fused_sm100a_gpu.py`, `tests/numerics/test_transition_fused_wide_sm100a_gpu.py`). Reductions run in
  a fixed order (no atomics): a replay is bit-identical.
- Research record (rounds v1-v23 for D128, w1-w6 and s1 for the other widths and small L, with profiles and rejected variants):
  `experiments/transition_fused_sm100/` on branch `perf/transition-sm100-b200`. At D ≥ 384 the training step is at this algorithm's
  energy floor on the power-capped card (round w5): scheduling changes no longer move it.

## Measurements (2026-09-29)

B200 (148 SMs, 1000 W limit), torch 2.13 + cu129, kernels built by CUDA 13.1; `benchmarks/runners/bench.py target=transition
level=module d_pair=D`, compiled, CUDA-graph replay, µs. *ours* = `modules.Transition` through `fused_sm100a` / `fused_wide_sm100a`;
the Triton row is the same module with `MINIWORLD_TRANSITION_FUSED_SM100A=0` (the path B200 ran before; D ≠ 128 values from the
research capsule's module run of the same bench, round w1). × = ours vs the fastest other row.

### Pair, n = 4 · Inference

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | Triton (engine) | ours | × |
|---|---|---|---|---|---|---|
| (384, 64) | 130.9 | n/a | 48.4 * | 85.8 | **36.7** | 1.32 |
| (768, 64) | 370.6 | n/a | 189.9 * | 313.2 | **114.6** | 1.66 |
| (384, 128) | 200.5 | n/a | 108.4 * | 205.8 | **63.3** | 1.71 |
| (768, 128) | 745.2 | n/a | 479.8 * | 755.7 | **227.1** | 2.11 |
| (384, 256) | 407.5 | n/a | 504.8 * | 530.2 | **192.4** | 2.12 |
| (768, 256) | 1604.6 | n/a | 2004.2 * | 2087.8 | **798.5** | 2.01 |
| (384, 384) | 686.1 | n/a | 4444.5 * | 1059.8 | **457.2** | 1.50 |
| (768, 384) | 2912.1 | n/a | 17669.6 * | 4179.5 | **1940.4** | 1.50 |
| (384, 512) | 1004.4 | n/a | n/a (no row) | 1670.0 | **749.3** | 1.34 |
| (768, 512) | 4336.4 | n/a | n/a (no row) | 6732.8 | **3202.8** | 1.35 |

### Pair, n = 4 · Training (forward + backward)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | Triton (engine) | ours | × |
|---|---|---|---|---|---|---|
| (384, 64) | 335.7 | n/a | n/a (no backward) | 329.6 | **132.9** | 2.48 |
| (768, 64) | 1063.8 | n/a | n/a (no backward) | 1182.7 | **448.1** | 2.37 |
| (384, 128) | 588.8 | n/a | n/a (no backward) | 715.8 | **290.6** | 2.03 |
| (768, 128) | 2073.4 | n/a | n/a (no backward) | 2654.0 | **1098.6** | 1.89 |
| (384, 256) | 1210.4 | n/a | n/a (no backward) | 1764.3 | **954.1** | 1.27 |
| (768, 256) | 4393.0 | n/a | n/a (no backward) | 6839.3 | **3833.7** | 1.15 |
| (384, 384) | 1972.1 | n/a | n/a (no backward) | 3360.3 | **1734.5** | 1.14 |
| (768, 384) | 8666.1 | n/a | n/a (no backward) | 13429.8 | **6850.0** | 1.27 |
| (384, 512) | 2894.7 | n/a | n/a (no backward) | 5210.0 | **2671.5** | 1.08 |
| (768, 512) | 12775.4 | n/a | n/a (no backward) | 21585.4 | **11183.1** | 1.14 |

\* Anthropic: the module-level runner refuses its rows on cc 10.0; the numbers are Anthropic's best Transition row at the kernel level
from the research capsule (`records/infer-all-s1.json`: `lnl` at D64, `v2` at D128, `pf` at D256 / D384; no row runs at D512).

Accuracy: every width passes "no further from fp32 than the Triton path" on 1 / 3 / 15 / 1152 tiles (and 1200 tiles at D ≥ 384,
the tile schedule), gradients included. Separate processes vary by up to ±10 % on this power-capped card (e.g. torch.compile D384
L768 training 7939 µs without / 8666 µs with CUDA graphs in the same run).

Host side: a training call through the autograd Function and the custom ops costs ~160 µs of CPU at D128 and ~210-300 µs at the
other widths (µs per forward + backward, measured with the GPU kept busy). Everywhere but D64 the GPU work is longer; D64 at L384
(133 µs of GPU work) is host-bound without CUDA graphs (210 µs).

The module-level forward is ~5-9 µs above the kernels alone (D128 L384: 63.3 vs 55.8 µs): the module casts its fp32 parameters to
bf16 on every call under `bf16-mixed`. Against the power-capped ceiling of this card (dense bf16 cuBLAS sustains 1.30 PFLOP/s at the
1000 W limit), the D128 training step reaches ~67 % (L384) / ~72 % (L768) of the design's tensor work at that rate; the research
record explains why the fusion cannot reach 90 % on this card with the bf16 contract.
