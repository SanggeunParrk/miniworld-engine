# Transition on B200 (sm100)

Kernel-level status of the Transition (`modules.Transition`, `y = x + W_s(silu(W_a·LN(x)) · (W_b·LN(x)))`, hidden n·D) on B200; the
module-level summary is in [b200.md](../b200.md). The hand-CUDA paths cover **bf16, n = 4 at D = 64 / 128 / 256 / 384 / 512 and
n = 2 at D = 64 / 128 / 256 / 384 / 512 / 768**, for any activation whose rows (every dimension but the last) are a whole number of
128-row tiles: the pair stream [B, L, L, D] (every registry length: L² is a multiple of 128 for L = 128 … 768), the MSA stream
[B, N, L, D] and a single stream [B, L, D]. Other (D, n), ragged row counts and fp32 run the generic tcgen05 kernels
([Generic path](#generic-path-any-width-row-count-and-fp32)). Columns are (Length, Dimension)
because the CUDA paths are bf16 only; the tables below are for the pair stream, the other streams run the same kernels on their
rows (the single stream at D = 384 below 17 tiles excepted, see [Small M](#small-m-d384-at-most-16-tiles)).

Dispatch: `modules/transition/module.py::_residual_forward` and `kernels/transition/whole_op.py` try
`kernels/transition/cuda/fused_sm100a.available()` (D = 128, n = 4) and then `fused_wide_sm100a.available()` (every other (D, n) in
its `SHAPES`) before the portable path: sm_100 with an even SM count, bf16 activations and weights, hidden = n D, rows a multiple of
128 (the two-role backwards also need enough SMs for both roles). Opt out of both with `settings.transition_fused_sm100a=0`. The
two-role backwards' weight-role replica counts are `DW_REPL = 9` (D128 n = 4; `settings.transition_sm100_repl` overrides it) and
`REPL` = 20 / 30 / 14 for (64, 4) / (64, 2) / (128, 2) (`settings.transition_wide_repl`). At D = 384 the small-M paths take calls of
at most 16 tiles (`SMALL_FWD_TILES`, `SMALL_TILES`; `settings.transition_wide_small_fwd_tiles`, `settings.transition_wide_small_tiles`).

**Build.** The kernels are compiled into cubins by the newest nvcc on the machine that knows sm_100a (`/usr/local/cuda*/bin/nvcc`
or the torch-matched one; `settings.transition_sm100_nvcc` overrides), one cubin group per (width, n) built on first use (each
kernel with `-DHID=<n D>`) and cached under `settings.engine_jit_root/transition_sm100a/`, and launched through the driver API from
the torch extension (`sm100/transition_sm100.cu`: the D128 entry points, plus a generic cubin / tensor-map / launch surface the other
widths are assembled on in Python). The reason is measured: the same D128 backward through CUDA 12.9's ptxas (the toolkit torch
cu129 pins) runs ~8 % slower than through 13.1's (L384 backward 250 vs 230 µs, L768 990 vs 885 µs). A CUDA 13.1 at
`/usr/local/cuda-13.1` is picked automatically. The sources under `sm100/` are generated from the standalone development build of
the same kernel sources (not shipped), with the experiment switches resolved to the adopted build; each generated file's SASS
equals that build's cubin.

Figures: one box per kernel, left to right, HBM reads (blue, left) and writes (red, right); generated from
`figures/transition_pair.json` by `python -m miniworld_engine.viz.kernel_flow`. (`rsvg-convert` was not available where the figures were
generated, so the pages embed the SVG until the PNGs are rendered.) The dW GEMMs of the D ≥ 256 backward are cuBLAS (fp32 output) and
appear in the figures only. The n = 2 builds run the kernels of the same width's figure (D768 as D512).

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
#### D384 path · every L

![Transition inference, D384](figures/transition_pair_inference_d384.svg)

##### F1 · LN + expand + SwiGLU + squeeze + residual

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
#### D512 path · every L

![Transition inference, D512](figures/transition_pair_inference_d512.svg)

##### F1 · LayerNorm

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F2 · expand + SwiGLU

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · squeeze + residual

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

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
#### D384 path · every L

![Transition training, D384](figures/transition_pair_training_d384.svg)

##### F1 · LN + expand + SwiGLU + squeeze + residual (saves x_n, rstd, c1, h, a, b)

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · gate + d_xn + LayerNorm backward + residual (writes dA | dB, dγ / dβ partials)

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B2 · partial-sum reduction

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
#### D512 path · every L

![Transition training, D512](figures/transition_pair_training_d512.svg)

##### F1 · LayerNorm (saves rstd, c1)

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F2 · expand + SwiGLU (saves h, a, b)

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · squeeze + residual

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · gate (dh GEMM + SwiGLU backward from the saved a, b)

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B3 · d_xn GEMM

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B4 · LayerNorm backward + residual (+ reduction)

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Transition, n = 2 (`Transition(d, n=2)`)

The same kernel sources built with `-DHID=2D`. In the consumer model code n = 2 is the diffusion conditioning's pair (D128) and
single (D384) Transition; the per-width paths are the n = 4 ones (D128 through `fused_wide_sm100a` with the D128 fused kernels at
hidden 256), plus D768, which runs the D512 path with the squeeze and d_xn GEMMs as two 384-column launches (one launch's accumulator
must fit 512 tensor-memory columns).

### Inference

##### F1 · D64 / D128 / D256 / D384: LN + expand + SwiGLU + squeeze + residual (one kernel)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · D512 / D768: LayerNorm

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) | (128, 768) | (256, 768) | (384, 768) | (512, 768) | (640, 768) | (768, 768) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F2 · D512 / D768: expand + SwiGLU

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) | (128, 768) | (256, 768) | (384, 768) | (512, 768) | (640, 768) | (768, 768) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · D512 / D768: squeeze + residual (D768: two 384-column launches)

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) | (128, 768) | (256, 768) | (384, 768) | (512, 768) | (640, 768) | (768, 768) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Training

##### F1 · D64 / D128 / D256 / D384: one fused kernel (saves x_n, rstd, c1; at D384 also h, a, b)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · D64 / D128: fused two-role backward

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B2 · D64 / D128: partial-sum reduction

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · D256: gate (recomputes a, b; writes h, dA | dB)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · D384: gate + d_xn + LayerNorm backward + residual

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B2 · D384: partial-sum reduction

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1-F3 · D512 / D768: LayerNorm (saves rstd, c1), expand + SwiGLU (saves h, a, b), squeeze + residual

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) | (128, 768) | (256, 768) | (384, 768) | (512, 768) | (640, 768) | (768, 768) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · D512 / D768: gate (dh GEMM + SwiGLU backward from the saved a, b)

| (Length, Dimension) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) | (128, 768) | (256, 768) | (384, 768) | (512, 768) | (640, 768) | (768, 768) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B3 · D256 / D512 / D768: d_xn GEMM (D768: two 384-column launches)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) | (128, 768) | (256, 768) | (384, 768) | (512, 768) | (640, 768) | (768, 768) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B4 · D256 / D512 / D768: LayerNorm backward + residual (+ reduction)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) | (128, 768) | (256, 768) | (384, 768) | (512, 768) | (640, 768) | (768, 768) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


## Small M (D384, at most 16 tiles)

A single-stream activation [1, L, 384] is L / 128 ≤ 6 tiles (the pairformer's single track, the diffusion conditioning at inference);
per-tile kernels would keep one SM pair per tile busy. Up to 16 tiles both directions run chains of programmatic dependent launches
(PDL: each kernel's prologue -- barriers, tensor memory, the weight K-blocks -- overlaps the previous kernel) sized for a few tiles on
all 148 SMs (`sm100/widths/tsx_*.cu`, round sx, 2026-10-29). Every operand is read in its own layout (W_s, W_a / W_b and the
activations MN-major where the product needs it), so no transposed or concatenated weight copy is made per call; with fp32 master
parameters the training forward's bf16 casts run as extra blocks of the LayerNorm launch. Columns below are (L, D) of [1, L, 384],
n = 2 or 4. Larger calls (the training batch [48, L, 384]) run the per-tile D384 path above.

### Inference and training

##### F1 · LayerNorm, one warp per row (training: saves x_n, rstd, c1; fp32 masters: + the three weight casts)

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F2 · expand + SwiGLU over (tile, 32- / 64-unit hidden chunk) CTAs (training: saves h, a, b)

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · squeeze + residual over (tile, 32- / 64-column block) clusters, hidden split over a CTA pair

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · gate (dh = dy W_s, W_s MN-major + SwiGLU backward from the saved a, b) over (tile, 64-unit chunk) CTAs

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B2 · d_xn = dA W_a + dB W_b + LayerNorm backward + residual, one cluster per tile (6 column blocks x K halves)

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B3 · dW_s, dW_a | dW_b (fp32, or bf16 for bf16 parameters) + dgamma / dbeta over (output tile) CTAs, 64 / 128 / 192 columns each: the narrowest whose CTAs fit beside B2's clusters

| (Length, Dimension) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Measured 2026-10-29 (B200, node 2 GPU 5, the module call in a CUDA graph under bf16 autocast, median of 5 × 50 replays, µs;
the B200 port session's harness `/NHNHOME/WORKSPACE/26mohw002_A/psk6950/scratch/port1028/bench_port.py`, not shipped; *before* = engine a014b518, the cluster forward `tsmall_w` + the D512-style
backward with per-call `W_s^T` / `[W_a; W_b]^T` copies; the training step includes the harness's `.grad` accumulation adds, ~11 µs):

| shape (n) | params | inference before | inference after | training before | training after |
|---|---|---|---|---|---|
| [1, 768, 384] (2) | bf16 | 16.5 | **10.4-12.3** | 90.1 | **39.2** |
| [1, 768, 384] (4) | bf16 | 20.5 | **14.4** | 110.6 | **49.2** |
| [1, 384, 384] (2) | bf16 | 16.5 | **10.3** | 88.1 | **36.9** |
| [1, 384, 384] (4) | bf16 | 20.5 | **12.3** | 108.6 | **45.1** |
| [1, 768, 384] (2) | fp32 masters | 22.6 | **14.4** | 89.7 | **41.0** |
| [1, 768, 384] (4) | fp32 masters | 26.7 | **16.7** | 111.0 | **53.3** |
| [1, 384, 384] (2) | fp32 masters | 22.6 | **12.4** | 88.4 | **39.9** |
| [1, 384, 384] (4) | fp32 masters | 26.7 | **14.9** | 108.7 | **47.8** |

(The [1, 768, 384] n = 2 inference cell measured 10.4 and 12.3 µs in different sessions with the same code -- before 16.5 in both;
fp32-master inference includes the in-graph weight cast of a capture without `static_weights`.) At 10-16 tiles the chains are ~2× the
per-tile path (L2048 n = 4: inference 28.7 vs 49.2 µs, training 94.2 vs 151.8 µs). H100 (generic kernels, 3 + PDL) does
[1, 768, 384] n = 2 in 11.8 / 33.1 µs.

**Round sx2 (2026-10-29).** B3's 108 CTAs (64-column tiles) next to B2's 72 exceeded the 148 SMs at [1, 768, 384]: the CTAs left over
started their whole K loop only after B2 ended, ~5 µs after it instead of ~2. B3 now takes the narrowest output tile width (64 / 128 /
192 columns, `SX_DW_NT`) whose CTAs fit the SMs B2 leaves free (n = 2: 128, n = 4: 192) and stores its tile before
`griddepcontrol.wait`, so only the dgamma / dbeta sum waits for B2. Training step, same conditions as above (the harness
`scratch/k2tr2/bench_tr.py`, not shipped; min of two runs of 5 × 50 replays; before = engine d879e180):

| shape (n) | bf16 params before | bf16 params after | fp32 masters before | fp32 masters after |
|---|---|---|---|---|
| [1, 768, 384] (2) | 41.0 | **37.0** | 41.0 | **39.0** |
| [1, 768, 384] (4) | 49.2 | **46.4** | 49.6 | **45.1** |
| [1, 384, 384] (2) | 36.9 | **34.9** | 36.9 | **36.9** |
| [1, 384, 384] (4) | 43.1 | **41.6** | 44.7 | **43.0** |

Of the ~37 µs at [1, 768, 384] n = 2 (bf16), ~9 µs are the harness's six `.grad` accumulation adds; the kernels end at ~28.5 µs
(nsys, graph replay): F1-F3 12.0, B1 +5.3, B2 +9.3, B3 +2.0. Tried and not kept: B2 starting per tile on arrival counters the gate
CTAs bump instead of `griddepcontrol.wait` (B2's 12-CTA clusters only become resident as gate CTAs leave their GPC, and the gate's
fence + atomic cost ~1 µs: 3 µs slower); dgamma / dbeta summed by B2's last CTA (an end-of-kernel atomic: ~2.5 µs slower than B3's
sum); half of the gate's sigmoids on the FMA pipe (no change outside the ±1 µs run-to-run spread). Left: one kernel for B1 + B2 per
row tile (the gate chunks of a tile on its B2 cluster), which also needs B3 to learn of `[dA | dB]` without B2's completion.

No kernel on these paths is autotuned, so nothing needs a cache (✓).

## Generic path (any width, row count and fp32)

`kernels/transition/cuda/fused_generic_sm100a.py` serves every call the tuned builds above do not take: any D that is a multiple of 64
up to 1024, any hidden width that is a multiple of 128, **any row count** (TMA zero-fills the tail of the last 128-row tile and clips
it on the store: no padding copy) and **fp32 activations on the TF32 tensor cores** (`tcgen05.mma.kind::tf32`, only where torch allows
TF32; strict fp32 keeps the PyTorch reference). The module routes fp32 and the widths without a build here; `ops.transition` (the
entry FoldForge's pairformer uses) pads a ragged row count for the tuned builds where one exists (measured faster: pair
[1, 300, 300, 128] n4 50 against 85 µs) and sends the rest here.

Kernels (`sm100/generic/`):

* `tg_gemm.cu`: one persistent, warp-specialised tcgen05 GEMM (320 threads: a TMA producer warp, an MMA warp, eight epilogue warps,
  two per TMEM lane quarter), accumulators double-buffered in tensor memory, operands K-major on both sides through a 128-B-swizzled
  ring, epilogue staged through swizzled shared memory and TMA-stored. Built per epilogue: SWIGLU (two accumulators, h = silu(a) b),
  RESID (+ x), PLAIN (fp32 d_xn; optionally split over K: slice s stores its fp32 partial in its own slab of rows), GATE (three
  accumulators over two operand phases: dh = dy Ws and a, b recomputed from x_n, writing h and [dA | dB]), SWIGLU_AB (SWIGLU that also
  stores a, b), GATE_AB (dh = dy Ws with a, b read from the forward's copy: one operand phase) and CTGATE (the ConditionedTransition's
  squeeze and gate projection, see `kernels/conditioned_transition/cuda/ct_sm100.py`); output tiles 128 / 64 / 32 columns, the widest
  where that gives every SM a tile, else the plan below.
* `tg_rows.cu`: LayerNorm forward / backward (16-B vectors, 8-32 lanes per row; the backward's per-block dgamma / dbeta partials reduced
  slot-parallel; `tg_ln_bwds_*` adds a split-K d_xn's slabs in slab order), the TF32 rounding of dy, the fixed-order column sums, the
  weight preparation (activation dtype, TF32 grid; with the backward's Ws^T and [Wa; Wb]^T transposed through 32 x 32 shared-memory tiles).

Forward: LN -> SWIGLU -> RESID. Backward: GATE -> cuBLAS dWs, [dWa; dWb] (fp32 out) -> PLAIN d_xn -> LN backward with the residual ->
column sums. The forward saves only x_n and the statistics. At a few row tiles (a single track, where the widest tiles leave SMs idle)
`_plan` picks each GEMM's tile width -- and for d_xn a K split -- by waves x per-CTA operand intake, the squeeze runs on cuBLAS `addmm`
(it splits K inside one launch: 0-4 µs faster than the split PLAIN plus a summing launch), and a training forward keeps a, b
(SWIGLU_AB) for GATE_AB while the gate grid is under one wave. Accuracy (tests): within 1.5x the error of the same-dtype PyTorch
statements against fp64, output and every gradient.

Measured 2026-10-29, CUDA graph replay, µs (before = the PyTorch statements B200 ran for these calls):

| call | before | after | speed-up |
|---|---|---|---|
| `ops.transition` bf16 inference [1, 300, 300, 128] n4 (padded tuned build) | 251.4 | 49.5 | 5.08x |
| `ops.transition` bf16 inference [1, 300, 300, 256] n4 / n2 (padded tuned) | 448.6 / 305.4 | 153.8 / 102.5 | 2.92x / 2.98x |
| `ops.transition` bf16 inference [1, 300, 300, 384] / [1, 300, 300, 512] n4 (padded tuned) | 697.2 / 1001.5 | 304.9 / 559.7 | 2.29x / 1.79x |
| `ops.transition` bf16 inference [1, 300, 384] n4 / n2 (padded tuned) | 22.1 / 20.3 | 21.5 / 18.3 | 1.03x / 1.11x |
| module fp32 [1, 384, 384, 128] n4 inference / training | 729.3 / 2195.5 | 215.2 / 926.1 | 3.39x / 2.37x |
| module fp32 [1, 384, 384, 256] n4 inference / training | 1392.1 / 4338.6 | 576.2 / 2465.0 | 2.42x / 1.76x |
| module fp32 [48, 384, 384] n2 inference / training | 167.4 / 652.3 | 85.1 / 419.3 | 1.97x / 1.56x |
| module fp32 [1, 768, 384] n4 inference / training | 31.6 / 99.6 | 24.8 / 106.3 | 1.27x / 0.94x |

Generic GEMMs alone (bf16, 90 000 rows): D128 n4 SWIGLU 43 µs (548 TFLOP/s), RESID 27 µs (441); D256 n4 SWIGLU 108 µs (876), RESID
70 µs (672).

**Small M (2026-10-29).** The module in fp32 (fp32 parameters and activations, TF32 allowed) and bf16-mixed, a single track; CUDA graph,
min of two runs of 5 × 50 replays, µs, training incl. the harness's `.grad` adds (`scratch/k2tr2/bench_tr.py`, not shipped; before =
engine d879e180, PyTorch = the module's PyTorch statements in the same harness):

| call | before | after | PyTorch |
|---|---|---|---|
| fp32 [1, 768, 384] n4 training / inference | 121.7 / 28.8 | **92.2 / 25.8** | 125.2 / 44.8 |
| fp32 [1, 768, 384] n2 training | 98.0 | **77.9** | 110.6 |
| fp32 [1, 384, 384] n4 training | 104.5 | **77.9** | 113.0 |
| fp32 [1, 300, 384] n4 training | 106.5 | **75.8** | 108.6 |
| fp32 [1, 768, 512] n2 training / inference | 143.4 / 34.9 | **91.9 / 26.7** | 120.8 / 41.0 |
| fp32 [1, 154, 768] n2 training | 145.4 | **98.3** | 112.7 |
| fp32 [1, 768, 256] n4 / [1, 768, 128] n2 training | 96.3 / 69.7 | **76.4 / 61.5** | 110.7 / 94.3 |
| bf16-mixed [1, 300, 768] n4 / [1, 2048, 768] n4 training | 190.9 / 330.9 | **105.3 / 209.8** | |

What moved (fp32 [1, 768, 384] n4, kernels alone): the weight preparation's scattered transposes 14 -> 4 µs, the LayerNorm backward
18 -> 4.7 µs (12 blocks of 64 rows each walking 8 rows, and 8-32 serial read-modify-write rounds per block for the partials), GATE
20 -> 13.5 µs (64-column tiles in one wave instead of 32 in two), d_xn split K 4 ways (9.9 µs; cuBLAS 10.8, the K-serial GEMM 19.6 at
D512). Larger calls are unchanged (fp32 pair [1, 384, 384, 128] / 256 n4 training 971 -> 964 / 2540 -> 2525 µs; the LayerNorm
backward's split-K variant is a separate instantiation so the large-M one keeps 64 registers and four blocks per SM). Every call above
is faster than the PyTorch statements. Left: the generic GEMMs at a few row tiles run at ~2x their operand-intake model (prologue and
epilogue latency: GATE_AB 12.5 µs at [768, 1536, 384] fp32, cuBLAS's plain GEMM of that size 5.5), and a fused one-kernel forward for
widths without a tuned build.

## How the kernels work

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
- **D384 F1** (`widths/tfwd_d384.cu`): one fused kernel. The output accumulator takes 384 tensor-memory columns, so there is one
  128-column `[a|b]` buffer (a 64-unit hidden chunk); the two SwiGLU warpgroups read their half of it with one 64-column load, hand it
  back at once and write `h` to shared memory, where the squeeze reads it. The weights stream through a 2-stage ring of 3 K-blocks
  (12 MMAs per wait) and a single W_s stage; the LayerNorm runs on 12 warps (three threads per row) and the epilogue of tile i
  (residual by TMA, `x + acc` in place, TMA store) overlaps the LayerNorm of tile i + 1. Training stores `h, a, b` from registers with
  a quad transpose (each store 8 rows × 64 B).
- **D384 B1** (`widths/tbwd_d384.cu`): gate, d_xn and the LayerNorm backward in one kernel. Per 64-unit chunk: `dh = dy W_s` (the dy
  tile resident as the A operand, `W_s^T` K-major) into one of two 64-column tensor-memory buffers; the gate reads `dh` and the saved
  `a, b` (TMA) and writes `[dA | dB]` back over `dh` as bf16 (half by half); `d_xn += [dA | dB][W_a; W_b]` takes A from tensor memory
  and B in the weights' own layout (MN-major, a 4-slot ring of 16 KB atoms, one N = 128 product per atom). Warps 8-11 copy `[dA | dB]`
  to HBM for the dW GEMMs; after the tile's last chunk its x is loaded over the weight region and 12 warps run the LayerNorm backward
  from `d_xn` in tensor memory, writing dx over dy. dγ / dβ are per-CTA partials summed by **B2** in a fixed order. The dW GEMMs stay
  cuBLAS (fp32 output); the host makes `W_s^T` per call.
- **D512 / D768**: the output accumulator alone is 512 (768) tensor-memory columns, so the forward is three kernels: **F1** LayerNorm
  (`widths/tln_w.cu`), **F2** expand + SwiGLU (`widths/tswiglu_w.cu`, writes `h`, and in training `a` and `b`), **F3** squeeze +
  residual (`widths/tgemm_nd.cu`; D768 as two 384-column launches). F2 deals (tile pair, 128-unit chunk) items over all SMs up to
  1152 tiles and whole tile pairs above. The backward's **B1** (`widths/tgate_ab.cu`) fuses `dh = dy W_s` with the SwiGLU backward
  from the saved `a, b`, then B3 / B4 as at D256.
- **Small M, D384** (`widths/tsx_ln.cu`, `tsx_expand.cu`, `tsx_squeeze.cu`, `tsx_gate.cu`, `tsx_dxn.cu`, `tsx_dw.cu`; see
  [Small M](#small-m-d384-at-most-16-tiles)). The bound is the per-SM TMA intake (~60 B/clk), not the products: the squeeze splits a
  row tile's hidden units over a CTA pair (fp32 partials through distributed shared memory) and B2 splits K into the `dA W_a` and
  `dB W_b` halves (pairwise mbarrier exchange) while the 12-CTA clusters are all resident (≤ 7 tiles); B2's row sums for the
  LayerNorm backward cross the cluster through distributed shared memory under split arrive / wait barriers; dgamma / dbeta are
  per-tile partials summed in tile order by B3, whose output tiles are sized so its CTAs run beside B2's clusters (round sx2). The LayerNorm runs once per row in F1 instead of in each of the 12-48 CTAs that share
  a tile (it is instruction bound: ~4k cycles per 128 × 384 tile).
- Numerics are the sm_90a contract (same rounding points; `x_n`, `h`, `dA`, `dB`, `d_xn` in bf16, fp32 accumulation; at D ≥ 384 `a, b`
  saved in bf16); vs the fp32 module every (D, n) is at least as close as the portable path
  (`tests/gpu/sm100/test_transition_fused_sm100a_gpu.py`, `tests/gpu/sm100/test_transition_fused_wide_sm100a_gpu.py`: 1 / 3 / 15 / 1152
  tiles, 1200 at D ≥ 384, the MSA and single-stream shapes). Reductions run in a fixed order (no atomics): a replay is bit-identical.
- The research record (rounds v1-v23 for D128, w1-w6, s1 and n2 for the other widths, n = 2 and D384, with profiles and rejected
  variants) belongs to the standalone development build and is not shipped.

## Measurements

### Pair, n = 4 (2026-09-29; D384 *ours* 2026-09-30; re-verified 2026-10-10)

B200 (148 SMs, 1000 W limit), torch 2.13 + cu129, kernels built by CUDA 13.1; `benchmarks/runners/bench.py target=transition
level=module d_pair=D`, compiled, CUDA-graph replay, µs. *ours* = `modules.Transition` through `fused_sm100a` / `fused_wide_sm100a`;
the Triton row is the same module with `settings.transition_fused_sm100a=0` (the path B200 ran before). × = ours vs the fastest other row.
Re-verified 2026-10-10 (engine v2.2.0 plus unreleased changes, driver 580.95.05, one GPU with no other process on it): the PyTorch compiled, Triton and ours **2026-10-11:** every CUDA-graph cell of these tables was re-measured with fp32 master parameters (`precision=bf16-mixed`, the harness default since that date: fp32 parameters over bf16 activations under `torch.autocast`; one run per cell, `parameter_dtype` = float32 in the CSV, the engine environment's python called directly); the cells above are those values. The no-CUDA-graph training tables are host-bound and keep their earlier values (bf16 parameters).
columns (60 cells) were re-measured with the command above, one run per width and mode, cells that moved by more than 8 % repeated;
median change 1.1 % (ours 3.5 %), 90th percentile 5.3 %, no cell replaced. The D384 *ours* values are the fused
D384 kernels, re-measured 2026-09-30 with the same command (before: inference 457.2 / 1940.4, training 1734.5 / 6850.0 µs at
L384 / L768); the other columns are unchanged. At these sizes (1152 / 4608 tiles) the training step barely moves: at L384 the fused
forward with its h / a / b stores is ~40 µs slower than the three-kernel forward (kernels alone: 363 vs 322 µs) and the fused
backward ~70 µs faster (648 vs 720 µs); inference gains 1.25x.

#### Inference

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Triton (engine) | ours | × |
|---|---|---|---|---|---|
| (384, 64) | 137.1 | n/a | 89.9 | **36.7** | 2.45 |
| (768, 64) | 382.3 | n/a | 317.2 | **114.6** | 2.77 |
| (384, 128) | 208.7 | n/a | 208.7 | **63.4** | 3.29 |
| (768, 128) | 760.8 | n/a | 760.7 | **225.2** | 3.38 |
| (384, 256) | 440.3 | n/a | 535.3 | **192.4** | 2.29 |
| (768, 256) | 1635.3 | n/a | 2067.3 | **798.7** | 2.05 |
| (384, 384) | 698.3 | n/a | 1063.8 | **367.4** | 1.90 |
| (768, 384) | 2892.8 | n/a | 4191.2 | **1616.8** | 1.79 |
| (384, 512) | 1049.4 | n/a | 1647.4 | **732.9** | 1.43 |
| (768, 512) | 4073.2 | n/a | 6513.7 | **3050.3** | 1.34 |

![Pair, n = 4 (2026-09-29; D384 *ours* 2026-09-30; re-verified 2026-10-10), length sweep at D128](figures/transition_pair_n_4_2026_09_29_d384_ours_2026_09_30_re_verified_2026_10_10_length.png) ![Pair, n = 4 (2026-09-29; D384 *ours* 2026-09-30; re-verified 2026-10-10), dimension sweep at L384](figures/transition_pair_n_4_2026_09_29_d384_ours_2026_09_30_re_verified_2026_10_10_dimension.png) <!-- measure_bars -->

#### Training (forward + backward)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Triton (engine) | ours | × |
|---|---|---|---|---|---|
| (384, 64) | 344.1 | n/a | 337.9 | **137.2** | 2.46 |
| (768, 64) | 1073.2 | n/a | 1192.7 | **452.5** | 2.37 |
| (384, 128) | 598.0 | n/a | 723.8 | **287.6** | 2.08 |
| (768, 128) | 2087.8 | n/a | 2660.2 | **1123.3** | 1.86 |
| (384, 256) | 1162.9 | n/a | 1672.0 | **1036.1** | 1.12 |
| (768, 256) | 4338.6 | n/a | 6771.4 | **3926.1** | 1.11 |
| (384, 384) | 2066.3 | n/a | 3221.3 | **1714.1** | 1.21 |
| (768, 384) | 7869.5 | n/a | 12841.0 | **6780.1** | 1.16 |
| (384, 512) | 2923.4 | n/a | 5227.1 | **2705.2** | 1.08 |
| (768, 512) | 12929.6 | n/a | 21567.5 | **11518.9** | 1.12 |

![Pair, n = 4 (2026-09-29; D384 *ours* 2026-09-30; re-verified 2026-10-10), length sweep at D128](figures/transition_pair_n_4_2026_09_29_d384_ours_2026_09_30_re_verified_2026_10_10_length.png) ![Pair, n = 4 (2026-09-29; D384 *ours* 2026-09-30; re-verified 2026-10-10), dimension sweep at L384](figures/transition_pair_n_4_2026_09_29_d384_ours_2026_09_30_re_verified_2026_10_10_dimension.png) <!-- measure_bars -->

Accuracy: every width passes "no further from fp32 than the portable path" on 1 / 3 / 15 / 1152 tiles (and 1200 tiles at D ≥ 384,
the tile schedule), gradients included. Separate processes vary by up to ±10 % on this power-capped card (e.g. torch.compile D384
L768 training 7939 µs without / 8666 µs with CUDA graphs in the same run).

Host side: a training call through the autograd Function and the custom ops costs ~160 µs of CPU at D128 and ~210-300 µs at the
other widths (µs per forward + backward, measured with the GPU kept busy). Everywhere but D64 the GPU work is longer; D64 at L384
(133 µs of GPU work) is host-bound without CUDA graphs (210 µs).

The module-level forward is ~5-9 µs above the kernels alone (D128 L384: 63.3 vs 55.8 µs): the module casts its fp32 parameters to
bf16 on every call under `bf16-mixed`. Against the power-capped ceiling of this card (dense bf16 cuBLAS sustains 1.30 PFLOP/s at the
1000 W limit), the D128 training step reaches ~67 % (L384) / ~72 % (L768) of the design's tensor work at that rate; the fusion
cannot reach 90 % on this card with the bf16 contract.

### n = 2, MSA and single streams (2026-09-30)

Same card and toolchain; module level with bf16 parameters, one eager call captured in a CUDA graph and replayed, µs (measured with a
benchmark of the standalone development build, not shipped; `benchmarks/runners/bench.py` has no n = 2 / MSA / single-stream Transition case). *ours* = `modules.Transition` on
this path; Triton = the same module with `settings.transition_fused_sm100a=0`; compiled = `torch.compile(dynamic=False)` of the
PyTorch implementation. × = ours vs the faster other column. The D64 / D128 / D256 and MSA rows were measured before the D384 fused
kernels landed (those widths' kernels are unchanged since).

| shape (n) | compiled inf. | Triton inf. | ours inf. | × | compiled train | Triton train | ours train | × |
|---|---|---|---|---|---|---|---|---|
| pair [1, 384, 384, 64] (2) | 61.7 | 51.6 | **24.8** | 2.08 | 243 | 221 | **115** | 1.92 |
| pair [1, 384, 384, 128] (2) | 150.7 | 121.1 | **44.1** | 2.75 | 435 | 454 | **230** | 1.89 |
| pair [1, 384, 384, 256] (2) | 241 | 298 | **125** | 1.93 | 750 | 988 | **615** | 1.22 |
| pair [1, 384, 384, 384] (2) | 403.8 | 578.3 | **220.3** | 1.83 | 1181.2 | 1771.6 | **977.3** | 1.21 |
| pair [1, 768, 768, 384] (2) | 1680.1 | 2368.8 | **839.5** | 2.00 | 4625.3 | 7309.5 | **4077.7** | 1.13 |
| MSA [1, 1024, 768, 64] (4) | 486 | 404 | **143** | 2.83 | 1436 | 1605 | **634** | 2.26 |
| MSA [1, 1024, 384, 128] (4) | 500.2 | 515.4 | **140.5** | 3.56 | 1487.8 | 1846.3 | **722.2** | 2.06 |
| MSA [1, 1024, 768, 128] (4) | 981.5 | 1034.7 | **275.2** | 3.57 | 2817.7 | 3602.1 | **1504.8** | 1.87 |
| [48, 384, 384] (2) | 57.7 | 81.3 | **34.7** | 1.66 | 228.1 | 279.3 | **174.2** | 1.31 |
| [48, 768, 384] (2) | 106.8 | 154.1 | **61.7** | 1.73 | 370.5 | 499.4 | **296.8** | 1.25 |
| [48, 384, 384] (4) | 94.9 | 144.4 | **51.5** | 1.84 | 330.2 | 459.7 | **262.9** | 1.26 |
| [48, 768, 384] (4) | 182.5 | 274.5 | **95.5** | 1.91 | 572.6 | 845.7 | **449.7** | 1.27 |
| single [1, 384, 384] (2) | 14.7 | 16.7 | **16.8** | 0.87 | 55.7 | 53.9 | **88.3** | 0.61 |
| single [1, 768, 384] (2) | 16.7 | 17.8 | **18.6** | 0.90 | 59.6 | 61.7 | **90.3** | 0.66 |

At ≤ 6 tiles (the single stream [1, L, 384]) the cluster forward was on par with Triton and ~12 % behind torch.compile at inference,
and training 1.5-1.6x slower than both; superseded on 2026-10-29 by the PDL chains (see [Small M](#small-m-d384-at-most-16-tiles):
[1, 768, 384] n = 2 inference 10.4-12.3, training 39.2 µs). Every other shape above is faster than both.
The diffusion conditioning's training batch [48, L, 384] n = 2 went from 220.7 / 328.9 µs (L384 / L768, three-kernel forward and
backward chain) to 174.2 / 296.8 µs with the fused D384 kernels.
