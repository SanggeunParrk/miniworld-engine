# TriangleAttention on H100: re-measurements of 2026-10-13 to 2026-10-15

Dated measurements of `TriangleAttention` (`use_self_attention=True`) on H100, kept as taken (the page
[triattn.md](../gpus/h100/triattn/triattn.md) carries the current tables).  Order: the 10-13 re-measurement on `h100/cuda-sweep`, the
10-14 grid at `f2c4cda4`, the ending node / key mask / B = 2 cases, the comparison with Anthropic's native cell and its correction
(the weight pack replayed inside every CUDA graph, fixed in `7f7851aa`), the 10-15 grids after the fix.  Protocols:

- `bench.py` rows: `benchmarks/runners/bench.py target=triangle_attention level=module cudagraph=manual compile=true`, bf16-mixed, B1,
  starting node, no mask.
- scratch-script rows (`~/miniworld-engine-scratch/triattn_cs/cases.py`, `cases_anth.py`): the same module and
  `compile_module_for_benchmark`, one CUDA graph per call, fp32 master parameters under bf16 autocast (PyTorch / cuEquivariance pay a cast
  per call there, so their ratios are larger than in `bench.py`; ours agrees between the two within 2 %).

## Re-measurement on `h100/cuda-sweep` (2026-10-13)

Same harness (`benchmarks/runners/bench.py target=triangle_attention level=module cudagraph=manual compile=true`, bf16-mixed, B1, starting
node, no mask, `allow_tf32=true`; training `n_augment=48 dropout=0.25`), one H100, ms, CUDA graph.  "other" = the fastest of PyTorch compiled,
cuEquivariance and (inference, d_pair 128 / 256 only) Anthropic's native cell, measured in the same session (d_pair 384 L768 and the
training d_pair 128 / 384 / 512 baselines are the 2026-10-05 values above).

Inference, ours (x = other / ours):

| d_pair | L128 | L256 | L384 | L512 | L640 | L768 |
|---|---|---|---|---|---|---|
| 64 | 0.030 (2.0x) | 0.080 (1.9x) | 0.172 (2.1x) | 0.309 (2.3x) | 0.508 (2.5x) | 0.772 (2.6x) |
| 128 | 0.033 (1.6x) | 0.113 (1.5x) | 0.267 (1.5x) | 0.507 (1.4x) | 0.862 (1.35x) | 1.346 (1.32x) |
| 256 | 0.074 (1.03x) | 0.264 (1.17x) | 0.603 (1.27x) | 1.146 (1.21x) | 1.950 (1.19x) | 3.051 (1.14x) |
| 384 | 0.114 (1.7x) | 0.389 (2.1x) | 0.968 (2.1x) | 1.869 (2.2x) | 3.217 (2.3x) | 5.094 (2.3x) |
| 512 | 0.154 (1.6x) | 0.560 (2.0x) | 1.383 (2.0x) | 2.682 (2.1x) | 4.684 (2.2x) | 7.173 (2.5x) |

Training (forward + backward), ours vs the faster of PyTorch compiled / cuEquivariance: d_pair 64 1.6-5.2x, 128 1.9-4.0x, 256 2.0-5.7x,
384 1.8-3.3x, 512 1.7-3.2x.  Anthropic ships no training path.

What changed on the generic path (inference): `torch.addmm(x, ...)` stages x into its output (a device copy of the whole pair tensor, 6 % of
the call at d_pair 256) -- the residual now runs as one cublasLt GEMM with D != C (`lt_addmm`); at L <= 256 the bias projection joins
the q | k | v | g GEMM.  Core experiments (what bounds the attention core and what did not help) are in
`docs/records/triattn_h100_core_experiments_20261013.md`.

## Full re-measurement at `f2c4cda4` (2026-10-14)

Same harness as above (`bench.py target=triangle_attention level=module cudagraph=manual compile=true`, bf16-mixed, B1, starting node,
no mask, `allow_tf32=true`; training `n_augment=48 dropout=0.25`), one H100, ms, CUDA graph, one run of the grid.  × = the fastest of the
others / ours.  Anthropic: its native cell (`block:triattn_native`), inference, d_pair 128 / 256 only.  The cuEquivariance training runs
use the harness with the fp32 execution-consistency limit relaxed from 1e-4 to 1e-3 (its gradients come from atomics: 1.3e-4 - 2.5e-4
between replays); "-" = PyTorch out of memory.

Inference (PyTorch compiled / cuEquivariance / Anthropic / ours, ×):

| d_pair | L128 | L256 | L384 | L512 | L640 | L768 |
|---|---|---|---|---|---|---|
| 64 | 0.069 / 0.059 / - / **0.025** 2.36× | 0.245 / 0.151 / - / **0.064** 2.37× | 0.827 / 0.359 / - / **0.151** 2.37× | 1.506 / 0.729 / - / **0.281** 2.59× | 3.822 / 1.257 / - / **0.476** 2.64× | 6.575 / 2.002 / - / **0.736** 2.72× |
| 128 | 0.107 / 0.080 / 0.051 / **0.033** 1.57× | 0.482 / 0.265 / 0.170 / **0.112** 1.52× | 1.635 / 0.715 / 0.398 / **0.265** 1.50× | 3.260 / 1.416 / 0.729 / **0.507** 1.44× | 8.284 / 2.454 / 1.180 / **0.873** 1.35× | 12.984 / 3.962 / 1.766 / **1.356** 1.30× |
| 256 | 0.193 / 0.127 / 0.075 / **0.065** 1.16× | 1.171 / 0.547 / 0.309 / **0.256** 1.21× | 3.621 / 1.395 / 0.758 / **0.607** 1.25× | 7.853 / 2.758 / 1.391 / **1.153** 1.21× | 16.636 / 4.854 / 2.325 / **1.974** 1.18× | 25.910 / 7.825 / 3.498 / **3.070** 1.14× |
| 384 | 0.276 / 0.193 / - / **0.104** 1.86× | 2.011 / 0.826 / - / **0.387** 2.13× | 5.936 / 2.070 / - / **0.974** 2.13× | 16.980 / 4.106 / - / **1.840** 2.23× | 31.753 / 7.312 / - / **3.255** 2.25× | 55.181 / 11.931 / - / **5.129** 2.33× |
| 512 | 0.461 / 0.249 / - / **0.147** 1.69× | 3.154 / 1.091 / - / **0.555** 1.96× | 9.676 / 2.802 / - / **1.398** 2.00× | 22.779 / 5.599 / - / **2.725** 2.05× | 42.567 / 9.993 / - / **4.663** 2.14× | 74.078 / 18.160 / - / **7.359** 2.47× |

Training, forward + backward (PyTorch compiled / cuEquivariance / ours, ×):

| d_pair | L128 | L256 | L384 | L512 | L640 | L768 |
|---|---|---|---|---|---|---|
| 64 | 0.216 / 0.209 / **0.140** 1.50× | 0.810 / 0.657 / **0.327** 2.01× | 2.315 / 1.697 / **0.671** 2.53× | 4.567 / 3.444 / **1.191** 2.89× | 10.04 / 6.072 / **1.956** 3.11× | 11.88 / 9.836 / **2.932** 3.36× |
| 128 | 0.344 / 0.304 / **0.133** 2.29× | 1.519 / 1.218 / **0.407** 2.99× | 4.665 / 3.246 / **0.938** 3.46× | 9.234 / 6.642 / **1.781** 3.73× | 14.94 / 11.85 / **3.092** 3.83× | 23.52 / 19.41 / **5.036** 3.85× |
| 256 | 0.615 / 0.523 / **0.314** 1.66× | 3.437 / 2.397 / **0.987** 2.43× | 9.463 / 6.416 / **2.244** 2.86× | 19.93 / 13.10 / **4.317** 3.03× | 42.56 / 23.74 / **7.496** 3.17× | - / 40.21 / **11.85** 3.39× |
| 384 | 0.908 / 0.805 / **0.446** 1.80× | 5.847 / 3.530 / **1.449** 2.44× | 16.73 / 9.510 / **3.494** 2.72× | 21.19 / 19.81 / **6.869** 2.88× | 44.98 / 37.46 / **12.06** 3.11× | 71.29 / 64.79 / **19.52** 3.32× |
| 512 | 1.496 / 1.022 / **0.584** 1.75× | 10.07 / 4.685 / **2.041** 2.30× | 29.50 / 12.63 / **4.852** 2.60× | 72.23 / 26.99 / **9.643** 2.80× | - / 52.44 / **17.40** 3.01× | - / 88.59 / **27.95** 3.17× |

Every one of the 60 cells is faster than every other measured implementation; the smallest margin is d_pair 256 inference L128 (1.16×
over Anthropic).  Not covered by this grid: ending node, mask, B > 1, fp32 inputs.

### Ending node, key mask, B = 2 (2026-10-14)

`bench.py` only builds B = 1, starting node, so these cells come from a scratch module-level script
(`~/miniworld-engine-scratch/triattn_cs/cases.py`: the same `TriangleAttention` module, `compile_module_for_benchmark`, one CUDA graph of a
module call, median of 5 x 10 replays; fp32 master parameters under bf16 autocast).  Cases: ending node; starting node with a key mask
(20 % masked); starting node with B = 2.  Widths 64-512, L 128 / 256 / 512 / 768, inference and training (forward + autograd.grad of the
input and every parameter, dropout 0.25), PyTorch compiled and cuEquivariance.  Anthropic is not measured here.

Control, same script, starting node, B = 1, no mask: ours matches `bench.py` (d_pair 128 L768 inference 1.346 vs 1.356 ms, training 4.97
vs 5.04 ms), but the two baselines are slower than in `bench.py` (cuEquivariance 5.74 vs 3.96 ms, PyTorch 25.0 vs 13.0 ms: their fp32 master
parameters are cast every call).  The "conservative" column therefore rescales the `bench.py` baseline of the same cell by
(case time / control time) of the same implementation.

| case, mode | cells | smallest x (conservative) | at | smallest x (same script) |
|---|---|---|---|---|
| ending, inference | 20 | 1.67 | d_pair 512 L128 | 1.65 |
| ending, training | 20 | 1.55 | d_pair 64 L128 | 1.53 |
| mask, inference | 20 | 1.56 | d_pair 512 L128 | 1.64 |
| mask, training | 20 | 1.53 | d_pair 64 L128 | 1.43 |
| B = 2, inference | 20 | 1.65 | d_pair 512 L128 | 1.73 |
| B = 2, training | 20 | 1.74 | d_pair 64 L128 | 1.71 |

Ours is the fastest in all 120 cells; largest margins (same script) are d_pair 128 at L >= 256 (4.1-4.8x inference, 4.5-4.8x training).
PyTorch compiled runs out of memory at L768 for d_pair 512 (B = 1, 2) and d_pair 256-384 at B = 2 and for d_pair 384 training;
those cells are against cuEquivariance only.  Ending node and mask cost nothing over the starting node without a mask (ours within
+-3 % of the control).  Not covered: fp32 activations, B > 2, L not a multiple of 128, Anthropic with these conditions.

### Against Anthropic's native cell, all four conditions (2026-10-14)

Same scratch script as the previous section (`cases_anth.py`: `integrations.anthropic.module_triangle_attention`, row
`block:triattn_native`, on the module's own parameters, inference; ours = the module with the sm90 path; one CUDA graph per cell, median
of 5 x 10 replays; fp32 master parameters under bf16 autocast).  Both outputs checked against the fp32 PyTorch module (relative error
1.7e-3 - 1.9e-3 for both, every cell).  ms, Anthropic / ours (Anthropic ÷ ours):

| case, d_pair | L128 | L256 | L512 | L768 |
|---|---|---|---|---|
| base, 128 | 0.047 / 0.036 (1.31x) | 0.162 / 0.115 (1.41x) | 0.702 / 0.512 (1.37x) | 1.712 / 1.367 (1.25x) |
| base, 256 | 0.070 / 0.074 (**0.95x**) | 0.300 / 0.265 (1.13x) | 1.388 / 1.187 (1.17x) | 3.462 / 3.155 (1.10x) |
| ending, 128 | 0.050 / 0.035 (1.43x) | 0.176 / 0.115 (1.53x) | 0.735 / 0.513 (1.43x) | 1.792 / 1.374 (1.30x) |
| ending, 256 | 0.076 / 0.074 (1.03x) | 0.324 / 0.266 (1.22x) | 1.442 / 1.189 (1.21x) | 3.609 / 3.182 (1.13x) |
| mask, 128 | 0.048 / 0.036 (1.33x) | 0.165 / 0.116 (1.42x) | 0.722 / 0.511 (1.41x) | 1.754 / 1.376 (1.27x) |
| mask, 256 | 0.070 / 0.074 (**0.95x**) | 0.305 / 0.265 (1.15x) | 1.412 / 1.176 (1.20x) | 3.480 / 3.122 (1.11x) |
| B = 2, 128 | 0.105 / 0.057 (1.84x) | 0.411 / 0.212 (1.94x) | 1.735 / 1.001 (1.73x) | 4.233 / 2.713 (1.56x) |
| B = 2, 256 | 0.195 / 0.137 (1.42x) | 0.764 / 0.489 (1.56x) | 3.458 / 2.373 (1.46x) | out of memory (reference check) |

Anthropic's cell does not serve d_pair 64 (`no-cell:fpf:prologue:64x2x32`), 384 (`pow2:c384h12d32`) or 512 (`Refused:pair_fused`).
d_pair 256 at L128 is a tie and, in this script, a small loss (0.070 vs 0.074 ms; `bench.py` measured 0.075 vs 0.065 ms for the same
cell on the same day): that cell is three kernels (fused front `front_v3` 28 us, core 22 us, cuBLAS 10 us of device time) and its
latency differs by a few us between the two harnesses.  Everywhere else ours is faster (1.03x - 1.94x).

#### Correction after `7f7851aa`: the weight pack

The d_pair 256 L128 cell above (a tie / small loss) was the weight packing: the sm90 paths repack the weights inside a captured CUDA graph
so a replay sees the current parameters, and with fp32 master parameters the generic path did it with twelve small torch kernels
(cast, pad, cat: 17 us of a 74 us replay at d_pair 256 L128).  `pack_generic_k` (generic) and `tri_pack_k` (fused inference) do it in
one launch each (3 / 2 us).  Re-measured, same script, ms, Anthropic / ours (Anthropic ÷ ours):

| case, d_pair | L128 | L256 | L512 | L768 |
|---|---|---|---|---|
| base, 128 | 0.047 / 0.030 (1.57x) | 0.161 / 0.110 (1.46x) | 0.700 / 0.503 (1.39x) | 1.718 / 1.369 (1.25x) |
| base, 256 | 0.070 / 0.061 (1.15x) | 0.301 / 0.255 (1.18x) | 1.388 / 1.168 (1.19x) | 3.474 / 3.158 (1.10x) |
| ending, 128 | 0.050 / 0.030 (1.67x) | 0.176 / 0.110 (1.60x) | 0.739 / 0.506 (1.46x) | 1.801 / 1.363 (1.32x) |
| ending, 256 | 0.076 / 0.062 (1.23x) | 0.322 / 0.258 (1.25x) | 1.439 / 1.185 (1.21x) | 3.638 / 3.208 (1.13x) |
| mask, 128 | 0.048 / 0.030 (1.60x) | 0.166 / 0.110 (1.51x) | 0.720 / 0.504 (1.43x) | 1.746 / 1.368 (1.28x) |
| mask, 256 | 0.070 / 0.061 (1.15x) | 0.305 / 0.256 (1.19x) | 1.388 / 1.173 (1.18x) | 3.495 / 3.167 (1.10x) |
| B = 2, 128 | 0.104 / 0.051 (2.04x) | 0.410 / 0.206 (1.99x) | 1.744 / 1.001 (1.74x) | 4.238 / 2.724 (1.56x) |
| B = 2, 256 | 0.195 / 0.120 (1.63x) | 0.771 / 0.479 (1.61x) | 3.458 / 2.284 (1.51x) | out of memory (reference check) |

Ours is faster than Anthropic's native cell in every cell; the thinnest margin is d_pair 256 L768 (1.10x), where the cuBLAS projections
dominate.  The same repack saving applies to every width and to training calls (their weight packs were the same torch kernels).

## Full re-measurement at `ea15e364` (2026-10-15)

After the one-launch weight pack (`7f7851aa`).  Same `bench.py` harness and settings as the 2026-10-14 grid above (PyTorch compiled /
cuEquivariance / ours; cuEquivariance training with the fp32 consistency limit relaxed to 1e-3).  ours in ms (x = the faster of PyTorch
compiled / cuEquivariance ÷ ours):

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

All 60 cells faster than both baselines (smallest: 1.69x).  Ending node, key mask, B = 2 (scratch script, 160 cells over base / ending /
mask / B = 2, inference and training, d_pair 64-512, L 128 / 256 / 512 / 768): every cell faster than both baselines, smallest 1.69x
(training, d_pair 64, L128).  Against Anthropic's native cell (d_pair 128 / 256, inference, all four cases): see the table after
`7f7851aa`, every cell faster, smallest 1.10x.

### Against Anthropic's native cell, complete grid (2026-10-15)

After `7f7851aa`.  Inference, d_pair 128 and 256 (the only widths its cell serves), L128-768 (640 included), four cases (scratch script
`cases_anth.py`, one CUDA graph per cell; ms, Anthropic / ours, Anthropic ÷ ours).  `bench.py` (`implementations=[anthropic,miniworld]`,
B = 1, starting, no mask) gives the same picture: d_pair 128 L128-768 1.57 / 1.52 / 1.49 / 1.43 / 1.35 / 1.30x, d_pair 256
1.17 / 1.21 / 1.25 / 1.21 / 1.19 / 1.11x.

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

48 of 48 cells faster; the thinnest margin is d_pair 256 L768 (1.07x in this script, 1.11x in `bench.py`: the cuBLAS projections dominate).
Anthropic's cell does not run d_pair 64 (`no-cell:fpf:prologue:64x2x32`), d_pair 384 (`pow2:c384h12d32`) or d_pair 512
(`Refused: pair_fused:epilogue`, Triton 3.7 PTXAS error), and ships no training path: no comparison is possible there.
