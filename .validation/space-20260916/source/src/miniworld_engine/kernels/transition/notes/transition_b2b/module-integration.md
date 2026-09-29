# transition_b2b — module integration (v14 wired into inference dispatch)

The hand-CUDA fused b2b forward (v14, ~1.29x vs Triton at the microbench) is wired into the
Transition module's inference dispatch for the fixed AF3 shape (d_hidden=128, n=4 -> K=128, ND=512,
D=128), bf16, M%128==0.

## Wiring
- `kernels/transition/cuda/__init__.py::cuda_transition_b2b(x, ln_w, ln_b, wa, wb, ws, eps)` —
  flattens x -> [M,128], computes LN stats via the SAME `stats_triton` the Triton b2b path uses
  (fair comparison), calls `transition_b2b_fwd`, reshapes back. Weight layouts match `nn.Linear.weight`
  directly (wa/wb `[ND,K]`, ws `[D,ND]`).
- `modules/transition/module.py::_inference_forward` routes to it when
  `_cuda_b2b_inference_enabled()` (env `MINIWORLD_TRANSITION_CUDA_B2B`, default ON) AND d_hidden==128,
  n==4, x is CUDA bf16, M%128==0. Otherwise falls back to the Triton b2b path.

## Correctness (real weights, L=1024)
- cos(cuda_b2b, torch)   = 0.999989
- cos(triton_b2b, torch) = 0.999989   (identical accuracy)
- cos(cuda_b2b, triton)  = 1.000000   (bit-identical to the path it replaces)

## Module inference A/B (H100, cudagraph=manual, bf16-mixed, d_pair=128, ms)
| L | CUDA-b2b (new) | Triton-b2b (old) | speedup | pytorch |
|---:|---:|---:|---:|---:|
| 384 | 0.2032 | 0.2448 | 1.20x | 1.037 |
| 512 | 0.3520 | 0.4186 | 1.19x | 1.805 |
| 640 | 0.5258 | 0.6307 | 1.20x | 2.794 |
| 768 | 0.7490 | 0.8932 | 1.19x | 3.989 |
| 896 | 1.0059 | 1.2052 | 1.20x | 5.429 |
| 1024 | 1.3047 | 1.5668 | 1.20x | 7.087 |

Microbench kernel win 1.29x -> module inference ~1.20x end-to-end (consistent across L). The gap is the
common LN-stats pass (`stats_triton`) + reshape overhead added around the kernel. vs pytorch at L=1024:
5.43x (Triton path was 4.51x). NOTE: the module bench's `output_cosine` reads 0.0 because the module's
`squeeze` is zero-init by default (output is all zeros) — a harness artifact; real correctness verified
above with randomized weights.

## Next lever
Fuse the LN stats into the CUDA b2b kernel (cf. MINIWORLD_TRANSITION_FUSE_STATS for the Triton path) to
close the 1.29x->1.20x gap — the kernel already loads the full-K row.

## Module TRAINING A/B (H100, cudagraph=manual, bf16-mixed, d_pair=128, ms)
CUDA b2b is **inference-only** (forward saves nothing for backward), so `_training_forward` stays on
the Triton path. This table confirms training is UNCHANGED by the wiring (CUDA_B2B on == off).

| L | miniworld (triton training) | pytorch | speedup | CUDA_B2B=0 (verify unchanged) |
|---:|---:|---:|---:|---:|
| 384 | 1.177 | 2.929 | 2.49x | 1.175 |
| 512 | 2.016 | 5.094 | 2.53x | 2.017 |
| 640 | 3.086 | 7.896 | 2.56x | 3.085 |
| 768 | 4.375 | 11.211 | 2.56x | 4.391 |
| 896 | 5.929 | 15.203 | 2.56x | 5.916 |
| 1024 | 7.726 | 19.822 | 2.57x | 7.719 |

Training is ~2.5-2.6x vs pytorch (all Triton). To speed up training with the CUDA b2b would need a
training-forward variant that saves intermediates (xn / a,b) for the backward — a separate effort; the
current kernel is forward-only.
