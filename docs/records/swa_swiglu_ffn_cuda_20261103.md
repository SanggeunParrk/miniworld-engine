# SWA SwiGLUFFN on H100 runs the hand-CUDA `swiglu_ffn` (2026-11-03)

Branch `green/engine-a`. H100 80GB HBM3 (node01), CUDA graph, warm, median of 30 replays of a graph holding 5 calls (module rows) or 20 calls (kernel rows).

## What changed

`modules.swa_atom_transformer.SwiGLUFFN` and `modules.swa_dit.SwiGLUFFN` ran the PyTorch statements on H100 (`restricted_card`: no Triton there, and the
module had no CUDA branch). They now call `kernels.swiglu_ffn` (`kernels/transition/interface.py` -> `kernels/conditioned_transition/cuda/swiglu_ffn.py`):
the fused expand + SwiGLU GEMM at inference, cuBLAS + the SwiGLU row kernels in training. Only bf16 is routed (the registry's `swiglu_ffn` rows declare
bfloat16 only); fp32 keeps the PyTorch statements, as does any call the kernel does not serve (it returns the PyTorch statements itself). Other cards
and `ImplementationType.PYTORCH` are unchanged.

Numerics: `tests/gpu/sm90/test_swa_swiglu_ffn_cuda_gpu.py` (6 passed). Both modules, d_model 128 / hidden 256, 5 x 1024 inference and 48 x 1024 training:
the CUDA kernel is called, and the output, dx and both weight gradients are within 1.5x of the bf16 PyTorch module's error against an fp64 run of the
same weights. fp32 is bitwise equal to the PyTorch module and calls no CUDA kernel.

## Before / after: the modules (bf16, d_model 128, hidden 256, registry atom_single)

Before = the PyTorch statements (what H100 ran); after = the CUDA route. `us` per call.

| module | A x atoms | mode | before PyTorch (us) | after CUDA (us) | before/after |
|---|---|---|---|---|---|
| atom | 5 x 1024 | infer | 20.4 | 18.2 | 1.12 |
| atom | 5 x 1024 | fwd+bwd | 74.4 | 55.1 | 1.35 |
| atom | 5 x 4096 | infer | 52.1 | 34.2 | 1.52 |
| atom | 5 x 4096 | fwd+bwd | 165.1 | 108.9 | 1.52 |
| atom | 5 x 8192 | infer | 101.8 | 53.8 | 1.89 |
| atom | 5 x 8192 | fwd+bwd | 315.6 | 196.3 | 1.61 |
| atom | 48 x 1024 | infer | 122.1 | 61.5 | 1.98 |
| atom | 48 x 1024 | fwd+bwd | 373.1 | 228.6 | 1.63 |
| dit | 5 x 1024 | infer | 21.5 | 19.0 | 1.13 |
| dit | 5 x 1024 | fwd+bwd | 73.3 | 54.2 | 1.35 |
| dit | 5 x 4096 | infer | 52.1 | 34.7 | 1.50 |
| dit | 5 x 4096 | fwd+bwd | 167.0 | 109.9 | 1.52 |
| dit | 5 x 8192 | infer | 102.6 | 54.2 | 1.89 |
| dit | 5 x 8192 | fwd+bwd | 315.7 | 196.7 | 1.61 |
| dit | 48 x 1024 | infer | 122.9 | 61.8 | 1.99 |
| dit | 48 x 1024 | fwd+bwd | 373.6 | 229.9 | 1.63 |

(`atom` = `swa_atom_transformer`, `dit` = `swa_dit`; the two modules share the arithmetic and the numbers.) Training rows at 48 x 1024 only; 48 x 4096+ not run.

## Token D768 bf16 inference was slower than PyTorch: found and fixed

`kernels.swiglu_ffn` at bf16 inference always took the fused `ct_inference` kernel. Measured per route (bf16, no grad, rows = tokens, CUDA graph):

| shape | rows | PyTorch (us) | `ct_inference` (us) | cuBLAS + row kernels (us) | public `swiglu_ffn` after the fix (us) | PyTorch / public |
|---|---|---|---|---|---|---|
| token D768 h1536 | 128 | 14.7 | 20.0 | 13.9 | 13.9 | 1.06 |
| token D768 h1536 | 256 | 16.4 | 20.1 | 14.8 | 14.8 | 1.11 |
| token D768 h1536 | 384 | 18.3 | 20.1 | 16.3 | 16.3 | 1.13 |
| token D768 h1536 | 512 | 19.1 | 20.4 | 17.4 | 17.4 | 1.09 |
| token D768 h1536 | 768 | 24.9 | 25.9 | 21.5 | 21.5 | 1.16 |
| token D768 h1536 | 1024 | 23.9 | 26.3 | 22.6 | 23.1 | 1.03 |
| token D768 h1536 | 16384 | 273.3 | 314.5 | 243.2 | 244.1 | 1.12 |
| token D768 h1536 | 65536 | 1074.6 | 1331.9 | 953.2 | 954.7 | 1.13 |
| token D384 h768 | 128 | 14.6 | 14.4 | 11.4 | 11.0 | 1.32 |
| token D384 h768 | 512 | 14.1 | 13.1 | 11.8 | 11.8 | 1.19 |
| token D384 h768 | 1024 | 15.3 | 13.6 | 13.3 | 13.2 | 1.16 |
| token D384 h768 | 16384 | 101.9 | 87.0 | 86.2 | 86.4 | 1.18 |
| token D384 h768 | 65536 | 386.0 | 411.9 | 350.8 | 324.7 | 1.19 |
| pair D128 h256 | 128 - 1024 | 11.4 - 12.0 | 8.5 - 9.1 | 9.1 - 9.7 | 8.5 - 8.8 | 1.32 - 1.37 |
| pair D128 h256 | 16384 / 65536 | 30.7 / 119.1 | 20.1 / 69.7 | 23.9 / 91.0 | 20.0 / 69.5 | 1.53 / 1.71 |
| pair D128 h512 | 128 - 1024 | 11.2 - 13.0 | 9.4 - 9.7 | 9.2 - 10.8 | 9.4 - 9.7 | 1.19 - 1.34 |
| pair D128 h512 | 16384 / 65536 | 50.4 / 218.0 | 31.2 / 125.7 | 42.0 / 167.3 | 31.5 / 126.0 | 1.60 / 1.73 |

Before the fix the "public" column equalled the `ct_inference` column: at hidden 1536 the fused kernel was 1.04 - 1.37x slower than PyTorch (first measurement of
this job: PyTorch / public 0.73 - 0.96), at hidden 768 it was within 0.92 - 1.18. The fused expand + SwiGLU GEMM writes a hidden activation that is large next to
the GEMM for hidden > 512, and cuBLAS + the SwiGLU row kernel (already the training route) is faster there.

Fix (`kernels/conditioned_transition/cuda/swiglu_ffn.py`, `_CT_MAX_HIDDEN_BF16 = 512`): bf16 inference with hidden > 512 takes the cuBLAS + row-kernel route; hidden <= 512
(pair D128, atom D128) keeps `ct_inference` (1.2 - 1.7x faster than PyTorch). The token D768 rows are now 1.03 - 1.16x faster than PyTorch. fp32 (TF32) is unchanged.
No new kernel and no new shape: both routes were already in the file.
