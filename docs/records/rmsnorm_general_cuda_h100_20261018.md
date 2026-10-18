# General CUDA RMSNorm on H100 (2026-10-18)

`kernels/rmsnorm/cuda`: the general LayerNorm kernel (`kernels/layernorm/cuda/general`,
[record](layernorm_general_cuda_h100_20261018.md)) instantiated with the mean and the bias dropped (a template flag; one extension
serves both). Any width (1 .. 64K bf16 / fp16, 32K fp32), any row count, bf16 / fp16 / fp32, optional weight, forward and backward with
dw reduced without atomics (bit-identical replays). It replaces the PyTorch statements `modules.RMSNorm` (MINIWORLD) ran on H100 / B200,
where the engine has no Triton fallback: the q/k norms of AttentionPairBias and AugmentedAttention and the SWA atom block norms.

H100 SXM, bf16 activations, fp32 weight (or none), eps = fp32 epsilon, CUDA-graph replay, best of 5 x 30, ms. "eager" is the former
PyTorch leg (`F.rms_norm`, cast to the input dtype); "compile" is `torch.compile` of the same function.

| shape [rows, width] | config (tpr, chunks, vec) | fwd ours | eager | compile | fwd+bwd ours | eager | compile |
|---|---|---|---|---|---|---|---|
| q/k norm, pair-bias attention L384 [384·16·8·4, 48], w | 8, 1, 8 | 0.013 | 0.191 | 0.012 | 0.042 | 0.643 | 0.049 |
| q/k norm, triangle attention L384 [384·384·4, 32], w | 4, 1, 8 | 0.029 | 0.409 | 0.180 | 0.076 | 1.297 | 0.451 |
| q/k norm, token d_head 64 [8192·16, 64], w | 8, 1, 8 | 0.010 | 0.159 | 0.043 | 0.039 | 0.559 | 0.163 |
| SWA atom norm [4096·4, 128], no w | 16, 1, 8 | 0.004 | 0.013 | 0.008 | 0.008 | 0.024 | 0.015 |
| SWA atom norm [4096·48, 128], no w | 16, 1, 8 | 0.037 | 0.124 | 0.062 | 0.091 | 0.252 | 0.136 |
| [2^20, 32], w | 4, 1, 8 | 0.048 | 0.712 | 0.318 | 0.128 | 2.234 | 0.787 |
| [2^18, 128], w | 16, 1, 8 | 0.049 | 0.553 | 0.083 | 0.126 | 2.040 | 0.489 |
| [2^17, 384], w | 64, 1, 8 | 0.081 | 0.796 | 0.096 | 0.193 | 2.774 | 0.682 |
| [2^16, 768], w | 128, 1, 8 | 0.080 | 0.797 | 0.100 | 0.192 | 2.748 | 0.694 |
| [2^14, 2048], w | 128, 2, 8 | 0.051 | 0.541 | 0.101 | 0.130 | 1.855 | 0.763 |
| [2^12, 8192], w | 256, 4, 8 | 0.050 | 0.538 | 0.158 | 0.145 | 1.841 | 0.347 |

Forward at ~33M elements moves ~2.7 TB/s, as the LayerNorm. Every row beats eager (fwd 3-16x, fwd+bwd 2.8-17x) and torch.compile in
fwd+bwd (1.2-6x); forward is 1 us behind torch.compile on the smallest q/k-norm shape (0.013 vs 0.012 ms, 4.7M elements) and beats
it elsewhere (1.2-6x).

Correctness (`tests/gpu/any/test_rmsnorm_general_cuda_gpu.py`): 17 widths 1..16384 x bf16 / fp16 / fp32, weighted and unweighted,
against fp64 `F.rms_norm`; bf16 weight gradient dtype; bit-identical replays; CUDA-graph capture; `torch.compile(fullgraph=True)`;
`modules.RMSNorm` routing to it on H100 with matching values and gradients. Not run on a B200.
