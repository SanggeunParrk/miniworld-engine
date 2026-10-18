# Bias-only token DiT on H100: superseded measurements and studies (2026-10-14)

Moved verbatim from [docs/gpus/h100/bias_only_dit/bias_only_dit.md](../gpus/h100/bias_only_dit/bias_only_dit.md) on 2026-10-18. The
page keeps the 2026-10-15 length sweep and head-layout tables.

## Measurements (2026-10-14)

H100 80GB HBM3 (cssb node01), torch 2.13.0+cu129, `benchmarks/runners/bench.py target=bias_only_dit level=module`, CUDA graph,
key mask 20 % (inference) / 12.5 % (training), ms, median. Accuracy columns are against an fp32 IEEE PyTorch block.

```
python benchmarks/runners/bench.py target=bias_only_dit level=module mode=inference precision=bf16-mixed cudagraph=manual \
    min_seq_len=384 max_seq_len=768 seq_len_step=384 'implementations=[pytorch,miniworld]' +shared_cond=true mask_prob=0.2
# fp32: precision=32; training: mode=training cudagraph=manual mask_prob=0.125
```

### Inference (S = 5, one conditioning shared by the samples)

| (Length, Dimension, dtype) | PyTorch compiled | ours | × | output rel. error: PyTorch / ours |
|---|---|---|---|---|
| (384, 768, bf16) | 0.208 | **0.090** | 2.3 | 3.9e-3 / 3.1e-3 |
| (768, 768, bf16) | 0.425 | **0.145** | 2.9 | 3.9e-3 / 3.1e-3 |
| (384, 768, fp32) | 0.367 | **0.163** | 2.3 | -- / 2.9e-4 |
| (768, 768, fp32) | 0.777 | **0.299** | 2.6 | -- / 2.9e-4 |

One conditioning per sample (bf16): L384 0.104, L768 0.179 ms (PyTorch 0.212 / 0.432).

### Training (A = 48)

| (Length, Dimension, dtype) | PyTorch compiled | ours | × |
|---|---|---|---|
| (384, 768, bf16) | 2.516 | **2.134** | 1.18 |
| (768, 768, bf16) | 4.949 | **4.314** | 1.15 |
| (384, 768, fp32) | 5.655 | **5.102** | 1.11 |
| (768, 768, fp32) | 11.073 | **10.284** | 1.08 |

Gradients (output, input, every parameter) against the fp32 IEEE block are within the PyTorch TF32 / bf16 error of the same
recipe (`tests/gpu/sm90/test_h100_bias_only_dit_gpu.py`), and a training step is bitwise repeatable.

### Where the time goes (bf16 training L768 A48, kernel sum 4.3 ms; `ncu`, % of the roofline = max(DRAM, tensor))

| | share | roofline | |
|---|---|---|---|
| cuBLAS GEMMs (forward, data and weight gradients) | 52 % | 64-91 % tensor | at its practical rate |
| row passes (LN / AdaLN / residual / gate / SwiGLU, fwd + bwd) | 39 % | 82-90 % DRAM | one read and one write of each tensor |
| `pvx` x 2, `dpb2`, `softmax_t`, pair bias fwd + bwd | 9 % | 45-80 % | `pvx` ~46 % |

The step is at ~75 % of the time-weighted roofline. What is left is the cuBLAS GEMMs and one HBM round trip per row pass;
removing those needs fusions with the GEMM epilogues, which were measured and lost on this card (below).
