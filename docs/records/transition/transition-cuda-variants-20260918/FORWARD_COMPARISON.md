# Transition forward comparison

Measured on node02 H100, B1 pair input [1,L,L,D], expansion4, nonzero squeeze, BF16 activations/weights and FP32 norm affine. All arms use static compile + manual CUDA Graph, two captures per row. Forward-only/inference, no optimizer. General Transition has no dropout.

**Old Triton means the retained split algorithm rerun on the current checkout/runtime, not a restored historical commit.** Current Triton uses full-K b2b at D128/256 and split at D384/512. New CUDA is the faster measured streamed/full-K native variant at each shape, not the existing legacy H100 auto backend. PyTorch is the official module harness reference, compiled and graphed too. PyTorch runs followed the backend runs; they were not a single interleaved experiment. Near-unity differences should be treated cautiously.

| D | L | PyTorch ms | Old Triton split ms | Current Triton ms | New CUDA ms | CUDA variant | Current Triton / PyTorch speedup | Current Triton / old speedup | New CUDA / PyTorch speedup | New CUDA / old speedup |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 128 | 384 | 0.4343 | 0.2755 | 0.1659 | 0.2108 | streamed_k | 2.618x | 1.660x | 2.061x | 1.307x |
| 128 | 768 | 1.6520 | 1.0324 | 0.5939 | 0.7734 | streamed_k | 2.781x | 1.738x | 2.136x | 1.335x |
| 256 | 384 | 0.8447 | 0.7078 | 0.5651 | 0.6465 | full_k | 1.495x | 1.252x | 1.307x | 1.095x |
| 256 | 768 | 3.2776 | 2.8965 | 2.1154 | 2.4305 | full_k | 1.549x | 1.369x | 1.349x | 1.192x |
| 384 | 384 | 1.4372 | 1.4128 | 1.4128 | 1.6431 | full_k | 1.017x | 1.000x | 0.875x | 0.860x |
| 384 | 768 | 5.7091 | 5.7812 | 5.7812 | 6.3429 | full_k | 0.988x | 1.000x | 0.900x | 0.911x |
| 512 | 384 | 2.1787 | 2.3445 | 2.3445 | 3.3561 | full_k | 0.929x | 1.000x | 0.649x | 0.699x |
| 512 | 768 | 8.7894 | 9.2645 | 9.2645 | 12.8118 | full_k | 0.949x | 1.000x | 0.686x | 0.723x |

Speedup = baseline time / implementation time; below 1 means slower. The new CUDA variants pass numerical/graph checks but are not faster than the current Triton choices here. They are explicit experimental module options, not promoted to automatic dispatch.

See [RESULTS.md](RESULTS.md) for separate CUDA variants, full training, forward/backward breakdown and validation; [CONFIGS.md](CONFIGS.md) for sampled winners and limits.
