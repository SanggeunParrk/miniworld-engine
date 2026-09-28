# Transition: PyTorch compiled versus installed native CUDA

H100 SXM, BF16, B1, input `[1,L,L,D]`, expansion 4. Full fresh forward +
backward, input and all five parameter gradients, residual included, no dropout.
Both actual modules are compiled with fullgraph=True and measured in manual CUDA graphs.
75 interleaved CUDA-event samples per implementation, medians below.

| L | D | PyTorch (ms) | Native (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 128 | 1.192 | 0.529 | 2.25x |
| 384 | 256 | 2.312 | 2.058 | 1.12x |
| 384 | 384 | 4.138 | 3.699 | 1.12x |
| 384 | 512 | 6.274 | 5.892 | 1.06x |
| 768 | 128 | 4.502 | 2.023 | 2.23x |
| 768 | 256 | 8.973 | 8.170 | 1.10x |
| 768 | 384 | 16.187 | 14.468 | 1.12x |
| 768 | 512 | 24.756 | 23.534 | 1.05x |

## Optional saved activation

`MINIWORLD_TRANSITION_WIDE_SAVE_H=1` reuses the forward SwiGLU activation at
D384/512. It is already implemented and remains opt-in. Additional retained
activation is per layer; activation checkpointing/layer count affect the total.

| L | D | Default F+B (ms) | Save-h F+B (ms) | Gain | Retained h (MiB/layer) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 384 | 384 | 3.699 | 3.556 | 3.9% | 432 |
| 384 | 512 | 5.892 | 5.831 | 1.0% | 576 |
| 768 | 384 | 14.468 | 13.954 | 3.6% | 1728 |
| 768 | 512 | 23.534 | 23.055 | 2.0% | 2304 |

## Qualification

- Native module dispatch observed directly, then CUDA kernel names recorded from compiled graph replay.
- Nonzero randomized weights and nontrivial FP32 LN affine parameters.
- Full output/input/all-parameter gradients pass the existing native-versus-Triton numerical gates.
- Save-h passes the existing native-versus-native gates, without tolerance changes.
- Compiled CUDA graphs agree with eager native execution and fresh compiled PyTorch execution.
- Input, every weight, LN affine and upstream gradient changed in place; all graph replays match fresh calls.
- Source hashes and raw sample lists are in each JSON. Sanitizer/pytest evidence is recorded separately in README.md.

PyTorch is the performance baseline. Engine Triton is only a numerical-contract check.
PyTorch casts LN affine parameters to BF16; native retains FP32 affine math, so the two paths
are numerically close rather than bit-identical. This table does not claim SOL or NCU results.

D128 uses the remeasured input-slot barrier fix (v2). The other six cells use
the unchanged wide native kernels (v1). See README.md for the original
race-instrumented gradient failure and the fix qualification.
