# General CUDA LayerNorm on H100 (2026-10-18)

`kernels/layernorm/cuda/general`: one CUDA kernel family for LayerNorm over the last dimension at any width (1 .. 64K bf16 / fp16,
32K fp32) and any row count, fp32 statistics, optional fp32 weight / bias, forward and backward (dx; dw / db as per-CTA partials plus a
fixed-order column pass, no atomics, so replays are bit-identical). It replaces the PyTorch statements that `modules.LayerNorm`
(MINIWORLD) and `architecture.primitives.LayerNorm` (MINIWORLD_ENGINE) ran on H100 / B200, where the engine has no Triton fallback.

H100 SXM, bf16 activations, fp32 affine, CUDA-graph replay, best of 5 x 30, ms. "eager" is the engine's former PyTorch leg
(`F.layer_norm` on the fp32 upcast, cast back); "compile" is `torch.compile` of the same function.

| shape [rows, width] | config (tpr, chunks, vec) | fwd ours | eager | compile | fwd+bwd ours | eager | compile |
|---|---|---|---|---|---|---|---|
| MiniWorld template `ln_out` [4·384², 64] | 8, 1, 8 | 0.054 | 1.075 | 0.054 | 0.146 | 1.921 | 0.173 |
| MiniWorld template `ln_query` / pair recycle [384², 128] | 16, 1, 8 | 0.029 | 0.321 | 0.095 | 0.080 | 0.643 | 0.242 |
| [2^20, 32] | 4, 1, 8 | 0.049 | 1.736 | 0.504 | 0.134 | 2.925 | 1.163 |
| [2^18, 128] | 16, 1, 8 | 0.049 | 0.564 | 0.166 | 0.132 | 1.115 | 0.411 |
| [2^17, 256] | 32, 1, 8 | 0.049 | 0.401 | 0.122 | 0.130 | 0.854 | 0.347 |
| [2^17, 384] | 64, 1, 8 | 0.084 | 0.515 | 0.161 | 0.213 | 1.146 | 0.481 |
| [2^16, 768] | 128, 1, 8 | 0.089 | 0.442 | 0.142 | 0.218 | 1.182 | 0.475 |
| [2^15, 1000] | 128, 1, 8 | 0.055 | 0.279 | 0.094 | 0.137 | 0.704 | 0.356 |
| [2^14, 2048] | 128, 2, 8 | 0.054 | 0.276 | 0.105 | 0.138 | 0.752 | 0.439 |
| [2^13, 4096] | 128, 4, 8 | 0.055 | 0.306 | 0.114 | 0.147 | 0.786 | 0.591 |
| [2^11, 16384] | 512, 4, 8 | 0.068 | 0.345 | 0.251 | 0.249 | 0.828 | 0.382 |

Forward moves ~2.7 TB/s and backward ~2.5 TB/s (75-82 % of HBM3 peak). Two findings shaped the kernel:

* The first version read the fp32 weight / bias per row with scalar loads; at TPR >= 32 lanes those strided loads cost many L1
  wavefronts each and capped the kernel at ~1.5 TB/s. Holding each thread's affine values in registers (loaded once, 16-byte vectors)
  lifted every width to the table above.
* Config sweep at ~33M elements: 128 threads per row x 2-4 vectors of 8 bf16 beat 256-1024 threads x 1 vector by 10-40 % at widths
  2048-8192 (fwd and bwd), and 8 vectors per thread lose (register pressure). `config()` picks the fewest idle lanes, <= 32 values per
  thread, TPR closest to 128.

Correctness (`tests/gpu/any/test_layernorm_general_cuda_gpu.py`): 19 widths 1..16384 x bf16 / fp16 / fp32 x affine variants against
fp64, bit-identical replays, CUDA-graph capture, `torch.compile(fullgraph=True)`, and both engine LayerNorm classes routing to it on
H100 with matching values and gradients. Not run on a B200 (no B200 on this cluster); the extension builds for sm_80 / sm_90 / sm_100.
