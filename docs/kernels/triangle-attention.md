# triangle attention

This document records the MiniWorld full triangular self-attention benchmark
path. It covers `TriangleAttention(use_self_attention=True)`: LayerNorm,
query/key/value/bias/gate projections, Triton pair-bias attention, and output
gating. The bias-only case is tracked separately in
`docs/kernels/bias-only-attention.md`.

## Scope

- Module target: `triangle_attention`
- Benchmark config: `benchmarks/modules/triangle_attention/configs/bench.yaml`
- Runner: `benchmarks/runners/bench.py`
- Run: `python benchmarks/runners/bench.py kernel=triangle_attention`
- Kernel: `src/miniworld_engine/kernels/triangle_attention/triton/main.py`
- Artifacts: `benchmarks/modules/triangle_attention/artifacts/`

Implementations in the benchmark:

- `pytorch`: compiled module reference
- `cuequivariance`: cuEquivariance triangle attention path
- `miniworld`: MiniWorld module path using the canonical Triton pair-bias
  attention kernel plus repo LayerNorm/gate dispatch

The benchmark uses `compile=true`, bf16 mixed precision, `mask_prob=0.0`,
`n_layers=1`, `L in {384, 512, 640, 768, 896, 1024}`, and
`d_pair in {128, 256, 512}` at fixed `L=384` for d sweeps.

## CUDA graph regime

The primary comparison is `compile=true + cudagraph=manual`. Full triangle
attention is a composed module with several launches around the core attention
kernel, so manual CUDA graph capture is the steady-state regime used for final
comparison.

`compile=true + cudagraph=disabled` is still useful as a diagnostic. It shows
where launch overhead and surrounding projection/LayerNorm/gate work remain
visible, especially for large `d_pair` training.

## H100 full benchmark: job 10273

Run:

```bash
sbatch --export=ALL,BENCH_TARGET=triangle_attention \
  --output=benchmarks/modules/triangle_attention/artifacts/triangle_attention_%j.out \
  submits/run_bench.sbatch
```

Result log:

```text
benchmarks/modules/triangle_attention/artifacts/triangle_attention_10273.out
MODULE BENCH DONE target=triangle_attention
```

One expected stress point appeared: PyTorch failed with OOM for manual training
at `L=1024, d_pair=128`; cuEquivariance and MiniWorld completed that point.

### Manual CUDA graph results

Inference L sweep, fixed `d_pair=128`:

| L | PyTorch | cuEquivariance | MiniWorld | speedup vs best baseline |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 3.729 ms | 1.441 ms | 0.700 ms | 2.06x |
| 512 | 8.135 ms | 2.677 ms | 1.308 ms | 2.05x |
| 640 | 15.119 ms | 4.414 ms | 2.161 ms | 2.04x |
| 768 | 25.329 ms | 6.767 ms | 3.317 ms | 2.04x |
| 896 | 39.101 ms | 9.932 ms | 4.791 ms | 2.07x |
| 1024 | 56.743 ms | 13.534 ms | 6.671 ms | 2.03x |

Training L sweep, fixed `d_pair=128`:

| L | best baseline | MiniWorld | speedup |
| ---: | ---: | ---: | ---: |
| 384 | 5.025 ms | 3.318 ms | 1.51x |
| 512 | 9.786 ms | 6.525 ms | 1.50x |
| 640 | 16.680 ms | 11.628 ms | 1.44x |
| 768 | 26.341 ms | 18.113 ms | 1.45x |
| 896 | 39.145 ms | 26.914 ms | 1.45x |
| 1024 | 56.031 ms | 37.387 ms | 1.50x |

Initial d sweep at `L=384` was weaker than the L sweep:

| mode | d_pair | MiniWorld | best baseline | speedup |
| --- | ---: | ---: | ---: | ---: |
| inference | 128 | 0.701 ms | 1.440 ms | 2.06x |
| inference | 256 | 1.286 ms | 2.381 ms | 1.85x |
| inference | 512 | 2.500 ms | 4.283 ms | 1.71x |
| training | 128 | 3.311 ms | 5.031 ms | 1.52x |
| training | 256 | 6.352 ms | 8.150 ms | 1.28x |
| training | 512 | 13.830 ms | 14.735 ms | 1.07x |

## d sweep tile tuning

The initial core attention autotune space was too narrow:

- forward had only two `64x64` candidates
- backward had only two candidates: `64x128` and `64x256`

This particularly hurt training at larger `d_pair`. The tuning change expanded
the candidate sets:

- forward: 2 -> 8 candidates, adding `32x64`, `64x128`, and `128x64`
- backward: 2 -> 8 candidates, adding `32x64`, `32x128`, `32x256`,
  `64x64`, `128x64`, and `128x128`

Validation runs:

```text
benchmarks/modules/triangle_attention/artifacts/triangle_dinf_tune_10277.out
benchmarks/modules/triangle_attention/artifacts/triangle_dtrain_tune_10276.out
benchmarks/modules/triangle_attention/artifacts/triangle_cdinf_tune_10278.out
benchmarks/modules/triangle_attention/artifacts/triangle_cdtrain_tune_10279.out
```

The tuned manual training d sweep improved materially:

| d_pair | before | after | best baseline | speedup after |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 3.311 ms | 3.303 ms | 5.022 ms | 1.52x |
| 256 | 6.352 ms | 5.570 ms | 8.160 ms | 1.47x |
| 512 | 13.830 ms | 11.568 ms | 14.733 ms | 1.27x |

The selected backward tiles after tuning at `L=384` were:

| d_pair | head_dim | selected backward tile |
| ---: | ---: | --- |
| 128 | 32 | `BLOCK_M=64, BLOCK_N=128, num_warps=4, num_stages=3` |
| 256 | 64 | `BLOCK_M=32, BLOCK_N=256, num_warps=8, num_stages=3` |
| 512 | 128 | `BLOCK_M=32, BLOCK_N=64, num_warps=4, num_stages=3` |

Compile-only d sweep also improved for training, but large `d_pair` still has
work left:

| d_pair | before | after | best baseline | speedup after |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 3.387 ms | 3.394 ms | 3.670 ms | 1.08x |
| 256 | 6.427 ms | 5.677 ms | 5.696 ms | 1.00x |
| 512 | 13.896 ms | 11.621 ms | 8.397 ms | 0.72x |

So the d sweep issue was real and mostly a backward tiling issue in the primary
manual CUDA graph regime. The remaining `compile=true, cudagraph=disabled,
d_pair=512` regression is broader than the attention tile itself: projection,
LayerNorm, gate/output, launch overhead, and backward reductions all remain
visible there.
