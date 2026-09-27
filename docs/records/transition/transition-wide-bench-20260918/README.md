# Transition D384/512: current Triton versus H100 — 2026-09-18

## Scope and method

D768 is excluded from new development and measurement plans at the user's request.
Existing support guards are unchanged. L768 is an input length, not channel width D768.

Measurements ran on **node02 only**, allocation 13274, two exclusive GPUs with one
width per GPU. Both steps completed successfully; the allocation was released.
Official `bench_module_transition` fixture, B=1, pair input `[1,L,L,D]`, n=4,
one layer, BF16, deterministic nonzero squeeze weights, residual enabled.
Static compile (`dynamic=False`, partial allowed, one observed graph per row),
manual CUDA graph. Training is forward+backward, without optimizer.
The general Transition has no dropout. Each cell is the median of two independent
captures, reversing backend order in the second pass.

This is the current **auto H100 route versus forced Triton**, not an exhaustive
config search. Triton cache misses search a heuristic subset of 24 candidates;
native/CuTe misses use the declared default. Warnings and raw samples are retained.
No kernel, dispatch default, installed package, or remote repository was changed.

## Results

See [timings](timings.md), [summary and individual samples](summary.json), and the
four `bench-L*-D*-auto.json` files alongside this record.

| L | D | Inference Triton / H100 (ms) | Speedup | Training Triton / H100 (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 384 | 384 | 1.4211 / 1.3458 | 1.056x | 4.6973 / 4.7674 | 0.985x |
| 768 | 384 | 5.8030 / 5.4502 | 1.065x | 19.0179 / 19.0350 | 0.999x |
| 384 | 512 | 2.3437 / 1.8080 | 1.296x | 7.5163 / 7.2017 | 1.044x |
| 768 | 512 | 9.3587 / 7.5726 | 1.236x | 29.7097 / 28.4141 | 1.046x |

D384 has a modest forward advantage, with training approximately tied (L384 H100
is 1.5% slower in these samples). D512 has a 24–30% inference speed advantage but
only 4–5% in complete training. This measures the full current paths, including
their different backward implementations; it does not isolate LN folding itself.

## Actual paths

- Forced Triton: LN saving xn → expand+SwiGLU → squeeze+residual.
  Backward uses saved-xn stacked gate gradients and four cuBLAS GEMMs.
- H100 auto for these aligned, large D384/512 workloads: Triton stats and weight
  fold → CuTe LN-folded expand+SwiGLU → CuTe rounded squeeze+residual.
  Backward creates xn, uses Triton separate dA/dB gate gradients and six cuBLAS
  GEMMs, then residual-fused LN backward. This is not an all-CuTe backward.

## Why the current b2b cannot simply dispatch D384/512

Sources are hashed in [sources.json](sources.json).

1. `autotune/hopper_cuda_config.py::candidates` supports b2b widths 128/256 only.
   The CUDA host entry also explicitly checks these two widths.
2. `transition_b2b_kernel.cu` fixes the squeeze tile at DN=128, and implements only
   one or two live output accumulators (`D/DN` equals 1 or 2). D384/512 require
   three/four output tiles. D384 additionally violates the normalization thread
   mapping requirement `(128 threads * 8 BF16 elements) % K == 0`.
3. The current layout keeps all xn plus staged Wa/Wb/Ws in shared memory. For the
   smallest current b2b BN=64, stages=1, CTA_M=128, K=D, the tensor/stat allocation is
   `2 * (128*D + 3*D*64) + 8*128` bytes, before pipeline barriers:

   | D | Shared memory before barriers |
   |---:|---:|
   | 256 | 161 KiB |
   | 384 | 241 KiB |
   | 512 | 321 KiB |

   D384/512 exceed the kernel's 227 KiB opt-in limit. Larger BN/stages only increase it.
4. At the current 64-row warpgroup tile, keeping all squeeze outputs alone costs
   192 FP32 values/thread at D384 and 256 at D512, before expand/gate operands and
   other state. A wider b2b needs a different storage/tiling design, not only a guard edit.

This is a limitation of the current implementation; it is not proof that every
possible wide-D fusion must lose. The historical commit `b2ce8134` file
`docs/kernel-optimization/transition_b2b/highdim-and-split.md` reports D512
single-warpgroup and K-tiled alternatives at approximately 3.21/3.07 ms versus
CuTe split 1.31 ms (M=131072, old kernels, no L2 flush). Those are historical
measurements, not a runnable current b2b result, and are not mixed into this table.

## Validation and reproduction

All 32 benchmark rows completed finite-value and graph-replay checks. Maximum
relative Frobenius difference from the fixture's compiled PyTorch reference was
0.004792 for output and 0.004883 for input gradient. Reference comparisons report
these metrics; graph replay separately asserts its field-specific tolerances.
This is not a fresh exhaustive parameter-gradient test suite or NCU profile.

`measure.py` and `env.sh` are the exact reused scripts. Run `--auto-only` for this
record; legacy b2b candidate options in the script are not valid for these widths.
Within a fresh **node02** allocation, run each width on a separate exclusive GPU:

```sh
bash env.sh measure.py --width 384 --length 384 --auto-only
bash env.sh measure.py --width 384 --length 768 --auto-only
bash env.sh measure.py --width 512 --length 384 --auto-only
bash env.sh measure.py --width 512 --length 768 --auto-only
```

Raw run directory: `/home/psk6950/MiniWorld/runs/transition_wide_bench_20260918`.
The recorded `summarize.py` is run from that original directory and uses only the
standard library. Native build artifacts use the isolated extension directory
configured in `env.sh`.
