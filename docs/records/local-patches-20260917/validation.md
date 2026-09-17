# Integration validation — 2026-09-17

## Environment

- Isolated checkout; original consumer repository, installed engine and running
  training/cache snapshots left unchanged.
- One NVIDIA H100 80GB, Slurm allocation 13217, PyTorch 2.10/CUDA 12.8,
  Python 3.10. Static gates use Ruff 0.15.20 and ty 0.0.74 with engine Python
  3.12 dependencies and this checkout's `src` as the first import root.
- Tests are correctness/dispatch checks, not a new performance benchmark.

## GPU results

- `pytest tests/consumer_updates -m gpu -q --tb=short`: **173 passed**.
  Includes TriMul packed buffers, F567 and backward fusions, direct launches
  with strided/ragged inputs, static/dynamic compile, dropout/residual graph
  replay, strict Triton routing, AdaLN alignment, bounded attention scratch,
  mixed affine LayerNorm, and FA4 native compiled backward.
- Additional merged-main SWA DiT coverage: **12 passed**, from
  `tests/numerics/test_swa_dit_kernels.py`. Its two CPU dispatch tests are
  covered by the CPU suite. The combined targeted GPU run was **32 passed**;
  the other 20 overlap the consumer suite above.
- KP=4096 public TriMul launch: **1 passed** independently against FP32
  products with the prescribed BF16 intermediate rounding; also included
  in the final 173 above.
- FA2 CUDA graph tests require an sm80-family GPU. They were not run on H100.

## Static and packaging checks

- Ruff: passed.
- ty: passed.
- `CUDA_VISIBLE_DEVICES='' python -m miniworld_engine.cli dev audit`: exit 0.
  Shard reachability and hardware coverage are warnings without real build
  shards/device context; the audit is not a complete-cache claim.
- `pip wheel --no-deps --no-build-isolation .`: passed. Asset inspection found
  the complete Python source set, runtime cache JSON, config/manifests, CUDA
  sources, registry and `py.typed`; archived incompatible caches are excluded.

## Derived build plan

`MINIWORLD_COMPILE_WRAP=disable python -m miniworld_engine.cli dev derive
--arch sm86 --workers 4`: **8,242 invocations, zero errors**. The verified plan
contains **52 kernels and 1,385 required cache keys**. The sweep page was
regenerated from that evidence. This derives calls and shapes without timing
or compiling the kernels; it does not claim the corresponding caches are full.

## CPU results

- Whole non-GPU suite: **3,771 passed, 40 skipped**, with one remaining failure
  in the ladder-completeness test collected before its final correction.
- Fresh `pytest tests/ -m 'not gpu' --lf -q --tb=short`: **1 passed**, exit 0.
  The complete affected ladder test file also passed independently: **5 passed**.
- The final guard retains the full driver-sweep requirement and additionally
  requires exact model-key coverage from a verified plan. The regression with
  equal bucket counts but different shapes passes. No test failure remains open.
- Final wheel SHA-256:
  `9bf25cc30c0119619f3a8485fd0e15dda197fccdedb5746238f6049697a111cf`.
  It contains 712 files; the checked asset groups include 289 runtime cache
  files, 89 configuration grids, 3 manifests and 13 CUDA sources.
