# fast_trimul (E5 baseline `fast_trimul`)

- Upstream: https://github.com/tiagomonteiro0715/fast_trimul, PyPI `fast_trimul` 3.0.6 (wheel sha256 in UPSTREAM.json). License Apache-2.0 (LICENSE + NOTICE in the wheel).
- What it is: a fused triangle multiplicative update for the AF3 family on CUTLASS CuTe DSL: LayerNorm, gated projections (one wide GEMM),
  batched tensor-core contraction (outgoing / incoming), LN_out, output gate, optional residual; `patch_openfold3()` etc. patch other libraries.
- Covers: pair `[B, N, N, d]`, `N % 8 == 0` (its dispatcher pads; our adapter calls the kernel path), fp16 kernels (other dtypes are cast), biased
  Linear layers (zero biases for our bias-free module), backward = a torch recompute (correct, not fast).
- Does not cover: bf16/fp32 arithmetic, masks (it masks z before LN_in: a different function), N % 8 != 0 in the adapter.
- Requires: `nvidia-cutlass-dsl` (engine envs of H100 / A100 have 4.8.0; the 5090 venv has none: `CUTLASS_DSL=1 fetch.sh` unpacks the 4.8.0 cu12 wheels next to it), `cuda-bindings` 12.x (`cuda-python<13`).
- Install: `third_party/fast_trimul/fetch.sh [TARGET]` (a pip-free wheel unpack, no dependency is touched). Patches: none.
- Adapter: `benchmarks/paper/E5_other_kernels/ext_fast_trimul.py`.
