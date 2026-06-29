# trimul_inproj micro-benchmark

Kernel-development diagnostic (`cute/perf.py`), forward-only, H100 80GB HBM3,
B=1, D=128, bf16. Median ms over `triton.do_bench`. All three variants include
the same plain-torch `gate = sigmoid(x@Wg)`, so deltas are the left/right path.

Variants:
- **new bdll** — left+right fused into ONE gated GEMM, `[B,D,L,L]` direct write
  (in-repo `_bdll_patch` shim, no permute).
- **new fallback** — same fused GEMM but n-major postact + `permute().contiguous()`.
- **tm1 2-launch** — tm1's two separate gated GEMMs (`out_layout="bdll_direct"`).

| L    | new bdll (ms) | new fallback (ms) | tm1 2-launch (ms) | bdll/fallback | bdll/tm1 |
|-----:|--------------:|------------------:|------------------:|--------------:|---------:|
| 384  |        0.136  |            0.448  |            0.145  |     **3.30×** |   1.07×  |
| 512  |        0.215  |            0.985  |            0.228  |     **4.58×** |   1.06×  |
| 768  |        0.433  |            4.950  |            0.462  |    **11.44×** |   1.07×  |
| 1024 |        0.738  |            9.589  |            0.802  |    **13.00×** |   1.09×  |

## Findings

1. **The `[B,D,L,L]` direct write is the dominant lever: 3.3×–13× over the
   permute fallback**, and the gap explodes with L (the `permute().contiguous()`
   is a full-tensor transpose that scales worse than the GEMM). This is exactly
   why stock quack's rejection of the M-major postact mattered, and why the
   in-repo `_bdll_patch` shim is load-bearing, not optional.
2. **Fusing left+right into one wide gated GEMM beats tm1's two launches by
   6–9%**, consistently across L. The single wider launch (N=4D→2D postact)
   raises epilogue register pressure but still wins on one fewer `x` read + one
   fewer launch — the register-pressure concern that motivated tm1's split does
   not dominate here.

![micro-bench](trimul_inproj_micro.png)
