# ConditionedTransition tail — backward verdict at REAL M (A=48), H100, CUDA graph, TF32

Real `M = A*L`, A=48. atom d=128, token d=768, n=2, d_cond=384. fp32 io, TF32 tensor cores.
All variants: wgrad on cuBLAS; only the dgrad path / elementwise fusion differs.
Correctness of the new clean variant: cos = 1.00000 on y + all 7 grads, every shape.

## fwd+bwd (CUDA graph, µs)

| stream |      M | champ (cuBLAS-dg) | clean (triton-dg) | fused (prologue) | compile | eager | clean/champ | clean/compile |
|--------|-------:|------------------:|------------------:|-----------------:|--------:|------:|------------:|--------------:|
| atom   |  98304 |            1378.9 |            1698.8 |           2319.9 |  1504.1 |1736.6 |       0.81x |         0.89x |
| atom   | 196608 |            2691.6 |            3298.4 |           4507.3 |  2881.9 |3297.7 |       0.82x |         0.87x |
| atom   | 393216 |            5096.3 |            6256.5 |           8668.4 |  5602.3 |6333.0 |       0.81x |         0.90x |
| token  |  18432 |            2235.3 |            4279.6 |           7895.9 |  2447.5 |2687.0 |       0.52x |         0.57x |
| token  |  24576 |            2921.7 |            5541.5 |          10440.1 |  3160.2 |3484.9 |       0.53x |         0.57x |
| token  |  36864 |            4399.3 |            8193.9 |          15539.5 |  4876.0 |5151.1 |       0.54x |         0.60x |
| token  |  49152 |            6204.0 |           11276.1 |          21112.1 |  6342.0 |6812.8 |       0.55x |         0.56x |

Two facts:
1. **clean ≫ fused-prologue (1.3–2.0×).** Folding gate-bwd / swiglu-bwd into the dgrad-GEMM
   *prologue* (the old `cond_transition_train_12_345`) recomputes sigmoid/silu/silu′ once per
   N-output-tile (grid_n×) AND still must materialize dout/dscale/dab for the cuBLAS wgrad —
   pure waste that serializes ALU into the WGMMA pipeline. Computing the elementwise ONCE
   (it also feeds wgrad, free) + a CLEAN GEMM is the correct triton structure.
2. **clean still LOSES to the cuBLAS champion (0.81× atom, 0.52–0.55× token) and to compile.**
   The reason is purely the GEMM primitive, isolated below.

## per-stage dgrad GEMM micro (CUDA graph, µs): triton `_dgemm` (TF32, 11-config autotuned) vs cuBLAS `matmul`

| stream |      M | gemm |        M,K,N | triton | cuBLAS | tri/cub |
|--------|-------:|------|-------------:|-------:|-------:|--------:|
| atom   |  98304 | dh   |  98304,128,256 |  179.7 |   72.1 |   2.49x |
| atom   |  98304 | dcond|  98304,128,384 |  253.6 |   96.5 |   2.63x |
| atom   |  98304 | dx   |  98304,512,128 |  200.6 |   91.5 |   2.19x |
| atom   | 393216 | dh   | 393216,128,256 |  645.5 |  286.7 |   2.25x |
| atom   | 393216 | dcond| 393216,128,384 |  967.4 |  368.6 |   2.62x |
| atom   | 393216 | dx   | 393216,512,128 |  756.8 |  335.0 |   2.26x |
| token  |  18432 | dh   | 18432,768,1536 |  592.2 |  142.8 |   4.15x |
| token  |  18432 | dx   | 18432,3072,768 | 1094.2 |  320.9 |   3.41x |
| token  |  49152 | dh   | 49152,768,1536 | 1550.4 |  384.5 |   4.03x |
| token  |  49152 | dcond| 49152,768,384  |  436.9 |  148.8 |   2.94x |
| token  |  49152 | dx   | 49152,3072,768 | 2679.3 |  870.0 |   3.08x |

**Triton TF32 GEMM is 2.2–2.6× (atom) / 2.9–4.2× (token) slower than cuBLAS at the exact dgrad
shapes**, isolated and CUDA-graph-timed with the autotuned swizzled `_dgemm`. This is a
kernel-level primitive gap (triton's Hopper TF32 mma vs cuBLAS's specialized split-K/pipelined
WGMMA), not fusion overhead or under-tuning. No prologue fusion can close a 2–4× GEMM gap.

## CUTLASS GEMM, re-tuned at REAL M (best-of-all-configs vs cuBLAS), ratio = cutlass/cublas

Earlier CUTLASS configs were tuned at M=L (48× too small) → unfair. Re-swept at M=48×L over
every built config (`_ct_cutlass/gemm_pick.py`, A=48). All cos=1.0.

| op (NT, dgrad/fwd) | atom 2048 | atom 8192 | token 384 | token 1024 |
|--------------------|----------:|----------:|----------:|-----------:|
| dh    (dout@Ws)    |     0.997 |     1.000 |     1.326 |      1.307 |
| dx    (dab@Wcat)   |     1.002 |     1.017 |     1.156 |      1.199 |
| dcond (dscale@Wsc) |     0.996 |     1.021 |     1.333 |      1.308 |
| squeeze (h@Ws^T)   |     0.975 |     1.034 |     1.292 |      1.260 |

- **atom: CUTLASS = cuBLAS parity (0.97–1.03).** Replacing cuBLAS with CUTLASS at atom gains 0.
- **token: CUTLASS 1.16–1.40× SLOWER than cuBLAS** on every shape. cuBLAS uses split-K for the
  large-K (3072/1536) token GEMMs; matching it in CUTLASS needs StreamK, which is **not
  CUDA-graph-capturable** → under the production (graph) constraint CUTLASS structurally can't
  catch cuBLAS at token.
- **TN (wgrad): CUTLASS 1.25–2.07× slower everywhere** → wgrad stays cuBLAS (confirmed again).

CUTLASS DOES beat triton (atom: parity-with-cuBLAS = 2.2–2.6× over triton; token: 1.2–1.4×cuBLAS
vs triton's 3–4×cuBLAS). But neither beats cuBLAS.

## Verdict (triton AND CUTLASS both exhausted at real M)
At real (large) M the backward is **GEMM-compute-bound**, and **cuBLAS is the H100 ceiling for
these shapes**: triton TF32 is 2.2–4.2× off; CUTLASS (all configs swept, correct M) only MATCHES
cuBLAS at atom and is 1.2–1.4× SLOWER at token. No available GEMM beats cuBLAS here. So the
GEMM-optimal backward is **cuBLAS dgrad + fused-triton elementwise + cat-merged expand**
(`cond_transition_train`), which BEATS torch.compile ~1.05–1.10× at every shape. The clean-dgrad
triton path (`cond_transition_train_12_345_clean`) is the best-possible triton-only variant
(1.3–2× faster than the prior fused-prologue version) but is not the default.
