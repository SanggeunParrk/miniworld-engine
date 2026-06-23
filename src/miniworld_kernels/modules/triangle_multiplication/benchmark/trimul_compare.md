# Triangle multiplication — ours vs NVIDIA kernels

> ⚠️ **Superseded for the compile/regime story** — the numbers below mixed
> measurement methodologies (compile for ours/cuequiv, README-eager for dtv1).
> The validated, apples-to-apples comparison (team-gm-faithful harness, all
> regimes) is in **[trimul_compile_analysis.md](trimul_compile_analysis.md)**.
> Verdict there: ours wins every L in every regime (compile-everything 1.37–1.65×).


Forward pass, H100 80GB, B=1, D=128, **bf16**. ALL paths measured in one harness
(`kernels/trimul_inproj/cute/compare_bench.py`): dtv1 and cuequiv are measured
here, not quoted. Two methodologies reported — **eager** (per-layer `do_bench`,
the team-gm methodology, fair to dtv1) and **torch.compile** (reduce-overhead,
K=8 stack).

## EAGER (ms/layer) — the honest, dtype-matched comparison

| L    | pytorch | nvidia dtv1 | cuequivariance | **ours v4** | ours v5 |
|-----:|--------:|------------:|---------------:|------------:|--------:|
| 384  | 1.180   | 0.344       | 0.391          | **0.247**   | 0.443   |
| 512  | 2.245   | 0.583       | 0.474          | **0.362**   | 0.470   |
| 768  | 7.805   | 1.288       | 1.036          | **0.811**   | 0.848   |
| 1024 | 14.992  | 2.437       | 1.892          | **1.496**   | 1.554   |

## torch.compile (reduce-overhead, K=8 stack, ms/layer)

| L    | pytorch | nvidia dtv1 | cuequivariance | **ours v4** | ours v5 |
|-----:|--------:|------------:|---------------:|------------:|--------:|
| 384  | 0.724   | 0.409       | 0.258          | **0.204**   | 0.430   |
| 512  | 1.473   | 0.715       | 0.464          | **0.353**   | 0.458   |
| 768  | 6.075   | 1.557       | 1.015          | **0.783**   | 0.834   |
| 1024 | 11.477  | 2.815       | 1.844          | **1.539**   | 1.551   |

dtv1's autograd Functions are `@torch.compiler.disable()`'d, so torch.compile /
cudagraphs can't capture them — that's why dtv1 looks *worse* under compile.
Eager is the fair view for dtv1.

## ours v4 speedup (eager, bf16)

| L | vs pytorch | vs dtv1 | vs cuequivariance |
|--:|--:|--:|--:|
| 384 | 4.8× | 1.39× | 1.58× |
| 512 | 6.2× | 1.61× | 1.31× |
| 768 | 9.6× | 1.59× | 1.28× |
| 1024 | 10.0× | 1.63× | **1.27×** |

**ours v4 beats BOTH NVIDIA kernels at every L, in both eager and compile.**

## Reconciling with team-gm (`docs/performance.md`)

team-gm's published table shows dtv1 **faster** than cuequiv (1.04–1.58× fwd,
1.10–2.01× fwd+bwd). That is **fp32** (their header: "H100 SXM, fp32"). Our
comparison is **bf16**. The dtype flips the dtv1↔cuequiv ordering at large L:

| @1024 fwd | fp32 (team-gm) | bf16 (here) | bf16 gain |
|---|---:|---:|---:|
| cuequivariance | 3.495 | 1.892 | **1.85×** |
| dtv1           | 3.360 | 2.437 | 1.38× |

cuequiv gets more out of bf16 tensor cores, so it overtakes dtv1 in bf16 at
L≥512. In fp32, dtv1 wins (as team-gm reports). Either way, **ours wins in
bf16** — the dtype the trimul module trains in. (team-gm's strongest dtv1 wins
are also *fwd+bwd*, where its optimized backward shines; this table is
forward-only. Our backward is in development — see below.)

## Autotuning (fairness)

| kernel | tile autotuning? |
|--------|------------------|
| ours v4 — front (`trimul_inproj` gemm_act) | ✅ quack `tuned=True` |
| ours v4 — back (triton dual-gemm) | ✅ `@triton.autotune` |
| ours v5 — back (`layernorm_linear_cute_fused`) | ✅ swept per-L; default 128×128 already optimal at d=128/N=128 |
| cuequivariance | ✅ AOT-tuned |
| nvidia dtv1 | ✅ Triton autotuner (occupancy/maxnreg/stages) |

All autotuned — fair. v5's per-L sweep found the default config best for d=128
(no meaningful gain), so v4 stays the headline.

## What "ours" is

```
LN_in -> trimul_inproj (left+right fused, ONE gated GEMM, bdll, m-major in-place)
      -> torch.bmm  -> fused back-half:
   v4: Triton dual-gemm  (LN_out(tri)@Wp ⊙ σ(x_n@Wg), gate in-kernel)   ← fastest
   v5: cute layernorm_linear + folded gate-mul (gate precomputed)
```

## Backward (in development)

- **B0 done**: manual backward (stages: back-half → bmm → front gated-GEMM →
  LN_in) verified bit-exact vs torch autograd (worst rel 6e-16, fp64) —
  `kernels/trimul_inproj/{autograd.py,bwd_test.py}`. Forward kernels swap in next
  (cute-first), targeting dtv1's fwd+bwd numbers.

![bars](trimul_compare_bars.png)
