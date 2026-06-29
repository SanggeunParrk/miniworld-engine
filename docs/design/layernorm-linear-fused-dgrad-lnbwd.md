# cute 1+4: fused dgrad GEMM + LN-backward (design)

## Goal
Fuse backward steps 1 (`dx_normed = dY @ W`) and 4 (`dx,dγ,dβ = LN_backward(dx_normed)`) into
ONE cute kernel so `dx_normed` (M,K) is never written/read back from HBM. The Triton version
(`triton/fused_bwd.py::dgrad_lnbwd`) is correct but loses because the **Triton GEMM is slower
than cuBLAS**; the win needs a **fast GEMM**, i.e. quack's `GemmSm90`. cuBLAS can't host the
epilogue, so we fork quack like M1 did.

## Why this is M1-level, NOT M2-level
M2 (inference) was hard because it reduced the **A operand in the mainloop** (sA recycle race,
pingpong barrier divergence, deadlocks). **1+4 needs NONE of that**: mean/rstd are saved inputs,
so there is no mainloop reduction. It's a **composable epilogue** on a standard `dY@W` GEMM —
the M1 pattern (`GemmDefaultEpiMixin` + `epi_ops`), not the M2 fork.

## Scope: d ≤ 256 only
The LN-backward needs the per-row reductions over the FULL output width K:
`c2 = meanₖ(dx̂)`, `c1 = meanₖ(dx̂·x̂)`, then `dx = rstd·(dx̂ − c2 − x̂·c1)`, `dx̂ = acc·γ`.
Set **tile_N = K** so the whole K-row lands in ONE epilogue subtile → reduce + apply in a single
pass (no two-pass, no cross-tile sync). tile_N ≤ 256 ⇒ d ≤ 256. (d>256 keeps the cuBLAS
unfused path — it can't fuse and loses nothing.) This is exactly the regime where the win is
real: at large M, `dx_normed` (M,K) > L2 so the round-trip is true HBM traffic (~45µs at
d=128 M=262144 ≈ 30% of the bwd).

## GEMM
A = dY (M, N), B = W (N, K) → out = dx_normed (M, K). Contraction = N. (W is already (N,K) =
(K_gemm, N_gemm); same operand orientation as the inference `gemm(dY, W)` in autograd.py.)

## Epilogue (custom epi op, single subtile = full K row)
Inputs broadcast onto the acc fragment:
- `γ` (K,) → `RowVecLoad("gamma")` (per output column).
- `rstd`,`mean` (M,) → `ColVecLoad("rstd")`, `ColVecLoad("mean")` (per row).
- `x` (M,K) → per-element load (like M2's `mX`) to form `xhat = (x-mean)*rstd`.
  Or save `xhat` in inference and load it directly; co-design with the fold-free
  inference path if adopted.
Per row (within the single subtile that holds all K):
1. `dx̂ = acc · γ`
2. warp/smem reduce over the subtile's N(=K): `c2 = Σ dx̂ / K`, `c1 = Σ (dx̂·x̂) / K`
   (reuse the butterfly/`ColVecReduce` reduce in `epi_ops.py` / M2's `_reduce` helper).
3. `dx = rstd·(dx̂ − c2 − x̂·c1)` → store to dx (M,K).
4. `dγ += Σ_m dx̂·x̂`, `dβ += Σ_m dx̂` — these reduce over M (across CTAs) → `atomic_add` to
   fp32 gmem (K,) buffers (can't fuse into the epilogue's per-tile scope).

## Reuse map
- `GemmSm90` + `GemmDefaultEpiMixin` (compose, don't fork the mainloop) — like
  `gemm_layernorm_linear.py` (M1).
- `RowVecLoad`/`ColVecLoad` from `quack.epi_ops`; the per-element x load + `SmemColVec` pattern
  from `gemm_layernorm_linear_fused.py` (M2) if x/x̂ needs kernel-side broadcast.
- `ColVecReduce` (`quack/epi_ops.py:539`) for the in-epilogue row reduction (c1, c2).
- Drive config tile_n = K (128 or 256), pingpong as M1.

## Verify / bench
- `verify`: dx,dγ,dβ cos vs fp32 autograd oracle (extend `cute/verify_bwd.py`), d∈{128,256}.
- `bench`: fused vs unfused (`gemm(dY,W)` cuBLAS + `_ln_backward`) at d∈{128,256} × large M.
  KEEP only if it wins (the Triton version was ~tie at d=128 M=262144 = 167 vs 156; cute should
  drop to ~110 if the round-trip (~45µs) is saved with a cuBLAS-speed GEMM).

## Risk
- In-epilogue row reduction over the acc fragment (cross-thread within the warpgroup) is the one
  new piece; `ColVecReduce` exists for exactly this. If the acc N-fragment spans multiple warps,
  needs a smem reduce (M2 has the pattern).
- dγ/dβ atomics over M: contention at large M; or a separate small reduction kernel.
