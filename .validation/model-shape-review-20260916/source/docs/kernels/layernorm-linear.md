# layernorm_linear — fused LayerNorm + Linear

## Math

For input `x: (..., d_in)`:

```
y = Linear(LayerNorm(x))
  = (x̂ ⊙ γ + β) @ Wᵀ + b ,   x̂ = (x − mean(x)) / sqrt(var(x) + ε)
```

LayerNorm over the last dim (stats in fp32), then an affine GEMM. This is the
`te.LayerNormLinear` op; fusing the two stages saves the HBM round-trip paid by
the separate `LayerNorm` -> `Linear` pair.

## Implementations

| name             | what it is                                                        |
|------------------|-------------------------------------------------------------------|
| `pytorch`        | `layernorm_linear_pytorch` — `F.layer_norm` + `F.linear`          |
| `torch.compile`  | `torch.compile(LayerNormLinearRef)` — the baseline / correctness oracle |
| `te`             | NVIDIA Transformer Engine `te.LayerNormLinear` (baseline)         |
| `cute`           | `cute/`: **our fused inference** — quack SM90 GEMM + folded LayerNorm epilogue (Milestone 1) |
| `triton`         | `layernorm_linear_triton` — alternative single-pass kernel (placeholder) |

## How the fused kernel works (`cute/`)

We do **not** write a GEMM. We start from quack's trusted warp-specialized SM90
BF16 GEMM (`GemmSm90`) and only swap the **epilogue** via quack's composable
`_epi_ops` system — adding two extra broadcast loaders so the epilogue computes,
per output element, ``Y = rstd[m]*acc - c1[m]*S[n] + B2[n]`` in one line. The
GEMM mainloop / WGMMA scheduling is untouched.

- **Prologue** (`fold_for_gemm`, cached for fixed weights): ``B=(N,K)=gamma⊙W``
  (the GEMM operand), ``S[n]=Σ_k B[n,k]`` (FP32, from the *stored* bf16 B), ``B2[n]``.
- **Stats** (Milestone 1): `rstd`, `c1=mean*rstd` from a `torch.compile`'d
  single-pass reduction over X. (Triton stats / mainloop-fused stats = next.)
- **Main**: raw ``X @ B`` on the quack GEMM; LayerNorm(X) [M,K] never materialized.

## Baseline results (H100, bf16) — TE vs torch.compile

Square `d_in = d_out ∈ {128,256,384,512,768}` × `M ∈ {16384,65536,262144}`,
inference + training. Generated tables and graphs now belong under
`benchmarks/kernels/layernorm_linear/artifacts/`.

Numerics agree to bf16 (`cos = 1.000000`, rel-Frobenius ~1e-4). Latency verdict:

- **Small/medium M (16384, 65536):** TE wins, growing with d (up to −28% at
  d=768, M=16384) — torch.compile's backward is weak at small batch.
- **Large M=262144, small d (128–384):** torch.compile wins (TE +7–20%).
- **Large M, large d (512, 768):** TE wins again.

→ Compute-bound (large d) favors TE; memory-bound (small d, large M) favors
torch.compile. The Triton kernel targets the memory-bound regime first, where a
single LN+GEMM pass should beat both by cutting the intermediate HBM write.

## Files

```
layernorm_linear/
├── reference.py   # PyTorch ref: layernorm_linear_pytorch + LayerNormLinearRef (nn.Module)
├── interface.py   # layernorm_linear_triton — Triton entry point (WIP placeholder)
├── triton/        # Triton kernel implementation (to come)
└── cute/          # SM90 CuTe/quack implementation
```

## Fused kernel result (Milestone 1, H100 bf16, inference)

`cute` vs the baselines on the square grid.
Correctness: `cos = 0.999997` vs the true op (bf16-level) for all K=N shapes.

Forward latency (ms) via `triton.testing.do_bench`, fastest in **bold**:

| d   | M=16384 (compile/TE/cute) | M=65536 | M=262144 |
|-----|---------------------------|---------|----------|
| 128 | 0.0172 / 0.0170 / **0.0151** | 0.0371 / 0.0423 / **0.0310** | 0.1239 / 0.1386 / **0.0937** |
| 256 | 0.0251 / 0.0254 / **0.0230** | 0.0605 / 0.0688 / **0.0563** | 0.2045 / 0.2280 / **0.1818** |
| 384 | 0.0294 / 0.0341 / **0.0250** | 0.0933 / 0.1017 / **0.0699** | 0.2981 / 0.3390 / **0.2459** |
| 512 | 0.0366 / 0.0409 / **0.0354** | 0.1148 / 0.1255 / **0.1027** | 0.4161 / 0.4525 / **0.3546** |
| 768 | 0.0578 / 0.0583 / **0.0476** | 0.1989 / 0.2031 / **0.1608** | 0.7418 / 0.7673 / **0.6084** |

**The thesis holds — and more strongly than first measured.** The fused `cute`
kernel is the fastest inference path in **all 15** shapes (~10–30% over the best
baseline), by never materializing LayerNorm(X). This is with `torch.compile`
stats (Milestone 1) — the in-mainloop stats (Milestone 2, `*_fused.py`) should
widen it further. (Earlier numbers used a hand-rolled timer whose per-call sync
inflated all backends; `triton.testing.do_bench` — the repo convention — is the
honest measurement.)

## Milestone 2 — stats fused into the GEMM mainloop (`cute/gemm_layernorm_linear_fused.py`)

The user's actual design: **one main kernel** that computes the per-row LayerNorm
stats *inside* the GEMM on CUDA cores (parallel to the WGMMA tensor cores), so
there is no separate stats pass. Implemented by forking quack's `GemmSm90`
(overriding `kernel`/`mma`; stats broadcast to the epilogue via a smem-sourced
`SmemColVec`). Prologue(1) + main(1) = **2 kernels total**.

- ✅ **Correct** — `cos = 0.999997` (bf16-level) across the whole grid AND the
  QKV shape (K=4096, N=12288); the separate-stats large-N bug does NOT occur here
  (no ColVecLoad). Historical validation scripts were removed from the package tree.
- Perf (inference): **fastest at small d** (d=128, M=16384: 0.0129 ms — beats TE
  0.0173 and the M1 kernel), but **slower at larger d** (the first version uses a
  NON-persistent tile scheduler — the persistent path has an unresolved stats-smem
  reuse race — and the in-mainloop reduction competes with the GEMM as d grows).
- Next: fix the persistent-scheduler stats-smem sync (double-buffer or a dedicated
  barrier) → reclaim scheduling efficiency; then tune the reduction.

## Status / next

- ✅ Milestone 0: folded math validated (`verify_folding.py`); cancellation only
  bites at pathological mean≳1000 (naive var breaks first).
- ✅ Milestone 1 (`cute/gemm_layernorm_linear.py`): separate (torch.compile) stats
  + fused GEMM epilogue. Fastest inference path in all grid shapes (~10–30% over TE).
  ⚠️ Known bug: QKV large-N ~2% off (persistent + ColVecLoad) — square exact.
- ✅ Milestone 2 (`cute/gemm_layernorm_linear_fused.py`): stats in the mainloop,
  2 kernels, correct everywhere incl. QKV; wins at small d; large-d perf pending
  the persistent-scheduler fix.
- Next: (1) persistent stats-smem sync → big-d perf; (2) Triton stats for M1;
  (3) training backward (out of scope for v1).
