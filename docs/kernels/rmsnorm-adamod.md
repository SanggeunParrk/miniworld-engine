# rmsnorm_adamod, and the adaLN modulate ladder

The atom DiT block conditions each sub-layer with adaLN-Zero:

```python
cs = F.silu(c)                                          # one activation, six chunks share it
shift, scale, gate = F.linear(cs, W).chunk(...)         # the projection
sub_in = rmsnorm(q) * (1 + scale) + shift               # the modulate
out    = gate * sub(sub_in)                             # the gated sub-layer
q      = q + out                                        # the residual
```

`rmsnorm_adamod` fuses the boxed middle — the projection, the RMSNorm, the modulate and the gate
projection — into one kernel per sub-layer. `cs = SiLU(c)` stays outside (it is one elementwise
pass all six chunks share; folding it in recomputes it per tile and helps nothing — measured, it
made the backward worse). The `gate * sub(...)` and the residual stay outside too (the sub-layer's
output is not available yet).

`triton_rmsnorm` is the bare normalization used by the SWA q/k and triangle-attention call sites;
it does not modulate. `triton_rmsnorm_adamod` is the fused modulate-with-projection.

## The ladder we measured, and why only the ends shipped

Three ways to compute the modulate, at A=48, S=8192, d_atom 128, d_cond 384 — one block's adaLN
portion, activation checkpointing off:

| | fwd | bwd | step | held (bwd) | peak |
|---|---:|---:|---:|---:|---:|
| **L0** eager (`F.rms_norm` + eager modulate) | 3.58 | 5.04 | 8.61 ms | 1,251 MB | 18,293 MB |
| **L2** shared 6-chunk GEMM + fused modulate | 2.83 | 3.93 | **6.76** | 1,059 MB | 16,373 MB |
| **L3** `rmsnorm_adamod` (projection fused in) | 2.16 | 4.87 | 7.03 | **291 MB** | 10,614 MB |

("held (bwd)" is what the forward keeps alive for the backward — the number that multiplies by
depth and decides whether a crop fits — not the transient peak. It is broken down per tensor
below.)

**L2 is the time optimum, L3 the memory optimum, and neither dominates.** L2 keeps the
conditioning projection as one big cuBLAS GEMM (`c @ W6^T`, six chunks at once), so both the
forward and the backward are one or two large, efficient GEMMs — but it materialises that
`[M, 6*d_atom]` projection (576 MiB here) and, because the gate chunks are views into it, holds
the whole thing for the backward. L3 splits the projection into the two/three slices each
sub-layer needs and contracts them inside the kernel, so nothing of the projection reaches HBM;
the price is that the split GEMMs are smaller and the backward recomputes `scale`.

Held-activation breakdown (per block, adaLN portion, output tensors excluded):

| saved for backward | L0 | L2 | L3 |
|---|---:|---:|---:|
| `SiLU(c)` [M,384] (weight grads need it) | 288 | 288 | 288 |
| `six` [M,768] (gate is a view → whole buffer pinned) | 576 | 576 | — |
| contiguous `scale` copy / `rmsnorm(q)` / `(1+scale)` | 384 | 192 | — |
| `rstd` [M] fp32 | 1.5 | 1.5 | 1.5 |
| **total** | **1,251** | **1,059** | **291** |

At `n_block = 3` per stack, encoder + decoder ≈ 6 blocks, that is ~7.3 GB (L0) / ~6.2 (L2) /
~1.7 (L3) of held adaLN activation before checkpointing — and the atom DiT activations are what
`swa_atom_transformer.py` calls out as dominating diffusion memory at large `num_augment`. L3
trades ~3% step time for that.

## Why only L3 is in the tree

We shipped L3 and deleted L2's code (`triton_rmsnorm_modulate`, `rmsnorm_modulate_reference`, and
the `HAS_MODULATION` branch of `rmsnorm_fwd/bwd_kernel`). Two entry points computing almost the
same thing is a maintenance surface and a doubled cache/build footprint, and the memory win is
the reason this family exists. L2 is a real point on the curve and is recorded here so the choice
is legible; if a future caller is time-bound rather than memory-bound, L2 is
`shared_gemm = F.linear(F.silu(c), W6)` followed by an elementwise `normed*(1+scale)+shift` —
the modulate fused, the projection left shared — and its numbers are the row above.

## How L3 got to 7.03 ms

Starting from a first-cut fusion at 8.88 ms:

- **backward GEMMs 4 → 2.** `dshift` IS `dy`, so the kernel writes `dscale` and `dy` side by side
  into one `[M, 2N]` buffer; `dWsc`/`dWsh` become one stacked `[dscale|dy]^T @ cs` and `dc` one
  `[dscale|dy] @ [Wsc;Wsh]`. 2.27 → 1.31 ms for that piece (−42%).
- **the gate chunk folded in.** adaLN's third chunk is accumulated from the same `c` tile — no
  extra read — and returned as `(y, gate)`; its two backward GEMMs join the stack, widening the
  buffer to `[M, 3N]` rather than adding launches. This removed the separate gate `Linear`, whose
  forward re-read all of `SiLU(c)` and whose backward was two more GEMMs. bwd 5.49 → 4.81.

## The limit

The forward kernel runs at ~131 TF/s and ~35% of HBM bandwidth — roughly 55–60% of what cuBLAS
gets on the same shape. Its grid is 1-D over rows (a program owns a row block's whole N and K),
which is forced by the RMSNorm: `rstd` needs the full row before any output column is final.
Widening the config ladder (BLOCK_M1 to 256, num_stages to 5, 720 configs) changed nothing, and a
sequential-chunk schedule that frees a register accumulator to open larger row tiles came out 24%
slower — the autotuner still chose BLOCK_M1=64. Closing the gap to cuBLAS is a smem-staging /
pipeline rewrite (the shape of work the transition family's hand-CUDA b2b kernel is on sm90), not
a tiling change, and is not done here. The backward kernel is already at ~78% of bandwidth, and
the cuBLAS GEMMs it feeds at 66–79% of tensor-core peak.

## Precision and tile naming

The family runs bf16 and fp32, like layernorm (`dtypes=bf16|fp32`). The pure-reduction kernels
(rmsnorm fwd/bwd) and rope land at ~2e-7 in fp32 -- true fp32 -- while `rmsnorm_adamod`'s `tl.dot`
uses TF32 tensor cores in fp32 io (~9e-4), the same "fp32 io with TF32" the layernorm_linear
family documents. Per-precision bands are in registry.csv.

Tile axes follow docs/kernels/grid-sweep.md's prefix rule (`BLOCK_M*`->M, `BLOCK_N*`->N,
`BLOCK_K*`->K). adamod is the GEMM `c[M,d_cond] @ W[d_cond,d_model]`, so `BLOCK_N` tiles d_model,
`BLOCK_K` tiles the d_cond contraction, `BLOCK_M1` the rows. Its three `tl.dot` axes floor at 16
(Triton's minimum); the reduce/elem kernels floor their rows at 1. Plain rmsnorm keeps `BLOCK_K`
for its normalized width, which is its reduction axis, exactly as layernorm does.
