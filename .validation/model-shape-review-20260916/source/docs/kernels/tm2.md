# tm2 — output gated GEMM

## Math

```
y = σ(x_normed @ W_g) ⊙ (out_normed @ W_p)
```

Unlike tm1's "single-A dual-acc" pattern, **tm2 has two distinct input
tensors**: the gate input (`x_normed`) is different from the projection
input (`out_normed`). So tm2 is a **dual-A dual-acc** gated GEMM:

* `gate_acc = x_normed @ W_g`  → `σ(gate_acc)`
* `proj_acc = out_normed @ W_p`
* `out = σ(gate_acc) * proj_acc`

`quack.gemm_act("glu")` only handles single-A → can't be used here. The
cuequivariance / nvidia-triton (dtv1) reference for the output-side gated
GEMM is `cuequivariance_ops_torch.gated_gemm_torch.fused_sigmoid_gated_dual_gemm_dual_x`,
which is the SM90 kernel cuequivariance uses internally for this op.

## Implementations

| name | what it is                                                                    |
|------|-------------------------------------------------------------------------------|
| `pt` | pure PyTorch (`@`, `torch.sigmoid`)                                           |
| `tn` | team-gm `psk/benchmark` `triton_tm2` (single Triton kernel, dual K-loop)      |
| `nv` | team-gm `perf/trimul` `triton_tm2` (TF32 / sigmoid precision fixes)           |
| `cu` | **`tm2_cute`** — `fused_sigmoid_gated_dual_gemm_dual_x` (cuequiv's SM90 kernel) |

A from-scratch CuTeDSL dual-A kernel (`triangle_multiplication/tm1/cute/tm2_cute_kernel.py`)
is in progress as a learning exercise. The cuequiv-backed path above is
the production cute baseline since it's the same kernel cuequivariance
uses and matches dtv1 bit-exactly.

## Results (H100, bf16, B=1, D=128)

Wall time (ms), tm2 in isolation:

| L    | pytorch | triton (tn) | nvidia-triton (nv) | cute (cu — cuequiv kernel) |
|------|--------:|------------:|-------------------:|----------------------------:|
| 384  |   0.14  |       0.05  |              0.06  |                       0.06  |
| 512  |   0.23  |       0.08  |              0.10  |                       0.09  |
| 768  |   0.48  |       0.17  |              0.20  |                       0.18  |
| 1024 |   0.83  |       0.29  |              0.34  |                       0.30  |

For our shapes (K=D=128) `triton_tm2` from team-gm is *slightly* faster
than the cuequiv-backed cute path (50–80 µs delta at small L, amortized
at L=1024). At full-TriMul level the difference is within noise, so we
keep `tm2_cute` for the "all cute" story; switch to `triton_tm2` if you
need the absolute fastest tm2.

## Files

```
tm2/
├── reference.py    # PyTorch reference
├── interface.py    # tm2_cute placeholder (the real wrapper lives in tm1/cute/tm2_cute.py)
├── bench.py        # 4-way bench (pt/tn/nv/cu)
```
```

The actual cute wrapper is in `triangle_multiplication/tm1/cute/tm2_cute.py`
(co-located with the rest of the cute env so they share pixi + imports).
