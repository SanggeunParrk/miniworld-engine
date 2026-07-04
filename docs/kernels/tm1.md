# tm1 — left/right gated GEMM

## Math

For pair input `x: (B, L, L, D)`:

```
left  = σ(x @ W_Lg) ⊙ (x @ W_L)     # (B, L, L, D)
right = σ(x @ W_Rg) ⊙ (x @ W_R)
```

Each side is a "gated GEMM": one matmul produces the sigmoid input
(gate), another produces the value (projection), elementwise gate ⊙ value.

`team-gm`'s Triton `tm1` keeps **four** accumulators (LA, LB, RA, RB) in
one CTA so it can reuse the loaded `x` tile across all four matmuls. The
CuTeDSL port goes the other way: **two side kernels** with two
accumulators each, launched serially. Register pressure halves and the
output layout can be tuned per side.

## Implementations

| name           | what it is                                                          |
|----------------|---------------------------------------------------------------------|
| `pt`           | pure PyTorch (`x @ W_*`, `torch.sigmoid`)                           |
| `tn`           | team-gm `psk/benchmark` `triton_tm1` (4-acc, single kernel)         |
| `nv`           | team-gm `perf/trimul` `triton_tm1` (same shape; TF32/sigmoid fixes) |
| `cu`           | CuTeDSL: side-split `quack.GemmGatedSm90` ×2 with patched M-major output (see `cute/launch.py`) |

The CuTeDSL path's payoff isn't tm1-in-isolation — it's that the kernel
writes its output **directly to `[B, D, L, L]` storage** (via an M-major
postact view of the destination buffer + two patched lines in
`quack.gemm_act`). That lets the downstream triangle contraction run as
a `bmm` instead of a 4-D einsum, which is where the big TriMul win comes
from. See `docs/kernels/trimul-inproj.md`.

## Results (H100, bf16, B=1, D=128)

Wall time (ms):

| L    | pytorch | triton (tn) | nvidia-triton (nv) | cute (cu) |
|------|--------:|------------:|-------------------:|----------:|
| 384  |   0.26  |       0.08  |              0.11  |     0.10  |
| 512  |   0.45  |       0.14  |              0.19  |     0.13  |
| 768  |   0.95  |       0.30  |              0.40  |     0.24  |
| 1024 |   1.66  |       0.52  |              0.73  |     0.40  |

Triton main is the fastest *single-kernel* tm1 here. The cute path's
`cu_bdll_direct` (output already in `[B, d, L, L]`) is competitive at
L ≥ 512 — and crucially, that layout pays off downstream.

For full numbers including TFLOPS and ratios, see the bench output
(`python -m triangle_multiplication.tm1.bench`) and
`tm1/cute/bench.py`.

## Files

```
tm1/
├── reference.py     # PyTorch reference (matches team-gm Triton math)
├── interface.py     # tm1_cute: thin Python wrapper, dispatches to cute/
└── cute/            # isolated CuTeDSL env
    ├── pixi.toml         # cu128 torch + nvidia-cutlass-dsl==4.4.2 + quack==0.3.11
    ├── launch.py         # tm1 cute launcher: 3 out_layout modes (blld/bdll/bdll_direct)
    ├── verify.py         # bf16/fp16 correctness, 6 shapes
    ├── _verify_direct.py # bdll_direct path: B=1, L∈{32..1024}
    ├── bench.py          # 4-way tm1 bench (pt/tn/cu_blld/cu_bdll/cu_bdll_direct)
    ├── bench_trimul.py   # full TriMul bench — pt/nv/cuequiv/cute (4-way)
    ├── fused_ln_mask.py  # custom Triton: LN(x) + per-row mask, 1 HBM pass
    ├── tm2_cute.py       # tm2 wrapper around cuequiv dual-x gated GEMM
    ├── tm2_cute_kernel.py# from-scratch CuTeDSL dual-A GEMM (WIP)
    └── _bench_stages.py  # per-stage timing inside trimul_cute
```

## Key engineering notes

* **Patched `quack.gemm_act`** — two sed edits to drop the
  `is_n_major_c()` assert on postact and to honor a detected M-major
  layout for the gated path. With those, the tm1 cute launcher using
  `out_layout="bdll_direct"` hands the GEMM an M-major postact view
  whose backing storage is `[B, D, L, L]` — kernel writes land directly
  there, no follow-up `permute().contiguous()`.

* **`fused_ln_mask.py`** — when an input mask is present, LayerNorm and
  the per-row mask multiply collapse into one Triton kernel (2.4× faster
  than `F.layer_norm` + separate `.mul_(mask)` on these shapes).

* **CuTeDSL from-scratch tm2** (`tm2_cute_kernel.py`) — **WIP, 28
  debug iterations in.** Significant progress; not yet correct.

  Known-good checkpoint: with the early ``return`` after
  ``mbarrier_wait``, the kernel reaches that point on H100 without
  hang or fault. That validates the entire **setup + compile + TMA
  load + mbarrier sync** path: smem layouts with swizzle split on
  ``get_tensor(layout.outer, swizzle=layout.inner)`` (the
  ``make_fragment_A/B`` composed-layout rejection fix), the `self`-
  based SharedStorage pattern (avoids tracer Constexpr issues), manual
  mbarrier_init with pre-extracted pointers, TMA loads via
  ``quack.copy_utils.tma_get_copy_fn`` inside ``with
  cute.arch.elect_one():``, and ``mbarrier_wait(ptr, Int32(1))`` —
  phase=1 is the correct parity to wait for on the first cycle.

  Open blocker: enabling the GMMA section hangs at runtime. Tested
  with ``quack_sm90.gemm`` + own acc, ``gemm_zero_init`` helper,
  single GMMA call, with/without extra ``cute.arch.barrier()``
  between mbar_wait and GMMA — all hang identically. The compile
  ``bad_variant_access`` ICE only shows up with the epilogue path
  active. See the docstring at the top of ``tm2_cute_kernel.py``
  for the suggested next-session investigation steps.
