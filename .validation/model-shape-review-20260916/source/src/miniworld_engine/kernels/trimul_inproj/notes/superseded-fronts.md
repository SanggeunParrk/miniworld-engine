# Fronts that were removed, and what they were for

Five kernels were tuned by every `build all` and checked by every numerics run while no path from
the module layer, the `ops` package or the top-level `__init__` reached any of them. They were the
trimul in-projection's earlier front-ends, left behind as the front moved. Removed in the commit
that added `tests/registry/test_a_built_kernel_is_one_something_launches.py`, which is what will
catch the next one.

This file is the part worth keeping: what each was, why it was built, and what replaced it. The
code itself is in git.

## `triton/front.py` — the two-kernel B200 schedule

`trimul_gemm_gate_packed_mmajor_triton` (`_lr_kernel`) and `trimul_outproj_gemm_sigmoid_triton`
(`_gate_kernel`), launched together by `_front_launch`. 248,832 config benches per build, for a
launcher nothing imported.

Its finding is worth restating, because it is about schedules and not about this kernel. Profiling
showed the op is **not tensor-core bound**: the GEMM is ~0.014 ms while the epilogue stores dominate
(lr bdll = 512 MB, gate = 256 MB). The original single kernel held one fp32 accumulator of
(BM, 2D+D) = (256, 384) in tensor memory, which capped resident blocks so the long store latency
could not overlap across blocks — effective store bandwidth ~0.9 TB/s against a pure coalesced
write's ~6.4 TB/s.

Splitting it in two recovered most of that:

* **lr** on a 2-D grid (M-blocks × the left/right half), so each program carries only
  (BM, 2D) = (BM, 256). Weights for one half are interleaved `[g0 p0 g1 p1 …]`, so one fat GEMM
  plus a reshape/split recovers (g, p) locally for `sigmoid(g) * p`. The bdll store transposes to
  (D, BM) so consecutive rows m are contiguous (address `c*LL + m`) and coalesce. At BM=128,
  num_warps=4 this roughly doubles store overlap against the fused kernel.
* **gate** as a separate lighter kernel with a (BM, D) accumulator and a fully contiguous (M, D)
  store, which reaches ~1.6 TB/s on its own.

Keeping the two epilogues apart — rather than folding the gate's accumulator into the lr programs —
holds each kernel at its own occupancy sweet spot, and measured faster than any single fused launch
tried. The trade is that x is read twice, which is far cheaper than the occupancy a wider
accumulator costs.

Replaced by the cute front (`cute/launch.py: trimul_inproj_cute_forward`), which both
`bidir_training` paths take.

## `cute/front_sm100.py`, `cute/front_sm100_fused.py` — the bdll transpose front

`trimul_transpose_triton` (`_transpose_kernel`), reached through `trimul_front_sm100` and
`trimul_front_sm100_fused`. A blld → bdll layout flip beside the sm100 GEMM. Neither launcher had
an importer: the sm100 training path takes `trimul_front_sm100_train_sig`, whose fused kernel
writes bdll directly and needs no flip. See `notes/trimul_bidir_b200/v1.md` and `v2.md` for the
measurements that chose between them.

## `front_train_sm100.py: _glu_bdll` — the v13 fallback

`gated_projection_gate_packed_mmajor_triton` (`_glu_bdll_kernel`). v13's front was a quack
non-gated m-major GEMM producing `preact[4D]`, plus this Triton kernel to RE-READ preact and emit
`left/right[2D]`. **v14 fused the GLU into the GEMM epilogue** (`notes/trimul_train_b200/v14.md`),
so the re-read disappeared and this kernel became a fallback in `trimul_front_sm100_train` — a
function the sig path replaced. Its own driver had recorded the situation exactly: *"that launcher
is not used because its other half is the quack/sm100 front GEMM, which is not this kernel."*

## `back_fused.py: front_bwd_dW_glogit` — a recorded negative result

`trimul_bwd_gate_transpose_packed_triton` (`_dconcat5_kernel`). It folded `d_glogit` in as a fifth
`d_concat` block to collapse the single-direction back half from four cuBLAS GEMMs to two:

    dWs5 = dconc5 @ x_n     (5D, D)  -> dWLg/dWL/dWRg/dWR + dWg, all weight grads in one GEMM
    dx_n = dconc5ᵀ @ W_all  (M, D)   = dconcᵀ@W_stack + d_glogit@Wgᵀ, both terms in one GEMM

Tried and not adopted — the docstring said so and the registry did not act on it. The sm90 path
takes `front_bwd_dW`, the sm100 path `front_bwd_dW_sig`, and neither wants the five-block form.
