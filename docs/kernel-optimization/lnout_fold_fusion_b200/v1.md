# LN_out + @Wp fold-fusion (sm100 B200) — v1: investigation, NO SHIP (all matmuls already tcgen05)

Round on the trimul TRAINING forward node ① `LN_out + @Wp` (the last remaining "our" GEMM stage
not fused): the contraction output `tri` is LayerNorm'd over the hidden channel and projected by
`to_out` (`@Wp`). Goal was to make it "B200-specific (tcgen05/TMEM)" and fuse away the LN↔GEMM HBM
round-trip. **Outcome: no shipping change.** The fold-fusion is blocked in the cute-DSL and a naive
raw-CUDA port is 10× slower; more importantly, a PTX check proved every matmul in the path *already*
lowers to tcgen05+TMEM, so the "B200 port" was moot — the only real lever was memory-traffic fusion,
which does not clear the bar here.

## Environment
- B200 (sm_100, cap 10.0), idle box, torch 2.10.0+cu128, Triton 3.6.0, cutlass-dsl 4.4.2, nvcc 13.1.
- bf16 in / fp32 acc / bf16 out. Shape: L=1024 → M=L²=1048576, K=N=d_hidden=128. ncu `Duration`.

## Baseline (current shipping: `te_style._te_forward`)
`_ln_mat_kernel` (Triton LN-materialize, reads the m-major `view` strided) + cuBLAS `@Wp`:
| kernel | µs | DRAM BW |
|---|---|---|
| `_ln_mat_kernel` | 240 | 27% (m-major strided read: x[m,k] at m + k·M) |
| cuBLAS `@Wp` (`nvjet_tst_*`) | 121 | 54% |
| **total** | **361** | |

## Experiments
| v | approach | correct | speed | verdict |
|---|---|---|---|---|
| v1 | existing fold micro `LnGateGemm` (proj+gate 2-GEMM, `micro_lngate.py`) | cos 0.999997 | 710µs | ❌ 2× slower (untuned v10 experiment) |
| v2 | `LnProjGemm` = proj-only algebraic-fold tcgen05 GEMM (RAW x + Wp'=γ⊙Wp, LN affine in epilogue) | cos 0.999997 | 220µs | 🟡 GEMM part beats the 352µs it replaces — but needs μ/rstd |
| v3 | separate var_mean (Triton, m-major strided) | ✓ | 162µs → 385µs total | ❌ strided read loses to baseline |
| v4 | in-kernel stats: scalar `sA[(m,k,0)]` (staged composed smem) | — | — | ❌ compile fail (swizzled ComposedLayout rejects indexing) |
| v5 | in-kernel stats: `cute.slice_` / `sA[None,None,0]` / `local_partition` | — | — | ❌ compile fail (same) |
| v6 | in-kernel stats: manual mbarrier+TMA into PLAIN smem, then reduce | — | — | ❌ runtime HANG (mbarrier/TMA deadlock) |
| v7 | in-kernel stats: `make_tiled_copy_A`+`partition_S`(single-stage)+`cute.copy`+smem-atomic scatter | compiles | — | ❌ runtime HANG |
| v8 | v7 minus atomics (isolate cause) | compiles | — | ❌ still HANG → the `cute.copy` s2r is the culprit (tcgen05 keeps A in smem for UMMA; no ldmatrix-style register A) |
| v9 | in-kernel stats: scalar `sA1[m,k]` (single-stage composed) | — | — | ❌ compile fail |
| v10 | **raw CUDA** fused LN-WMMA-GEMM (`mma.sync`, plain smem, in-kernel stats, one x read) | cos 0.999997, μ/rstd cos 1.0 | **3745µs** | ❌ 10× slower (naive: 128KB dyn smem → 1 block/SM → ~6% occupancy, no cp.async pipeline) |
| v11 | **PTX verification**: does Triton `tl.dot` (sm_100) use tcgen05/TMEM? | — | — | ✅ **decisive**: it does (below) |

## v11 — the decisive finding
Triton 3.6.0 lowering of a bf16 `tl.dot` for sm_100 emits the full Blackwell 5th-gen tensor-core +
tensor-memory sequence (from the compiled PTX):
```
1  tcgen05.alloc.cta_group                     # TMEM allocation
8  tcgen05.mma.cta_group                        # 5th-gen tensor-core MMA
1  tcgen05.ld.sync.aligned.32x32b.x128.b32      # TMEM accumulator load
1  tcgen05.st.sync.aligned.32x32b.x128.b32      # TMEM store
2  tcgen05.commit / 2 tcgen05.wait / dealloc / relinquish_alloc_permit
```
So **every matmul in the trimul path already runs on tcgen05+TMEM**: cuBLAS `nvjet_*` (@Wp,
contraction, dW) are cuBLAS's Blackwell tensor-core kernels; Triton `tl.dot` auto-lowers to tcgen05
(above); the front GEMM is explicit cute-DSL tcgen05. The remaining "our" Triton kernels (LN, gate
elementwise, dconcat) have **no matmul**, so tcgen05/TMEM is inapplicable — they are memory-bound and
already run at BW.

## Conclusion — no ship
- The premise "the non-front kernels are generic, not B200-specific" is misleading: *generic
  cuBLAS/Triton matmuls ARE tcgen05 on Blackwell.* There was no hand-port to tcgen05 to do.
- The only genuine lever was **fusion** (cut the `x_normed` HBM round-trip, ~90µs of the 361µs). The
  fusions that mattered were already shipped (front σ(gate) `front_fused_gemm_b200/v2`, gate fwd
  `gate_forward_fused/v1`). This last one (LN_out+@Wp) is:
  - blocked in cute-DSL (v4–v9: cannot reduce the swizzled UMMA A-operand smem in-kernel; the H100
    `gemm_layernorm_linear_fused` does this via WGMMA `sA[m,0:K]` logical indexing, which does not
    port to tcgen05 — the sm100 module `ln_linear_sm100` is deliberately a two-kernel design and
    marks the single-kernel fold as an undone "later round"),
  - and not worth a raw-CUDA WMMA optimization project (v10 naive = 10× slower; would need many
    tuning rounds just to reclaim ~90µs of memory traffic, i.e. ~3-5% end-to-end).
- Note: `ln_linear_sm100.proj_gemm_sm100` (tcgen05 proj_only GEMM, used by the *inference* split
  back-half `back_split_sm100`) already exists; v2 above was an independent re-derivation of it. The
  *training* path keeps cuBLAS `@Wp` because the tcgen05 module is forward-only (no saved μ/rstd,
  no m-major dx, no dW/dγ/dβ backward).

**Kept the shipping path unchanged.** trimul kernels are already tensor-core-optimal for every matmul
on B200. All experiment artifacts were scratch on the dev box (`~/psk/ncu/`, not in-repo).
