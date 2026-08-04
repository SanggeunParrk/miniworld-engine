"""cute 1+4: fused dgrad GEMM (dY@W) + LN-norm-backward epilogue → dx.  [VERIFIED cos=1.0]

See docs/design/layernorm-linear-fused-dgrad-lnbwd.md. Forks the composable epilogue (GemmSm90 + GemmDefaultEpiMixin)
on a `dY @ W` GEMM and does the LN-normalize backward in the epilogue, so dx_normed (M,K) is
never written/read back from HBM. x̂ is fed as the C operand (tRS_rC).

    A = dY (M,N)   B = Wᵀ (K,N)   C = x̂ (M,K)   D = dx (M,K)
    dx̂ = acc·γ ;  c2 = meanₖ(dx̂) ;  c1 = meanₖ(dx̂·x̂) ;  dx = rstd·(dx̂ − c2 − x̂·c1)

This kernel produces ONLY dx. dγ/dβ/dW are derived in the composed backward (autograd.py) from
T = dYᵀ@x̂ (one wgrad GEMM) — db=Σ_m dY, dW=γ⊙T+outer(db,β), dγ=(W⊙T).sum(0), dβ=db@W — so NO
in-epilogue M-reduction (RowVecReduce doesn't exist in quack) and NO dx_normed round-trip anywhere.

KEY FIXES during bring-up (now resolved, cos 1.0 at K∈{128,256}, varied N):
 - GEMM orientation: quack computes A@Bᵀ, so dx_normed=dY@W needs B=Wᵀ (K,N), NOT W (N,K)
   (which silently computed dY@Wᵀ; N=K=d hid it).  cos 0→0.48.
 - SINGLE epilogue subtile: the default epi_tile_N = gcd(32,tile_N) = 32 splits K=128 into 4
   subtiles → each visit's warp_reduction over N saw only 32 cols → PARTIAL c1/c2 → cos≈0.48.
   We override `_compute_tile_shape_or_override` to force epi_tile = the full CTA tile so the
   whole K-row is in ONE subtile → single-pass reduce+apply. cos 0.48→1.0. Safe ONLY with
   atom_layout 1×1 (tile_m=64, non-pingpong); a cooperative 2×1 atom iterates N-first and a
   full-M epi_tile would mis-order (see the warning in that gemm_sm90 helper).
 - warp_reduction uses shuffle_sync_bfly (butterfly) → already broadcasts to all N lanes (the
   earlier "missing broadcast" hypothesis was wrong; the real cause was the subtile split).

Verified by dgrad_lnbwd_verify.py (srun). Wired into autograd.py LayerNormLinear backward (K≤128).

PERF (archived dgrad_lnbwd bench, H100 bf16, full backward fused vs unfused cuBLAS-dgrad+Triton-LN-bwd):
    M=16384  d=128 → 1.29x   |  d=256 → 1.20x
    M=65536  d=128 → 0.99x   |  d=256 → 0.73x
    M=262144 d=128 → 1.02x   |  d=256 → 0.75x
  d=128 wins/ties everywhere (saving the dx_normed round-trip + the separate LN-bwd kernel). d=256
  LOSES at large M: the full-N=256 epilogue subtile holds D+C (x̂) in smem, starving the mainloop
  (ab_stage≈2 even after forcing epi_stage=epi_c_stage=1), so the dgrad GEMM falls behind quack's
  tuned tile_m=128 cooperative gemm. The single-subtile invariant pins tile_m=64/atom-1×1, so we
  can't use the faster cooperative tile. NEXT to rescue d=256: load x̂ directly from gmem in the
  epilogue (M2's mX per-element pattern) instead of as the C operand → frees ~64KB epi-C smem →
  more ab_stages. Until then the wired default gates fused to K≤128.
"""

from __future__ import annotations

import operator
from typing import NamedTuple, Optional

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack.cute_dsl_utils import mlir_namedtuple, torch2cute_dtype_map, get_device_capacity, get_max_active_clusters
from quack.epi_ops import RowVecLoad, ColVecLoad, ColVecReduce, colvec_reduce_accumulate
from quack.gemm_sm90 import GemmSm90
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.rounding import RoundingMode
from quack.compile_utils import make_fake_tensor as fake_tensor
from miniworld_engine.kernels._quack_compat import jit_cache
from miniworld_engine.kernels._quack_compat import default_config
from quack.gemm_tvm_ffi_utils import (
    get_majors, get_dtypes, perm3d, make_scheduler_args, make_varlen_args,
    make_fake_scheduler_args, make_fake_varlen_args, make_fake_gemm_tensors, compile_gemm_kernel,
)

_LANES_IN_N = 4  # SM90 WGMMA epilogue: 4 lanes share an M row (ColVecReduce.end)


class _DgradLNBwdMixin(GemmDefaultEpiMixin):
    """dx = rstd·(acc·γ − meanₖ(acc·γ) − x̂·meanₖ(acc·γ·x̂)).  γ=RowVec(n), rstd=ColVec(m), x̂=C.

    CRITICAL: the LN-backward row-reduction (c1,c2 over the full K) + apply must happen in ONE
    epilogue subtile. The framework's default epi_tile_N = gcd(32, tile_N) = 32 → K=128 splits into
    4 subtiles, each visit sees only 32 cols → PARTIAL c1/c2 → dx cos≈0.5 (the iter-6 bug). We force
    epi_tile = the full CTA tile so the whole K-row lands in one subtile. Safe ONLY when atom_layout
    is 1×1 (tile_m=64 non-pingpong): with a cooperative 2×1 atom the accumulator iterates N-first and
    a full-M epi_tile would mis-order — see the warning in gemm_sm90._compute_tile_shape_or_override.
    """

    @classmethod
    def _compute_stages(cls, cta_tile_shape_mnk, epi_tile, a_dtype, b_dtype, d_dtype, c_dtype,
                        epilogue_args, smem_capacity, occupancy, warp_shape_mnk=None):
        """Single full-N epi subtile ⇒ epi_stage=epi_c_stage=1 suffices (no subtile double-buffer).
        The stock heuristic reserves 2+2 epi stages, which at K=256 starves the mainloop down to
        ~2 ab stages (slow GEMM). Reclaiming that smem ~doubles ab_stage → competitive dgrad."""
        epi_stage = 1
        epi_c_stage = 0 if c_dtype is None else 1
        d_bytes_per_stage = cute.size(epi_tile) * d_dtype.width // 8 if d_dtype is not None else 0
        # quack 0.5.0: epi_smem_bytes_per_stage(int) -> epi_smem_bytes(...).{unstaged,d_stage,c_stage}.
        esb = cls.epi_smem_bytes(epilogue_args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk)
        epi_bytes_per_stage = d_bytes_per_stage + esb.d_stage
        epi_bytes = esb.unstaged + epi_bytes_per_stage * epi_stage
        if c_dtype is not None:
            epi_bytes += cute.size(epi_tile) * c_dtype.width // 8 * epi_c_stage
        if esb.c_stage > 0:  # tile-load ops carry their own per-c-stage smem
            epi_bytes += esb.c_stage * epi_c_stage
        a_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (cute.size(a_shape) * a_dtype.width // 8
                              + cute.size(b_shape) * b_dtype.width // 8)
        remaining_bytes = smem_capacity // occupancy - 1024 - epi_bytes
        ab_stage = remaining_bytes // ab_bytes_per_stage
        return ab_stage, epi_stage, epi_c_stage

    @staticmethod
    def _compute_tile_shape_or_override(cta_tile_shape_mnk, atom_layout_mnk,
                                             element_type=None, epi_tile_override=None):
        # Force a single epilogue subtile (full N = K) so the per-row reduction+apply is single-pass.
        # Requires atom_layout 1×1 (assert below catches a mis-set config).
        assert atom_layout_mnk[0] == 1 and atom_layout_mnk[1] == 1, (
            "_DgradLNBwd needs atom_layout 1×1 (tile_m=64, non-pingpong) for a full-N epi subtile")
        return (cta_tile_shape_mnk[0], cta_tile_shape_mnk[1])

    # mC2red/mC1red: ColVecReduce gives correctly (m,n)→m -laid-out per-row accumulators via
    # epi_loop_tensors (begin allocates + zeros). We accumulate Σdx̂ and Σ(dx̂·x̂) in visit, then
    # — because tile_N=K is a SINGLE subtile — finalize the N-lane warp reduction IN visit (the
    # framework's ColVecReduce.end would only write to gmem, too late for the dx apply).
    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, RowVecLoad("mGamma"),
                ColVecReduce("mC2red"), ColVecReduce("mC1red"))
    _extra_param_fields = (("inv_k", Float32, Float32(1.0)),)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None   # unused slot (default)
        mColVecBroadcast: Optional[cute.Tensor] = None   # rstd[m]
        mGamma: Optional[cute.Tensor] = None             # gamma[n] (= K output col)
        mC2red: Optional[cute.Tensor] = None             # scratch (M,) for Σdx̂ (gmem ignored)
        mC1red: Optional[cute.Tensor] = None             # scratch (M,) for Σdx̂·x̂
        sr_seed: Optional[cute.Tensor] = None
        inv_k: Optional[Float32] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        d = self._epi_ops_to_params_dict(args)
        d["inv_k"] = args.inv_k
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        rstd = epi_loop_tensors["mColVecBroadcast"]   # per-element (broadcast over n)
        gamma = epi_loop_tensors["mGamma"]            # per-element (broadcast over m)
        c2buf = epi_loop_tensors["mC2red"]            # per-m accumulator (stride (1,0))
        c1buf = epi_loop_tensors["mC1red"]
        inv_k = params.inv_k
        nfrag = cute.size(tRS_rD)
        dxhat = cute.make_rmem_tensor_like(tRS_rD, Float32)
        xh = cute.make_rmem_tensor_like(tRS_rD, Float32)
        for i in cutlass.range(nfrag, unroll_full=True):
            dxhat[i] = tRS_rD[i].to(Float32) * gamma[i].to(Float32)
            xh[i] = tRS_rC[i].to(Float32)
        # per-m partial sums (n collapsed via the colvec stride-(1,0) layout)
        colvec_reduce_accumulate(self, c2buf, dxhat)
        colvec_reduce_accumulate(self, c1buf, dxhat, rScale=xh)
        # finalize across the N lanes, in register (single subtile → end is too late)
        c2f = cute.filter_zeros(c2buf)
        c1f = cute.filter_zeros(c1buf)
        for j in cutlass.range(cute.size(c2f), unroll_full=True):
            c2f[j] = cute.arch.warp_reduction(c2f[j], operator.add, threads_in_group=_LANES_IN_N)
            c1f[j] = cute.arch.warp_reduction(c1f[j], operator.add, threads_in_group=_LANES_IN_N)
        for i in cutlass.range(nfrag, unroll_full=True):
            tRS_rD[i] = rstd[i].to(Float32) * (dxhat[i] - c2buf[i] * inv_k - xh[i] * c1buf[i] * inv_k)
        return None


class _DgradLNBwdSm90(_DgradLNBwdMixin, GemmSm90):
    pass


@jit_cache
def _compile(a_dtype, b_dtype, d_dtype, c_dtype, a_major, b_major, d_major, c_major,
             vec_dtype, tile_mn, cluster_mnk, pingpong, persistent, is_dyn, device_capacity):
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype, b_dtype, d_dtype, c_dtype, a_major, b_major, d_major, c_major
    )
    mColVec = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)  # rstd
    mGamma = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)   # gamma over output K
    n_tiles = cute.sym_int()
    mC2red = fake_tensor(Float32, (l, m, n_tiles), leading_dim=2, divisibility=1)   # scratch (l,M,n_tiles)
    mC1red = fake_tensor(Float32, (l, m, n_tiles), leading_dim=2, divisibility=1)
    epi_args = _DgradLNBwdSm90.EpilogueArguments(
        mColVecBroadcast=mColVec, mGamma=mGamma, mC2red=mC2red, mC1red=mC1red, inv_k=Float32(1.0))
    sched = make_fake_scheduler_args((is_dyn and device_capacity[0] == 9), False, l)
    varlen = make_fake_varlen_args(False, False, False, None)
    return compile_gemm_kernel(
        _DgradLNBwdSm90, a_dtype, tile_mn, cluster_mnk, pingpong, persistent, False, is_dyn,
        device_capacity, mA, mB, mD, mC, epi_args, sched, varlen,
    )


def dgrad_lnbwd_cute(dY: Tensor, W: Tensor, xhat: Tensor, _gamma: Tensor, rstd: Tensor):
    """dx (M,K) = LN-backward(dY@W). dY (M,N), W (N,K), xhat=(x-mean)*rstd (M,K), gamma (K,), rstd (M,)."""
    dev = get_device_capacity(dY.device)
    assert dev[0] == 9, "SM90 only"
    M, N = dY.shape
    K = W.shape[1]
    cfg = default_config(dY.device)
    # tile_N must cover K (single-subtile reduction): use K (<=256) as tile_n.
    # tile_m=64 + non-pingpong → atom_layout 1×1, the only regime where forcing a full-N
    # epi subtile is safe (see _compute_tile_shape_or_override override above).
    tile_m = 64
    tile_mn = (tile_m, K)
    pingpong = True   # tile_m=64 pingpong → still atom_layout 1×1 (single full-N epi subtile holds)
    dx = torch.empty(M, K, device=dY.device, dtype=dY.dtype)
    # GEMM computes A @ Bᵀ (contract last dims). We want dx_normed = dY @ W (contract N), so
    # B must be Wᵀ (K,N): dY @ (Wᵀ)ᵀ = dY @ W. (Passing W (N,K) computes dY@Wᵀ — wrong.)
    Wt = W.t().contiguous()  # (K, N)
    A, B, C, Dt = dY.unsqueeze(0), Wt.unsqueeze(0), xhat.unsqueeze(0), dx.unsqueeze(0)
    A_p, B_p, D_p, C_p = perm3d(A, B, Dt, C)
    a_maj, b_maj, d_maj, c_maj = get_majors(A_p, B_p, D_p, C_p)
    a_dt, b_dt, d_dt, c_dt = get_dtypes(dY, W, dx, xhat)
    vec_dt = torch2cute_dtype_map[rstd.dtype]
    cluster_mnk = (1, 1, 1)
    fn = _compile(a_dt, b_dt, d_dt, c_dt, a_maj, b_maj, d_maj, c_maj, vec_dt,
                  tile_mn, cluster_mnk, pingpong, True,
                  cfg.is_dynamic_persistent, dev)
    from miniworld_engine.kernels._quack_compat import is_compile_only
    if is_compile_only():
        return dx
    mac = get_max_active_clusters(1)
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=dY.device)
        if (cfg.is_dynamic_persistent and dev[0] == 9) else None
    )
    rstd2 = rstd.float().contiguous().view(1, M)
    gamma2 = _gamma.float().contiguous().view(1, K)
    c2scratch = torch.empty(1, M, 1, dtype=torch.float32, device=dY.device)  # (l, M, n_tiles=1)
    c1scratch = torch.empty(1, M, 1, dtype=torch.float32, device=dY.device)
    epi_args = _DgradLNBwdSm90.EpilogueArguments(
        mColVecBroadcast=rstd2, mGamma=gamma2, mC2red=c2scratch, mC1red=c1scratch,
        inv_k=Float32(1.0 / K), add_to_output=None, rounding_mode=None,
    )
    sched = make_scheduler_args(mac, cfg.max_swizzle_size, tile_count_semaphore)
    varlen = make_varlen_args(None, None, None)
    fn(A_p, B_p, D_p, C_p, epi_args, sched, varlen, None)
    return dx
