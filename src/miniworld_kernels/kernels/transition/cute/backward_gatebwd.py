"""Fused expand + SwiGLU-gate-BACKWARD GEMM for the Transition module, on quack SM90.

Mirrors the triton ``_transition_expand_gatebwd_kernel`` (triton/fused.py): recompute the
two pre-activations ONCE via a single dual-accumulator WGMMA and apply a gate-backward
epilogue that, given the per-pair upstream grad ``ge = grad_expand``, emits THREE tensors:

    a = xn @ Wa^T ; b = xn @ Wb^T            (recomputed; LN folded into the GEMM operand)
    sig = sigmoid(a) ; silu = a*sig
    dA  = ge * b * silu'(a)   (silu'(a) = sig + silu*(1-sig))
    dB  = ge * silu(a)
    h   = silu(a) * b                        (= the forward expand, for dWs = go^T @ h)

This is the *mirror image* of quack's ``GemmDGatedMixin`` (gemm_dact.py): there the GEMM
result IS the upstream grad and the pre-activations are the C operand; here the GEMM result
is the (LN-folded) pre-activations and ``grad_expand`` is the C operand. We compose:

  * the FORWARD ``GemmLnGatedMixin`` (gemm_transition_swiglu.py) prologue/LN-fold — the
    accumulator is the stacked [gate|up] = [a|b] with the LN affine recovered in registers,
  * the ``dswiglu(a, b, ge) -> (dA, dB, h)`` activation (quack activation.py),
  * the standard D output (2N wide = packed [dA|dB] interleaved) AND the gated PostAct
    (N wide = h) — both fire in the same epilogue (gemm_sm90.epilogue() has independent
    copy_D and copy_postact paths).

``grad_expand`` (M, N) is duplicated to interleaved (M, 2N) on the host so the trusted C
load path hands it back element-aligned to the [a|b] accumulator (tRS_rC[2i]==ge_i).
"""

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack.cute_dsl_utils import (
    mlir_namedtuple,
    torch2cute_dtype_map,
    get_device_capacity,
    get_max_active_clusters,
    ParamsBase,
)
from quack.epi_ops import RowVecLoad, ColVecLoad
from quack.gemm_sm90 import GemmSm90
from quack.gemm_act import GemmGatedMixin
from quack.activation import dgate_fn_map
from quack.rounding import RoundingMode
from quack.compile_utils import make_fake_tensor as fake_tensor
from miniworld_kernels.kernels._quack_compat import jit_cache
from quack.gemm_config import GemmConfig
from quack.gemm_tvm_ffi_utils import (
    perm3d_single,
    get_major,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    div_for_dtype,
    make_fake_gemm_tensors,
    compile_gemm_kernel,
)


# fmt: off
@triton.jit
def _cdup_interleave_kernel(g_ptr, o_ptr, M, N, N2, sgm, sgn, som, BM: tl.constexpr, BN: tl.constexpr):
    # Duplicate grad_expand (M,N) -> interleaved (M,2N): out[m,2j]=out[m,2j+1]=ge[m,j], so the
    # C operand aligns to the [a|b]-interleaved accumulator. Coalesced via tl.interleave (a
    # single contiguous 2*BN store), ~2x faster than the eager expand().reshape().contiguous().
    pidm = tl.program_id(0).to(tl.int64)
    pidn = tl.program_id(1).to(tl.int64)
    rows = pidm * BM + tl.arange(0, BM)
    cn = pidn * BN + tl.arange(0, BN)
    rm = rows < M
    v = tl.load(g_ptr + rows[:, None] * sgm + cn[None, :] * sgn, mask=rm[:, None] & (cn < N)[None, :], other=0.0)
    vi = tl.interleave(v, v)  # (BM, 2*BN) = [v0, v0, v1, v1, ...]
    co = pidn * 2 * BN + tl.arange(0, 2 * BN)
    tl.store(o_ptr + rows[:, None] * som + co[None, :], vi, mask=rm[:, None] & (co < N2)[None, :])
# fmt: on


def _cdup_interleave(ge: Tensor) -> Tensor:
    # ge may be a strided/transposed VIEW (col stride != 1); the kernel reads it with an
    # explicit col stride, fusing an upstream transpose into this (already-present) copy.
    M, N = ge.shape
    o = torch.empty(M, 2 * N, device=ge.device, dtype=ge.dtype)
    _cdup_interleave_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 128))](
        ge, o, M, N, 2 * N, ge.stride(0), ge.stride(1), o.stride(0), BM=64, BN=128
    )
    return o


class GemmDLnGatedMixin(GemmGatedMixin):
    """Epilogue: LN-fold affine over the stacked [a|b] accumulator, then SwiGLU-backward.

    Like ``GemmLnGatedMixin`` it adds ``mC1`` (per-m) + ``mB2`` (per-n) for the LN fold on
    top of the gated mixin's ``mRowVecBroadcast`` (=S) / ``mColVecBroadcast`` (=rstd). Unlike
    the forward, it (a) takes a real C operand = duplicated grad_expand, and (b) applies
    ``act_bwd_fn = dswiglu`` to emit (dA, dB) into the 2N D accumulator + h into PostAct.
    """

    _epi_ops = (*GemmGatedMixin._epi_ops, ColVecLoad("mC1"), RowVecLoad("mB2"))
    _extra_param_fields = (("act_bwd_fn", cutlass.Constexpr, None),)
    _epi_param_bases = (ParamsBase,)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Optional[object]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None  # S[n]   (stacked 2N, interleaved)
        mColVecBroadcast: Optional[cute.Tensor] = None  # rstd[m]
        mC1: Optional[cute.Tensor] = None               # c1[m] = mean*rstd
        mB2: Optional[cute.Tensor] = None               # B2[n]  (stacked 2N, interleaved)
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        # Mirror GemmGatedMixin: set up the gated postact tile (N//2) + dtype/layout.
        assert args.mPostAct.element_type.width == 16, "gated postact must be 16-bit"
        assert cutlass.utils.LayoutEnum.from_tensor(args.mPostAct).is_n_major_c()
        if self.arch == 90:
            assert self.cta_tile_shape_mnk[1] % 32 == 0, "gated SM90 needs tileN % 32 == 0"
        self.rounding_mode = args.rounding_mode
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        self.cta_tile_shape_postact_mn = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1] // 2,
        )
        d = self._epi_ops_to_params_dict(args)
        d["act_bwd_fn"] = args.act_bwd_fn
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        assert tRS_rC is not None, "gate-backward needs grad_expand as the C operand"
        rstd = epi_loop_tensors["mColVecBroadcast"]
        c1 = epi_loop_tensors["mC1"]
        S = epi_loop_tensors["mRowVecBroadcast"]
        B2 = epi_loop_tensors["mB2"]
        # 1) LN-fold affine on the full stacked accumulator: acc' = rstd*acc - c1*S + B2.
        #    After this, even cols = a (gate pre-act), odd cols = b (up pre-act).
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            tRS_rD[i] = rstd[i] * tRS_rD[i] - c1[i] * S[i] + B2[i]
        # 2) grad_expand (C operand) -> f32 registers, element-aligned to the 2N accumulator
        #    (host duplicated it: tRS_rC[2i] == tRS_rC[2i+1] == ge for pair i).
        tRS_rC_acc = cute.make_rmem_tensor_like(tRS_rC, self.acc_dtype)
        tRS_rC_acc.store(tRS_rC.load().to(self.acc_dtype))
        # 3) SwiGLU backward per pair: (dA, dB, h) = dswiglu(a, b, ge). dA/dB overwrite the
        #    2N accumulator IN PLACE (-> D output); h -> the N-wide gated postact.
        tRS_rPostAct_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rPostAct_layout.shape, self.acc_dtype)
        for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
            dA, dB, h = params.act_bwd_fn(
                tRS_rD[2 * i], tRS_rD[2 * i + 1], tRS_rC_acc[2 * i]
            )
            tRS_rD[2 * i] = dA
            tRS_rD[2 * i + 1] = dB
            tRS_rPostAct[i] = h
        return tRS_rPostAct

    # epi_convert_postact (permute_gated_Cregs_b16 on the N-wide h only) is inherited.


class GemmDLnGatedSm90(GemmDLnGatedMixin, GemmSm90):
    pass


@jit_cache
def _compile_gemm_dln_gatebwd(
    a_dtype, b_dtype, d_dtype, c_dtype, postact_dtype,
    a_major, b_major, d_major, c_major, postact_major,
    vec_dtype,
    tile_shape_mn, cluster_shape_mnk,
    pingpong, is_dynamic_persistent, device_capacity,
):
    act_bwd_fn = dgate_fn_map["swiglu"]
    GemmCls = GemmDLnGatedSm90
    # mD = (M, 2N) = packed [dA|dB] interleaved; mC = (M, 2N) = duplicated grad_expand.
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype, b_dtype, d_dtype, c_dtype, a_major, b_major, d_major, c_major
    )
    pa_n = cute.sym_int()  # gated postact width = n // 2
    div_pa = div_for_dtype(postact_dtype)
    mPostAct = fake_tensor(postact_dtype, (m, pa_n, l), leading_dim=1, divisibility=div_pa)
    mRowVec = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)  # S (2N)
    mB2 = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)
    mColVec = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)  # rstd
    mC1 = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)
    epi_args = GemmCls.EpilogueArguments(
        mPostAct,
        act_bwd_fn,
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        mC1=mC1,
        mB2=mB2,
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    varlen_args = make_fake_varlen_args(False, False, False, None)
    return compile_gemm_kernel(
        GemmCls, a_dtype, tile_shape_mn, cluster_shape_mnk,
        pingpong, True, False, is_dynamic_persistent, device_capacity,
        mA, mB, mD, mC, epi_args, scheduler_args, varlen_args,
    )


def gemm_dln_gatebwd(
    A: Tensor,        # (M, K) bf16, k-major  = xn (pre-normalized, fed as raw x@W2 via LN fold)
    B: Tensor,        # (2N, K) bf16, k-major = interleaved [gate|up] W2 = gamma ⊙ [Wa,Wb]
    D: Tensor,        # (M, 2N) bf16, n-major = packed [dA|dB] interleaved (OUTPUT)
    PostAct: Tensor,  # (M, N) bf16, n-major  = h = silu(a)*b (OUTPUT)
    C: Tensor,        # (M, 2N) bf16, n-major = duplicated grad_expand (INPUT, ge[2j]=ge[2j+1])
    rstd: Tensor,     # (1, M) fp32
    c1: Tensor,       # (1, M) fp32
    S: Tensor,        # (1, 2N) fp32  interleaved
    B2: Tensor,       # (1, 2N) fp32  interleaved
    *,
    config: GemmConfig | None = None,
) -> None:
    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] == 9, "SM90 (H100) only"
    if config is None:
        # tile_m=192 pingpong (1,2) is correct + fast across K here. NOTE: tile_m=256 (the
        # forward's K<=128 pick) gives WRONG results in THIS backward — the gated epilogue
        # with a real C operand + 3 outputs at tile_m=256 corrupts ~0.7% of elements
        # (cos~0.993); tile_m=192/128 are bit-exact (cos 1.00000). So we do NOT inherit the
        # forward's 256x128 config for the backward.
        config = GemmConfig(
            tile_m=192, tile_n=128, pingpong=True, is_dynamic_persistent=False,
            cluster_m=1, cluster_n=2, swap_ab=False, max_swizzle_size=8, device_capacity=9,
        )

    A3 = A.unsqueeze(0) if A.dim() == 2 else A
    B3 = B.unsqueeze(0) if B.dim() == 2 else B
    D3 = D.unsqueeze(0) if D.dim() == 2 else D
    C3 = C.unsqueeze(0) if C.dim() == 2 else C
    PA3 = PostAct.unsqueeze(0) if PostAct.dim() == 2 else PostAct
    A_p = perm3d_single(A3)
    B_p = perm3d_single(B3)
    D_p = perm3d_single(D3)
    C_p = perm3d_single(C3)
    PA_p = perm3d_single(PA3)
    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n")
    c_major = get_major(C_p, "m", "n")
    postact_major = get_major(PA_p, "m", "n")
    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype]
    c_dtype = torch2cute_dtype_map[C.dtype]
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    vec_dtype = torch2cute_dtype_map[rstd.dtype]
    is_dynamic_persistent = config.is_dynamic_persistent

    compiled_fn = _compile_gemm_dln_gatebwd(
        a_dtype, b_dtype, d_dtype, c_dtype, postact_dtype,
        a_major, b_major, d_major, c_major, postact_major,
        vec_dtype,
        (config.tile_m, config.tile_n),
        (config.cluster_m, config.cluster_n, 1),
        config.pingpong, is_dynamic_persistent, device_capacity,
    )

    from miniworld_kernels.kernels._quack_compat import is_compile_only

    if is_compile_only():
        return

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if (is_dynamic_persistent and device_capacity[0] == 9)
        else None
    )
    epi_args = GemmDLnGatedMixin.EpilogueArguments(
        PA_p,
        None,            # act_bwd_fn is Constexpr, baked at compile
        mRowVecBroadcast=S,
        mColVecBroadcast=rstd,
        mC1=c1,
        mB2=B2,
        rounding_mode=None,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters, config.max_swizzle_size, tile_count_semaphore
    )
    varlen_args = make_varlen_args(None, None, None)
    compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None)


def transition_expand_gatebwd_cute(
    xn: Tensor,           # (M, K) bf16 — PRE-NORMALIZED x (Version B: saved xn)
    grad_expand: Tensor,  # (M, N) bf16 — upstream grad of the expand (= go @ Ws)
    Wa: Tensor,           # (N, K) bf16 = expand_a.weight
    Wb: Tensor,           # (N, K) bf16 = expand_b.weight
    *,
    prefolded_B: Tensor | None = None,   # (2N, K) interleaved [Wa|Wb] (NO gamma fold for bwd)
    config: GemmConfig | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """SwiGLU-gate backward via one dual-accumulator WGMMA.

    Returns ``(h, dAB, Bw)``:
      * ``h``   (M, N)  = silu(a)*b           (for dWs = go^T @ h)
      * ``dAB`` (M, 2N) = interleaved [dA|dB]  (even cols = dA, odd cols = dB)
      * ``Bw``  (2N, K) = interleaved [Wa|Wb]

    The interleaved packing lets the downstream wgrad/d_xn GEMMs run as SINGLE contiguous
    GEMMs over the 2N dim instead of deinterleaving to two (M, N) tensors (which is a huge
    memory-bound copy at large N):

        d_xn      = dAB @ Bw                    # (M,2N)@(2N,K) = dA@Wa + dB@Wb
        dW_stack  = dAB^T @ xn                  # (2N,K), even rows = dWa, odd rows = dWb

    The pre-activations a,b are recomputed from the *already normalized* xn (Version B path:
    gamma/beta were applied when xn was produced), so the LN-fold affine is a no-op
    (rstd=1, c1=S=B2=0) and B = plain interleaved [Wa|Wb].
    """
    assert xn.is_cuda and xn.dim() == 2
    M, K = xn.shape
    N = Wa.shape[0]
    if prefolded_B is None:
        Bw = torch.empty(2 * N, K, dtype=xn.dtype, device=xn.device)
        Bw[0::2] = Wa
        Bw[1::2] = Wb
        Bw = Bw.contiguous()
    else:
        Bw = prefolded_B
    # LN-fold affine no-op: acc' = 1*acc - 0*S + 0  ==  acc.
    rstd = torch.ones(1, M, dtype=torch.float32, device=xn.device)
    c1 = torch.zeros(1, M, dtype=torch.float32, device=xn.device)
    S = torch.zeros(1, 2 * N, dtype=torch.float32, device=xn.device)
    B2 = torch.zeros(1, 2 * N, dtype=torch.float32, device=xn.device)
    # Duplicate grad_expand to interleaved (M, 2N) so C aligns to the [a|b] accumulator.
    C = _cdup_interleave(grad_expand)
    dAB = torch.empty(M, 2 * N, dtype=xn.dtype, device=xn.device)
    h = torch.empty(M, N, dtype=xn.dtype, device=xn.device)
    gemm_dln_gatebwd(xn, Bw, dAB, h, C, rstd, c1, S, B2, config=config)
    return h, dAB, Bw
