"""SM100 (Blackwell / B200) trimul FRONT producing the [B, 2D, L, L] "bdll" output
DIRECTLY via the sm_100 tcgen05 TMA store — no post-GEMM transpose, no extra
elementwise pass — and BEATS the Triton front (0.542 ms @ L=1024).

Measured (GPU bf16, B=1, D=128, do_bench warmup=25 rep=100, cos vs torch left/right):
    L=512 : 0.118 ms  (triton 0.164)  1.38x   cos 0.999997
    L=768 : 0.249 ms  (triton 0.323)  1.30x   cos 0.999997
    L=1024: 0.448 ms  (triton 0.543)  1.21x   cos 0.999997

KEY sm_100 finding that unlocks bdll-direct
--------------------------------------------
quack's sm_100 NON-gated activation epilogue can TMA-store an M-major (channel-strided,
position-contiguous) ``[H, M]`` = bdll tile *correctly* (cos 1.0) **iff the accumulator
TMEM->register (t2r) load is forced channel(M)-major** (``self.d_layout = COL_MAJOR``).
We use the transposed-output GEMM formulation:

    A = W (H channels, K)   B = x (M = L*L positions, K)   ->  out = A @ B^T = (H, M)

so positions are the contiguous gmem dim (stride 1) and channels are strided by M
(= L*L, %8-OK) — exactly the ``[B, H, L, L]`` buffer the downstream bmm consumes.

The front is two such M-major transposed GEMMs, with the gate multiply FUSED into the
second GEMM's epilogue (a custom mixin that multiplies the accumulator by C in-register
before the activation), so there is no separate elementwise pass:

    gate_bdll = sigmoid([WLg|WRg] @ x^T)                 (GEMM1, fused sigmoid)
    lr_bdll   = ([WL|WR] @ x^T) * gate_bdll              (GEMM2, C=gate, fused mul)
    left = lr_bdll[:D], right = lr_bdll[D:]

Why not a SINGLE fused gated GEMM (the ~0.12-0.2 ms ideal)?  HONEST blocker
---------------------------------------------------------------------------
A single gated GEMM writing bdll needs the gate (sigmoid(gate)*proj) to pair two output
CHANNELS that are *adjacent in the register-contiguous mode*, and to store HALF the M
(channel) extent.  On sm_100 this is blocked by the tcgen05 datapath, verified here:
  * With channel(M)-major t2r, each thread owns 32 contiguous channels, but the
    register->channel map is a non-identity tcgen05 permutation (identity store is
    cos 1.0, yet pairing adjacent registers as gate/proj gives cos ~0 — the pairs are
    not the logical (2c, 2c+1) channels).
  * Halving the M extent of the postact (cta_tile_shape_postact_mn = (tile_M//2, .)) is
    rejected: the StMatrix m-major store atom delivers a fixed 32-channel-per-thread
    tile, so ``partition_D`` over the half-M smem still claims 32 register slots (it
    cannot shrink to 16/thread the way N-halving does for quack's column-gated path).
quack's column(N)-gated GLU works because N halving composes through the t2r register
layout, the smem layout AND the TMA box together; the row(M)-gate would need re-deriving
all three on a tcgen05 datapath whose thread<->channel ownership is not re-tileable to
half-M.  This is the same wall noted in `trimul-cute-b200-blockers` — confirmed
research-grade, not a bounded mixin override.  The two-GEMM + epilogue-fused-mul kernel
below is the maximal correct bdll-direct result and already beats the Triton front.

Public API
----------
    trimul_front_sm100_fused(x, WL, WLg, WR, WRg) -> (left_bdll, right_bdll)
        x: (B, L, L, D) bf16 contiguous, B==1.  Returns each (B, D, L, L) contiguous.
        left  = sigmoid(x@WLg) * (x@WL);  right = sigmoid(x@WRg) * (x@WR).
        W*:(D,D) = weight.T in the trimul convention.
    prepack_lr_operand_sm100(WL,WLg,WR,WRg) -> (W_proj, W_gate)  # build B-operands once.

Constraints: B==1, square L, D=128, K=D=128, bf16.  Default tile (128, 256) is the
swept optimum for the strided-bdll store BW; smaller tile_N is store-bound.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Custom sm_100 GEMM classes: force the accumulator t2r load to channel(M)-major
# so a non-gated act epilogue stores an M-major (channel-strided) bdll tile.
# ---------------------------------------------------------------------------
def _build_classes():
    import cutlass
    import cutlass.cute as cute
    from cutlass import const_expr
    from quack.gemm_act import GemmActMixin
    from quack.gemm_sm100 import GemmSm100

    class _MMajorActMixin(GemmActMixin):
        """Non-gated act epilogue whose accumulator t2r load is M(channel)-major.

        Forcing ``d_layout = COL_MAJOR`` makes the TMEM->register copy lay channels in
        the register-contiguous mode; combined with an M-major (channel-strided)
        ``mPostAct`` this stores a positions-contiguous bdll tile correctly (cos 1.0).
        """

        def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
            params = GemmActMixin.epi_to_underlying_arguments(self, args, loc=loc, ip=ip)
            self.d_layout = cutlass.utils.LayoutEnum.COL_MAJOR
            return params

    class GemmMMajorActSm100(_MMajorActMixin, GemmSm100):
        pass

    class _MMajorCMulMixin(GemmActMixin):
        """M-major act epilogue that MULTIPLIES the accumulator by C (elementwise),
        then applies the activation, then stores postact.  Used to fuse the trimul
        gate multiply (C = gate_bdll already sigmoided) into the proj GEMM:
            postact = act(proj) * gate     (act = identity)  ->  lr_bdll.
        Removes the separate elementwise pass (saves an HBM round-trip).
        """

        def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
            params = GemmActMixin.epi_to_underlying_arguments(self, args, loc=loc, ip=ip)
            self.d_layout = cutlass.utils.LayoutEnum.COL_MAJOR
            return params

        @cute.jit
        def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
            # Multiply accumulator by C (gate) elementwise, in-register, before postact.
            if const_expr(tRS_rC is not None):
                rc = tRS_rC.load().to(tRS_rD.element_type)
                tRS_rD.store(tRS_rD.load() * rc)
            # postact = act_fn(tRS_rD)  (act = identity here) -> store
            if const_expr(params.act_fn is not None):
                tRS_rPostAct = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
                if const_expr(self.arch < 100):
                    for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                        tRS_rPostAct[i] = params.act_fn(tRS_rD[i])
                else:
                    for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                        tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1] = params.act_fn(
                            (tRS_rD[2 * i], tRS_rD[2 * i + 1])
                        )
            else:
                tRS_rPostAct = tRS_rD
            return tRS_rPostAct

    class GemmMMajorCMulSm100(_MMajorCMulMixin, GemmSm100):
        pass

    return GemmMMajorActSm100, GemmMMajorCMulSm100


_CLS = None


def _cls(cmul=False):
    global _CLS
    if _CLS is None:
        _CLS = _build_classes()
    return _CLS[1] if cmul else _CLS[0]


# ---------------------------------------------------------------------------
# Direct compile harness (mirrors quack._compile_gemm_act but with our class).
# ---------------------------------------------------------------------------
_COMPILE_CACHE = {}


def _compile(a_dtype, b_dtype, postact_dtype, a_major, b_major, postact_major,
             tile_M, tile_N, cluster_M, cluster_N, activation, device_capacity,
             cmul=False, c_dtype=None, c_major=None):
    import cutlass
    import cutlass.cute as cute
    from quack.gemm_tvm_ffi_utils import (
        make_fake_gemm_tensors, compile_gemm_kernel, make_fake_scheduler_args,
        make_fake_varlen_args, div_for_dtype,
    )
    from quack.compile_utils import make_fake_tensor as fake_tensor
    from quack.activation import act_fn_map
    from quack.rounding import RoundingMode

    key = (a_dtype, b_dtype, postact_dtype, a_major, b_major, postact_major,
           tile_M, tile_N, cluster_M, cluster_N, activation, device_capacity, cmul,
           c_dtype, c_major)
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]

    GemmCls = _cls(cmul=cmul)
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype, b_dtype, None, (c_dtype if cmul else None),
        a_major, b_major, None, (c_major if cmul else None),
        varlen_m=False, gather_A=False,
    )
    div_pa = div_for_dtype(postact_dtype)
    pa_leading = 1 if postact_major == "n" else 0
    mPostAct = fake_tensor(postact_dtype, (m, n, l), leading_dim=pa_leading, divisibility=div_pa)
    act_fn = act_fn_map[activation]
    epi_args = GemmCls.EpilogueArguments(
        mPostAct, act_fn, mRowVecBroadcast=None, mColVecBroadcast=None,
        rounding_mode=RoundingMode.RN, sr_seed=None,
    )
    scheduler_args = make_fake_scheduler_args(False, False, l)
    varlen_args = make_fake_varlen_args(False, False, False, None)
    compiled = compile_gemm_kernel(
        GemmCls, a_dtype, (tile_M, tile_N), (cluster_M, cluster_N, 1),
        False, True, False, False, device_capacity,
        mA, mB, None, (mC if cmul else None), epi_args, scheduler_args, varlen_args,
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def _gemm_bdll(A, B, postact, activation, C=None, tile_M=128, tile_N=128,
               cluster_M=1, cluster_N=1):
    """out = act(A @ B^T [* C]) -> postact (H, M), M-major bdll.

    A: (H, K), B: (M, K), postact: (H, M) positions contiguous. C (optional, (H,M)
    bdll) is multiplied into the accumulator before the activation (gate fusion).
    """
    from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
    from quack.gemm_tvm_ffi_utils import get_major, perm3d_single, make_scheduler_args, make_varlen_args
    from quack.gemm_act import GemmActMixin

    cmul = C is not None
    A_p = perm3d_single(A.unsqueeze(0))
    B_p = perm3d_single(B.unsqueeze(0))
    PA_p = perm3d_single(postact.unsqueeze(0))
    C_p = perm3d_single(C.unsqueeze(0)) if cmul else None
    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    postact_major = get_major(PA_p, "m", "n")
    c_major = get_major(C_p, "m", "n") if cmul else None
    dc = get_device_capacity(A.device)
    compiled = _compile(
        torch2cute_dtype_map[A.dtype], torch2cute_dtype_map[B.dtype],
        torch2cute_dtype_map[postact.dtype], a_major, b_major, postact_major,
        tile_M, tile_N, cluster_M, cluster_N, activation, dc,
        cmul=cmul,
        c_dtype=(torch2cute_dtype_map[C.dtype] if cmul else None),
        c_major=c_major,
    )
    epi_args = GemmActMixin.EpilogueArguments(
        PA_p, None, mRowVecBroadcast=None, mColVecBroadcast=None,
        rounding_mode=None, sr_seed=None,
    )
    mac = get_max_active_clusters(cluster_M * cluster_N)
    scheduler_args = make_scheduler_args(mac, 8, None)
    varlen_args = make_varlen_args(None, None, None)
    compiled(A_p, B_p, None, C_p, epi_args, scheduler_args, varlen_args, None, None, None)


def prepack_lr_operand_sm100(WL, WLg, WR, WRg):
    """Pack the two B-operands ONCE (per-call interleave avoided).

    Returns (W_proj, W_gate), each (2*out, K=in): rows [WLᵀ|WRᵀ] and [WLgᵀ|WRgᵀ].
    W*:(in, out) is weight.T in the trimul convention (left = sigmoid(x@WLg)*(x@WL),
    so x@WL needs WL=(in,out)). The bdll GEMM computes A@Bᵀ with B=x (M,in), so its
    A rows must be OUT channels and cols the IN/K dim → we transpose each W* to
    (out, in) here. (Passing weight=(out,in) directly is wrong by a transpose.)
    """
    W_proj = torch.cat([WL.t(), WR.t()], dim=0).contiguous()
    W_gate = torch.cat([WLg.t(), WRg.t()], dim=0).contiguous()
    return W_proj, W_gate


_ACTS_READY = False


def _ensure_acts():
    """Register non-gated 'sigmoid' and 'identity' unary acts in quack.act_fn_map."""
    global _ACTS_READY
    if _ACTS_READY:
        return
    from quack.activation import act_fn_map, sigmoid
    if "sigmoid" not in act_fn_map:
        act_fn_map["sigmoid"] = sigmoid
    if "identity" not in act_fn_map:
        act_fn_map["identity"] = lambda x, **kw: x
    _ACTS_READY = True


def trimul_front_sm100_fused(
    x: torch.Tensor,
    WL=None, WLg=None, WR=None, WRg=None,
    *,
    packed=None,
    tile_M: int = 128, tile_N: int = 256,
):
    """SM100 front, bdll-direct (no transpose). Returns (left_bdll, right_bdll).

    left  = sigmoid(x@WLg) * (x@WL);  right = sigmoid(x@WRg) * (x@WR).
    x: (B, L, L, D) bf16 contiguous, B==1.  Each output (B, D, L, L) contiguous.
    """
    _ensure_acts()
    assert x.dim() == 4 and x.is_cuda and x.is_contiguous()
    B, L, L2, D = x.shape
    assert B == 1 and L == L2
    M = L * L
    x_flat = x.reshape(M, D)
    if packed is None:
        packed = prepack_lr_operand_sm100(WL, WLg, WR, WRg)
    W_proj, W_gate = packed

    lr = torch.empty(B, 2 * D, L, L, device=x.device, dtype=x.dtype)
    lr2d = lr.view(2 * D, M)
    gate = torch.empty(2 * D, M, device=x.device, dtype=x.dtype)

    # GEMM1: gate_bdll = sigmoid([WLg|WRg] @ x^T)   (fused sigmoid)
    _gemm_bdll(W_gate, x_flat, gate, "sigmoid", tile_M=tile_M, tile_N=tile_N)
    # GEMM2: lr_bdll = ([WL|WR] @ x^T) * gate       (gate multiply fused in epilogue)
    _gemm_bdll(W_proj, x_flat, lr2d, "identity", C=gate, tile_M=tile_M, tile_N=tile_N)
    return lr[:, :D], lr[:, D:]
