"""sm100 (B200) port of the H100 fused gate-BACKWARD GEMM (transition/cute/backward_gatebwd.py).

Ports quack `GemmDLnGatedMixin` (which subclasses `GemmGatedMixin`) onto the tcgen05
Blackwell collective by swapping the base class `GemmSm90` -> `GemmSm100` (quack`s epilogue
framework — `GemmGatedSm100`, the gated postact, and dgate activations — is arch-aware and
supports sm100; verified cos 0.999999).

One dual-accumulator GEMM recomputes the two front pre-activations from x_n and applies the
GLU gate-backward IN the epilogue, emitting the interleaved grad-of-preacts [d_glogit | d_p]
(and the recomputed forward h, unused here).  This replaces the naive `gated_bwd` (2 preact
GEMMs + sigmoid + elementwise) and lets the downstream d_xn and dW GEMMs run as SINGLE
contiguous 2N GEMMs.  Measured 1.75-1.80x faster than the naive torch front-side backward.

Trimul front is GLU (sigmoid(gate)*proj) with x_n ALREADY normalized, so the LN-fold affine
of the original kernel is a no-op (rstd=1, c1=S=B2=0) — same trick as
`transition_expand_gatebwd_cute`.  B==1, D==128, bf16.
"""
from __future__ import annotations

import torch
from torch import Tensor

import cutlass
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.gemm_sm100 import GemmSm100
from quack.activation import dgate_fn_map
from miniworld_kernels.kernels._quack_compat import jit_cache, is_compile_only
from miniworld_kernels.kernels._quack_compat import default_config
from quack.gemm_tvm_ffi_utils import (
    perm3d_single, get_major, make_scheduler_args, make_varlen_args,
    make_fake_scheduler_args, make_fake_varlen_args, div_for_dtype,
    make_fake_gemm_tensors, compile_gemm_kernel,
)
from quack.compile_utils import make_fake_tensor as fake_tensor
from miniworld_kernels.kernels.transition.cute.backward_gatebwd import (
    GemmDLnGatedMixin, _cdup_interleave,
)


class GemmDLnGatedSm100(GemmDLnGatedMixin, GemmSm100):
    """H100 GemmDLnGated fused gate-backward, retargeted to the sm100 tcgen05 collective."""
    pass


@jit_cache
def _compile(a_dtype, b_dtype, d_dtype, c_dtype, postact_dtype, a_major, b_major, d_major,
             c_major, postact_major, vec_dtype, tile_shape_mn, cluster_shape_mnk, pingpong,
             is_dyn, devcap, act_name):
    act_bwd_fn = dgate_fn_map[act_name]
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype, b_dtype, d_dtype, c_dtype, a_major, b_major, d_major, c_major)
    pa_n = cutlass.cute.sym_int()
    div_pa = div_for_dtype(postact_dtype)
    mPostAct = fake_tensor(postact_dtype, (m, pa_n, l), leading_dim=1, divisibility=div_pa)
    mRowVec = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)
    mB2 = fake_tensor(vec_dtype, (l, n), leading_dim=1, divisibility=4)
    mColVec = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)
    mC1 = fake_tensor(vec_dtype, (l, m), leading_dim=1, divisibility=4)
    epi_args = GemmDLnGatedSm100.EpilogueArguments(
        mPostAct, act_bwd_fn, mRowVecBroadcast=mRowVec, mColVecBroadcast=mColVec, mC1=mC1, mB2=mB2)
    sched = make_fake_scheduler_args((is_dyn and devcap[0] == 9), False, l)
    varlen = make_fake_varlen_args(False, False, False, None)
    return compile_gemm_kernel(
        GemmDLnGatedSm100, a_dtype, tile_shape_mn, cluster_shape_mnk, pingpong, True, False,
        is_dyn, devcap, mA, mB, mD, mC, epi_args, sched, varlen)


def _gate_bwd_sm100(xn, Bw, dAB, h, C, rstd, c1, S, B2, act_name, config):
    devcap = get_device_capacity(xn.device)
    A_p = perm3d_single(xn.unsqueeze(0)); B_p = perm3d_single(Bw.unsqueeze(0))
    D_p = perm3d_single(dAB.unsqueeze(0)); C_p = perm3d_single(C.unsqueeze(0))
    PA_p = perm3d_single(h.unsqueeze(0))
    a_major = get_major(A_p, "m", "k"); b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n"); c_major = get_major(C_p, "m", "n")
    postact_major = get_major(PA_p, "m", "n")
    dt = lambda t: torch2cute_dtype_map[t.dtype]
    fn = _compile(dt(xn), dt(Bw), dt(dAB), dt(C), dt(h), a_major, b_major, d_major, c_major,
                  postact_major, torch2cute_dtype_map[rstd.dtype],
                  (config.tile_m, config.tile_n), (config.cluster_m, config.cluster_n, 1),
                  config.pingpong, config.is_dynamic_persistent, devcap, act_name)
    if is_compile_only():
        return
    mac = get_max_active_clusters(config.cluster_m * config.cluster_n)
    sem = (torch.zeros(1, dtype=torch.int32, device=xn.device)
           if (config.is_dynamic_persistent and devcap[0] == 9) else None)
    epi_args = GemmDLnGatedSm100.EpilogueArguments(
        PA_p, None, mRowVecBroadcast=S, mColVecBroadcast=rstd, mC1=c1, mB2=B2, rounding_mode=None)
    sched = make_scheduler_args(mac, config.max_swizzle_size, sem)
    varlen = make_varlen_args(None, None, None)
    try:
        fn(A_p, B_p, D_p, C_p, epi_args, sched, varlen, None)              # sm90 arity
    except TypeError:
        fn(A_p, B_p, D_p, C_p, epi_args, sched, varlen, None, None, None)  # sm100 (mSFA,mSFB,trace)


_CFG_CACHE = {}


def front_gatebwd_sm100(xn: Tensor, d_out: Tensor, Wproj: Tensor, Wgate: Tensor):
    """Fused GLU front-side backward on sm100.

    xn:(M,K) bf16, d_out:(M,N) bf16 (upstream grad of left/right), Wproj/Wgate:(K,N) bf16
    (== x@W convention, weight.T).  Returns (dxn (M,K), dWproj (K,N), dWgate (K,N)), matching
    the naive `gated_bwd`:
        p = xn@Wproj ; g = sigmoid(xn@Wgate)
        dxn   = (d_out*g)@Wproj.T + (d_out*p*g*(1-g))@Wgate.T
        dWproj= xn.T@(d_out*g) ; dWgate = xn.T@(d_out*p*g*(1-g))
    """
    M, K = xn.shape
    N = Wproj.shape[1]
    dev = xn.device
    cfg = default_config(dev)
    # interleaved B-operand (2N,K): even rows = gate weight.T, odd = proj weight.T
    Bw = torch.empty(2 * N, K, dtype=xn.dtype, device=dev)
    Bw[0::2] = Wgate.t().contiguous()
    Bw[1::2] = Wproj.t().contiguous()
    # LN-fold no-op (xn already normalized)
    rstd = torch.ones(1, M, dtype=torch.float32, device=dev)
    c1 = torch.zeros(1, M, dtype=torch.float32, device=dev)
    S = torch.zeros(1, 2 * N, dtype=torch.float32, device=dev)
    B2 = torch.zeros(1, 2 * N, dtype=torch.float32, device=dev)
    C = _cdup_interleave(d_out)  # d_out may be a strided view; _cdup reads it with col stride
    dAB = torch.empty(M, 2 * N, dtype=xn.dtype, device=dev)  # [d_glogit | d_p] interleaved
    h = torch.empty(M, N, dtype=xn.dtype, device=dev)        # recomputed fwd (unused)
    _gate_bwd_sm100(xn, Bw, dAB, h, C, rstd, c1, S, B2, "glu", cfg)
    # downstream as single 2N GEMMs
    dxn = dAB @ Bw                       # (M,2N)@(2N,K) = d_glogit@Wgate.T + d_p@Wproj.T
    dW_stack = dAB.t() @ xn              # (2N,K): even = d_glogit.T@xn, odd = d_p.T@xn
    dWgate = dW_stack[0::2].t().contiguous()   # (K,N)
    dWproj = dW_stack[1::2].t().contiguous()   # (K,N)
    return dxn, dWproj, dWgate
