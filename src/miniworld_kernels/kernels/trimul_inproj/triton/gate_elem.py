"""GateElem — the second half of the SPLIT trimul back (forward + backward).

Forward (one GEMM, gate computed in-kernel, no gate materialization unless asked):

    gate = sigmoid(x_n @ Wg)               # (M, N)
    y    = proj ⊙ gate                     # (M, N)   proj from ① LayerNormLinear

Backward (given dy):

    d_proj   = dy ⊙ gate                                  -> feeds ① (grad into proj)
    d_glogit = dy ⊙ proj ⊙ gate ⊙ (1 - gate)             # sigmoid'
    dx_gate  = d_glogit @ Wgᵀ                             # gate-path grad into x_n
    dWg      = x_nᵀ @ d_glogit

Save design (training): save `gate` and `proj` (both M×N); `x_n` is already saved
(front/LN_in backward use it) so dWg needs no extra. No gate-GEMM recompute in bwd.
The elementwise (d_proj, d_glogit) is one fused triton pass (shares the dy/proj/gate
reads); the two backward GEMMs are cuBLAS (tiny/standard shapes → already optimal),
mirroring `triton/back_fused.py`.

x_n : (M, K=d_pair) row-major. proj/gate/y : (M, N) row-major. Wg : (K, N) = to_gate.weight.T.
B=1, bf16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_kernels.kernels.trimul_inproj.triton._autotune import get_seq_group


@triton.autotune(
    configs=[triton.Config({"BLK": b}, num_warps=nw) for b in (1024, 2048, 4096) for nw in (4, 8)],
    key=["GROUP_M", "N", "ADD_RESIDUAL", "USE_DROPOUT"],
)
@triton.jit
def _gate_mul_kernel(glogit_ptr, proj_ptr, y_ptr, gate_ptr, res_ptr, ds_ptr, n_elem, L,
                     N: tl.constexpr, BLK: tl.constexpr, SAVE_GATE: tl.constexpr,
                     GROUP_M: tl.constexpr, ADD_RESIDUAL: tl.constexpr, USE_DROPOUT: tl.constexpr):
    """Elementwise (1D, D-general): gate = sigmoid(glogit); y = proj ⊙ gate; [save gate].

    Optionally fuses the pairformer residual+dropout: y = residual + dropscale ⊙ (proj⊙gate).
    ``ds_ptr`` is the row-broadcast drop scale [L, N] (== drop_row mask/(1-p), broadcast over the
    i-index); for flat element ``off`` the scale is ds[(off//N) % L, off%N]. ``res_ptr`` is the
    module input pair [M, N]."""
    off = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK).to(tl.int64)
    mask = off < n_elem
    g = tl.sigmoid(tl.load(glogit_ptr + off, mask=mask, other=0.0).to(tl.float32))
    p = tl.load(proj_ptr + off, mask=mask, other=0.0).to(tl.float32)
    y = p * g
    if USE_DROPOUT:
        m = off // N
        n = off - m * N
        j = m % L
        ds = tl.load(ds_ptr + j * N + n, mask=mask, other=0.0).to(tl.float32)
        y = y * ds
    if ADD_RESIDUAL:
        y = y + tl.load(res_ptr + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + off, y.to(y_ptr.dtype.element_ty), mask=mask)
    if SAVE_GATE:
        tl.store(gate_ptr + off, g.to(gate_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BM": bm}, num_warps=nw, num_stages=ns)
        for bm in (32, 64, 128)
        for nw in (4, 8)
        for ns in (2, 3, 4)
    ],
    key=["GROUP_M", "N", "USE_DROPOUT"],
)
@triton.jit
def _gate_elem_bwd_ew_kernel(
    dy_ptr, proj_ptr, gate_ptr,    # (M, N)
    dproj_ptr, dglogit_ptr,        # (M, N) out
    ds_ptr, L,                     # row-broadcast drop scale [L, N] (used iff USE_DROPOUT)
    M, N: tl.constexpr, BM: tl.constexpr, GROUP_M: tl.constexpr,
    FROM_PREACT: tl.constexpr = False, USE_DROPOUT: tl.constexpr = False,
):
    """Fused elementwise: d_proj = dy⊙gate ; d_glogit = dy⊙proj⊙gate⊙(1-gate).
    One pass over (dy, proj, gate). If FROM_PREACT, `gate_ptr` holds the PREACT
    (glogit=x_n@Wg) instead of gate, and gate=sigmoid(preact) is recomputed here —
    lets the fused fwd (gate_elem_quack_fused) save preact instead of gate.

    If USE_DROPOUT, the incoming grad is scaled by the row-broadcast drop scale
    (dy_eff = dy ⊙ ds[m%L, n]) before the gate backward — the grad of y = ds ⊙ (proj⊙gate)."""
    pid = tl.program_id(0).to(tl.int64)
    # int64 M-index: off = m*N + n with M=B*L*L overflows int32 at large L.
    rm = pid.to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    rn = tl.arange(0, N)
    mmask = rm[:, None] < M
    off = rm[:, None] * N + rn[None, :]
    dy = tl.load(dy_ptr + off, mask=mmask, other=0.0).to(tl.float32)
    if USE_DROPOUT:
        j = rm % L
        ds = tl.load(ds_ptr + j[:, None] * N + rn[None, :], mask=mmask, other=0.0).to(tl.float32)
        dy = dy * ds
    proj = tl.load(proj_ptr + off, mask=mmask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + off, mask=mmask, other=0.0).to(tl.float32)
    if FROM_PREACT:
        gate = tl.sigmoid(gate)
    dproj = dy * gate
    dglogit = dy * proj * gate * (1.0 - gate)
    tl.store(dproj_ptr + off, dproj.to(dproj_ptr.dtype.element_ty), mask=mmask)
    tl.store(dglogit_ptr + off, dglogit.to(dglogit_ptr.dtype.element_ty), mask=mmask)


def gate_elem_triton(x_n, proj, Wg, *, return_gate: bool = False,
                     residual=None, dropscale=None, seq_len=None):
    """gate = sigmoid(x_n @ Wg); y = proj ⊙ gate. Returns y, or (y, gate) if return_gate.
    x_n:(M,K) or (B,L,L,K); proj:(M,N); Wg:(K,N)=to_gate.weight.T. B=1.

    ``residual`` [M,N] (== module input pair) and ``dropscale`` [L,N] (== drop_row mask/(1-p),
    broadcast over the i-index) optionally fuse the pairformer residual+dropout into the store:
    y = residual + dropscale ⊙ (proj⊙gate). ``seq_len`` (L) is required with dropscale.

    gate GEMM via cuBLAS (D-general), then a 1D elementwise (sigmoid + mul + residual/dropout)."""
    xn_flat = x_n.reshape(-1, x_n.shape[-1])
    M = xn_flat.shape[0]
    N = proj.shape[-1]
    proj_flat = proj.reshape(M, N)
    glogit = xn_flat @ Wg                                          # (M, N) cuBLAS
    y = torch.empty(M, N, device=xn_flat.device, dtype=xn_flat.dtype)
    gate = torch.empty(M, N, device=xn_flat.device, dtype=xn_flat.dtype) if return_gate \
        else proj_flat  # dummy (kernel won't write it)
    add_residual = residual is not None
    use_dropout = dropscale is not None
    L = int(seq_len) if seq_len is not None else (x_n.shape[1] if x_n.dim() == 4 else 0)
    res_flat = residual.reshape(M, N) if add_residual else proj_flat  # dummy ptr when off
    ds_flat = dropscale.reshape(L, N) if use_dropout else proj_flat   # dummy ptr when off
    n_elem = M * N
    grid = lambda meta: (triton.cdiv(n_elem, meta["BLK"]),)  # noqa: E731
    _gate_mul_kernel[grid](glogit, proj_flat, y, gate, res_flat, ds_flat, n_elem, L, N=N,
                           SAVE_GATE=return_gate, GROUP_M=get_seq_group(M),
                           ADD_RESIDUAL=add_residual, USE_DROPOUT=use_dropout)
    return (y, gate) if return_gate else y


def gate_elem_quack(x_n, proj, Wg, *, return_gate: bool = False):
    """Same as gate_elem_triton but the gate GEMM+sigmoid is ONE quack `gemm_act` launch
    (sigmoid fused into the GEMM epilogue → no separate glogit HBM round-trip), then a single
    elementwise `y = proj ⊙ gate`. x_n:(M,K)/(B,L,L,K); proj:(M,N); Wg:(K,N). B=1."""
    from quack.gemm_interface import gemm_act

    from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
    _bdll_patch.ensure_sigmoid_act()                       # register "sigmoid" in quack act map
    xn_flat = x_n.reshape(-1, x_n.shape[-1])               # (M, K), contiguous
    M, N = xn_flat.shape[0], proj.shape[-1]
    proj_flat = proj.reshape(M, N)
    _, gate = gemm_act(A=xn_flat, B=Wg, activation="sigmoid", store_preact=False)  # σ(x_n@Wg)
    gate = gate.reshape(M, N)
    y = proj_flat * gate                                   # elementwise mul (one aten kernel)
    return (y, gate) if return_gate else y


def gate_elem_quack_fused(x_n, proj, Wg, *, return_preact: bool = False):
    """FULLY-FUSED gate in ONE quack launch: y = sigmoid(x_n @ Wg) ⊙ proj, via the custom
    `act(A@B)⊙C` epilogue (C=proj). Kills the separate mul + the gate (M,N) round-trip.
    Returns y, or (y, preact=x_n@Wg) if return_preact (backward recomputes gate=σ(preact))."""
    from quack.gemm_interface import gemm_act

    from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch, _gate_mul_patch
    _bdll_patch.ensure_sigmoid_act()
    _gate_mul_patch.apply()
    xn_flat = x_n.reshape(-1, x_n.shape[-1])
    M, N = xn_flat.shape[0], proj.shape[-1]
    proj_flat = proj.reshape(M, N)
    preact, y = gemm_act(A=xn_flat, B=Wg, C=proj_flat, activation="sigmoid",
                         store_preact=return_preact)
    return (y.reshape(M, N), preact) if return_preact else y.reshape(M, N)


def gate_elem_bwd_ew(dy, proj, gate, *, from_preact: bool = False,
                     dropscale=None, seq_len=None):
    """Just the GateElem bwd ELEMENTWISE (no GEMMs): returns (d_proj, d_glogit), both (M,N).
    For the merged BidirBackHalf, which fuses dx_gate (=d_glogit@Wgᵀ) into the dxn GEMM and
    does dWg itself — so it wants d_glogit raw, not gate_elem_bwd's dx_gate/dWg.
    If from_preact, `gate` is actually the saved preact (glogit) and gate=σ(preact) is
    recomputed in-kernel (fused fwd path saves preact, not gate).

    ``dropscale`` [L,N] (+ ``seq_len``=L): scale the incoming grad by the row-broadcast drop
    scale before the gate bwd (grad of y = dropscale ⊙ (proj⊙gate))."""
    M, N = dy.reshape(-1, dy.shape[-1]).shape
    dy, proj, gate = dy.reshape(M, N), proj.reshape(M, N), gate.reshape(M, N)
    d_proj = torch.empty_like(dy)
    d_glogit = torch.empty_like(dy)
    use_dropout = dropscale is not None
    L = int(seq_len) if seq_len is not None else 0
    ds_flat = dropscale.reshape(L, N) if use_dropout else dy  # dummy ptr when off
    grid = lambda meta: (triton.cdiv(M, meta["BM"]),)  # noqa: E731
    _gate_elem_bwd_ew_kernel[grid](dy, proj, gate, d_proj, d_glogit, ds_flat, L, M, N=N,
                                   GROUP_M=get_seq_group(M), FROM_PREACT=from_preact,
                                   USE_DROPOUT=use_dropout)
    return d_proj, d_glogit


def gate_elem_bwd(dy, x_n, proj, gate, Wg):
    """Backward of GateElem. Returns (d_proj, dx_gate, dWg).
    dy/proj/gate:(M,N); x_n:(M,K); Wg:(K,N). d_proj feeds ①; dx_gate is the gate-path
    grad into x_n (sum it with the front/LN contributions upstream); dWg:(K,N)."""
    M, N = dy.reshape(-1, dy.shape[-1]).shape
    dy = dy.reshape(M, N)
    proj = proj.reshape(M, N)
    gate = gate.reshape(M, N)
    xn_flat = x_n.reshape(M, -1)
    d_proj = torch.empty_like(dy)
    d_glogit = torch.empty_like(dy)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]),)  # noqa: E731
    _gate_elem_bwd_ew_kernel[grid](dy, proj, gate, d_proj, d_glogit, dy, 0, M, N=N,
                                   GROUP_M=get_seq_group(M), USE_DROPOUT=False)
    dx_gate = d_glogit @ Wg.t()            # (M, K)  cuBLAS
    dWg = xn_flat.t() @ d_glogit           # (K, N)  cuBLAS
    return d_proj, dx_gate, dWg


class GateElem(torch.autograd.Function):
    """autograd.Function over 2D tensors. x_n:(M,K) proj:(M,N) Wg:(K,N) -> y:(M,N).
    Saves gate+proj+x_n (the simple training save design). Caller reshapes to/from 4D."""

    @staticmethod
    def forward(ctx, x_n, proj, Wg):
        # TRAINING: triton gate (cuBLAS gemm + one triton sigmoid·mul pass, gate saved FREE).
        # The fully-fused quack gate (gate_elem_quack_fused) is an INFERENCE-only win: training
        # needs `gate` for the bwd, and the fused path forces an extra preact write + a bwd
        # σ(preact) recompute that together cost MORE than the fwd fusion saves (measured
        # regression: d128 L1024 15.25→16.04). So keep triton for the GateElem autograd path.
        y, gate = gate_elem_triton(x_n, proj, Wg, return_gate=True)
        ctx.save_for_backward(x_n, proj, gate, Wg)
        return y

    @staticmethod
    def backward(ctx, dy):
        x_n, proj, gate, Wg = ctx.saved_tensors
        d_proj, dx_gate, dWg = gate_elem_bwd(dy, x_n, proj, gate, Wg)
        return dx_gate, d_proj, dWg
