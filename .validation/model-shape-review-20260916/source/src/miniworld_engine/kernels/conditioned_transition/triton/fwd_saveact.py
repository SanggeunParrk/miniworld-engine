"""The save-for-backward FORWARD pair for the token (d>=256) training path.

Two kernels, each fusing a GEMM with the elementwise that follows it, emitting the activations the
backward needs: expand+SwiGLU -> (h, ab), then squeeze+gate -> (y, out, scale). `training.py`'s
`_composed_fwd_train` is the only caller and the only thing that needs to be.

This file used to also hold a fully fused BACKWARD and the autograd Function that drove it,
`cond_transition_train_fused`. Nothing ever selected it -- `training.py` was always the production
default -- and its own measurement on an H100 is why: the fused-triton dgrad lost to
cuBLAS+elementwise by 1.6-7.1x at every stage. Six registry kernels existed only to serve it, and
one of those (`_wgrad_kernel`) was dead inside the dead path, since the fused backward did wgrad on
cuBLAS unconditionally and the `_WGRAD_BACKEND` switch it exported was never read. All of it is
gone; what survives is the half production runs.
"""

from __future__ import annotations
# The gate backward here was gated_projection's `_sigmul_bwd` minus the `.to(tl.float32)`
# on its loads. That is not a low-precision variant, it is a missing upcast: tl.sigmoid goes
# through tl.exp, which is @_check_dtype(fp32, fp64), so the kernel did not compile at all in
# bf16. Nothing reached it -- it has no bench path -- so the break sat here unnoticed until a
# driver launched it directly. Use the one that upcasts.
from miniworld_engine.kernels.gated_projection.triton.main import _sigmul_bwd
from miniworld_engine.autotune.configs import configs_for

import torch

from miniworld_engine.kernels._compile import opaque
import triton
import triton.language as tl

from miniworld_engine.kernels._tiles import tile_grid, tile_order



# Flat elementwise stages tile ONE axis (the linear element index) — canonical 1-D sweep,
# replacing the literal BLOCK=2048 each was launched with.

# shape_key's value is L -- the ATOM count (this family is level=atom in kernels/registry.csv) --
# never the row count, and never the flat element count, a kernel receives.
#
# WHERE THAT L COMES FROM, and it is the one thing to know about this family: every entry point
# here is handed an ALREADY-FLATTENED (M, K) activation -- modules/conditioned_transition/module.py
# does `x.reshape(-1, d)` before it calls -- so `length_of` of that 2-D matrix is M = B*A, which is
# the atom count A only when B == 1. The module therefore reads A off the un-flattened activation
# and passes it down as the `length` argument of every entry point; the entry point buckets it once
# and hands the result to the inner launchers as `shape_key`. `length=None` falls back to
# `length_of(x.shape)` == M for the direct callers that have no un-flattened tensor to read (the
# registry drivers/checkers, and train_12_345.py), which is exactly the old behaviour.
from miniworld_engine.autotune.shape_key import atom_key, both_key, length_of, pack, rows_of


# ============================================================================
# FORWARD (training): reuse the fused inference structure, emit saved tensors.
# ============================================================================



# fmt: off


@triton.autotune(configs=configs_for("cond_transition_expand_swiglu_saveact_triton"),
                 key=['shape_key'])
@triton.jit
def _fwd_expand_swiglu_kernel(
    x_ptr, wa_ptr, wb_ptr, h_ptr, ab_ptr,
    M, ND, K, ND2,
    stride_xm, stride_xk,
    stride_wn, stride_wk,    # Wa, Wb: (ND, K) row-major
    stride_hm, stride_hn,    # h:  (M, ND)
    stride_abm, stride_abn,  # ab: (M, 2*ND) packed [a | b]
    BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    shape_key,
):
    # Visit order, tuned: see kernels/_tiles.py. Same GEMM as composed.py's, saving [a|b] as well.
    pid_m, pid_n = tile_order(
        tl.program_id(0).to(tl.int64),
        tl.cdiv(M, BLOCK_M1), tl.cdiv(ND, BLOCK_N), GROUP_M)
    rows = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < ND
    a = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    b = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        x = tl.load(x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        wa = tl.load(wa_ptr + cols[None, :] * stride_wn + k[:, None] * stride_wk,
                     mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        wb = tl.load(wb_ptr + cols[None, :] * stride_wn + k[:, None] * stride_wk,
                     mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        a += tl.dot(x, wa, out_dtype=tl.float32, input_precision="tf32")
        b += tl.dot(x, wb, out_dtype=tl.float32, input_precision="tf32")
    h = a * tl.sigmoid(a) * b
    m = row_mask[:, None] & col_mask[None, :]
    tl.store(h_ptr + rows[:, None] * stride_hm + cols[None, :] * stride_hn, h, mask=m)
    # save pre-activations a, b (packed) for the backward swiglu-grad.
    tl.store(ab_ptr + rows[:, None] * stride_abm + cols[None, :] * stride_abn, a, mask=m)
    tl.store(ab_ptr + rows[:, None] * stride_abm + (cols + ND)[None, :] * stride_abn, b, mask=m)
# fmt: on


# BLOCK_K_DC tiles the (looped) DC contraction and comes from the CSV like every other tile here.


# fmt: off


@triton.autotune(configs=configs_for("cond_transition_squeeze_gate_saveact_triton"),
                 key=['shape_key'])
@triton.jit
def _fwd_squeeze_gate_kernel(
    h_ptr, cond_ptr, ws_ptr, wsc_ptr, bsc_ptr, y_ptr, out_ptr, scale_ptr,
    M, ND, D,
    DC: tl.constexpr,
    stride_hm, stride_hn,
    stride_cm, stride_cc,
    stride_sd, stride_sn,     # Ws: (D, ND)
    stride_scd, stride_scc,   # Wsc: (D, DC)
    stride_ym, stride_yd,
    stride_om, stride_od,     # out:  (M, D)
    stride_sm, stride_sc,     # scale:(M, D)
    BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K_ND: tl.constexpr, BLOCK_K_DC: tl.constexpr,
    GROUP_M: tl.constexpr,
    shape_key,
):
    # Visit order, tuned: see kernels/_tiles.py. Squeeze: the (M, ND) h is the big operand, Ws (D, ND) the small one.
    pid_m, pid_d = tile_order(
        tl.program_id(0).to(tl.int64),
        tl.cdiv(M, BLOCK_M1), tl.cdiv(D, BLOCK_N), GROUP_M)
    rows = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    dcols = pid_d * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    d_mask = dcols < D
    out_acc = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    for n0 in range(0, ND, BLOCK_K_ND):
        n = n0 + tl.arange(0, BLOCK_K_ND)
        n_mask = n < ND
        h = tl.load(h_ptr + rows[:, None] * stride_hm + n[None, :] * stride_hn,
                    mask=row_mask[:, None] & n_mask[None, :], other=0.0)
        ws_t = tl.load(ws_ptr + n[:, None] * stride_sn + dcols[None, :] * stride_sd,
                       mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        out_acc += tl.dot(h, ws_t, out_dtype=tl.float32, input_precision="tf32")
    scale = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    for c0 in range(0, DC, BLOCK_K_DC):
        dc = c0 + tl.arange(0, BLOCK_K_DC)
        dc_mask = dc < DC
        cond = tl.load(cond_ptr + rows[:, None] * stride_cm + dc[None, :] * stride_cc,
                       mask=row_mask[:, None] & dc_mask[None, :], other=0.0)
        wsc_t = tl.load(wsc_ptr + dcols[None, :] * stride_scd + dc[:, None] * stride_scc,
                        mask=dc_mask[:, None] & d_mask[None, :], other=0.0)
        scale += tl.dot(cond, wsc_t, out_dtype=tl.float32, input_precision="tf32")
    bsc = tl.load(bsc_ptr + dcols, mask=d_mask, other=0.0)
    scale += bsc[None, :]
    y = tl.sigmoid(scale) * out_acc
    m = row_mask[:, None] & d_mask[None, :]
    tl.store(y_ptr + rows[:, None] * stride_ym + dcols[None, :] * stride_yd, y, mask=m)
    tl.store(out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od, out_acc, mask=m)
    tl.store(scale_ptr + rows[:, None] * stride_sm + dcols[None, :] * stride_sc, scale, mask=m)
# fmt: on


def _fwd_expand_swiglu_fake(x, wa, wb, shape_key=None):
    """(M, ND) h and (M, 2*ND) ab -- ab packs the saved pre-activations [a | b]."""
    return (x.new_empty((x.shape[0], wa.shape[0])),
            x.new_empty((x.shape[0], 2 * wa.shape[0])))


@opaque(fake=_fwd_expand_swiglu_fake, name="conditioned_transition_train_fused_expand_swiglu")
def _fwd_expand_swiglu(x: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
                       shape_key: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """a = x@Waᵀ, b = x@Wbᵀ, h = silu(a)*b -- the expand half, both GEMMs in one launch.
    Training forward: the SwiGLU is the GEMM epilogue, and the pre-activations are packed into ab
    because the backward's silu'(a) needs a and b, which h alone cannot give back.
    """
    M, K = x.shape
    if shape_key is None:
        raise ValueError(
            "shape_key is required here: this launcher receives an already-flattened "
            "(M, D) matrix, and M alone cannot say whether it is L or L*L. Compute the key "
            "at the caller that still holds the pre-flatten shape -- atom_key(length_of(x.shape)) "
            "-- and pass it down. The `None` default is the signature the @opaque fakes share, "
            "not a working fallback: length_of refuses a rank-2 shape."
        )
    ND = wa.shape[0]
    h = torch.empty(M, ND, device=x.device, dtype=x.dtype)
    ab = torch.empty(M, 2 * ND, device=x.device, dtype=x.dtype)
    grid = lambda meta: tile_grid(M, ND, meta["BLOCK_M1"], meta["BLOCK_N"])  # noqa: E731
    _fwd_expand_swiglu_kernel[grid](
        x, wa, wb, h, ab, M, ND, K, 2 * ND,
        x.stride(0), x.stride(1), wa.stride(0), wa.stride(1),
        h.stride(0), h.stride(1), ab.stride(0), ab.stride(1),
        shape_key=pack(shape_key, ND=ND, K=K),
    )
    return h, ab


def _fwd_squeeze_gate_fake(h, cond, ws, wsc, bsc, shape_key=None):
    """y, out and scale, all (M, D) -- D = ws.shape[0], the squeeze output width, not h's ND."""
    return (h.new_empty((h.shape[0], ws.shape[0])),
            h.new_empty((h.shape[0], ws.shape[0])),
            h.new_empty((h.shape[0], ws.shape[0])))


@opaque(fake=_fwd_squeeze_gate_fake, name="conditioned_transition_train_fused_squeeze_gate")
def _fwd_squeeze_gate(h: torch.Tensor, cond: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor,
                      shape_key: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """out = h@Wsᵀ, scale = cond@Wscᵀ + bsc, y = sigmoid(scale)*out -- the squeeze half, one launch.
    Training forward: out and scale are materialized next to y because the gate backward and the
    dWs/dWsc wgrads both read them.
    """
    M, ND = h.shape
    if shape_key is None:
        raise ValueError(
            "shape_key is required here: this launcher receives an already-flattened "
            "(M, D) matrix, and M alone cannot say whether it is L or L*L. Compute the key "
            "at the caller that still holds the pre-flatten shape -- atom_key(length_of(x.shape)) "
            "-- and pass it down. The `None` default is the signature the @opaque fakes share, "
            "not a working fallback: length_of refuses a rank-2 shape."
        )
    D = ws.shape[0]
    DC = cond.shape[1]
    y = torch.empty(M, D, device=h.device, dtype=h.dtype)
    out = torch.empty(M, D, device=h.device, dtype=h.dtype)
    scale = torch.empty(M, D, device=h.device, dtype=h.dtype)
    grid = lambda meta: tile_grid(M, D, meta["BLOCK_M1"], meta["BLOCK_N"])  # noqa: E731
    _fwd_squeeze_gate_kernel[grid](
        h, cond, ws, wsc, bsc, y, out, scale, M, ND, D, DC,
        h.stride(0), h.stride(1), cond.stride(0), cond.stride(1),
        ws.stride(0), ws.stride(1), wsc.stride(0), wsc.stride(1),
        y.stride(0), y.stride(1), out.stride(0), out.stride(1),
        scale.stride(0), scale.stride(1),
        shape_key=pack(shape_key, ND=ND, D=D, DC=DC),
    )
    return y, out, scale


# ============================================================================
# BACKWARD dgrad (outputs (M,*)) — fuse elementwise into the consuming GEMM.
# ============================================================================

# --- gate-bwd: dout = sg*dy ; dscale = out*sg*(1-sg)*dy  (one HBM pass over (M,D)) ---


