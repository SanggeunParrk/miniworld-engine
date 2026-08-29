"""Training (forward-saving + backward) for the post-AdaLN ConditionedTransition tail.

Mirrors the trimul split: the forward-only path is "inference" (saves nothing); this
"forward" saves the minimum for backward. The math (all fp32, TF32 tensor cores):

    a = x @ Wa^T ; b = x @ Wb^T                 # (M, ND)
    h = silu(a) * b                             # SwiGLU
    out = h @ Ws^T                              # (M, D)
    scale = cond @ Wsc^T + b_sc                 # (M, D)
    y = sigmoid(scale) * out

Backward (sg = sigmoid(scale), sa = sigmoid(a)):
    dout   = sg * dy
    dscale = out * sg * (1 - sg) * dy
    dcond  = dscale @ Wsc        ; dWsc = dscale^T @ cond ; db_sc = dscale.sum(0)
    dh     = dout @ Ws           ; dWs  = dout^T @ h
    silu'(a) = sa + silu(a)*(1 - sa) = sa*(1 + a*(1 - sa))
    da = dh * b * silu'(a)       ; db = dh * silu(a)
    dx = da @ Wa + db @ Wb       ; dWa = da^T @ x ; dWb = db^T @ x

GEMMs go through cuBLAS (torch.matmul, TF32). Triton fuses the two elementwise stages:
the forward SwiGLU + gate, and the backward gate-grad + SwiGLU-grad.
"""

from __future__ import annotations
# _gate_fwd_kernel / _gate_bwd_kernel used to be defined here. They were bitwise equal to
# bias_only_attention's copies (.bench/direct.out), so both now come from one home.
from miniworld_engine.kernels.gated_projection.triton.main import _sigmul_bwd, _sigmul_fwd
from miniworld_engine.autotune.configs import configs_for

import torch

from miniworld_engine.kernels._compile import opaque
import triton
import triton.language as tl

from miniworld_engine.kernels._tiles import check_tile_axes, tile_grid, tile_order



# autograd Functions cannot be @typecheck'd cleanly; keep precision policy explicit.

# The flat elementwise stages tile ONE axis (the linear element index), so their config space is
# the canonical 1-D block sweep. Each used to be launched with a literal BLOCK (2048 / 1024) —
# a hardcoded tile that no measurement chose and that no card can move.



# --- forward elementwise: h = silu(a)*b , scale = s@... already done, gate -----------
# FLAT 1-D, not a 2-D tile. The 2-D form was tried on the argument that a/b are strided views
# of the packed ab, so a flat index has to recover (row, col) with a runtime `//ND` and `%ND`
# per element. That argument is real but it optimises the wrong resource: this kernel is
# bandwidth-bound, and the flat index is fully coalesced whereas a (BLOCK_M1, BLOCK_N) tile of a
# strided view breaks each row into its own segment. MEASURED on an A6000 at M=4096/ND=512:
# 2-D cost _swiglu_fwd 467us + _swiglu_bwd 822us of GPU time against <100us each for the flat
# form — 1.5x on the whole conditioned_transition step (3406us -> 5107us of kernel time, same
# 83 launches, so it is GPU work and not launch overhead). Trading ALU for bandwidth is the
# wrong direction here. BLOCK_E still comes from the config space, so the tile is tuned either way.
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


@triton.autotune(configs=configs_for("cond_transition_swiglu_triton"), key=['shape_key'])
@triton.jit
def _swiglu_fwd_kernel(
    a_ptr, b_ptr, h_ptr, M, ND,
    stride_m, stride_n,      # a, b: (M, ND), possibly strided views (same strides)
    BLOCK_E: tl.constexpr,
    shape_key,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offs < M * ND
    row = offs // ND
    col = offs % ND
    idx = row * stride_m + col * stride_n
    a = tl.load(a_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    h = a * tl.sigmoid(a) * b
    tl.store(h_ptr + offs, h.to(h_ptr.dtype.element_ty), mask=mask)   # h is contiguous (M, ND)




def _swiglu_fake(a, b, shape_key=None):
    """(M, ND) h -- same shape as `a`, but contiguous where `a` may be a strided view."""
    return a.new_empty(a.shape)


@opaque(fake=_swiglu_fake, name="conditioned_transition_swiglu_fwd")
def _swiglu(a: torch.Tensor, b: torch.Tensor,
            shape_key: int | None = None) -> torch.Tensor:
    """h = silu(a)*b, reading a and b through their shared strides and writing h contiguous."""
    M, ND = a.shape
    if shape_key is None:
        raise ValueError(
            "shape_key is required here: this launcher receives an already-flattened "
            "(M, D) matrix, and M alone cannot say whether it is L or L*L. Compute the key "
            "at the caller that still holds the pre-flatten shape -- atom_key(length_of(x.shape)) "
            "-- and pass it down. The `None` default is the signature the @opaque fakes share, "
            "not a working fallback: length_of refuses a rank-2 shape."
        )
    h = torch.empty(M, ND, device=a.device, dtype=a.dtype)  # contiguous output
    grid = lambda meta: (triton.cdiv(M * ND, meta["BLOCK_E"]),)  # noqa: E731
    _swiglu_fwd_kernel[grid](a, b, h, M, ND, a.stride(0), a.stride(1),
                             shape_key=pack(shape_key, ND=ND))
    return h


def _gate_fake(out, scale, shape_key=None):
    """(M, D) y -- same shape as `out`."""
    return torch.empty_like(out)


@opaque(fake=_gate_fake, name="conditioned_transition_gate_fwd")
def _gate(out: torch.Tensor, scale: torch.Tensor,
          shape_key: int | None = None) -> torch.Tensor:
    """y = sigmoid(scale)*out, through the gated_projection ``_sigmul_fwd`` launch."""
    y = torch.empty_like(out)
    n = out.numel()
    if shape_key is None:
        # `n` is a flat element count (M*D), not a shape. The kernel is `_sigmul_fwd`, which belongs
        # to the gated_projection family (registry level=BOTH), so it keys against the union bucket
        # set -- otherwise the same L would bucket differently depending on the launching family.
        shape_key = both_key(rows_of(out.shape))
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_E"]),)  # noqa: E731
    _sigmul_fwd[grid](scale, out, y, n, shape_key=shape_key)
    return y


# --- backward elementwise (fused) ----------------------------------------------------


# FLAT 1-D — same measurement as _swiglu_fwd_kernel; this one was the larger of the two
# regressions (822us of GPU time as a 2-D tile).
@triton.autotune(configs=configs_for("cond_transition_bwd_swiglu_flat_triton"), key=['shape_key'])
@triton.jit
def _swiglu_bwd_kernel(
    a_ptr, b_ptr, dh_ptr, dab_ptr, M, ND,
    stride_m, stride_n,          # a, b: (M, ND) (possibly strided views, same strides)
    stride_dhm, stride_dhn,      # dh: (M, ND) (own strides — may differ from a/b)
    stride_pm, stride_pn,        # dab: (M, 2*ND) packed [da | db]
    BLOCK_E: tl.constexpr,
    shape_key,
):
    # Packs da into dab[:, :ND] and db into dab[:, ND:] so the expand-bwd is one GEMM.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offs < M * ND
    row = offs // ND
    col = offs % ND
    a = tl.load(a_ptr + row * stride_m + col * stride_n, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * stride_m + col * stride_n, mask=mask, other=0.0).to(tl.float32)
    dh = tl.load(dh_ptr + row * stride_dhm + col * stride_dhn, mask=mask, other=0.0).to(tl.float32)
    sa = tl.sigmoid(a)
    silu = a * sa
    silu_prime = sa * (1.0 + a * (1.0 - sa))  # sa + silu*(1 - sa)
    da = dh * b * silu_prime
    db = dh * silu
    base = row * stride_pm
    tl.store(dab_ptr + base + col * stride_pn, da.to(dab_ptr.dtype.element_ty), mask=mask)
    tl.store(dab_ptr + base + (col + ND) * stride_pn, db.to(dab_ptr.dtype.element_ty), mask=mask)


def _gate_bwd_fake(out, scale, dy, shape_key=None):
    """dout and dscale, both (M, D) like `out`."""
    return torch.empty_like(out), torch.empty_like(out)


@opaque(fake=_gate_bwd_fake, name="conditioned_transition_gate_bwd")
def _gate_bwd(out: torch.Tensor, scale: torch.Tensor, dy: torch.Tensor,
              shape_key: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """dout = sg*dy and dscale = out*sg*(1-sg)*dy for sg = sigmoid(scale), one pass over (M, D),
    through the gated_projection ``_sigmul_bwd`` launch."""
    dout = torch.empty_like(out)
    dscale = torch.empty_like(out)
    n = out.numel()
    if shape_key is None:
        shape_key = both_key(rows_of(out.shape))   # `_sigmul_bwd` is gated_projection, level=both
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_E"]),)  # noqa: E731
    _sigmul_bwd[grid](dy, scale, out, dscale, dout, n, shape_key=shape_key)
    return dout, dscale


# NB: a "fused" gate-bwd that also reduces db_sc=dscale.sum(0) in-kernel was tried and
# REGRESSED hard (full-D tile + tl.sum: token 1.25->0.33x compile). torch's dscale.sum(0)
# is a fast cuBLAS-adjacent reduction; keep it separate.


def _swiglu_bwd_packed_fake(a, b, dh, shape_key=None):
    """(M, 2*ND) -- the packed [da | db], twice the width of `a`."""
    return a.new_empty((a.shape[0], 2 * a.shape[1]))


@opaque(fake=_swiglu_bwd_packed_fake, name="conditioned_transition_swiglu_bwd_packed")
def _swiglu_bwd_packed(a: torch.Tensor, b: torch.Tensor, dh: torch.Tensor,
                       shape_key: int | None = None) -> torch.Tensor:
    """Return dab = [da | db] : (M, 2*ND), contiguous, for a single concatenated expand-bwd GEMM."""
    M, ND = a.shape
    if shape_key is None:
        raise ValueError(
            "shape_key is required here: this launcher receives an already-flattened "
            "(M, D) matrix, and M alone cannot say whether it is L or L*L. Compute the key "
            "at the caller that still holds the pre-flatten shape -- atom_key(length_of(x.shape)) "
            "-- and pass it down. The `None` default is the signature the @opaque fakes share, "
            "not a working fallback: length_of refuses a rank-2 shape."
        )
    dab = torch.empty(M, 2 * ND, device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M * ND, meta["BLOCK_E"]),)  # noqa: E731
    _swiglu_bwd_kernel[grid](
        a, b, dh, dab, M, ND,
        a.stride(0), a.stride(1),
        dh.stride(0), dh.stride(1),
        dab.stride(0), dab.stride(1),
        shape_key=pack(shape_key, ND=ND),
    )
    return dab


# ============================================================================
# FUSED TRITON FORWARD for training (emits the tensors backward needs).
# Reuses the inference fused structure (GEMM<->elem fusion, no h/out/scale HBM re-read
# between GEMMs) but additionally WRITES ab=[a|b], h, out, scale for the backward.
# ============================================================================

# --- atom (d<=128): single-kernel b2b forward, emits ab,h,out,scale,y -----------------
# EVERY tile axis is now a searched knob. The old grid tuned only BLOCK_M1/BLOCK_N and pinned the
# rest at the launch site or to a shape: BLOCK_K = next_power_of_2(K) (whole-K register tile, no
# K loop at all), BLOCK_DC = min(128, next_power_of_2(DC)), and the D axis was not tiled — `D`
# itself was the extent of `tl.arange(0, D)` and of the `(BLOCK_M1, D)` accumulators, so a wider D
# silently grew the register tile and there was no config the tuner could move.
#
# Now: the K contraction loops in BLOCK_K tiles, D is tiled by BLOCK_D over grid axis 1 (as in
# `composed.py::_squeeze_gate_kernel`), and the DC contraction reuses BLOCK_K. Reusing BLOCK_K for
# the DC loop is deliberate: both are contraction widths of the same kernel and both draw from the
# same candidate set, and a fifth independent axis would multiply this grid by another 5x (15k ->
# 75k configs) for a loop that costs a few percent of the runtime.
#
# The D-tiling makes the expand half (a/b/h) per-d-tile work, so it is recomputed by every d
# program; the `pid_d == 0` guard keeps the saved-tensor stores single-writer. With BLOCK_D >= D
# (which the tuner reaches: D <= 128 on this atom path and BLOCK_D sweeps up to 256) there is one
# d program and the recompute is zero — i.e. the previous behaviour is still inside the space.


# fmt: off


# `D` is GONE from this kernel, not merely unkeyed. It came from `ws.shape[0]` and `K` from
# `x.shape[1]`, and both are the module's d_hidden: ConditionedTransition's docstring states it
# ("K = D = d_hidden"), its Linears are expand (d_hidden -> n*d_hidden) / squeeze
# (n*d_hidden -> d_hidden), and every launcher in the repo (module.py,
# drivers.conditioned_transition._ct_args, checks.conditioned_transition) builds ws as (D, ND)
# with D == K. Its one read was the squeeze output mask, which `K` states exactly as well. ND is
# a different matter and stays keyed: n varies per module (2 here, 4 in transition/), and the
# driver harness perturbs ND on its own axis, so ND is not recoverable from K.
@triton.autotune(configs=configs_for("cond_transition_fwd_b2b_saveact_triton"),
                 key=['shape_key'])
@triton.jit
def _b2b_fwd_train_kernel(
    x_ptr, cond_ptr, wa_ptr, wb_ptr, ws_ptr, wsc_ptr, bsc_ptr,
    y_ptr, ab_ptr, h_ptr, out_ptr, scale_ptr,
    M, ND,
    K: tl.constexpr, DC: tl.constexpr,
    stride_xm, stride_xk,
    stride_cm, stride_cc,
    stride_wn, stride_wk,     # Wa, Wb: (ND, K)
    stride_sd, stride_sn,     # Ws: (D, ND)
    stride_scd, stride_scc,   # Wsc: (D, DC)
    stride_ym, stride_yd,
    stride_abm, stride_abn,   # ab: (M, 2*ND) packed [a|b]
    stride_hm, stride_hn,     # h:  (M, ND)
    stride_om, stride_od,     # out:(M, D)
    stride_sm, stride_sc,     # scale:(M, D)
    BLOCK_M1: tl.constexpr, BLOCK_K_ND: tl.constexpr,
    BLOCK_K_D: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    shape_key,
):
    # Visit order, tuned: see kernels/_tiles.py. This is the b2b, so ND is looped INSIDE the
    # program and every program reads the whole of Wa and Wb -- the weights are what gets re-read
    # here, not x, which is the opposite of the composed expand GEMM. So the row-first walk this
    # kernel has always done may well be the right one; the ladder carries 65536 for exactly that,
    # and the tuner decides per card and shape instead of the grid shape deciding for it.
    # `K` is the output width: this kernel drops D as an argument because D == K.
    pid_m, pid_d = tile_order(
        tl.program_id(0).to(tl.int64),
        tl.cdiv(M, BLOCK_M1), tl.cdiv(K, BLOCK_N), GROUP_M)
    rows = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    row_mask = rows < M
    dcols = pid_d * BLOCK_N + tl.arange(0, BLOCK_N)
    d_mask = dcols < K
    out_acc = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    for n0 in range(0, ND, BLOCK_K_ND):
        cols = n0 + tl.arange(0, BLOCK_K_ND)
        col_mask = cols < ND
        a = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)
        b = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K_D):
            k = k0 + tl.arange(0, BLOCK_K_D)
            k_mask = k < K
            x = tl.load(x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                        mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            wa = tl.load(wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                         mask=k_mask[:, None] & col_mask[None, :], other=0.0)
            wb = tl.load(wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                         mask=k_mask[:, None] & col_mask[None, :], other=0.0)
            a += tl.dot(x, wa, out_dtype=tl.float32, input_precision="tf32")
            b += tl.dot(x, wb, out_dtype=tl.float32, input_precision="tf32")
        # Cast before the squeeze dot: `a` and `b` are fp32 accumulators and `ws_t` carries the
        # weight's own dtype, and `tl.dot` requires one dtype for both operands. Without it this
        # kernel does not COMPILE at bf16 -- "Both operands must be same dtype. Got fp32 and bf16"
        # -- while its inference twin (inference.py) and the transition b2b (fused.py) both cast
        # here and compile. registry.csv declares this family fp32, where the cast is a no-op, so
        # nothing had ever asked it to build in the precision that fails.
        h = (a * tl.sigmoid(a) * b).to(x_ptr.dtype.element_ty)
        ws_t = tl.load(ws_ptr + cols[:, None] * stride_sn + dcols[None, :] * stride_sd,
                       mask=col_mask[:, None] & d_mask[None, :], other=0.0)
        out_acc += tl.dot(h, ws_t, out_dtype=tl.float32, input_precision="tf32")
        # emit saved tensors for backward (write the chunk as we go). Only the pid_d==0 column
        # of programs writes them: a/b/h do not depend on the d tile, so every other d program
        # would re-store identical bytes. Folded into the store MASK rather than an `if`, so
        # there is no divergent control flow around the stores.
        cm = row_mask[:, None] & col_mask[None, :] & (pid_d == 0)
        tl.store(ab_ptr + rows[:, None] * stride_abm + cols[None, :] * stride_abn, a, mask=cm)
        tl.store(ab_ptr + rows[:, None] * stride_abm + (cols + ND)[None, :] * stride_abn, b, mask=cm)
        tl.store(h_ptr + rows[:, None] * stride_hm + cols[None, :] * stride_hn, h, mask=cm)
    scale = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    for c0 in range(0, DC, BLOCK_K_D):
        dc = c0 + tl.arange(0, BLOCK_K_D)
        dc_mask = dc < DC
        cond = tl.load(cond_ptr + rows[:, None] * stride_cm + dc[None, :] * stride_cc,
                       mask=row_mask[:, None] & dc_mask[None, :], other=0.0)
        wsc_t = tl.load(wsc_ptr + dcols[None, :] * stride_scd + dc[:, None] * stride_scc,
                        mask=dc_mask[:, None] & d_mask[None, :], other=0.0)
        scale += tl.dot(cond, wsc_t, out_dtype=tl.float32, input_precision="tf32")
    bsc = tl.load(bsc_ptr + dcols, mask=d_mask, other=0.0)
    scale += bsc[None, :]
    y = tl.sigmoid(scale) * out_acc
    rm = row_mask[:, None] & d_mask[None, :]
    tl.store(y_ptr + rows[:, None] * stride_ym + dcols[None, :] * stride_yd, y, mask=rm)
    tl.store(out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od, out_acc, mask=rm)
    tl.store(scale_ptr + rows[:, None] * stride_sm + dcols[None, :] * stride_sc, scale, mask=rm)
# fmt: on


def _b2b_fwd_train_fake(x, cond, wa, wb, ws, wsc, bsc, shape_key=None):
    """(y (M, D), ab (M, 2*ND), h (M, ND), out (M, D), scale (M, D)) -- the expand width ND comes
    off wa and the squeeze width D off ws; neither is readable from x, whose width is K."""
    m = x.shape[0]
    nd = wa.shape[0]
    d = ws.shape[0]
    return (x.new_empty((m, d)), x.new_empty((m, 2 * nd)), x.new_empty((m, nd)),
            x.new_empty((m, d)), x.new_empty((m, d)))


@opaque(fake=_b2b_fwd_train_fake, name="conditioned_transition_b2b_fwd_train")
def _b2b_fwd_train(x: torch.Tensor, cond: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
                   ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor,
                   shape_key: int | None = None) -> tuple[
                       torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """atom fused b2b training forward -> (y, ab=[a|b], h, out, scale)."""
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
    D = ws.shape[0]
    DC = cond.shape[1]
    y = torch.empty(M, D, device=x.device, dtype=x.dtype)
    ab = torch.empty(M, 2 * ND, device=x.device, dtype=x.dtype)
    h = torch.empty(M, ND, device=x.device, dtype=x.dtype)
    out = torch.empty(M, D, device=x.device, dtype=x.dtype)
    scale = torch.empty(M, D, device=x.device, dtype=x.dtype)
    check_tile_axes("cond_transition_fwd_b2b_saveact_triton", D, K, "D (ws rows)", "K (x columns)")
    grid = lambda meta: tile_grid(M, D, meta["BLOCK_M1"], meta["BLOCK_N"])  # noqa: E731
    _b2b_fwd_train_kernel[grid](
        x, cond, wa, wb, ws, wsc, bsc, y, ab, h, out, scale, M, ND, K, DC,
        x.stride(0), x.stride(1), cond.stride(0), cond.stride(1),
        wa.stride(0), wa.stride(1), ws.stride(0), ws.stride(1),
        wsc.stride(0), wsc.stride(1),
        y.stride(0), y.stride(1), ab.stride(0), ab.stride(1), h.stride(0), h.stride(1),
        out.stride(0), out.stride(1), scale.stride(0), scale.stride(1),
        shape_key=pack(shape_key, ND=ND, K=K, DC=DC),
    )
    return y, ab, h, out, scale


# --- token (d>=256): composed 2-kernel forward emitting ab,h,out,scale -----------------
def _composed_fwd_train(x, cond, wa, wb, ws, wsc, bsc, *, shape_key=None):
    """token fused composed training forward -> (y, ab=[a|b], h, out, scale)."""
    from .fwd_saveact import _fwd_expand_swiglu, _fwd_squeeze_gate

    # One key for both kernels: they are two halves of one call at one L, and `h` between them is
    # (M, ND), so kernel B could not re-derive the same L from its own input anyway.
    h, ab = _fwd_expand_swiglu(x, wa, wb, shape_key=shape_key)   # kernel A: expand+swiglu -> h, ab
    y, out, scale = _fwd_squeeze_gate(h, cond, ws, wsc, bsc,     # kernel B: squeeze+gate -> out,scale
                                      shape_key=shape_key)
    return y, ab, h, out, scale


_ATOM_D_MAX = 128


def _fused_fwd_train(x, cond, wa, wb, ws, wsc, bsc, *, shape_key=None):
    """d-aware fused triton training forward; returns (y, ab, h, out, scale) for backward."""
    if x.shape[1] <= _ATOM_D_MAX:
        return _b2b_fwd_train(x, cond, wa, wb, ws, wsc, bsc, shape_key=shape_key)
    return _composed_fwd_train(x, cond, wa, wb, ws, wsc, bsc, shape_key=shape_key)


# Forward backend for training.
#   "cublas" = cat-merged-expand cuBLAS GEMMs + fused triton elementwise.
#   "fused"  = fused-triton b2b (atom) / composed (token) forward (GEMM<->elem fused).
#   "auto"   = per-regime pick of the measured-best (default).
# MEASURED (H100, CUDA-graph fwd+bwd; identical backward, only the forward differs):
# both beat eager 1.12-1.28x. fused-vs-cublas: fused WINS atom large-M (8192: 1.03x) and
# small token (384/512: 1.01-1.02x), ~ties mid, REGRESSES token>=768 (0.93-0.95x) because
# the forward must additionally write ab,h,out,scale (saved-for-bwd) that inference never
# writes, eroding the inference-proven GEMM<->elem fusion win. "auto" routes to the winner.
_FWD_MODE = "auto"  # {"auto", "cublas", "fused"}


def set_forward_mode(name: str):
    global _FWD_MODE
    assert name in ("auto", "cublas", "fused")
    _FWD_MODE = name


def _pick_fwd(d_hidden: int, M: int) -> str:
    """Measured-best forward backend per regime (CUDA-graph fwd+bwd)."""
    if d_hidden <= _ATOM_D_MAX:
        return "fused" if M >= 8192 else "cublas"   # fused wins only at large atom M
    return "fused" if M <= 512 else "cublas"        # fused wins only at small token M


class ConditionedTransitionTailFunction(torch.autograd.Function):
    """fp32 / TF32 forward + backward for the post-AdaLN ConditionedTransition tail.

    forward(x, cond, Wa, Wb, Ws, Wsc, bsc, length) -> y ; saves (x, cond, a, b, out, scale,
    weights). Backward: cuBLAS GEMMs (dgrad+wgrad) + fused-triton elementwise (gate-bwd,
    swiglu-bwd).

    ``length`` is L -- the ATOM count A of the un-flattened activation (see the CAVEAT up top).
    It is a POSITIONAL argument because ``Function.apply`` takes no keywords, and it is an input
    to ``forward``, so ``backward`` returns one extra ``None`` for it.
    """

    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc, length=None):
        # FORWARD backend (see _FWD_MODE): cuBLAS GEMMs + fused-triton elementwise (default,
        # measured-best e2e under CUDA graph) OR the fused-triton b2b/composed forward.
        # Both emit the SAME saved tensors (ab=[a|b], h, out, scale) + wcat for the backward.
        x = x.contiguous()
        wa = wa.contiguous(); wb = wb.contiguous()
        ws = ws.contiguous(); wsc = wsc.contiguous(); bsc = bsc.contiguous()
        cond = cond.contiguous()
        ND = wa.shape[0]
        wcat = torch.cat([wa, wb], dim=0)         # (2*ND, K); backward dx=dab@wcat, dWcat=dab^T@x
        # ONE L for the whole call, computed once here and reused by the backward via ctx: the
        # module hands us the real atom count A; without it all we could see is M = B*A.
        L = int(length) if length is not None else length_of(x.shape)
        ctx.length = L
        mode = _pick_fwd(x.shape[1], x.shape[0]) if _FWD_MODE == "auto" else _FWD_MODE
        if x.dtype == torch.bfloat16 and mode == "fused":
            mode = "cublas"  # bf16 fused b2b train kernel is broken (dtype/spill); use cuBLAS split
        if mode == "fused":
            y, ab, h, out, scale = _fused_fwd_train(x, cond, wa, wb, ws, wsc, bsc,
                                                    shape_key=atom_key(L))
        else:  # "cublas": cat-merged expand (one GEMM) + cuBLAS GEMMs + triton elementwise
            ab = x @ wcat.t()                     # (M, 2*ND)
            a, b = ab[:, :ND], ab[:, ND:]
            h = _swiglu(a, b, shape_key=atom_key(L))
            out = h @ ws.t()
            scale = torch.addmm(bsc, cond, wsc.t())
            # `_gate` launches gated_projection's `_sigmul_fwd`, which is level=BOTH in
            # registry.csv, so it keys on the ROW count while the atom-level kernels above key
            # on L. `out` is the flattened (M, D) matrix, and M is what a both-level bucket is.
            y = _gate(out, scale, shape_key=both_key(out.shape[0]))
        ctx.save_for_backward(x, cond, ab, h, out, scale, wcat, ws, wsc)
        ctx.ND = ND
        return y

    @staticmethod
    def backward(ctx, dy):
        x, cond, ab, h, out, scale, wcat, ws, wsc = ctx.saved_tensors
        ND = ctx.ND
        a, b = ab[:, :ND], ab[:, ND:]
        dy = dy.contiguous()
        # gate bwd (fused elementwise: dout, dscale in one flat pass — measured-best)
        # The SAME L the forward keyed on, carried on ctx (x is (M, K) here, so it cannot be
        # re-read). `_sigmul_bwd` is a level=both kernel, so its label uses the union bucket set.
        L = ctx.length
        shape_key = atom_key(L)
        # both_key buckets ROWS, and `out` is the flattened (M, D) the forward keyed with
        # `both_key(out.shape[0])` twenty lines up. Passing L here fed a LENGTH to a row
        # bucket: the two agreed only at B == 1, and otherwise the forward and the backward
        # of the same `_sigmul_*` kernel landed in different buckets. That is the mismatch
        # 6948c77 measured at 1.73x, in the same function.
        dout, dscale = _gate_bwd(out, scale, dy, shape_key=both_key(out.shape[0]))
        # conditioning grads
        dcond = dscale @ wsc                            # (M, DC)
        dWsc = dscale.t() @ cond                        # (D, DC)
        db_sc = dscale.sum(0)                           # (D,) — cheap cuBLAS-adjacent reduction
        del dscale                                      # last use; see the `del` note below
        # squeeze bwd
        dh = dout @ ws                                  # (M, ND)
        dWs = dout.t() @ h                              # (D, ND)
        del dout, dy      # dy is the same (M, D) block and `_gate_bwd` was its last reader
        # swiglu bwd (fused elementwise) -> packed [da | db] : (M, 2*ND)
        dab = _swiglu_bwd_packed(a, b, dh, shape_key=shape_key)
        # `del` after last use, and it is not cosmetic. A hand-written backward holds every
        # intermediate in a LOCAL until the function returns, where autograd frees each one as soon
        # as its consumer node has run -- so the peak here carried dscale, dout and dh (48 + 48 +
        # 96 = 192 MiB at B=32, L=1024, d=768, bf16) that were dead by the time the biggest
        # allocation, dab @ wcat, ran. Measured with the allocator trace: at the peak instant torch
        # held three (M, ND) blocks and NO (M, d) block, while this function held three of them.
        # That difference was the whole of this module's training-memory disadvantage.
        del dh
        # expand bwd: one concatenated GEMM each (vs 2 + add).
        dx = dab @ wcat                                 # (M, K)
        dWcat = dab.t() @ x                             # (2*ND, K)
        del dab                                         # (M, 2*ND) -- the largest block here
        dWa, dWb = dWcat[:ND], dWcat[ND:]
        # 8 returns for 8 forward inputs: the trailing None is `length` (an int shape key, not a
        # differentiable tensor). A missing one is an arity error, which is the point.
        return dx, dcond, dWa.contiguous(), dWb.contiguous(), dWs, dWsc, db_sc, None


def cond_transition_train(x, cond, wa, wb, ws, wsc, bsc, length=None):
    """Differentiable ConditionedTransition tail (training fwd+bwd via autograd Function).

    ``length`` is L -- the ATOM count A of the activation BEFORE the caller flattened it to
    (M, K) -- and is passed POSITIONALLY to ``.apply`` because ``Function.apply`` takes no
    keyword arguments. None falls back to M inside ``forward``.
    """
    return ConditionedTransitionTailFunction.apply(x, cond, wa, wb, ws, wsc, bsc, length)
