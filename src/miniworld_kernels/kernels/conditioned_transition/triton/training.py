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

import torch
import triton
import triton.language as tl

from miniworld_kernels.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of

# autograd Functions cannot be @typecheck'd cleanly; keep precision policy explicit.


# --- forward elementwise: h = silu(a)*b , scale = s@... already done, gate -----------
@triton.jit
def _swiglu_fwd_kernel(
    a_ptr, b_ptr, h_ptr, M, ND,
    stride_m, stride_n,      # a, b: (M, ND), possibly strided views
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * ND
    row = offs // ND
    col = offs % ND
    idx = row * stride_m + col * stride_n
    a = tl.load(a_ptr + idx, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + idx, mask=mask).to(tl.float32)
    h = a * tl.sigmoid(a) * b
    tl.store(h_ptr + offs, h, mask=mask)   # h is contiguous (M, ND)


@triton.jit
def _gate_fwd_kernel(out_ptr, scale_ptr, y_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    out = tl.load(out_ptr + offs, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr + offs, mask=mask).to(tl.float32)
    y = tl.sigmoid(scale) * out
    tl.store(y_ptr + offs, y, mask=mask)


def _swiglu(a, b):
    M, ND = a.shape
    h = torch.empty(M, ND, device=a.device, dtype=a.dtype)  # contiguous output
    n = M * ND
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _swiglu_fwd_kernel[grid](a, b, h, M, ND, a.stride(0), a.stride(1), BLOCK=2048)
    return h


def _gate(out, scale):
    y = torch.empty_like(out)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _gate_fwd_kernel[grid](out, scale, y, n, BLOCK=1024)
    return y


# --- backward elementwise (fused) ----------------------------------------------------
@triton.jit
def _gate_bwd_kernel(out_ptr, scale_ptr, dy_ptr, dout_ptr, dscale_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    out = tl.load(out_ptr + offs, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr + offs, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask).to(tl.float32)
    sg = tl.sigmoid(scale)
    tl.store(dout_ptr + offs, sg * dy, mask=mask)
    tl.store(dscale_ptr + offs, out * sg * (1.0 - sg) * dy, mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    a_ptr, b_ptr, dh_ptr, dab_ptr, M, ND, ND2,
    stride_m, stride_n,          # a, b: (M, ND) (possibly strided views)
    stride_dhm, stride_dhn,      # dh: (M, ND) (own strides — may differ from a/b)
    stride_pm, stride_pn,        # dab: (M, 2*ND) packed [da | db]
    BLOCK: tl.constexpr,
):
    # Packs da into dab[:, :ND] and db into dab[:, ND:] so the expand-bwd is one GEMM.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * ND
    row = offs // ND
    col = offs % ND
    a = tl.load(a_ptr + row * stride_m + col * stride_n, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + row * stride_m + col * stride_n, mask=mask).to(tl.float32)
    dh = tl.load(dh_ptr + row * stride_dhm + col * stride_dhn, mask=mask).to(tl.float32)
    sa = tl.sigmoid(a)
    silu = a * sa
    silu_prime = sa * (1.0 + a * (1.0 - sa))  # sa + silu*(1 - sa)
    da = dh * b * silu_prime
    db = dh * silu
    base = row * stride_pm
    tl.store(dab_ptr + base + col * stride_pn, da, mask=mask)
    tl.store(dab_ptr + base + (col + ND) * stride_pn, db, mask=mask)


def _gate_bwd(out, scale, dy):
    dout = torch.empty_like(out)
    dscale = torch.empty_like(out)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _gate_bwd_kernel[grid](out, scale, dy, dout, dscale, n, BLOCK=2048)
    return dout, dscale


# NB: a "fused" gate-bwd that also reduces db_sc=dscale.sum(0) in-kernel was tried and
# REGRESSED hard (full-D tile + tl.sum: token 1.25->0.33x compile). torch's dscale.sum(0)
# is a fast cuBLAS-adjacent reduction; keep it separate.


def _swiglu_bwd_packed(a, b, dh):
    """Return dab = [da | db] : (M, 2*ND), contiguous, for a single concatenated expand-bwd GEMM."""
    M, ND = a.shape
    dab = torch.empty(M, 2 * ND, device=a.device, dtype=a.dtype)
    n = M * ND
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _swiglu_bwd_kernel[grid](
        a, b, dh, dab, M, ND, 2 * ND,
        a.stride(0), a.stride(1),
        dh.stride(0), dh.stride(1),
        dab.stride(0), dab.stride(1),
        BLOCK=2048,
    )
    return dab


# ============================================================================
# FUSED TRITON FORWARD for training (emits the tensors backward needs).
# Reuses the inference fused structure (GEMM<->elem fusion, no h/out/scale HBM re-read
# between GEMMs) but additionally WRITES ab=[a|b], h, out, scale for the backward.
# ============================================================================

# --- atom (d<=128): single-kernel b2b forward, emits ab,h,out,scale,y -----------------
# num_stages=2 only: the b2b ALSO writes ab,h,out,scale and pipelines x[BM,BK=128] +
# wa/wb[BK,BN] + ws_t[BN,D] + out_acc[BM,D]; stages>=3 or BLOCK_N=128 overflows the 232KB
# SM90 smem budget (saw 245760B). Keep tiles modest.
_cfgs_b2b = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
]


# fmt: off
_cond_transition_training_b2b_fwd_prune = make_cache_prune(
    "cond_transition_training_b2b_fwd", dtype_of=tensor_dtype_of("x_ptr"),
    bucket_of=key_bucket_of("ND", "K", "D", "DC"),
)


@triton.autotune(configs=_cfgs_b2b, key=["ND", "K", "D", "DC"],
                 prune_configs_by={"early_config_prune": _cond_transition_training_b2b_fwd_prune})
@triton.jit
def _b2b_fwd_train_kernel(
    x_ptr, cond_ptr, wa_ptr, wb_ptr, ws_ptr, wsc_ptr, bsc_ptr,
    y_ptr, ab_ptr, h_ptr, out_ptr, scale_ptr,
    M, ND,
    K: tl.constexpr, D: tl.constexpr, DC: tl.constexpr,
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_DC: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    k = tl.arange(0, BLOCK_K)
    k_mask = k < K
    x = tl.load(x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                mask=row_mask[:, None] & k_mask[None, :], other=0.0)
    dcols = tl.arange(0, D)
    out_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for n0 in range(0, ND, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        col_mask = cols < ND
        wa = tl.load(wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                     mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        wb = tl.load(wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                     mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        a = tl.dot(x, wa, out_dtype=tl.float32, input_precision="tf32")
        b = tl.dot(x, wb, out_dtype=tl.float32, input_precision="tf32")
        h = a * tl.sigmoid(a) * b
        ws_t = tl.load(ws_ptr + cols[:, None] * stride_sn + dcols[None, :] * stride_sd,
                       mask=col_mask[:, None], other=0.0)
        out_acc += tl.dot(h, ws_t, out_dtype=tl.float32, input_precision="tf32")
        # emit saved tensors for backward (write the chunk as we go)
        cm = row_mask[:, None] & col_mask[None, :]
        tl.store(ab_ptr + rows[:, None] * stride_abm + cols[None, :] * stride_abn, a, mask=cm)
        tl.store(ab_ptr + rows[:, None] * stride_abm + (cols + ND)[None, :] * stride_abn, b, mask=cm)
        tl.store(h_ptr + rows[:, None] * stride_hm + cols[None, :] * stride_hn, h, mask=cm)
    scale = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for c0 in range(0, DC, BLOCK_DC):
        dc = c0 + tl.arange(0, BLOCK_DC)
        dc_mask = dc < DC
        cond = tl.load(cond_ptr + rows[:, None] * stride_cm + dc[None, :] * stride_cc,
                       mask=row_mask[:, None] & dc_mask[None, :], other=0.0)
        wsc_t = tl.load(wsc_ptr + dcols[None, :] * stride_scd + dc[:, None] * stride_scc,
                        mask=dc_mask[:, None], other=0.0)
        scale += tl.dot(cond, wsc_t, out_dtype=tl.float32, input_precision="tf32")
    bsc = tl.load(bsc_ptr + dcols)
    scale += bsc[None, :]
    y = tl.sigmoid(scale) * out_acc
    rm = row_mask[:, None]
    tl.store(y_ptr + rows[:, None] * stride_ym + dcols[None, :] * stride_yd, y, mask=rm)
    tl.store(out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od, out_acc, mask=rm)
    tl.store(scale_ptr + rows[:, None] * stride_sm + dcols[None, :] * stride_sc, scale, mask=rm)
# fmt: on


def _b2b_fwd_train(x, cond, wa, wb, ws, wsc, bsc):
    """atom fused b2b training forward -> (y, ab=[a|b], h, out, scale)."""
    M, K = x.shape
    ND = wa.shape[0]
    D = ws.shape[0]
    DC = cond.shape[1]
    y = torch.empty(M, D, device=x.device, dtype=x.dtype)
    ab = torch.empty(M, 2 * ND, device=x.device, dtype=x.dtype)
    h = torch.empty(M, ND, device=x.device, dtype=x.dtype)
    out = torch.empty(M, D, device=x.device, dtype=x.dtype)
    scale = torch.empty(M, D, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _b2b_fwd_train_kernel[grid](
        x, cond, wa, wb, ws, wsc, bsc, y, ab, h, out, scale, M, ND, K, D, DC,
        x.stride(0), x.stride(1), cond.stride(0), cond.stride(1),
        wa.stride(0), wa.stride(1), ws.stride(0), ws.stride(1),
        wsc.stride(0), wsc.stride(1),
        y.stride(0), y.stride(1), ab.stride(0), ab.stride(1), h.stride(0), h.stride(1),
        out.stride(0), out.stride(1), scale.stride(0), scale.stride(1),
        BLOCK_K=triton.next_power_of_2(K),
        BLOCK_DC=min(128, triton.next_power_of_2(DC)),
    )
    return y, ab, h, out, scale


# --- token (d>=256): composed 2-kernel forward emitting ab,h,out,scale -----------------
def _composed_fwd_train(x, cond, wa, wb, ws, wsc, bsc):
    """token fused composed training forward -> (y, ab=[a|b], h, out, scale)."""
    from .train_fused import _fwd_expand_swiglu, _fwd_squeeze_gate

    h, ab = _fwd_expand_swiglu(x, wa, wb)          # kernel A: expand+swiglu, emits h, ab
    y, out, scale = _fwd_squeeze_gate(h, cond, ws, wsc, bsc)  # kernel B: squeeze+gate, emits out,scale
    return y, ab, h, out, scale


_ATOM_D_MAX = 128


def _fused_fwd_train(x, cond, wa, wb, ws, wsc, bsc):
    """d-aware fused triton training forward; returns (y, ab, h, out, scale) for backward."""
    if x.shape[1] <= _ATOM_D_MAX:
        return _b2b_fwd_train(x, cond, wa, wb, ws, wsc, bsc)
    return _composed_fwd_train(x, cond, wa, wb, ws, wsc, bsc)


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

    forward(x, cond, Wa, Wb, Ws, Wsc, bsc) -> y ; saves (x, cond, a, b, out, scale, weights).
    Backward: cuBLAS GEMMs (dgrad+wgrad) + fused-triton elementwise (gate-bwd, swiglu-bwd).
    """

    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc):
        # FORWARD backend (see _FWD_MODE): cuBLAS GEMMs + fused-triton elementwise (default,
        # measured-best e2e under CUDA graph) OR the fused-triton b2b/composed forward.
        # Both emit the SAME saved tensors (ab=[a|b], h, out, scale) + wcat for the backward.
        x = x.contiguous()
        wa = wa.contiguous(); wb = wb.contiguous()
        ws = ws.contiguous(); wsc = wsc.contiguous(); bsc = bsc.contiguous()
        cond = cond.contiguous()
        ND = wa.shape[0]
        wcat = torch.cat([wa, wb], dim=0)         # (2*ND, K); backward dx=dab@wcat, dWcat=dab^T@x
        mode = _pick_fwd(x.shape[1], x.shape[0]) if _FWD_MODE == "auto" else _FWD_MODE
        if x.dtype == torch.bfloat16 and mode == "fused":
            mode = "cublas"  # bf16 fused b2b train kernel is broken (dtype/spill); use cuBLAS split
        if mode == "fused":
            y, ab, h, out, scale = _fused_fwd_train(x, cond, wa, wb, ws, wsc, bsc)
        else:  # "cublas": cat-merged expand (one GEMM) + cuBLAS GEMMs + triton elementwise
            ab = x @ wcat.t()                     # (M, 2*ND)
            a, b = ab[:, :ND], ab[:, ND:]
            h = _swiglu(a, b)
            out = h @ ws.t()
            scale = torch.addmm(bsc, cond, wsc.t())
            y = _gate(out, scale)
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
        dout, dscale = _gate_bwd(out, scale, dy)        # (M, D), (M, D)
        # conditioning grads
        dcond = dscale @ wsc                            # (M, DC)
        dWsc = dscale.t() @ cond                        # (D, DC)
        db_sc = dscale.sum(0)                           # (D,) — cheap cuBLAS-adjacent reduction
        # squeeze bwd
        dh = dout @ ws                                  # (M, ND)
        dWs = dout.t() @ h                              # (D, ND)
        # swiglu bwd (fused elementwise) -> packed [da | db] : (M, 2*ND)
        dab = _swiglu_bwd_packed(a, b, dh)
        # expand bwd: one concatenated GEMM each (vs 2 + add).
        dx = dab @ wcat                                 # (M, K)
        dWcat = dab.t() @ x                             # (2*ND, K)
        dWa, dWb = dWcat[:ND], dWcat[ND:]
        return dx, dcond, dWa.contiguous(), dWb.contiguous(), dWs, dWsc, db_sc


def cond_transition_train(x, cond, wa, wb, ws, wsc, bsc):
    """Differentiable ConditionedTransition tail (training fwd+bwd via autograd Function)."""
    return ConditionedTransitionTailFunction.apply(x, cond, wa, wb, ws, wsc, bsc)
