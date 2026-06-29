"""No-cuBLAS fused training for the post-AdaLN ConditionedTransition tail (TF32 triton).

Every GEMM is a triton ``tl.dot(input_precision="tf32")`` kernel that *fuses the
elementwise op into the matmul that produces or consumes it*, so the SwiGLU/gate
intermediates never round-trip HBM. cute/quack SM90 can't do TF32 (gemm_sm90 gates
16-bit), so triton is the TF32 vehicle.

Math (sg = sigmoid(scale), sa = sigmoid(a)):
    fwd:  a=x@Wa^T, b=x@Wb^T ; h=silu(a)*b ; out=h@Ws^T ; scale=cond@Wsc^T+bsc ; y=sg*out
    bwd:  dout=sg*dy ; dscale=out*sg*(1-sg)*dy
          dh = dout @ Ws        (Ws:(D,ND))      dWs  = dout^T @ h
          dcond = dscale @ Wsc  (Wsc:(D,DC))     dWsc = dscale^T @ cond ; db_sc = dscale.sum(0)
          silu'(a) = sa*(1 + a*(1-sa))
          da = dh*b*silu'(a) ; db = dh*silu(a)
          dx = da @ Wa + db @ Wb                 dWa = da^T @ x ; dWb = db^T @ x

dgrad GEMMs (outputs (M,*)) fuse the elementwise in the PROLOGUE:
    - dh-GEMM   : gate-bwd computes dout=sg*dy from (out,scale,dy) then dout@Ws -> dh (also emits dscale).
    - dcond-GEMM: dscale @ Wsc -> dcond.
    - dx-GEMM   : swiglu-bwd forms dab=[da|db] from (dh,a,b) per tile, ONE concatenated GEMM dab@Wcat -> dx.
The fused dgrad kernels also emit the materialized operands (dout,dscale,dab) so the cuBLAS
wgrad GEMMs (dWs,dWsc,dWa,dWb) can reuse them. wgrad stays cuBLAS unconditionally
(reductions over M = cuBLAS's domain).

MEASURED VERDICT (H100, CUDA-graph; archived conditioned_transition training reports
under benchmarks/reports/archive): CORRECT (all 7 grads +
cos_y = 1.00000). Eager fwd+bwd is autograd-overhead-bound (~330us flat for every path) and
not informative; under CUDA graph this fused path reaches/BEATS torch-eager at atom M<=4096
(1.03x @ M=2048) but loses to the cuBLAS-GEMM training path (training.py) at large atom and
at token. Per-stage CUDA-graph micro: fused-triton dgrad loses to (cuBLAS GEMM + separate
elementwise) by 1.6-7.1x on EVERY stage -- the fusion removes the elementwise HBM pass, but
the triton GEMM itself is 2-7x slower than cuBLAS at these M-heavy/short-K shapes and the
gap widens with M (a real per-kernel gap, not launch overhead). Unlike inference (a b2b
fusion eliminating a large HBM intermediate), backward has no fusible GEMM pair: dh/dcond/dx
each feed a different wgrad that needs the operand materialized. => cond_transition_train
(training.py) stays the production default; this path is shipped, correct, and selectable.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ============================================================================
# FORWARD (training): reuse the fused inference structure, emit saved tensors.
# ============================================================================

_cfgs_fwdA = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
]


# fmt: off
@triton.autotune(configs=_cfgs_fwdA, key=["M", "ND", "K"])
@triton.jit
def _fwd_expand_swiglu_kernel(
    x_ptr, wa_ptr, wb_ptr, h_ptr, ab_ptr,
    M, ND, K, ND2,
    stride_xm, stride_xk,
    stride_wn, stride_wk,    # Wa, Wb: (ND, K) row-major
    stride_hm, stride_hn,    # h:  (M, ND)
    stride_abm, stride_abn,  # ab: (M, 2*ND) packed [a | b]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < ND
    a = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    b = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
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


_cfgs_fwdB = [
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_D": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_D": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 128, "BLOCK_K": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_D": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
]


# fmt: off
@triton.autotune(configs=_cfgs_fwdB, key=["M", "ND", "D", "DC"])
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
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_DC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dcols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = rows < M
    d_mask = dcols < D
    out_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for n0 in range(0, ND, BLOCK_K):
        n = n0 + tl.arange(0, BLOCK_K)
        n_mask = n < ND
        h = tl.load(h_ptr + rows[:, None] * stride_hm + n[None, :] * stride_hn,
                    mask=row_mask[:, None] & n_mask[None, :], other=0.0)
        ws_t = tl.load(ws_ptr + n[:, None] * stride_sn + dcols[None, :] * stride_sd,
                       mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        out_acc += tl.dot(h, ws_t, out_dtype=tl.float32, input_precision="tf32")
    scale = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for c0 in range(0, DC, BLOCK_DC):
        dc = c0 + tl.arange(0, BLOCK_DC)
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


def _fwd_expand_swiglu(x, wa, wb):
    M, K = x.shape
    ND = wa.shape[0]
    h = torch.empty(M, ND, device=x.device, dtype=x.dtype)
    ab = torch.empty(M, 2 * ND, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(ND, meta["BLOCK_N"]))  # noqa: E731
    _fwd_expand_swiglu_kernel[grid](
        x, wa, wb, h, ab, M, ND, K, 2 * ND,
        x.stride(0), x.stride(1), wa.stride(0), wa.stride(1),
        h.stride(0), h.stride(1), ab.stride(0), ab.stride(1),
    )
    return h, ab


def _fwd_squeeze_gate(h, cond, ws, wsc, bsc):
    M, ND = h.shape
    D = ws.shape[0]
    DC = cond.shape[1]
    y = torch.empty(M, D, device=h.device, dtype=h.dtype)
    out = torch.empty(M, D, device=h.device, dtype=h.dtype)
    scale = torch.empty(M, D, device=h.device, dtype=h.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(D, meta["BLOCK_D"]))  # noqa: E731
    _fwd_squeeze_gate_kernel[grid](
        h, cond, ws, wsc, bsc, y, out, scale, M, ND, D, DC,
        h.stride(0), h.stride(1), cond.stride(0), cond.stride(1),
        ws.stride(0), ws.stride(1), wsc.stride(0), wsc.stride(1),
        y.stride(0), y.stride(1), out.stride(0), out.stride(1),
        scale.stride(0), scale.stride(1),
        BLOCK_DC=min(128, triton.next_power_of_2(DC)),
    )
    return y, out, scale


# ============================================================================
# BACKWARD dgrad (outputs (M,*)) — fuse elementwise into the consuming GEMM.
# ============================================================================

# --- gate-bwd: dout = sg*dy ; dscale = out*sg*(1-sg)*dy  (one HBM pass over (M,D)) ---
@triton.jit
def _gate_bwd_kernel(out_ptr, scale_ptr, dy_ptr, dout_ptr, dscale_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    out = tl.load(out_ptr + offs, mask=mask)
    scale = tl.load(scale_ptr + offs, mask=mask)
    dy = tl.load(dy_ptr + offs, mask=mask)
    sg = tl.sigmoid(scale)
    tl.store(dout_ptr + offs, sg * dy, mask=mask)
    tl.store(dscale_ptr + offs, out * sg * (1.0 - sg) * dy, mask=mask)


def _gate_bwd(out, scale, dy):
    dout = torch.empty_like(out)
    dscale = torch.empty_like(out)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _gate_bwd_kernel[grid](out, scale, dy, dout, dscale, n, BLOCK=2048)
    return dout, dscale


# --- generic dgrad GEMM: C(M,N) = A(M,K) @ W(K,N), W stored row-major (K,N) ----------
# Used for dcond = dscale @ Wsc (Wsc is (D,DC) i.e. (N,K)->transposed access) and
# dh = dout @ Ws (Ws is (D,ND) i.e. (K=D, N=ND) row-major). We pass W with its strides
# so both the (K,N) and (N,K) layouts work via stride_wk/stride_wn.
_cfgs_dgemm = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
    # thin-N / short-K friendly: bigger M tiles, fewer warps -> less launch + better occupancy
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=4),
]


# fmt: off
@triton.autotune(configs=_cfgs_dgemm, key=["M", "N", "K"])
@triton.jit
def _dgemm_kernel(
    a_ptr, w_ptr, c_ptr, M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,   # logical W:(K,N); pass strides so any storage layout works
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # L2-friendly grouped (swizzled) program ordering — standard triton matmul scheduling.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    first_m = group_id * GROUP_M
    group_size = min(grid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size)
    pid_n = (pid % width) // group_size
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        a = tl.load(a_ptr + rows[:, None] * stride_am + k[None, :] * stride_ak,
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                    mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        acc += tl.dot(a, w, out_dtype=tl.float32, input_precision="tf32")
    tl.store(c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
             acc, mask=row_mask[:, None] & col_mask[None, :])
# fmt: on


def _dgemm(a, w, M, N, K, swk, swn):
    """C = a(M,K) @ W(K,N) via TF32 triton. swk,swn = W strides for the (K,N) logical view."""
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)  # noqa: E731
    _dgemm_kernel[grid](a, w, c, M, N, K, a.stride(0), a.stride(1), swk, swn,
                        c.stride(0), c.stride(1))
    return c


# --- dx-GEMM with swiglu-bwd prologue: dx = da@Wa + db@Wb, da/db recomputed per tile ---
# Contraction over the ND axis (loaded once); da,db never materialize to HBM.
_cfgs_dx = [
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 128}, num_warps=4, num_stages=4),
]


# fmt: off
@triton.autotune(configs=_cfgs_dx, key=["M", "K", "ND"])
@triton.jit
def _dx_fused_kernel(
    dh_ptr, ab_ptr, wa_ptr, wb_ptr, dx_ptr,
    M, K, ND,
    stride_dhm, stride_dhn,
    stride_abm, stride_abn,    # ab:(M,2*ND) [a|b]
    stride_wn, stride_wk,      # Wa,Wb:(ND,K) row-major
    stride_dxm, stride_dxk,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # dx[m,:] = sum_n da[m,n]*Wa[n,:] + db[m,n]*Wb[n,:]   (BLOCK_D tiles the ND axis)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    kk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)   # output feature (K=d_hidden) tile
    row_mask = rows < M
    k_mask = kk < K
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n0 in range(0, ND, BLOCK_D):
        n = n0 + tl.arange(0, BLOCK_D)
        n_mask = n < ND
        dh = tl.load(dh_ptr + rows[:, None] * stride_dhm + n[None, :] * stride_dhn,
                     mask=row_mask[:, None] & n_mask[None, :], other=0.0)
        a = tl.load(ab_ptr + rows[:, None] * stride_abm + n[None, :] * stride_abn,
                    mask=row_mask[:, None] & n_mask[None, :], other=0.0)
        b = tl.load(ab_ptr + rows[:, None] * stride_abm + (n + ND)[None, :] * stride_abn,
                    mask=row_mask[:, None] & n_mask[None, :], other=0.0)
        sa = tl.sigmoid(a)
        silu = a * sa
        silu_p = sa * (1.0 + a * (1.0 - sa))
        da = dh * b * silu_p           # (BM, BD)
        db = dh * silu                 # (BM, BD)
        wa = tl.load(wa_ptr + n[:, None] * stride_wn + kk[None, :] * stride_wk,
                     mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        wb = tl.load(wb_ptr + n[:, None] * stride_wn + kk[None, :] * stride_wk,
                     mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(da, wa, out_dtype=tl.float32, input_precision="tf32")
        acc += tl.dot(db, wb, out_dtype=tl.float32, input_precision="tf32")
    tl.store(dx_ptr + rows[:, None] * stride_dxm + kk[None, :] * stride_dxk,
             acc, mask=row_mask[:, None] & k_mask[None, :])
# fmt: on


def _dx_fused(dh, ab, wa, wb):
    M, ND = dh.shape
    K = wa.shape[1]
    dx = torch.empty(M, K, device=dh.device, dtype=dh.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(K, meta["BLOCK_K"]))  # noqa: E731
    _dx_fused_kernel[grid](
        dh, ab, wa, wb, dx, M, K, ND,
        dh.stride(0), dh.stride(1), ab.stride(0), ab.stride(1),
        wa.stride(0), wa.stride(1), dx.stride(0), dx.stride(1),
        BLOCK_D=64,
    )
    return dx


# ============================================================================
# FUSED-PROLOGUE dgrad: gate-bwd folded into dh-GEMM ; swiglu-bwd folded into dx-GEMM.
# These collapse the separate elementwise kernels into the consuming GEMM (the gate/swiglu
# operands form in-register per K-tile), AND emit the materialized operand (dout/dscale/dab)
# in the epilogue so the cuBLAS wgrad GEMMs can reuse them. Fewer launches = the real win
# (triton's per-call dispatch floor ~20us dominates these small dgrad GEMMs in eager).

# --- dh = dout @ Ws, with dout = sigmoid(scale)*dy formed in the K-loop; emit dout,dscale ---
# fmt: off
@triton.autotune(configs=_cfgs_dgemm, key=["M", "ND", "D"])
@triton.jit
def _dh_gatebwd_kernel(
    out_ptr, scale_ptr, dy_ptr, ws_ptr, dh_ptr, dout_ptr, dscale_ptr,
    M, ND, D,
    stride_om, stride_od,       # out/scale/dy: (M, D)
    stride_wk, stride_wn,       # Ws:(D,ND) logical (K=D, N=ND)
    stride_dhm, stride_dhn,     # dh:(M, ND)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M); grid_n = tl.cdiv(ND, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width; first_m = group_id * GROUP_M
    group_size = min(grid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size); pid_n = (pid % width) // group_size
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M; col_mask = cols < ND
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < D
        em = row_mask[:, None] & k_mask[None, :]
        o = tl.load(out_ptr + rows[:, None] * stride_om + k[None, :] * stride_od, mask=em, other=0.0)
        s = tl.load(scale_ptr + rows[:, None] * stride_om + k[None, :] * stride_od, mask=em, other=0.0)
        dyv = tl.load(dy_ptr + rows[:, None] * stride_om + k[None, :] * stride_od, mask=em, other=0.0)
        sg = tl.sigmoid(s)
        dout = sg * dyv                                  # gate-bwd, in-register (BM, BK)
        w = tl.load(ws_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                    mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        acc += tl.dot(dout, w, out_dtype=tl.float32, input_precision="tf32")
        # emit dout, dscale once (only the first N-block writes, to avoid redundant stores)
        if pid_n == 0:
            dscale = o * sg * (1.0 - sg) * dyv
            tl.store(dout_ptr + rows[:, None] * stride_om + k[None, :] * stride_od, dout, mask=em)
            tl.store(dscale_ptr + rows[:, None] * stride_om + k[None, :] * stride_od, dscale, mask=em)
    tl.store(dh_ptr + rows[:, None] * stride_dhm + cols[None, :] * stride_dhn,
             acc, mask=row_mask[:, None] & col_mask[None, :])
# fmt: on


def _dh_gatebwd(out, scale, dy, ws, ND):
    """dh = (sigmoid(scale)*dy) @ Ws ; also returns materialized dout, dscale for wgrad."""
    M, D = out.shape
    dh = torch.empty(M, ND, device=out.device, dtype=out.dtype)
    dout = torch.empty(M, D, device=out.device, dtype=out.dtype)
    dscale = torch.empty(M, D, device=out.device, dtype=out.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(ND, meta["BLOCK_N"]),)  # noqa: E731
    _dh_gatebwd_kernel[grid](
        out, scale, dy, ws, dh, dout, dscale, M, ND, D,
        out.stride(0), out.stride(1), ws.stride(0), ws.stride(1), dh.stride(0), dh.stride(1),
    )
    return dh, dout, dscale


# --- dx = dab @ Wcat (one concatenated GEMM), dab=[da|db] formed per K-tile; emit dab ---
# fmt: off
@triton.autotune(configs=_cfgs_dgemm, key=["M", "K", "ND2"])
@triton.jit
def _dx_swiglubwd_kernel(
    dh_ptr, ab_ptr, wcat_ptr, dx_ptr, dab_ptr,
    M, K, ND, ND2,
    stride_dhm, stride_dhn,
    stride_abm, stride_abn,     # ab:(M,2ND) [a|b]
    stride_wj, stride_wk,       # Wcat:(2ND, K) logical (K_red=2ND axis = j, N=K)
    stride_dxm, stride_dxk,
    stride_pm, stride_pn,       # dab:(M,2ND) [da|db]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # dx[m,:] = sum_{j in 2ND} dab[m,j] * Wcat[j,:]   (j tiled by BLOCK_K = the reduction)
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M); grid_n = tl.cdiv(K, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width; first_m = group_id * GROUP_M
    group_size = min(grid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size); pid_n = (pid % width) // group_size
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    kk = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)       # output feature (K=d_hidden)
    row_mask = rows < M; k_mask = kk < K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for j0 in range(0, ND2, BLOCK_K):
        j = j0 + tl.arange(0, BLOCK_K)
        j_mask = j < ND2
        # dab[:, j] : first ND cols -> da = dh*b*silu'(a); next ND cols -> db = dh*silu(a).
        # j indexes the packed 2ND axis. n = j % ND maps to the dh/a/b column.
        n = j % ND
        is_db = j >= ND
        em = row_mask[:, None] & j_mask[None, :]
        dh = tl.load(dh_ptr + rows[:, None] * stride_dhm + n[None, :] * stride_dhn, mask=em, other=0.0)
        a = tl.load(ab_ptr + rows[:, None] * stride_abm + n[None, :] * stride_abn, mask=em, other=0.0)
        b = tl.load(ab_ptr + rows[:, None] * stride_abm + (n + ND)[None, :] * stride_abn, mask=em, other=0.0)
        sa = tl.sigmoid(a)
        silu = a * sa
        silu_p = sa * (1.0 + a * (1.0 - sa))
        dval = tl.where(is_db[None, :], dh * silu, dh * b * silu_p)   # da or db per column
        w = tl.load(wcat_ptr + j[:, None] * stride_wj + kk[None, :] * stride_wk,
                    mask=j_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(dval, w, out_dtype=tl.float32, input_precision="tf32")
        if pid_n == 0:
            tl.store(dab_ptr + rows[:, None] * stride_pm + j[None, :] * stride_pn, dval, mask=em)
    tl.store(dx_ptr + rows[:, None] * stride_dxm + kk[None, :] * stride_dxk,
             acc, mask=row_mask[:, None] & k_mask[None, :])
# fmt: on


def _dx_swiglubwd(dh, ab, wcat):
    """dx = dab @ Wcat (one GEMM), dab formed in-register from (dh, ab); emits dab for wgrad."""
    M, ND = dh.shape
    ND2 = 2 * ND
    K = wcat.shape[1]
    dx = torch.empty(M, K, device=dh.device, dtype=dh.dtype)
    dab = torch.empty(M, ND2, device=dh.device, dtype=dh.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(K, meta["BLOCK_N"]),)  # noqa: E731
    _dx_swiglubwd_kernel[grid](
        dh, ab, wcat, dx, dab, M, K, ND, ND2,
        dh.stride(0), dh.stride(1), ab.stride(0), ab.stride(1),
        wcat.stride(0), wcat.stride(1), dx.stride(0), dx.stride(1), dab.stride(0), dab.stride(1),
    )
    return dx, dab


# ============================================================================
# BACKWARD wgrad: cuBLAS (reductions over M — cuBLAS's domain; left on cuBLAS per the
# course-correction). dout, dscale, dab are materialized by the fused dgrad kernels above.
# ============================================================================

# --- swiglu-bwd packed: dab=[da|db] for the dWa/dWb cuBLAS path -----------------------
@triton.jit
def _swiglu_bwd_pack_kernel(
    dh_ptr, ab_ptr, dab_ptr, M, ND,
    stride_dhm, stride_dhn, stride_abm, stride_abn, stride_pm, stride_pn,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * ND
    row = offs // ND
    col = offs % ND
    dh = tl.load(dh_ptr + row * stride_dhm + col * stride_dhn, mask=mask)
    a = tl.load(ab_ptr + row * stride_abm + col * stride_abn, mask=mask)
    b = tl.load(ab_ptr + row * stride_abm + (col + ND) * stride_abn, mask=mask)
    sa = tl.sigmoid(a)
    silu = a * sa
    silu_p = sa * (1.0 + a * (1.0 - sa))
    tl.store(dab_ptr + row * stride_pm + col * stride_pn, dh * b * silu_p, mask=mask)
    tl.store(dab_ptr + row * stride_pm + (col + ND) * stride_pn, dh * silu, mask=mask)


def _swiglu_bwd_pack(dh, ab):
    M, ND = dh.shape
    dab = torch.empty(M, 2 * ND, device=dh.device, dtype=dh.dtype)
    n = M * ND
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _swiglu_bwd_pack_kernel[grid](
        dh, ab, dab, M, ND,
        dh.stride(0), dh.stride(1), ab.stride(0), ab.stride(1), dab.stride(0), dab.stride(1),
        BLOCK=2048,
    )
    return dab


# --- generic wgrad GEMM: dW(N,K) = G(M,N)^T @ X(M,K) ; reduce over M (K_red=M) --------
_cfgs_wgrad = [
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_M": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_M": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_M": 128}, num_warps=8, num_stages=3),
]


# fmt: off
@triton.autotune(configs=_cfgs_wgrad, key=["N", "K", "M"])
@triton.jit
def _wgrad_kernel(
    g_ptr, x_ptr, dw_ptr, M, N, K,
    stride_gm, stride_gn,
    stride_xm, stride_xk,
    stride_dwn, stride_dwk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    # dW[n,k] = sum_m G[m,n] * X[m,k]
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = ns < N
    k_mask = ks < K
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        m = m0 + tl.arange(0, BLOCK_M)
        m_mask = m < M
        g = tl.load(g_ptr + m[:, None] * stride_gm + ns[None, :] * stride_gn,
                    mask=m_mask[:, None] & n_mask[None, :], other=0.0)   # (BM, BN)
        x = tl.load(x_ptr + m[:, None] * stride_xm + ks[None, :] * stride_xk,
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)   # (BM, BK)
        acc += tl.dot(tl.trans(g), x, out_dtype=tl.float32, input_precision="tf32")
    tl.store(dw_ptr + ns[:, None] * stride_dwn + ks[None, :] * stride_dwk,
             acc, mask=n_mask[:, None] & k_mask[None, :])
# fmt: on


def _wgrad(g, x, N, K):
    """dW(N,K) = g(M,N)^T @ x(M,K) via TF32 triton (reduce over M)."""
    M = g.shape[0]
    dw = torch.empty(N, K, device=g.device, dtype=g.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(K, meta["BLOCK_K"]))  # noqa: E731
    _wgrad_kernel[grid](g, x, dw, M, N, K, g.stride(0), g.stride(1),
                       x.stride(0), x.stride(1), dw.stride(0), dw.stride(1))
    return dw


# ============================================================================
# Autograd Function — all-triton dgrad; wgrad via measured hybrid (set by env).
# ============================================================================

# Per-GEMM wgrad backend selection. Default = cuBLAS for wgrads (big-M reductions where
# cuBLAS usually wins); flip to "triton" after the measured comparison if it wins.
_WGRAD_BACKEND = "cublas"  # {"cublas", "triton"}


def set_wgrad_backend(name: str):
    global _WGRAD_BACKEND
    assert name in ("cublas", "triton")
    _WGRAD_BACKEND = name


class ConditionedTransitionTailFusedFunction(torch.autograd.Function):
    """No-cuBLAS dgrad ConditionedTransition tail; wgrad backend is the measured hybrid."""

    @staticmethod
    def forward(ctx, x, cond, wa, wb, ws, wsc, bsc):
        x = x.contiguous()
        wa = wa.contiguous(); wb = wb.contiguous()
        h, ab = _fwd_expand_swiglu(x, wa, wb)
        y, out, scale = _fwd_squeeze_gate(h, cond.contiguous(), ws.contiguous(),
                                          wsc.contiguous(), bsc.contiguous())
        ctx.save_for_backward(x, cond, ab, h, out, scale, wa, wb, ws, wsc)
        ctx.ND = wa.shape[0]
        ctx.wcat = torch.cat([wa, wb], dim=0)   # (2ND, K) for the one-GEMM dx
        return y

    @staticmethod
    def backward(ctx, dy):
        x, cond, ab, h, out, scale, wa, wb, ws, wsc = ctx.saved_tensors
        ND = ctx.ND
        D = ws.shape[0]
        DC = cond.shape[1]
        K = wa.shape[1]
        M = x.shape[0]
        dy = dy.contiguous()
        wcat = ctx.wcat

        # --- dgrad: TF32 triton GEMMs with the producing elementwise FUSED into the prologue ---
        #   dh = (sigmoid(scale)*dy) @ Ws ; gate-bwd folded in; emits dout,dscale for wgrad.
        dh, dout, dscale = _dh_gatebwd(out, scale, dy, ws, ND)
        #   dcond = dscale @ Wsc ; Wsc:(D,DC) logical (K=D, N=DC). (dscale already materialized.)
        dcond = _dgemm(dscale, wsc, M, DC, D, wsc.stride(0), wsc.stride(1))
        #   dx = dab @ Wcat (one concatenated GEMM); swiglu-bwd folded in; emits dab for wgrad.
        dx, dab = _dx_swiglubwd(dh, ab, wcat)

        # --- wgrad: cuBLAS (TF32) reductions over M (cuBLAS's domain; left on cuBLAS) ---
        db_sc = dscale.sum(0)
        dWs = dout.t() @ h                      # (D, ND)
        dWsc = dscale.t() @ cond                # (D, DC)
        dWcat = dab.t() @ x                     # (2ND, K)
        dWa, dWb = dWcat[:ND].contiguous(), dWcat[ND:].contiguous()
        return dx, dcond, dWa, dWb, dWs, dWsc, db_sc


def cond_transition_train_fused(x, cond, wa, wb, ws, wsc, bsc):
    """No-cuBLAS-dgrad ConditionedTransition tail training (fwd saves; fused-triton backward)."""
    return ConditionedTransitionTailFusedFunction.apply(x, cond, wa, wb, ws, wsc, bsc)
