"""Composed (two-kernel) INFERENCE path for the token stream (d_hidden >= 256).

The fully-fused b2b (``inference.py``) keeps the whole ``(M, ND)`` SwiGLU tile and
``out_acc[BM, D]`` live in registers, which only fits when K = d_hidden is small (the
atom stream, d=128). For the token stream (d=768) ``BLOCK_K = next_pow2(768) = 1024``
won't compile and ``out_acc[BM, 768]`` spills. So we *compose* two autotuned kernels and
let ``h:(M, ND)`` round-trip HBM:

    kernel A  (expand + SwiGLU):   h = silu(x @ Wa^T) * (x @ Wb^T)        -> HBM
    kernel B  (squeeze + gate):    out = h @ Ws^T
                                   scale = cond @ Wsc^T + b_sc            (DC-tiled, fused)
                                   y     = sigmoid(scale) * out

Both matmuls are K-tiled (loop BLOCK_K over the contraction dim), so any K compiles.
fp32 io with TF32 tensor cores (``input_precision="tf32"``).
"""

import torch
import triton
import triton.language as tl


# --- kernel A: expand + SwiGLU -> h:(M, ND) ----------------------------------
_cfgs_a = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
]


# fmt: off
@triton.autotune(configs=_cfgs_a, key=["M", "ND", "K"])
@triton.jit
def _expand_swiglu_kernel(
    x_ptr, wa_ptr, wb_ptr, h_ptr,
    M, ND, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,    # Wa, Wb: (ND, K) row-major
    stride_hm, stride_hn,
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
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=row_mask[:, None] & k_mask[None, :], other=0.0,
        )
        wa = tl.load(
            wa_ptr + cols[None, :] * stride_wn + k[:, None] * stride_wk,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        wb = tl.load(
            wb_ptr + cols[None, :] * stride_wn + k[:, None] * stride_wk,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        a += tl.dot(x, wa, out_dtype=tl.float32, input_precision="tf32")
        b += tl.dot(x, wb, out_dtype=tl.float32, input_precision="tf32")
    h = a * tl.sigmoid(a) * b
    tl.store(
        h_ptr + rows[:, None] * stride_hm + cols[None, :] * stride_hn,
        h, mask=row_mask[:, None] & col_mask[None, :],
    )
# fmt: on


# --- kernel B: squeeze + conditioning gate -> y:(M, D) -----------------------
# D is tiled too (BLOCK_D, power-of-2) so D need not be a power of 2 (D=768) and the
# (BLOCK_M, BLOCK_D) out/scale tiles stay register-sized. Grid = (M tiles, D tiles).
_cfgs_b = [
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_D": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_D": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 128, "BLOCK_K": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_D": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    # occupancy-friendly for the token stream (small M=384-1024, D=768): smaller M/D tiles
    # -> more CTAs so all 132 SMs are busy (BM=64/BD=64 alone gives only ~72 CTAs). Keeps the
    # gate fused (unlike split-K, which would un-fuse it).
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 128, "BLOCK_K": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 32, "BLOCK_K": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
]


# fmt: off
@triton.autotune(configs=_cfgs_b, key=["M", "ND", "D", "DC"])
@triton.jit
def _squeeze_gate_kernel(
    h_ptr, cond_ptr, ws_ptr, wsc_ptr, bsc_ptr, out_ptr,
    M, ND, D,
    DC: tl.constexpr,
    stride_hm, stride_hn,
    stride_cm, stride_cc,
    stride_sd, stride_sn,     # Ws: (D, ND) row-major
    stride_scd, stride_scc,   # Wsc: (D, DC) row-major
    stride_om, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_DC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dcols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = rows < M
    d_mask = dcols < D

    # out[:, d_tile] = h @ Ws[d_tile, :]^T   (K = ND, tiled)
    out_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for n0 in range(0, ND, BLOCK_K):
        n = n0 + tl.arange(0, BLOCK_K)
        n_mask = n < ND
        h = tl.load(
            h_ptr + rows[:, None] * stride_hm + n[None, :] * stride_hn,
            mask=row_mask[:, None] & n_mask[None, :], other=0.0,
        )
        ws_t = tl.load(  # (BLOCK_K, BLOCK_D): Ws[d_tile, n]^T
            ws_ptr + n[:, None] * stride_sn + dcols[None, :] * stride_sd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        )
        out_acc += tl.dot(h, ws_t, out_dtype=tl.float32, input_precision="tf32")

    # scale[:, d_tile] = cond @ Wsc[d_tile, :]^T + b_sc  (DC tiled), y = sigmoid(scale) * out
    scale = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for c0 in range(0, DC, BLOCK_DC):
        dc = c0 + tl.arange(0, BLOCK_DC)
        dc_mask = dc < DC
        cond = tl.load(
            cond_ptr + rows[:, None] * stride_cm + dc[None, :] * stride_cc,
            mask=row_mask[:, None] & dc_mask[None, :], other=0.0,
        )
        wsc_t = tl.load(  # (BLOCK_DC, BLOCK_D): Wsc[d_tile, dc]^T
            wsc_ptr + dcols[None, :] * stride_scd + dc[:, None] * stride_scc,
            mask=dc_mask[:, None] & d_mask[None, :], other=0.0,
        )
        scale += tl.dot(cond, wsc_t, out_dtype=tl.float32, input_precision="tf32")
    bsc = tl.load(bsc_ptr + dcols, mask=d_mask, other=0.0)
    scale += bsc[None, :]
    y = tl.sigmoid(scale) * out_acc
    tl.store(
        out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od,
        y, mask=row_mask[:, None] & d_mask[None, :],
    )
# fmt: on


def _expand_swiglu(x, wa, wb):
    """h = silu(x @ Wa^T) * (x @ Wb^T) -> (M, ND)."""
    M, K = x.shape
    ND = wa.shape[0]
    h = torch.empty(M, ND, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(ND, meta["BLOCK_N"]))  # noqa: E731
    _expand_swiglu_kernel[grid](
        x, wa, wb, h,
        M, ND, K,
        x.stride(0), x.stride(1),
        wa.stride(0), wa.stride(1),
        h.stride(0), h.stride(1),
    )
    return h


def _squeeze_gate(h, cond, ws, wsc, bsc):
    """y = sigmoid(cond @ Wsc^T + b_sc) * (h @ Ws^T) -> (M, D)."""
    M, ND = h.shape
    D = ws.shape[0]
    DC = cond.shape[1]
    out = torch.empty(M, D, device=h.device, dtype=h.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(D, meta["BLOCK_D"]))  # noqa: E731
    _squeeze_gate_kernel[grid](
        h, cond, ws, wsc, bsc, out,
        M, ND, D, DC,
        h.stride(0), h.stride(1),
        cond.stride(0), cond.stride(1),
        ws.stride(0), ws.stride(1),
        wsc.stride(0), wsc.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_DC=min(128, triton.next_power_of_2(DC)),
    )
    return out


def cond_transition_inference_composed(x, cond, wa, wb, ws, wsc, bsc):
    """Two-kernel composed inference for the token stream (large d_hidden).

    Same math/signature as ``cond_transition_inference`` but K-tiled, so d>=256 compiles.
    """
    wa = wa.contiguous(); wb = wb.contiguous(); ws = ws.contiguous()
    wsc = wsc.contiguous(); bsc = bsc.contiguous()
    h = _expand_swiglu(x, wa, wb)
    return _squeeze_gate(h, cond, ws, wsc, bsc)


def cond_transition_fwd_12_345(x, cond, wa, wb, ws, wsc, bsc):
    """The 1+2 | 3+4+5 two-triton-kernel forward — UNIFORM for atom (d=128) and token (d=768).

    Numbering the post-AdaLN ops: 1=expand(a=x@Wa^T,b=x@Wb^T) 2=SwiGLU(h=silu(a)*b)
    3=squeeze(out=h@Ws^T) 4=to_scale(scale=cond@Wsc^T+b_sc) 5=gate(y=sigmoid(scale)*out).

    Exactly TWO kernels, h:(M,ND) round-trips HBM between them (no register-resident squeeze,
    no b2b, no spill — works for any d):
      - Kernel 1 (1+2): expand + SwiGLU -> h          (``_expand_swiglu``, K-tiled, tl.dot tf32)
      - Kernel 2 (3+4+5): squeeze + to_scale + gate -> y  (``_squeeze_gate``, ND- & DC-tiled, fused)
    fp32 io, TF32 tensor cores. This is the structure to ship (simpler than the b2b/CUTLASS paths).
    """
    wa = wa.contiguous(); wb = wb.contiguous(); ws = ws.contiguous()
    wsc = wsc.contiguous(); bsc = bsc.contiguous()
    h = _expand_swiglu(x, wa, wb)            # 1+2
    return _squeeze_gate(h, cond, ws, wsc, bsc)  # 3+4+5
