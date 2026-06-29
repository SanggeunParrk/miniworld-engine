"""Fully fused INFERENCE kernel for the post-AdaLN ConditionedTransition tail.

The ConditionedTransition forward, AFTER the (separately-optimized) AdaLN, is:

    a     = x @ Wa^T                       # (M, ND)   ND = n*d_hidden
    b     = x @ Wb^T                       # (M, ND)
    h     = silu(a) * b                    # SwiGLU
    out   = h @ Ws^T                       # (M, D)    D = d_hidden
    scale = cond @ Wsc^T + b_sc            # (M, D)    cond = (M, DC), DC = d_cond
    y     = sigmoid(scale) * out           # (M, D)    output gate

This is the INFERENCE path: forward only, saves nothing for backward, maximal fusion.
One program owns BLOCK_M rows and ALL of ND: it builds the gated h tile-by-tile and
accumulates the squeeze ``out[BM, D] += h_chunk @ Ws[:, chunk]^T`` in registers (the
(M, ND) intermediate h never touches HBM), then fuses the conditioning gate
``sigmoid(cond @ Wsc^T + b_sc)`` straight onto ``out`` before the single write.

fp32 inputs with TF32 tensor-core matmuls (input_precision="tf32"). Practical when K
(= d_hidden) fits one BLOCK_K and the working set fits smem — i.e. the atom stream
(d_hidden=128). The token stream (d_hidden=768) routes to the cute TF32 path.
"""

import torch
import triton
import triton.language as tl


_cfgs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]


# fmt: off
@triton.autotune(configs=_cfgs, key=["ND", "K", "D", "DC"])
@triton.jit
def _cond_transition_inference_kernel(
    x_ptr, cond_ptr, wa_ptr, wb_ptr, ws_ptr, wsc_ptr, bsc_ptr, out_ptr,
    M, ND,
    K: tl.constexpr, D: tl.constexpr, DC: tl.constexpr,
    stride_xm, stride_xk,
    stride_cm, stride_cc,
    stride_wn, stride_wk,     # Wa, Wb: (ND, K) row-major
    stride_sd, stride_sn,     # Ws: (D, ND) row-major
    stride_scd, stride_scc,   # Wsc: (D, DC) row-major
    stride_om, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_DC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    k = tl.arange(0, BLOCK_K)
    k_mask = k < K

    # x is the AdaLN output (no LN fold here — AdaLN is a separate kernel).
    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
        mask=row_mask[:, None] & k_mask[None, :], other=0.0,
    )

    dcols = tl.arange(0, D)
    out_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for n0 in range(0, ND, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        col_mask = cols < ND
        wa = tl.load(
            wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        wb = tl.load(
            wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        a = tl.dot(x, wa, out_dtype=tl.float32, input_precision="tf32")
        b = tl.dot(x, wb, out_dtype=tl.float32, input_precision="tf32")
        h = a * tl.sigmoid(a) * b  # (BM, BN) fp32
        ws_t = tl.load(  # (BN, D): Ws[d, cols]^T
            ws_ptr + cols[:, None] * stride_sn + dcols[None, :] * stride_sd,
            mask=col_mask[:, None], other=0.0,
        )
        out_acc += tl.dot(h, ws_t, out_dtype=tl.float32, input_precision="tf32")

    # Conditioning gate: scale = cond @ Wsc^T + b_sc ; y = sigmoid(scale) * out.
    # DC is tiled (the full (BLOCK_DC, D) Wsc^T tile would blow smem in fp32 at DC=384).
    scale = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for c0 in range(0, DC, BLOCK_DC):
        dc = c0 + tl.arange(0, BLOCK_DC)
        dc_mask = dc < DC
        cond = tl.load(
            cond_ptr + rows[:, None] * stride_cm + dc[None, :] * stride_cc,
            mask=row_mask[:, None] & dc_mask[None, :], other=0.0,
        )
        wsc_t = tl.load(  # (BLOCK_DC, D): Wsc[d, dc]^T
            wsc_ptr + dcols[None, :] * stride_scd + dc[:, None] * stride_scc,
            mask=dc_mask[:, None], other=0.0,
        )
        scale += tl.dot(cond, wsc_t, out_dtype=tl.float32, input_precision="tf32")
    bsc = tl.load(bsc_ptr + dcols)
    scale += bsc[None, :]
    y = tl.sigmoid(scale) * out_acc
    tl.store(
        out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od,
        y, mask=row_mask[:, None],
    )
# fmt: on


def cond_transition_inference(
    x: torch.Tensor,     # (M, K)  AdaLN output, K = d_hidden
    cond: torch.Tensor,  # (M, DC) conditioning, DC = d_cond
    wa: torch.Tensor,    # (ND, K) expand_a.weight
    wb: torch.Tensor,    # (ND, K) expand_b.weight
    ws: torch.Tensor,    # (D, ND) squeeze.weight, D = d_hidden
    wsc: torch.Tensor,   # (D, DC) to_scale.weight
    bsc: torch.Tensor,   # (D,)    to_scale.bias
) -> torch.Tensor:
    """Fused inference: SwiGLU expand+squeeze + sigmoid(cond-gate). y never round-trips h."""
    M, K = x.shape
    ND = wa.shape[0]
    D = ws.shape[0]
    DC = cond.shape[1]
    out = torch.empty(M, D, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _cond_transition_inference_kernel[grid](
        x, cond, wa.contiguous(), wb.contiguous(), ws.contiguous(),
        wsc.contiguous(), bsc.contiguous(), out,
        M, ND, K, D, DC,
        x.stride(0), x.stride(1),
        cond.stride(0), cond.stride(1),
        wa.stride(0), wa.stride(1),
        ws.stride(0), ws.stride(1),
        wsc.stride(0), wsc.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=triton.next_power_of_2(K),
        BLOCK_DC=min(128, triton.next_power_of_2(DC)),
    )
    return out
