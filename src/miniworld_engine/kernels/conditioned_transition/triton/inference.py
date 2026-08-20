"""Fully fused INFERENCE kernel for the post-AdaLN ConditionedTransition tail.

The ConditionedTransition forward, AFTER the (separately-optimized) AdaLN, is:

    a     = x @ Wa^T                       # (M, ND)   ND = n*d_hidden
    b     = x @ Wb^T                       # (M, ND)
    h     = silu(a) * b                    # SwiGLU
    out   = h @ Ws^T                       # (M, D)    D = d_hidden
    scale = cond @ Wsc^T + b_sc            # (M, D)    cond = (M, DC), DC = d_cond
    y     = sigmoid(scale) * out           # (M, D)    output gate

This is the INFERENCE path: forward only, saves nothing for backward, maximal fusion.
One program owns BLOCK_M1 rows and ALL of ND: it builds the gated h tile-by-tile and
accumulates the squeeze ``out[BM, D] += h_chunk @ Ws[:, chunk]^T`` in registers (the
(M, ND) intermediate h never touches HBM), then fuses the conditioning gate
``sigmoid(cond @ Wsc^T + b_sc)`` straight onto ``out`` before the single write.

fp32 inputs with TF32 tensor-core matmuls (input_precision="tf32"). Practical when K
(= d_hidden) fits one BLOCK_K and the working set fits smem — i.e. the atom stream
(d_hidden=128). The token stream (d_hidden=768) routes to the cute TF32 path.
"""

from miniworld_engine.autotune.configs import configs_for
import torch
import triton
import triton.language as tl


from miniworld_engine.autotune import key_bucket_of, tensor_dtype_of


# Every tile comes from the CSV. BLOCK_K_D tiles the d_hidden contraction; a row that sets it >= K
# pins the whole row on-chip, which is the schedule this kernel was designed around, and the k-loop
# makes the smaller rows correct. Nothing filters rows the running card cannot fit.
#
# The squeeze/gate output width D was the other shape constant (`tl.arange(0, D)`, `[BM, D]`
# accumulators, which also silently required D to be a power of two). It is a free axis, so it
# moves onto the grid tiled by BLOCK_N -- reusing that knob rather than adding a fifth independent
# one, which would multiply this sweep by 5x for a (narrow ND tile, wide D tile) combination the
# tuner has no reason to prefer.


# fmt: off
from miniworld_engine.autotune.buckets import bucket_mixed as _bucket


def get_seq_group(rows) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    return _bucket(rows)


@triton.autotune(configs=configs_for("cond_transition_fwd_b2b_triton"), key=['ND', 'K', 'DC', 'seq_group'])
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
    BLOCK_M1: tl.constexpr, BLOCK_K_ND: tl.constexpr,
    BLOCK_K_D: tl.constexpr, BLOCK_K_DC: tl.constexpr,
    seq_group,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    rows = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    row_mask = rows < M

    dcols = pid_d * BLOCK_K_ND + tl.arange(0, BLOCK_K_ND)   # output tile of the squeeze/gate
    d_mask = dcols < D
    out_acc = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)
    for n0 in range(0, ND, BLOCK_K_ND):
        cols = n0 + tl.arange(0, BLOCK_K_ND)
        col_mask = cols < ND
        a = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)
        b = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K_D):
            k = k0 + tl.arange(0, BLOCK_K_D)
            k_mask = k < K
            # x is the AdaLN output (no LN fold here — AdaLN is a separate kernel).
            x = tl.load(
                x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            )
            wa = tl.load(
                wa_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                mask=k_mask[:, None] & col_mask[None, :], other=0.0,
            )
            wb = tl.load(
                wb_ptr + k[:, None] * stride_wk + cols[None, :] * stride_wn,
                mask=k_mask[:, None] & col_mask[None, :], other=0.0,
            )
            a = tl.dot(x, wa, a, out_dtype=tl.float32, input_precision="tf32")
            b = tl.dot(x, wb, b, out_dtype=tl.float32, input_precision="tf32")
        h = (a * tl.sigmoid(a) * b).to(x_ptr.dtype.element_ty)  # operand dtype -> squeeze dot in bf16/fp32
        ws_t = tl.load(  # (BN, BN): Ws[d, cols]^T
            ws_ptr + cols[:, None] * stride_sn + dcols[None, :] * stride_sd,
            mask=col_mask[:, None] & d_mask[None, :], other=0.0,
        )
        out_acc = tl.dot(h, ws_t, out_acc, out_dtype=tl.float32, input_precision="tf32")

    # Conditioning gate: scale = cond @ Wsc^T + b_sc ; y = sigmoid(scale) * out.
    # DC is tiled (the full (BLOCK_K_DC, D) Wsc^T tile would blow smem in fp32 at DC=384).
    scale = tl.zeros((BLOCK_M1, BLOCK_K_ND), dtype=tl.float32)
    for c0 in range(0, DC, BLOCK_K_DC):
        dc = c0 + tl.arange(0, BLOCK_K_DC)
        dc_mask = dc < DC
        cond = tl.load(
            cond_ptr + rows[:, None] * stride_cm + dc[None, :] * stride_cc,
            mask=row_mask[:, None] & dc_mask[None, :], other=0.0,
        )
        wsc_t = tl.load(  # (BLOCK_K_DC, BN): Wsc[d, dc]^T
            wsc_ptr + dcols[None, :] * stride_scd + dc[:, None] * stride_scc,
            mask=dc_mask[:, None] & d_mask[None, :], other=0.0,
        )
        scale = tl.dot(cond, wsc_t, scale, out_dtype=tl.float32, input_precision="tf32")
    bsc = tl.load(bsc_ptr + dcols, mask=d_mask, other=0.0)
    scale += bsc[None, :]
    y = tl.sigmoid(scale) * out_acc
    tl.store(
        out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od,
        y, mask=row_mask[:, None] & d_mask[None, :],
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
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M1"]), triton.cdiv(D, meta["BLOCK_K_ND"]),
    )
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
        seq_group=get_seq_group(M),
    )
    return out
