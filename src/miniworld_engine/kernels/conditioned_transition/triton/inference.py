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

from miniworld_engine.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of


# BLOCK_DC (the DC-loop tile, `for c0 in range(0, DC, BLOCK_DC)`) is now a SEARCHED knob,
# crossed {64, 128} over the existing BLOCK_M/BLOCK_N/num_warps/num_stages combos (it used to be
# pinned at the launch site to min(128, next_pow2(DC))). Both values are <= the old 128 cap, so
# smem stays within the previously safe envelope.
_cfgs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_DC": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_DC": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_DC": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_DC": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_DC": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_DC": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_DC": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_DC": 128}, num_warps=8, num_stages=3),
]


def _smem_early_prune(configs, named_args, **kwargs):  # noqa: ARG001
    """Drop configs whose shared-memory footprint exceeds the device limit BEFORE compile.

    Per program the b2b inference kernel keeps a persistent x tile ``[BLOCK_M, BLOCK_K]`` (loaded
    once outside the loops) and pipelines, across ``num_stages``: in the ND loop the expand-GEMM
    weights wa + wb ``[BLOCK_K, BLOCK_N]`` and the squeeze weight ws_t ``[BLOCK_N, D]``; in the DC
    loop cond ``[BLOCK_M, BLOCK_DC]`` and wsc_t ``[BLOCK_DC, D]`` (out_acc/scale are fp32 register
    accumulators, not smem). bf16 = 2 B. Large BLOCK_M/BLOCK_N/BLOCK_DC combos overflow A100's
    ~163 KB, so prune up front: Triton's bench-time OOM pruning is unsafe under CUDA-graph capture
    (fires mid-capture, poisons the stream). Device-aware via ``max_shared_mem`` — sm90/sm100 keep
    more configs. This is composed as the base_prune UNDER the cache-narrowing prune.
    """
    import triton as _triton

    try:
        limit = _triton.runtime.driver.active.utils.get_device_properties(
            torch.cuda.current_device(),
        )["max_shared_mem"]
    except Exception:  # noqa: BLE001 -- conservative sm100 budget
        limit = 227 * 1024

    def _nget(name):  # BLOCK_K is pinned at launch (=next_pow2(K)); D is a positional constexpr
        if hasattr(named_args, "get") and name in named_args:
            return named_args[name]
        return kwargs.get(name)

    bk = _nget("BLOCK_K")
    d = _nget("D")

    def _smem(c):
        bm = c.kwargs["BLOCK_M"]
        bn = c.kwargs["BLOCK_N"]
        bdc = c.kwargs["BLOCK_DC"]
        # The ND loop and the DC loop are SEQUENTIAL (ND completes, then DC), with disjoint
        # buffer liveness, so Triton reuses the smem -> peak is the MAX of the two phases, not
        # their sum. (Summing them over-prunes to a single config on A100.) x[BM,BK] is loaded
        # once and lives only through the ND phase.
        nd_phase = bm * bk + c.num_stages * (2 * bk * bn + bn * d)   # x + (wa,wb) + ws_t
        dc_phase = c.num_stages * (bm * bdc + bdc * d)               # cond + wsc_t
        return max(nd_phase, dc_phase) * 2

    kept = [c for c in configs if _smem(c) <= limit]
    return kept or [min(configs, key=_smem)]


_cond_transition_infer_prune = make_cache_prune(
    "cond_transition_infer", dtype_of=tensor_dtype_of("x_ptr"),
    bucket_of=key_bucket_of("ND", "K", "D", "DC"), base_prune=_smem_early_prune,
)


# fmt: off
@triton.autotune(
    configs=_cfgs, key=["ND", "K", "D", "DC"],
    prune_configs_by={"early_config_prune": _cond_transition_infer_prune},
)
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
    pid_m = tl.program_id(0).to(tl.int64)
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
        h = (a * tl.sigmoid(a) * b).to(x.dtype)  # cast to operand dtype -> squeeze dot works in bf16 (no-op for fp32)
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
    )
    return out
