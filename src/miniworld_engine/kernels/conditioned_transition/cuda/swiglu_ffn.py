"""H100 (sm_90) CUDA wiring of ``swiglu_ffn`` (``W_down( SiLU(Wa @ x) * (Wb @ x) )``, weights as arguments) on the kernels of the conditioned-transition tail.

``swiglu_ffn`` answers ``None`` when no CUDA kernel serves the call -- another card, a dtype / width outside the kernel's contract, a CPU tensor, a
``torch.compile`` trace -- and the caller runs the PyTorch statements. Nothing here launches Triton and nothing here is a new kernel.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels._device import on_h100


def _tf32_or_bf16(x: torch.Tensor) -> bool:
    return x.dtype is torch.bfloat16 or (x.dtype is torch.float32 and torch.backends.cuda.matmul.allow_tf32)


class _SwiGLUFFN(torch.autograd.Function):
    """``Ws (silu(Wa x) * (Wb x))``: one cuBLAS GEMM over [Wa; Wb], the CUDA SwiGLU row kernel (``ct_rows``), one GEMM; backward the
    kernels of the conditioned-transition tail without its gate."""

    @staticmethod
    def forward(ctx, x, wa, wb, ws):  # noqa: D102
        from miniworld_engine.kernels.conditioned_transition.cuda.ct_train import swiglu_fwd

        dt = x.dtype
        wab = torch.cat([wa, wb]).to(dt)
        wsb = ws.to(dt)
        ab = torch.mm(x, wab.t())
        h = swiglu_fwd(ab)
        out = torch.mm(h, wsb.t())
        ctx.save_for_backward(x, ab, h, wab, wsb)
        ctx.dtypes = (wa.dtype, wb.dtype, ws.dtype)
        return out

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        from miniworld_engine.kernels.conditioned_transition.cuda.ct_train import swiglu_bwd

        x, ab, h, wab, wsb = ctx.saved_tensors
        dy = dy.contiguous()
        dws = torch.mm(dy.t(), h)
        dab = swiglu_bwd(torch.mm(dy, wsb), ab)
        dwab = torch.mm(dab.t(), x)
        dx = torch.mm(dab, wab)
        nd = ab.shape[1] // 2
        da, db, ds = ctx.dtypes
        return dx, dwab[:nd].to(da), dwab[nd:].to(db), dws.to(ds)


_GATE_ONE: dict = {}

#: Widest hidden (rows of ``wa``) at which bf16 inference takes the fused ``ct_inference`` kernel; wider ones take cuBLAS + the SwiGLU row kernel.
_CT_MAX_HIDDEN_BF16 = 512


def _gate_of_one(x: torch.Tensor):
    """(Wsc, bias) of a gate that is exactly one for any cond: zero weights and bias 30 (sigmoid(30) rounds to 1.0f). Cached per (device, dtype,
    width) so the weight-pack cache of ``ct_inference`` keeps one entry instead of one per call."""
    key = (x.device, x.dtype, x.shape[-1])
    got = _GATE_ONE.get(key)
    if got is None:
        d = x.shape[-1]
        got = _GATE_ONE[key] = (x.new_zeros(d, d), x.new_full((d,), 30.0))
    return got


def swiglu_ffn(x, wa, wb, ws):
    """``swiglu_ffn`` on the CUDA kernels of the conditioned-transition tail, or None. Inference (bf16, or fp32 under TF32): the fused
    expand + SwiGLU GEMM and the squeeze GEMM of ``ct_inference`` (the hidden activation is the only HBM round trip) with a gate that is exactly
    one (``cond = x``, see ``_gate_of_one``). Otherwise (training, fp32 at the default matmul precision): ``_SwiGLUFFN``."""
    if not on_h100(x, wa, wb, ws) or x.dtype not in (torch.bfloat16, torch.float32) or x.shape[-1] % 8 or wa.shape[0] % 8:
        return None
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, wa, wb, ws)):
        return _SwiGLUFFN.apply(x2, wa, wb, ws).reshape(*shape[:-1], ws.shape[0])
    from miniworld_engine.kernels.conditioned_transition.cuda import ct_inference as ct

    d = shape[-1]
    # bf16 with a hidden width above 512: the fused expand + SwiGLU GEMM writes a hidden activation that is no longer small next to the GEMM, and
    # cuBLAS + the SwiGLU row kernel is as fast or faster (CUDA graph, H100, hidden 768 / 1536, 128 - 65536 rows: 0.8 - 1.0x of ct_inference's time;
    # the fused kernel was 1.04 - 1.37x SLOWER than the PyTorch statements at hidden 1536). At hidden <= 512 the fused kernel stays (1.2 - 1.8x faster).
    wide_bf16 = x.dtype is torch.bfloat16 and wa.shape[0] > _CT_MAX_HIDDEN_BF16
    if not wide_bf16 and _tf32_or_bf16(x) and ct.supported(x2, x2, wa.shape[0], d):
        wsc, bsc = _gate_of_one(x2)
        return ct.ct_inference(x2, x2, wa, wb, ws, wsc, bsc).reshape(*shape[:-1], ws.shape[0])
    return _SwiGLUFFN.apply(x2, wa, wb, ws).reshape(*shape[:-1], ws.shape[0])
