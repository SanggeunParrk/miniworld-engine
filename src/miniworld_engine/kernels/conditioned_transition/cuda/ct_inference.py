"""H100 (sm_90a) ConditionedTransition tail, inference: ``ct_sm90.cu``.

    h = silu(x Wa^T) * (x Wb^T)                            ``expand_swiglu``: one wgmma GEMM, [a | b] never reaches HBM
    y = sigmoid(cond Wsc^T + b) * (h Ws^T)  (+ residual)   ``squeeze_gate``: both products in one pass over the output tile

``x`` is the output of the module's AdaLN (the kernel is the post-AdaLN tail, as the Triton family's). bf16 on wgmma k16; fp32 on the TF32 tensor
cores (the weights are rounded to the TF32 grid to nearest once per weight pack, ``h`` by the epilogue that writes it). Two launches, the second
under programmatic dependent launch. Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.adaln.cuda.infer import num_sms, pick_ht, rna_tf32

_dir = Path(__file__).parent
_kernels = _dir.parent.parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="ct_sm90" + ("_trace" if os.environ.get("CT_TRACE") else "") + ("_nomc" if os.environ.get("CT_NOMC") else ""),
        sources=[str(_dir / "ct_sm90.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            *gencodes("90a"),
            "--expt-relaxed-constexpr",
            f"-I{_kernels / 'transition' / 'cuda' / 'anthropic_v5'}",
            f"-I{_dir}",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *(["-DCT_TRACE"] if os.environ.get("CT_TRACE") else []),
            *(["-DCT_NOMC_BUILD"] if os.environ.get("CT_NOMC") else []),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def supported(x: torch.Tensor, cond: torch.Tensor, nd: int, d: int) -> bool:
    """sm_90 only; bf16, or fp32 under TF32 (``allow_tf32``); K = d and ND, DC multiples of the K slab, D a multiple of 64."""
    if not (x.is_cuda and cond.is_cuda and x.dtype == cond.dtype):
        return False
    if x.dtype == torch.float32:
        if not torch.backends.cuda.matmul.allow_tf32:
            return False
        kq = 32
    elif x.dtype == torch.bfloat16:
        kq = 64
    else:
        return False
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    dc = cond.shape[-1]
    return torch.cuda.get_device_capability(idx) == (9, 0) and d % max(kq, 64) == 0 and nd % 128 == 0 and dc % kq == 0


def pack_weights(wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor, ht: int, dtype: torch.dtype):
    """([Wa; Wb] per tile of ``ht`` hidden units [2 ND, K], Ws [D, ND], Wsc [D, DC], the gate bias fp32): ``dtype`` (fp32: on the TF32 grid)."""
    nd, k = wa.shape
    a, b = wa.to(dtype), wb.to(dtype)
    wab = torch.stack([a.reshape(nd // ht, ht, k), b.reshape(nd // ht, ht, k)], 1).reshape(2 * nd, k)
    s, g = ws.to(dtype).contiguous(), wsc.to(dtype).contiguous()
    if dtype == torch.float32:
        wab, s, g = rna_tf32(wab), rna_tf32(s), rna_tf32(g)
    return wab.contiguous(), s, g, bsc.to(dtype).float().contiguous()


def packed(wa, wb, ws, wsc, bsc, ht, dtype):
    return _capture.param_forms(
        f"ct_cuda_{dtype}_{ht}", (wa, wb, ws, wsc, bsc), lambda: pack_weights(wa, wb, ws, wsc, bsc, ht, dtype)
    )


def _infer_fake(x, cond, wab, ws, wsc, bsc, residual, ht, bn, rnd, cl):
    """y [M, D]."""
    return x.new_empty((x.shape[0], ws.shape[0]))


@opaque(fake=_infer_fake, name="conditioned_transition_cuda_inference")
def _infer(x: torch.Tensor, cond: torch.Tensor, wab: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor,
           residual: torch.Tensor | None, ht: int, bn: int, rnd: bool, cl: int) -> torch.Tensor:
    """The tail: expand + SwiGLU GEMM, then squeeze + gate (+ residual) GEMM, the second under programmatic dependent launch."""
    e = ext()
    m = x.shape[0]
    nd = wab.shape[0] // 2
    nsm = num_sms(x.device)
    h = torch.empty(m, nd, device=x.device, dtype=x.dtype)
    out = torch.empty(m, ws.shape[0], device=x.device, dtype=x.dtype)
    e.expand_swiglu(x, wab, h, nsm, ht, rnd, True, cl)
    e.squeeze_gate(h, ws, cond, wsc, bsc, residual, out, nsm, bn, True)
    return out


def ct_inference(x, cond, wa, wb, ws, wsc, bsc, residual=None, *, ht=None, bn=None, cl=1):
    """``residual + sigmoid(cond Wsc^T + b) * (silu(x Wa^T) * (x Wb^T)) Ws^T`` for ``x`` (M, D) and ``cond`` (M, DC) of one dtype (bf16 / fp32
    under TF32); the weights in any dtype (cast and packed once per weight pack). ``residual`` (M, D) is added after the gated update is rounded to
    ``x``'s dtype, as the unfused ``residual + y``."""
    m, d = x.shape
    nd = wa.shape[0]
    ht = ht or pick_ht(m, nd, num_sms(x.device))
    if bn is None:   # output tile columns of the squeeze: a small M gives 30 tiles of 128 (5 row blocks x 6) on 132 SMs, each streaming K = 1920: 64 columns double the CTAs
        bn = 64 if ((m + 127) // 128) * (d // 128) * 2 <= num_sms(x.device) else 128
    wab, s, g, bs = packed(wa, wb, ws, wsc, bsc, ht, x.dtype)
    x = x if x.stride(1) == 1 else x.contiguous()
    cond = cond if cond.stride(1) == 1 else cond.contiguous()
    return _infer(x, cond, wab, s, g, bs, residual, ht, bn, x.dtype == torch.float32, cl)


__all__ = ["ct_inference", "ext", "pack_weights", "supported"]

