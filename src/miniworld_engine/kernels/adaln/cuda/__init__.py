"""H100 hand-CUDA adaptive LayerNorm ``y = sigmoid(scale) * LN(x) + bias``, bf16 or fp32 activations, forward and backward.

``adaln_rows.cu``: one-warp-per-row kernels around the two cuBLAS GEMMs (the [scale | bias] projection of LN(cond) in one GEMM, its
weight gradient and dcn in two more). The PyTorch leg runs the same computation as ten fp32 elementwise / LayerNorm kernels
(casts, two statistics passes, the gate chain, a cast back); here it is ``ln_cond_fwd`` -> GEMM -> ``adaln_fwd`` and, backward,
``adaln_bwd`` -> GEMM x3 -> ``ln_cond_bwd``. Statistics are fp32 and recomputed from the row in registers; every reduction has one
fixed order (block partials summed in order), so the op is bitwise repeatable.

Contract (``supported``): sm_90, bf16 x / cond, widths in ``WIDTHS`` (a multiple of 64), x.shape[:-1] == cond.shape[:-1].
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque

from miniworld_engine import settings

_dir = Path(__file__).parent
WIDTHS = (128, 256, 384, 512, 768, 1024)


@functools.lru_cache(maxsize=None)
def _ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="adaln_rows_sm90",
        sources=[str(_dir / "adaln_rows.cu")],
        extra_cuda_cflags=[
            *host_flags(), "-std=c++17", "-O3", *gencodes("90a"), "--expt-relaxed-constexpr",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


_BUILD_FAILED = False


def supported(x: torch.Tensor, cond: torch.Tensor) -> bool:
    """The contract above (and the ``adaln_h100_cuda`` setting)."""
    if not settings.current().adaln_h100_cuda or not x.is_cuda or not cond.is_cuda:
        return False
    if x.dtype is torch.float32 and torch.is_autocast_enabled("cuda"):
        return False  # autocast would run the fp32 call's GEMMs in bf16: the PyTorch leg decides there
    if x.dtype not in (torch.bfloat16, torch.float32) or cond.dtype is not x.dtype or x.shape[:-1] != cond.shape[:-1]:
        return False
    if x.shape[-1] not in WIDTHS or cond.shape[-1] not in WIDTHS or x.numel() == 0:
        return False
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(idx) == (9, 0)


def available(x: torch.Tensor, cond: torch.Tensor) -> bool:
    """``supported()`` plus a successful build. Fake tensors outside a ``torch.compile`` trace (``dev derive``) keep the PyTorch leg; inside a
    trace the route is taken as the opaque ops below (the extension is built when the op first runs)."""
    global _BUILD_FAILED
    if _BUILD_FAILED or not supported(x, cond):
        return False
    from torch._subclasses.fake_tensor import FakeTensor

    if torch.compiler.is_compiling():
        return True
    if isinstance(x, FakeTensor) or isinstance(cond, FakeTensor):
        return False
    try:
        _ext()
    except Exception as exc:  # noqa: BLE001
        import warnings

        _BUILD_FAILED = True
        warnings.warn(f"H100 CUDA adaLN unavailable, keeping the PyTorch leg: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _wcat_bcat(x, ws, bs, wb):
    """[Ws | Wb] [2 d, dc] and [bs | 0] [2 d] in the activations' dtype (bf16: one pack launch; fp32: the GEMMs follow the process's TF32 setting)."""
    d, dc = ws.shape
    if x.dtype is torch.bfloat16:
        wcat = torch.empty(2 * d, dc, device=x.device, dtype=torch.bfloat16)
        bcat = torch.empty(2 * d, device=x.device, dtype=torch.bfloat16)
        _ext().pack(ws.contiguous(), wb.contiguous(), None if bs is None else bs.contiguous(), wcat, bcat)
        return wcat, bcat
    wcat = torch.cat([ws, wb]).to(torch.float32)
    bcat = torch.cat([bs if bs is not None else ws.new_zeros(d), ws.new_zeros(d)]).to(torch.float32)
    return wcat, bcat


def _fwd_fake(x2, c2, wc, ws, bs, wb, eps_x, eps_c):
    r, d = x2.shape
    return (
        x2.new_empty((r, d)),
        c2.new_empty(c2.shape),
        x2.new_empty((r, 2), dtype=torch.float32),
        x2.new_empty((r, 2), dtype=torch.float32),
        x2.new_empty((r, 2 * d)),
    )


@opaque(fake=_fwd_fake, name="adaln_h100_fwd")
def _fwd(
    x2: torch.Tensor, c2: torch.Tensor, wc: torch.Tensor, ws: torch.Tensor, bs: torch.Tensor | None, wb: torch.Tensor, eps_x: float, eps_c: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(y, LN(cond) wc, its statistics, x's statistics, [scale | bias] pre-activations) of the rows x2 [r, d], c2 [r, dc]."""
    e = _ext()
    cn, stc = e.ln_cond_fwd(c2, wc.contiguous(), eps_c)
    wcat, bcat = _wcat_bcat(x2, ws, bs, wb)
    sb = torch.addmm(bcat, cn, wcat.t())
    y, stx = e.adaln_fwd(x2, sb, eps_x)
    return y, cn, stc, stx, sb


def _bwd_fake(dy2, x2, c2, cn, stc, stx, sb, wc, ws, bs, wb, has_bs):
    d, dc = ws.shape
    return (
        x2.new_empty(x2.shape),
        c2.new_empty(c2.shape),
        wc.new_empty((dc,), dtype=torch.float32),
        ws.new_empty((d, dc), dtype=torch.float32),
        ws.new_empty((d if has_bs else 0,), dtype=torch.float32),
        ws.new_empty((d, dc), dtype=torch.float32),
    )


@opaque(fake=_bwd_fake, name="adaln_h100_bwd")
def _bwd(
    dy2: torch.Tensor, x2: torch.Tensor, c2: torch.Tensor, cn: torch.Tensor, stc: torch.Tensor, stx: torch.Tensor, sb: torch.Tensor,
    wc: torch.Tensor, ws: torch.Tensor, bs: torch.Tensor | None, wb: torch.Tensor, has_bs: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dcond, dwc, dWs, dbs (empty without a scale bias), dWb) from dy2 [r, d] and the forward's saved rows."""
    e = _ext()
    d = x2.shape[-1]
    wcat, _ = _wcat_bcat(x2, ws, bs, wb)
    dsb, dx, pbs = e.adaln_bwd(dy2, x2, sb, stx)
    dcn = torch.mm(dsb, wcat)
    dwcat = torch.mm(dsb.t(), cn, out_dtype=torch.float32) if dsb.dtype is torch.bfloat16 else torch.mm(dsb.t(), cn)  # [2 d, dc]
    dcond, pbc = e.ln_cond_bwd(dcn, c2, stc, wc.contiguous())
    dbs = pbs.sum(0) if has_bs else pbs.new_empty(0)
    return dx, dcond, pbc.sum(0), dwcat[:d].clone(), dbs, dwcat[d:].clone()


class _AdaLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, wc, ws, bs, wb, eps_x, eps_c):  # noqa: D102
        d, dc = x.shape[-1], cond.shape[-1]
        x2, c2 = x.reshape(-1, d).contiguous(), cond.reshape(-1, dc).contiguous()
        from . import infer

        if infer.supported(x2, c2, d, dc):   # the sm_90a wgmma forward (ln_rows + one GEMM with the gate and the LayerNorm in its epilogue)
            y, cn, stc, stx, sb = infer.fwd_saved(x2, c2, wc, ws, bs if bs is not None else ws.new_zeros(d), wb, eps_x, eps_c,
                                                  infer.pick_ht(x2.shape[0], d, infer.num_sms(x2.device)))
        else:
            y, cn, stc, stx, sb = _fwd(x2, c2, wc, ws, bs, wb, eps_x, eps_c)
        if any(ctx.needs_input_grad):
            # the weights are saved too: autograd then refuses an in-place update between the passes
            ctx.save_for_backward(x2, c2, cn, stc, stx, sb, wc, ws, wb, *([] if bs is None else [bs]))
            ctx.shape, ctx.cshape, ctx.has_bs = x.shape, cond.shape, bs is not None
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):  # noqa: D102
        x2, c2, cn, stc, stx, sb, wc, ws, wb, *rest = ctx.saved_tensors
        bs = rest[0] if ctx.has_bs else None
        dy2 = dy.reshape(-1, x2.shape[-1]).contiguous()
        dx, dcond, dwc, dws, dbs, dwb = _bwd(dy2, x2, c2, cn, stc, stx, sb, wc, ws, bs, wb, ctx.has_bs)
        return dx.view(ctx.shape), dcond.view(ctx.cshape), dwc, dws, (dbs if ctx.has_bs else None), dwb, None, None


def adaln(x, cond, ln_cond_weight, w_scale, b_scale, w_bias, eps_x: float = 1e-5, eps_cond: float = 1e-5, *, rna: bool = False):
    """``sigmoid(LN(c) w_scale^T + b_scale) * LN(x) + LN(c) w_bias^T`` with LN(c) = LayerNorm(cond) * ln_cond_weight; bf16 or fp32 x / cond
    [..., d] / [..., dc]; the weights may be fp32 masters (rounded to bf16 in one pack launch for bf16 activations) or bf16. Returns x's dtype.

    One entry, two routes by mode: a call that records no backward runs the sm_90a wgmma kernels of ``infer.py`` (``ln_rows`` + one GEMM with the
    gate and the LayerNorm of x in its epilogue, two launches; bf16, or fp32 under TF32; ``rna``: the fp32 result is rounded to the TF32 grid for a
    TF32 GEMM that consumes it); everything else (training, strict fp32, widths the wgmma tile does not take) the row kernels + cuBLAS below.
    A ``torch.compile`` trace sees opaque ops (``adaln_cuda_inference`` or ``adaln_h100_fwd`` / ``adaln_h100_bwd``)."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, cond, ln_cond_weight, w_scale, b_scale, w_bias) if t is not None)):
        from . import infer

        if infer.supported(x, cond, x.shape[-1], cond.shape[-1]):
            return infer.adaln_inference(x, cond, ln_cond_weight, w_scale, b_scale, w_bias, eps_x, eps_cond, rna=rna)
    return _AdaLN.apply(x, cond, ln_cond_weight, w_scale, b_scale, w_bias, eps_x, eps_cond)


__all__ = ["adaln", "available", "supported"]
