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
import os
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque

from miniworld_engine import settings

_dir = Path(__file__).parent
WIDTHS = (128, 256, 384, 512, 768, 1024)
FEATURE_MAJOR_WIDTHS = (256, 384, 512, 768)   # fp32 backward row pass writing the gradient feature-major (``adaln_bwd_t``)
_FM = os.environ.get("ADALN_FM", "auto")   # "1": always, "0": never, "auto": a cond of 768 columns or more


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


def pack_fp32(ws, wb, bs):
    """([Ws; Wb] [2 d, dc], its transpose [dc, 2 d] (the K-major right operand of the TF32 dgrad GEMM), [bs | 0] [2 d]) fp32, in ONE launch."""
    d, dc = ws.shape
    dev = ws.device
    f32 = torch.float32
    wcat = torch.empty(2 * d, dc, device=dev, dtype=f32)
    wcatT = torch.empty(dc, 2 * d, device=dev, dtype=f32)
    bcat = torch.zeros(2 * d, device=dev, dtype=f32)
    ws, wb = ws.to(f32).contiguous(), wb.to(f32).contiguous()
    srcs, dsts, spec = [ws, wb, ws, wb], [wcat, wcat, wcatT, wcatT], [[0, 0, 0, 1], [0, d, 0, 1], [1, 0, 0, 1], [1, 0, d, 1]]
    if bs is not None:
        srcs.append(bs.to(f32).contiguous().view(1, d))
        dsts.append(bcat.view(1, 2 * d))
        spec.append([0, 0, 0])
    _ext().pack_f32(srcs, dsts, spec)
    return wcat, wcatT, bcat


def _fwd_fake(x2, c2, wc, ws, bs, wb, eps_x, eps_c):
    """Output structure of ``adaln_h100_fwd``."""
    r, d = x2.shape
    return (
        x2.new_empty((r, d)),
        c2.new_empty(c2.shape),
        x2.new_empty((r, 2), dtype=torch.float32),
        x2.new_empty((r, 2), dtype=torch.float32),
        x2.new_empty((r, 2 * d)),
        x2.new_empty((2 * d, c2.shape[1])),
        x2.new_empty((c2.shape[1], 2 * d) if x2.dtype is torch.float32 else (0,)),
    )


@opaque(fake=_fwd_fake, name="adaln_h100_fwd")
def _fwd(
    x2: torch.Tensor, c2: torch.Tensor, wc: torch.Tensor, ws: torch.Tensor, bs: torch.Tensor | None, wb: torch.Tensor, eps_x: float, eps_c: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(y, LN(cond) wc, its statistics, x's statistics, [scale | bias] pre-activations, the packed [Ws; Wb], and for fp32 its K-major transpose [dc, 2 d] for
    the backward) of the rows x2 [r, d], c2 [r, dc]."""
    e = _ext()
    cn, stc = e.ln_cond_fwd(c2, wc.contiguous(), eps_c)
    if x2.dtype is torch.float32:
        wcat, wcatT, bcat = pack_fp32(ws, wb, bs)
    else:
        wcat, bcat = _wcat_bcat(x2, ws, bs, wb)
        wcatT = wcat.new_empty(0)
    sb = torch.addmm(bcat, cn, wcat.t())
    y, stx = e.adaln_fwd(x2, sb, eps_x)
    return y, cn, stc, stx, sb, wcat, wcatT


def _bwd_fake(dy2, x2, c2, cn, stc, stx, sb, wc, ws, bs, wb, has_bs, dres=None, dcres=None, wcat=None, wcatT=None):
    """Output structure of ``adaln_h100_bwd``."""
    d, dc = ws.shape
    return (
        x2.new_empty(x2.shape),
        c2.new_empty(c2.shape),
        wc.new_empty((dc,), dtype=torch.float32),
        ws.new_empty((2 * d, dc), dtype=torch.float32),
        ws.new_empty((d if has_bs else 0,), dtype=torch.float32),
    )


@opaque(fake=_bwd_fake, name="adaln_h100_bwd")
def _bwd(
    dy2: torch.Tensor, x2: torch.Tensor, c2: torch.Tensor, cn: torch.Tensor, stc: torch.Tensor, stx: torch.Tensor, sb: torch.Tensor,
    wc: torch.Tensor, ws: torch.Tensor, bs: torch.Tensor | None, wb: torch.Tensor, has_bs: bool,
    dres: torch.Tensor | None = None, dcres: torch.Tensor | None = None, wcat: torch.Tensor | None = None, wcatT: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dcond, dwc, [dWs; dWb] as one [2 d, dc] fp32 tensor, dbs (empty without a scale bias)) from dy2 [r, d] and the forward's saved rows. ``dres`` / ``dcres``: gradients of the
    stream outputs (x, cond returned as they are for a caller that adds them to something later), which join dx / dcond inside the row kernels that
    write them instead of in a separate add of the two paths."""
    e = _ext()
    d = x2.shape[-1]
    if x2.dtype is torch.float32 and (wcat is None or wcat.numel() == 0 or wcatT is None or wcatT.numel() == 0):   # not handed over by the forward: pack again, one launch
        wcat, wcatT, _ = pack_fp32(ws, wb, bs)
    elif wcat is None or wcat.numel() == 0:
        wcat, _ = _wcat_bcat(x2, ws, bs, wb)
    from . import infer

    if d == 128 and c2.shape[1] == 128 and infer.atom_fused(128, 128):
        # the atom width: dgrad + the LayerNorm(cond) backward in one wgmma kernel (dcn never reaches HBM); the weight gradient stays a cuBLAS GEMM on dsb
        dsb, dx, pbs = e.adaln_bwd(dy2, x2, sb, stx, dres, x2.dtype is torch.float32)
        wt = wcatT if x2.dtype is torch.float32 else wcat.t().contiguous()
        dcond, pbc = infer.ext().adaln_dgrad_atom(dsb, wt, c2, stc, wc.contiguous(), dcres, infer.num_sms(x2.device))
        if dsb.dtype is torch.bfloat16:
            dwcat = torch.mm(dsb.t(), cn, out_dtype=torch.float32)
        else:
            dwcat = torch.mm(cn.t(), dsb).t()
        dbs, dwc = infer.colsum2(pbs, pbc) if has_bs else (pbs.new_empty(0), infer.colsum(pbc))
        return dx, dcond, dwc, dwcat.contiguous(), dbs
    fm = _FM
    if x2.dtype is torch.float32 and d in FEATURE_MAJOR_WIDTHS and (fm == "1" or (fm == "auto" and c2.shape[1] >= 768)):
        # TF32 weight gradients want the activation rows (K) contiguous: the row pass writes [dscale ; dy]^T feature-major (tpose.cuh)
        m = x2.shape[0]
        dfm, dx, pbs = e.adaln_bwd_t(dy2, x2, sb, stx, dres)
        dfm = dfm[:, :m]
        dcn = torch.mm(dfm.t(), wcatT.t())
        dwcat = torch.mm(dfm, cn)   # [2 d, dc]
        dcond, pbc = e.ln_cond_bwd(dcn, c2, stc, wc.contiguous(), dcres)
        from .infer import colsum, colsum2

        dbs, dwc = colsum2(pbs, pbc) if has_bs else (pbs.new_empty(0), colsum(pbc))
        return dx, dcond, dwc, dwcat.contiguous(), dbs
    dsb, dx, pbs = e.adaln_bwd(dy2, x2, sb, stx, dres)
    if dsb.dtype is torch.bfloat16:
        dcn = torch.mm(dsb, wcat)
        dwcat = torch.mm(dsb.t(), cn, out_dtype=torch.float32)  # [2 d, dc]
    else:  # TF32: the layouts cuBLAS runs fastest (measured at 18432 rows: dW 148 -> 119 us, dcn 127 -> 91 us)
        dcn = torch.mm(dsb, wcatT.t())
        dwcat = torch.mm(cn.t(), dsb).t()  # [2 d, dc]
    dcond, pbc = e.ln_cond_bwd(dcn, c2, stc, wc.contiguous(), dcres)
    from .infer import colsum, colsum2   # fixed-order column sums of the block partials: one launch each instead of a PyTorch reduction

    dbs, dwc = colsum2(pbs, pbc) if has_bs else (pbs.new_empty(0), colsum(pbc))
    return dx, dcond, dwc, dwcat.contiguous(), dbs


class _AdaLN(torch.autograd.Function):
    """``wgmma``: the forward is the sm_90a wgmma one (``infer.fwd_saved``) instead of ``ln_cond_fwd`` -> cuBLAS -> ``adaln_fwd``; the saved tensors and the
    backward are the same. ``streams``: also return x and cond themselves, as outputs of this node, so that their gradients (arriving from whatever adds
    them to its result) join the backward's row kernels."""

    @staticmethod
    def forward(ctx, x, cond, wc, ws, bs, wb, eps_x, eps_c, wgmma=False, streams=False):  # noqa: D102
        d, dc = x.shape[-1], cond.shape[-1]
        x2, c2 = x.reshape(-1, d).contiguous(), cond.reshape(-1, dc).contiguous()
        if wgmma:
            from . import infer

            y, cn, stc, stx, sb, wcat = infer.fwd_saved(x2, c2, wc, ws, bs if bs is not None else ws.new_zeros(d), wb, eps_x, eps_c,
                                                  infer.pick_ht(x2.shape[0], d, infer.num_sms(x2.device)))
            wcatT = wcat.new_empty(0)
        else:
            y, cn, stc, stx, sb, wcat, wcatT = _fwd(x2, c2, wc, ws, bs, wb, eps_x, eps_c)
        if any(ctx.needs_input_grad):
            # the weights are saved too: autograd then refuses an in-place update between the passes
            ctx.save_for_backward(x2, c2, cn, stc, stx, sb, wc, ws, wb, wcat, wcatT, *([] if bs is None else [bs]))
            ctx.shape, ctx.cshape, ctx.has_bs = x.shape, cond.shape, bs is not None
        return (y.view(x.shape), x.view_as(x), cond.view_as(cond)) if streams else y.view(x.shape)

    @staticmethod
    def backward(ctx, dy, dxs=None, dcs=None):  # noqa: D102
        x2, c2, cn, stc, stx, sb, wc, ws, wb, wcat, wcatT, *rest = ctx.saved_tensors
        bs = rest[0] if ctx.has_bs else None
        dy2 = dy.reshape(-1, x2.shape[-1]).contiguous()
        dres = None if dxs is None else dxs.reshape(-1, x2.shape[-1]).contiguous()
        dcres = None if dcs is None else dcs.reshape(-1, c2.shape[-1]).contiguous()
        dx, dcond, dwc, dwcat, dbs = _bwd(dy2, x2, c2, cn, stc, stx, sb, wc, ws, bs, wb, ctx.has_bs, dres, dcres, wcat, wcatT)
        d = x2.shape[-1]
        return dx.view(ctx.shape), dcond.view(ctx.cshape), dwc, dwcat[:d], (dbs if ctx.has_bs else None), dwcat[d:], None, None, None, None


def adaln(x, cond, ln_cond_weight, w_scale, b_scale, w_bias, eps_x: float = 1e-5, eps_cond: float = 1e-5, *, rna: bool = False, streams: bool = False):
    """``sigmoid(LN(c) w_scale^T + b_scale) * LN(x) + LN(c) w_bias^T`` with LN(c) = LayerNorm(cond) * ln_cond_weight; bf16 or fp32 x / cond
    [..., d] / [..., dc]; the weights may be fp32 masters (rounded to bf16 in one pack launch for bf16 activations) or bf16. Returns x's dtype.

    The ONE entry of the H100 AdaLN; the route is chosen here by mode, dtype and shape:

    * no backward recorded: the sm_90a wgmma kernels of ``infer.py`` (``ln_rows`` + one GEMM with the gate and the LayerNorm of x in its epilogue, the
      second launch under PDL). ``rna``: round an fp32 result to the TF32 grid (its consumer is a TF32 GEMM);
    * a backward recorded: the same wgmma forward for bf16 (saving what the backward reads), the row kernels + cuBLAS of ``adaln_rows.cu`` for fp32
      (measured faster there: 171 against 208 us at 18432 rows), then their backward.

    ``streams`` (training): return ``(y, x, cond)``, the last two being x and cond as outputs of the same autograd node: a caller that adds them to
    its own result (the residual stream, the gate projection of cond) gets their gradients folded into the backward's row kernels.
    A ``torch.compile`` trace sees opaque ops (``adaln_cuda_*`` / ``adaln_h100_fwd`` / ``adaln_h100_bwd``)."""
    from . import infer

    grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, cond, ln_cond_weight, w_scale, b_scale, w_bias))
    wgmma = infer.supported(x, cond, x.shape[-1], cond.shape[-1])
    if wgmma and not grad:
        y = infer.adaln_inference(x, cond, ln_cond_weight, w_scale, b_scale, w_bias, eps_x, eps_cond, rna=rna)
        return (y, x, cond) if streams else y
    return _AdaLN.apply(x, cond, ln_cond_weight, w_scale, b_scale, w_bias, eps_x, eps_cond, wgmma and infer.train_supported(x, cond), streams)


def wgmma_supported(x, cond) -> bool:
    """Whether ``adaln()`` takes the sm_90a wgmma route for this call (a consumer that chains a wgmma GEMM on the result asks)."""
    from . import infer

    return infer.supported(x, cond, x.shape[-1], cond.shape[-1])


__all__ = ["adaln", "available", "supported", "wgmma_supported"]
