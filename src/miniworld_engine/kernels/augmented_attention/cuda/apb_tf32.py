"""AttentionPairBias in fp32 (TF32) on H100: the Pairformer single track's attention as hand-written CUDA.

``og = sigmoid(g) softmax(q k^T / sqrt(d) + wf . LN(pair)) v`` for the packed projections of ``LN(single)`` (q with its bias, k, v, g)
and an fp32 pair [1, L, L, d_pair], one autograd function around two opaque ops (forward, backward). The pieces (``apb_tf32.cu`` and
the TF32 wgmma core ``attn_tf32.cu`` of the token stack, built for the layout's heads):

  forward   one cuBLAS GEMM over [Wq | Wk | Wv | Wg]   ->  ``apb_prep`` (q scaled to exp2 units and TF32-rounded, k, v, v^T)
            ``pb_fwd`` (LayerNorm + the head projection of the pair in ONE pass, the core's key order, the key mask folded in)
            ->  the core (``tf32_forward``)  ->  ``apb_gate``
  backward  ``apb_bwd_prep`` (dO, dg), exact-fp32 cuBLAS GEMMs for S and dP, ``apb_bwd_rows`` (softmax, D, dS), GEMMs for dq, dk, dv, ``apb_post``
            (the packed gradient of the projections), two TF32 cuBLAS GEMMs (dx, dW), ``pb_bwd_tc`` / ``pb_bwd_rows`` (dpair, pair weights).
            The attention gradients are exact fp32: the TF32 core rounding of dS / dP left dq, dk at 2.4x PyTorch's error

The to_out projection and the residual stay in the module (one cuBLAS addmm). 24-wide heads run as 32-wide ones (zero columns); the
operands are rounded to the TF32 grid like every fp32 matmul of the process (``supported`` is False when TF32 is off).

Contract: sm_90, fp32 single / pair, one batch row, L a multiple of 128, heads x width in ``LAYOUTS``, d_pair 128 (any listed layout) or
256 (8 heads), an optional key mask [1, L].
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

from . import LOG2E, _ext, _is_hopper, tf32_defs, tf32_forward

NEG = -1e30
#: (heads, head width) -> padded width of the attention core
LAYOUTS = {(8, 48): 48, (12, 32): 32, (16, 24): 32, (16, 32): 32, (24, 16): 32}
_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def _rows():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="augattn_sm90_apb_tf32",
        sources=[str(_dir / "apb_tf32.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), "--expt-relaxed-constexpr"],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def _tf32_on() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest")


def supported(x, pair, n_head: int, d_hidden: int, mask) -> bool:
    """The contract above (``x`` is LN(single) [1, L, d_single])."""
    if settings.current().force_portable_kernels or not _tf32_on():
        return False
    if x.dtype is not torch.float32 or pair.dtype is not torch.float32 or not x.is_cuda or x.dim() != 3 or pair.dim() != 4:
        return False
    B, L, _ = x.shape
    dp = pair.shape[-1]
    if B != 1 or L % 128 or tuple(pair.shape[:3]) != (1, L, L) or (n_head, d_hidden) not in LAYOUTS:
        return False
    if not (dp == 128 or (dp == 256 and n_head == 8)):
        return False
    if mask is not None and (mask.dim() != 2 or tuple(mask.shape) != (1, L) or mask.dtype is not torch.bool):
        return False
    return _is_hopper(x.device.index if x.device.index is not None else torch.cuda.current_device())


_BUILD_FAILED = False


def available(x, pair, n_head: int, d_hidden: int, mask) -> bool:
    """``supported()`` plus successful builds; never under fake tensors (dev derive)."""
    global _BUILD_FAILED
    if _BUILD_FAILED or not supported(x, pair, n_head, d_hidden, mask):
        return False
    from torch._subclasses.fake_tensor import FakeTensor

    if torch.compiler.is_compiling():
        return True
    if isinstance(x, FakeTensor) or isinstance(pair, FakeTensor):
        return False
    try:
        from . import tf32_fwd_defs

        _rows()
        dh = LAYOUTS[(n_head, d_hidden)]
        _ext("attn_tf32", tf32_fwd_defs(n_head, dh))
    except Exception as exc:  # noqa: BLE001
        import warnings

        _BUILD_FAILED = True
        warnings.warn(f"TF32 AttentionPairBias kernels unavailable, keeping the PyTorch leg: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _fwd_fake(x, pair, wq, bq, wk, wv, wg, lnw, wb, mask, eps, gb, nh, dr, train):
    """Shapes of the TF32 AttentionPairBias forward: og, O (padded), bias, pair stats and q + bias (training only), qkvg, packed weight; all like x's dtype."""
    L = x.shape[1]
    N, dm = nh * dr, nh * LAYOUTS[(nh, dr)]
    e = x.new_empty
    return (e((L, N)), e((L, dm)), e((nh, L, L)), e((L * L, 2) if train else (0,)), e((L, 4 * N)), e((L, N) if train else (0,)), e((4 * N, x.shape[2])))


@opaque(fake=_fwd_fake, name="augmented_attention_apb_tf32_fwd")
def _fwd(
    x: torch.Tensor, pair: torch.Tensor, wq: torch.Tensor, bq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wg: torch.Tensor,
    lnw: torch.Tensor, wb: torch.Tensor, mask: torch.Tensor | None, eps: float, gb: float, nh: int, dr: int, train: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(og [L, N], O, bias in the core's layout, pair stats, qkvg, q + bias, packed projection weight)."""
    rows = _rows()
    L, dp = x.shape[1], pair.shape[-1]
    dh = LAYOUTS[(nh, dr)]
    wpack = torch.cat([wq, wk, wv, wg], 0)
    qkvg = torch.mm(x.reshape(L, -1), wpack.t())
    qs, kr, qf, vt = rows.apb_prep(qkvg, bq, nh, dr, LOG2E / math.sqrt(dr), train)
    wf = (wb.float() * lnw.float() * LOG2E).contiguous()
    bias_p, pst = rows.pb_fwd(pair.view(L * L, dp), wf, None if mask is None else mask.reshape(-1), L, float(eps), NEG, train)
    o, _ = tf32_forward(qs, kr, vt, bias_p, None, 1, L, nh=nh, dh=dh)
    og = rows.apb_gate(o, qkvg, nh, dr, float(gb))
    return og, o, bias_p, pst, qkvg, qf, wpack


def _bwd_fake(dog, x, o, bias_p, pst, qkvg, qf, wpack, pair, lnw, wb, eps, gb, nh, dr):
    """Shapes of the TF32 AttentionPairBias backward: dx, dpair, dW of the packed projections, dbq (like x's dtype), dlnw and dwb (fp32)."""
    N = nh * dr
    return (x.new_empty(x.shape[1:]), pair.new_empty(pair.shape), x.new_empty((4 * N, x.shape[2])), x.new_empty((N,)),
            lnw.new_empty(lnw.shape, dtype=torch.float32), wb.new_empty(wb.shape, dtype=torch.float32))


def _heads(t, L, nh, dr):
    """[L, >= nh dr] column block (row stride kept) as [nh, L, dr]."""
    return t.as_strided((nh, L, dr), (dr, t.stride(0), 1))


@opaque(fake=_bwd_fake, name="augmented_attention_apb_tf32_bwd")
def _bwd(
    dog: torch.Tensor, x: torch.Tensor, o: torch.Tensor, bias_p: torch.Tensor, pst: torch.Tensor, qkvg: torch.Tensor, qf: torch.Tensor,
    wpack: torch.Tensor, pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor, eps: float, gb: float, nh: int, dr: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx [L, d_single], dpair, dW [4N, d_single] of the packed projections, dbq, dlnw, dwb). The attention gradients run in exact fp32."""
    rows = _rows()
    L, dp = x.shape[1], pair.shape[-1]
    N = nh * dr
    xs = x.reshape(L, -1)
    dO, dqkvg = rows.apb_bwd_prep(dog.contiguous(), o, qkvg, nh, dr, float(gb))
    qh, kh, vh, oh = _heads(qf, L, nh, dr), _heads(qkvg[:, N:], L, nh, dr), _heads(qkvg[:, 2 * N :], L, nh, dr), _heads(dO, L, nh, dr)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        s = torch.bmm(qh, kh.transpose(1, 2))
        dpm = torch.bmm(oh, vh.transpose(1, 2))
        p, ds = rows.apb_bwd_rows(s, dpm, bias_p, LOG2E / math.sqrt(dr))
        dq = torch.bmm(ds, kh)
        dk = torch.bmm(ds.transpose(1, 2), qh)
        dv = torch.bmm(p.transpose(1, 2), oh)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    rows.apb_post(dq, dk, dv, dqkvg, nh, dr)
    dx = torch.mm(dqkvg, wpack)
    dw = torch.mm(dqkvg.t(), xs)
    dbq = dqkvg[:, :N].sum(0)
    wfn = (wb.float() * lnw.float()).contiguous()
    if dp == 128:
        dz, accp = rows.pb_bwd_tc(ds, pair.view(L * L, dp), pst, wfn)
    else:
        dz, accp = rows.pb_bwd_rows(ds, pair.view(L * L, dp), wfn, float(eps))
    dwf = accp.sum(0).view(nh, dp)
    return dx, dz.view(pair.shape), dw, dbq, (dwf * wb.float()).sum(0), dwf * lnw.float()


class _Apb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pair, wq, bq, wk, wv, wg, lnw, wb, mask, eps, gb, nh, dr):  # noqa: D102
        train = any(ctx.needs_input_grad)
        og, o, bias_p, pst, qkvg, qf, wpack = _fwd(x, pair, wq, bq, wk, wv, wg, lnw, wb, mask, eps, gb, nh, dr, train)
        if train:
            ctx.save_for_backward(x, o, bias_p, pst, qkvg, qf, wpack, pair, lnw, wb)
            ctx.cfg = (eps, gb, nh, dr)
        return og

    @staticmethod
    def backward(ctx, dog):  # noqa: D102
        x, o, bias_p, pst, qkvg, qf, wpack, pair, lnw, wb = ctx.saved_tensors
        eps, gb, nh, dr = ctx.cfg
        dx, dz, dw, dbq, dlnw, dwb = _bwd(dog, x, o, bias_p, pst, qkvg, qf, wpack, pair, lnw, wb, eps, gb, nh, dr)
        N = nh * dr
        return (dx.view(x.shape), dz, dw[:N], dbq, dw[N : 2 * N], dw[2 * N : 3 * N], dw[3 * N :], dlnw, dwb, None, None, None, None, None)


def attention_pair_tf32(x, pair, wq, bq, wk, wv, wg, lnw, wb, mask=None, eps: float = 1e-5, gate_bias: float = 0.0, n_head: int = 8, d_hidden: int = 48):
    """The gated attention output ``og`` [L, n_head * d_hidden] for x = LN(single) [1, L, d_single]. Call ``available()`` first."""
    return _Apb.apply(x, pair, wq, bq, wk, wv, wg, lnw, wb, mask, eps, gate_bias, n_head, d_hidden)


__all__ = ["LAYOUTS", "attention_pair_tf32", "available", "supported"]
