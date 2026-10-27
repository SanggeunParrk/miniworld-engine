"""The fp32 token stack's pair-biased attention on H100: TF32 wgmma cores (``attn_tf32.cu``) with the pair bias folded in.

``softmax(q k^T / sqrt(48) + wf . LN(pair)) v`` for fp32 q / k / v [A, 1, L, 16, 48] and an fp32 pair [1, L, L, 128]: the bias
comes out of ONE pass over the pair (``pair_bias_tok_cuda``: LayerNorm + the 16 head projections + the TF32 core's key
permutation + the key mask), the core is the TF32 forward / dq-dbias / dk-dv trio of the token DiT's fp32 block, and the
backward of the pair side is PyTorch (LayerNorm backward, two TF32 GEMMs). Operands are rounded to TF32 inside the tensor core
like every fp32 matmul of the process (``supported`` is False when TF32 is switched off: the call then keeps full-fp32 math).

Contract: sm_90, fp32 q / k / v / pair, 16 x 48, d_pair 128, one batch row, L a multiple of 128, a key mask shared by all
samples (or none).
"""

from __future__ import annotations

import math

import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

from . import LOG2E, _is_hopper, tf32_backward, tf32_forward

NEG = -1e30
H, D, DP = 16, 48, 128


def _tf32_on() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest")


def supported(q, pair, lnw, wb, mask) -> bool:
    """The contract above."""
    if not settings.current().augattn_tf32_sm90 or not _tf32_on():
        return False
    if q.dim() != 5 or q.dtype is not torch.float32 or pair.dtype is not torch.float32 or not q.is_cuda:
        return False
    A, B, L, h, d = q.shape
    if (h, d) != (H, D) or B != 1 or L % 128 or tuple(pair.shape) != (1, L, L, DP) or tuple(wb.shape) != (H, DP) or tuple(lnw.shape) != (DP,):
        return False
    if mask is not None and (mask.dim() != 1 or mask.numel() != L or mask.dtype is not torch.bool):
        return False
    return _is_hopper(q.device.index if q.device.index is not None else torch.cuda.current_device())


_BUILD_FAILED = False


def available(q, pair, lnw, wb, mask) -> bool:
    """``supported()`` plus successful builds; never under torch.compile tracing or fake tensors."""
    global _BUILD_FAILED
    if _BUILD_FAILED or not supported(q, pair, lnw, wb, mask):
        return False
    from torch._subclasses.fake_tensor import FakeTensor

    if torch.compiler.is_compiling():
        return True  # two opaque ops in the trace; the extensions build when the op first runs
    if isinstance(q, FakeTensor) or isinstance(pair, FakeTensor):
        return False  # dev derive
    try:
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32

        from . import _ext, tf32_fwd_defs

        rows32()
        from .fused_pair import _rows_ext

        _rows_ext()
        _ext("attn_tf32", tf32_fwd_defs())
    except Exception as exc:  # noqa: BLE001
        import warnings

        _BUILD_FAILED = True
        warnings.warn(f"TF32 pair-fused attention unavailable, keeping the PyTorch leg: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


_PERMS: dict = {}


def _perm(L, device):
    """key_perm and its inverse, built once per (L, device) (outside CUDA-graph capture the first call builds them)."""
    key = (L, str(device))
    if key not in _PERMS:
        from miniworld_engine.kernels.conditioned_transition.cuda import key_perm

        p = key_perm(L, device)
        _PERMS[key] = (p, torch.argsort(p))
    return _PERMS[key]

def _fwd_fake(q, k, v, pair, lnw, wb, mask, eps):
    """Shapes of the TF32 pair-fused attention forward (outputs and saved tensors; all like q's dtype)."""
    A, _, L, _, _ = q.shape
    return (
        q.new_empty((A * L, H * D)),
        q.new_empty((A, H, L)),
        q.new_empty((H, L, L)),
        q.new_empty((A * L, H * D)),
        q.new_empty((A * L, H * D)),
        q.new_empty((A * L, H * D)),
    )


@opaque(fake=_fwd_fake, name="augmented_attention_tf32_fwd")
def _fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor, mask: torch.Tensor | None, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(O, LSE, key-permuted bias [H, L, L] in log2 units, q scaled, k, v rounded to the TF32 grid [A L, 768]): the pair pass and the core."""
    from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32

    from .fused_pair import _rows_ext

    A, _, L, _, _ = q.shape
    qs, kr, vr, vt = _rows_ext().prep_fwd32(
        q.reshape(A * L, H * D).contiguous(), k.reshape(A * L, H * D).contiguous(), v.reshape(A * L, H * D).contiguous(), LOG2E / math.sqrt(D)
    )
    wf = (wb.float() * lnw.float() * LOG2E).contiguous()
    bias_p = torch.empty(H, L, L, device=q.device, dtype=torch.float32)
    rows32().pair_bias_tok_cuda(pair.view(L * L, DP), wf, bias_p, mask, int(L), float(eps), True, NEG)
    o, lse = tf32_forward(qs, kr, vt, bias_p, None, A, L)
    return o, lse, bias_p, qs, kr, vr


def _bwd_fake(do, o, lse, bias_p, qs, kr, vr, pair, lnw, wb, eps):
    """Shapes of the TF32 pair-fused attention backward: (dq, dk, dv, dpair, dlnw fp32, dwb fp32)."""
    return (
        qs.new_empty(qs.shape),
        qs.new_empty(qs.shape),
        qs.new_empty(qs.shape),
        pair.new_empty(pair.shape),
        lnw.new_empty(lnw.shape, dtype=torch.float32),
        wb.new_empty(wb.shape, dtype=torch.float32),
    )


@opaque(fake=_bwd_fake, name="augmented_attention_tf32_bwd")
def _bwd(
    do: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, bias_p: torch.Tensor, qs: torch.Tensor, kr: torch.Tensor, vr: torch.Tensor,
    pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dq, dk, dv [A L, 768], dpair, dlnw, dwb) of ``_fwd``."""
    from .fused_pair import _rows_ext

    A = o.shape[0] // pair.shape[1]
    L = pair.shape[1]
    e = _rows_ext()
    dd, dd_p, lse_p, dobr = e.dd_perm(do.contiguous().float(), o, lse, A, L)  # D = rowsum(dO O), its / the LSE's query-permuted copies, dO rounded
    dq, dk, dv, db_p = tf32_backward(
        qs, kr, vr, dobr, e.transpose32(qs), e.transpose32(kr), e.transpose32(dobr), bias_p, e.bias_transposed(bias_p, L),
        lse, dd, lse_p, dd_p, None, None, A, L,
    )
    _, inv = _perm(L, qs.device)
    db = db_p[..., inv].contiguous()  # natural key order, natural units: [H, L, L]
    wf = (wb.float() * lnw.float()).contiguous()
    dz, accp = e.pair_rows_bwd32(pair.view(L * L, DP), db, wf, L, eps)
    dwf = accp.sum(0).view(H, DP)
    return dq, dk, dv, dz.view(pair.shape), (dwf * wb.float()).sum(0), dwf * lnw.float()


class _AttnTf32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pair, lnw, wb, mask, eps):  # noqa: D102
        A, _, L, _, _ = q.shape
        o, lse, bias_p, qs, kr, vr = _fwd(q, k, v, pair, lnw, wb, mask, eps)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(o, lse, bias_p, qs, kr, vr, pair, lnw, wb)
            ctx.eps, ctx.shape = eps, q.shape
        return o.view(A, 1, L, H, D)

    @staticmethod
    def backward(ctx, do):  # noqa: D102
        o, lse, bias_p, qs, kr, vr, pair, lnw, wb = ctx.saved_tensors
        dq, dk, dv, dz, dlnw, dwb = _bwd(do.reshape(o.shape).contiguous(), o, lse, bias_p, qs, kr, vr, pair, lnw, wb, ctx.eps)
        shp = ctx.shape
        return dq.view(shp), dk.view(shp), dv.view(shp), dz, dlnw, dwb, None, None


def attention_pair_tf32(q, k, v, pair, lnw, wb, mask=None, eps: float = 1e-5):
    """See the module docstring; returns fp32 [A, 1, L, 16, 48]. Call ``available()`` first. A ``torch.compile`` trace sees two opaque ops."""
    return _AttnTf32.apply(q, k, v, pair, lnw, wb, mask, eps)


__all__ = ["attention_pair_tf32", "available", "supported"]
