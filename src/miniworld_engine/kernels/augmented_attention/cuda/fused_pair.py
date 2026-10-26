"""AugmentedAttention's H100 core with the pair bias folded in: ``softmax(q k^T / sqrt(d) + wf . LN(pair)) v``.

One autograd function around hand-written sm_90a kernels: the pair -> bias pass (LayerNorm folded into the head projection: one
read of the pair tensor, the bias comes out head-major in log2 units, the key mask as a large negative), the wgmma attention
cores of this family (``attn_fwd.cu`` / ``attn_dqb.cu`` / ``attn_dkv.cu``, O and the gradients written as bf16) and, backward,
the pair pass again (dpair and dWf in one read of pair + dbias). The unfused route (``augmented_attention_bf16_sm90``) runs the
pair LayerNorm, the head projection, a permute copy and a scale pass in PyTorch / Triton: four passes over the L x L x d_pair
pair tensor against one here.

Two layouts (``LAYOUTS``): the token stack (16 heads x 48, d_pair 128: ``apb/apb_rows.cu`` ``pair_bias``, tensor-core
projection) and the dense atom stack (4 heads x 32, d_pair 16: ``atom_pair.cu``, one thread per pair row).

Contract (``supported``): sm_90, bf16 q / k / v / pair, one batch row, L a multiple of 128, a key mask shared by all samples (or
none), no extra pair bias. The caller owns every projection; this op takes the LayerNorm weight ``lnw`` [d_pair] and the head
projection ``wb`` [H, d_pair] of the pair bias as leaves (their gradients come back fp32).

Numerics: as the unfused core: bf16 operands, fp32 sums, P and dS bf16 for the tensor core; the bias is rounded to bf16 once, in
log2 units; for d_pair 128 ``wf = wb * lnw`` is rounded to bf16 first (the module under autocast rounds LN(pair) and wb to bf16
separately), for d_pair 16 it stays fp32.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import torch
import triton
import triton.language as tl

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

from . import (
    FWD_OP,
    LOG2E,
    _ext as _core_ext,
    _is_hopper,
    _scale_bf16_kernel,
    backward,
)

NEG = -1e30  # a masked key's bias in log2 units: finite, so a fully masked row softmaxes to uniform (the reference's finfo.min fill)
#: (heads, head width) -> d_pair of the stack the pair kernels of that layout serve
LAYOUTS = {(16, 48, 128), (4, 32, 16), (8, 48, 256)}  # (heads, head width, d_pair)


def _kind(nh, dp):
    """Which pair kernel serves a layout: the tensor-core ``pair_bias`` (16 heads, d_pair 128), the atom thread-per-row kernel (d_pair 16) or the generic warp-per-row one."""
    return "apb" if (nh, dp) == (16, 128) else "atom" if dp == 16 else "rows"


@functools.lru_cache(maxsize=None)
def _mma_ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="augattn_sm90_pair_mma",
        sources=[str(_dir / "pair_mma.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), "--expt-relaxed-constexpr"],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def _rows_ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="augattn_sm90_pair_rows",
        sources=[str(_dir / "pair_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), "--expt-relaxed-constexpr"],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )
_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def _atom_ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="augattn_sm90_atom_pair",
        sources=[str(_dir / "atom_pair.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), "--expt-relaxed-constexpr"],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


@triton.jit
def _wf_kernel(wb, lnw, out_l2, out_nat, n, scale, BLOCK: tl.constexpr, NAT: tl.constexpr, DPK: tl.constexpr, F32: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    w = tl.load(wb + offs, mask=m).to(tl.float32) * tl.load(lnw + offs % DPK, mask=m).to(tl.float32)
    if F32:
        tl.store(out_l2 + offs, w, mask=m)
    else:
        tl.store(out_l2 + offs, (w * scale).to(tl.bfloat16), mask=m)
        if NAT:
            tl.store(out_nat + offs, w.to(tl.bfloat16), mask=m)


def _wf(wb: torch.Tensor, lnw: torch.Tensor, need_nat: bool):
    """d_pair 128: (wf in log2 units, wf natural) bf16 [H, 128] = wb * lnw (* log2 e); the natural copy only when asked."""
    wb, lnw = wb.contiguous(), lnw.contiguous()
    n = wb.numel()
    l2 = torch.empty(wb.shape, device=wb.device, dtype=torch.bfloat16)
    nat = torch.empty_like(l2) if need_nat else l2
    _wf_kernel[(triton.cdiv(n, 1024),)](wb, lnw, l2, nat, n, LOG2E, BLOCK=1024, NAT=need_nat, DPK=wb.shape[1], F32=False)
    return l2, (nat if need_nat else None)


def _wf32(wb: torch.Tensor, lnw: torch.Tensor):
    """d_pair 16: wf = wb * lnw, fp32 [H, 16] (the kernel applies log2 e itself)."""
    wb, lnw = wb.contiguous(), lnw.contiguous()
    n = wb.numel()
    out = torch.empty(wb.shape, device=wb.device, dtype=torch.float32)
    _wf_kernel[(triton.cdiv(n, 1024),)](wb, lnw, out, out, n, 1.0, BLOCK=1024, NAT=False, DPK=wb.shape[1], F32=True)
    return out


@triton.jit
def _dd_kernel(do, o, dd, L, NH: tl.constexpr, DH: tl.constexpr, DPAD: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)  # token rows (a * L + l)
    hh = tl.arange(0, NH)
    dcol = tl.arange(0, DPAD)
    ptr = r[:, None, None] * (NH * DH) + (hh[:, None] * DH + dcol[None, :])[None, :, :]
    msk = (dcol[None, :] < DH)[None, :, :] & (r[:, None, None] >= 0)
    g = tl.load(do + ptr, mask=msk, other=0.0).to(tl.float32)
    x = tl.load(o + ptr, mask=msk, other=0.0).to(tl.float32)
    s = tl.sum(g * x, axis=2)
    tl.store(dd + ((r // L)[:, None] * NH + hh[None, :]) * L + (r % L)[:, None], s)


def _dd(do2, o2, A, L, nh, dh):
    """D = rowsum(dO * O) [A, H, L] fp32."""
    dd = torch.empty(A, nh, L, device=do2.device, dtype=torch.float32)
    _dd_kernel[((A * L) // 8,)](do2, o2, dd, L, NH=nh, DH=dh, DPAD=64, ROWS=8)
    return dd


def _as_bf16_rows(t: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    t = t.reshape(rows, width)
    if t.dtype is not torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t if t.is_contiguous() else t.contiguous()


def supported(q: torch.Tensor, pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor, mask: torch.Tensor | None) -> bool:
    """The contract above."""
    if q.dim() != 5 or q.dtype is not torch.bfloat16 or pair.dtype is not torch.bfloat16:
        return False
    A, B, L, h, d = q.shape
    dp = pair.shape[-1]
    if (h, d, dp) not in LAYOUTS or tuple(pair.shape) != (1, L, L, dp) or tuple(wb.shape) != (h, dp) or tuple(lnw.shape) != (dp,):
        return False
    if mask is not None and (mask.dim() != 1 or mask.numel() != L or mask.dtype is not torch.bool):
        return False
    # the attention core's own gate: sm_90, B == 1, L % 128, the setting
    return bool(
        settings.current().augmented_attention_bf16_sm90
        and q.is_cuda
        and pair.is_cuda
        and B == 1
        and L % 128 == 0
        and _is_hopper(q.device.index if q.device.index is not None else torch.cuda.current_device())
    )


def _fwd_fake(q, k, v, pair, lnw, wb, mask, eps):
    A, _, L, nh, dh = q.shape
    kind = _kind(nh, pair.shape[-1])
    return (
        q.new_empty((A * L, nh * dh), dtype=torch.bfloat16),
        q.new_empty((A, nh, L), dtype=torch.float32),
        q.new_empty((nh, L, L), dtype=torch.bfloat16),
        q.new_empty((L * L, 2) if kind == "rows" else (0, 2), dtype=torch.float32),
    )


@opaque(fake=_fwd_fake, name="augmented_attention_pair_fused_fwd")
def _fused_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor, mask: torch.Tensor | None, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(O bf16 [A L, W], LSE [A, H, L], bias [H, L, L] bf16 in log2 units, pair-row statistics or an empty [0, 2]): the pair pass and the core."""
    A, _, L, nh, dh = q.shape
    W, dp = nh * dh, pair.shape[-1]
    pair2 = pair.view(L * L, dp)
    kind = _kind(nh, dp)
    pst = q.new_empty((0, 2), dtype=torch.float32)
    if kind == "apb":
        from .apb import pair_bias

        wf_l2, _ = _wf(wb, lnw, False)
        bias = pair_bias(pair2, wf_l2, mask, L, eps, NEG)  # [H, L, L] bf16, log2 units
    elif kind == "atom":
        bias = _atom_ext().atom_pair_fwd(pair2, _wf32(wb, lnw), mask, L, eps, LOG2E, NEG)
    elif dp == 256:  # tensor-core forward (bf16 wf in log2 units like the APB kernel), CUDA-core backward
        wf_l2, _ = _wf(wb, lnw, False)
        bias, pst = _mma_ext().pair_mma_fwd(pair2, wf_l2, mask, L, eps, NEG)
    else:
        bias, pst = _rows_ext().pair_rows_fwd(pair2, _wf32(wb, lnw), mask, L, eps, LOG2E, NEG)
    qs, kb, vb = _prep(q, k, v, A, L, W, dh)
    o, lse = FWD_OP(qs, kb, vb, bias, None, True)
    return o, lse, bias, pst


def _prep(q, k, v, A, L, W, dh):
    """q scaled into the core's log2 units and the bf16 row views of q, k, v."""
    qs = torch.empty(A * L, W, device=q.device, dtype=torch.bfloat16)
    qc = q.contiguous()
    n = qc.numel()
    _scale_bf16_kernel[(triton.cdiv(n, 4096),)](qc, qs, n, LOG2E / math.sqrt(dh), BLOCK=4096)
    return qs, _as_bf16_rows(k, A * L, W), _as_bf16_rows(v, A * L, W)


def _bwd_fake(do, q, k, v, pair, lnw, wb, o, lse, bias, pst, eps):
    return (
        q.new_empty(q.shape, dtype=torch.bfloat16),
        q.new_empty(q.shape, dtype=torch.bfloat16),
        q.new_empty(q.shape, dtype=torch.bfloat16),
        pair.new_empty(pair.shape),
        lnw.new_empty(lnw.shape, dtype=torch.float32),
        wb.new_empty(wb.shape, dtype=torch.float32),
    )


@opaque(fake=_bwd_fake, name="augmented_attention_pair_fused_bwd")
def _fused_bwd(
    do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor,
    o: torch.Tensor, lse: torch.Tensor, bias: torch.Tensor, pst: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dq, dk, dv bf16, dpair, dlnw, dwb fp32) of ``_fused_fwd``."""
    A, _, L, nh, dh = q.shape
    dp = pair.shape[-1]
    qs, kb, vb = _prep(q, k, v, A, L, nh * dh, dh)
    do2 = _as_bf16_rows(do, A * L, nh * dh)
    dd = _dd(do2, o, A, L, nh, dh)
    dq, dk, dv, db = backward(qs, kb, vb, do2, bias, None, lse, dd, A, L, nh, dh, out_bf16=True)
    if not db.is_contiguous():  # A not a multiple of 3: the dK / dV pass leaves it transposed
        db = db.contiguous()
    kind = _kind(nh, dp)
    if kind == "apb":
        from .apb import acc_n, ext, pair_bias_bwd

        _, wf_nat = _wf(wb, lnw, True)
        acc = torch.empty(ext().pair_bwd_blocks(pair, nh, L), acc_n(nh, 384), device=pair.device)
        dz, _, _ = pair_bias_bwd(pair.view(L * L, dp), db, wf_nat, L, eps, acc, 384)
        o_off = 2 * 384 + 512  # width(16, 384) = 16 x 32: the accumulator's dWf columns
        dwf = acc[:, o_off : o_off + nh * dp].sum(0).view(nh, dp)
    elif kind == "atom":
        dz, accp = _atom_ext().atom_pair_bwd(pair.view(L * L, dp), db, _wf32(wb, lnw), L, eps)
        dwf = accp.sum(0).view(nh, dp)
    else:
        dz, accp = _rows_ext().pair_rows_bwd(pair.view(L * L, dp), db, _wf32(wb, lnw), pst, L, eps)
        dwf = accp.sum(0).view(nh, dp)
    shp = (A, 1, L, nh, dh)
    return dq.view(shp), dk.view(shp), dv.view(shp), dz.view(pair.shape), (dwf * wb.float()).sum(0), dwf * lnw.float()


class _AttnFusedPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pair, lnw, wb, mask, eps):  # noqa: D102
        A, _, L, nh, dh = q.shape
        o, lse, bias, pst = _fused_fwd(q, k, v, pair, lnw, wb, mask, eps)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(q, k, v, pair, lnw, wb, o, lse, bias, pst)
            ctx.eps = eps
        return o.view(A, 1, L, nh, dh)

    @staticmethod
    def backward(ctx, do):  # noqa: D102
        q, k, v, pair, lnw, wb, o, lse, bias, pst = ctx.saved_tensors
        dq, dk, dv, dz, dlnw, dwb = _fused_bwd(do.contiguous(), q, k, v, pair, lnw, wb, o, lse, bias, pst, ctx.eps)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dz, dlnw, dwb, None, None


def attention_pair_fused(q, k, v, pair, lnw, wb, mask=None, eps: float = 1e-5):
    """``softmax(q k^T / sqrt(d) + (wb * lnw) . LN(pair)) v`` for q, k, v [A, 1, L, h, d] bf16, pair [1, L, L, d_pair] bf16, lnw [d_pair]
    and wb [h, d_pair] (fp32 leaves ok; layouts in ``LAYOUTS``), mask a key mask [L] bool (True = attend) or None; returns bf16
    [A, 1, L, h, d]. Call ``available()`` first."""
    return _AttnFusedPair.apply(q, k, v, pair, lnw, wb, mask, eps)


_BUILD_FAILED = False


def available(q, pair, lnw, wb, mask) -> bool:
    """``supported()`` plus successful builds (a failure warns once and keeps the unfused route); never on
    fake tensors outside a ``torch.compile`` trace (``dev derive``)."""
    global _BUILD_FAILED
    if _BUILD_FAILED or not supported(q, pair, lnw, wb, mask):
        return False
    from torch._subclasses.fake_tensor import FakeTensor

    if torch.compiler.is_compiling():
        return True  # the route is two opaque ops in the trace; the extensions build when the op first runs
    if isinstance(q, FakeTensor) or isinstance(pair, FakeTensor):
        return False  # dev derive: the portable kernels carry the autotune keys
    try:
        from . import fwd_defs

        _core_ext("attn_fwd", fwd_defs(q.shape[3], q.shape[4]))
        kd = _kind(q.shape[3], pair.shape[-1])
        if kd == "apb":
            from .apb import ext

            ext()
        elif kd == "atom":
            _atom_ext()
        else:
            _rows_ext()
            _mma_ext()
    except Exception as exc:  # noqa: BLE001
        import warnings

        _BUILD_FAILED = True
        warnings.warn(f"pair-fused sm90 attention unavailable, keeping the unfused core: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


__all__ = ["attention_pair_fused", "available", "supported"]
