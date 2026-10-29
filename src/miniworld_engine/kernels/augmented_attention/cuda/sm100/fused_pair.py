"""AugmentedAttention's B200 (sm_100a) core with the pair bias folded in: ``softmax(q k^T / sqrt(d) + wf . LN(pair)) v``.

The B200 counterpart of the H100 ``cuda/fused_pair.py`` (bf16) and ``cuda/fused_pair_tf32.py`` (fp32): one autograd function around
hand-written kernels, each pass an opaque op (``torch.compile`` keeps them as single nodes):

  forward    the pair -> bias pass: LayerNorm folded into the head projection, one read of the pair tensor, the bias head-major
             [H, L, L] in natural units, the key mask folded in as -1e30 (bf16 pair: ``apb_rows.cu`` ``pair_bias_fwd``, mma.sync on a
             TMA ring; fp32 pair: ``bias_only_dit_f32_rows.cu`` ``pair_bias_tc`` / ``pair_bias_tok``, TF32 mma.sync), then this family's
             tcgen05 attention forward (``attn_fwd2`` bf16 / ``attn_fwd_tf32`` TF32).
  backward   ``prep_do_*`` (D = rowsum(dO O), dO as bf16 for the bf16 core), the tcgen05 dK / dV and dQ + dbias passes built with
             ``-DDET=1`` (each CTA's two MMA issuers accumulate in a fixed order) and ``-DDQPART=1`` (one dQ partial per key chunk, stored,
             summed in chunk order by ``finish_*``), then the pair side: dpair and the dWf block partials in one more read of the pair
             (``pair_bias_bwd`` / ``pair_bias_bwd_tc``), dWf summed over the partial rows in order. Every gradient is bitwise the same
             run to run.

Precision follows q: bf16 q / k / v -> the bf16 core (bf16 operands, fp32 sums, O fp32); fp32 q / k / v -> the TF32 core (kind::tf32
MMAs, fp32 softmax; only when the process allows TF32 matmuls, as the H100 TF32 core). The pair side follows the pair's dtype: a bf16
pair takes the bf16 pair pass (wf = wb * lnw rounded to bf16), an fp32 pair the TF32 one (an fp32 pair under a bf16 core -- autocast over
fp32 activations -- has its fp32 bias rounded to bf16 once).

Contract (``supported``): sm_100, q / k / v [A, 1, L, H, d] at a token DiT head layout (``TDIT_HEADS``: 16 x 48, 24 x 32, 12 x 64,
16 x 64), any A (an odd A runs one zero sample more: the cores pair the samples), L a multiple of 128, pair [1, L, L, d_pair] with d_pair
128 (bf16 pair: also 256 at 16 heads), a key mask [L] shared by the samples or none. ``settings.augmented_attention_bf16_sm100`` turns
the B200 cores off. A fully masked query row softmaxes to uniform (as the PyTorch statements' finfo.min fill); its dbias at the masked
keys is not zeroed (as on H100).
"""

from __future__ import annotations

import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

from . import TDIT_HEADS, _grid, _index, _is_blackwell, _rs, _sm100_kernel, _tdit_defs, _tm, bias_transpose, forward, forward_tf32

NEG = -1e30  # a masked key's bias (natural units): finite, so a fully masked row softmaxes to uniform
BF = torch.bfloat16
F32 = torch.float32


def _tf32_on() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest")


def supported(
    q: torch.Tensor, pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor, mask: torch.Tensor | None, dtype: torch.dtype | None = None
) -> bool:
    """The contract above; ``dtype``: the core's precision when q is cast to it first (default q's)."""
    if not settings.current().augmented_attention_bf16_sm100:
        return False
    if q.dim() != 5 or not q.is_cuda or not pair.is_cuda:
        return False
    A, B, L, h, d = q.shape
    dp = pair.shape[-1]
    if (h, d) not in TDIT_HEADS or B != 1 or A < 1 or L == 0 or L % 128:
        return False
    if tuple(pair.shape) != (1, L, L, dp) or tuple(wb.shape) != (h, dp) or tuple(lnw.shape) != (dp,):
        return False
    dt = dtype or q.dtype
    if dt is BF:
        if pair.dtype is BF:
            ok = dp == 128 or (dp == 256 and h == 16)
        else:
            ok = pair.dtype is F32 and dp == 128
    elif dt is F32:
        ok = pair.dtype is F32 and dp == 128 and _tf32_on()
    else:
        ok = False
    if not ok:
        return False
    if mask is not None and (mask.dim() != 1 or mask.numel() != L or mask.dtype is not torch.bool):
        return False
    return _is_blackwell(_index(q))


def _core_defs(H: int, DH: int) -> tuple[str, ...]:
    return (*_tdit_defs(H, DH), "DET=1")


def _build(q: torch.Tensor, pair: torch.Tensor, dtype: torch.dtype) -> None:
    """Load every kernel the call will launch (cubins, the pair extension)."""
    dev, (H, DH) = _index(q), tuple(q.shape[-2:])
    td, cd = _tdit_defs(H, DH), _core_defs(H, DH)
    if dtype is BF:
        _sm100_kernel("attn_fwd2", "augattn_fwd2_sm100", dev, defs=td)
        _sm100_kernel("attn_dkv", "augattn_dkv_sm100", dev, defs=cd)
        _sm100_kernel("attn_dqb", "augattn_dqb_sm100", dev, defs=(*cd, "DQPART=1"))
    else:
        _sm100_kernel("attn_fwd_tf32", "augattn_fwd_tf32_sm100", dev, defs=td)
        _sm100_kernel("attn_dkv_tf32", "augattn_dkv_tf32_sm100", dev, defs=cd)
        _sm100_kernel("attn_dqb_tf32", "augattn_dqb_tf32_sm100", dev, defs=(*cd, "DQPART=1"))
    for name in ("prep_do_f32r", "prep_do_b16", "prep_do_f32", "finish_f32", "finish_bf16", "round3_tf32", "bias_transpose_bf16"):
        _sm100_kernel("glue", name, dev, smem=0)
    if pair.dtype is BF:
        from ..apb import ext

        ext()
    else:
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32

        rows32()


def available(q, pair, lnw, wb, mask, dtype: torch.dtype | None = None) -> bool:
    """``supported()`` plus successful builds. A build or load failure raises (B200 has no silent fallback). Under a
    ``torch.compile`` trace the route is taken (its passes are opaque ops; the kernels load when they first run); fake tensors outside
    a trace keep the other route."""
    if not supported(q, pair, lnw, wb, mask, dtype):
        return False
    from torch._subclasses.fake_tensor import FakeTensor

    if torch.compiler.is_compiling():
        return True
    if isinstance(q, FakeTensor) or isinstance(pair, FakeTensor):
        return False
    try:
        _build(q, pair, dtype or q.dtype)
    except Exception as exc:
        raise RuntimeError(f"sm100 pair-fused augmented attention kernels failed to build or load on B200: {exc!r}") from exc
    return True


# --------------------------------------------------------------------------------------------------- helpers
def _even(A: int) -> int:
    return A + (A & 1)


def _rows(t: torch.Tensor, A: int, Ap: int, L: int, W: int) -> torch.Tensor:
    """[A, 1, L, H, d] (or [A L, W]) -> [Ap L, W] rows, contiguous; the pad sample (odd A) zero."""
    t2 = t.reshape(A * L, W)
    if Ap == A:
        return t2 if t2.stride(1) == 1 and t2.stride(0) == W else t2.contiguous()
    out = t2.new_empty(Ap * L, W)
    out[: A * L].copy_(t2)
    out[A * L :].zero_()
    return out


def _wf(wb: torch.Tensor, lnw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Wf = wb diag(lnw) [H, d_pair] in ``dtype`` (the LayerNorm weight folded into the head projection; natural units)."""
    return (wb.float() * lnw.float()).to(dtype).contiguous()


def _pair_bias(pair, lnw, wb, mask, eps, core_bf16: bool, train: bool):
    """(bias [H, L, L] in the core's dtype, natural units, -1e30 on masked keys; the pair rows' (mean, rstd) [L L, 2] when the fp32
    pair pass's backward will need them, else an empty [0, 2])."""
    L, dp = pair.shape[1], pair.shape[-1]
    H = wb.shape[0]
    p2 = pair.reshape(L * L, dp)
    m = None if mask is None else mask.reshape(L).contiguous()
    pst = pair.new_empty((0, 2), dtype=F32)
    if pair.dtype is BF:
        from ..apb import pair_bias

        bias = pair_bias(p2, _wf(wb, lnw, BF), m, L, eps, NEG)  # [H, L, L] bf16 (a bf16 pair only meets the bf16 core)
        return bias, pst
    from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32

    wf = _wf(wb, lnw, F32)
    bias = torch.empty(H, L, L, device=pair.device, dtype=F32)
    if train:
        pst = torch.empty(L * L, 2, device=pair.device, dtype=F32)
        rows32().pair_bias_tc_cuda(p2, wf, bias, pst, float(eps), 1)
        if m is not None:
            bias.masked_fill_(~m.view(1, 1, L), NEG)
    else:
        rows32().pair_bias_tok_cuda(p2, wf, bias, m, int(L), float(eps), False, NEG)
    return (bias.to(BF) if core_bf16 else bias), pst


def _f32maps(t, dims, rows):
    """K-major fp32 row maps of a head row: columns 0-31 (128-B swizzle) and the 16-column rest (64-B swizzle)."""
    return (_tm(t, dims, _rs(t), [32, rows], swizzle=128, dtype="f32"), _tm(t, dims, _rs(t), [16, rows], swizzle=64, dtype="f32"))


def _bwd_bf16(q2, k2, v2, dob, bias, LSE, Dd, A, L, H, DH):
    """The bf16 core's backward, deterministic: (dQ partials [L / 128, A L, W], dK, dV [A L, W] fp32, dbias [H, L, L] fp32)."""
    dev = _index(q2)
    W, M, nch = H * DH, A * L, L // 128
    f32 = dict(dtype="f32")
    bias_t = bias_transpose(bias)
    DQ = torch.empty(nch, M, W, device=q2.device, dtype=F32)
    DK, DV = (torch.empty(M, W, device=q2.device, dtype=F32) for _ in range(2))
    DB = torch.empty(H, L, L, device=q2.device, dtype=F32)
    dims = [W, M]
    dkv_maps = (
        _tm(q2, dims, _rs(q2), [DH, 64]),
        _tm(k2, dims, _rs(k2), [DH, 128]),
        _tm(v2, dims, _rs(v2), [DH, 128]),
        _tm(dob, dims, _rs(dob), [DH, 64]),
        _tm(bias_t, [L, H * L], L * 2, [64, 128]),
        _tm(DK, dims, W * 4, [32, 128], swizzle=128, **f32),
        _tm(DK, dims, W * 4, [16, 128], swizzle=64, **f32),
        _tm(DV, dims, W * 4, [32, 128], swizzle=128, **f32),
        _tm(DV, dims, W * 4, [16, 128], swizzle=64, **f32),
    )
    dq_dims = [W, nch * M]
    dqb_maps = (
        *(_tm(t, dims, _rs(t), [DH, 128]) for t in (q2, k2, v2, dob)),
        _tm(bias, [L, H * L], L * 2, [64, 128]),
        _tm(DQ, dq_dims, W * 4, [32, 128], swizzle=128, **f32),
        _tm(DQ, dq_dims, W * 4, [16, 128], swizzle=64, **f32),
    )
    cd = _core_defs(H, DH)
    _sm100_kernel("attn_dkv", "augattn_dkv_sm100", dev, defs=cd)(
        _grid(A * H * (L // 128), dev), (384, 1, 1), *dkv_maps, LSE, Dd, DK, DV, None, int(L), int(A)
    )  # no dQ zero fill: the dQ partials are stored
    _sm100_kernel("attn_dqb", "augattn_dqb_sm100", dev, defs=(*cd, "DQPART=1"))(
        _grid(H * (L // 128) * nch, dev), (384, 1, 1), *dqb_maps, LSE, Dd, DQ, DB, int(L), int(A)
    )
    return DQ, DK, DV, DB


def _bwd_tf32(q, k, v, do, bias, LSE, Dd, A, L, H, DH):
    """The TF32 core's backward, deterministic: (dQ partials [L / 64, A L, W], dK, dV [A L, W], dbias [H, L, L]), all fp32."""
    dev = _index(q)
    W, M, nch = H * DH, A * L, L // 64
    f32 = dict(dtype="f32")
    DQ = torch.empty(nch, M, W, device=q.device, dtype=F32)
    DK, DV = (torch.empty(M, W, device=q.device, dtype=F32) for _ in range(2))
    DB = torch.empty(H, L, L, device=q.device, dtype=F32)
    dims = [W, M]
    a = lambda t, rows: _tm(t, dims, _rs(t), [32, rows], swizzle=128, **f32)  # noqa: E731  K-major, columns 0-31
    b = lambda t, rows: _tm(t, dims, _rs(t), [16, rows], swizzle=64, **f32)  # noqa: E731  K-major, the 16-column rest
    m = lambda t, rows: _tm(t, dims, _rs(t), [32, rows], swizzle="128a32", **f32)  # noqa: E731  MN-major, 32-column atoms
    dkv_maps = (
        a(q, 32), b(q, 32), m(q, 32), a(do, 32), b(do, 32), m(do, 32),
        a(k, 128), b(k, 128), a(v, 128), b(v, 128),
        _tm(bias, [L, H * L], L * 4, [32, 32], swizzle=0, **f32),
        a(DK, 128), b(DK, 128), a(DV, 128), b(DV, 128),
    )
    dqb_maps = (
        a(q, 128), b(q, 128), a(do, 128), b(do, 128), a(v, 64), b(v, 64), a(k, 64), b(k, 64), m(k, 64),
        _tm(bias, [L, H * L], L * 4, [32, 128], swizzle=128, **f32),
    )
    cd = _core_defs(H, DH)
    _sm100_kernel("attn_dkv_tf32", "augattn_dkv_tf32_sm100", dev, defs=cd)(
        _grid(A * H * (L // 128), dev), (384, 1, 1), *dkv_maps, LSE, Dd, None, int(L), int(A), int(W)
    )  # no dQ zero fill
    _sm100_kernel("attn_dqb_tf32", "augattn_dqb_tf32_sm100", dev, defs=(*cd, "DQPART=1"))(
        _grid(H * (L // 128) * nch, dev), (384, 1, 1), *dqb_maps, LSE, Dd, DQ, DB, int(L), int(A), 1, int(W)
    )  # one sample group: dbias written once per (head, query tile, key chunk), no reductions
    return DQ, DK, DV, DB


def _prep_do(do2, O, dob, dd, M, L, H, DH):
    """D = rowsum_head(dO O) into ``dd`` and (``dob`` given) dO as bf16, over the first M rows."""
    name = "prep_do_b16" if do2.dtype is BF else "prep_do_f32"
    _sm100_kernel("glue", name, _index(do2), smem=0)(((M * H + 255) // 256, 1, 1), (256, 1, 1), do2, O, dob, dd, int(M), int(L), int(H), int(DH))


def _finish_grid(n4: int, dev: int, y: int = 1):
    from miniworld_engine.kernels._device import num_sms

    return (min((n4 + 255) // 256, 8 * num_sms(dev)), y, 1)


# --------------------------------------------------------------------------------------------------- the two passes
def _round3(q, k, v, A, Ap, L, W):
    """fp32 q, k, v -> [A' L, W] copies on the TF32 grid (rounded to nearest; the pad sample zero): ``glue.cu`` ``round3_tf32``."""
    src = [t.reshape(A * L, W) for t in (q, k, v)]
    src = [t if t.is_contiguous() else t.contiguous() for t in src]
    out = [torch.empty(Ap * L, W, device=q.device, dtype=F32) for _ in range(3)]
    n4, n4p = A * L * W // 4, Ap * L * W // 4
    dev = _index(q)
    _sm100_kernel("glue", "round3_tf32", dev, smem=0)(_finish_grid(n4p, dev, 3), (256, 1, 1), *src, *out, int(n4), int(n4p))
    return out


def _fwd_fake(q, k, v, pair, lnw, wb, mask, eps, train):
    """Shapes of the pair-fused forward: (O [A L, W] fp32, LSE [A', H, L] fp32 (A' = A rounded up to even), bias [H, L, L] in q's
    dtype, the fp32 pair's row statistics [L L, 2] in training, else [0, 2], and (fp32) q, k, v on the TF32 grid [A' L, W], else
    three empty [0, W])."""
    A, _, L, H, DH = q.shape
    st = (L * L, 2) if (train and pair.dtype is F32) else (0, 2)
    r = (_even(A) * L if q.dtype is F32 else 0, H * DH)
    return (
        q.new_empty((A * L, H * DH), dtype=F32),
        q.new_empty((_even(A), H, L), dtype=F32),
        q.new_empty((H, L, L), dtype=q.dtype),
        q.new_empty(st, dtype=F32),
        *(q.new_empty(r, dtype=F32) for _ in range(3)),
    )


@opaque(fake=_fwd_fake, name="augmented_attention_pair_sm100_fwd")
def _fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor,
    mask: torch.Tensor | None, eps: float, train: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(O fp32 [A L, W], LSE [A', H, L] (log2 units), bias [H, L, L] (natural units, the core's dtype), pair statistics, and for the
    TF32 core the rounded q, k, v rows its backward reads): the pair pass and the core's forward."""
    A, _, L, H, DH = q.shape
    W, Ap = H * DH, _even(A)
    bf = q.dtype is BF
    bias, pst = _pair_bias(pair, lnw, wb, mask, eps, bf, train)
    if bf:
        q2, k2, v2 = (_rows(t, A, Ap, L, W) for t in (q, k, v))
        O, LSE = forward(q2, k2, v2, bias, Ap, L, H, DH)
        e = q.new_empty((0, W), dtype=F32)
        rq = (e, e.clone(), e.clone())
    else:
        rq = _round3(q, k, v, A, Ap, L, W)
        O, LSE = forward_tf32(*rq, bias, Ap, L, H, DH)
    return (O if Ap == A else O[: A * L]), LSE, bias, pst, *rq


def _bwd_fake(do, q2, k2, v2, pair, lnw, wb, o, lse, bias, pst, eps, A):
    """Shapes of the pair-fused backward: (dq, dk, dv [A L, W] in the core's dtype, dpair like pair, dlnw [d_pair], dwb [H, d_pair]
    fp32)."""
    M, W = o.shape
    dt = BF if q2.dtype is BF else F32
    return (
        o.new_empty((M, W), dtype=dt),
        o.new_empty((M, W), dtype=dt),
        o.new_empty((M, W), dtype=dt),
        pair.new_empty(pair.shape),
        lnw.new_empty(lnw.shape, dtype=F32),
        wb.new_empty(wb.shape, dtype=F32),
    )


@opaque(fake=_bwd_fake, name="augmented_attention_pair_sm100_bwd")
def _bwd(
    do: torch.Tensor, q2: torch.Tensor, k2: torch.Tensor, v2: torch.Tensor, pair: torch.Tensor, lnw: torch.Tensor, wb: torch.Tensor,
    o: torch.Tensor, lse: torch.Tensor, bias: torch.Tensor, pst: torch.Tensor, eps: float, A: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dq, dk, dv [A L, W] in the core's dtype, dpair, dlnw, dwb fp32) of ``_fwd``. q2 / k2 / v2: bf16 q, k, v as the caller holds
    them (any [A, 1, L, H, d] / [A L, W] view), or the TF32 core's rounded [A' L, W] rows from ``_fwd``."""
    Ap, H, L = lse.shape
    M, W = o.shape
    DH = W // H
    dev = _index(o)
    bf = q2.dtype is BF
    do2 = do.reshape(M, W)
    if do2.dtype not in (BF, F32):
        do2 = do2.float()
    if not do2.is_contiguous():
        do2 = do2.contiguous()
    dd = torch.empty(Ap, H, L, device=o.device, dtype=F32)
    if Ap != A:
        dd[A:].zero_()
    n4 = M * W // 4
    if bf:
        q2, k2, v2 = (_rows(t, A, Ap, L, W) for t in (q2, k2, v2))
        dob = torch.empty(Ap * L, W, device=o.device, dtype=BF)
        if Ap != A:
            dob[M:].zero_()
        _prep_do(do2, o, dob, dd, M, L, H, DH)
        DQ, DK, DV, DB = _bwd_bf16(q2, k2, v2, dob, bias, lse, dd, Ap, L, H, DH)
        dq, dk, dv = (torch.empty(M, W, device=o.device, dtype=BF) for _ in range(3))
        _sm100_kernel("glue", "finish_bf16", dev, smem=0)(
            _finish_grid(n4, dev, 3), (256, 1, 1), DQ, int(DQ.shape[0]), int(Ap * L * W), DK, DV, dq, dk, dv, int(n4)
        )
    else:
        do32 = do2.float() if do2.dtype is not F32 else do2
        dor = torch.empty(Ap * L, W, device=o.device, dtype=F32)   # dO on the TF32 grid (pad rows zero)
        if Ap != A:
            dor[M:].zero_()
        _sm100_kernel("glue", "prep_do_f32r", dev, smem=0)(
            ((M * H + 255) // 256, 1, 1), (256, 1, 1), do32, o, dor, dd, int(M), int(L), int(H), int(DH)
        )
        DQ, DK, DV, DB = _bwd_tf32(q2, k2, v2, dor, bias, lse, dd, Ap, L, H, DH)
        dq = torch.empty(M, W, device=o.device, dtype=F32)
        _sm100_kernel("glue", "finish_f32", dev, smem=0)(_finish_grid(n4, dev), (256, 1, 1), DQ, int(DQ.shape[0]), int(Ap * L * W), dq, int(n4))
        dk, dv = (DK, DV) if Ap == A else (DK[:M], DV[:M])
    # the pair side: dpair and dWf from dbias (natural units; the masked keys carry zero dbias)
    dp = pair.shape[-1]
    p2 = pair.reshape(L * L, dp)
    if pair.dtype is BF:
        from ..apb import acc_n, ext, pair_bias_bwd, width

        acc = torch.empty(ext().pair_bwd_blocks(pair, H, L, 1), acc_n(H, 384, dp), device=o.device, dtype=F32)
        dz, _, _ = pair_bias_bwd(p2, DB, _wf(wb, lnw, BF), L, eps, acc, 384)
        off = 2 * 384 + width(H, 384)
        dwf = acc[:, off : off + H * dp].sum(0).view(H, dp)   # block partials in row order
    else:
        from miniworld_engine.kernels.bias_only_dit.cuda.tf32 import rows32

        F = rows32()
        dz = torch.empty_like(p2)
        pwf = torch.empty(F.partial_rows(M), H, dp, device=o.device, dtype=F32)
        nwf = F.pair_bias_bwd_tc_cuda(DB.view(H, L * L), p2, pst, _wf(wb, lnw, F32), dz, pwf, 1)
        dwf = pwf[:nwf].sum(0)
    return dq, dk, dv, dz.view(pair.shape), (dwf * wb.float()).sum(0), dwf * lnw.float()


class _AttnPairSm100(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pair, lnw, wb, mask, eps):  # noqa: D102
        A, _, L, H, DH = q.shape
        o, lse, bias, pst, rq, rk, rv = _fwd(q, k, v, pair, lnw, wb, mask, eps, True)
        qkv = (q, k, v) if q.dtype is BF else (rq, rk, rv)   # the TF32 core's backward reads the rounded rows
        ctx.save_for_backward(*qkv, pair, lnw, wb, o, lse, bias, pst)
        ctx.eps, ctx.shape = eps, q.shape
        return o.view(A, 1, L, H, DH)

    @staticmethod
    def backward(ctx, do):  # noqa: D102
        q2, k2, v2, pair, lnw, wb, o, lse, bias, pst = ctx.saved_tensors
        shp = ctx.shape
        dq, dk, dv, dz, dlnw, dwb = _bwd(do.contiguous(), q2, k2, v2, pair, lnw, wb, o, lse, bias, pst, ctx.eps, shp[0])
        return dq.view(shp), dk.view(shp), dv.view(shp), dz, dlnw.to(lnw.dtype), dwb.to(wb.dtype), None, None


def attention_pair_sm100(q, k, v, pair, lnw, wb, mask=None, eps: float = 1e-5):
    """``softmax(q k^T / sqrt(d) + (wb * lnw) . LN(pair)) v`` for q, k, v [A, 1, L, h, d] (bf16 or fp32, one dtype), pair [1, L, L,
    d_pair], lnw [d_pair] and wb [h, d_pair] (leaves of any float dtype), mask a key mask [L] bool (True = attend) or None; returns fp32
    [A, 1, L, h, d]. Call ``available()`` first."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, pair, lnw, wb))):
        A, _, L, H, DH = q.shape   # inference: nothing for a backward is made or kept
        return _fwd(q, k, v, pair, lnw, wb, mask, eps, False)[0].view(A, 1, L, H, DH)
    return _AttnPairSm100.apply(q, k, v, pair, lnw, wb, mask, eps)


__all__ = ["attention_pair_sm100", "available", "supported"]
