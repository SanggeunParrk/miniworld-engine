"""Block-local window attention (AlphaFold 3's atom attention: 32 queries x ``keys`` keys per window, per-window pair bias) in Triton.

The portable core of :class:`~miniworld_engine.modules.local_dit.LocalDiTBlock` and of the atom variant stacks
(``modules.dit_variant.AtomDiTVariantStack``): the PyTorch path gathers a key window per query window, runs an fp32 einsum and
materialises ``[A, B, H, nwin, 32, keys]`` scores and probabilities; this one keeps a window's scores in registers.

    out[a, b, 32 w + i, h] = softmax_j( q . k[start[b, w] + j] * d^-1/2 + bias[b, h, w, i, j] ) v[start[b, w] + j]

* ``q, k, v``  ``[A, B, N, H, D]`` (any strides, unit stride on D), ``bias`` ``[B, H, nwin, 32, keys]`` (any strides and dtype),
* ``starts``   ``[B, nwin]`` first key of every window (may lie before 0 or past ``N``; ``plan`` makes them for AlphaFold 3's
               ``"slide"`` and for the clipped ``"pad"`` window),
* a key is live when it lies in ``[0, N)``, ``keep[b, key]`` is set, it is within ``window_half`` of the query position (if given) and
  ``uid`` agrees (if given: a *real* query sees only keys of its own conformer, a padding query sees every live key).
  A row with no live key gives 0 (the reference's ``nan_to_num``); query rows ``>= N`` are not written.

Backward (``WindowAttentionFunction``): one program per window computes ``dq``, ``dbias`` (``= dscores``, summed over the A samples
in a fixed order: A is split into G strided groups, one fp32 partial per group, added by torch) and the window's *partial* ``dk`` /
``dv`` ``[A, B, nwin, keys, H, D]`` (fp32); a second kernel adds each key's partials over the windows that cover it in a fixed
order, so there are no atomics and the gradients are bitwise reproducible.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.kernels._tf32 import tf32_dot

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import atom_key, pack
from miniworld_engine.kernels._compile import opaque

QUERIES = 32
_LOG2E = tl.constexpr(1.4426950408889634)
MAX_HEAD_DIM = 64
MAX_KEYS = 256


def plan(starts_rule: str, n: int, nwin: int, keys: int, keep: torch.Tensor | None, device) -> torch.Tensor:
    """``[B, nwin]`` int32 first key of every window. ``"pad"``: ``32 w - (keys - 32) / 2`` (the clipped window; keys outside
    ``[0, N)`` and masked keys are off). ``"slide"``: AlphaFold 3's rule, shifted back inside the real atoms (``keep`` counts them;
    real atoms come first), as ``modules.local_dit.slide_starts``."""
    first = QUERIES * torch.arange(nwin, device=device) - (keys - QUERIES) // 2
    if starts_rule == "pad":
        b = 1 if keep is None else keep.shape[0]
        return first[None].expand(b, -1).to(torch.int32).contiguous()
    from miniworld_engine.modules.local_dit.module import slide_starts

    n_real = (torch.full((1,), n, device=device, dtype=torch.long) if keep is None else keep.long().sum(-1))
    return slide_starts(n_real, nwin, keys).to(torch.int32).contiguous()


def fold_range(starts: torch.Tensor, n: int, keys: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B, ntile]`` first and last window that can cover each 32-key tile ``[32 t, 32 t + 32)`` (a conservative contiguous range,
    exact when ``starts`` is non-decreasing, which every rule above is except below ``keys`` real atoms)."""
    s = starts.long()
    ntile = (n + QUERIES - 1) // QUERIES
    tile = QUERIES * torch.arange(ntile, device=s.device)[None].expand(s.shape[0], -1).contiguous()
    ends = torch.cummax(s + keys, dim=1).values                              # non-decreasing: max end over windows <= w
    suffix_min = torch.flip(torch.cummin(torch.flip(s, [1]), dim=1).values, [1])  # non-decreasing: min start over windows >= w
    lo = torch.searchsorted(ends.contiguous(), tile, right=True)             # first window whose (running) end passes the tile
    hi = torch.searchsorted(suffix_min.contiguous(), tile + (QUERIES - 1), right=True) - 1
    return lo.to(torch.int32).contiguous(), hi.to(torch.int32).contiguous()


@triton.jit
def _live(w, start, N, keep_ptr, uid_ptr, b, iq, ik, KEYS: tl.constexpr, WINDOW_HALF: tl.constexpr, HAS_UID: tl.constexpr):
    """[32, KEYS_PAD] liveness of every (query i, key slot j) of window ``w``."""
    kpos = start + ik
    kin = (ik < KEYS) & (kpos >= 0) & (kpos < N)
    kept = tl.load(keep_ptr + b * N + kpos, mask=kin, other=0) != 0
    kin = kin & kept
    qpos = w * 32 + iq
    live = kin[None, :]
    if WINDOW_HALF >= 0:
        live = live & (tl.abs(qpos[:, None] - kpos[None, :]) <= WINDOW_HALF)
    if HAS_UID:
        qin = qpos < N
        qpad = tl.load(keep_ptr + b * N + qpos, mask=qin, other=0) == 0
        quid = tl.load(uid_ptr + b * N + qpos, mask=qin, other=0)
        kuid = tl.load(uid_ptr + b * N + kpos, mask=kin, other=0)
        live = live & ((quid[:, None] == kuid[None, :]) | qpad[:, None])
    return live, kpos, kin


@triton.autotune(configs=configs_for("augmented_attention_fwd_window_triton"), key=["shape_key"])
@triton.jit
def _window_fwd(
    Q, K, V, Bias, Starts, Keep, Uid, Out, Lse, sm_scale,
    sqa, sqb, sqn, sqh, ska, skb, skn, skh, sva, svb, svn, svh, soa, sob, son, soh,
    sbb, sbh, sbw, sbq, sbk,
    N, NW, B,
    H: tl.constexpr, HEAD_DIM: tl.constexpr, HD_PAD: tl.constexpr, KEYS: tl.constexpr, BLOCK_N: tl.constexpr,
    WINDOW_HALF: tl.constexpr, HAS_UID: tl.constexpr, PRECISION: tl.constexpr, shape_key,
):
    w = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    a = tl.program_id(2).to(tl.int64)
    b = bh // H
    h = bh % H
    start = tl.load(Starts + b * NW + w).to(tl.int64)
    iq = tl.arange(0, 32)
    d = tl.arange(0, HD_PAD)
    dm = d < HEAD_DIM
    qpos = w * 32 + iq
    q = tl.load(Q + a * sqa + b * sqb + h * sqh + qpos[:, None] * sqn + d[None, :], mask=(qpos < N)[:, None] & dm[None, :], other=0.0)
    m_i = tl.zeros([32], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([32], dtype=tl.float32)
    acc = tl.zeros([32, HD_PAD], dtype=tl.float32)
    for n0 in range(0, KEYS, BLOCK_N):
        ik = n0 + tl.arange(0, BLOCK_N)
        live, kpos, kin = _live(w, start, N, Keep, Uid, b, iq, ik, KEYS, WINDOW_HALF, HAS_UID)
        k = tl.load(K + a * ska + b * skb + h * skh + kpos[None, :] * skn + d[:, None], mask=kin[None, :] & dm[:, None], other=0.0)
        bias = tl.load(Bias + b * sbb + h * sbh + w * sbw + iq[:, None] * sbq + ik[None, :] * sbk,
                       mask=(ik < KEYS)[None, :], other=0.0).to(tl.float32)
        s = tf32_dot(q, k, input_precision=PRECISION) * sm_scale + bias
        s = tl.where(live, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2((m_i - m_safe) * _LOG2E)
        p = tl.math.exp2((s - m_safe[:, None]) * _LOG2E)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + a * sva + b * svb + h * svh + kpos[:, None] * svn + d[None, :], mask=kin[:, None] & dm[None, :], other=0.0)
        acc = acc * alpha[:, None] + tf32_dot(p.to(v.dtype), v, input_precision=PRECISION)
        m_i = m_new
    o = acc / tl.maximum(l_i, 1e-30)[:, None]
    m_fin = tl.where(m_i == float("-inf"), 0.0, m_i)
    lse = tl.where(l_i > 0.0, m_fin + tl.log(tl.maximum(l_i, 1e-30)), float("inf"))
    tl.store(Lse + ((a * B + b) * H + h) * (NW * 32) + qpos, lse)
    tl.store(Out + a * soa + b * sob + h * soh + qpos[:, None] * son + d[None, :], o.to(Out.type.element_ty),
             mask=(qpos < N)[:, None] & dm[None, :])


@triton.autotune(configs=configs_for("augmented_attention_bwd_window_triton"), key=["shape_key"])
@triton.jit
def _window_bwd(
    Q, K, V, Bias, Starts, Keep, Uid, Out, Lse, DO, DQ, DKP, DVP, DB, sm_scale,
    sqa, sqb, sqn, sqh, ska, skb, skn, skh, sva, svb, svn, svh, soa, sob, son, soh,
    sbb, sbh, sbw, sbq, sbk,
    N, NW, B, A, G,
    H: tl.constexpr, HEAD_DIM: tl.constexpr, HD_PAD: tl.constexpr, KEYS: tl.constexpr, BLOCK_N: tl.constexpr,
    WINDOW_HALF: tl.constexpr, HAS_UID: tl.constexpr, NEED_DB: tl.constexpr, NEED_DKV: tl.constexpr, PRECISION: tl.constexpr,
    shape_key,
):
    w = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    g = tl.program_id(2).to(tl.int64)
    b = bh // H
    h = bh % H
    start = tl.load(Starts + b * NW + w).to(tl.int64)
    iq = tl.arange(0, 32)
    d = tl.arange(0, HD_PAD)
    dm = d < HEAD_DIM
    qpos = w * 32 + iq
    qin = qpos < N
    for a in range(g, A, G):
        row = (a * B + b) * H + h
        q = tl.load(Q + a * sqa + b * sqb + h * sqh + qpos[:, None] * sqn + d[None, :], mask=qin[:, None] & dm[None, :], other=0.0)
        o = tl.load(Out + a * soa + b * sob + h * soh + qpos[:, None] * son + d[None, :], mask=qin[:, None] & dm[None, :], other=0.0)
        do = tl.load(DO + a * soa + b * sob + h * soh + qpos[:, None] * son + d[None, :], mask=qin[:, None] & dm[None, :], other=0.0)
        lse = tl.load(Lse + row * (NW * 32) + qpos)
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
        dq = tl.zeros([32, HD_PAD], dtype=tl.float32)
        for n0 in range(0, KEYS, BLOCK_N):
            ik = n0 + tl.arange(0, BLOCK_N)
            live, kpos, kin = _live(w, start, N, Keep, Uid, b, iq, ik, KEYS, WINDOW_HALF, HAS_UID)
            k = tl.load(K + a * ska + b * skb + h * skh + kpos[:, None] * skn + d[None, :], mask=kin[:, None] & dm[None, :], other=0.0)
            v = tl.load(V + a * sva + b * svb + h * svh + kpos[:, None] * svn + d[None, :], mask=kin[:, None] & dm[None, :], other=0.0)
            bias = tl.load(Bias + b * sbb + h * sbh + w * sbw + iq[:, None] * sbq + ik[None, :] * sbk,
                           mask=(ik < KEYS)[None, :], other=0.0).to(tl.float32)
            s = tf32_dot(q, tl.trans(k), input_precision=PRECISION) * sm_scale + bias
            p = tl.where(live, tl.math.exp2((s - lse[:, None]) * _LOG2E), 0.0)
            dp = tf32_dot(do, tl.trans(v), input_precision=PRECISION)
            ds = p * (dp - delta[:, None])
            dq += tf32_dot(ds.to(k.dtype), k, input_precision=PRECISION)
            if NEED_DB:
                dbp = DB + (((g * B + b) * H + h) * NW + w) * (32 * KEYS) + iq[:, None] * KEYS + ik[None, :]
                dbm = (ik < KEYS)[None, :]
                if a == g:
                    tl.store(dbp, ds, mask=dbm)
                else:
                    tl.store(dbp, tl.load(dbp, mask=dbm, other=0.0) + ds, mask=dbm)
                # the next sample's read-modify-write of this slot may be done by another thread of the block (the tile's layouts differ
                # between the dot's result and the load): a block-wide barrier makes this store visible to it
                tl.debug_barrier()
            if NEED_DKV:
                dv = tf32_dot(tl.trans(p).to(do.dtype), do, input_precision=PRECISION)
                dk = tf32_dot(tl.trans(ds).to(q.dtype), q, input_precision=PRECISION) * sm_scale
                part = (((a * B + b) * NW + w) * KEYS + ik[:, None]) * (H * HEAD_DIM) + h * HEAD_DIM + d[None, :]
                pm = (ik < KEYS)[:, None] & dm[None, :]
                tl.store(DKP + part, dk, mask=pm)
                tl.store(DVP + part, dv, mask=pm)
        tl.store(DQ + a * sqa + b * sqb + h * sqh + qpos[:, None] * sqn + d[None, :], (dq * sm_scale).to(DQ.type.element_ty),
                 mask=qin[:, None] & dm[None, :])


@triton.autotune(configs=configs_for("augmented_attention_bwd_reduce_window_triton"), key=["shape_key"])
@triton.jit
def _window_fold(
    DKP, DVP, DK, DV, Starts, Lo, Hi,
    sda, sdb, sdn, sdh,
    N, NW, NT, B,
    H: tl.constexpr, HEAD_DIM: tl.constexpr, KEYS: tl.constexpr, BLOCK_D: tl.constexpr, shape_key,
):
    t = tl.program_id(0).to(tl.int64)
    ab = tl.program_id(1).to(tl.int64)
    hd = tl.program_id(2).to(tl.int64)
    chunks = (HEAD_DIM + BLOCK_D - 1) // BLOCK_D
    h = hd // chunks
    a = ab // B
    b = ab % B
    lo = tl.load(Lo + b * NT + t).to(tl.int64)
    hi = tl.load(Hi + b * NT + t).to(tl.int64)
    ip = tl.arange(0, 32)
    d = (hd % chunks) * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = d < HEAD_DIM
    pos = t * 32 + ip
    acc_k = tl.zeros([32, BLOCK_D], dtype=tl.float32)
    acc_v = tl.zeros([32, BLOCK_D], dtype=tl.float32)
    for w in range(lo, hi + 1):
        start = tl.load(Starts + b * NW + w).to(tl.int64)
        rel = pos - start
        cover = (rel >= 0) & (rel < KEYS) & (pos < N)
        off = (((a * B + b) * NW + w) * KEYS + rel[:, None]) * (H * HEAD_DIM) + h * HEAD_DIM + d[None, :]
        msk = cover[:, None] & dm[None, :]
        acc_k += tl.load(DKP + off, mask=msk, other=0.0)
        acc_v += tl.load(DVP + off, mask=msk, other=0.0)
    dst = a * sda + b * sdb + h * sdh + pos[:, None] * sdn + d[None, :]
    smk = (pos < N)[:, None] & dm[None, :]
    tl.store(DK + dst, acc_k.to(DK.type.element_ty), mask=smk)
    tl.store(DV + dst, acc_v.to(DV.type.element_ty), mask=smk)


# ------------------------------------------------------------------------------------------------------------------ launchers
def _precision(dtype: torch.dtype) -> str:
    return "tf32" if dtype == torch.float32 else "ieee"


def _geometry(q, keys):
    a, b, n, h, d = q.shape
    return a, b, n, h, d, (n + QUERIES - 1) // QUERIES, max(16, triton.next_power_of_2(d))


def _uid_or_dummy(uid, keep):
    return (uid if uid is not None else keep), uid is not None


def _launch_fwd_fake(q, k, v, bias, starts, keep, uid, keys, window_half, shape_key):
    """``(out, lse)``: ``out`` like ``q``; ``lse`` the per-row natural logsumexp ``[A, B, H, nwin * 32]`` in fp32."""
    a, b, n, h, d = q.shape
    return torch.empty_like(q), q.new_empty((a, b, h, ((n + 31) // 32) * 32), dtype=torch.float32)


@opaque(fake=_launch_fwd_fake, name="augmented_attention_fwd_window")
def _launch_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, starts: torch.Tensor, keep: torch.Tensor,
                uid: torch.Tensor | None, keys: int, window_half: int, shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The window attention forward launch -> ``(out, lse)``; ``lse`` is what the backward recomputes the probabilities from."""
    a, b, n, h, d, nwin, hd_pad = _geometry(q, keys)
    out = torch.empty_like(q)
    lse = torch.empty(a, b, h, nwin * QUERIES, device=q.device, dtype=torch.float32)
    uid_t, has_uid = _uid_or_dummy(uid, keep)
    _window_fwd[(nwin, b * h, a)](
        q, k, v, bias, starts, keep, uid_t, out, lse, d**-0.5,
        *q.stride()[:4], *k.stride()[:4], *v.stride()[:4], *out.stride()[:4], *bias.stride(),
        n, nwin, b,
        H=h, HEAD_DIM=d, HD_PAD=hd_pad, KEYS=keys, WINDOW_HALF=window_half, HAS_UID=has_uid,
        PRECISION=_precision(q.dtype), shape_key=pack(shape_key, H=h, HEAD_DIM=d),
    )
    return out, lse


def _launch_bwd_fake(do, q, k, v, bias, starts, keep, uid, out, lse, lo, hi, keys, window_half, need_db, need_dkv, shape_key):
    """``(dq, dk, dv, dbias_partials)``: ``dk`` / ``dv`` are empty when no key or value needs a gradient, the partials ``[G, B, H, nwin, 32, keys]`` fp32."""
    a, b, n, h, d = q.shape
    nwin = (n + 31) // 32
    return (torch.empty_like(q), torch.empty_like(k) if need_dkv else k.new_empty(0), torch.empty_like(v) if need_dkv else v.new_empty(0),
            q.new_empty((_groups(a, nwin * b * h), b, h, nwin, 32, keys) if need_db else (0,), dtype=torch.float32))


def _groups(a: int, programs: int) -> int:
    """Strided groups of the A samples: enough programs to fill the card (~264 = 2 per SM of an H100), at most 8 partials."""
    return max(1, min(a, 8, -(-264 // max(programs, 1))))


@opaque(fake=_launch_bwd_fake, name="augmented_attention_bwd_window")
def _launch_bwd(do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, starts: torch.Tensor,
                keep: torch.Tensor, uid: torch.Tensor | None, out: torch.Tensor, lse: torch.Tensor, lo: torch.Tensor,
                hi: torch.Tensor, keys: int, window_half: int, need_db: bool, need_dkv: bool,
                shape_key: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The window attention backward launches -> ``(dq, dk, dv, dbias_partials)``: the per-window kernel (``dq``, the bias gradient
    summed over strided groups of the A samples, the windows' partial ``dk`` / ``dv``), then the fold adding each key's partials."""
    a, b, n, h, d, nwin, hd_pad = _geometry(q, keys)
    uid_t, has_uid = _uid_or_dummy(uid, keep)
    groups = _groups(a, nwin * b * h)
    dq = torch.empty_like(q)
    dkp = torch.empty(a, b, nwin, keys, h, d, device=q.device, dtype=torch.float32) if need_dkv else q.new_empty(1)
    dvp = torch.empty_like(dkp)
    db = torch.empty(groups, b, h, nwin, 32, keys, device=q.device, dtype=torch.float32) if need_db else q.new_empty(1)
    _window_bwd[(nwin, b * h, groups)](
        q, k, v, bias, starts, keep, uid_t, out, lse, do, dq, dkp, dvp, db, d**-0.5,
        *q.stride()[:4], *k.stride()[:4], *v.stride()[:4], *out.stride()[:4], *bias.stride(),
        n, nwin, b, a, groups,
        H=h, HEAD_DIM=d, HD_PAD=hd_pad, KEYS=keys, WINDOW_HALF=window_half, HAS_UID=has_uid,
        NEED_DB=need_db, NEED_DKV=need_dkv, PRECISION=_precision(q.dtype), shape_key=pack(shape_key, H=h, HEAD_DIM=d),
    )
    if need_dkv:
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        ntile = (n + QUERIES - 1) // QUERIES
        _window_fold[lambda meta: (ntile, a * b, h * triton.cdiv(d, meta['BLOCK_D']))](
            dkp, dvp, dk, dv, starts, lo, hi, dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            n, nwin, ntile, b, H=h, HEAD_DIM=d, KEYS=keys, shape_key=pack(shape_key, H=h, HEAD_DIM=d),
        )
    else:
        dk, dv = k.new_empty(0), v.new_empty(0)
    return dq, dk, dv, (db if need_db else q.new_empty(0, dtype=torch.float32))


class WindowAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, starts, keep, uid, keys: int, window_half: int):
        if q.shape[-1] > MAX_HEAD_DIM or keys > MAX_KEYS:
            raise ValueError(f"window attention supports head_dim <= {MAX_HEAD_DIM} and keys <= {MAX_KEYS}, got {q.shape[-1]} / {keys}")
        n = q.shape[2]
        out, lse = _launch_fwd(q, k, v, bias, starts, keep, uid, keys, window_half, atom_key(n))
        lo, hi = fold_range(starts, n, keys) if (k.requires_grad or v.requires_grad) else (starts, starts)
        ctx.save_for_backward(q, k, v, bias, starts, keep, uid, out, lse, lo, hi)
        ctx.cfg = (keys, window_half, n, bias.requires_grad, k.requires_grad or v.requires_grad)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, bias, starts, keep, uid, out, lse, lo, hi = ctx.saved_tensors
        keys, window_half, n, need_db, need_dkv = ctx.cfg
        dq, dk, dv, db = _launch_bwd(do.contiguous().to(q.dtype), q, k, v, bias, starts, keep, uid, out, lse, lo, hi, keys,
                                     window_half, need_db, need_dkv, atom_key(n))
        dbias = db.sum(0).to(bias.dtype) if need_db else None   # [B, H, nwin, 32, keys]
        return dq, (dk if need_dkv else None), (dv if need_dkv else None), dbias, None, None, None, None, None


def window_attention(q, k, v, bias, starts, keep, *, uid=None, keys: int = 128, window_half: int | None = None) -> torch.Tensor:
    """``q, k, v`` ``[A, B, N, H, D]``; ``bias`` ``[B, H, nwin, 32, keys]``; ``starts`` ``[B, nwin]`` int32; ``keep`` ``[B, N]`` bool
    (live keys / real queries). Returns ``[A, B, N, H, D]`` in ``q``'s dtype. Differentiable in q, k, v and bias."""
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        q, k, v = (t.contiguous() for t in (q, k, v))
    keep = keep.to(torch.uint8).contiguous() if keep.dtype != torch.uint8 else keep.contiguous()
    uid = None if uid is None else uid.to(torch.int32).contiguous()
    return WindowAttentionFunction.apply(q, k, v, bias, starts, keep, uid, keys, -1 if window_half is None else int(window_half))
