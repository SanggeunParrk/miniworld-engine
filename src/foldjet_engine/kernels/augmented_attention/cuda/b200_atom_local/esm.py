"""ESMFold2's atom block kernels on the B200 (sm_100a, card ``b200_sm100``): ``origin/b200/ladder-final``'s ``sm100_atom_local`` ESMFold2
wrappers (dc425caa5), B200 halves only. ``kernels/integrations/local_dit_esm_b200`` (``AtomDiTESMFold2Pair``) is the only caller.

  lattn_esm.cu          the 192-key rank-window attention (tcgen05; the +/- 64 band by position, the slid 192-key window, the
                        position-indexed pair bias [4, nwin, 32, 160] of ``esm_bias``), forward with the optional row LSE.
  lattn_esm_dq.cu       dQ and the bias gradient (per sample-split partial, summed in a fixed order).
  lattn_esm_dkv.cu      dK, dV.
  lesm.cu               the pair bias by position, the q / k RMSNorm + rotary (and backward), the conditioning SiLU backward, the fp32
                        bias scatter.
  lattn_esm_*_tf32.cu   the fp32 path's twins (tcgen05 kind::tf32).
  lbias.cu              ``local_esm_bias_bwd`` (the bf16 pair bias backward by position) and ``local_bias_fin``.
"""

from __future__ import annotations

import foldjet_engine._envcompat  # noqa: F401  (old MINIWORLD_* environment names)

import functools

import torch

from foldjet_engine.kernels._device import num_sms
from foldjet_engine.kernels.augmented_attention.cuda.b200_atom_local import (
    DM,
    DP,
    NH,
    WQ,
    _dev,
    _out32,
    _params,
    _rows,
    _rows32,
    _tmap,
    _u8,
    _vec,
    cubin,
    nwindows,
    pair_bias_bwd32,
)

ESM_KEYS, ESM_SPAN = 192, 160
SOURCES = ("lattn_esm", "lattn_esm_dq", "lattn_esm_dkv", "lattn_esm_tf32", "lattn_esm_dq_tf32", "lattn_esm_dkv_tf32", "lesm")


#: ESMFold2's atom attention on sm_100a (``LocalDiTBlock(**ESMFOLD2)``, bf16 inference), loaded apart from KERNELS (``load_esm``)
KERNELS_ESM = {
    "esm": ("lattn_esm", "local_attn_esm", 191232),
    "esm_bias": ("lesm", "local_esm_bias", 0),
    "esm_qk": ("lesm", "local_esm_qkrope", 0),
}


@functools.lru_cache(maxsize=None)
def _load_esm(name: str, device_index: int):
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_ESM[name]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem), func, smem)


@functools.lru_cache(maxsize=None)
def _load_esm_pdl(device_index: int):
    """``local_attn_esm`` built ``-DLATTN_PDL`` and launched as a programmatic dependent (round 3 part 11): its setup overlaps the bias
    kernel and it lets ``post_fwd`` stage its weights while it runs. MW_LD11_EP=0: the plain launch (A / B)."""
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_ESM["esm"]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem, False, ("LATTN_PDL=1",)), func, smem, pdl=True)


@functools.lru_cache(maxsize=None)
def _load_esm_bias_early(device_index: int):
    """``local_esm_bias`` built ``-DEARLY=1`` and launched as a programmatic dependent of the projection kernel (round 3 part 12)."""
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_ESM["esm_bias"]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem, False, ("EARLY=1",)), func, smem, pdl=True)


def esm_bias_early(z, gamma, wb, kmask, n0, eps=1e-5, ctas=8):
    """``esm_bias`` as a programmatic dependent of the projection kernel, beside it (round 3 part 12; ``lbias_tf32.cu``
    ``local_bias_fwd_f32_early``'s contract): -> (bias2, launch). Call it BEFORE the chain's launches (z, the mask, the parameters and the
    output are made here) and ``launch()`` right after the projection kernel, whose dependents start after its own wait. Same bits."""
    nwin = z.shape[0]
    N = 32 * nwin
    assert tuple(z.shape) == (nwin, 32, ESM_KEYS, DP) and z.is_contiguous() and N % 128 == 0
    d = _dev(z)
    out = torch.empty(NH, nwin, 32, ESM_SPAN, device=z.device, dtype=torch.float32)
    km = _u8(kmask)
    if km is not None and km.data_ptr() % 16:
        km = km.clone()
    wb_, g_ = wb.contiguous(), gamma.contiguous()
    flags = int(wb_.dtype == torch.bfloat16) | (int(g_.dtype == torch.bfloat16) << 1) | (int(z.dtype == torch.bfloat16) << 2)
    assert {wb_.dtype, g_.dtype, z.dtype} <= {torch.bfloat16, torch.float32}
    k = _load_esm_bias_early(d)
    grid = (min(ctas * num_sms(d), (nwin * 32 * ESM_SPAN + 255) // 256), 1, 1)

    def launch():
        k(grid, (256, 1, 1), z, g_, wb_, km, out, int(N), int(n0), float(eps), flags)

    return out, launch


def load_esm(device_index: int) -> None:
    """Build and load ESMFold2's attention kernels on the device (raises on any failure)."""
    for name in KERNELS_ESM:
        _load_esm(name, device_index)


def esm_bias(z, gamma, wb, kmask, n0, eps=1e-5):
    """ESMFold2's pair bias by position (``lesm.cu`` ``local_esm_bias``): z the trunked pair [nwin, 32, 192, 16] (bf16 or fp32,
    contiguous), gamma [16], Wb [4, 16], kmask [N] or None (then all n0 = N atoms are real) -> bias2 [4, nwin, 32, 160] fp32 (log2 units,
    -inf where the position is not a key of the query)."""
    nwin = z.shape[0]
    N = 32 * nwin
    assert tuple(z.shape) == (nwin, 32, ESM_KEYS, DP) and z.is_contiguous() and N % 128 == 0
    d = _dev(z)
    out = torch.empty(NH, nwin, 32, ESM_SPAN, device=z.device, dtype=torch.float32)
    km = _u8(kmask)
    if km is not None and km.data_ptr() % 16:
        km = km.clone()
    wb_, g_ = wb.contiguous(), gamma.contiguous()
    flags = int(wb_.dtype == torch.bfloat16) | (int(g_.dtype == torch.bfloat16) << 1) | (int(z.dtype == torch.bfloat16) << 2)
    assert {wb_.dtype, g_.dtype, z.dtype} <= {torch.bfloat16, torch.float32}
    _load_esm("esm_bias", d)(
        (min(8 * num_sms(d), (nwin * 32 * ESM_SPAN + 255) // 256), 1, 1),  # (8 CTAs an SM: lesm.cu)
        (256, 1, 1),
        z,
        g_,
        wb_,
        km,
        out,
        int(N),
        int(n0),
        float(eps),
        flags,
    )
    return out


def esm_qkrope(q, k, wq, wk, cos, sin, eps):
    """ESMFold2's q / k RMSNorm (per head, weights fp32 [32], contiguous) and rotary (tables cos / sin fp32 [n, 16], row r is atom r % n): q, k [M, 128]
    bf16 views with one row stride -> qo, ko [M, 128] bf16 (``lesm.cu`` ``local_esm_qkrope``)."""
    M = q.shape[0]
    assert q.stride(0) == k.stride(0) and q.stride(1) == k.stride(1) == 1 and q.dtype == k.dtype == torch.bfloat16
    assert cos.dtype == sin.dtype == torch.float32 and cos.is_contiguous() and sin.is_contiguous() and cos.shape[1] == 16
    qo, ko = torch.empty(M, DM, device=q.device, dtype=q.dtype), torch.empty(M, DM, device=q.device, dtype=q.dtype)
    assert wq.dtype == wk.dtype == torch.float32 and wq.is_contiguous() and wk.is_contiguous()
    _load_esm("esm_qk", _dev(q))(
        ((M + 7) // 8, 1, 1), (256, 1, 1), q, k, int(q.stride(0)), wq, wk, cos, sin, qo, ko, int(M), int(cos.shape[0]), float(eps)
    )
    return qo, ko


def attn_esm_sm100(q, k, v, bias2, lse=None):
    """ESMFold2's atom attention (``lattn_esm.cu``, tcgen05): q, k, v [A, N, 128] bf16 (contiguous; q / k normalised and rotated), bias2
    [4, nwin, 32, 160] (``esm_bias``) -> O [A, N, 128] bf16 (and the row LSE, log2 units, into ``lse`` [A, 4, N] when given)."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    o = torch.empty_like(q)
    nq = (N + 127) // 128
    sp = max(1, min(A, num_sms(d) // (nq * NH)))
    out = _tmap(o, (DM, N, A), (DM * 2, N * DM * 2), (16, 32, 1), 0)
    import os

    (_load_esm_pdl(d) if os.environ.get("MW_LD11_EP", "1") != "0" else _load_esm("esm", d))(
        (nq * NH * sp, 1, 1),
        (544, 1, 1),
        _rows(q, 128),
        _rows(k, 256),
        _rows(v, 256),
        out,
        bias2,
        lse,
        int(N),
        int(A),
        int(nwin),
        int(sp),
    )
    return o


#: ESMFold2's training kernels on sm_100a (bf16): the attention backward (``lattn_esm_dq.cu``: dQ + the bias gradient; ``lattn_esm_dkv.cu``: dK,
#: dV), the q / k RMSNorm + rotary backward and the conditioning's SiLU backward (``lesm.cu``), the position-indexed pair bias backward
#: (``lbias.cu``; its block partials reduced by ``local_bias_fin``). Loaded apart from the inference kernels (``load_esm_train``).
KERNELS_ESM_TRAIN = {
    "esm_dq": ("lattn_esm_dq", "local_attn_esm_dq", 193280),
    "esm_dkv": ("lattn_esm_dkv", "local_attn_esm_dkv", 200960),
    "esm_qk_b": ("lesm", "local_esm_qkrope_bwd", 0),
    "esm_qk_fin": ("lesm", "local_esm_qk_fin", 0),
    "esm_silu_b": ("lesm", "local_esm_silu_bwd", 0),
    "esm_bias_b": ("lbias", "local_esm_bias_bwd", 0),
    "pb_fin": ("lbias", "local_bias_fin", 0),
}


@functools.lru_cache(maxsize=None)
def _load_esm_train(name: str, device_index: int):
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_ESM_TRAIN[name]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem), func, smem)


def load_esm_train(device_index: int) -> None:
    """Build and load ESMFold2's training kernels on the device (raises on any failure)."""
    for name in KERNELS_ESM_TRAIN:
        _load_esm_train(name, device_index)


def attn_esm_bwd_sm100(q, k, v, do, bias2, lse, dd, dkv):
    """The backward of :func:`attn_esm_sm100` (tcgen05): q, k, v, do [A, N, 128] bf16 contiguous (q / k the normalised, rotated ones), bias2
    [4, nwin, 32, 160] (log2), lse [A, 4, N] (the forward's, log2), dd [A, 4, N] = rowsum(dO O) per head -> dQ, dK, dV (bf16) into the column
    blocks 0 / 1 / 2 of dkv [A N, 512] (row stride dkv.stride(0)); returns the bias gradient [4, nwin, 32, 160] fp32 (natural units, by
    position, summed over the samples in a fixed order: deterministic)."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    ldd = dkv.stride(0)
    nq = N // 128
    out = _tmap(dkv, (dkv.shape[1], N, A), (ldd * 2, N * ldd * 2), (16, 32, 1), 0)
    sp = max(1, min(A, num_sms(d) // (nq * NH)))
    dbias = torch.empty((sp, NH, nwin, WQ, ESM_SPAN) if sp > 1 else (NH, nwin, WQ, ESM_SPAN), device=q.device, dtype=torch.float32)
    _load_esm_train("esm_dq", d)(
        (nq * NH * sp, 1, 1),
        (512, 1, 1),
        _rows(q, 128),
        _rows(k, 256),
        _rows(v, 256),
        _rows(do, 128),
        _vec(lse, 128),
        _vec(dd, 128),
        out,
        bias2,
        dbias,
        int(N),
        int(A),
        int(nwin),
        int(sp),
    )
    sp2 = max(1, min(A, num_sms(d) // (nq * NH)))
    _load_esm_train("esm_dkv", d)(
        (nq * NH * sp2, 1, 1),
        (512, 1, 1),
        _rows(q, 256),
        _rows(k, 128),
        _rows(v, 128),
        _rows(do, 256),
        _vec(lse, 256),
        _vec(dd, 256),
        out,
        bias2,
        int(N),
        int(A),
        int(nwin),
        int(sp2),
    )
    return dbias.sum(0) if sp > 1 else dbias


def esm_qkrope_bwd(dq, dk, q, k, wq, wk, cos, sin, eps, weights=True):
    """The backward of :func:`esm_qkrope`, in place: dq, dk [M, 128] bf16 views with one row stride (the gradients of the rotated q / k) become
    the gradients of q / k [M, 128] (the projections' bf16 outputs, one row stride). -> the norm weights' gradients [64] fp32 (q 0..31, k 32..63,
    summed over the rows and heads in a fixed order). ``weights`` False (constant norm weights): their gradients are not finished (None)."""
    M = q.shape[0]
    assert (
        dq.stride(0) == dk.stride(0)
        and dq.stride(1) == dk.stride(1) == 1
        and q.stride(0) == k.stride(0)
        and q.stride(1) == k.stride(1) == 1
    )
    assert cos.dtype == sin.dtype == torch.float32 and cos.is_contiguous() and sin.is_contiguous() and cos.shape[1] == 16
    d = _dev(q)
    nb = max(1, min(4 * num_sms(d), (M + 7) // 8))
    part = torch.empty(nb, 64, device=q.device, dtype=torch.float32)
    out = torch.empty(64, device=q.device, dtype=torch.float32)
    _load_esm_train("esm_qk_b", d)(
        (nb, 1, 1),
        (256, 1, 1),
        dq,
        dk,
        int(dq.stride(0)),
        q,
        k,
        int(q.stride(0)),
        wq,
        wk,
        cos,
        sin,
        part,
        int(M),
        int(cos.shape[0]),
        float(eps),
    )
    if not weights:
        return None
    _load_esm_train("esm_qk_fin", d)((1, 1, 1), (512, 1, 1), part, int(nb), out)
    return out


def esm_silu_bwd(c, g):
    """c [M, 128] bf16 (the conditioning), g [M, 128] fp32 (the tables' gradient times their weights) -> dc = rn(g silu'(c)), sc = rn(silu(c))."""
    assert c.is_contiguous() and g.is_contiguous() and g.dtype == torch.float32 and c.shape == g.shape
    dc, sc = torch.empty_like(c), torch.empty_like(c)
    n4 = c.numel() // 4
    d = _dev(c)
    _load_esm_train("esm_silu_b", d)((max(1, min(8 * num_sms(d), (n4 + 255) // 256)), 1, 1), (256, 1, 1), c, g, dc, sc, int(n4))
    return dc, sc


def esm_bias_bwd(z, gamma, wb, dbias2, kmask, n0, eps=1e-5):
    """The backward of :func:`esm_bias`: z the trunked pair [nwin, 32, 192, 16] bf16 (contiguous), dbias2 [4, nwin, 32, 160] fp32 (natural
    units) -> dz (z's shape, bf16), dgamma [16], dWb [4, 16] (fp32; the block partials summed in order)."""
    nwin = z.shape[0]
    N = 32 * nwin
    assert tuple(z.shape) == (nwin, 32, ESM_KEYS, DP) and z.is_contiguous() and z.dtype == torch.bfloat16 and N % 128 == 0
    d = _dev(z)
    wb, gamma, bf = _params(wb, gamma)
    km = _u8(kmask)
    if km is not None and km.data_ptr() % 16:
        km = km.clone()
    rows = z.numel() // DP
    dz = torch.empty_like(z)
    nb = min(num_sms(d), (rows + 255) // 256)
    part = torch.empty(nb, NH * DP, device=z.device, dtype=torch.float32)
    dgamma = torch.empty(DP, device=z.device, dtype=torch.float32)
    dwb = torch.empty(NH, DP, device=z.device, dtype=torch.float32)
    _load_esm_train("esm_bias_b", d)(
        (nb, 1, 1), (256, 1, 1), z, wb, gamma, dbias2.contiguous(), km, dz, part, int(N), int(n0), float(eps), bf
    )
    _load_esm_train("pb_fin", d)((1, 1, 1), (64, 1, 1), part, int(nb), wb, gamma, dgamma, dwb, bf)
    return dz, dgamma, dwb


#: ESMFold2's fp32-path kernels on sm_100a (TF32 attention: ``lattn_esm_tf32.cu``; the fp32 q / k norm + rotary of ``lesm.cu``), loaded apart
#: (``load_esm32``).
KERNELS_ESM32 = {
    "esm32": ("lattn_esm_tf32", "local_attn_esm_tf32", 188672),
    "esm_qk32": ("lesm", "local_esm_qkrope_f32", 0),
    "esm_qk_b32": ("lesm", "local_esm_qkrope_bwd_f32", 0),
    "esm_qk_fin": ("lesm", "local_esm_qk_fin", 0),
    "esm_dq32": ("lattn_esm_dq_tf32", "local_attn_esm_dq_tf32", 229632),
    "esm_dkv32": ("lattn_esm_dkv_tf32", "local_attn_esm_dkv_tf32", 200960),
    "esm_bias_s": ("lesm", "local_esm_bias_scatter", 0),
}


@functools.lru_cache(maxsize=None)
def _load_esm32(name: str, device_index: int):
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_ESM32[name]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem), func, smem)


def load_esm32(device_index: int) -> None:
    """Build and load ESMFold2's fp32-path kernels on the device (raises on any failure)."""
    for name in KERNELS_ESM32:
        _load_esm32(name, device_index)


def esm_qkrope32(q, k, wq, wk, cos, sin, eps):
    """:func:`esm_qkrope` on fp32 q, k [M, 128] views (one row stride, unit column stride) -> qo, ko [M, 128] fp32, tf32-rounded (RNA)."""
    M = q.shape[0]
    assert q.stride(0) == k.stride(0) and q.stride(1) == k.stride(1) == 1 and q.dtype == k.dtype == torch.float32
    assert cos.dtype == sin.dtype == torch.float32 and cos.is_contiguous() and sin.is_contiguous() and cos.shape[1] == 16
    qo, ko = torch.empty(M, DM, device=q.device, dtype=q.dtype), torch.empty(M, DM, device=q.device, dtype=q.dtype)
    _load_esm32("esm_qk32", _dev(q))(
        ((M + 7) // 8, 1, 1), (256, 1, 1), q, k, int(q.stride(0)), wq, wk, cos, sin, qo, ko, int(DM), int(M), int(cos.shape[0]), float(eps)
    )
    return qo, ko


def esm_qkrope_bwd32(dq, dk, q, k, wq, wk, cos, sin, eps, weights=True):
    """:func:`esm_qkrope_bwd` on fp32: dq, dk [M, 128] fp32 views (in place), q / k the fp32 projections -> the norm weights' gradients [64] (``weights`` False: not finished, None)."""
    M = q.shape[0]
    assert (
        dq.stride(0) == dk.stride(0)
        and dq.stride(1) == dk.stride(1) == 1
        and q.stride(0) == k.stride(0)
        and q.stride(1) == k.stride(1) == 1
    )
    d = _dev(q)
    nb = max(1, min(4 * num_sms(d), (M + 7) // 8))
    part = torch.empty(nb, 64, device=q.device, dtype=torch.float32)
    out = torch.empty(64, device=q.device, dtype=torch.float32)
    _load_esm32("esm_qk_b32", d)(
        (nb, 1, 1),
        (256, 1, 1),
        dq,
        dk,
        int(dq.stride(0)),
        q,
        k,
        int(q.stride(0)),
        wq,
        wk,
        cos,
        sin,
        part,
        int(M),
        int(cos.shape[0]),
        float(eps),
    )
    if not weights:
        return None
    _load_esm32("esm_qk_fin", d)((1, 1, 1), (512, 1, 1), part, int(nb), out)
    return out


def attn_esm_bwd32(q, k, v, do, bias2, lse, dd, dq, dkv):
    """The backward of :func:`attn_esm32` (tcgen05 kind::tf32): q, k, v, do fp32 [A, N, 128] views (tf32-rounded; unit column stride),
    bias2, lse [A, 4, N], dd [A, 4, N] -> dQ fp32 into ``dq`` [A N, 128] and dK, dV into the columns 128..255 / 256..383 of ``dkv``
    [A N, >= 384] (fp32 views with one row stride); returns the bias gradient [4, nwin, 32, 160] fp32 (natural units, summed over the
    samples in a fixed order)."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    ldd = dkv.stride(0)
    assert dkv.dtype == dq.dtype == torch.float32 and dkv.stride(1) == dq.stride(1) == 1 and dq.stride(0) == ldd
    nq = N // 128
    sp = max(1, min(A, num_sms(d) // (nq * NH)))
    dbias = torch.empty((sp, NH, nwin, WQ, ESM_SPAN) if sp > 1 else (NH, nwin, WQ, ESM_SPAN), device=q.device, dtype=torch.float32)
    _load_esm32("esm_dq32", d)(
        (nq * NH * sp, 1, 1),
        (512, 1, 1),
        _rows32(q, 128),
        _rows32(k, 256),
        _rows32(v, 256),
        _rows32(do, 128),
        lse,
        dd,
        _rows32(k, 256, "128a32"),
        bias2,
        _tmap(dq, (128, N, A), (ldd * 4, N * ldd * 4), (16, 32, 1), 64, "f32"),
        dbias,
        int(N),
        int(A),
        int(nwin),
        int(sp),
    )
    out = dkv[:, 128:384]  # dK at its column 128, dV at 256: TMA stores of 32 rows x 16 columns
    _load_esm32("esm_dkv32", d)(
        (nq * NH * sp, 1, 1),
        (512, 1, 1),
        _rows32(q, 224),
        _rows32(q, 32),
        _rows32(q, 224, "128a32"),
        _rows32(q, 32, "128a32"),
        _rows32(do, 224),
        _rows32(do, 32),
        _rows32(do, 224, "128a32"),
        _rows32(do, 32, "128a32"),
        _rows32(k, 128),
        _rows32(v, 128),
        _vec(lse, 256),
        _vec(dd, 256),
        _tmap(out, (256, N, A), (ldd * 4, N * ldd * 4), (16, 32, 1), 64, "f32"),
        bias2,
        int(N),
        int(A),
        int(nwin),
        int(sp),
    )
    return dbias.sum(0) if sp > 1 else dbias


def esm_bias_bwd32(z, gamma, wb, dbias2, kmask, n0, eps=1e-5):
    """:func:`esm_bias_bwd` for an fp32 trunked pair z [nwin, 32, 192, 16]: dbias2 scattered to the pair's slots (``lesm.cu``), then the fp32
    pair-bias backward (``lbias_tf32.cu``) -> dz fp32, dgamma [16], dWb [4, 16]."""
    nwin = z.shape[0]
    N = 32 * nwin
    assert tuple(z.shape) == (nwin, 32, ESM_KEYS, DP) and z.dtype == torch.float32 and N % 128 == 0
    d = _dev(z)
    km = _u8(kmask)
    if km is not None and km.data_ptr() % 16:
        km = km.clone()
    rows = nwin * 32 * ESM_KEYS
    db = torch.empty(NH, rows, device=z.device, dtype=torch.float32)
    _load_esm32("esm_bias_s", d)(
        (min(4 * num_sms(d), (rows + 255) // 256), 1, 1), (256, 1, 1), dbias2.contiguous(), km, db, int(N), int(n0)
    )
    return pair_bias_bwd32(z, gamma, wb, db, eps)


def attn_esm32(q, k, v, bias2, lse=None):
    """ESMFold2's atom attention on the fp32 path (``lattn_esm_tf32.cu``, tcgen05 kind::tf32): q, k, v fp32 [A, N, 128] views (unit column
    stride, any row stride; tf32-rounded), bias2 [4, nwin, 32, 160] (``esm_bias``) -> O [A, N, 128] fp32 (and the row LSE, log2 units, into
    ``lse`` [A, 4, N] when given)."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    o = torch.empty(A, N, DM, device=q.device, dtype=torch.float32)
    nq = N // 128
    sp = max(1, min(A, num_sms(d) // (nq * NH)))
    _load_esm32("esm32", d)(
        (nq * NH * sp, 1, 1),
        (512, 1, 1),
        _rows32(q, 128),
        _rows32(k, 256),
        _rows32(v, 256, "128a32"),
        _out32(o),
        bias2,
        lse,
        int(N),
        int(A),
        int(nwin),
        int(sp),
    )
    return o
