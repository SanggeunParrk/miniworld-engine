"""The token DiT training block's few-row GEMMs with the neighbouring row pass in the epilogue, on sm_100a (B200):
``tdt_epi_sm100.cu`` (tcgen05, one 128 x BN tile per CTA, a cluster of CN CTAs multicasting its shared A tile, PDL with the weight
boxes loaded before the wait).

    res_c(h, Wsq, x1, Gg, bg2, z, out)        z = h Wsq^T; out = x1 + sigmoid(Gg[:, D:2D] + bg2) z      (forward: + res_c)
    swiglu(xt, Wab, ab, h)                    [a | b] = xt Wab^T; h = silu(a) b                         (forward: + swiglu_rows)
    swiglu_bwd(dz, Wsq, ab, dab, h)           dh = dz Wsq; dab, h as ``swiglu_bwd``                     (backward: + swiglu_bwd)
    mm(a, w, trb=0)                           the plain GEMM (tests)

At one sample (M = 128 - 1024 rows) the training step is a chain of ~30 dependent kernels of 2-9 us; each of these removes one row
pass launch from the chain. ``supported`` is the whole gate; the callers keep cuBLAS + the row kernel elsewhere.

Routed (round 3 part 6, docs/b200/token-dit.md): ``mm`` only -- the training backward's dchat = dG Wn up to 256 rows (one launch for
cuBLAS's split K + reduce). The three epilogues measured mixed at the step level (res_c: -2 % to +10 %, the 1024-row step slower;
swiglu / swiglu_bwd 0.4-3 us slower than cuBLAS + the row pass at 128 - 1024 rows) and are not routed."""

from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path

import torch

_dir = str(Path(__file__).parent)
BM, BK = 128, 64
#: rows up to which the fused GEMMs serve (above it cuBLAS's tiles fill the card)
MAX_M = int(os.environ.get("MINIWORLD_TDT_EPI_MAX_M", "1024"))
ENABLED = os.environ.get("MINIWORLD_TDT_EPI", "1") != "0"
#: per-epilogue switches (``MINIWORLD_TDT_EPI_OFF=resc,swf,swb``)
_OFF = set(filter(None, os.environ.get("MINIWORLD_TDT_EPI_OFF", "").split(",")))


class _Args(ctypes.Structure):
    _fields_ = [
        ("out", ctypes.c_void_p),
        ("ldo", ctypes.c_long),
        ("aux", ctypes.c_void_p),
        ("lda", ctypes.c_long),
        ("in0", ctypes.c_void_p),
        ("ld0", ctypes.c_long),
        ("in1", ctypes.c_void_p),
        ("ld1", ctypes.c_long),
        ("vec", ctypes.c_void_p),
        ("nh", ctypes.c_int),
    ]


def _cfg(name: str, default: int) -> int:
    return int(os.environ.get(f"MINIWORLD_TDT_EPI_{name}", str(default)))


def _smem(bn: int, nst: int) -> int:
    return 1024 + nst * (BM * BK * 2 + bn * BK * 2) + BM * (bn + 4) * 4 + 256


def _nst(bn: int) -> int:
    return min(6, (232448 - 2048 - BM * (bn + 4) * 4) // (BM * BK * 2 + bn * BK * 2))


@functools.lru_cache(maxsize=None)
def _kernel(device_index: int, bn: int, trb: int, ks: int, epi: int, nst: int, outf32: int, out_bf16: int):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    defs = (f"BN={bn}", f"TRB={trb}", f"NST={nst}", f"EPI={epi}", "PDL=1", f"OUTF32={outf32}", f"OUT_BF16={out_bf16}")
    return sm100._sm100_kernel(
        "tdt_epi_sm100", "tdt_epi_sm100", device_index, pdl=True, src_dir=_dir, defs=defs, cluster=ks, smem=_smem(bn, nst)
    )


def _index(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def _ok2(t: torch.Tensor) -> bool:
    return t.dim() == 2 and t.stride(1) == 1 and (t.stride(0) * t.element_size()) % 16 == 0 and t.data_ptr() % 16 == 0


def supported(kind: str, a: torch.Tensor) -> bool:
    """bf16 rows on sm_100, M a multiple of 128 and <= MAX_M; ``kind`` one of resc / swf / swb."""
    if not ENABLED or kind in _OFF or not (a.is_cuda and a.dtype is torch.bfloat16 and a.dim() == 2):
        return False
    if torch.cuda.get_device_capability(_index(a)) != (10, 0):
        return False
    M, K = a.shape
    return 0 < M <= MAX_M and M % BM == 0 and K % BK == 0 and _ok2(a)


def _ks(tiles: int, nk: int, nsm: int) -> int:
    """the largest power-of-two K split (<= MAX_KS, <= the K-block count) that keeps every CTA of the grid resident at once"""
    ks = 1
    while ks * 2 <= _cfg("MAX_KS", 16) and ks * 2 <= nk and tiles * ks * 2 <= nsm:
        ks *= 2
    return ks


def _launch(a, w, trb, bn, ks, epi, N, args, outf32=0, out_bf16=1, nst=None):
    from miniworld_engine.kernels._device import num_sms
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    M, K = a.shape
    if nst is None:
        nst = _nst(bn)
    ma = sm100._tm(a, [K, M], a.stride(0) * 2, [BK, BM])
    if trb:
        mb = sm100._tm(w, [w.shape[1], K], w.stride(0) * 2, [64, BK])
    elif epi == 3:
        mb = sm100._tm(w, [K, w.shape[0]], w.stride(0) * 2, [BK, bn // 2])
    else:
        mb = sm100._tm(w, [K, w.shape[0]], w.stride(0) * 2, [BK, bn])
    tiles = (N // (bn // 2 if epi == 3 else bn)) * (M // BM)
    dev = _index(a)
    if ks is None:
        ks = _ks(tiles, K // BK, num_sms(dev))
    k = _kernel(dev, bn, int(trb), ks, epi, nst, outf32, out_bf16)
    k((tiles // (M // BM) * ks, M // BM, 1), (192, 1, 1), ma, mb, M, K, ks, args)


def _ptr(t):
    return None if t is None else t.data_ptr()


def _pick_bn(N: int, M: int, half: bool = False) -> tuple[int, int | None]:
    """(BN, KS) for an N-wide output: env overrides (KS 0: automatic), else BN 64 (EPI 3: 64, 32 columns of a and of b)"""
    bn = _cfg("BN_SWF" if half else "BN", 64)
    ks = _cfg("KS", 0)
    return bn, (ks or None)


def res_c(h, wsq, x1, gg, bg2, z, out):
    """z = h wsq^T (bf16), out = x1 + sigmoid(gg[:, D:2D] + bg2) z (bf16 or fp32 out)."""
    M = h.shape[0]
    D = wsq.shape[0]
    bn, ks = _pick_bn(D, M)
    assert x1.dtype is torch.float32 and z.dtype is torch.bfloat16 and gg.dtype is torch.bfloat16
    g2 = gg[:, D : 2 * D]
    args = _Args(_ptr(out), out.stride(0), _ptr(z), z.stride(0), _ptr(x1), x1.stride(0), _ptr(g2), gg.stride(0), _ptr(bg2), 0)
    _launch(h, wsq, 0, bn, ks, 1, D, args, out_bf16=int(out.dtype is torch.bfloat16))


def swiglu(xt, wab, ab, h):
    """[a | b] = xt wab^T (bf16 into ab), h = silu(a) b."""
    M = xt.shape[0]
    nh = wab.shape[0] // 2
    bn, ks = _pick_bn(nh, M, half=True)
    args = _Args(_ptr(ab), ab.stride(0), _ptr(h), h.stride(0), None, 0, None, 0, None, nh)
    _launch(xt, wab, 0, bn, ks, 3, nh, args)


def swiglu_bwd(dz, wsq, ab, dab, h):
    """dh = dz wsq (bf16), dab = d[a | b], h = silu(a) b (as ``swiglu_bwd``)."""
    M = dz.shape[0]
    nh = wsq.shape[1]
    bn, ks = _pick_bn(nh, M)
    args = _Args(_ptr(dab), dab.stride(0), _ptr(h), h.stride(0), _ptr(ab), ab.stride(0), None, 0, None, nh)
    _launch(dz, wsq, 1, bn, ks, 2, nh, args)


def mm(a, w, trb=0, bias=None, out_dtype=torch.bfloat16, bn=64, ks=None, nst=None, out=None):
    M = a.shape[0]
    N = w.shape[1] if trb else w.shape[0]
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=out_dtype)
    args = _Args(_ptr(out), out.stride(0), None, 0, _ptr(bias), 0, None, 0, None, 0)
    _launch(a, w, trb, bn, ks, 0, N, args, outf32=int(out.dtype is torch.float32), nst=nst)
    return out


__all__ = ["mm", "res_c", "supported", "swiglu", "swiglu_bwd"]
