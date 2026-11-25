"""The bias-only token DiT inference runner's weight pack on B200 in ONE launch (``bo_pack_sm100.cu``; ``runner.FusedBiasOnlyDiT._pack_b200``).

A :class:`Pack` collects segments -- a copy (optionally times a per-column scale: the folded LayerNorm weights), a zero fill, a pair pack
of a [768, K] weight for the tail kernels -- each into a contiguous destination, then issues them as one kernel launch (up to
``MAXSEG`` segments; more launch again). Sources are read in their own dtype (bf16 or fp32), destinations are bf16, fp32, or fp32 rounded
to the nearest TF32 (``ddt=2``); the arithmetic is the torch pack's (one fp32 product, one rounding), so the values are the same bit for bit.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

_dir = Path(__file__).parent
MAXSEG, TPB, VEC, UNR = 48, 256, 4, 4  # (bo_pack_sm100.cu)
PER_BLK = TPB * VEC * UNR
BF, F32 = torch.bfloat16, torch.float32


class _Seg(ctypes.Structure):
    _fields_ = [
        ("src", ctypes.c_void_p),
        ("dst", ctypes.c_void_p),
        ("scale", ctypes.c_void_p),
        ("n", ctypes.c_longlong),
        ("b0", ctypes.c_int),
        ("sdt", ctypes.c_int),
        ("ddt", ctypes.c_int),
        ("scdt", ctypes.c_int),
        ("mode", ctypes.c_int),
        ("period", ctypes.c_int),
        ("K", ctypes.c_int),
        ("cl", ctypes.c_int),
        ("bw", ctypes.c_int),
        ("zo", ctypes.c_int),
        ("cs", ctypes.c_float),
        ("dh", ctypes.c_int),
        ("dhp", ctypes.c_int),
        ("cols", ctypes.c_int),
        ("pad_", ctypes.c_int),
    ]


class _Segs(ctypes.Structure):
    _fields_ = [("s", _Seg * MAXSEG), ("n", ctypes.c_int)]


def _kernel(device_index: int):
    from foldjet_engine.kernels.augmented_attention.cuda import sm100

    return sm100._sm100_kernel("bo_pack_sm100", "bo_pack_sm100", device_index, src_dir=str(_dir), smem=0)


def tab_kernel(device):
    """``bo_tab_sm100.cu``: the inference runner's one-block conditioning table pass (bias + sigmoids in place; ``runner._tab_b200``)."""
    from foldjet_engine.kernels.augmented_attention.cuda import sm100

    idx = device.index if device.index is not None else torch.cuda.current_device()
    return sm100._sm100_kernel("bo_tab_sm100", "bo_tab_sm100", idx, src_dir=str(_dir), smem=0)


def _dt(t: torch.Tensor) -> int:
    assert t.dtype in (BF, F32) and t.is_contiguous(), (t.dtype, t.shape)
    return 1 if t.dtype is F32 else 0


class Pack:
    """Segments of one pack launch; ``dtype`` is the default destination dtype (the runner's)."""

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.segs: list[tuple] = []  # (fields, tensors kept alive)

    def _add(self, dst, mode, src=None, scale=None, ddt=None, K=0, cl=0, bw=0, zo=0, cs=1.0, dh=0, dhp=0, cols=0):
        assert dst.is_contiguous() and dst.numel() % VEC == 0 and dst.numel() < 2**31, (dst.shape, dst.stride())
        for t in (src, dst, scale):  # the kernel's 4-element vector accesses
            assert t is None or t.data_ptr() % (4 * t.element_size()) == 0, "pack segment not 4-element aligned"
        dd = 2 if ddt == 2 else _dt(dst)
        assert dd != 2 or dst.dtype is F32
        f = dict(
            src=0 if src is None else src.data_ptr(),
            dst=dst.data_ptr(),
            scale=0 if scale is None else scale.data_ptr(),
            n=dst.numel(),
            sdt=0 if src is None else _dt(src),
            ddt=dd,
            scdt=0 if scale is None else _dt(scale),
            mode=mode,
            period=0 if scale is None else scale.numel(),
            K=K,
            cl=cl,
            bw=bw,
            zo=zo,
            cs=cs,
            dh=dh,
            dhp=dhp,
            cols=cols,
        )
        self.segs.append((f, (src, dst, scale)))
        return dst

    def copy(self, src, dst, scale=None, ddt=None, cs: float = 1.0):
        """dst = src (x scale per column: ``scale`` [src row length]) x ``cs``; ``ddt=2``: fp32 rounded to the nearest TF32."""
        src = src.detach().contiguous()
        assert src.numel() == dst.numel(), (src.shape, dst.shape)
        if scale is not None:
            scale = scale.detach().contiguous()
            assert src.shape[-1] == scale.numel(), (src.shape, scale.shape)
        return self._add(dst, 0, src, scale, ddt, cs=cs)

    def zero(self, dst, value: float = 0.0):
        """dst = ``value`` everywhere."""
        return self._add(dst, 1, cs=value)

    def pad_rows(self, src, dst, dh: int, dhp: int, cs: float = 1.0):
        """dst [H dhp, C] = src [H dh, C] with zero rows after each head's dh, x ``cs``."""
        src = src.detach().contiguous()
        C = dst.shape[-1]
        assert src.numel() * dhp == dst.numel() * dh and C % VEC == 0, (src.shape, dst.shape, dh, dhp)
        return self._add(dst, 3, src, cs=cs, dh=dh, dhp=dhp, cols=C)

    def pad_cols(self, src, dst, dh: int, dhp: int, cs: float = 1.0):
        """dst [R, H dhp] = src [R, H dh] with zero columns after each head's dh, x ``cs`` (dh a multiple of 4)."""
        src = src.detach().contiguous()
        W = dst.shape[-1]
        assert src.numel() * dhp == dst.numel() * dh and dh % VEC == 0 and W % dhp == 0, (src.shape, dst.shape, dh, dhp)
        return self._add(dst, 4, src, cs=cs, dh=dh, dhp=dhp, cols=W)

    def pairs(self, w, cl: int, bw: int, zo: int = 0, ddt=None):
        """The pair-packed form of ``w`` [768, K] for cluster ``cl``, k-blocks of ``bw`` columns (``zo``: each CTA's z order), a new tensor."""
        w = w.detach().contiguous()
        n, K = w.shape
        assert n == 768 and K % (2 * bw) == 0 and 768 % cl == 0, (w.shape, cl, bw)
        dst = torch.empty(n * K // bw, bw, device=w.device, dtype=F32 if ddt == 2 else self.dtype)
        return self._add(dst, 2, w, None, ddt, K=K, cl=cl, bw=bw, zo=zo)

    def launch(self, device) -> None:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        k = _kernel(idx)
        for i in range(0, len(self.segs), MAXSEG):
            part = self.segs[i : i + MAXSEG]
            segs, b = _Segs(), 0
            for j, (f, _) in enumerate(part):
                s = segs.s[j]
                for name, v in f.items():
                    setattr(s, name, v)
                s.b0 = b
                b += max(1, -(-f["n"] // PER_BLK))
            segs.n = len(part)
            k((b, 1, 1), (TPB, 1, 1), segs)
        self.keep = [t for _, ts in self.segs for t in ts if t is not None]  # the sources / destinations stay alive with the pack
