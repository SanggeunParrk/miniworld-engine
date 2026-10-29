"""The token DiT blocks' GEMMs with their row passes in the epilogue on B200 (``dit_gemm_sm100.cu``, sm_100a, bf16 operands):

* :func:`glu_gemm` -- h = silu(a) b [u] for [a | b | u] = x [Wa; Wb; Wu]^T: the transition's expand GEMM with the GLU (SwiGLU, or
  Boltz-2's up-gated GLU) in its epilogue. 2-CTA persistent, (256-row tile pair, 128-unit (up-gate: 64-unit) chunk) items.
* :func:`res_gemm` -- x += g1 (a1 W1^T) [+ g2 (a2 W2^T)], then the stack's bf16 output or x written back and the next part's AdaLN
  (LN(x) f(ms) + mb, one or two tables): the to_out / squeeze GEMMs with the residual + gate pass and the following AdaLN in the
  epilogue. One 8-CTA cluster per 128-row tile (96 columns a CTA); the row's LayerNorm statistics meet over distributed shared
  memory.

Both serve rows of D = 768 (the token DiT width) and any M; tables [L, *] bf16 indexed by row % L (the conditioning shared by the
samples, or L = M for per-row tables). Launch bindings (TMA descriptors) are cached by the addresses and geometry they encode, without
holding the tensors. Graph-capturable (driver launches on torch's current stream), programmatic dependent launch (``PDL``).
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._device import num_sms

_dir = str(Path(__file__).parent)
D = 768
_THREADS = (384, 1, 1)


def _index(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def available(t: torch.Tensor) -> bool:
    return t.is_cuda and torch.cuda.get_device_capability(_index(t)) == (10, 0)


def _rows_ok(t: torch.Tensor) -> bool:
    """bf16 rows with unit column stride, 16-byte aligned rows and start (a TMA operand / a vector load)."""
    return (t.dtype is torch.bfloat16 and t.dim() == 2 and t.stride(1) == 1 and (t.stride(0) * 2) % 16 == 0
            and t.data_ptr() % 16 == 0)


def _tm(t, dims, stride_bytes, box, **kw):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    m = sm100._tm(t, dims, stride_bytes, box, **kw)
    m.keep = None   # cached by address: the descriptor must not pin the tensor
    return m


# ---------------------------------------------------------------------------------------------------------------- GLU
def _glu_geometry(nops: int, units: int) -> tuple[int, int]:
    n = nops * units
    stage = 128 * 128 + (n // 2) * 128
    nst = 5 if n == 256 else 6
    return nst, nst * stage + 2 * 128 * units * 2 + 256


#: programmatic dependent launch between consecutive kernels of a step: a kernel's prologue (barriers, tensor-memory allocation, the
#: cluster barrier, ~1 us) overlaps the previous kernel's tail; every kernel waits (griddepcontrol.wait) before it reads an operand
PDL = True


@functools.lru_cache(maxsize=16)
def _glu_kernel(device_index: int, nops: int, units: int, k: int, h: int, pdl: bool):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    _, smem = _glu_geometry(nops, units)
    defs = ("GLU", f"NOPS={nops}", f"UNITS={units}", f"KDIM={k}", f"HID={h}", *(("PDL",) if pdl else ()))
    return sm100._sm100_kernel("dit_gemm_sm100", "dit_glu_gemm_sm100", device_index, src_dir=_dir, defs=defs, cluster=2, smem=smem,
                               newest=True, pdl=pdl)


def glu_units(nops: int, m: int) -> int:
    """Hidden units per item: 128 (N = 256 per pair) for the plain SwiGLU, 64 with the up-gate (N = 192)."""
    return 64 if nops == 3 else 128


def glu_supported(x: torch.Tensor, w: torch.Tensor, nops: int) -> bool:
    """bf16 on sm_100, K a multiple of 64, the hidden width a multiple of the item's units; x rows TMA-aligned, w contiguous."""
    if nops not in (2, 3) or not available(x) or not _rows_ok(x) or w.dtype is not torch.bfloat16 or not w.is_contiguous():
        return False
    k = x.shape[1]
    h = w.shape[0] // nops
    return w.shape[1] == k and k % 64 == 0 and w.shape[0] == nops * h and h % glu_units(nops, x.shape[0]) == 0


_GLU_MAPS: dict = {}


@torch.compiler.disable
def glu_gemm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, nops: int) -> torch.Tensor:
    """out [M, H] = bf16(silu(a) b [u]) for [a | b | u] = x w^T, w = [Wa; Wb (; Wu)] ([nops H, K] bf16, contiguous)."""
    m, k = x.shape
    h = w.shape[0] // nops
    units = glu_units(nops, m)
    assert glu_supported(x, w, nops) and out.shape == (m, h) and out.is_contiguous() and out.dtype is torch.bfloat16
    idx = _index(x)
    kern = _glu_kernel(idx, nops, units, k, h, PDL)
    key = (idx, x.data_ptr(), tuple(x.shape), x.stride(0), w.data_ptr(), tuple(w.shape), out.data_ptr(), nops)
    maps = _GLU_MAPS.get(key)
    if maps is None:
        if len(_GLU_MAPS) >= 512:
            _GLU_MAPS.clear()
        maps = _GLU_MAPS[key] = (
            _tm(x, [k, m], x.stride(0) * 2, [64, 128]),
            _tm(w, [k, nops * h], k * 2, [64, 32]),
            _tm(out, [h, m], h * 2, [64, 128]),
        )
    items = (m + 255) // 256 * (h // units)
    grid = (min(num_sms(idx) & ~1, 2 * items), 1, 1)
    kern(grid, _THREADS, *maps, int(m))
    return out


# ---------------------------------------------------------------------------------------------------------------- RES
class _ResArgs(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint64), ("g1", ctypes.c_uint64), ("g2", ctypes.c_uint64),
        ("ms0", ctypes.c_uint64), ("ms1", ctypes.c_uint64), ("mb0", ctypes.c_uint64), ("mb1", ctypes.c_uint64),
        ("sg", ctypes.c_int64), ("sms", ctypes.c_int64),
        ("M", ctypes.c_int32), ("L", ctypes.c_int32), ("nout", ctypes.c_int32), ("sig", ctypes.c_int32), ("fin", ctypes.c_int32),
        ("eps", ctypes.c_float), ("sig_g", ctypes.c_int32),
    ]


#: CTAs per 128-row tile (a cluster): each takes 768 / RES_CLUSTER columns
RES_CLUSTER = 8


def _res_smem(cl: int) -> int:
    return 232448      # the kernel sizes its ring to the whole budget (dit_gemm_sm100.cu, RES)


@functools.lru_cache(maxsize=16)
def _res_kernel(device_index: int, k1: int, k2: int, cl: int, pdl: bool):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    defs = ("RES", f"CL={cl}", f"K1={k1}", f"K2={k2}", *(("PDL",) if pdl else ()))
    return sm100._sm100_kernel("dit_gemm_sm100", "dit_res_gemm_sm100", device_index, src_dir=_dir, defs=defs, cluster=cl,
                               smem=_res_smem(cl), newest=True, pdl=pdl)


def _table_ok(t: torch.Tensor) -> bool:
    return t.dtype is torch.bfloat16 and t.dim() == 2 and t.shape[1] == D and t.stride(1) == 1 and t.data_ptr() % 16 == 0 \
        and (t.stride(0) * 2) % 16 == 0


def res_supported(x: torch.Tensor, a1: torch.Tensor, w1: torch.Tensor, a2: torch.Tensor | None = None,
                  w2: torch.Tensor | None = None) -> bool:
    """sm_100, x fp32 [M, 768] contiguous, a bf16 TMA-aligned rows with K a multiple of 64, w bf16 [768, K] contiguous."""
    if not available(x) or x.dtype is not torch.float32 or x.dim() != 2 or x.shape[1] != D or not x.is_contiguous():
        return False
    for a, w in ((a1, w1), *(((a2, w2),) if a2 is not None else ())):
        if not (_rows_ok(a) and w.dtype is torch.bfloat16 and w.is_contiguous() and w.shape == (D, a.shape[1])
                and a.shape[1] % 64 == 0 and a.shape[0] == x.shape[0]):
            return False
    return True


_RES_MAPS: dict = {}
_OUT_MAPS: dict = {}


@torch.compiler.disable
def res_gemm(x: torch.Tensor, a1: torch.Tensor, w1: torch.Tensor, g1: torch.Tensor, L: int, *, a2: torch.Tensor | None = None,
             w2: torch.Tensor | None = None, g2: torch.Tensor | None = None, mods=(), sig: bool = True, eps: float = 1e-5,
             final: torch.Tensor | None = None, sig_g: bool = True) -> None:
    """x += sigmoid(g1) (a1 w1^T) [+ sigmoid(g2) (a2 w2^T)]; then ``final`` = bf16(x) (x not written back) or x written back and, for
    each (ms, mb, out) of ``mods`` (at most two), out = LN(x) f(ms) + mb with f = sigmoid when ``sig``. g / ms / mb: bf16 [L, 768]
    table views indexed by row % L (the g's share one row stride, the ms / mb another). ``sig_g`` False: the g tables already hold
    sigmoid values (per-token tables: L sigmoids instead of M in the epilogue), and likewise ``sig`` for the scales."""
    m = x.shape[0]
    assert res_supported(x, a1, w1, a2, w2), "res_gemm: unsupported operands"
    assert len(mods) <= 2 and (final is None or not mods)
    tables = [g1, *((g2,) if a2 is not None else ())]
    assert all(_table_ok(t) and t.stride(0) == g1.stride(0) for t in tables)
    sms = mods[0][0].stride(0) if mods else 0
    for ms, mb, out in mods:
        assert _table_ok(ms) and _table_ok(mb) and ms.stride(0) == sms and mb.stride(0) == sms
        assert out.dtype is torch.bfloat16 and out.shape == (m, D) and out.is_contiguous()
    if final is not None:
        assert final.dtype is torch.bfloat16 and final.shape == (m, D) and final.is_contiguous()
    idx = _index(x)
    k1, k2 = a1.shape[1], (a2.shape[1] if a2 is not None else 0)
    cl = RES_CLUSTER
    nt = D // cl
    kern = _res_kernel(idx, k1, k2, cl, PDL)
    key = (idx, cl, a1.data_ptr(), tuple(a1.shape), a1.stride(0), w1.data_ptr(), k1,
           None if a2 is None else (a2.data_ptr(), tuple(a2.shape), a2.stride(0), w2.data_ptr(), k2))
    maps = _RES_MAPS.get(key)
    if maps is None:
        if len(_RES_MAPS) >= 512:
            _RES_MAPS.clear()
        ma1, mw1 = _tm(a1, [k1, m], a1.stride(0) * 2, [64, 16]), _tm(w1, [k1, D], k1 * 2, [64, nt])
        if a2 is not None:
            ma2, mw2 = _tm(a2, [k2, m], a2.stride(0) * 2, [64, 16]), _tm(w2, [k2, D], k2 * 2, [64, nt])
        else:
            ma2, mw2 = ma1, mw1
        maps = _RES_MAPS[key] = (ma1, mw1, ma2, mw2)
    outs = [final] if final is not None else [t[2] for t in mods]
    tabs = [g1, *((g2,) if a2 is not None else ()), *(t for mod in mods for t in mod[:2])]
    okey = (idx, x.data_ptr(), m, *(o.data_ptr() for o in outs), *((t.data_ptr(), t.shape[0], t.stride(0)) for t in tabs))
    omaps = _OUT_MAPS.get(okey)
    if omaps is None:
        if len(_OUT_MAPS) >= 512:
            _OUT_MAPS.clear()
        mx = _tm(x, [D, m], D * 4, [32, 128], swizzle=128, dtype="f32")
        mo = [_tm(o, [D, m], D * 2, [32, 128], swizzle=64) for o in outs]
        tm = {id(t): _tm(t, [D, t.shape[0]], t.stride(0) * 2, [32, 128], swizzle=64) for t in tabs}
        mg = [tm[id(g1)], tm[id(g2)] if a2 is not None else mx]
        mt = [tm[id(t)] for mod in mods for t in mod[:2]]
        omaps = _OUT_MAPS[okey] = (mx, *(mo + [mx] * (2 - len(mo))), *mg, *(mt + [mx] * (4 - len(mt))))
    ms = [t[0].data_ptr() for t in mods] + [0] * (2 - len(mods))
    mb = [t[1].data_ptr() for t in mods] + [0] * (2 - len(mods))
    args = _ResArgs(x.data_ptr(), g1.data_ptr(), g2.data_ptr() if a2 is not None else 0, ms[0], ms[1], mb[0], mb[1], g1.stride(0), sms,
                    m, int(L), len(mods), int(bool(sig)), int(final is not None), float(eps), int(bool(sig_g)))
    kern((cl * ((m + 127) // 128), 1, 1), _THREADS, *maps, *omaps, args)


__all__ = ["available", "glu_gemm", "glu_supported", "res_gemm", "res_supported"]
