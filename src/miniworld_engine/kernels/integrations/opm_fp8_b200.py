"""EXPERIMENTAL B200 (sm_100a) e4m3 product of OuterProductMean inference (opt-in, default OFF; docs/b200/fp8.md "OuterProductMean").

The H100 recipe of ``opm_fp8.py`` unchanged where it is not instruction specific: policy ``bound`` (static power-of-two scales from the weights alone:
|LN(x)_k| <= sqrt(d_msa - 1)|gamma_k| + |beta_k|, so no input can saturate the cast: no calibration, no guard), the product is the only e4m3 stage
(O = sa sb A2q . BTq^T with fp32 accumulation, bf16 O), the count table / epilogue are the default's. What is B200 specific: A2 / BT come out of the B200
bf16 prologue (``opm_sm100.cu`` / ``opm_wide_sm100.cu``, untouched) and one cast launch makes the e4m3 operands (``opm_fp8_b200.cu``); the product is
``torch._scaled_mm`` (cuBLASLt, tcgen05 e4m3). Switch: the H100 module's ``opm_fp8.POLICY`` (``MINIWORLD_OPM_FP8=bound``), so ``fp8.enable`` already
reaches it; depth gate ``MIN_S``.
"""

from __future__ import annotations

import functools
import weakref
from pathlib import Path

import torch

from . import opm_fp8 as _h

_dir = Path(__file__).parent
E4M3 = torch.float8_e4m3fn

#: the fewest padded depths / lengths the B200 route takes (module time, docs/b200/fp8.md): S >= 1024 at every L (L128 S1024 0.89), S 512 from N 256 where the fused bf16 kernel does not serve (0.90-0.95); the fused kernel (S <= 512, N >= 512) and below keep bf16
MIN_S: int | None = None


def _gate(s: int, n: int) -> bool:
    if MIN_S is not None:
        return s >= MIN_S
    return s >= 1024 or (s >= 512 and n >= 256)


_CACHES: dict = {}
STATS = {"calls": 0, "declined": 0}


@functools.lru_cache(maxsize=None)
def _ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="opm_fp8_b200",
        sources=[str(_dir / "csrc" / "sm100" / "opm_fp8_b200.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("100", ptx=("100",))],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def prebuild_items():
    yield _ext


def _scales(args, device):
    """The module's static scales (made on its first eager call, one per weight; None under a capture with no entry: the bf16 route runs)."""
    wa = args[4]
    key = wa.data_ptr()
    c = _CACHES.get(key)
    if c is not None:
        return c
    if torch.cuda.is_current_stream_capturing():
        return None
    lnw, lnb, wb = args[2], args[3], args[5]
    pbias = torch.cat([args[9], args[11]]) if args[9].numel() and len(args) > 11 and args[11].numel() else None
    s = _h.bound_scales(lnw, lnb, wa, wb, pbias).to(device).contiguous()
    qh = (1.0 / s).tolist()
    c = _CACHES[key] = {"qa_host": float(qh[0]), "qb_host": float(qh[1]), "s": s, "q": (1.0 / s).contiguous(), "sa": s[0].clone(), "sb": s[1].clone()}
    weakref.finalize(wa, _CACHES.pop, key, None)
    return c


def serves(args, n: int, s: int, save: bool):
    """The scale entry when the route takes this ``opm_sm100._forward`` call (inference, bf16 activations, ``MINIWORLD_OPM_FP8=bound``, shape gate), else None."""
    if save or _h.POLICY != "bound" or args[0].dtype != torch.bfloat16 or not _gate(s, n):
        return None
    c = _scales(args, args[0].device)
    if c is None:
        STATS["declined"] += 1
        return None
    return c


def product(c, a2, bt, pre8: bool = False):
    """O [(i, c), (j, e)] bf16 = sa sb (A2q . BTq^T). ``pre8``: a2 / bt are already e4m3 (the fused prologue ``opm_prologue_f8``); else the bf16 operands of the prologue (one cast launch)."""
    if pre8:
        a8, b8 = a2, bt
    else:
        a8 = torch.empty(a2.shape, device=a2.device, dtype=E4M3)
        b8 = torch.empty(bt.shape, device=a2.device, dtype=E4M3)
        _ext().opm_cast2(a2, bt, a8, b8, c["q"])
    STATS["calls"] += 1
    _h.GUARD_STATS["calls"] += 1
    return torch._scaled_mm(a8, b8.t(), scale_a=c["sa"], scale_b=c["sb"], out_dtype=torch.bfloat16)
