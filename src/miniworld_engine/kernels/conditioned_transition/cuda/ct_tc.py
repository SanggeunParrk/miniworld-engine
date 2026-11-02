"""B200 (sm_100a) fp32 (TF32) ConditionedTransition GEMMs with their elementwise passes in the epilogue (``ct_tc_sm100.cu``):

    swab   [a | b] = xn [Wa; Wb]^T, h = silu(a) b          -> h, ab            (expand GEMM + swiglu_fwd)
    gab    dh = dout Ws, the SwiGLU backward on the saved ab -> dab             (dh GEMM + swiglu_bwd)
    sw     h = silu(xn Wa^T) (xn Wb^T)                       -> h                (inference: expand GEMM + swiglu_fwd)

Operands fp32 on the TF32 grid, K-major; any row count; N a multiple of the tile width. Each epilogue is built at a few tile widths, on single
CTAs and on CTA pairs; ``_pick`` takes the pair where its 256-row tiles still fill the card, else the widest single-CTA tile that does.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._device import num_sms

_dir = Path(__file__).parent
_MODES = {"swab": 0, "gab": 1, "sw": 3}
#: (epilogue, tile width, CTA pair) builds
_BUILDS = (("swab", 128, 1), ("swab", 128, 0), ("swab", 64, 0), ("gab", 256, 1), ("gab", 128, 1), ("gab", 128, 0), ("gab", 64, 0),
           ("sw", 128, 1), ("sw", 128, 0))
NEPI = 8


@functools.lru_cache(maxsize=1)
def _load():
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import _ext, build_cubins

    ext = _ext()
    specs = tuple((f"{m}{bn}{'x2' if c2 else ''}", str(_dir / "ct_tc_sm100.cu"), (f"-DMODE={_MODES[m]}", f"-DBN={bn}", f"-DCTA2={c2}",
                                                                                   f"-DNEPI={NEPI}"))
                  for m, bn, c2 in _BUILDS)
    for name, path in build_cubins("ct_tc", specs).items():
        ext.load_cubin(f"ct_tc_{name}", path)
    return ext


def available() -> bool:
    try:
        _load()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"ConditionedTransition TF32 tcgen05 kernels failed to build or load on B200: {exc!r}") from exc
    return True


def _pick(mode: str, m: int, n: int, nsm: int, force=None) -> tuple[int, bool]:
    """(tile width, CTA pair): the pair's widest tile while its tiles fill every SM pair, else the widest single-CTA tile that gives every
    SM a tile, else the narrowest."""
    if force is not None:
        return force
    builds = [(bn, bool(c2)) for md, bn, c2 in _BUILDS if md == mode and n % bn == 0]
    for bn, c2 in sorted((b for b in builds if b[1]), reverse=True):
        if -(-m // 256) * (n // bn) >= nsm // 2:
            return bn, True
    singles = sorted((b for b in builds if not b[1]), reverse=True)
    for bn, c2 in singles:
        if -(-m // 128) * (n // bn) >= nsm:
            return bn, False
    return singles[-1]


def gemm(mode: str, m: int, n: int, a0: torch.Tensor, b0: torch.Tensor, o0: torch.Tensor, *, o1=None, o2=None, a1=None, b1=None, aux=None,
         bias=None, k0: int, k1: int = 0, ld0: int = 0, ld1: int = 0, ldaux: int = 0, hoff: int = 0, force=None) -> None:
    """C[m, n] by the ``mode`` epilogue (see the module docstring). a* [m, K] / b* [n, K] fp32 K-major, outputs fp32 row-major (row strides
    ``ld*``)."""
    ext = _load()
    nsm = num_sms(a0.device)
    bn, pair = _pick(mode, m, n, nsm, force)
    ncta = 2 if pair else 1
    bnl = bn // ncta
    t = ext.tmap
    ma0 = t(a0, k0, m, 32, 128)
    ma1 = t(a1, k1, m, 32, 128) if a1 is not None else ma0
    mb0 = t(b0, b0.shape[1], b0.shape[0], 32, bnl)
    mb1 = t(b1, b1.shape[1], b1.shape[0], 32, bnl) if b1 is not None else mb0
    tiles = -(-m // (128 * ncta)) * (n // bn)
    ext.launch_kernel(f"ct_tc_{mode}{bn}{'x2' if pair else ''}", "ctc_sm100", ncta * max(1, min(nsm // ncta, tiles)), 64 + 32 * NEPI, ncta,
                      [ma0, ma1, mb0, mb1, o0, o1 if o1 is not None else o0, o2 if o2 is not None else o0, aux if aux is not None else o0,
                       bias if bias is not None else o0, m, n, k0, k1, ld0, ld1, ldaux, hoff])
