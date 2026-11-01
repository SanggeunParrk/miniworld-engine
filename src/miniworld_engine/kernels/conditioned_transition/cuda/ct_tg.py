"""B200 (sm_100a) bf16 tcgen05 GEMMs of the ConditionedTransition tail: the generic persistent GEMM of the Transition family
(``kernels/transition/cuda/sm100/generic/tg_gemm.cu``) built in this family's own cubin group, so that the CTA-pair builds and the
epilogues the Transition routing never takes exist without changing what the Transition builds or launches:

    swiglu   h = silu(A Wa^T) * (A Wb^T)                                   (MODE 0)
    ctgate   y = res + sigmoid(cond Wsc^T + bsc) * (h Ws^T)                (MODE 4, two operand phases)
    gate7    [dA | dB] of dh = dout Ws and the saved [a | b]               (MODE 7: the SwiGLU backward in the epilogue, h not stored)

``pair``: a ``cta_group::2`` build (256-row products, each CTA of the pair bringing half of the B tile). Launched through the sm_100a host
binding of the Transition kernels (``fused_sm100a._ext``: tensor maps, cubin registry). bf16 operands only (K-major, TMA 128-B swizzle).
"""

from __future__ import annotations

import functools

import torch

from miniworld_engine.kernels._device import num_sms

MODES = {"swiglu": 0, "ctgate": 4, "swiglu_ab": 5, "gate7": 7}


@functools.lru_cache(maxsize=None)
def _cubin(name: str, bn: int, pair: bool):
    """(host binding, registered cubin name) of one build, compiled on first use."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import _ext as sm100_ext, build_cubins

    ext = sm100_ext()
    key = f"{name}{bn}{'x2' if pair else ''}"
    flags = (f"-DMODE={MODES[name]}", "-DF32=0", f"-DBN={bn}") + (("-DCTA2=1",) if pair else ())
    for k, path in build_cubins("ct_bf16", ((key, "generic/tg_gemm.cu", flags),)).items():
        ext.load_cubin(f"ct_bf16_{k}", path)
    return ext, f"ct_bf16_{key}"


def is_b200(t: torch.Tensor) -> bool:
    return t.is_cuda and torch.cuda.get_device_capability(t.device) == (10, 0)


def gemm(name: str, bn: int, pair: bool, m: int, n: int, a0: torch.Tensor, b0: torch.Tensor, o0: torch.Tensor, *, k0: int,
         a1: torch.Tensor | None = None, b1: torch.Tensor | None = None, o1: torch.Tensor | None = None, res: torch.Tensor | None = None,
         bias: torch.Tensor | None = None, k1: int = 0, ldres: int = 0, hoff: int = 0) -> None:
    """C[m, n] by the ``name`` epilogue (see the module docstring); a* [m, K] and b* [n, K] K-major bf16, outputs bf16. ``n`` a multiple
    of ``bn``; any ``m``. gate7: ``res`` is the saved [a | b] and ``o1`` receives [dA | dB] (b / dB ``hoff`` columns after a / dA)."""
    ext, cub = _cubin(name, bn, pair)
    t = ext.tmap
    bb = bn // 2 if pair else bn
    ma0 = t(a0, k0, m, 64, 128)
    ma1 = t(a1, k1, m, 64, 128) if a1 is not None else ma0
    mb0 = t(b0, b0.shape[1], b0.shape[0], 64, bb)
    mb1 = t(b1, b1.shape[1], b1.shape[0], 64, bb) if b1 is not None else mb0
    if name == "gate7":
        mb1 = t(res, res.shape[1], res.shape[0], 64, 32)
    mo0 = t(o0, o0.shape[1], o0.shape[0], 64, 32)
    mo1 = t(o1, o1.shape[1], o1.shape[0], 64, 32) if o1 is not None else mo0
    args = [ma0, ma1, mb0, mb1, mb0, mo0, mo1, res if res is not None else o0, bias if bias is not None else o0, m, n, k0, k1, ldres,
            hoff, 1, 0]
    nsm = num_sms(a0.device)
    if pair:
        tiles = -(-m // 256) * (n // bn)
        ext.launch_kernel(cub, "tg_gemm_sm100", 2 * max(1, min(nsm // 2, tiles)), 320, 2, args)
    else:
        tiles = -(-m // 128) * (n // bn)
        ext.launch_kernel(cub, "tg_gemm_sm100", max(1, min(nsm, tiles)), 320, 1, args)


__all__ = ["gemm", "is_b200"]
