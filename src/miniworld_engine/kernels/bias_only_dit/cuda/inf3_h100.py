"""The H100 (sm_90a) bf16 inference step as three kernels per block: ``bo_inf3_sm90.cu`` (front, tail) around ``pvx`` launched with
programmatic dependent launch (``pv_gate_inf.cu``'s ``pv_cta_pdl``) -- the Hopper twin of B200's ``inf3_bf16.py``.

  front  LN + AdaLN of the block input + the v|g GEMM            a cluster of CL CTAs per 128-row tile (wgmma, TMA multicast)
  core   sigmoid(g) (P v)                                         ``pvx``, PDL
  tail   out GEMM, residual + gate, LN + AdaLN, a|b GEMM, SwiGLU, squeeze GEMM, residual + gate      the same cluster geometry

The conditioning tables are the 12-launch step's (LayerNorm rows + two cuBLAS GEMMs per call, raw: the kernels apply the sigmoids).
The extension builds on first use (``ready``); a failed build warns once and the runner keeps the 12-launch step.
"""

from __future__ import annotations

import functools
import warnings

import torch

from miniworld_engine.kernels._device import num_sms

from miniworld_engine import settings
from miniworld_engine.kernels.bias_only_dit.cuda import _dir, h100_ext

#: Cluster of the front and the tail: None = ``cluster`` (the round model); 8 or 16 forces one (A/B).
_FORCE_CL: int | None = None
#: Programmatic dependent launch of front / core / tail (False: plain launches; A/B).
_PDL = True


@functools.lru_cache(maxsize=None)
def ext():
    from miniworld_engine.kernels._nvcc import (
        ensure_cuda_home,
        gencodes,
        host_flags,
        load_extension,
    )

    ensure_cuda_home()
    k = _dir.parent.parent
    return load_extension(
        name="bias_only_dit_inf3_sm90a",
        sources=[str(_dir / "bo_inf3_sm90.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            *gencodes("90a"),
            "--expt-relaxed-constexpr",
            f"-I{k / 'transition' / 'cuda' / 'anthropic_v5'}",
            f"-I{k / 'transition' / 'cuda' / 'wide' / 'kernels'}",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


_FAILED: list = []


def ready() -> bool:
    """The extension (and the PDL core) load; a failure warns once and the runner keeps the 12-launch bf16 step."""
    if _FAILED:
        return False
    try:
        ext()
        h100_ext()
        return True
    except (
        Exception
    ) as err:  # a build failure must not break inference: the 12-launch step serves
        _FAILED.append(err)
        warnings.warn(
            f"bias-only DiT H100 three-kernel step unavailable ({type(err).__name__}: {err}); keeping the 12-launch step"
        )
        return False


#: The H100 three-kernel step serves only when this is True (off: measured slower than the 12-launch step at every shape, see
#: docs/gpus/h100/bias_only_dit/bias_only_dit.md); settings.bias_only_dit_inf3_bf16=0 also turns it off. Kept for A/B.
_ON = False


def on() -> bool:
    """The H100 three-kernel step is on: ``_ON`` and settings.bias_only_dit_inf3_bf16 (read per call)."""
    return _ON and settings.current().bias_only_dit_inf3_bf16


@functools.lru_cache(maxsize=None)
def _max_clusters(device_index: int, cl: int) -> int:
    with torch.cuda.device(device_index):
        return min(int(ext().max_clusters(0, cl)), int(ext().max_clusters(1, cl)))


def cluster(M: int, device_index: int) -> int:
    """The front's and the tail's cluster: 16 CTAs per 128-row tile (half the weight rows per CTA) while the tiles fit in one round of
    16-clusters, else 8."""
    if _FORCE_CL:
        return int(_FORCE_CL)
    tiles = M // 128
    n16 = _max_clusters(device_index, 16)
    return 16 if 0 < tiles <= n16 else 8


class Inf3H100:
    """front / core / tail launchers of one device."""

    def __init__(self, device_index: int, nh: int, dh: int):
        self.device_index, self.nh, self.dh = device_index, nh, dh
        self.nsm = num_sms(device_index)
        self.x = ext()
        self.core_ext = h100_ext()

    def front(self, x, g1, w, vg, xa, cl, eps):
        self.x.front(x, g1, w, vg, xa, cl, _PDL, eps)

    def core(self, v, P, out, S, g):
        if _PDL:
            self.core_ext.pv_cta_pdl(v, P, out, S, g, self.nh, self.dh, self.nsm)
        else:
            self.core_ext.pv_cta(v, P, out, S, g, self.nh, self.dh, self.nsm, 0, 0, 0)

    def tail(self, a, x, g1, g2, wo, wab, wsq, xt, h, out, cl, eps):
        self.x.tail(a, x, g1, g2, wo, wab, wsq, xt, h, out, cl, _PDL, eps)
