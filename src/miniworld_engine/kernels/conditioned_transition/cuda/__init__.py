"""CUDA row kernels of the fused token DiT (``token_dit_rows.cu``): the passes between the GEMMs, one block per row.

Same entry points and semantics as the Triton module ``kernels/conditioned_transition/triton/token_dit_kernels.py``
(``adaln_rows``, ``resgate_adaln_rows``, ``gate_rows``, ``swiglu_rows``, ``pair_bias_all``), so the runner can take
either. ``pair_bias_all`` is a CUDA LayerNorm of the pair rows and one cuBLAS GEMM that writes every block's bias
head-major (a fused WMMA LayerNorm + projection kernel measured 3.4-3.6x slower on B200 and was dropped). Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent
STAT_W = 128          # kept for the runner's buffer layout (the Triton module's statistics width)


@functools.lru_cache(maxsize=None)
def _ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="token_dit_rows_cuda",
        sources=[str(_dir / "token_dit_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                           *gencodes("80", "90", "100", ptx=("100",))],
        extra_cflags=["-std=c++17"], verbose=False)


def available() -> bool:
    """The extension builds (a failure is reported once by the caller, which keeps the Triton rows)."""
    _ext()
    return True


def adaln_rows(x, ms, mb, out, L, eps=1e-5):
    """out = LN(x) * sigmoid(ms[row % L]) + mb[row % L]; x fp32 [M, 768], ms / mb per-token tables."""
    _ext().adaln_rows_cuda(x, ms, mb, out, int(L), float(eps))


def resgate_adaln_rows(x, y, gl, ms, mb, out, L, eps=1e-5):
    """x += sigmoid(gl[row % L]) * y in place (fp32); then, when ms is given, out = AdaLN(x) for the next half-block."""
    _ext().resgate_adaln_rows_cuda(x, y, gl, ms, mb, out, int(L), float(eps))


def gate_rows(o, g, out):
    """out = o * sigmoid(g)."""
    _ext().gate_rows_cuda(o, g, out)


def swiglu_rows(ab, out):
    """out = silu(a) * b for ab = [a | b]."""
    _ext().swiglu_rows_cuda(ab, out)


#: ``pair_bias_all`` applies the key mask itself (the runner then skips its own -inf fill).
PAIR_BIAS_MASK = True


def pair_bias_all(z2d, wt, out, L, eps=1e-5, mask=None):
    """Every block's pair bias, head-major: out [NB, L, L] = (LN(z) @ wt)^T with wt [C, NB] (the LayerNorm weights and
    any log2 e scale already folded in), -inf on the columns of masked keys (``mask`` [L] bool). z2d [L L, 128].
    A CUDA LayerNorm and one cuBLAS GEMM (a fused one-pass LayerNorm + projection kernel, one thread per pair row, measured
    slower -- 450 against 266 us for a whole L = 768 block call -- and was dropped)."""
    nb = wt.shape[1]
    m = None if mask is None else mask.reshape(-1).to(torch.bool).contiguous()
    zh = torch.empty(z2d.shape[0], 128, device=z2d.device, dtype=wt.dtype)
    _ext().layernorm128_rows(z2d, zh, float(eps))
    torch.mm(wt.t(), zh.t(), out=out.view(nb, -1))
    if m is not None:
        out.view(nb, L, L)[:, :, ~m] = float("-inf")


__all__ = ["STAT_W", "adaln_rows", "available", "gate_rows", "pair_bias_all", "resgate_adaln_rows", "swiglu_rows"]
