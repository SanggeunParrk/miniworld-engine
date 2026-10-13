"""CUDA row kernels of the fused token DiT (``token_dit_rows.cu``): the passes between the GEMMs, one block per row.

Same entry points and semantics as the Triton module ``kernels/conditioned_transition/triton/token_dit_kernels.py``
(``adaln_rows``, ``resgate_adaln_rows``, ``gate_rows``, ``swiglu_rows``, ``pair_bias_all``; ``qknorm_rows`` has no
Triton twin: QK-norm on the fused step is B200-only), so the runner can take
either. A100 BF16 / 16-head ``pair_bias_all`` uses the fused APB tensor-core producer, including the key mask.
Other configurations use a CUDA LayerNorm of the pair rows and one cuBLAS GEMM that writes every block's bias
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


@functools.lru_cache(maxsize=1)
def _pack_ext():
    from ..._nvcc import ensure_cuda_home, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="token_dit_pack_sm80", sources=[str(_dir / "token_dit_pack.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "--fmad=false",
                           "-gencode=arch=compute_80,code=sm_80"],
        extra_cflags=["-O3", "-std=c++17"], verbose=False)


def pack_token(block, qscale, bscale):
    from miniworld_engine.integrations.token_dit import WEIGHTS

    weights = []
    for path in WEIGHTS:
        node = block
        for name in path.split("."):
            node = getattr(node, name)
        weights.append(node.contiguous())
    return _pack_ext().pack_token(weights, float(qscale), float(bscale))


def adaln_rows(x, ms, mb, out, L, eps=1e-5):
    """out = LN(x) * sigmoid(ms[row % L]) + mb[row % L]; x fp32 [M, D] (D 768 or 1024), ms / mb per-token tables."""
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


def adaln_in_rows(xin, x, ms, mb, out, L, eps=1e-5):
    """x = xin (the step's input rows, fp32 or bf16, into the fp32 residual [M, 768]) and out = LN(x) * sigmoid(ms) + mb,
    one pass (adaln_rows after a copy, without the copy)."""
    _ext().adaln_in_rows_cuda(xin, x, ms, mb, out, int(L), float(eps))


def resgate_out_rows(x, y, gl, out, L):
    """out = x + sigmoid(gl[row % L]) * y in out's dtype: the block's last residual, straight into the step's output."""
    _ext().resgate_out_rows_cuda(x, y, gl, out, int(L))


def layernorm_rows(z, out, eps=1e-5):
    """out = LN(z) over 384 columns, no affine (the conditioning rows), z / out fp32 or bf16."""
    _ext().layernorm_rows_cuda(z, out, float(eps))


def qknorm_rows(qk, wq, wk, eq, ek, d=768):
    """QK-norm in place: q = qk[:, 0:d], k = qk[:, d:2d] of every row, RMSNorm per head (head dim = len(wq): 48, 32 or 64 at
    d 768, 64 at d 1024) times wq / wk (fp32, any logit scale folded in), eps eq / ek."""
    _ext().qknorm_rows_cuda(qk, wq, wk, float(eq), float(ek), int(d))


#: ``pair_bias_all`` applies the key mask itself (the runner then skips its own -inf fill).
PAIR_BIAS_MASK = True


def pair_bias_all(z2d, wt, out, L, eps=1e-5, mask=None):
    """Every block's pair bias, head-major: out [NB, L, L] = (LN(z) @ wt)^T with wt [C, NB] (the LayerNorm weights and
    any log2 e scale already folded in), -inf on the columns of masked keys (``mask`` [L] bool). z2d [L L, 128].
    A100 BF16 / 16 heads uses the tiled APB CUDA producer. Other cases retain CUDA LayerNorm + cuBLAS;
    the older one-thread-per-row fused experiment was slower and is not used."""
    nb = wt.shape[1]
    m = None if mask is None else mask.reshape(-1).to(torch.bool).contiguous()
    from miniworld_engine import settings
    if (z2d.dtype is torch.bfloat16 and wt.dtype is torch.bfloat16 and nb == 16
            and L % 128 == 0 and torch.cuda.get_device_capability(z2d.device) == (8, 0)
            and settings.current().token_dit_prep):
        from miniworld_engine.kernels.augmented_attention.cuda import apb

        apb.ext().pair_bias_fwd(z2d, wt.t().contiguous(), m, out, L, float(eps), float("-inf"))
        return
    zh = torch.empty(z2d.shape[0], 128, device=z2d.device, dtype=wt.dtype)
    _ext().layernorm128_rows(z2d, zh, float(eps))
    torch.mm(wt.t(), zh.t(), out=out.view(nb, -1))
    if m is not None:
        out.view(nb, L, L)[:, :, ~m] = float("-inf")


__all__ = ["STAT_W", "adaln_rows", "available", "gate_rows", "pair_bias_all", "resgate_adaln_rows", "swiglu_rows"]
