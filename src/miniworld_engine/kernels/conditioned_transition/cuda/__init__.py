"""CUDA row kernels of the fused token DiT (``token_dit_rows.cu``): the passes between the GEMMs, one block per row.

Same entry points and semantics as the Triton module ``kernels/conditioned_transition/triton/token_dit_kernels.py``
(``adaln_rows``, ``resgate_adaln_rows``, ``gate_rows``, ``swiglu_rows``, ``pair_bias_all``; ``qknorm_rows`` has no
Triton twin: QK-norm on the fused step needs CUDA rows on H100 or B200), so the runner can take
either. ``pair_bias_all`` is a CUDA LayerNorm of the pair rows and one cuBLAS GEMM that writes every block's bias
head-major (a fused WMMA LayerNorm + projection kernel measured 3.4-3.6x slower on B200 and was dropped). Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_dir = Path(__file__).parent
STAT_W = 128          # kept for the runner's buffer layout (the Triton module's statistics width)


def _ext():
    arch = "90a" if torch.cuda.get_device_capability() == (9, 0) else "100"
    return _arch_ext(arch)


@functools.lru_cache(maxsize=None)
def _arch_ext(arch):
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name=f"token_dit_rows_cuda_sm{arch}",
        sources=[str(_dir / "token_dit_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                           *gencodes(arch, ptx=() if arch == "90a" else ("100",))],
        extra_cflags=["-std=c++17"], verbose=False)


def available() -> bool:
    """The extension builds (a failure is reported once by the caller, which keeps the Triton rows)."""
    _ext()
    return True


def token_pack(weights, heads, qk, qs, dtype=torch.bfloat16):
    if dtype is torch.float32 or os.environ.get("MINIWORLD_TOKEN_DIT_H100_PACK_VECTOR", "0") == "1":
        return _ext().token_pack_vector(weights, int(heads), bool(qk), float(qs), dtype is torch.float32)
    return _ext().token_pack(weights, int(heads), bool(qk), float(qs))


_TOKEN_PACKS: dict = {}
_TOKEN_STREAMS: dict = {}


@functools.lru_cache(maxsize=8)
def _token_swiglu(device_index):
    from .gemm_swiglu import GemmSwigluSm90
    return GemmSwigluSm90(device_index)


def token_inference(single, cond, pair, mask, weights, qk, eq, ek, heads):
    """Hopper port of the B200 token block's fusion and saved-buffer boundaries.

    Live CUDA weight packing, fused pair normalization/bias, gated TMA/WGMMA
    attention, fused residual/AdaLN rows and WGMMA SwiGLU. All other products
    are cuBLAS. Unsupported calls raise and never enter a portable backend.
    """
    import math
    from ... import _capture
    from ...augmented_attention import cuda as attention
    from ...augmented_attention.cuda import apb

    A, B, L, D = single.shape
    if (torch.is_grad_enabled() or single.dtype not in (torch.bfloat16, torch.float32) or pair.dtype != single.dtype
        or B != 1 or D != 768 or heads != 16 or L % 128 or cond.shape[-1] != 384
        or tuple(pair.shape) != (1, L, L, 128) or not pair.is_contiguous()
        or torch.cuda.get_device_capability(single.device) != (9, 0)):
        raise ValueError("Hopper token CUDA candidate requires inference BF16/FP32 bucketed 768/384/128, 16 heads")
    qs = math.log2(math.e) / math.sqrt(48)
    fp32 = single.dtype is torch.float32
    asynchronous = os.environ.get("MINIWORLD_TOKEN_DIT_H100_INFER_ASYNC", "0") == "1"
    key = (asynchronous, qk, heads, single.dtype, os.environ.get("MINIWORLD_TOKEN_DIT_H100_PACK_VECTOR", "0"), *((w.data_ptr(), w._version) for w in weights))
    def prepare():
        if not asynchronous:
            return token_pack(weights, heads, qk, qs, single.dtype), weights, None, None
        idx = single.device.index if single.device.index is not None else torch.cuda.current_device()
        if idx not in _TOKEN_STREAMS:
            _TOKEN_STREAMS[idx] = torch.cuda.Stream(device=single.device)
        stream = _TOKEN_STREAMS[idx]
        out = _ext().token_alloc90(weights, heads, qk, fp32)
        stream.wait_stream(torch.cuda.current_stream(single.device))
        cond_ready, matrix_ready = torch.cuda.Event(), torch.cuda.Event()
        with torch.cuda.stream(stream):
            _ext().token_stage90(weights, out, heads, qk, qs, fp32, 0, 4)
            cond_ready.record(stream)
            _ext().token_stage90(weights, out, heads, qk, qs, fp32, 5, 13)
            matrix_ready.record(stream)
        _ext().token_stage90(weights, out, heads, qk, qs, fp32, 4, 5)
        return out, weights, cond_ready, matrix_ready
    packed, _, cond_ready, matrix_ready = _capture.lookup(_TOKEN_PACKS, key, prepare, limit=8)
    w1, b1, w2, b2, wf, wqkvg, bqkvg, wo, wab_fp32, ws, nq, nk, wab = packed
    mask = None if mask is None else mask.reshape(L)
    # The augmented-attention module fills fully masked rows with a finite
    # negative value, so its reference softmax is uniform on that edge.
    # AttentionPairBias has a separate zero-probability reference contract.
    if fp32:
        zh = single.new_empty((L * L, 128))
        _ext().layernorm128_rows(pair.view(L * L, 128), zh, 1e-5, True)
        bias = torch.mm(wf, zh.t()).view(heads, L, L)
        if mask is not None:
            _ext().token_mask90(bias, mask)
    else:
        bias = apb.pair_bias(pair.view(L * L, 128), wf, mask, L, 1e-5, -1e30)
    shared = cond.shape[0] == 1 or cond.stride(0) == 0
    c = cond[0, 0] if shared else cond.reshape(-1, 384)
    P = c.shape[0]
    cn = single.new_empty((P, 384)); cc = torch.empty_like(cn)
    cond_rows(c, cn, cc, 1e-5)
    if cond_ready is not None:
        torch.cuda.current_stream(single.device).wait_event(cond_ready)
    g1 = torch.addmm(b1, cn, w1.t()).view(P, 4, D)
    g2 = torch.addmm(b2, cc, w2.t()).view(P, 2, D)
    M = A * L
    x = single.new_empty((M, D), dtype=torch.float32)
    xa = single.new_empty((M, D))
    adaln_in_rows(single.view(M, D), x, g1[:, 0], g1[:, 1], xa, P, 1e-5)
    if matrix_ready is not None:
        torch.cuda.current_stream(single.device).wait_event(matrix_ready)
    qkvg = (torch.addmm(bqkvg[:3 * D], xa, wqkvg[:3 * D].t()) if fp32
            else torch.addmm(bqkvg, xa, wqkvg.t()))
    if qk:
        qknorm_rows(qkvg, nq, nk, eq, ek, D)
    if fp32:
        vt = torch.mm(wqkvg[3 * D:], xa.t())
        attention.tf32_gated_inference(qkvg, vt, bias, 0, A, 2 * D)
    else:
        attention.gated_inference(qkvg, bias, 0, A)
    y = torch.mm(qkvg[:, :D], wo.t())
    resgate_adaln_rows(x, y, g2[:, 0], g1[:, 2], g1[:, 3], xa, P, 1e-5)
    h = single.new_empty((M, ws.shape[1]))
    if fp32:
        ab = torch.mm(xa, wab_fp32.t())
        swiglu_rows(ab, h)
    else:
        op = _token_swiglu(single.device.index)
        op.ext.gemm_swiglu(xa, wab, h, op.nsm)
    y = torch.mm(h, ws.t())
    out = torch.empty_like(xa)
    resgate_out_rows(x, y, g2[:, 1], out, P)
    return out.view(A, B, L, D)


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


def cond_rows(c, cn, cc, eps=1e-5):
    """cn = LN(c), cc = c, both in the GEMM dtype, in one warp-per-row pass over 384 conditioning columns."""
    _ext().cond_rows_cuda(c, cn, cc, float(eps))


def qknorm_rows(qk, wq, wk, eq, ek, d=768):
    """QK-norm in place: q = qk[:, 0:d], k = qk[:, d:2d] of every row, RMSNorm per head (head dim = len(wq): 48, 32 or 64 at
    d 768, 64 at d 1024) times wq / wk (fp32, any logit scale folded in), eps eq / ek."""
    _ext().qknorm_rows_cuda(qk, wq, wk, float(eq), float(ek), int(d))


#: ``pair_bias_all`` applies the key mask itself (the runner then skips its own -inf fill).
PAIR_BIAS_MASK = True


def key_perm(L, device):
    """Position p of each group of 8 keys holds key 4 i + j for p = 2 j + i: the key order of the H100 TF32 attention core
    (``augmented_attention/cuda/attn_tf32.cu``), in which its bias and key mask are laid out."""
    p = torch.arange(L, device=device)
    r = p % 8
    return p - r + 4 * (r % 2) + r // 2


def pair_bias_all(z2d, wt, out, L, eps=1e-5, mask=None, perm=False):
    """Every block's pair bias, head-major: out [NB, L, L] = (LN(z) @ wt)^T with wt [C, NB] (the LayerNorm weights and
    any log2 e scale already folded in), -inf on the columns of masked keys (``mask`` [L] bool). z2d [L L, 128].
    A CUDA LayerNorm and one cuBLAS GEMM (a fused one-pass LayerNorm + projection kernel, one thread per pair row, measured
    slower -- 450 against 266 us for a whole L = 768 block call -- and was dropped). ``perm``: the key columns in
    ``key_perm`` order (the TF32 core's; the LayerNorm writes the pair rows there, so it costs nothing)."""
    nb = wt.shape[1]
    m = None if mask is None else mask.reshape(-1).to(torch.bool)
    if m is not None and perm:
        m = m[key_perm(L, m.device)]
    m = None if m is None else m.contiguous()
    zh = torch.empty(z2d.shape[0], 128, device=z2d.device, dtype=wt.dtype)
    _ext().layernorm128_rows(z2d, zh, float(eps), bool(perm))
    torch.mm(wt.t(), zh.t(), out=out.view(nb, -1))
    if m is not None:
        out.view(nb, L, L)[:, :, ~m] = float("-inf")


__all__ = ["STAT_W", "adaln_in_rows", "adaln_rows", "available", "cond_rows", "gate_rows", "key_perm", "layernorm_rows", "pair_bias_all", "qknorm_rows", "resgate_adaln_rows", "resgate_out_rows", "swiglu_rows"]
