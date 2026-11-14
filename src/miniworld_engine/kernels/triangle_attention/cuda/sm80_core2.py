"""The generalised A100 (sm_80) triangle-attention core (``sm80/attn2_fwd_sm80.cuh``, ``attn2_bwd_sm80.cuh``): head dim 16, 32 or 64, element strides for every operand (token-major packed
buffers and head-major ``[A, B, H, L, D]`` tensors alike), an optional per-pair-row key mask.  The arithmetic and the schedules are those of ``sm80.py``'s core (the same kernels with the
widths and the addressing as template parameters); this module only builds the extension and wraps the launches -- callers (the module-level path of ``sm80_wide.py``, the
``projected_attention`` leaf) put them inside their own opaque ops.

A layout is the 4-tuple of element strides ``(token, row, batch, head)`` of a ``[batch, row, token, head, channel]`` tensor (the channel stride is 1).
"""

import functools
import math
from pathlib import Path

import torch

from . import sm80_arch as _arch
from ..._nvcc import build_failed, ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"
_L2E = 1.4426950408889634
BF = torch.bfloat16
F32 = torch.float32
#: partial-gradient groups of the query side: rows per CTA (the bias gradient is summed from L / DQ_ROWS bf16 partials)
DQ_ROWS = 4

#: forward schedule of the 16-channel fixed-shift core (0 = automatic by L; 1-12: see attn2_ops.cu), a module-level knob for sweeps
SCHED = 0

#: softmax mode of the 16-channel fixed-shift core: 1 = the first key tile's maximum as the shift, 2 = no shift (weights 2^s; the online fallback decides on overflow / an empty sum)
FIXMODE = 2

#: schedules of the unmasked 16-channel backward kernels (0 = the base schedule; the defaults measured -2.4..-3 % of a D64 training step; see attn2_ops.cu), module-level knobs for sweeps
DQ_SCHED = 3  # 3 stages x 32 keys x 2 CTAs / SM
DKV_SCHED = 5  # 2 stages x 3 CTAs / SM

Layout = tuple[int, int, int, int]


@functools.lru_cache(maxsize=1)
def ext():
    ensure_cuda_home()
    return load_extension(
        name=_arch.name("triattn_sm80_core2"),
        sources=[str(_dir / "attn2_ops.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            *_arch.gencode(),
            f"-I{_dir}",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


@torch.compiler.assume_constant_result
def loads() -> bool:
    """Builds (first call) or loads the extension; a failed build / load raises ``_nvcc.KernelBuildError`` (never a fallback).  A process-level constant: ``torch.compile``
    evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    try:
        ext()
    except Exception as exc:  # a build / load failure is an error, never a fallback
        build_failed("sm80 triangle-attention core (head dim 16 / 32)", exc)
        return False  # reached only while planning (build_failed returns there)
    return True


def token_major(length: int, ld: int, head_dim: int) -> Layout:
    """The layout of a token-major ``[Z, L, L, H * head_dim]`` buffer with row stride ``ld`` (the channels of head h at ``h * head_dim``)."""
    return (ld, length * ld, length * length * ld, head_dim)


def head_major(heads: int, length: int, head_dim: int, batch: int) -> Layout:
    """The layout of a contiguous ``[A, B, H, L, D]`` tensor (``A`` = the pair rows): token j of row a, batch b, head h at ``(((a B + b) H + h) L + j) D``."""
    return (
        head_dim,
        batch * heads * length * head_dim,
        heads * length * head_dim,
        length * head_dim,
    )


def forward(
    q,
    k,
    v,
    bias,
    rowmask,
    out,
    lse,
    lays,
    length,
    heads,
    batch,
    head_dim,
    sm_scale,
    fixed=False,
    key_mask=None,
) -> None:
    """``out = softmax(sm_scale q k^T + bias [+ the per-row key mask]) v`` for q / k / v / out the strided views ``lays = (lq, lk, lv, lo)``; ``bias`` ``[B, H, L, L]`` bf16 contiguous,
    ``rowmask`` ``[B, L, L]`` uint8 or an empty tensor, ``lse`` ``[B, H, L, L]`` fp32 or an empty tensor (the base-2 log-sum-exp the backward reads).  ``key_mask``: the global
    ``[B, L]`` key mask already folded into ``bias`` (or None): an inference call without ``rowmask`` streams the keys only up to each sample's last live one."""
    lq, lk, lv, lo = lays
    ext().attn2_fwd(
        q,
        k,
        v,
        bias,
        rowmask,
        out,
        lse,
        list(lq),
        list(lk),
        list(lv),
        list(lo),
        length,
        heads,
        batch,
        head_dim,
        sm_scale * _L2E,
        (FIXMODE if fixed else 0),
        SCHED,
        key_mask if key_mask is not None else torch.empty(0, device=q.device),
    )


def delta_rows(o, dov, out, lays, length, heads, batch, head_dim) -> None:
    """``out[b, h, row, query] = sum_d o . dov`` (fp32, ``[B, H, L, L]``) for the strided views ``lays = (lo, ld)``: the backward's row term, for callers whose back stage does not compute it."""
    lo, ld = lays
    ext().attn2_delta(o, dov, out, list(lo), list(ld), length, heads, batch, head_dim)


def dq_rows(head_dim: int) -> int:
    """The pair rows per bias-gradient partial of the query side: 4 (``DQ_ROWS``), 2 for 64-channel heads (their Q and dO tiles of 2 rows fill the shared memory)."""
    return 2 if head_dim == 64 else DQ_ROWS


#: the transient memory of a training step (the bias gradient's partials, cubic in L) above which the callers' Triton path serves the gradient
MAX_DBP_BYTES = 16 << 30


def dbp_bytes(length: int, heads: int, batch: int, head_dim: int = 32) -> int:
    """The bias gradient's bf16 partials of ``backward`` (one ``[B H, L, L]`` plane per group of ``DQ_ROWS`` pair rows): the transient memory of the training step (cubic in L)."""
    return (length // dq_rows(head_dim)) * batch * heads * length * length * 2


def dbp_fits(length: int, heads: int, batch: int, head_dim: int = 32) -> bool:
    return dbp_bytes(length, heads, batch, head_dim) <= MAX_DBP_BYTES


def backward(
    q,
    k,
    v,
    dov,
    bias,
    rowmask,
    lse,
    delta,
    dq,
    dk,
    dv,
    lays,
    length,
    heads,
    batch,
    head_dim,
    sm_scale,
) -> torch.Tensor:
    """The backward of ``forward``: dq / dk / dv are written into the strided views ``lays = (lq, lk, lv, ld, ldq, ldk, ldv)`` and the pair bias' gradient ``[B, H, L, L]`` bf16
    (the sum over the pair rows of dS, from bf16 partials over groups of ``DQ_ROWS`` rows and a fixed-order reduction) is returned.  ``delta`` = sum_d o do per (b, h, row, query)."""
    lq, lk, lv, ld, ldq, ldk, ldv = lays
    e = ext()
    scl = sm_scale * _L2E
    e.attn2_set_sched(DQ_SCHED, DKV_SCHED)
    groups = length // dq_rows(head_dim)
    dbp = torch.empty((groups, batch * heads, length, length), dtype=BF, device=q.device)
    e.attn2_bwd_dq(
        q,
        k,
        v,
        dov,
        bias,
        rowmask,
        lse,
        delta,
        dq,
        dbp,
        list(lq),
        list(lk),
        list(lv),
        list(ld),
        list(ldq),
        length,
        heads,
        batch,
        head_dim,
        scl,
        sm_scale,
    )
    db = torch.empty((batch, heads, length, length), dtype=BF, device=q.device)
    e.attn2_db_reduce(dbp, db, groups)
    del dbp
    bias_t = torch.empty_like(bias)
    e.attn2_bias_transpose(bias, bias_t, length)
    e.attn2_bwd_dkv(
        q,
        k,
        v,
        dov,
        bias_t,
        rowmask,
        lse,
        delta,
        dk,
        dv,
        list(lq),
        list(lk),
        list(lv),
        list(ld),
        list(ldk),
        list(ldv),
        length,
        heads,
        batch,
        head_dim,
        scl,
        sm_scale,
    )
    return db


def default_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)
