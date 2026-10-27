"""sm_90a pair-bias attention core with bf16 operands, fp32 accumulation, forward AND backward.

The token DiT's attention (16 heads x 48, one pair bias per head shared by all A augmented samples) run as four
hand-written wgmma/TMA kernels, developed in the research branch ``research/token-dit-overlap``
(``archive/experiments-20260928:experiments/token_dit_train``). Measured on an H100 SXM at A = 48 (do_bench, L2 evicted), against this engine's bf16
Triton core and torch SDPA:

    op                    L=384      L=768
    forward               107 us     334 us     (engine bf16 659 us at L768; no-math pattern floor 273 us)
    backward              451 us    1507 us     (engine bf16 ~3977 us at L768)
    forward + backward    558 us    1841 us     (engine bf16 4636, SDPA 4543 us at L768)

Token DiT block fwd+bwd (24 blocks, per-block checkpointing, pair-bias hoist, fp32 model): 22.22 -> 18.77 ms/block at
L768, 10.70 -> 9.52 at L384; fwd+bwd peak memory 8.50 -> 6.59 GB at L768 (the Triton backward writes dbias per sample).

The kernels:
  attn_fwd.cu   O = softmax(q k^T / sqrt 48 + bias) v and the row LSE (log2 units). The bias is seeded into the score
                accumulator and the QK wgmma accumulates onto it; q is pre-scaled by log2 e / sqrt 48 so every
                exponential is one ex2; the running max moves only when a block exceeds it by 8 (log2 units).
  attn_dqb.cu   dQ and dbias. The CTA's three warpgroups are three SAMPLES of one query tile; their dS tiles meet in
                shared memory and one warpgroup issues the L2 reds, so dbias costs a third of the atomics.
  attn_dkv.cu   dK and dV (built with DBIAS=0 next to attn_dqb; DBIAS=1 adds dbias with per-sample L2 atomics, the
                fallback when A is not a multiple of 3, paired with attn_dq.cu).
  attn_dq.cu    dQ alone (the fallback's partner).

Numerics: the operands are rounded to bf16 (q after the scale, the bias after the log2 e scale), P and dS are bf16 for
the tensor core, every sum is fp32. Against an fp64 truth, O / dq / dk / dv / dbias sit at the bf16-input-rounding
floor (O 4.7e-3 against a floor of 4.4e-3 at unit-variance inputs); in the token DiT block the error is that of the
engine's bf16 Triton core (tests/gpu/sm90/test_augmented_attention_bf16_sm90_gpu.py).

``supported()`` is the whole gate: sm_90, 16 heads x 48, B == 1, L a multiple of 128. Everything else keeps the
Triton path, and so do fake tensors and torch.compile tracing (``available()``): the op is a plain autograd Function
around native extensions, with no fake implementation.
"""

from __future__ import annotations

import functools
import math
import os
import warnings
from pathlib import Path

import torch
import triton
import triton.language as tl

from miniworld_engine import settings

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_dir = Path(__file__).parent
_tmn = (
    _dir.parent.parent / "transition" / "cuda" / "anthropic_v5"
)  # tmn_kernels.cuh (TMA / mbarrier helpers)

H, D = 16, 48
LOG2E = 1.0 / math.log(2.0)
# AttentionPairBias (B = 1): key splits per 64 query rows -- APB_KS warpgroups, each walking every APB_KS-th key block, merged in
# shared memory. Measured on H100 (L768, 8 x 48): 2 splits take 1-2 us off each core (about 0.5 % of the step); the default stays 1.
_APB_KS = {"fwd": 1, "inf": 1, "dq": 1, "dkv": 1}


@functools.lru_cache(maxsize=None)
def _ext(name: str, defs: tuple[str, ...] = ()):
    ensure_cuda_home()
    return load_extension(
        name=f"augattn_sm90_{name}"
        + "".join(
            "_" + d.replace("=", "").replace(".", "p").replace("-", "m") for d in defs
        ),
        sources=[str(_dir / f"{name}.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            *gencodes("90a"),
            f"-I{_tmn}",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *("-D" + d for d in defs),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


class ApbInferenceCore:
    """Hopper gated APB core: packed projections in, gated attention over Q out.

    Weight packing supplies exp2-scaled Q and bias. This entry point launches only
    hand-written TMA/WGMMA CUDA and keeps the B200 inference tensor lifetimes.
    """

    def __init__(self, device_index: int, heads: int, d: int = 384):
        from .apb import GEOMETRY, width

        _, dh = GEOMETRY[(heads, d)]
        self.heads = heads
        with torch.cuda.device(device_index):
            self.extension = _ext(
                "attn_fwd",
                (
                    f"DHP={dh}",
                    f"ROW_WIDTH={width(heads, d)}",
                    "NWG=1",
                    f"APB_KS={_APB_KS['inf']}",
                ),
            )

    def __call__(self, qkvg: torch.Tensor, bias: torch.Tensor):
        if bias.shape[0] != self.heads or qkvg.shape[0] != bias.shape[1]:
            raise ValueError(
                "Hopper APB inference requires one sample and head-major bias"
            )
        self.extension.attn_inf(qkvg, bias, 0, 1)
        return qkvg


def _apb_training_defs(heads: int, d: int):
    from .apb import GEOMETRY, width

    real, padded = GEOMETRY[(heads, d)]
    return (
        f"DHP={padded}",
        f"ROW_WIDTH={width(heads, d)}",
        "NWG=1",
        "APB_NATURAL=1",
        f"RSQDV={real**-0.5!r}f",
    )


def apb_forward(q, k, v, bias, L, heads: int = 8, d: int = 384):
    """Natural-unit packed column views; B200's saved O/LSE lifetimes on Hopper."""
    O = torch.empty_like(q, memory_format=torch.contiguous_format, dtype=torch.float32)
    LSE = torch.empty(heads, L, device=q.device, dtype=torch.float32)
    _ext("attn_fwd", _apb_training_defs(heads, d)).attn_fwd(q, k, v, bias, None, O, LSE)
    return O, LSE


def apb_forward_gated(qkvg, bias, L, heads: int = 8, d: int = 384):
    """The Pairformer track's training forward with its gate in the epilogue: (O fp32, og = sigmoid(g) O bf16, LSE) from the
    packed q | k | v | g GEMM output [L, 4 W] (natural units) and the head-major bias."""
    from .apb import width

    W = width(heads, d)
    O = torch.empty(L, W, device=qkvg.device, dtype=torch.float32)
    og = torch.empty(L, W, device=qkvg.device, dtype=torch.bfloat16)
    LSE = torch.empty(heads, L, device=qkvg.device, dtype=torch.float32)
    _ext(
        "attn_fwd", _apb_training_defs(heads, d) + (f"APB_KS={_APB_KS['fwd']}",)
    ).attn_fwd_o_og(qkvg, bias, O, og, LSE)
    return O, og, LSE


def apb_backward(q, k, v, dob, bias, LSE, Dd, L, heads: int = 8, d: int = 384):
    """dQB and dKV own their complete rows: no cross-CTA attention atomics."""
    opts = dict(memory_format=torch.contiguous_format, dtype=torch.float32)
    DQ, DK, DV = (torch.empty_like(q, **opts) for _ in range(3))
    DB = torch.empty(heads, L, L, device=q.device, dtype=torch.float32)
    defs = _apb_training_defs(heads, d)
    _ext("attn_dq", defs + (f"APB_KS={_APB_KS['dq']}",)).apb_dqb(
        q, k, v, dob, bias, LSE, Dd, DQ, DB
    )
    _ext("attn_dkv", defs + ("DBIAS=0", f"APB_KS={_APB_KS['dkv']}")).attn_dkv(
        q, k, v, dob, bias, None, LSE, Dd, DK, DV, DB
    )
    return DQ, DK, DV, DB


def natural_forward(
    q, k, v, bias, *, heads: int, head_dim: int, output_bf16: bool = False
):
    """Generic B1 natural-unit attention for fused callers; q/k/v [A*L,W].

    Returns O [A*L,W] (FP32, or BF16 directly from the epilogue when requested)
    and log2 LSE [A,H,L]. No Torch conversion kernels run here.
    """
    L, W = bias.shape[1], q.shape[1]
    if q.ndim != 2 or W % heads or W // heads not in (16, 32, 48) or L % 128:
        raise ValueError(
            "Hopper natural attention requires packed bucketed head geometry"
        )
    defs = (
        f"DHP={W // heads}",
        f"ROW_WIDTH={W}",
        "NWG=1",
        "APB_NATURAL=1",
        f"RSQDV={head_dim**-0.5!r}f",
    )
    if output_bf16:
        defs += ("FWD_BF16=1",)
    O = torch.empty_like(
        q,
        memory_format=torch.contiguous_format,
        dtype=torch.bfloat16 if output_bf16 else torch.float32,
    )
    LSE = torch.empty(q.shape[0] // L, heads, L, device=q.device, dtype=torch.float32)
    _ext("attn_fwd", defs).attn_fwd(q, k, v, bias, None, O, LSE)
    return O, LSE


def natural_backward(
    q,
    k,
    v,
    dob,
    bias,
    LSE,
    Dd,
    *,
    heads: int,
    head_dim: int,
    output_bf16: bool = False,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    return_bias: bool = True,
    bias_transposed: bool = True,
):
    """dQ/dK/dV and deterministic FP32 per-sample dBiasT [A,H,key,query].

    Caller reduces the sample axis in fixed order. ``output_bf16`` writes the
    activation gradients directly as BF16; default FP32. ``out`` may provide
    three disjoint packed column views with a common row stride. Every dBiasT
    element is written, so no zero-fill. ``return_bias=False`` allocates no
    bias workspace. For A1, ``bias_transposed=False`` instead writes DB[H,L,L]
    in query/key order directly in the dQ pass.
    """
    L, W = bias.shape[1], q.shape[1]
    if (
        q.ndim != 2
        or W % heads
        or W // heads not in (16, 32, 48)
        or L % 128
        or q.shape[0] % L
    ):
        raise ValueError(
            "Hopper natural attention requires packed bucketed head geometry"
        )
    A = q.shape[0] // L
    if return_bias and not bias_transposed and A != 1:
        raise ValueError("Direct query/key dBias requires one sample")
    defs = (
        f"DHP={W // heads}",
        f"ROW_WIDTH={W}",
        "NWG=1",
        "APB_NATURAL=1",
        f"RSQDV={head_dim**-0.5!r}f",
    )
    if output_bf16:
        defs += ("BWD_BF16=1",)
    dtype = torch.bfloat16 if output_bf16 else torch.float32
    if out is None:
        DQ, DK, DV = (
            torch.empty_like(q, memory_format=torch.contiguous_format, dtype=dtype)
            for _ in range(3)
        )
    else:
        if len(out) != 3 or any(
            t.shape != q.shape
            or t.device != q.device
            or t.dtype != dtype
            or t.stride(1) != 1
            or t.stride(0) != out[0].stride(0)
            or t.stride(0) < W
            for t in out
        ):
            raise ValueError(
                "Gradient outputs must be matching packed column views with a common row stride"
            )
        DQ, DK, DV = out
        if DQ.stride(0) != W:
            defs += (f"BWD_ROW_STRIDE={DQ.stride(0)}",)
    direct_bias = return_bias and not bias_transposed
    DBT = (
        torch.empty(
            (heads, L, L) if direct_bias else (A, heads, L, L),
            device=q.device,
            dtype=torch.float32,
        )
        if return_bias
        else None
    )
    if direct_bias:
        _ext("attn_dq", defs + ("DQ_BIAS=1",)).apb_dqb(
            q, k, v, dob, bias, LSE, Dd, DQ, DBT
        )
    else:
        _ext("attn_dq", defs + ("DQ_BIAS=0",)).attn_dq(
            q, k, v, dob, bias, None, LSE, Dd, DQ
        )
    if return_bias and bias_transposed:
        _ext("attn_dkv", defs + ("DB_PER_SAMPLE=1",)).attn_dkv(
            q, k, v, dob, bias, None, LSE, Dd, DK, DV, DBT
        )
    else:
        # DBIAS=0 never reads/writes this pointer; reuse an existing contiguous
        # FP32 tensor instead of allocating or initializing a bias workspace.
        _ext("attn_dkv", defs + ("DBIAS=0",)).attn_dkv(
            q, k, v, dob, bias, None, LSE, Dd, DK, DV, LSE
        )
    return DQ, DK, DV, DBT


# --------------------------------------------------------------------------------------------------- gate
@functools.lru_cache(maxsize=8)
def _is_hopper(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (9, 0)


#: (heads, head width) the bf16 cores serve: the token stack (16 x 48) and the dense atom stack (4 x 32).
BF16_LAYOUTS = ((16, 48), (4, 32))


def supported(
    q: torch.Tensor, bias: torch.Tensor, bias_head_major: bool = False
) -> bool:
    """q [A, 1, L, h, d] with (h, d) in ``BF16_LAYOUTS`` (any float dtype) and a bias of [1, L, L, h] (``bias_head_major``: [h, 1, L, L]) on sm_90
    with L a multiple of 128. The requirements are the kernels' own tile shapes, not a policy."""
    if not settings.current().augmented_attention_bf16_sm90:
        return False
    if not q.is_cuda or q.dim() != 5 or not q.is_floating_point():
        return False
    A, B, L, h, d = q.shape
    if B != 1 or (h, d) not in BF16_LAYOUTS or L % 128 or L == 0:
        return False
    if tuple(bias.shape) != ((h, 1, L, L) if bias_head_major else (1, L, L, h)):
        return False
    return _is_hopper(
        q.device.index if q.device.index is not None else torch.cuda.current_device()
    )


_BUILD_FAILED = False


def available(
    q: torch.Tensor, bias: torch.Tensor, bias_head_major: bool = False
) -> bool:
    """``supported()`` plus successful builds; a build failure warns once and keeps the Triton path."""
    global _BUILD_FAILED
    if _BUILD_FAILED or not supported(q, bias, bias_head_major):
        return False
    # Under FakeTensorMode (``dev derive``) or while torch.compile traces, keep the Triton path: a native extension
    # cannot run on fake tensors, and the Triton path is the one with fakes and autotune keys to record.
    from torch._subclasses.fake_tensor import FakeTensor

    if (
        torch.compiler.is_compiling()
        or isinstance(q, FakeTensor)
        or isinstance(bias, FakeTensor)
    ):
        return False
    try:
        _ext("attn_fwd", fwd_defs(q.shape[3], q.shape[4]))
    except Exception as exc:  # noqa: BLE001
        _BUILD_FAILED = True
        warnings.warn(
            f"sm90 bf16 augmented attention unavailable, keeping the portable path: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


# --------------------------------------------------------------------------------------------------- glue
def _bf16_rows(t, w):
    """[rows, w] bf16 view of ``t`` (a copy only when it is not already contiguous bf16)."""
    t = t.reshape(-1, w)
    if t.dtype is not torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t if t.is_contiguous() else t.contiguous()


def _prep_qkv(q, k, v, scale):
    """q scaled into the core's log2 units (one pass) and the bf16 rows of q, k, v: k and v are used as they are when already contiguous bf16."""
    w = q.shape[-2] * q.shape[-1]
    qc = q.contiguous()
    n = qc.numel()
    qs = torch.empty(n, device=q.device, dtype=torch.bfloat16)
    _scale_bf16_kernel[(triton.cdiv(n, 4096),)](qc, qs, n, scale, BLOCK=4096)
    return qs.view(-1, w), _bf16_rows(k, w), _bf16_rows(v, w)


@triton.jit
def _prep_do_kernel(
    do,
    o,
    dob,
    dd,
    L,
    NH: tl.constexpr,
    DH: tl.constexpr,
    DP: tl.constexpr,
    ROWS: tl.constexpr,
    WRITE_DOB: tl.constexpr,
):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)  # token rows (a * L + l)
    hh = tl.arange(0, NH)
    dcol = tl.arange(0, DP)
    ptr = r[:, None, None] * (NH * DH) + (hh[:, None] * DH + dcol[None, :])[None, :, :]
    msk = (dcol[None, :] < DH)[None, :, :] & (r[:, None, None] >= 0)
    g = tl.load(do + ptr, mask=msk, other=0.0).to(tl.float32)
    x = tl.load(o + ptr, mask=msk, other=0.0)
    if WRITE_DOB:
        tl.store(dob + ptr, g.to(tl.bfloat16), mask=msk)
    s = tl.sum(g * x, axis=2)
    tl.store(dd + ((r // L)[:, None] * NH + hh[None, :]) * L + (r % L)[:, None], s)


def _prep_do(do, o, A, L, nh=H, dh=D, write_dob=True):
    """dO -> bf16 and D = rowsum(dO * O) as [A, H, L] fp32. ``write_dob=False``: D only (dO is already the bf16 rows the core reads; it is returned as is)."""
    do = do.reshape(A * L, nh * dh).contiguous()
    dob = torch.empty(A * L, nh * dh, device=do.device, dtype=torch.bfloat16) if write_dob else do
    dd = torch.empty(A, nh, L, device=do.device, dtype=torch.float32)
    _prep_do_kernel[((A * L) // 8,)](
        do, o.reshape(A * L, nh * dh), dob, dd, L, NH=nh, DH=dh, DP=64, ROWS=8, WRITE_DOB=write_dob
    )
    return dob, dd


@triton.jit
def _scale_bf16_kernel(b, out, n, scale, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(
        out + offs,
        (tl.load(b + offs, mask=m).to(tl.float32) * scale).to(tl.bfloat16),
        mask=m,
    )


def _bias_prep(bias_hll):
    """[H, L, L] bias -> bf16 bias * log2 e (log2 units, as the kernels seed it)."""
    bias_hll = bias_hll.contiguous()
    out = torch.empty(bias_hll.shape, device=bias_hll.device, dtype=torch.bfloat16)
    n = bias_hll.numel()
    _scale_bf16_kernel[(triton.cdiv(n, 4096),)](bias_hll, out, n, LOG2E, BLOCK=4096)
    return out


# --------------------------------------------------------------------------------------------------- forward op
# The forward kernel is a torch.library op so selective activation checkpointing can name it: with
# ``checkpoint_context_keeping_attention()`` a checkpointed block keeps O and the LSE from its first forward and the
# recompute skips the kernel (see that function).
#: Forward kernel launches in this process (tests use it to see a checkpoint recompute skip the kernel).
FWD_LAUNCHES = [0]


def _fwd_launch_fake(qs, kb, vb, bb, km, out_bf16: bool = False):
    """O [A L, H D] (qs's shape; fp32, or bf16 with ``out_bf16``) and the LSE fp32 [A, H, L]."""
    L = bb.shape[1]
    return (
        qs.new_empty(qs.shape, dtype=torch.bfloat16 if out_bf16 else torch.float32),
        qs.new_empty((qs.shape[0] // L, bb.shape[0], L), dtype=torch.float32),
    )


@opaque(fake=_fwd_launch_fake, name="augmented_attention_bf16_sm90_fwd")
def _fwd_launch(
    qs: torch.Tensor,
    kb: torch.Tensor,
    vb: torch.Tensor,
    bb: torch.Tensor,
    km: torch.Tensor | None,
    out_bf16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention forward on the prepared operands: (O fp32 or bf16, LSE fp32)."""
    FWD_LAUNCHES[0] += 1
    L = bb.shape[1]
    nh = bb.shape[0]
    return forward(qs, kb, vb, bb, km, qs.shape[0] // L, L, nh=nh, dh=qs.shape[1] // nh, out_bf16=out_bf16)


def forward(qs, kb, vb, bb, km, A, L, qkvg=None, nh=H, dh=D, out_bf16=False):
    """The forward kernel on prepared operands, no op wrapper (for callers that are themselves one opaque op, as the fused
    token DiT training block): qs = q log2 e / sqrt 48, kb, vb [A L, 768] bf16 contiguous; bb [16, L, L] bf16 in log2
    units; km None or an additive key mask [A, L] fp32. Returns O fp32 [A L, 768] and the LSE [A, 16, L] (log2); with
    ``qkvg`` (the block's [A L, 3072] q|k|v|g GEMM output) og = sigmoid(g) O in bf16 instead of O -- all the token DiT
    block keeps (its backward takes D = rowsum(dO O) = rowsum(d og * og))."""
    lse = torch.empty(A, nh, L, device=qs.device, dtype=torch.float32)
    ext = _ext("attn_fwd", (*fwd_defs(nh, dh), *(("FWD_BF16=1",) if out_bf16 else ())))
    if qkvg is not None:
        og = torch.empty(A * L, nh * dh, device=qs.device, dtype=torch.bfloat16)
        ext.attn_fwd_og(qs, kb, vb, bb, km, qkvg, og, lse)
        return og, lse
    o = torch.empty(A * L, nh * dh, device=qs.device, dtype=torch.bfloat16 if out_bf16 else torch.float32)
    ext.attn_fwd(qs, kb, vb, bb, km, o, lse)
    return o, lse


def inference_core_supported(
    dtype: torch.dtype, L: int, d: int, h: int, device_index: int
) -> bool:
    """The fused token DiT step's gated core fits: bf16, 16 heads x 48, L a multiple of 128, H100."""
    return (
        settings.current().augmented_attention_bf16_sm90
        and dtype is torch.bfloat16
        and (d, h) == (H * D, H)
        and L % 128 == 0
        and _is_hopper(device_index)
    )


def fwd_defs(nh: int = H, dh: int = D) -> tuple[str, ...]:
    """Build defines of ``attn_fwd.cu`` for a token head layout (default 16 x 48 builds without them)."""
    return () if (nh, dh) == (H, D) else (f"DHP={dh}", f"ROW_WIDTH={nh * dh}")


def dq_defs(nh: int = H, dh: int = D) -> tuple[str, ...]:
    """Build defines of ``attn_dq.cu``: ``fwd_defs`` plus the dQ output scale 1/sqrt(head_dim) its epilogue multiplies by (the source's
    default is 1/sqrt(48); without it a 12 x 64 dQ is 15 % too large and a 24 x 32 one 18 % too small)."""
    return fwd_defs(nh, dh) + (() if dh == D else (f"RSQDV={dh**-0.5!r}f",))


def gated_inference(qkvg, bias_all, block, S, nh=H, dh=D):
    """The token DiT step's inference core (attn_fwd.cu, GATED): qkvg [S L, 3072] bf16 with q in exp2 units, bias_all
    [nb 16, L, L] bf16 in log2 units (masked keys -inf); writes sigmoid(g) softmax(q k^T + bias) v over q for ``block``'s
    slice of the bias."""
    _ext("attn_fwd", fwd_defs(nh, dh)).attn_inf(qkvg, bias_all, int(block), int(S))


def tf32_defs(nh: int = H, dh: int = D) -> tuple[str, ...]:
    """Build defines of ``attn_tf32.cu`` for a head layout (the default 16 x 48 builds without them)."""
    return () if (nh, dh) == (H, D) else (f"NHP={nh}", f"DHP={dh}")


TF32_LAYOUTS = ((16, 48), (24, 32), (12, 64), (16, 64))


def tf32_fwd_defs(nh: int = H, dh: int = D) -> tuple[str, ...]:
    """The forward's build: ``tf32_defs`` plus the tuning of the layout (the backward keeps ``tf32_defs``: one warpgroup per CTA and a
    128-key block are forward-only). Width 32 (24 x 32): 128-key blocks, one warpgroup per CTA, 3 CTAs / SM, p written back into the
    score registers -- 110 -> 90 us at L768, A = 5 (the other widths measured slower with it)."""
    extra = ()  # (the width-32 candidate "FWD_BN=128", "NWG=1", "BLKSM=3", "INPLACE_P=1" is timed 90 us but not yet correct: off)
    return (*tf32_defs(nh, dh), *extra)


def tf32_inference_core_supported(
    dtype: torch.dtype, L: int, d: int, h: int, device_index: int
) -> bool:
    """The fp32 step's TF32 core (attn_tf32.cu) fits: fp32, h heads of width d / h (``TF32_LAYOUTS``; ``d`` is the attention width), L a multiple of 128, H100.
    ``settings.augattn_tf32_sm90 = False`` keeps the Triton core."""
    return (
        settings.current().augattn_tf32_sm90
        and dtype is torch.float32
        and h > 0
        and d % h == 0
        and (h, d // h) in TF32_LAYOUTS
        and L % 128 == 0
        and _is_hopper(device_index)
    )


def tf32_gated_inference(qkg, vt, bias_all, block, S, goff, nh=H, dh=D):
    """The fp32 step's inference core (attn_tf32.cu, TF32 wgmma): qkg [S L, >= 2304] fp32 with q | k at columns 0 | 768 (q
    in exp2 units) and g at ``goff``; vt = v^T [768, S L]; bias_all [nb 16, L, L] fp32 in log2 units with its key columns in
    ``conditioned_transition.cuda.key_perm`` order (masked keys -inf). Writes sigmoid(g) softmax(q k^T + bias) v over q."""
    _ext("attn_tf32", tf32_fwd_defs(nh, dh)).attn_tf32_inf(qkg, vt, bias_all, int(block), int(S), int(goff))


def tf32_forward(q, k, vt, bias, km, A, L, qkvg=None, nh=H, dh=D):
    """The TF32 training forward: q (exp2 units), k [A L, 768] fp32, vt = v^T [768, A L]; bias [16, L, L] fp32 (log2 units)
    and km [A, L] additive or None, both key-permuted (``key_perm``). Returns O [A L, 768] fp32 and the LSE [A, 16, L]; with
    ``qkvg`` [A L, 3072] og = sigmoid(g) O instead of O (as ``forward``)."""
    o = torch.empty(A * L, nh * dh, device=q.device, dtype=torch.float32)
    lse = torch.empty(A, nh, L, device=q.device, dtype=torch.float32)
    ext = _ext("attn_tf32", tf32_fwd_defs(nh, dh))
    if qkvg is not None:
        ext.attn_tf32_fwd_og(q, k, vt, bias, km, qkvg, o, lse)
    else:
        ext.attn_tf32_fwd(q, k, vt, bias, km, o, lse)
    return o, lse


def tf32_backward(
    qs, k, v, dob, qt, kt, dot, bias_p, bias_tp, lse, dd, lse_p, dd_p, km, km_p, A, L, nh=H, dh=D, dpack=None
):
    """The TF32 attention backward (attn_tf32.cu dqb + dkv). qs (q in exp2 units), k, v, dob = dO [A L, 768] fp32 and their
    transposes qt, kt, dot [768, A L]; bias_p the forward's key-permuted bias [16, L, L]; bias_tp [16, L(key), L] the bias
    transposed with its query columns permuted; lse, dd [A, 16, L] and lse_p, dd_p their query-permuted copies; km / km_p the
    natural / permuted additive key masks (or None). Returns dQ (for the unscaled q), dK, dV [A L, 768] fp32 and dbias
    [16, L, L] fp32 with its key columns still permuted (``key_perm`` order). ``dpack`` [A L, >= 3 nh dh]: dQ, dK, dV are written into its first three column blocks
    (and returned as views of it) instead of three fresh tensors."""
    ext = _ext("attn_tf32", tf32_defs(nh, dh))
    n = nh * dh
    if dpack is None:
        dq, dk, dv = (torch.empty_like(qs) for _ in range(3))
    else:
        dq, dk, dv = (dpack[:, i * n : (i + 1) * n] for i in range(3))
    dbc = torch.zeros(tf32_dqb_chunks(A), nh, L, L, device=qs.device, dtype=torch.float32)
    ext.attn_tf32_dqb(qs, dob, k, v, kt, bias_p, lse, dd, km_p, dq, dbc)
    db_p = dbc.sum(0)  # the chunk buffers in order: bitwise repeatable
    ext.attn_tf32_dkv(qs, dob, qt, dot, k, v, bias_tp, lse_p, dd_p, km, dk, dv)
    return dq, dk, dv, db_p


def tf32_dqb_chunks(A: int) -> int:
    """Partial dbias buffers of the TF32 ``attn_tf32_dqb`` (see ``dqb_chunks``): the largest divisor of A up to 4."""
    return max(c for c in (1, 2, 3, 4) if A % c == 0)


def dqb_chunks(A: int) -> int:
    """Partial dbias buffers of ``attn_dqb``: each CTA adds its A / 3 / C sample groups to ONE buffer in order (the TMA reduce-adds
    of two CTAs into one buffer land in schedule order: not bitwise repeatable); the C buffers are summed in order afterwards.
    The largest divisor of A / 3 up to 4: C x 16 x (L / 64) CTAs, 4 buffers = 4 x 38 MB fp32 at L768."""
    groups = A // 3
    return max(c for c in (1, 2, 3, 4) if groups % c == 0)


def backward(qs, kb, vb, dob, bb, km, lse, dd, A, L, nh=H, dh=D, out_bf16=False):
    """dQ, dK, dV [A L, 768] fp32 (bf16 with ``out_bf16``; with respect to the unscaled q, k, v) and dbias [16, L, L] fp32 (natural units;
    a transposed view when A is not a multiple of 3). dob [A L, 768] bf16 and dd = rowsum(dO O) [A, 16, L] fp32; the other operands as ``forward`` took them."""
    odt = torch.bfloat16 if out_bf16 else torch.float32
    bd = ("BWD_BF16=1",) if out_bf16 else ()
    dq = torch.empty(A * L, nh * dh, device=qs.device, dtype=odt)
    dk, dv = torch.empty_like(dq), torch.empty_like(dq)
    ld = fwd_defs(nh, dh)
    if (
        A % 3 == 0
    ):  # dbias in the dQ pass, three samples summed on chip before the L2 reds
        chunks = dqb_chunks(A)
        dbc = torch.zeros(chunks, nh, L, L, device=qs.device, dtype=torch.float32)
        _ext("attn_dqb", (*ld, *bd)).attn_dqb(qs, kb, vb, dob, bb, km, lse, dd, dq, dbc)
        db = dbc.sum(0)  # the chunks in order: bitwise repeatable
        _ext("attn_dkv", ("DBIAS=0", *ld, *bd)).attn_dkv(
            qs, kb, vb, dob, bb, km, lse, dd, dk, dv, db
        )
        return dq, dk, dv, db
    # dbias by per-sample L2 atomics in the dK / dV pass (transposed)
    _ext("attn_dq", (*(() if L % 192 == 0 else ("NWG=2",)), *dq_defs(nh, dh), *bd)).attn_dq(
        qs, kb, vb, dob, bb, km, lse, dd, dq
    )
    dbt = torch.zeros(nh, L, L, device=qs.device, dtype=torch.float32)
    _ext("attn_dkv", (*ld, *bd)).attn_dkv(qs, kb, vb, dob, bb, km, lse, dd, dk, dv, dbt)
    return dq, dk, dv, dbt.transpose(1, 2)


# --------------------------------------------------------------------------------------------------- per-sample bias
# The same cores with one bias PER SAMPLE (build define ``PSB=1``): bias [H, A, L, L], the layout a stacked bias GEMM leaves
# (rows of the [H A L, L] TMA map are (head A + sample) L + query). Nothing is shared between the samples any more, so the
# backward writes each sample's dbias itself (``attn_dkv`` ``DB_PER_SAMPLE``: plain stores, no atomics, no reduction).
def forward_psb(qs, kb, vb, bb, km, A, L, qkvg=None, nh=H, dh=D):
    """``forward`` with a per-sample bias: bb [H, A, L, L] bf16 in log2 units. Returns (O, LSE), or (og, LSE) with ``qkvg``."""
    lse = torch.empty(A, nh, L, device=qs.device, dtype=torch.float32)
    ext = _ext("attn_fwd", (*fwd_defs(nh, dh), "PSB=1"))
    if qkvg is not None:
        og = torch.empty(A * L, nh * dh, device=qs.device, dtype=torch.bfloat16)
        ext.attn_fwd_og(qs, kb, vb, bb, km, qkvg, og, lse)
        return og, lse
    o = torch.empty(A * L, nh * dh, device=qs.device, dtype=torch.float32)
    ext.attn_fwd(qs, kb, vb, bb, km, o, lse)
    return o, lse


def backward_psb(qs, kb, vb, dob, bb, km, lse, dd, A, L, nh=H, dh=D):
    """``backward`` with a per-sample bias bb [H, A, L, L]: dQ, dK, dV [A L, 768] fp32 and the bias's gradient [H, A, L(query),
    L(key)] bf16 in the bias's own (log2) units -- the dQ pass writes it from the dS registers (``PSB_DB``); the dK / dV pass
    keeps no dbias. Deterministic: every output element has one writer."""
    dq = torch.empty(A * L, nh * dh, device=qs.device, dtype=torch.float32)
    dk, dv = torch.empty_like(dq), torch.empty_like(dq)
    dbias = torch.empty(nh, A, L, L, device=qs.device, dtype=torch.bfloat16)
    ld = fwd_defs(nh, dh)
    _ext("attn_dq", (*(() if L % 192 == 0 else ("NWG=2",)), *dq_defs(nh, dh), "PSB=1", "PSB_DB=1")).attn_dq_psb(
        qs, kb, vb, dob, bb, km, lse, dd, dq, dbias
    )
    _ext("attn_dkv", ("DBIAS=0", "PSB=1", *ld)).attn_dkv(
        qs, kb, vb, dob, bb, km, lse, dd, dk, dv, torch.empty(1, device=qs.device)
    )
    return dq, dk, dv, dbias


# What the checkpoint policy matches: the registered OpOverload under compile_wrap="custom_op" (the default). Under
# "disable" there is no op to name, so FWD_OP is the plain launcher and the policy recomputes it like everything else --
# same results, without the saving.
FWD_OP = (
    torch.ops.miniworld_engine.augmented_attention_bf16_sm90_fwd.default
    if hasattr(torch.ops.miniworld_engine, "augmented_attention_bf16_sm90_fwd")
    else _fwd_launch
)


def checkpoint_context_keeping_attention():
    """``context_fn`` for ``torch.utils.checkpoint`` (non-reentrant): keep this core's forward outputs (O fp32 and the
    LSE, 115 MB a call at L768 / A48) and recompute everything else. Under per-block checkpointing the backward's
    recompute then skips the attention forward (334 us a block at L768); the cost is O and LSE held from the forward
    to the backward for every checkpointed block."""
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    def policy(ctx, op, *args, **kwargs):
        return (
            CheckpointPolicy.MUST_SAVE
            if op == FWD_OP
            else CheckpointPolicy.PREFER_RECOMPUTE
        )

    return create_selective_checkpoint_contexts(policy)


# --------------------------------------------------------------------------------------------------- op
class _AttentionBf16Sm90(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, mask, bias_head_major):  # noqa: D102
        A, _, L, nh, dh = q.shape
        qs, kb, vb = _prep_qkv(q, k, v, LOG2E / math.sqrt(dh))
        bb = _bias_prep(
            bias.reshape(nh, L, L) if bias_head_major else bias[0].permute(2, 0, 1)
        )
        km = None
        if mask is not None:
            km = torch.where(mask.reshape(A, L), 0.0, -1e30).float().contiguous()
        o, lse = FWD_OP(qs, kb, vb, bb, km)  # nh, dh follow from bb and qs
        if torch.is_grad_enabled() or any(ctx.needs_input_grad):
            ctx.save_for_backward(
                qs,
                kb,
                vb,
                bb,
                km if km is not None else torch.empty(0, device=q.device),
                o,
                lse,
            )
        ctx.has_mask, ctx.head_major, ctx.dims = km is not None, bias_head_major, (A, L, nh, dh)
        return o.view(A, 1, L, nh, dh)

    @staticmethod
    def backward(ctx, do):  # noqa: D102
        qs, kb, vb, bb, km, o, lse = ctx.saved_tensors
        km = km if ctx.has_mask else None
        A, L, nh, dh = ctx.dims
        dob, dd = _prep_do(do, o, A, L, nh, dh)
        dq, dk, dv, db = backward(qs, kb, vb, dob, bb, km, lse, dd, A, L, nh, dh)
        shp = (A, 1, L, nh, dh)
        dbias = db.unsqueeze(1) if ctx.head_major else db.permute(1, 2, 0).unsqueeze(0)
        return dq.view(shp), dk.view(shp), dv.view(shp), dbias, None, None


def augmented_attention_bf16_sm90(
    q, k, v, bias, mask=None, *, bias_head_major: bool = False
):
    """``softmax(q k^T / sqrt(48) + bias) v`` in bf16 with fp32 accumulation; returns fp32 [A, 1, L, 16, 48].

    q, k, v: [A, 1, L, 16, 48] (fp32 or bf16). bias: [1, L, L, 16], or [16, 1, L, L] with ``bias_head_major`` (the
    layout the token DiT's hoisted pair bias is produced in -- no permute copy). mask: key mask [A, 1, L] or [1, L]
    (True = attend). Call ``available()`` first."""
    if mask is not None and mask.dim() == 2:
        mask = mask[None].expand(q.shape[0], *mask.shape)
    return _AttentionBf16Sm90.apply(q, k, v, bias, mask, bias_head_major)


class _AttentionBf16Sm90Psb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, mask):  # noqa: D102
        A, _, L, _, _ = q.shape
        qs, kb, vb = _prep_qkv(q, k, v, LOG2E / math.sqrt(D))
        bb = _bias_prep(bias.reshape(H, A, L, L))
        km = None
        if mask is not None:
            km = torch.where(mask.reshape(A, L), 0.0, -1e30).float().contiguous()
        o, lse = forward_psb(qs, kb, vb, bb, km, A, L)
        if torch.is_grad_enabled() or any(ctx.needs_input_grad):
            ctx.save_for_backward(
                qs, kb, vb, bb, km if km is not None else torch.empty(0, device=q.device), o, lse
            )
        ctx.has_mask, ctx.dims = km is not None, (A, L)
        return o.view(A, 1, L, H, D)

    @staticmethod
    def backward(ctx, do):  # noqa: D102
        qs, kb, vb, bb, km, o, lse = ctx.saved_tensors
        km = km if ctx.has_mask else None
        A, L = ctx.dims
        dob, dd = _prep_do(do, o, A, L)
        dq, dk, dv, dbias = backward_psb(qs, kb, vb, dob, bb, km, lse, dd, A, L)
        shp = (A, 1, L, H, D)
        # the kernel's gradient is wrt the log2-unit bias the cores read; this op's input is the natural-unit bias times log2 e
        return dq.view(shp), dk.view(shp), dv.view(shp), dbias * LOG2E, None


def augmented_attention_bf16_sm90_psb(q, k, v, bias, mask=None):
    """``softmax(q k^T / sqrt(48) + bias[sample]) v`` with a bias of its own per sample, in bf16 with fp32 accumulation.

    q, k, v: [A, 1, L, 16, 48]; bias: [16, A, L, L] head-major (``A`` samples, as the token DiT's stacked bias GEMM leaves it);
    mask: key mask [A, 1, L] or [1, L] (True = attend). Returns fp32 [A, 1, L, 16, 48]; the gradient of the bias comes back in
    the bias's layout. Same gate as ``augmented_attention_bf16_sm90``."""
    if mask is not None and mask.dim() == 2:
        mask = mask[None].expand(q.shape[0], *mask.shape)
    return _AttentionBf16Sm90Psb.apply(q, k, v, bias, mask)


__all__ = [
    "FWD_OP",
    "augmented_attention_bf16_sm90",
    "augmented_attention_bf16_sm90_psb",
    "backward_psb",
    "dq_defs",
    "forward_psb",
    "available",
    "backward",
    "checkpoint_context_keeping_attention",
    "forward",
    "gated_inference",
    "inference_core_supported",
    "supported",
    "tf32_forward",
    "tf32_gated_inference",
    "tf32_backward",
    "tf32_inference_core_supported",
]
