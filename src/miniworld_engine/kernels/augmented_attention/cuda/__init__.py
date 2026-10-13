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
                "attn_fwd", (f"DHP={dh}", f"ROW_WIDTH={width(heads, d)}", "NWG=1")
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


def apb_backward(q, k, v, dob, bias, LSE, Dd, L, heads: int = 8, d: int = 384):
    """dQB and dKV own their complete rows: no cross-CTA attention atomics."""
    opts = dict(memory_format=torch.contiguous_format, dtype=torch.float32)
    DQ, DK, DV = (torch.empty_like(q, **opts) for _ in range(3))
    DB = torch.empty(heads, L, L, device=q.device, dtype=torch.float32)
    defs = _apb_training_defs(heads, d)
    _ext("attn_dq", defs).apb_dqb(q, k, v, dob, bias, LSE, Dd, DQ, DB)
    _ext("attn_dkv", defs + ("DBIAS=0",)).attn_dkv(
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


def supported(
    q: torch.Tensor, bias: torch.Tensor, bias_head_major: bool = False
) -> bool:
    """q [A, 1, L, 16, 48] (any float dtype) and a bias of [1, L, L, 16] (``bias_head_major``: [16, 1, L, L]) on sm_90
    with L a multiple of 128. The requirements are the kernels' own tile shapes, not a policy."""
    if not settings.current().augattn_bf16_sm90:
        return False
    if not q.is_cuda or q.dim() != 5 or not q.is_floating_point():
        return False
    A, B, L, h, d = q.shape
    if B != 1 or h != H or d != D or L % 128 or L == 0:
        return False
    if tuple(bias.shape) != ((H, 1, L, L) if bias_head_major else (1, L, L, H)):
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
        _ext("attn_fwd")
    except Exception as exc:  # noqa: BLE001
        _BUILD_FAILED = True
        warnings.warn(
            f"sm90 bf16 augmented attention unavailable, keeping the Triton path: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


# --------------------------------------------------------------------------------------------------- glue
@triton.jit
def _prep_qkv_kernel(q, k, v, qs, kb, vb, n, scale, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(
        qs + offs,
        (tl.load(q + offs, mask=m).to(tl.float32) * scale).to(tl.bfloat16),
        mask=m,
    )
    tl.store(kb + offs, tl.load(k + offs, mask=m).to(tl.bfloat16), mask=m)
    tl.store(vb + offs, tl.load(v + offs, mask=m).to(tl.bfloat16), mask=m)


def _prep_qkv(q, k, v, scale):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    n = q.numel()
    qs, kb, vb = (
        torch.empty(n, device=q.device, dtype=torch.bfloat16) for _ in range(3)
    )
    _prep_qkv_kernel[(triton.cdiv(n, 4096),)](q, k, v, qs, kb, vb, n, scale, BLOCK=4096)
    return qs.view(-1, H * D), kb.view(-1, H * D), vb.view(-1, H * D)


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
):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)  # token rows (a * L + l)
    hh = tl.arange(0, NH)
    dcol = tl.arange(0, DP)
    ptr = r[:, None, None] * (NH * DH) + (hh[:, None] * DH + dcol[None, :])[None, :, :]
    msk = (dcol[None, :] < DH)[None, :, :] & (r[:, None, None] >= 0)
    g = tl.load(do + ptr, mask=msk, other=0.0).to(tl.float32)
    x = tl.load(o + ptr, mask=msk, other=0.0)
    tl.store(dob + ptr, g.to(tl.bfloat16), mask=msk)
    s = tl.sum(g * x, axis=2)
    tl.store(dd + ((r // L)[:, None] * NH + hh[None, :]) * L + (r % L)[:, None], s)


def _prep_do(do, o, A, L):
    """dO -> bf16 and D = rowsum(dO * O) as [A, H, L] fp32."""
    do = do.reshape(A * L, H * D).contiguous()
    dob = torch.empty(A * L, H * D, device=do.device, dtype=torch.bfloat16)
    dd = torch.empty(A, H, L, device=do.device, dtype=torch.float32)
    _prep_do_kernel[((A * L) // 8,)](
        do, o.reshape(A * L, H * D), dob, dd, L, NH=H, DH=D, DP=64, ROWS=8
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


def _fwd_launch_fake(qs, kb, vb, bb, km):
    """O fp32 [A L, H D] (qs's shape) and the LSE fp32 [A, H, L]."""
    L = bb.shape[1]
    return (
        qs.new_empty(qs.shape, dtype=torch.float32),
        qs.new_empty((qs.shape[0] // L, H, L), dtype=torch.float32),
    )


@opaque(fake=_fwd_launch_fake, name="augmented_attention_bf16_sm90_fwd")
def _fwd_launch(
    qs: torch.Tensor,
    kb: torch.Tensor,
    vb: torch.Tensor,
    bb: torch.Tensor,
    km: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention forward on the prepared operands: (O fp32, LSE fp32)."""
    FWD_LAUNCHES[0] += 1
    L = bb.shape[1]
    return forward(qs, kb, vb, bb, km, qs.shape[0] // L, L)


def forward(qs, kb, vb, bb, km, A, L, qkvg=None):
    """The forward kernel on prepared operands, no op wrapper (for callers that are themselves one opaque op, as the fused
    token DiT training block): qs = q log2 e / sqrt 48, kb, vb [A L, 768] bf16 contiguous; bb [16, L, L] bf16 in log2
    units; km None or an additive key mask [A, L] fp32. Returns O fp32 [A L, 768] and the LSE [A, 16, L] (log2); with
    ``qkvg`` (the block's [A L, 3072] q|k|v|g GEMM output) og = sigmoid(g) O in bf16 instead of O -- all the token DiT
    block keeps (its backward takes D = rowsum(dO O) = rowsum(d og * og))."""
    lse = torch.empty(A, H, L, device=qs.device, dtype=torch.float32)
    if qkvg is not None:
        og = torch.empty(A * L, H * D, device=qs.device, dtype=torch.bfloat16)
        _ext("attn_fwd").attn_fwd_og(qs, kb, vb, bb, km, qkvg, og, lse)
        return og, lse
    o = torch.empty(A * L, H * D, device=qs.device, dtype=torch.float32)
    _ext("attn_fwd").attn_fwd(qs, kb, vb, bb, km, o, lse)
    return o, lse


def inference_core_supported(
    dtype: torch.dtype, L: int, d: int, h: int, device_index: int
) -> bool:
    """The fused token DiT step's gated core fits: bf16, 16 heads x 48, L a multiple of 128, H100."""
    return (
        settings.current().augattn_bf16_sm90
        and dtype is torch.bfloat16
        and (d, h) == (H * D, H)
        and L % 128 == 0
        and _is_hopper(device_index)
    )


def gated_inference(qkvg, bias_all, block, S):
    """The token DiT step's inference core (attn_fwd.cu, GATED): qkvg [S L, 3072] bf16 with q in exp2 units, bias_all
    [nb 16, L, L] bf16 in log2 units (masked keys -inf); writes sigmoid(g) softmax(q k^T + bias) v over q for ``block``'s
    slice of the bias."""
    _ext("attn_fwd").attn_inf(qkvg, bias_all, int(block), int(S))


def tf32_inference_core_supported(
    dtype: torch.dtype, L: int, d: int, h: int, device_index: int
) -> bool:
    """The fp32 step's TF32 core (attn_tf32.cu) fits: fp32, 16 heads x 48, L a multiple of 128, H100.
    ``settings.augattn_tf32_sm90 = False`` keeps the Triton core."""
    return (
        settings.current().augattn_tf32_sm90
        and dtype is torch.float32
        and (d, h) == (H * D, H)
        and L % 128 == 0
        and _is_hopper(device_index)
    )


def tf32_gated_inference(qkg, vt, bias_all, block, S, goff):
    """The fp32 step's inference core (attn_tf32.cu, TF32 wgmma): qkg [S L, >= 2304] fp32 with q | k at columns 0 | 768 (q
    in exp2 units) and g at ``goff``; vt = v^T [768, S L]; bias_all [nb 16, L, L] fp32 in log2 units with its key columns in
    ``conditioned_transition.cuda.key_perm`` order (masked keys -inf). Writes sigmoid(g) softmax(q k^T + bias) v over q."""
    _ext("attn_tf32").attn_tf32_inf(qkg, vt, bias_all, int(block), int(S), int(goff))


def tf32_forward(q, k, vt, bias, km, A, L, qkvg=None):
    """The TF32 training forward: q (exp2 units), k [A L, 768] fp32, vt = v^T [768, A L]; bias [16, L, L] fp32 (log2 units)
    and km [A, L] additive or None, both key-permuted (``key_perm``). Returns O [A L, 768] fp32 and the LSE [A, 16, L]; with
    ``qkvg`` [A L, 3072] og = sigmoid(g) O instead of O (as ``forward``)."""
    o = torch.empty(A * L, H * D, device=q.device, dtype=torch.float32)
    lse = torch.empty(A, H, L, device=q.device, dtype=torch.float32)
    if qkvg is not None:
        _ext("attn_tf32").attn_tf32_fwd_og(q, k, vt, bias, km, qkvg, o, lse)
    else:
        _ext("attn_tf32").attn_tf32_fwd(q, k, vt, bias, km, o, lse)
    return o, lse


def tf32_backward(
    qs, k, v, dob, qt, kt, dot, bias_p, bias_tp, lse, dd, lse_p, dd_p, km, km_p, A, L
):
    """The TF32 attention backward (attn_tf32.cu dqb + dkv). qs (q in exp2 units), k, v, dob = dO [A L, 768] fp32 and their
    transposes qt, kt, dot [768, A L]; bias_p the forward's key-permuted bias [16, L, L]; bias_tp [16, L(key), L] the bias
    transposed with its query columns permuted; lse, dd [A, 16, L] and lse_p, dd_p their query-permuted copies; km / km_p the
    natural / permuted additive key masks (or None). Returns dQ (for the unscaled q), dK, dV [A L, 768] fp32 and dbias
    [16, L, L] fp32 with its key columns still permuted (``key_perm`` order)."""
    ext = _ext("attn_tf32")
    dq, dk, dv = (torch.empty_like(qs) for _ in range(3))
    db_p = torch.zeros(H, L, L, device=qs.device, dtype=torch.float32)
    ext.attn_tf32_dqb(qs, dob, k, v, kt, bias_p, lse, dd, km_p, dq, db_p)
    ext.attn_tf32_dkv(qs, dob, qt, dot, k, v, bias_tp, lse_p, dd_p, km, dk, dv)
    return dq, dk, dv, db_p


def backward(qs, kb, vb, dob, bb, km, lse, dd, A, L):
    """dQ, dK, dV [A L, 768] fp32 (with respect to the unscaled q, k, v) and dbias [16, L, L] fp32 (natural units;
    a transposed view when A is not a multiple of 3). dob [A L, 768] bf16 and dd = rowsum(dO O) [A, 16, L] fp32; the other operands as ``forward`` took them."""
    dq = torch.empty(A * L, H * D, device=qs.device, dtype=torch.float32)
    dk, dv = torch.empty_like(dq), torch.empty_like(dq)
    if (
        A % 3 == 0
    ):  # dbias in the dQ pass, three samples summed on chip before the L2 reds
        db = torch.zeros(H, L, L, device=qs.device, dtype=torch.float32)
        _ext("attn_dqb").attn_dqb(qs, kb, vb, dob, bb, km, lse, dd, dq, db)
        _ext("attn_dkv", ("DBIAS=0",)).attn_dkv(
            qs, kb, vb, dob, bb, km, lse, dd, dk, dv, db
        )
        return dq, dk, dv, db
    # dbias by per-sample L2 atomics in the dK / dV pass (transposed)
    _ext("attn_dq", () if L % 192 == 0 else ("NWG=2",)).attn_dq(
        qs, kb, vb, dob, bb, km, lse, dd, dq
    )
    dbt = torch.zeros(H, L, L, device=qs.device, dtype=torch.float32)
    _ext("attn_dkv").attn_dkv(qs, kb, vb, dob, bb, km, lse, dd, dk, dv, dbt)
    return dq, dk, dv, dbt.transpose(1, 2)


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
        A, _, L, _, _ = q.shape
        qs, kb, vb = _prep_qkv(q, k, v, LOG2E / math.sqrt(D))
        bb = _bias_prep(
            bias.reshape(H, L, L) if bias_head_major else bias[0].permute(2, 0, 1)
        )
        km = None
        if mask is not None:
            km = torch.where(mask.reshape(A, L), 0.0, -1e30).float().contiguous()
        o, lse = FWD_OP(qs, kb, vb, bb, km)
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
        ctx.has_mask, ctx.head_major, ctx.dims = km is not None, bias_head_major, (A, L)
        return o.view(A, 1, L, H, D)

    @staticmethod
    def backward(ctx, do):  # noqa: D102
        qs, kb, vb, bb, km, o, lse = ctx.saved_tensors
        km = km if ctx.has_mask else None
        A, L = ctx.dims
        dob, dd = _prep_do(do, o, A, L)
        dq, dk, dv, db = backward(qs, kb, vb, dob, bb, km, lse, dd, A, L)
        shp = (A, 1, L, H, D)
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


__all__ = [
    "FWD_OP",
    "augmented_attention_bf16_sm90",
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
