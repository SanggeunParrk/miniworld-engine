"""sm_100a (B200) pair-bias attention core, forward AND backward in bf16 (fp32 accumulation) and in fp32 (TF32 tensor cores),
plus the gated inference cores the fused token DiT step uses.

The token DiT's attention (16 heads x 48, one pair bias per head shared by all A augmented samples) as tcgen05 / TMEM /
TMA kernels, developed on branch ``b200/token-dit`` (``experiments/augattn_sm100``, rounds v1-v3). B200, A = 48,
CUDA-graph replay, 2026-09-29:

    op                    L=384      L=768
    forward               65.7 us    210.0 us    (engine bf16 Triton 152.5 / 517.4 us)
    forward + backward    283.5 us   962.6 us    (engine bf16 Triton 706.0 / 2895.1 us)

The kernels (``sm100.cuh`` holds the tcgen05 / TMA / mbarrier helpers):
  attn_fwd2.cu  O = softmax(q k^T / sqrt 48 + bias) v (fp32) and the row LSE (log2 units). Persistent CTAs, 64-key
                blocks, two softmax warpgroups (one per sample of a pair) that take turns on the exponentials (PP).
  attn_dqb.cu   dQ and dbias. A CTA owns (head, 128 queries, 128-key chunk) and walks all samples, so dbias leaves once
                with plain stores; dQ partials leave through TMA bulk reduce-adds into a buffer attn_dkv zero-fills.
  attn_dkv.cu   dK and dV (one key row per thread, the bias read transposed).
  attn_inf.cu   the inference step's core: q | k | v | g read as column views of the q|k|v|g GEMM output (logits
                pre-scaled into exp2 units), sigmoid(g) * o written over q.
  attn_fwd_tf32.cu, attn_dkv_tf32.cu, attn_dqb_tf32.cu, attn_inf_tf32.cu
                the fp32 twins (kind::tf32 MMAs, fp32 softmax): 32-key / 32-query blocks, 48-wide fp32 rows as a
                32-column 128-B-swizzled box + a 16-column 64-B one, MN-major operands in the 128-B swizzle with 32-B atoms
                (sm100.cuh); the inference twin reads v^T (q | k | g + v^T). Built per token DiT head layout like the bf16
                cores (``_tdit_defs``: 16 x 48 default, 24 x 32, 12 x 64, 16 x 64): a 32-wide fp32 head row is one 128-B box,
                a 64-wide one two; the 64-wide builds drop a pipeline stage where three do not fit (attn_fwd_tf32 /
                attn_inf_tf32: 2 stages, attn_dqb_tf32: one K slot).
  glue.cu       dO -> bf16 with D = rowsum(dO O), and the bf16 bias transpose attn_dkv reads.

Numerics against fp64 at A = 48: O 1.6e-3, dq 3.0e-3, dk 2.9e-3, dv 2.9e-3, dbias 2.4e-3 (bf16-input-rounding floor). The fp32
kernels: O, dq, dk, dv, dbias within 3e-3 of fp64 (tests/gpu/sm100/test_augmented_attention_tf32_sm100_gpu.py).

The cubins are built with this environment's nvcc on first use and cached under ``settings.engine_jit_root``
(keyed by the sources and flags). They are launched through the CUDA driver (``driver.py``, cuda.bindings) with 2-D
TMA descriptors encoded on the host; launches go on torch's current stream and are CUDA-graph capturable.

``supported()`` is the whole gate: sm_100, 16 heads x 48, B == 1, an even A, L a multiple of 128, no key mask.
Everything else keeps the Triton path, and so do fake tensors and torch.compile tracing (``available()``).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._device import num_sms
from miniworld_engine import settings

H, D = 16, 48
_dir = Path(__file__).parent
_SMEM = 232448
_NVCC_FLAGS = ("-std=c++17", "-O3", "-arch=sm_100a", "--cubin", "-lineinfo")


# --------------------------------------------------------------------------------------------------- build
def _jit_root() -> Path:
    return settings.engine_jit_root() / "augattn_sm100"


@functools.lru_cache(maxsize=None)
def cubin(stem: str, defs: tuple[str, ...] = (), src_dir: str | None = None, newest: bool = False) -> str:
    """Path of ``<src_dir>/<stem>.cu`` (default: this directory) built for sm_100a with ``-D`` ``defs`` and this
    directory's ``sm100.cuh`` on the include path; rebuilt only when a source or a flag changes. ``newest``: by the newest nvcc here
    that knows sm_100a (toolchain ``sm100``, ``transition.cuda.fused_sm100a.kernel_toolchain``: 13.1 on the B200 box) instead of the
    torch-matched one, under its own artifact name (the key does not see the compiler, so the two builds must not share a name)."""
    from miniworld_engine.kernels import _nvcc

    src = Path(src_dir or _dir) / f"{stem}.cu"
    flags = (*_NVCC_FLAGS, f"-I{_dir}", *("-D" + d for d in defs))
    if newest:
        return str(_nvcc.cubin(f"augattn_sm100_newest/{stem}", flags, source=src, toolchain="sm100", timeout=None))
    # no -ccbin: this family has always been built with nvcc's default host compiler
    return str(_nvcc.cubin(f"augattn_sm100/{stem}", flags, source=src, toolchain="torch-nohost", timeout=None))


@functools.lru_cache(maxsize=None)
def _sm100_kernel(
    stem: str,
    func: str,
    device_index: int,
    pdl: bool = False,
    src_dir: str | None = None,
    defs: tuple[str, ...] = (),
    cluster: int | None = None,
    smem: int = _SMEM,
    newest: bool = False,
):
    from . import driver

    with torch.cuda.device(device_index):
        return driver.Kernel(
            cubin(stem, defs, src_dir=src_dir, newest=newest), func, smem, cluster=cluster, pdl=pdl
        )


# --------------------------------------------------------------------------------------------------- gate
@functools.lru_cache(maxsize=8)
def _is_blackwell(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (10, 0)


def _index(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def supported(
    q: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None = None,
    bias_head_major: bool = False,
) -> bool:
    """q [A, 1, L, 16, 48] (any float dtype) with A even, a bias of [1, L, L, 16] (``bias_head_major``: [16, 1, L, L]),
    no key mask, on sm_100 with L a multiple of 128 (the kernels' tile shapes and sample pairing, not a policy)."""
    if (not settings.current().augmented_attention_bf16_sm100) or mask is not None:
        return False
    if not q.is_cuda or q.dim() != 5 or not q.is_floating_point():
        return False
    A, B, L, h, d = q.shape
    if B != 1 or h != H or d != D or L % 128 or L == 0 or A % 2:
        return False
    if tuple(bias.shape) != ((H, 1, L, L) if bias_head_major else (1, L, L, H)):
        return False
    return _is_blackwell(_index(q))


_BUILD_FAILED = False


def available(
    q: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None = None,
    bias_head_major: bool = False,
) -> bool:
    """``supported()`` plus successful builds; a build failure raises."""
    global _BUILD_FAILED
    if _BUILD_FAILED or not supported(q, bias, mask, bias_head_major):
        return False
    from torch._subclasses.fake_tensor import FakeTensor

    if (
        torch.compiler.is_compiling()
        or isinstance(q, FakeTensor)
        or isinstance(bias, FakeTensor)
    ):
        return False
    try:
        for stem, func in (
            ("attn_fwd2", "augattn_fwd2_sm100"),
            ("attn_dqb", "augattn_dqb_sm100"),
            ("attn_dkv", "augattn_dkv_sm100"),
        ):
            _sm100_kernel(stem, func, _index(q))
    except Exception as exc:  # B200 has no engine Triton / silent PyTorch fallback: a build or load failure is an error
        raise RuntimeError(f"sm100 bf16 augmented attention kernels failed to build or load on B200: {exc!r}") from exc
    return True


# --------------------------------------------------------------------------------------------------- launches
def _tm(t, dims, stride_bytes, box, **kw):
    from .driver import TensorMap

    return TensorMap(t, dims, stride_bytes, box, **kw)


def _grid(items: int, device_index: int) -> tuple[int, int, int]:
    return (
        min(
            num_sms(device_index), items
        ),
        1,
        1,
    )


def _rs(t: torch.Tensor) -> int:
    """Row stride in bytes: q / k / v may be column views of a wider projection output (rows 16-byte aligned)."""
    assert (
        t.stride(1) == 1
        and (t.stride(0) * t.element_size()) % 16 == 0
        and t.data_ptr() % 16 == 0
    )
    return t.stride(0) * t.element_size()


#: token DiT head layouts the bf16 kernels take: (heads, head dim) -> d = heads x head dim (16 x 48 is the default build)
TDIT_HEADS = ((16, 48), (24, 32), (12, 64), (16, 64))


def _tdit_defs(heads: int = H, dh: int = D):
    """-D flags for a token DiT head layout (none for the default 16 x 48)."""
    if (heads, dh) == (H, D):
        return ()
    assert (heads, dh) in TDIT_HEADS, (heads, dh)
    return (f"NHEAD={heads}", f"DHP={dh}", f"RSQDV={dh**-0.5!r}f")


def forward(q2, k2, v2, bias_hll, A, L, heads: int = H, dh: int = D):
    """q2, k2, v2 [A L, heads dh] bf16 rows, bias [heads, L, L] bf16 (natural units) -> O [A L, heads dh] fp32, LSE [A, heads, L]
    (log2). Default 16 x 48 (d 768)."""
    dev = _index(q2)
    W = heads * dh
    O = torch.empty(A * L, W, device=q2.device, dtype=torch.float32)
    LSE = torch.empty(A, heads, L, device=q2.device, dtype=torch.float32)
    maps = (
        _tm(q2, [W, A * L], _rs(q2), [dh, 128]),
        _tm(k2, [W, A * L], _rs(k2), [dh, 64]),
        _tm(v2, [W, A * L], _rs(v2), [dh, 64]),
        _tm(bias_hll, [L, heads * L], L * 2, [64, 128]),
        _tm(O, [W, A * L], W * 4, [32, 128], swizzle=128, dtype="f32"),
        _tm(O, [W, A * L], W * 4, [16, 128], swizzle=64, dtype="f32"),
    )
    _sm100_kernel("attn_fwd2", "augattn_fwd2_sm100", dev, defs=_tdit_defs(heads, dh))(
        _grid((A // 2) * heads * (L // 128), dev),
        (384, 1, 1),
        *maps,
        O,
        LSE,
        int(L),
        int(A),
    )
    return O, LSE


def forward_tf32(q, k, v, bias_hll, A, L, heads: int = H, dh: int = D):
    """The fp32 training forward (``attn_fwd_tf32.cu``, TF32 tensor cores): q, k, v [A L, heads dh] fp32 rows, bias [heads, L, L]
    fp32 (natural units) -> O [A L, heads dh] fp32, LSE [A, heads, L] (log2). Default 16 x 48 (d 768); any of TDIT_HEADS (the
    kernel built with ``_tdit_defs``). A fp32 head row is 32-column 128-B-swizzled boxes and, at 48, a 16-column 64-B one: the
    16-column maps are encoded for every layout and read only where the head has that box."""
    dev = _index(q)
    W = heads * dh
    O = torch.empty(A * L, W, device=q.device, dtype=torch.float32)
    LSE = torch.empty(A, heads, L, device=q.device, dtype=torch.float32)
    f32 = dict(dtype="f32")
    dims, rs = [W, A * L], W * 4
    maps = (
        _tm(q, dims, _rs(q), [32, 128], swizzle=128, **f32),
        _tm(q, dims, _rs(q), [16, 128], swizzle=64, **f32),
        _tm(k, dims, _rs(k), [32, 32], swizzle=128, **f32),
        _tm(k, dims, _rs(k), [16, 32], swizzle=64, **f32),
        _tm(v, dims, _rs(v), [32, 32], swizzle="128a32", **f32),
        _tm(bias_hll, [L, heads * L], L * 4, [32, 128], swizzle=128, **f32),
        _tm(O, dims, rs, [32, 128], swizzle=128, **f32),
        _tm(O, dims, rs, [16, 128], swizzle=64, **f32),
    )
    _sm100_kernel(
        "attn_fwd_tf32", "augattn_fwd_tf32_sm100", dev, defs=_tdit_defs(heads, dh)
    )(
        _grid((A // 2) * heads * (L // 128), dev),
        (384, 1, 1),
        *maps,
        LSE,
        int(L),
        int(A),
    )
    return O, LSE


def _dqb_split(items: int, A: int, dev: int) -> int:
    """attn_dqb_tf32's sample groups: the largest NS dividing A with items x NS <= the SM count (1 when the items alone fill
    the GPU). L = 128 at 16 heads: 32 items -> NS = 4 at A = 48."""
    nsm = num_sms(dev)
    return max(
        [
            s
            for s in (1, 2, 3, 4, 6, 8, 12, 16)
            if A % s == 0 and (s == 1 or items * s <= nsm)
        ]
    )


def backward_tf32(
    q,
    k,
    v,
    do,
    bias_hll,
    LSE,
    Dd,
    A,
    L,
    heads: int = H,
    dh: int = D,
    dq=None,
    dk=None,
    dv=None,
):
    """The fp32 training backward (``attn_dkv_tf32.cu`` then ``attn_dqb_tf32.cu``, TF32 tensor cores): q, k, v, do [A L, heads dh]
    fp32 rows, bias [heads, L, L] fp32 as given to ``forward_tf32``, LSE from it, Dd = rowsum(dO O) [A, heads, L] fp32 -> dQ, dK, dV
    [A L, heads dh] fp32 and dbias [heads, L, L] fp32. Default 16 x 48 (d 768); any of TDIT_HEADS. ``dq`` / ``dk`` / ``dv``: where
    to write them (fp32 [A L, heads dh] with unit column stride and 16-byte rows, e.g. column blocks of the caller's q|k|v|g
    gradient), else fresh buffers."""
    dev = _index(q)
    W = heads * dh
    out = (
        lambda t: torch.empty(A * L, W, device=q.device, dtype=torch.float32)
        if t is None
        else t
    )  # noqa: E731
    DQ, DK, DV = out(dq), out(dk), out(dv)
    for t in (DQ, DK, DV):
        assert t.dtype == torch.float32 and tuple(t.shape) == (A * L, W)
        _rs(t)
    items = heads * (L // 128) * (L // 64)
    ns = _dqb_split(items, A, dev)
    # sample groups add their dbias into it (attn_dqb_tf32 -DNS > 1): zeroed then, else written once
    DB = (torch.zeros if ns > 1 else torch.empty)(
        heads, L, L, device=q.device, dtype=torch.float32
    )
    f32 = dict(dtype="f32")
    dims = [W, A * L]
    a = lambda t, rows: _tm(t, dims, _rs(t), [32, rows], swizzle=128, **f32)  # noqa: E731  K-major, columns 0-31
    b = lambda t, rows: _tm(t, dims, _rs(t), [16, rows], swizzle=64, **f32)  # noqa: E731  K-major, columns 32-47
    m = lambda t, rows: _tm(t, dims, _rs(t), [32, rows], swizzle="128a32", **f32)  # noqa: E731  MN-major, 32-column atoms
    dkv_maps = (
        a(q, 32),
        b(q, 32),
        m(q, 32),
        a(do, 32),
        b(do, 32),
        m(do, 32),
        a(k, 128),
        b(k, 128),
        a(v, 128),
        b(v, 128),
        _tm(bias_hll, [L, heads * L], L * 4, [32, 32], swizzle=0, **f32),
        a(DK, 128),
        b(DK, 128),
        a(DV, 128),
        b(DV, 128),
    )
    dqb_maps = (
        a(q, 128),
        b(q, 128),
        a(do, 128),
        b(do, 128),
        a(v, 64),
        b(v, 64),
        a(k, 64),
        b(k, 64),
        m(k, 64),
        _tm(bias_hll, [L, heads * L], L * 4, [32, 128], swizzle=128, **f32),
    )
    defs = _tdit_defs(heads, dh)
    ldq = DQ.stride(0)
    # attn_dkv_tf32 zero-fills dQ on the way; attn_dqb_tf32 then adds one partial per 64-key chunk (and sample group).
    _sm100_kernel("attn_dkv_tf32", "augattn_dkv_tf32_sm100", dev, defs=defs)(
        _grid(A * heads * (L // 128), dev),
        (384, 1, 1),
        *dkv_maps,
        LSE,
        Dd,
        DQ,
        int(L),
        int(A),
        int(ldq),
    )
    _sm100_kernel("attn_dqb_tf32", "augattn_dqb_tf32_sm100", dev, defs=defs)(
        _grid(items * ns, dev),
        (384, 1, 1),
        *dqb_maps,
        LSE,
        Dd,
        DQ,
        DB,
        int(L),
        int(A),
        int(ns),
        int(ldq),
    )
    return DQ, DK, DV, DB


def backward(q2, k2, v2, dob, bias_hll, LSE, Dd, A, L, heads: int = H, dh: int = D):
    """dQ, dK, dV [A L, heads dh] fp32 and dbias [heads, L, L] fp32. dob [A L, heads dh] bf16, Dd = rowsum(dO O) [A, heads, L]
    fp32. Default 16 x 48 (d 768)."""
    dev = _index(q2)
    W = heads * dh
    bias_t = bias_transpose(bias_hll)
    DQ = torch.empty(A * L, W, device=q2.device, dtype=torch.float32)
    DB = torch.empty(heads, L, L, device=q2.device, dtype=torch.float32)
    DK, DV = torch.empty_like(DQ), torch.empty_like(DQ)
    f32 = dict(dtype="f32")
    dims = [W, A * L]
    dkv_maps = (
        _tm(q2, dims, _rs(q2), [dh, 64]),
        _tm(k2, dims, _rs(k2), [dh, 128]),
        _tm(v2, dims, _rs(v2), [dh, 128]),
        _tm(dob, dims, _rs(dob), [dh, 64]),
        _tm(bias_t, [L, heads * L], L * 2, [64, 128]),
        _tm(DK, dims, W * 4, [32, 128], swizzle=128, **f32),
        _tm(DK, dims, W * 4, [16, 128], swizzle=64, **f32),
        _tm(DV, dims, W * 4, [32, 128], swizzle=128, **f32),
        _tm(DV, dims, W * 4, [16, 128], swizzle=64, **f32),
    )
    dqb_maps = (
        *(_tm(t, dims, _rs(t), [dh, 128]) for t in (q2, k2, v2, dob)),
        _tm(bias_hll, [L, heads * L], L * 2, [64, 128]),
        _tm(DQ, dims, W * 4, [32, 128], swizzle=128, **f32),
        _tm(DQ, dims, W * 4, [16, 128], swizzle=64, **f32),
    )
    defs = _tdit_defs(heads, dh)
    # attn_dkv zero-fills dQ on the way; attn_dqb then adds one partial per 128-key chunk.
    _sm100_kernel("attn_dkv", "augattn_dkv_sm100", dev, defs=defs)(
        _grid(A * heads * (L // 128), dev),
        (384, 1, 1),
        *dkv_maps,
        LSE,
        Dd,
        DK,
        DV,
        DQ,
        int(L),
        int(A),
    )
    _sm100_kernel("attn_dqb", "augattn_dqb_sm100", dev, defs=defs)(
        _grid(heads * (L // 128) * (L // 128), dev),
        (384, 1, 1),
        *dqb_maps,
        LSE,
        Dd,
        DQ,
        DB,
        int(L),
        int(A),
    )
    return DQ, DK, DV, DB


# --------------------------------------------------------------------------------------------------- glue
def prep_do(do, O, A, L):
    """dO -> bf16 and D = rowsum(dO O) as [A, H, L] fp32 (``glue.cu``)."""
    do = do.reshape(A * L, H * D).float().contiguous()
    dob = torch.empty(A * L, H * D, device=do.device, dtype=torch.bfloat16)
    dd = torch.empty(A, H, L, device=do.device, dtype=torch.float32)
    _sm100_kernel("glue", "prep_do_bf16", _index(do), smem=0)(
        ((A * L * H + 255) // 256, 1, 1),
        (256, 1, 1),
        do,
        O,
        dob,
        dd,
        int(A * L),
        int(L),
    )
    return dob, dd


def bias_transpose(bias):
    """[H, L, L] bf16 -> [H, L(key), L(query)] (attn_dkv reads its key rows contiguously; ``glue.cu``)."""
    Hh, L, _ = bias.shape
    out = torch.empty_like(bias)
    _sm100_kernel("glue", "bias_transpose_bf16", _index(bias), smem=0)(
        (L // 32, L // 32, Hh), (32, 8, 1), bias, out, int(L)
    )
    return out


# --------------------------------------------------------------------------------------------------- op
class _AttentionBf16Sm100(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, bias_head_major):  # noqa: D102
        A, _, L, _, _ = q.shape
        q2, k2, v2 = (
            t.reshape(A * L, H * D).to(torch.bfloat16).contiguous() for t in (q, k, v)
        )
        bb = (
            (bias.reshape(H, L, L) if bias_head_major else bias[0].permute(2, 0, 1))
            .to(torch.bfloat16)
            .contiguous()
        )
        O, LSE = forward(q2, k2, v2, bb, A, L)
        if torch.is_grad_enabled() or any(ctx.needs_input_grad):
            ctx.save_for_backward(q2, k2, v2, bb, O, LSE)
        ctx.head_major, ctx.dims = bias_head_major, (A, L)
        return O.view(A, 1, L, H, D)

    @staticmethod
    def backward(ctx, do):  # noqa: D102
        q2, k2, v2, bb, O, LSE = ctx.saved_tensors
        A, L = ctx.dims
        dob, dd = prep_do(do, O, A, L)
        DQ, DK, DV, DB = backward(q2, k2, v2, dob, bb, LSE, dd, A, L)
        shp = (A, 1, L, H, D)
        dbias = DB.unsqueeze(1) if ctx.head_major else DB.permute(1, 2, 0).unsqueeze(0)
        return DQ.view(shp), DK.view(shp), DV.view(shp), dbias, None


def augmented_attention_bf16_sm100(q, k, v, bias, *, bias_head_major: bool = False):
    """``softmax(q k^T / sqrt(48) + bias) v`` in bf16 with fp32 accumulation; returns fp32 [A, 1, L, 16, 48].

    q, k, v: [A, 1, L, 16, 48] (fp32 or bf16). bias: [1, L, L, 16], or [16, 1, L, L] with ``bias_head_major``.
    Call ``available()`` first."""
    return _AttentionBf16Sm100.apply(q, k, v, bias, bias_head_major)


# --------------------------------------------------------------------------------------------------- inference core
class GatedInferenceCore:
    """The fused token DiT step's core on sm_100a: reads the block's head-major hoisted bias ([nb H, L, L] rows) and the
    step's projections (logits pre-scaled into exp2 units), writes sigmoid(g) * o over q.

    * bf16 (``attn_inf.cu``): ``qkvg`` [S L, 4 D], q | k | v | g column views of one GEMM output.
    * fp32 with TF32 tensor cores (``attn_inf_tf32.cu``): ``qkvg`` is q | k | g, [S L, 3 D], and ``vt`` is v transposed,
      [D, S L] -- a kind::tf32 MMA takes v only K-major. Every 48-wide fp32 q / k / g tile is a 32-column 128-B-swizzled
      box and a 16-column 64-B-swizzled one (32: one 32-column box, 64: two); the v^T tile is dh rows x 32 keys.

    Both at every layout of TDIT_HEADS. Bindings (TMA descriptors) are cached per buffer."""

    def __init__(
        self,
        device_index: int,
        dtype: torch.dtype = torch.bfloat16,
        heads: int = H,
        dh: int = D,
    ):
        self.fp32 = dtype is torch.float32
        self.heads, self.dh = heads, dh
        assert (heads, dh) in TDIT_HEADS, (heads, dh)
        self.k = (
            _sm100_kernel(
                "attn_inf_tf32",
                "augattn_inf_tf32_sm100",
                device_index,
                defs=_tdit_defs(heads, dh),
            )
            if self.fp32
            else _sm100_kernel(
                "attn_inf",
                "augattn_inf_sm100",
                device_index,
                defs=_tdit_defs(heads, dh),
            )
        )
        self.nsm = num_sms(device_index)
        self.runs: dict = {}

    def _bind(self, qkvg, vt, bias, block, S, L):
        # 3-D maps (columns, row of the sample, sample | head): the last tile's rows past L load as zeros and are clipped
        # on the store, so L need only keep the rows 16-byte aligned (a multiple of 8)
        M, W = qkvg.shape
        es = qkvg.element_size()
        rs = W * es
        Hh = self.heads
        bv = bias[block * Hh : (block + 1) * Hh]
        grid = (min(self.nsm, ((S + 1) // 2) * Hh * -(-L // 128)), 1, 1)
        bias_map = lambda box, **kw: _tm(
            bv, [L, L, Hh], [L * es, L * L * es], [*box, 1], **kw
        )  # noqa: E731
        if self.fp32:
            Dm = W // 3
            q, k, g = (qkvg[:, i * Dm : (i + 1) * Dm] for i in range(3))
            f32 = dict(dtype="f32")
            a = lambda t, rows: _tm(
                t, [Dm, L, S], [rs, L * rs], [32, rows, 1], swizzle=128, **f32
            )  # noqa: E731
            b = lambda t, rows: _tm(
                t, [Dm, L, S], [rs, L * rs], [16, rows, 1], swizzle=64, **f32
            )  # noqa: E731
            maps = (
                a(q, 128),
                b(q, 128),
                a(k, 32),
                b(k, 32),
                _tm(
                    vt, [L, S, Dm], [L * 4, M * 4], [32, 1, self.dh], swizzle=128, **f32
                ),  # v^T: key, sample, channel
                bias_map([32, 128], swizzle=128, **f32),
                a(g, 128),
                b(g, 128),
            )

            def run():
                self.k(grid, (384, 1, 1), *maps, int(L), int(S))
        else:
            Dm, DH = W // 4, self.dh
            q, k, v, g = (qkvg[:, i * Dm : (i + 1) * Dm] for i in range(4))
            t3 = lambda t, rows: _tm(t, [Dm, L, S], [rs, L * rs], [DH, rows, 1])  # noqa: E731
            maps = (t3(q, 128), t3(k, 64), t3(v, 64), bias_map([64, 128]), t3(g, 128))
            mq, mk, mv, mb, mg = maps

            def run():
                self.k(grid, (384, 1, 1), mq, mk, mv, mb, mg, mq, int(L), int(S))

        run.keep = (maps, qkvg, vt, bias)
        return run

    def __call__(self, qkvg, bias, block, S, vt=None):
        """``vt`` (fp32 only): v^T [D, S L]; ``qkvg`` is then q | k | g."""
        assert self.fp32 == (vt is not None), (
            "the TF32 core takes q | k | g and v^T; the bf16 core q | k | v | g"
        )
        L = qkvg.shape[0] // S
        key = (
            qkvg.data_ptr(),
            None if vt is None else vt.data_ptr(),
            bias.data_ptr(),
            block,
            S,
            L,
        )
        run = self.runs.get(key)
        if run is None:
            run = self.runs[key] = self._bind(qkvg, vt, bias, block, S, L)
        run()
        return qkvg


# --------------------------------------------------------------------------------------------------- AttentionPairBias
# One sample, d_single 384 as 8 heads x 48, 12 x 32, 24 x 16 (-DDHP=16) or 16 x 24, or d_single 512 as 16 x 32: the same kernels built with -DQPAIR=1 (a work item pairs two 128-query
# tiles of a head, sharing its k / v blocks; attn_inf, attn_fwd2), -DNHEAD (attn_dkv, attn_dqb: A = 1 needs no other change) and,
# for 16 x 24, -DDHP=32: each head's rows are 32 wide in memory and in the MMAs, the caller's projection writing its 24 channels
# and 8 zeros (so the pads of q, k, v, dO, and with them of O, dQ, dK, dV, are exactly 0); -DRSQDV is 1 / sqrt(24).
# (heads, d_single) -> (head width in memory, real head dim); 16 x 32 at d_single 512
APB_GEOMETRY = {
    (8, 384): (48, 48),
    (12, 384): (32, 32),
    (16, 384): (32, 24),
    (24, 384): (16, 16),
    (16, 512): (32, 32),
    # the token DiT's head layouts (TDIT_HEADS at d 768 / 1024): its per-sample-bias training block (``token_dit_train``, B200
    # psb) runs these BPS / QPAIR builds with every sample's own bias planes
    (16, 768): (48, 48),
    (24, 768): (32, 32),
    (12, 768): (64, 64),
    (16, 1024): (64, 64),
}


def apb_width(heads: int, d: int = 384) -> int:
    """Row width of q / k / v / g / O (d_single, or 512 for 16 x 24 padded to 32)."""
    return heads * APB_GEOMETRY[(heads, d)][0]


def _apb_defs(heads: int, qpair: bool, d: int = 384, ksplit: bool = False):
    """-D flags of an AttentionPairBias build: every core takes per-sample bias planes (BPS; the backward cores' flag, the
    forward cores' QPAIR mode)."""
    dhp, dreal = APB_GEOMETRY[(heads, d)]
    defs = (("QPAIR=1",) if qpair else ("BPS=1",)) + (f"NHEAD={heads}",)
    if dhp != 48 or dreal != 48:
        defs += (f"DHP={dhp}", f"RSQDV={dreal**-0.5!r}f")
    return defs + (("KSPLIT=1",) if ksplit else ())


#: most chunks a work item's keys are split into (``apb_rows`` ksplit_combine's NSMAX)
KSPLIT_MAX = 16


def apb_ksplit(L: int, heads: int, d: int = 384, nsm: int = 148) -> int:
    """Key chunks per work item of the forward / inference cores at (L, heads): a function of the layout and L only (never of
    the batch, so a batch of B samples is bitwise B single-sample calls). ``settings.apb_b200_ksplit`` overrides it (1: the
    unsplit cores). The work items of one sample are (heads x query-tile pairs); a split into ns chunks makes ns times as many,
    each 1 / ns as long, so the core takes ~ waves(ns) / ns of the unsplit time, waves = ceil(items ns / ``nsm``). The choice is
    the smallest ns (a divisor of the key blocks ceil(L / 64), each item keeping at least two) that minimises it, kept only when it
    at least halves the unsplit estimate (the partials and the combine pass cost ~4-5 us: two waves of a third each, 16 heads at
    L1536, measured even). E.g. 8 heads L768 (24 items) ns 6, 16 heads L768 (48) ns 3, 16 heads L1536 (96) unsplit."""
    nbt = -(-L // 64)
    forced = settings.current().apb_b200_ksplit
    if forced is not None:
        if forced < 1 or forced > KSPLIT_MAX or nbt % forced:
            raise ValueError(f"apb_b200_ksplit={forced} must divide the {nbt} key blocks of L={L} (at most {KSPLIT_MAX})")
        return int(forced)
    base = ((-(-L // 128) + 1) // 2) * heads
    best, best_cost = 1, -(-base // nsm)
    for ns in range(2, KSPLIT_MAX + 1):
        if nbt % ns or nbt // ns < 2:
            continue
        cost = -(-base * ns // nsm) / ns
        if cost < best_cost - 1e-9:
            best, best_cost = ns, cost
    return best if best_cost <= 0.5 * -(-base // nsm) + 1e-9 else 1


def apb_qsplit(L: int, heads: int, nsm: int = 148) -> int:
    """Query chunks per work item of the backward dK / dV core (``attn_dkv -DQSPLIT``): one sample has heads x L / 128 work
    items (key tiles); the same waves / chunks rule as ``apb_ksplit`` over the divisors of L / 128. ``settings.apb_b200_qsplit``
    overrides it (1: the unsplit core)."""
    kt = L // 128
    forced = settings.current().apb_b200_qsplit
    if forced is not None:
        if forced < 1 or kt % forced:
            raise ValueError(f"apb_b200_qsplit={forced} must divide the {kt} key tiles of L={L}")
        return int(forced)
    base = heads * kt
    best, best_cost = 1, -(-base // nsm)
    for nq in range(2, kt + 1):
        if kt % nq:
            continue
        cost = -(-base * nq // nsm) / nq
        if cost < best_cost - 1e-9:
            best, best_cost = nq, cost
    return best


def _rows_ext():
    from ..apb import ext

    return ext()


_apb_runs: dict = {}


def _apb_bound(key, build):
    """Prepared launches cached by the buffers' addresses (and L): in steady state the caching allocator hands the same
    addresses back every step, so the TMA descriptors are encoded once. The descriptors hold no tensors: a key match means
    the current tensors sit at exactly those addresses."""
    run = _apb_runs.get(key)
    if run is None:
        if len(_apb_runs) >= 64:
            _apb_runs.clear()
        run = _apb_runs[key] = build()
    return run


def _tmn(t, dims, stride_bytes, box, **kw):
    m = _tm(t, dims, stride_bytes, box, **kw)
    m.keep = None
    return m


class ApbInferenceCore:
    """AttentionPairBias's inference core: ``qkvg`` [S L, 4 W] bf16 (q | k | v | g, W = ``apb_width(heads)``, logits pre-scaled
    into exp2 units), the head-major bias [S heads, L, L] bf16 (exp2 units, masked keys very negative); sigmoid(g) * o written
    over q. With a key split (``apb_ksplit``) the core leaves per-chunk partials and ``ksplit_combine`` gates them over q."""

    def __init__(self, device_index: int, heads: int = 8, d: int = 384):
        self.heads, self.d, self.dhp = heads, d, APB_GEOMETRY[(heads, d)][0]
        self.dev = device_index
        self.nsm = num_sms(device_index)

    def _kernel(self, ksplit: bool):
        return _sm100_kernel(
            "attn_inf",
            "augattn_inf_sm100",
            self.dev,
            defs=_apb_defs(self.heads, True, self.d, ksplit),
        )

    def _bind(self, qkvg, bias, L, S, ns, op, lp):
        W, es, H_, dh = qkvg.shape[1], qkvg.element_size(), self.heads, self.dhp
        Dm = W // 4
        q, k, v, g = (qkvg[:, i * Dm : (i + 1) * Dm] for i in range(4))
        t3 = lambda t, rows: _tmn(t, [Dm, L, S], [W * es, L * W * es], [dh, rows, 1])  # noqa: E731
        mq, mk, mv, mg = t3(q, 128), t3(k, 64), t3(v, 64), t3(g, 128)
        mb = _tmn(bias, [L, L, S * H_], [L * es, L * L * es], [64, 128, 1])
        grid = (min(self.nsm, ((-(-L // 128) + 1) // 2) * H_ * ns * S), 1, 1)
        if ns == 1:
            return self._kernel(False).bind(grid, (384, 1, 1), mq, mk, mv, mb, mg, mq, int(L), int(S))
        f32 = dict(dtype="f32")
        dims, st = [Dm, L, ns * S], [Dm * 4, L * Dm * 4]
        mpa = _tmn(op, dims, st, [32, 128, 1], swizzle=128, **f32)
        mpb = _tmn(op, dims, st, [16, 128, 1], swizzle=64, **f32)
        return self._kernel(True).bind(
            grid, (384, 1, 1), mq, mk, mv, mb, mg, mq, int(L), int(S), mpa, mpb, lp, int(ns)
        )

    def __call__(self, qkvg, bias, L: int | None = None):
        L = qkvg.shape[0] if L is None else L
        S = qkvg.shape[0] // L
        W = apb_width(self.heads, self.d)
        assert qkvg.shape[1] == 4 * W and bias.shape[0] == S * self.heads and S * L == qkvg.shape[0]
        ns = apb_ksplit(L, self.heads, self.d, self.nsm)
        op = lp = None
        if ns > 1:
            op = torch.empty(ns, S * L, W, device=qkvg.device, dtype=torch.float32)
            lp = torch.empty(ns, S, self.heads, L, device=qkvg.device, dtype=torch.float32)
        _apb_bound(
            (
                "inf",
                self.heads,
                self.d,
                qkvg.device.index,
                qkvg.data_ptr(),
                bias.data_ptr(),
                L,
                S,
                ns,
                None if op is None else op.data_ptr(),
                None if lp is None else lp.data_ptr(),
            ),
            lambda: self._bind(qkvg, bias, L, S, ns, op, lp),
        )()
        if ns > 1:
            _rows_ext().ksplit_combine(op, lp, qkvg, None, None, None, L, self.heads, self.dhp)
        return qkvg


def apb_forward(q, k, v, bias, L, heads: int = 8, d: int = 384):
    """AttentionPairBias training forward (``attn_fwd2 -DQPAIR``): q, k, v [S L, W] bf16 rows (W = ``apb_width(heads)``, column
    views welcome), bias [S heads, L, L] bf16 natural units -> O [S L, W] fp32, LSE [S heads, L] (log2)."""
    dev = _index(q)
    dh, W = APB_GEOMETRY[(heads, d)][0], apb_width(heads, d)
    M = q.shape[0]
    S = M // L
    O = torch.empty(M, W, device=q.device, dtype=torch.float32)
    LSE = torch.empty(S * heads, L, device=q.device, dtype=torch.float32)

    def build():
        maps = (
            _tmn(q, [W, M], _rs(q), [dh, 128]),
            _tmn(k, [W, M], _rs(k), [dh, 64]),
            _tmn(v, [W, M], _rs(v), [dh, 64]),
            _tmn(bias, [L, S * heads * L], L * 2, [64, 128]),
            _tmn(O, [W, M], W * 4, [32, 128], swizzle=128, dtype="f32"),
            _tmn(O, [W, M], W * 4, [16, 128], swizzle=64, dtype="f32"),
        )
        grid = _grid(((L // 128 + 1) // 2) * heads * S, dev)
        return _sm100_kernel(
            "attn_fwd2", "augattn_fwd2_sm100", dev, defs=_apb_defs(heads, True, d)
        ).bind(grid, (384, 1, 1), *maps, O, LSE, int(L), int(S))

    key = (
        "fwd",
        heads,
        d,
        dev,
        L,
        S,
        *(t.data_ptr() for t in (q, k, v, bias, O, LSE)),
        _rs(q),
        _rs(k),
        _rs(v),
    )
    _apb_bound(key, build)()
    return O, LSE


def apb_forward_og(qkvg, bias, L, heads: int = 8, d: int = 384):
    """The training forward with the gate: ``qkvg`` [S L, 4 W] bf16 (q | k | v | g, natural units), bias [S heads, L, L] bf16
    -> O [S L, W] fp32, LSE [S heads, L] (log2), og = sigmoid(g) O [S L, W] bf16. Unsplit (``apb_ksplit`` 1): ``attn_fwd2``
    then ``gate_rows``; split: ``attn_fwd2 -DKSPLIT`` (per-chunk partials) then ``ksplit_combine`` (O, LSE and og in one pass)."""
    dev = _index(qkvg)
    dh, W = APB_GEOMETRY[(heads, d)][0], apb_width(heads, d)
    M = qkvg.shape[0]
    S = M // L
    q, k, v = qkvg[:, :W], qkvg[:, W : 2 * W], qkvg[:, 2 * W : 3 * W]
    ns = apb_ksplit(L, heads, d, num_sms(dev))
    og = torch.empty(M, W, device=qkvg.device, dtype=torch.bfloat16)
    if ns == 1:
        O, LSE = apb_forward(q, k, v, bias, L, heads, d)
        _rows_ext().gate_rows(O, qkvg, og)
        return O, LSE, og
    OP = torch.empty(ns, M, W, device=qkvg.device, dtype=torch.float32)
    LP = torch.empty(ns, S, heads, L, device=qkvg.device, dtype=torch.float32)

    def build():
        maps = (
            _tmn(q, [W, M], _rs(q), [dh, 128]),
            _tmn(k, [W, M], _rs(k), [dh, 64]),
            _tmn(v, [W, M], _rs(v), [dh, 64]),
            _tmn(bias, [L, S * heads * L], L * 2, [64, 128]),
            _tmn(OP, [W, ns * M], W * 4, [32, 128], swizzle=128, dtype="f32"),
            _tmn(OP, [W, ns * M], W * 4, [16, 128], swizzle=64, dtype="f32"),
        )
        grid = _grid(((L // 128 + 1) // 2) * heads * ns * S, dev)
        return _sm100_kernel(
            "attn_fwd2", "augattn_fwd2_sm100", dev, defs=_apb_defs(heads, True, d, True)
        ).bind(grid, (384, 1, 1), *maps, OP, LP, int(L), int(S), int(ns))

    key = ("fwd_ks", heads, d, dev, L, S, ns, *(t.data_ptr() for t in (qkvg, bias, OP, LP)))
    _apb_bound(key, build)()
    O = torch.empty(M, W, device=qkvg.device, dtype=torch.float32)
    LSE = torch.empty(S * heads, L, device=qkvg.device, dtype=torch.float32)
    _rows_ext().ksplit_combine(OP, LP, qkvg, O, LSE, og, L, heads, dh)
    return O, LSE, og


def apb_backward(q, k, v, dob, bias, LSE, Dd, L, heads: int = 8, d: int = 384):
    """AttentionPairBias training backward (``attn_dkv``, ``attn_dqb`` with -DBPS -DNHEAD, A = S samples): q, k, v, dob [S L, W]
    bf16 rows, bias [S heads, L, L] bf16 as given to ``apb_forward``, LSE from it, Dd = rowsum(dO O) [S heads, L] fp32 -> dQ, dK,
    dV [S L, W] fp32 and dbias [S heads, L, L] fp32. dQ comes as L / 128 key-chunk partials [L / 128, S L, W] (``attn_dqb
    -DDQPART=1`` stores each chunk's slice instead of reduce-adding into one buffer, so the sum -- the caller's, in chunk order --
    is deterministic)."""
    dev = _index(q)
    dh, W = APB_GEOMETRY[(heads, d)][0], apb_width(heads, d)
    nch = L // 128
    M = q.shape[0]
    S = M // L
    nq = apb_qsplit(L, heads, torch.cuda.get_device_properties(dev).multi_processor_count)
    bias_t = torch.empty_like(bias)
    DQ = torch.empty(nch, M, W, device=q.device, dtype=torch.float32)
    DB = torch.empty(S * heads, L, L, device=q.device, dtype=torch.float32)
    DK, DV = (torch.empty(nq, M, W, device=q.device, dtype=torch.float32) for _ in range(2))

    def build():
        f32 = dict(dtype="f32")
        dims = [W, M]
        dkv_maps = (
            _tmn(q, dims, _rs(q), [dh, 64]),
            _tmn(k, dims, _rs(k), [dh, 128]),
            _tmn(v, dims, _rs(v), [dh, 128]),
            _tmn(dob, dims, _rs(dob), [dh, 64]),
            _tmn(bias_t, [L, S * heads * L], L * 2, [64, 128]),
            _tmn(DK, [W, nq * M], W * 4, [32, 128], swizzle=128, **f32),
            _tmn(DK, [W, nq * M], W * 4, [16, 128], swizzle=64, **f32),
            _tmn(DV, [W, nq * M], W * 4, [32, 128], swizzle=128, **f32),
            _tmn(DV, [W, nq * M], W * 4, [16, 128], swizzle=64, **f32),
        )
        dq_dims = [W, nch * M]
        dqb_maps = (
            *(_tmn(t, dims, _rs(t), [dh, 128]) for t in (q, k, v, dob)),
            _tmn(bias, [L, S * heads * L], L * 2, [64, 128]),
            _tmn(DQ, dq_dims, W * 4, [32, 128], swizzle=128, **f32),
            _tmn(DQ, dq_dims, W * 4, [16, 128], swizzle=64, **f32),
        )
        defs = _apb_defs(heads, False, d)
        tr = _sm100_kernel("glue", "bias_transpose_bf16", dev, smem=0).bind(
            (L // 32, L // 32, S * heads), (32, 8, 1), bias, bias_t, int(L)
        )
        dkv = _sm100_kernel(
            "attn_dkv", "augattn_dkv_sm100", dev, defs=defs + (("QSPLIT=1",) if nq > 1 else ())
        ).bind(
            _grid(S * heads * (L // 128) * nq, dev),
            (384, 1, 1),
            *dkv_maps,
            LSE,
            Dd,
            DK,
            DV,
            None,
            int(L),
            int(S),
            *((int(nq),) if nq > 1 else ()),
        )  # no dQ zero fill
        dqb = _sm100_kernel(
            "attn_dqb", "augattn_dqb_sm100", dev, defs=defs + ("DQPART=1",)
        ).bind(
            _grid(S * heads * (L // 128) * (L // 128), dev),
            (384, 1, 1),
            *dqb_maps,
            LSE,
            Dd,
            DQ,
            DB,
            int(L),
            int(S),
        )

        def run():
            tr()
            dkv()
            dqb()

        run.keep = (tr, dkv, dqb)
        return run

    key = (
        "bwd",
        heads,
        d,
        dev,
        L,
        S,
        nq,
        *(t.data_ptr() for t in (q, k, v, dob, bias, LSE, Dd, bias_t, DQ, DB, DK, DV)),
        _rs(q),
        _rs(k),
        _rs(v),
        _rs(dob),
    )
    _apb_bound(key, build)()
    return DQ, DK, DV, DB


def inference_core_supported(
    dtype: torch.dtype, L: int, d: int, h: int, device_index: int
) -> bool:
    """bf16, or fp32 (TF32 MMA), at a head layout of TDIT_HEADS (d = heads x head dim); L a multiple of 8 (the fused step pads
    other lengths to one), sm_100."""
    if (not settings.current().augmented_attention_bf16_sm100) or d % h:
        return False
    ok = dtype in (torch.float32, torch.bfloat16) and (h, d // h) in TDIT_HEADS
    return ok and L % 8 == 0 and L >= 8 and _is_blackwell(device_index)


__all__ = [
    "APB_GEOMETRY",
    "TDIT_HEADS",
    "ApbInferenceCore",
    "GatedInferenceCore",
    "apb_backward",
    "apb_forward",
    "apb_forward_og",
    "apb_ksplit",
    "apb_qsplit",
    "apb_width",
    "augmented_attention_bf16_sm100",
    "available",
    "cubin",
    "inference_core_supported",
    "supported",
]
