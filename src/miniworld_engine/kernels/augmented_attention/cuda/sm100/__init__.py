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

import collections
import functools
import os
from pathlib import Path

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._device import num_sms
from miniworld_engine.kernels._nvcc import build_failed, engine_jit_root

H, D = 16, 48
_dir = Path(__file__).parent
#: B200 fp32 token DiT inference at one sample: the TF32 core's query-tile-pair build (0: the sample-pair schedule)
_TF32_QPAIR = os.environ.get("MINIWORLD_TOKEN_B200_TF32_QPAIR", "1") != "0"
#: B200 fp32 token DiT inference at one sample (2026-11-17, AttentionPairBias part 7): the TF32 core's key split (``attn_inf_tf32 -DQPAIR
#: -DKSPLIT`` + ``ksplit_combine32``) where the query-tile pairs leave most SMs idle. MINIWORLD_TOKEN_B200_TF32_KSPLIT=n forces n chunks (1: off).
_TF32_KSPLIT = os.environ.get("MINIWORLD_TOKEN_B200_TF32_KSPLIT")
_KS32_NS = (2, 3, 4, 5, 6, 8, 10, 12, 16)  # chunk counts ksplit_combine32 is built for


def tf32_ksplit(L: int, heads: int, nsm: int = 148) -> int:
    """Key chunks per work item of the one-sample TF32 inference core: the largest divisor ns of the 32-key blocks ceil(L / 32) with
    items x ns <= nsm, when the unsplit core has fewer than nsm / 2 items (items = heads x ceil(ceil(L / 128) / 2)); else 1."""
    nbt, items = -(-L // 32), heads * ((-(-L // 128) + 1) // 2)
    if _TF32_KSPLIT is not None:
        f = int(_TF32_KSPLIT)
        return f if f in _KS32_NS and nbt % f == 0 else 1
    if 2 * items >= nsm:
        return 1
    best = 1
    for ns in _KS32_NS:
        if nbt % ns == 0 and items * ns <= nsm:
            best = ns
    return best


_SMEM = 232448
_NVCC_FLAGS = ("-std=c++17", "-O3", "-arch=sm_100a", "--cubin", "-lineinfo")


# --------------------------------------------------------------------------------------------------- build
def _jit_root() -> Path:
    return engine_jit_root() / "augattn_sm100"


@functools.lru_cache(maxsize=None)
def cubin(stem: str, defs: tuple[str, ...] = (), src_dir: str | None = None) -> str:
    """Path of ``<src_dir>/<stem>.cu`` (default: this directory) built for sm_100a with ``-D`` ``defs`` and this
    directory's ``sm100.cuh`` on the include path; rebuilt only when a source or a flag changes."""
    from miniworld_engine.kernels import _nvcc

    src = Path(src_dir or _dir) / f"{stem}.cu"
    flags = (*_NVCC_FLAGS, f"-I{_dir}", *("-D" + d for d in defs))
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
):
    from . import driver

    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem, defs, src_dir=src_dir), func, smem, cluster=cluster, pdl=pdl)


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
    if mask is not None:
        return False
    if not q.is_cuda or q.dim() != 5 or not q.is_floating_point():
        return False
    A, B, L, h, d = q.shape
    if B != 1 or h != H or d != D or L % 128 or L == 0 or A % 2:
        return False
    if tuple(bias.shape) != ((H, 1, L, L) if bias_head_major else (1, L, L, H)):
        return False
    return _is_blackwell(_index(q))


def available(
    q: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None = None,
    bias_head_major: bool = False,
) -> bool:
    """``supported()`` plus successful builds; a failed build / load raises ``_nvcc.KernelBuildError`` (never a fallback)."""
    if not supported(q, bias, mask, bias_head_major):
        return False
    from torch._subclasses.fake_tensor import FakeTensor

    if torch.compiler.is_compiling() or isinstance(q, FakeTensor) or isinstance(bias, FakeTensor):
        return False
    try:
        for stem, func in (
            ("attn_fwd2", "augattn_fwd2_sm100"),
            ("attn_dqb", "augattn_dqb_sm100"),
            ("attn_dkv", "augattn_dkv_sm100"),
        ):
            _sm100_kernel(stem, func, _index(q))
    except Exception as exc:  # a build / load failure is an error, never a fallback
        build_failed("sm100 bf16 augmented attention", exc)
        return False  # reached only while planning (build_failed returns there)
    return True


# --------------------------------------------------------------------------------------------------- launches
def _tm(t, dims, stride_bytes, box, **kw):
    from .driver import TensorMap

    return TensorMap(t, dims, stride_bytes, box, **kw)


def _grid(items: int, device_index: int) -> tuple[int, int, int]:
    return (
        min(num_sms(device_index), items),
        1,
        1,
    )


def _rs(t: torch.Tensor) -> int:
    """Row stride in bytes: q / k / v may be column views of a wider projection output (rows 16-byte aligned)."""
    assert t.stride(1) == 1 and (t.stride(0) * t.element_size()) % 16 == 0 and t.data_ptr() % 16 == 0
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
    # A = 1 (a training block of one sample): the -DQPAIR build pairs two 128-query tiles of a head (AttentionPairBias's schedule) where
    # the sample-pair schedule needed a zero partner sample -- the whole block padded to A = 2 (2026-11-17)
    qpair = A == 1
    _sm100_kernel("attn_fwd2", "augattn_fwd2_sm100", dev, defs=((("QPAIR=1",) if qpair else ()) + _tdit_defs(heads, dh)))(
        _grid(((L // 128 + 1) // 2) * heads if qpair else (A // 2) * heads * (L // 128), dev),
        (384, 1, 1),
        *maps,
        O,
        LSE,
        int(L),
        int(A),
    )
    return O, LSE


def forward_tf32(q, k, v, bias_hll, A, L, heads: int = H, dh: int = D, defs: tuple[str, ...] | None = None):
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
    _sm100_kernel("attn_fwd_tf32", "augattn_fwd_tf32_sm100", dev, defs=_tdit_defs(heads, dh) if defs is None else defs)(
        _grid(((A + 1) // 2) * heads * (L // 128), dev),
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
    return max([s for s in (1, 2, 3, 4, 6, 8, 12, 16) if A % s == 0 and (s == 1 or items * s <= nsm)])


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
    defs: tuple[str, ...] | None = None,
    dq_parts: bool = False,
    det: bool = False,
):
    """The fp32 training backward (``attn_dkv_tf32.cu`` then ``attn_dqb_tf32.cu``, TF32 tensor cores): q, k, v, do [A L, heads dh]
    fp32 rows, bias [heads, L, L] fp32 as given to ``forward_tf32``, LSE from it, Dd = rowsum(dO O) [A, heads, L] fp32 -> dQ, dK, dV
    [A L, heads dh] fp32 and dbias [heads, L, L] fp32. Default 16 x 48 (d 768); any of TDIT_HEADS. ``dq`` / ``dk`` / ``dv``: where
    to write them (fp32 [A L, heads dh] with unit column stride and 16-byte rows, e.g. column blocks of the caller's q|k|v|g
    gradient), else fresh buffers. ``defs``: the -D flags of another head layout (AttentionPairBias's, e.g.
    8 x 48), default the token DiT layout's ``_tdit_defs``. ``dq_parts`` (bitwise repeatable dQ): ``attn_dqb_tf32 -DDQPART=1 -DDET=1``
    stores each 64-key chunk's dQ partial as its own slice instead of reduce-adding them in arrival order; the first return value is
    then the slices [L / 64, A L, W] (the caller adds them in order; ``dq`` is only zero-filled). ``det`` (the token DiT
    training block, 2026-11-17): every output bitwise repeatable at any A -- the ``dq_parts`` build with the sample groups kept, each
    group's dbias stored as its own slice (``-DDBPART=1``) and added here in group order, ``attn_dkv_tf32 -DDET=1``, and the dQ slices
    added here in chunk order into ``dq`` (no zero fill)."""
    dev = _index(q)
    W = heads * dh
    out = lambda t: torch.empty(A * L, W, device=q.device, dtype=torch.float32) if t is None else t  # noqa: E731
    DQ, DK, DV = out(dq), out(dk), out(dv)
    for t in (DQ, DK, DV):
        assert t.dtype == torch.float32 and tuple(t.shape) == (A * L, W)
        _rs(t)
    items = heads * (L // 128) * (L // 64)
    ns = 1 if dq_parts and not det else _dqb_split(items, A, dev)
    DQP = torch.empty(L // 64, A * L, W, device=q.device, dtype=torch.float32) if dq_parts or det else None
    # sample groups add their dbias into it (attn_dqb_tf32 -DNS > 1): zeroed then, else written once
    DB = (
        torch.empty(ns, heads, L, L, device=q.device, dtype=torch.float32)
        if det
        else (torch.zeros if ns > 1 else torch.empty)(heads, L, L, device=q.device, dtype=torch.float32)
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
    defs = _tdit_defs(heads, dh) if defs is None else defs
    ldq = DQ.stride(0)
    # attn_dkv_tf32 zero-fills dQ on the way; attn_dqb_tf32 then adds one partial per 64-key chunk (and sample group).
    nodet = tuple(d for d in defs if d != "DET=1")
    _sm100_kernel("attn_dkv_tf32", "augattn_dkv_tf32_sm100", dev, defs=(*nodet, "DET=1") if det else defs)(
        _grid(A * heads * (L // 128), dev),
        (384, 1, 1),
        *dkv_maps,
        LSE,
        Dd,
        None if det else DQ,
        int(L),
        int(A),
        int(ldq),
    )
    _sm100_kernel(
        "attn_dqb_tf32",
        "augattn_dqb_tf32_sm100",
        dev,
        defs=(*nodet, "DQPART=1", "DET=1", *(("DBPART=1",) if det else ())) if DQP is not None else defs,
    )(
        _grid(items * ns, dev),
        (384, 1, 1),
        *dqb_maps,
        LSE,
        Dd,
        DQ if DQP is None else DQP,
        DB,
        int(L),
        int(A),
        int(ns),
        int(ldq if DQP is None else W),
    )
    if det:  # fixed-order sums: the 64-key chunks' dQ slices, the sample groups' dbias slices
        torch.sum(DQP, dim=0, out=DQ)
        return DQ, DK, DV, (DB[0] if ns == 1 else DB.sum(0))
    return (DQ if DQP is None else DQP), DK, DV, DB


def backward(q2, k2, v2, dob, bias_hll, LSE, Dd, A, L, heads: int = H, dh: int = D, det: bool = False, bias_nt: bool = False, side=None):
    """dQ, dK, dV [A L, heads dh] fp32 and dbias [heads, L, L] fp32. dob [A L, heads dh] bf16, Dd = rowsum(dO O) [A, heads, L]
    fp32. Default 16 x 48 (d 768). ``det`` (the token DiT training block, 2026-11-17): bitwise repeatable -- ``attn_dkv -DDET=1``
    (dK / dV accumulated in block order) and ``attn_dqb -DDQPART=1 -DDQDET=1`` (each 128-key chunk's dQ partial stored as its own slice,
    added here in chunk order, no zero fill; a step's two dQ MMA groups in a fixed order). ``bias_nt`` (the token DiT, 2026-11-17 part 7;
    any A since part 10 -- the bias is shared by the samples, the tile attn_dkv loads does not depend on the sample): attn_dkv built
    -DBIASNT=1 reads the bias as stored -- no bias_transpose launch (the same values). Only the token DiT passes it."""
    dev = _index(q2)
    W = heads * dh
    bias_t = bias_hll if bias_nt else bias_transpose(bias_hll)
    nch = L // 128 if det else 1
    DQ = torch.empty(nch * A * L, W, device=q2.device, dtype=torch.float32)
    DB = torch.empty(heads, L, L, device=q2.device, dtype=torch.float32)
    DK, DV = (torch.empty(A * L, W, device=q2.device, dtype=torch.float32) for _ in range(2))
    f32 = dict(dtype="f32")
    dims = [W, A * L]
    dkv_maps = (
        _tm(q2, dims, _rs(q2), [dh, 64]),
        _tm(k2, dims, _rs(k2), [dh, 128]),
        _tm(v2, dims, _rs(v2), [dh, 128]),
        _tm(dob, dims, _rs(dob), [dh, 64]),
        _tm(bias_hll, [L, heads * L], L * 2, [128, 64], swizzle=0) if bias_nt else _tm(bias_t, [L, heads * L], L * 2, [64, 128]),
        _tm(DK, dims, W * 4, [32, 128], swizzle=128, **f32),
        _tm(DK, dims, W * 4, [16, 128], swizzle=64, **f32),
        _tm(DV, dims, W * 4, [32, 128], swizzle=128, **f32),
        _tm(DV, dims, W * 4, [16, 128], swizzle=64, **f32),
    )
    dqb_maps = (
        *(_tm(t, dims, _rs(t), [dh, 128]) for t in (q2, k2, v2, dob)),
        _tm(bias_hll, [L, heads * L], L * 2, [64, 128]),
        _tm(DQ, [W, nch * A * L], W * 4, [32, 128], swizzle=128, **f32),
        _tm(DQ, [W, nch * A * L], W * 4, [16, 128], swizzle=64, **f32),
    )
    defs = _tdit_defs(heads, dh)

    def dqb():
        _sm100_kernel("attn_dqb", "augattn_dqb_sm100", dev, defs=defs + (("DQPART=1", "DQDET=1") if det else ()))(
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

    # side (det only; the token DiT, 2026-11-17 part 7): (run, wait) -- attn_dqb launched through ``run`` (a side stream ordered after
    # the inputs) beside attn_dkv, ``wait`` joins it before dQ is read: with det, attn_dkv fills no dQ, and the two read only their inputs
    if side is not None and det:
        side[0](dqb)
    # attn_dkv zero-fills dQ on the way; attn_dqb then adds one partial per 128-key chunk (det: stores it to the chunk's slice).
    _sm100_kernel("attn_dkv", "augattn_dkv_sm100", dev, defs=defs + (("DET=1",) if det else ()) + (("BIASNT=1",) if bias_nt else ()))(
        _grid(A * heads * (L // 128), dev),
        (384, 1, 1),
        *dkv_maps,
        LSE,
        Dd,
        DK,
        DV,
        None if det else DQ,
        int(L),
        int(A),
    )
    if side is not None and det:
        side[1]()
    else:
        dqb()
    if det:
        DQ = DQ if nch == 1 else DQ.view(nch, A * L, W).sum(0)
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
    _sm100_kernel("glue", "bias_transpose_bf16", _index(bias), smem=0)((L // 32, L // 32, Hh), (32, 8, 1), bias, out, int(L))
    return out


def bias_transpose64(bias, blocks: int = 64):
    """``bias_transpose`` in 64 x 64 tiles on a persistent grid of ``blocks`` blocks (``glue.cu`` bias_transpose64_bf16; L % 64 == 0): the
    AttentionPairBias B200 training forward runs it beside the attention core and leaves that core its SMs (2026-11-17)."""
    Hh, L, _ = bias.shape
    out = torch.empty_like(bias)
    tiles = (L // 64) ** 2 * Hh
    assert L % 64 == 0
    _sm100_kernel("glue", "bias_transpose64_bf16", _index(bias), smem=0)(
        (min(blocks, tiles), 1, 1), (256, 1, 1), bias, out, int(L), int(Hh)
    )
    return out


# --------------------------------------------------------------------------------------------------- op
class _AttentionBf16Sm100(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, bias_head_major):  # noqa: D102
        A, _, L, _, _ = q.shape
        q2, k2, v2 = (t.reshape(A * L, H * D).to(torch.bfloat16).contiguous() for t in (q, k, v))
        bb = (bias.reshape(H, L, L) if bias_head_major else bias[0].permute(2, 0, 1)).to(torch.bfloat16).contiguous()
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
class BoundedRuns(collections.OrderedDict):
    """Bound launches (``GatedInferenceCore.runs``): an LRU of at most ``cap`` entries, keyed by the buffers' addresses and layouts.

    A run holds only its TMA descriptors (they encode addresses, not tensors) and the scratch it owns -- never the caller's tensors -- so an
    entry costs kilobytes, the address of a freed buffer can be reused by the allocator and hit the same (valid) run, and a sampler that
    allocates fresh tensors every step cannot grow the cache past ``cap``. A graph that captured a run keeps the run's scratch alive through
    ``_capture.hold`` (done by the core), so an eviction never frees memory a replay reads."""

    cap = 128
    #: bytes of scratch the runs own (``run.nbytes``: the key-split partials), at most, beyond the newest run
    scratch_budget = 256 << 20

    def bind(self, key, make):
        run = self.get(key)
        if run is not None:
            self.move_to_end(key)
            return run
        run = make()
        self[key] = run
        while len(self) > 1 and (len(self) > self.cap or sum(getattr(r, "nbytes", 0) for r in self.values()) > self.scratch_budget):
            self.popitem(last=False)
        return run


def _release_maps(maps) -> None:
    """The maps' ``keep`` pins the tensor a descriptor was encoded from; a cached run must not own the caller's buffers."""
    for m in maps:
        m.keep = None


class GatedInferenceCore:
    """The fused token DiT step's core on sm_100a: reads the block's head-major hoisted bias ([nb H, L, L] rows) and the
    step's projections (logits pre-scaled into exp2 units), writes sigmoid(g) * o over q.

    * bf16 (``attn_inf.cu``): ``qkvg`` [S L, 4 D], q | k | v | g column views of one GEMM output.
    * fp32 with TF32 tensor cores (``attn_inf_tf32.cu``): ``qkvg`` is q | k | g, [S L, 3 D], and ``vt`` is v transposed,
      [D, S L] -- a kind::tf32 MMA takes v only K-major. Every 48-wide fp32 q / k / g tile is a 32-column 128-B-swizzled
      box and a 16-column 64-B-swizzled one (32: one 32-column box, 64: two); the v^T tile is dh rows x 32 keys.

    Both at every layout of TDIT_HEADS. Bindings (TMA descriptors) are cached per buffer address and layout, in a bounded LRU
    (``BoundedRuns``): they do not own the caller's tensors."""

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
        self.runs = BoundedRuns()
        self.device_index = device_index

    def _qpair(self):
        """bf16, one sample: ``attn_inf -DQPAIR=1`` at this head layout (built on first use)."""
        return _sm100_kernel(
            "attn_inf",
            "augattn_inf_sm100",
            self.device_index,
            defs=("QPAIR=1", f"NHEAD={self.heads}", f"DHP={self.dh}", f"RSQDV={self.dh**-0.5!r}f"),
        )

    def _bind(self, qkvg, vt, bias, block, S, L, srow=None, vmn=False):
        # 3-D maps (columns, row of the sample, sample | head): the last tile's rows past L load as zeros and are clipped
        # on the store, so L need only keep the rows 16-byte aligned (a multiple of 8)
        M, W = qkvg.shape
        es = qkvg.element_size()
        rs = W * es
        ss = (L if srow is None else srow) * rs  # bytes between consecutive samples (B200 R18 folds: B L rows)
        Hh = self.heads
        bv = bias[block * Hh : (block + 1) * Hh]
        grid = (min(self.nsm, ((S + 1) // 2) * Hh * -(-L // 128)), 1, 1)
        bias_map = lambda box, **kw: _tm(bv, [L, L, Hh], [L * es, L * L * es], [*box, 1], **kw)  # noqa: E731
        if self.fp32:
            # vmn (several samples, -DVMN): qkvg is q | k | v | g and v is read MN-major from its column view (no v^T)
            Dm = W // 4 if vmn else W // 3
            if vmn:
                q, k, v, g = (qkvg[:, i * Dm : (i + 1) * Dm] for i in range(4))
            else:
                q, k, g = (qkvg[:, i * Dm : (i + 1) * Dm] for i in range(3))
            f32 = dict(dtype="f32")
            a = lambda t, rows, ns=1: _tm(t, [Dm, L, S], [rs, ss], [32, rows, ns], swizzle=128, **f32)  # noqa: E731
            b = lambda t, rows, ns=1: _tm(t, [Dm, L, S], [rs, ss], [16, rows, ns], swizzle=64, **f32)  # noqa: E731
            # vmn: k of both samples of a pair in one box (dh 32 / 48: one 32-column box a sample), v as one 4-D box (column, key,
            # 32-column atom, sample)
            nk = 2 if vmn and self.dh <= 48 else 1
            nva = -(-self.dh // 32)
            maps = (
                a(q, 128),
                b(q, 128),
                a(k, 32, nk),
                b(k, 32, nk),
                # v: 32-column MN atoms (vmn); v^T: key, sample, channel
                _tm(v, [Dm, L, nva, S], [rs, 128, ss], [32, 32, nva, 2], swizzle="128a32", **f32)
                if vmn
                else _tm(vt, [L, S, Dm], [(L if srow is None else srow) * 4, vt.stride(0) * 4], [32, 1, self.dh], swizzle=128, **f32),
                bias_map([32, 128], swizzle=128, **f32),
                a(g, 128),
                b(g, 128),
            )

            kf = self.k
            if vmn:
                assert S > 1 and srow is None, "the MN-major v build serves several samples"
                kf = _sm100_kernel("attn_inf_tf32", "augattn_inf_tf32_sm100", self.device_index, defs=("VMN=1", *_tdit_defs(self.heads, self.dh)))
            own = getattr(self, "tf32_qpair", True) and not vmn  # AttentionPairBias's -DKPERM core opts out of the token DiT's one-sample forms
            ns = tf32_ksplit(L, Hh, self.nsm) if S == 1 and srow is None and own else 1
            if ns > 1:
                # one sample, few work items (2026-11-17, AttentionPairBias part 7): the query-tile pairs x ns key chunks, each leaving its
                # normalised partial o and LSE; ksplit_combine32 merges them in chunk order, gates and writes o over q
                from .. import apb as _apb_rows

                ks = _sm100_kernel(
                    "attn_inf_tf32",
                    "augattn_inf_tf32_sm100",
                    self.device_index,
                    defs=("QPAIR=1", "KSPLIT=1", *_tdit_defs(self.heads, self.dh)),
                )
                op = torch.empty((ns, L, Dm), dtype=torch.float32, device=qkvg.device)
                lsep = torch.empty((ns, Hh, L), dtype=torch.float32, device=qkvg.device)
                pa = _tm(op, [Dm, L, ns], [Dm * 4, L * Dm * 4], [32, 128, 1], swizzle=128, **f32)
                pb = _tm(op, [Dm, L, ns], [Dm * 4, L * Dm * 4], [16, 128, 1], swizzle=64, **f32)
                grid = (min(self.nsm, ((-(-L // 128) + 1) // 2) * Hh * ns), 1, 1)
                E = _apb_rows.ext()
                dh = self.dh

                def run(qkvg_):
                    ks(grid, (384, 1, 1), *maps, int(L), int(S), pa, pb, lsep, int(ns))
                    E.ksplit_combine32(op, lsep, qkvg_[:L], 2 * Dm, Hh, dh)

                _release_maps(maps)
                run.keep = (maps, op, lsep, pa, pb)  # its own scratch only
                run.nbytes = op.nbytes + lsep.nbytes  # ty: ignore[unresolved-attribute]
                return run
            if S == 1 and L > 128 and _TF32_QPAIR and own:
                # one sample: ``attn_inf_tf32 -DQPAIR=1`` (two 128-query tiles of a head share its k / v^T blocks; two stages at 48 wide)
                # where the sample-pair schedule computed a dropped partner sample (2026-11-17, part 3)
                kf = _sm100_kernel(
                    "attn_inf_tf32", "augattn_inf_tf32_sm100", self.device_index, defs=("QPAIR=1", *_tdit_defs(self.heads, self.dh))
                )
                grid = (min(self.nsm, ((-(-L // 128) + 1) // 2) * Hh), 1, 1)

            def run(qkvg_=None):
                kf(grid, (384, 1, 1), *maps, int(L), int(S))
        else:
            Dm, DH = W // 4, self.dh
            q, k, v, g = (qkvg[:, i * Dm : (i + 1) * Dm] for i in range(4))
            t3 = lambda t, rows: _tm(t, [Dm, L, S], [rs, ss], [DH, rows, 1])  # noqa: E731
            maps = (t3(q, 128), t3(k, 64), t3(v, 64), bias_map([64, 128]), t3(g, 128))
            mq, mk, mv, mb, mg = maps
            k = self.k
            if S == 1 and L > 128:
                # one sample: the -DQPAIR build (AttentionPairBias's schedule) pairs two 128-query tiles of a head, sharing its k / v
                # blocks, where the sample-pair schedule computed a dropped partner sample (half of every work item)
                k = self._qpair()
                grid = (min(self.nsm, ((-(-L // 128) + 1) // 2) * Hh), 1, 1)

            def run(qkvg_=None):
                k(grid, (384, 1, 1), mq, mk, mv, mb, mg, mq, int(L), int(S))

        _release_maps(maps)
        run.keep = (maps,)
        return run

    def __call__(self, qkvg, bias, block, S, vt=None, L=None, srow=None, vmn=False):
        """``vt`` (fp32 only): v^T [D, S L]; ``qkvg`` is then q | k | g. ``L`` / ``srow`` (optional): the rows of a sample and the
        rows between consecutive samples (default: qkvg is S samples of L contiguous rows; the B200 R18 folds pass a fold's first row
        and B L)."""
        assert self.fp32 == (vt is not None or vmn), "the TF32 core takes q | k | g and v^T (or q | k | v | g: vmn); the bf16 core q | k | v | g"
        if L is None:
            L = qkvg.shape[0] // S
        key = (
            qkvg.data_ptr(),
            None if vt is None else vt.data_ptr(),
            bias.data_ptr(),
            block,
            S,
            L,
            srow,
            vmn,
            # the layout the descriptors encode: the address alone no longer pins a tensor (a freed buffer's address is reused)
            qkvg.shape[1],
            qkvg.dtype,
            None if vt is None else vt.stride(0),
        )
        run = self.runs.bind(key, lambda: self._bind(qkvg, vt, bias, block, S, L, srow, vmn))
        _capture.hold(run)  # a capture reads the run's scratch (key-split partials) at every replay
        run(qkvg)
        return qkvg

    def folds(self, qkvg, bias, S, NB, L):
        """bf16, NB folds in ONE launch (``attn_inf -DFOLDARG=1``): ``qkvg`` [S NB L, 4 D] rows [S][NB][L] (the B200 R18 trunk_seeds),
        ``bias`` [heads, NB, L, L] (``apb.pair_bias_batch``); a work item pairs two samples of one fold. 2026-11-17, part 3: the per-fold
        launches were 5 x 7.8 us at L 128 (48 work items each on 148 SMs)."""
        assert not self.fp32
        key = ("folds", qkvg.data_ptr(), bias.data_ptr(), S, NB, L, qkvg.shape[1], qkvg.dtype)

        def make():
            k = _sm100_kernel("attn_inf", "augattn_inf_sm100", self.device_index, defs=(*_tdit_defs(self.heads, self.dh), "FOLDARG=1"))
            W = qkvg.shape[1]
            es = qkvg.element_size()
            rs = W * es
            Dm, DH, Hh = W // 4, self.dh, self.heads
            q, kk, v, g = (qkvg[:, i * Dm : (i + 1) * Dm] for i in range(4))
            t3 = lambda t, rows: _tm(t, [Dm, L, S * NB], [rs, L * rs], [DH, rows, 1])  # noqa: E731
            bv = bias.reshape(Hh * NB, L, L)
            mb = _tm(bv, [L, L, Hh * NB], [L * es, L * L * es], [64, 128, 1])
            mq = t3(q, 128)
            maps = (mq, t3(kk, 64), t3(v, 64), mb, t3(g, 128))
            grid = (min(self.nsm, ((S + 1) // 2) * Hh * -(-L // 128) * NB), 1, 1)

            def run(qkvg_=None):
                k(grid, (384, 1, 1), *maps, mq, int(L), int(S), int(NB))

            _release_maps(maps)
            run.keep = (maps,)  # ty: ignore[unresolved-attribute]
            return run

        run = self.runs.bind(key, make)
        _capture.hold(run)
        run(qkvg)
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
}


def apb_width(heads: int, d: int = 384) -> int:
    """Row width of q / k / v / g / O (d_single, or 512 for 16 x 24 padded to 32)."""
    return heads * APB_GEOMETRY[(heads, d)][0]


def _apb_defs(heads: int, qpair: bool, d: int = 384):
    dhp, dreal = APB_GEOMETRY[(heads, d)]
    defs = (("QPAIR=1",) if qpair else ()) + (f"NHEAD={heads}",)
    if dhp != 48 or dreal != 48:
        defs += (f"DHP={dhp}", f"RSQDV={dreal**-0.5!r}f")
    if (
        not qpair
    ):  # the backward cores (attn_dkv -DDET=1: dK / dV accumulated in block order, bitwise repeatable; attn_dqb does not read it)
        defs += ("DET=1",)
    return defs


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


#: most chunks a work item's keys are split into (``apb_rows_sm100.cu`` ksplit_combine's NSMAX)
KSPLIT_MAX = 16
#: MINIWORLD_APB_B200_KSPLIT: force the inference core's key split (1: the unsplit core); unset: ``apb_ksplit``'s rule
_KSPLIT_FORCED = os.environ.get("MINIWORLD_APB_B200_KSPLIT")
#: the shortest L the rule splits at
_KSPLIT_MIN_L = 384


def apb_ksplit(L: int, heads: int, nsm: int = 148) -> int:
    """Key chunks per work item of AttentionPairBias's B = 1 inference core (``attn_inf -DKSPLIT=1``, from origin/b200/ladder-final):
    the unsplit core has heads x ceil(L / 256) work items for ``nsm`` SMs (12-48 of 148 at L <= 768). A split into ns chunks makes ns
    times as many items, each 1 / ns as long, so the core takes ~ waves(ns) / ns of the unsplit time, waves = ceil(items ns / nsm). The
    choice is the smallest ns (a divisor of the key blocks ceil(L / 64), each item keeping at least two) that minimises it, kept only
    when it at least halves the unsplit estimate, and only from L 512 on: the partials and the combine pass cost a launch (measured
    2026-11-17, master eval step, unsplit -> split: 16 x 128 L 256 32.6 -> 34.6 us, L 384 39.0 -> 38.9, L 512 48.9 -> 44.8, L 768
    67.7 -> 65.4, L 1024 96.1 -> 92.1; 12 x 128 L 512 47.0 -> 44.8, L 768 69.4 -> 63.3; 24 heads level). v20 (2026-11-17, the core and
    the combine as programmatic dependents): from L 384 (``_KSPLIT_MIN_L``; master eval L 384 unsplit -> split, trimmed means: 16 x 128
    40.9 -> 39.0 us, 12 x 128 39.2 -> 37.6, 16 x 256 50.8 -> 49.7, 8 x 384 57.4 -> 55.9, 24 heads and d 512 level; L 256 1.6-2.1 us slower)."""
    nbt = -(-L // 64)
    if _KSPLIT_FORCED is not None:
        forced = int(_KSPLIT_FORCED)
        return forced if 1 <= forced <= KSPLIT_MAX and nbt % forced == 0 else 1
    if L < _KSPLIT_MIN_L:
        return 1
    base = ((-(-L // 128) + 1) // 2) * heads
    best, best_cost = 1, float(-(-base // nsm))
    for ns in range(2, KSPLIT_MAX + 1):
        if nbt % ns or nbt // ns < 2:
            continue
        cost = -(-base * ns // nsm) / ns
        if cost < best_cost - 1e-9:
            best, best_cost = ns, cost
    return best if best_cost <= 0.5 * -(-base // nsm) + 1e-9 else 1


class ApbInferenceCore:
    """AttentionPairBias's inference core: ``qkvg`` [L, 4 W] bf16 (q | k | v | g, W = ``apb_width(heads)``, logits pre-scaled into
    exp2 units), the head-major bias [heads, L, L] bf16 (exp2 units, masked keys very negative); sigmoid(g) * o written over q.
    With a key split (``apb_ksplit`` > 1) the core leaves per-chunk partials (fp32 o, LSE) and ``ksplit_combine`` merges and gates
    them over q."""

    def __init__(self, device_index: int, heads: int = 8, d: int = 384):
        self.heads, self.d, self.dhp = heads, d, APB_GEOMETRY[(heads, d)][0]
        self.dev = device_index
        self.k = _sm100_kernel(
            "attn_inf",
            "augattn_inf_sm100",
            device_index,
            defs=_apb_defs(heads, True, d),
        )
        self.ks = None  # the key-split build, made at its first use
        self.ksp = None  # its -DPDLW=1 build (v20)
        self.kp = None  # the -DPDLW=1 build launched as a programmatic dependent (part 6), made at its first use
        self.nsm = num_sms(device_index)

    def _bind(self, qkvg, bias, L, ns=1, op=None, lp=None, pdl=False):
        W, es, H_, dh = qkvg.shape[1], qkvg.element_size(), self.heads, self.dhp
        Dm = W // 4
        q, k, v, g = (qkvg[:, i * Dm : (i + 1) * Dm] for i in range(4))
        t3 = lambda t, rows: _tmn(t, [Dm, L, 1], [W * es, L * W * es], [dh, rows, 1])  # noqa: E731
        mq, mk, mv, mg = t3(q, 128), t3(k, 64), t3(v, 64), t3(g, 128)
        mb = _tmn(bias, [L, L, H_], [L * es, L * L * es], [64, 128, 1])
        grid = (min(self.nsm, ((-(-L // 128) + 1) // 2) * H_ * ns), 1, 1)
        if ns == 1 and pdl:
            if self.kp is None:
                self.kp = _sm100_kernel(
                    "attn_inf", "augattn_inf_sm100", self.dev, pdl=True, defs=(*_apb_defs(self.heads, True, self.d), "PDLW=1")
                )
            kp = self.kp  # bind() packs plain launches only: the PDL launch goes through __call__
            return lambda: kp(grid, (384, 1, 1), mq, mk, mv, mb, mg, mq, int(L), 1)
        if ns == 1:
            return self.k.bind(grid, (384, 1, 1), mq, mk, mv, mb, mg, mq, int(L), 1)
        if pdl:  # v20 (2026-11-17): the key-split build as a programmatic dependent too (-DPDLW=1; bind() takes PDL launches)
            if self.ksp is None:
                self.ksp = _sm100_kernel(
                    "attn_inf", "augattn_inf_sm100", self.dev, pdl=True, defs=(*_apb_defs(self.heads, True, self.d), "KSPLIT=1", "PDLW=1")
                )
            kk = self.ksp
        else:
            if self.ks is None:
                self.ks = _sm100_kernel("attn_inf", "augattn_inf_sm100", self.dev, defs=(*_apb_defs(self.heads, True, self.d), "KSPLIT=1"))
            kk = self.ks
        f32 = dict(dtype="f32")
        dims, st = [Dm, L, ns], [Dm * 4, L * Dm * 4]
        mpa = _tmn(op, dims, st, [32, 128, 1], swizzle=128, **f32)
        mpb = _tmn(op, dims, st, [16, 128, 1], swizzle=64, **f32)
        return kk.bind(grid, (384, 1, 1), mq, mk, mv, mb, mg, mq, int(L), 1, mpa, mpb, lp, int(ns))

    def __call__(self, qkvg, bias, pdl: bool = False):
        """``pdl`` (2026-11-17, part 6): the unsplit core launched as a programmatic dependent of the previous kernel on the stream (its
        prologue overlaps that kernel's tail; the -DPDLW=1 build waits before its first load)."""
        L = qkvg.shape[0]
        W = apb_width(self.heads, self.d)
        assert qkvg.shape[1] == 4 * W and bias.shape[0] == self.heads
        ns = apb_ksplit(L, self.heads, self.nsm)
        op = lp = None
        if ns > 1:
            op = torch.empty(ns, L, W, device=qkvg.device, dtype=torch.float32)
            lp = torch.empty(ns, 1, self.heads, L, device=qkvg.device, dtype=torch.float32)
        _apb_bound(
            (
                "inf",
                self.heads,
                self.d,
                qkvg.device.index,
                qkvg.data_ptr(),
                bias.data_ptr(),
                L,
                ns,
                None if op is None else op.data_ptr(),
                None if lp is None else lp.data_ptr(),
                pdl,
            ),
            lambda: self._bind(qkvg, bias, L, ns, op, lp, pdl),
        )()
        if ns > 1:
            from ..apb import _ext

            E = _ext("100")
            was = E.pdl(1 if pdl else 0)  # v20: the combine as a programmatic dependent of the core
            E.ksplit_combine(op, lp, qkvg, None, None, None, L, self.heads, self.dhp)
            E.pdl(was)
        return qkvg


def apb_forward(q, k, v, bias, L, heads: int = 8, d: int = 384, pdl: bool = False):
    """AttentionPairBias training forward (``attn_fwd2 -DQPAIR``): q, k, v [L, W] bf16 rows (W = ``apb_width(heads)``, column
    views welcome), bias [heads, L, L] bf16 natural units -> O [L, W] fp32, LSE [heads, L] (log2). ``pdl`` (v20, 2026-11-17): the -DPDLW=1 build
    launched as a programmatic dependent of the previous kernel on the stream."""
    dev = _index(q)
    dh, W = APB_GEOMETRY[(heads, d)][0], apb_width(heads, d)
    O = torch.empty(L, W, device=q.device, dtype=torch.float32)
    LSE = torch.empty(heads, L, device=q.device, dtype=torch.float32)

    def build():
        maps = (
            _tmn(q, [W, L], _rs(q), [dh, 128]),
            _tmn(k, [W, L], _rs(k), [dh, 64]),
            _tmn(v, [W, L], _rs(v), [dh, 64]),
            _tmn(bias, [L, heads * L], L * 2, [64, 128]),
            _tmn(O, [W, L], W * 4, [32, 128], swizzle=128, dtype="f32"),
            _tmn(O, [W, L], W * 4, [16, 128], swizzle=64, dtype="f32"),
        )
        grid = _grid(((L // 128 + 1) // 2) * heads, dev)
        defs = _apb_defs(heads, True, d) + (("PDLW=1",) if pdl else ())
        return _sm100_kernel("attn_fwd2", "augattn_fwd2_sm100", dev, pdl=pdl, defs=defs).bind(grid, (384, 1, 1), *maps, O, LSE, int(L), 1)

    key = (
        "fwd",
        pdl,
        heads,
        d,
        dev,
        L,
        *(t.data_ptr() for t in (q, k, v, bias, O, LSE)),
        _rs(q),
        _rs(k),
        _rs(v),
    )
    _apb_bound(key, build)()
    return O, LSE


def apb_backward(q, k, v, dob, bias, LSE, Dd, L, heads: int = 8, d: int = 384):
    """AttentionPairBias training backward (``attn_dkv``, ``attn_dqb`` with -DNHEAD, A = 1): q, k, v, dob [L, W] bf16 rows, bias
    [heads, L, L] bf16 as given to ``apb_forward``, LSE from it, Dd = rowsum(dO O) [heads, L] fp32 -> dQ, dK, dV [L, W] fp32 and
    dbias [heads, L, L] fp32. dQ comes as L / 128 key-chunk partials [L / 128, L, W] (``attn_dqb -DDQPART=1`` stores each
    chunk's slice instead of reduce-adding into one buffer, so the sum -- the caller's, in chunk order -- is deterministic)."""
    stages, outs = apb_backward_stages(q, k, v, dob, bias, LSE, Dd, L, heads, d)
    for run in stages:
        run()
    return outs[:4]


def _no_launch() -> None:
    pass


def apb_backward_stages(
    q,
    k,
    v,
    dob,
    bias,
    LSE,
    Dd,
    L,
    heads: int = 8,
    d: int = 384,
    bias_t: torch.Tensor | None = None,
    bias_nt: bool = False,
    dkv_pdl: bool = False,
):
    """``apb_backward``'s three launches, unlaunched: ((bias_transpose, attn_dkv, attn_dqb), (dQ, dK, dV, dbias, bias^T)). The caller
    holds bias^T until attn_dkv has run (2026-11-17: it was not returned, so the caching allocator handed its block to the next allocation on
    the caller's stream -- the dO GEMM's output -- while the transpose wrote it on a side stream: wrong dK / dV on a cold
    first call). Each runs on the stream current when it is called, so a caller can overlap them: the transpose needs only the bias; attn_dkv needs the transpose;
    attn_dqb (dQ chunk partials, no zero fill) needs neither -- it may run beside attn_dkv."""
    dev = _index(q)
    dh, W = APB_GEOMETRY[(heads, d)][0], apb_width(heads, d)
    nch = L // 128
    # bias_nt (2026-11-17, part 6): attn_dkv built -DBIASNT=1 reads the bias as stored (no transpose: the stage does nothing)
    made_t = bias_t is None and not bias_nt  # given: the forward made bias^T already (the returned transpose stage then does nothing)
    if bias_nt:
        bias_t = bias
    elif bias_t is None:
        bias_t = torch.empty_like(bias)
    DQ = torch.empty(nch, L, W, device=q.device, dtype=torch.float32)
    DB = torch.empty(heads, L, L, device=q.device, dtype=torch.float32)
    DK, DV = (torch.empty(L, W, device=q.device, dtype=torch.float32) for _ in range(2))

    def build():
        f32 = dict(dtype="f32")
        dims = [W, L]
        dkv_maps = (
            _tmn(q, dims, _rs(q), [dh, 64]),
            _tmn(k, dims, _rs(k), [dh, 128]),
            _tmn(v, dims, _rs(v), [dh, 128]),
            _tmn(dob, dims, _rs(dob), [dh, 64]),
            _tmn(bias, [L, heads * L], L * 2, [128, 64], swizzle=0) if bias_nt else _tmn(bias_t, [L, heads * L], L * 2, [64, 128]),
            _tmn(DK, dims, W * 4, [32, 128], swizzle=128, **f32),
            _tmn(DK, dims, W * 4, [16, 128], swizzle=64, **f32),
            _tmn(DV, dims, W * 4, [32, 128], swizzle=128, **f32),
            _tmn(DV, dims, W * 4, [16, 128], swizzle=64, **f32),
        )
        dq_dims = [W, nch * L]
        dqb_maps = (
            *(_tmn(t, dims, _rs(t), [dh, 128]) for t in (q, k, v, dob)),
            _tmn(bias, [L, heads * L], L * 2, [64, 128]),
            _tmn(DQ, dq_dims, W * 4, [32, 128], swizzle=128, **f32),
            _tmn(DQ, dq_dims, W * 4, [16, 128], swizzle=64, **f32),
        )
        defs = _apb_defs(heads, False, d)
        tr = _sm100_kernel("glue", "bias_transpose_bf16", dev, smem=0).bind((L // 32, L // 32, heads), (32, 8, 1), bias, bias_t, int(L))
        # dkv_pdl (v20, 2026-11-17): the -DPDLW=1 build launched as a programmatic dependent of the previous kernel on the stream
        dkv_defs = defs + (("BIASNT=1",) if bias_nt else ()) + (("PDLW=1",) if dkv_pdl else ())
        dkv = _sm100_kernel("attn_dkv", "augattn_dkv_sm100", dev, pdl=dkv_pdl, defs=dkv_defs).bind(
            _grid(heads * (L // 128), dev),
            (384, 1, 1),
            *dkv_maps,
            LSE,
            Dd,
            DK,
            DV,
            None,
            int(L),
            1,
        )  # no dQ zero fill
        dqb_defs = defs + ("DQPART=1",)
        dqb = _sm100_kernel("attn_dqb", "augattn_dqb_sm100", dev, defs=dqb_defs).bind(
            _grid(heads * (L // 128) * (L // 128), dev),
            (384, 1, 1),
            *dqb_maps,
            LSE,
            Dd,
            DQ,
            DB,
            int(L),
            1,
        )

        return (tr if made_t else _no_launch), dkv, dqb

    key = (
        "bwd3",
        made_t,
        bias_nt,
        dkv_pdl,
        heads,
        d,
        dev,
        L,
        *(t.data_ptr() for t in (q, k, v, dob, bias, LSE, Dd, bias_t, DQ, DB, DK, DV)),
        _rs(q),
        _rs(k),
        _rs(v),
        _rs(dob),
    )
    return _apb_bound(key, build), (DQ, DK, DV, DB, bias_t)


#: pair_bwd.cu builds a = a_hi + a_lo (PB_SPLIT, its default; 2026-11-17) at <= 16 heads: row 148 L 128 grad:pair 1.25 -> 1.00 x tol, no
#: measurable cost there; 24 heads keep the single bf16 a (the split doubles K of D = a Wf: L 768 +8 us). MINIWORLD_APB_B200_PB_SPLIT=0: never.
_PB_SPLIT_OFF = os.environ.get("MINIWORLD_APB_B200_PB_SPLIT") == "0"


def _pb_split_defs(heads: int) -> tuple[str, ...]:
    return ("PB_SPLIT=0",) if _PB_SPLIT_OFF or heads > 16 else ()


def _pair_fwd32_smem(dp: int) -> int:
    """apb_pair_fwd32_sm100.cuh smem_bytes: stages x 32 KB units + Wf's boxes (4 KB per 32 columns) + barriers."""
    return (5 if dp >= 384 else 6) * 128 * 64 * 4 + (dp // 32) * 32 * 128 + 1024


def apb_pair_fwd32(pair2d, wf, mask, out, L: int, eps: float, neg: float, perm: bool = True):
    """AttentionPairBias's fp32 inference pair -> bias on tcgen05 kind::tf32 (``pair_fwd32.cu``, 2026-11-17 part 6): ``out`` [H, L, L] fp32
    = Wf . LN(pair) (Wf [H, dp] fp32 as packed: exp2 units), keys in the TF32 core's key_perm order when ``perm``, ``neg`` on the keys
    where ``mask`` ([L] bool or None) is False -- apb_rows.cu pair_bias32_fwd's result. pair2d [L L, dp] fp32 contiguous."""
    heads, dp = wf.shape
    dev = _index(pair2d)
    assert pair2d.dtype == torch.float32 and pair2d.is_contiguous() and pair2d.numel() == L * L * dp and dp % 64 == 0
    assert wf.dtype == torch.float32 and wf.is_contiguous() and out.dtype == torch.float32 and out.is_contiguous() and heads <= 24
    k = _sm100_kernel("pair_fwd32", "apb_pair_fwd32_sm100", dev, defs=(f"PF_NH={heads}", f"PF_DPW={dp}"), smem=_pair_fwd32_smem(dp))
    grid = (min(num_sms(dev), L * -(-L // 128)), 1, 1)
    _apb_bound(
        (
            "pfwd32",
            heads,
            dp,
            dev,
            pair2d.data_ptr(),
            wf.data_ptr(),
            None if mask is None else mask.data_ptr(),
            out.data_ptr(),
            L,
            float(eps),
            float(neg),
            bool(perm),
        ),
        lambda: k.bind(
            grid,
            (192, 1, 1),
            _tmn(pair2d, [dp, L * L], dp * 4, [32, 128], dtype="f32"),
            wf,
            mask,
            out,
            int(L),
            float(eps),
            float(neg),
            int(perm),
        ),
    )()
    return out


#: AttentionPairBias pair -> bias on tcgen05 (``pair_fwd.cu``): d_pair -> its shared memory bytes (apb_pair_fwd_sm100.cuh smem_bytes)
_PAIR_FWD_SMEM = {128: 140288, 256: 214016, 384: 222208, 512: 230400}


def apb_pair_fwd(pair2d, wsrc, lpw, ws: float, mask, L: int, eps: float, neg: float, *, nb: int = 1, stats=None, unmasked=None):
    """bias [H, L, L] bf16 (``nb`` > 1: [H, nb, L, L]) = Wf . LN(pair) of AttentionPairBias on the tcgen05 pass (``pair_fwd.cu``, a
    build per (heads, d_pair)); ``neg`` on the keys where ``mask`` ([nb L] bool or None) is False. pair2d [nb L L, dp] bf16 contiguous.
    ``lpw`` None: ``wsrc`` is Wf [H, dp] bf16. Otherwise ``wsrc`` = to_bias.weight [H, dp] (bf16, or an fp32 master) and ``lpw`` =
    ln_pair.weight [dp] fp32: Wf = bf16(bf16(wsrc) lpw ws) is formed in the kernel (``prep``'s arithmetic), so nothing is packed
    before the pass. ``stats`` ([L L, 2] fp32) receives every row's (mean, rstd), ``unmasked`` ([H, L, L] fp32) the bias before the mask."""
    heads, dp = wsrc.shape
    dev = _index(pair2d)
    k = _sm100_kernel("pair_fwd", "apb_pair_fwd_sm100", dev, defs=(f"PF_NH={heads}", f"PF_DPW={dp}"), smem=_PAIR_FWD_SMEM[dp])
    shape = (heads, L, L) if nb == 1 else (heads, nb, L, L)
    out = torch.empty(shape, device=pair2d.device, dtype=torch.bfloat16)
    assert pair2d.dtype == torch.bfloat16 and pair2d.is_contiguous() and pair2d.numel() == nb * L * L * dp
    assert wsrc.is_contiguous() and (wsrc.dtype == torch.bfloat16 or (lpw is not None and wsrc.dtype == torch.float32))
    assert lpw is None or (lpw.dtype == torch.float32 and lpw.numel() == dp)
    assert mask is None or (mask.dtype == torch.bool and mask.is_contiguous() and mask.numel() == nb * L)
    ptr = lambda t: None if t is None else t.data_ptr()  # noqa: E731
    fp32 = int(wsrc.dtype == torch.float32)
    grid = (min(num_sms(dev), nb * L * -(-L // 128)), 1, 1)
    _apb_bound(
        (
            "pfwd",
            heads,
            dp,
            dev,
            pair2d.data_ptr(),
            wsrc.data_ptr(),
            fp32,
            ptr(lpw),
            float(ws),
            ptr(mask),
            out.data_ptr(),
            ptr(stats),
            ptr(unmasked),
            L,
            nb,
            float(eps),
            float(neg),
        ),
        lambda: k.bind(
            grid,
            (192, 1, 1),
            _tmn(pair2d, [dp, nb * L * L], dp * 2, [64, 128]),
            wsrc,
            lpw,
            float(ws),
            fp32,
            mask,
            out,
            stats,
            unmasked,
            int(L),
            float(eps),
            float(neg),
            int(nb),
        ),
    )()
    return out


#: AttentionPairBias pair backward on tcgen05 (``pair_bwd.cu``, d_pair 128): heads -> its shared memory bytes (SMEM_BYTES there)
_PAIR_BWD_SMEM = {8: 147456 + 61440, 12: 155648 + 61440, 16: 163840 + 61440, 24: 135168 + 61440}


def _pair_bwd_defs(heads: int) -> tuple[str, ...]:
    return (f"PB_NH={heads}",) + _pb_split_defs(heads)


def apb_pair_bwd(pair2d, dbias, wf, L: int, eps: float, acc_cols, dz, reserve: int = 0, trace=None) -> int:
    """dpair [L L, 128] bf16 into ``dz`` and the block partial rows of (dWf [H, 128], per-head dbias sums [H]) into ``acc_cols`` (a
    [rows, >= H 129] fp32 column view of the backward's accumulator: row b = block b) of AttentionPairBias's d_pair 128 pair backward on
    tcgen05 (``pair_bwd.cu``; the results of ``apb_rows.cu`` pair_bias_bwd). pair2d [L L, 128] bf16, dbias [H, L, L] fp32, wf [H, 128]
    bf16; L a multiple of 128. ``reserve``: SMs left free. Returns the grid (= the partial rows written)."""
    heads = wf.shape[0]
    dev = _index(pair2d)
    assert L % 128 == 0 and pair2d.dtype == torch.bfloat16 and pair2d.is_contiguous() and dz.is_contiguous()
    assert dbias.dtype == torch.float32 and dbias.is_contiguous() and wf.dtype == torch.bfloat16 and wf.is_contiguous()
    assert acc_cols.dtype == torch.float32 and acc_cols.stride(1) == 1
    grid = min(num_sms(dev) - reserve, L * (L // 128))
    assert acc_cols.shape[0] >= grid and acc_cols.shape[1] >= heads * 129
    defs = _pair_bwd_defs(heads)
    k = _sm100_kernel("pair_bwd", "apb_pair_bwd_sm100", dev, defs=defs, smem=_PAIR_BWD_SMEM[heads])
    _apb_bound(
        (
            "pbwd",
            heads,
            dev,
            L,
            grid,
            defs,
            float(eps),
            *(t.data_ptr() for t in (pair2d, dbias, wf, acc_cols, dz)),
            acc_cols.stride(0),
        )
        + (() if trace is None else (trace.data_ptr(),)),
        lambda: k.bind(
            (grid, 1, 1),
            (352, 1, 1),
            _tmn(pair2d, [128, L * L], 256, [64, 128]),
            _tmn(dz, [128, L * L], 256, [64, 128]),
            _tmn(dbias, [L, L, heads], [L * 4, L * L * 4], [128, 1, heads], swizzle=0, dtype="f32"),
            wf,
            acc_cols,
            int(acc_cols.stride(0)),
            int(L),
            float(eps),
            trace,
        ),
    )()
    return grid


def inference_core_supported(dtype: torch.dtype, L: int, d: int, h: int, device_index: int) -> bool:
    """bf16, or fp32 (TF32 MMA), at a head layout of TDIT_HEADS (d = heads x head dim); L a multiple of 8 (the fused step pads
    other lengths to one), sm_100."""
    if d % h:
        return False
    ok = dtype in (torch.float32, torch.bfloat16) and (h, d // h) in TDIT_HEADS
    return ok and L % 8 == 0 and L >= 8 and _is_blackwell(device_index)


__all__ = [
    "APB_GEOMETRY",
    "TDIT_HEADS",
    "ApbInferenceCore",
    "GatedInferenceCore",
    "apb_backward",
    "apb_backward_stages",
    "apb_forward",
    "apb_pair_fwd",
    "apb_width",
    "augmented_attention_bf16_sm100",
    "available",
    "cubin",
    "inference_core_supported",
    "supported",
]
