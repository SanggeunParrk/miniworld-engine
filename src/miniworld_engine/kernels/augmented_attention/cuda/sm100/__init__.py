"""sm_100a (B200) pair-bias attention core, bf16 operands, fp32 accumulation, forward AND backward, plus the gated
inference core the fused token DiT step uses.

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

Numerics against fp64 at A = 48: O 1.6e-3, dq 3.0e-3, dk 2.9e-3, dv 2.9e-3, dbias 2.4e-3 (bf16-input-rounding floor).

The cubins are built with this environment's nvcc on first use and cached under ``MINIWORLD_ENGINE_JIT_ROOT``
(keyed by the sources and flags). They are launched through the CUDA driver (``driver.py``, cuda.bindings) with 2-D
TMA descriptors encoded on the host; launches go on torch's current stream and are CUDA-graph capturable.

``supported()`` is the whole gate: sm_100, 16 heads x 48, B == 1, an even A, L a multiple of 128, no key mask.
Everything else keeps the Triton path, and so do fake tensors and torch.compile tracing (``available()``).
"""

from __future__ import annotations

import functools
import hashlib
import os
import subprocess
import warnings
from pathlib import Path

import torch
import triton
import triton.language as tl

H, D = 16, 48
_dir = Path(__file__).parent
_SMEM = 232448
_NVCC_FLAGS = ("-std=c++17", "-O3", "-arch=sm_100a", "--cubin", "-lineinfo")


# --------------------------------------------------------------------------------------------------- build
def _jit_root() -> Path:
    return Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit")) / "augattn_sm100"


@functools.lru_cache(maxsize=None)
def cubin(stem: str, defs: tuple[str, ...] = (), src_dir: str | None = None) -> str:
    """Path of ``<src_dir>/<stem>.cu`` (default: this directory) built for sm_100a with ``-D`` ``defs`` and this
    directory's ``sm100.cuh`` on the include path; rebuilt only when a source or a flag changes."""
    from miniworld_engine.kernels._nvcc import nvcc_path

    nvcc = nvcc_path()
    if not nvcc:
        raise RuntimeError("no nvcc matching torch.version.cuda")
    src = Path(src_dir or _dir) / f"{stem}.cu"
    flags = (*_NVCC_FLAGS, f"-I{_dir}", *("-D" + d for d in defs))
    h = hashlib.sha256()
    for f in (_dir / "sm100.cuh", src):
        h.update(f.read_bytes())
    h.update(" ".join((nvcc, *flags)).encode())
    out = _jit_root() / f"{stem}_{h.hexdigest()[:16]}.cubin"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        res = subprocess.run([nvcc, *flags, str(src), "-o", str(tmp)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"nvcc {stem}.cu failed:\n{res.stderr[-4000:]}")
        os.replace(tmp, out)
    return str(out)


@functools.lru_cache(maxsize=None)
def _sm100_kernel(stem: str, func: str, device_index: int, pdl: bool = False, src_dir: str | None = None,
            defs: tuple[str, ...] = (), cluster: int | None = None, smem: int = _SMEM):
    from . import driver

    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem, defs, src_dir=src_dir), func, smem, cluster=cluster, pdl=pdl)


# --------------------------------------------------------------------------------------------------- gate
@functools.lru_cache(maxsize=8)
def _is_blackwell(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (10, 0)


def _index(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def supported(q: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor | None = None, bias_head_major: bool = False) -> bool:
    """q [A, 1, L, 16, 48] (any float dtype) with A even, a bias of [1, L, L, 16] (``bias_head_major``: [16, 1, L, L]),
    no key mask, on sm_100 with L a multiple of 128 (the kernels' tile shapes and sample pairing, not a policy)."""
    if os.environ.get("MINIWORLD_AUGATTN_BF16_SM100", "1") == "0" or mask is not None:
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


def available(q: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor | None = None, bias_head_major: bool = False) -> bool:
    """``supported()`` plus successful builds; a build failure warns once and keeps the Triton path."""
    global _BUILD_FAILED
    if _BUILD_FAILED or not supported(q, bias, mask, bias_head_major):
        return False
    from torch._subclasses.fake_tensor import FakeTensor
    if torch.compiler.is_compiling() or isinstance(q, FakeTensor) or isinstance(bias, FakeTensor):
        return False
    try:
        for stem, func in (("attn_fwd2", "augattn_fwd2_sm100"), ("attn_dqb", "augattn_dqb_sm100"), ("attn_dkv", "augattn_dkv_sm100")):
            _sm100_kernel(stem, func, _index(q))
    except Exception as exc:  # noqa: BLE001
        _BUILD_FAILED = True
        warnings.warn(f"sm100 bf16 augmented attention unavailable, keeping the Triton path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


# --------------------------------------------------------------------------------------------------- launches
def _tm(t, dims, stride_bytes, box, **kw):
    from .driver import TensorMap
    return TensorMap(t, dims, stride_bytes, box, **kw)


def _grid(items: int, device_index: int) -> tuple[int, int, int]:
    return (min(torch.cuda.get_device_properties(device_index).multi_processor_count, items), 1, 1)


def forward(q2, k2, v2, bias_hll, A, L):
    """q2, k2, v2 [A L, 768] bf16 contiguous, bias [16, L, L] bf16 (natural units) -> O [A L, 768] fp32, LSE [A, 16, L] (log2)."""
    dev = _index(q2)
    O = torch.empty(A * L, H * D, device=q2.device, dtype=torch.float32)
    LSE = torch.empty(A, H, L, device=q2.device, dtype=torch.float32)
    maps = (_tm(q2, [H * D, A * L], H * D * 2, [D, 128]), _tm(k2, [H * D, A * L], H * D * 2, [D, 64]),
            _tm(v2, [H * D, A * L], H * D * 2, [D, 64]), _tm(bias_hll, [L, H * L], L * 2, [64, 128]),
            _tm(O, [H * D, A * L], H * D * 4, [32, 128], swizzle=128, dtype="f32"),
            _tm(O, [H * D, A * L], H * D * 4, [16, 128], swizzle=64, dtype="f32"))
    _sm100_kernel("attn_fwd2", "augattn_fwd2_sm100", dev)(_grid((A // 2) * H * (L // 128), dev), (384, 1, 1), *maps, O, LSE, int(L), int(A))
    return O, LSE


def backward(q2, k2, v2, dob, bias_hll, LSE, Dd, A, L):
    """dQ, dK, dV [A L, 768] fp32 and dbias [16, L, L] fp32. dob [A L, 768] bf16, Dd = rowsum(dO O) [A, 16, L] fp32."""
    dev = _index(q2)
    bias_t = bias_transpose(bias_hll)
    DQ = torch.empty(A * L, H * D, device=q2.device, dtype=torch.float32)
    DB = torch.empty(H, L, L, device=q2.device, dtype=torch.float32)
    DK, DV = torch.empty_like(DQ), torch.empty_like(DQ)
    f32 = dict(dtype="f32")
    dkv_maps = (_tm(q2, [H * D, A * L], H * D * 2, [D, 64]), _tm(k2, [H * D, A * L], H * D * 2, [D, 128]),
                _tm(v2, [H * D, A * L], H * D * 2, [D, 128]), _tm(dob, [H * D, A * L], H * D * 2, [D, 64]),
                _tm(bias_t, [L, H * L], L * 2, [64, 128]),
                _tm(DK, [H * D, A * L], H * D * 4, [32, 128], swizzle=128, **f32), _tm(DK, [H * D, A * L], H * D * 4, [16, 128], swizzle=64, **f32),
                _tm(DV, [H * D, A * L], H * D * 4, [32, 128], swizzle=128, **f32), _tm(DV, [H * D, A * L], H * D * 4, [16, 128], swizzle=64, **f32))
    dqb_maps = (*(_tm(t, [H * D, A * L], H * D * 2, [D, 128]) for t in (q2, k2, v2, dob)), _tm(bias_hll, [L, H * L], L * 2, [64, 128]),
                _tm(DQ, [H * D, A * L], H * D * 4, [32, 128], swizzle=128, **f32), _tm(DQ, [H * D, A * L], H * D * 4, [16, 128], swizzle=64, **f32))
    # attn_dkv zero-fills dQ on the way; attn_dqb then adds one partial per 128-key chunk.
    _sm100_kernel("attn_dkv", "augattn_dkv_sm100", dev)(_grid(A * H * (L // 128), dev), (384, 1, 1), *dkv_maps, LSE, Dd, DK, DV, DQ, int(L), int(A))
    _sm100_kernel("attn_dqb", "augattn_dqb_sm100", dev)(_grid(H * (L // 128) * (L // 128), dev), (384, 1, 1), *dqb_maps, LSE, Dd, DQ, DB, int(L), int(A))
    return DQ, DK, DV, DB


# --------------------------------------------------------------------------------------------------- glue
@triton.jit
def _prep_do_kernel(do, o, dob, dd, L, NH: tl.constexpr, DH: tl.constexpr, DP: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    hh = tl.arange(0, NH)
    dcol = tl.arange(0, DP)
    ptr = r[:, None, None] * (NH * DH) + (hh[:, None] * DH + dcol[None, :])[None, :, :]
    msk = (dcol[None, :] < DH)[None, :, :] & (r[:, None, None] >= 0)
    g = tl.load(do + ptr, mask=msk, other=0.0).to(tl.float32)
    x = tl.load(o + ptr, mask=msk, other=0.0)
    tl.store(dob + ptr, g.to(tl.bfloat16), mask=msk)
    s = tl.sum(g * x, axis=2)
    tl.store(dd + ((r // L)[:, None] * NH + hh[None, :]) * L + (r % L)[:, None], s)


def prep_do(do, O, A, L):
    """dO -> bf16 and D = rowsum(dO O) as [A, H, L] fp32."""
    do = do.reshape(A * L, H * D).contiguous()
    dob = torch.empty(A * L, H * D, device=do.device, dtype=torch.bfloat16)
    dd = torch.empty(A, H, L, device=do.device, dtype=torch.float32)
    _prep_do_kernel[((A * L) // 8,)](do, O, dob, dd, L, NH=H, DH=D, DP=64, ROWS=8)
    return dob, dd


@triton.jit
def _transpose_kernel(src, dst, L, BT: tl.constexpr):
    h, i, j = tl.program_id(2), tl.program_id(0) * BT, tl.program_id(1) * BT
    ri, rj = i + tl.arange(0, BT), j + tl.arange(0, BT)
    x = tl.load(src + h * L * L + ri[:, None] * L + rj[None, :])
    tl.store(dst + h * L * L + rj[:, None] * L + ri[None, :], tl.trans(x))


def bias_transpose(bias):
    """[H, L, L] -> [H, L(key), L(query)] (attn_dkv reads its key rows contiguously)."""
    Hh, L, _ = bias.shape
    out = torch.empty_like(bias)
    _transpose_kernel[(L // 64, L // 64, Hh)](bias, out, L, BT=64)
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
class GatedInferenceCore:
    """The fused token DiT step's core on sm_100a (``attn_inf.cu``): reads q | k | v | g as column views of the
    [S L, 4 D] q|k|v|g GEMM output (logits pre-scaled into exp2 units), the block's head-major hoisted bias (bf16,
    [nb H, L, L] rows), and writes sigmoid(g) * o over q in bf16. Bindings (TMA descriptors) are cached per buffer."""

    def __init__(self, device_index: int):
        self.k = _sm100_kernel("attn_inf", "augattn_inf_sm100", device_index)
        self.nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
        self.runs: dict = {}

    def _bind(self, qkvg, bias, block, S, L):
        M, D4 = qkvg.shape
        Dm, DH = D4 // 4, D4 // 4 // H
        rs = D4 * qkvg.element_size()
        q, k, v, g = (qkvg[:, i * Dm:(i + 1) * Dm] for i in range(4))
        bv = bias[block * H:(block + 1) * H]
        maps = (_tm(q, [Dm, M], rs, [DH, 128]), _tm(k, [Dm, M], rs, [DH, 64]), _tm(v, [Dm, M], rs, [DH, 64]),
                _tm(bv, [L, H * L], L * bias.element_size(), [64, 128]), _tm(g, [Dm, M], rs, [DH, 128]))
        grid = (min(self.nsm, ((S + 1) // 2) * H * (L // 128)), 1, 1)
        mq, mk, mv, mb, mg = maps

        def run():
            self.k(grid, (384, 1, 1), mq, mk, mv, mb, mg, mq, int(L), int(S))
        run.keep = (maps, qkvg, bias)
        return run

    def __call__(self, qkvg, bias, block, S):
        L = qkvg.shape[0] // S
        key = (qkvg.data_ptr(), bias.data_ptr(), block, S, L)
        run = self.runs.get(key)
        if run is None:
            run = self.runs[key] = self._bind(qkvg, bias, block, S, L)
        run()
        return qkvg


def inference_core_supported(dtype: torch.dtype, L: int, d: int, h: int, device_index: int) -> bool:
    """bf16, d 768 as 16 x 48, L a multiple of 128, sm_100."""
    if os.environ.get("MINIWORLD_AUGATTN_BF16_SM100", "1") == "0":
        return False
    return dtype is torch.bfloat16 and d == H * D and h == H and L % 128 == 0 and _is_blackwell(device_index)


__all__ = ["GatedInferenceCore", "augmented_attention_bf16_sm100", "available", "cubin", "inference_core_supported", "supported"]
