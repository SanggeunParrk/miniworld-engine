"""Round-to-nearest TF32 operands for the portable (Triton) fp32 kernels.

``tl.dot(a, b, input_precision="tf32")`` hands the raw fp32 bits to the tensor core, which ignores the low 13 mantissa bits: it TRUNCATES
(rounds toward zero). cuBLAS under ``allow_tf32`` rounds each operand to nearest. Truncation is biased (every operand shrinks by 2^-11 on
average), so a kernel built on it measured 3-6x the error of the PyTorch TF32 statements against an fp64 run (Transition, ConditionedTransition,
AdaLN, the attention cores), where one rounded to nearest first measures 1.0-1.3x.

Four spellings of the one rounding, all here (what each costs was measured on an H100, ConditionedTransition inference at d768, 5 samples x 384 tokens, 102 us
before: rounding the loaded tiles in the kernel +57 us, rounding everything in a launcher pass +54, rounding only the WEIGHTS in a launcher pass and the
activation tiles in the kernel +17 (launches), the same with the rounded weights cached +4):

* ``tf32_round`` (jit): adds half a TF32 ulp (0x1000) to the bit pattern of an fp32 register tile; the tensor core then drops the low 13 bits,
  which is round half away from zero, exactly ``cvt.rna.tf32.f32``, for one integer add per element. The value it returns is only meant to be
  an MMA operand (it is not a grid point itself). For the A operand of a dot (the activation, which reaches the MMA through registers anyway) it
  is free; for a B operand (weights, which the MMA reads from shared memory) it breaks the async-copy / TMA path and costs 10-50 %.
* ``tf32_dot`` (jit): ``tl.dot`` with that applied to fp32 operands when the precision is ``"tf32"``. bf16 / fp16 operands and every other
  precision compile to the plain ``tl.dot``. ``pre_a`` / ``pre_b`` say the operand arrives already rounded (see below).
* ``tf32_grid`` / ``tf32_grid_many`` (host): the exact grid value of whole tensors (add and clear the low 13 bits), up to six in ONE launch; fp32
  only, anything else is returned as is. A launcher rounds the weights it hands a kernel and passes ``pre_b``. ``tf32_cached`` memoises that
  while grad mode is off (inference: the weights do not change between calls, a launch costs more than the small kernels it feeds).
* ``tf32_grid_tile`` (jit): the exact grid value of a register tile, for a store another TF32 kernel reads (the SwiGLU output ``h`` of the expand
  kernel, read by the squeeze kernel as a pre-rounded A operand); rounding is idempotent, so a cuBLAS GEMM that reads it sees the operand it would
  have rounded itself.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def tf32_grid(t: torch.Tensor) -> torch.Tensor:
    """``t`` (fp32) rounded to the nearest TF32 value (10 mantissa bits, ties away from zero), as an fp32 tensor; other dtypes unchanged."""
    if t.dtype is not torch.float32:
        return t
    return ((t.view(torch.int32) + 0x1000) & -0x2000).view(torch.float32)


@triton.jit
def tf32_round(x):
    """fp32 ``x`` + half a TF32 ulp, bitwise: as an MMA operand it is ``x`` rounded to nearest; other dtypes are returned unchanged."""
    if x.dtype == tl.float32:
        x = (x.to(tl.int32, bitcast=True) + 0x1000).to(tl.float32, bitcast=True)
    return x


@triton.jit
def tf32_grid_tile(x):
    """The exact TF32 grid value of an fp32 register tile (for a store another TF32 kernel will read); other dtypes unchanged."""
    if x.dtype == tl.float32:
        x = ((x.to(tl.int32, bitcast=True) + 0x1000) & -0x2000).to(tl.float32, bitcast=True)
    return x


@triton.jit
def tf32_dot(a, b, acc=None, input_precision: tl.constexpr = "tf32", out_dtype: tl.constexpr = tl.float32,
             pre_a: tl.constexpr = False, pre_b: tl.constexpr = False):
    """``tl.dot(a, b, acc)`` whose fp32 operands are rounded to nearest first when ``input_precision`` is ``"tf32"`` (unless ``pre_a`` /
    ``pre_b``: that operand was rounded by the launcher or the kernel that wrote it)."""
    if input_precision == "tf32":
        if not pre_a:
            a = tf32_round(a)
        if not pre_b:
            b = tf32_round(b)
    return tl.dot(a, b, acc, input_precision=input_precision, out_dtype=out_dtype)


_BLOCK = 4096
_MAX_MANY = 6


@triton.jit
def _grid_block(src, dst, n, b, BLOCK: tl.constexpr):
    offs = b.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(src + offs, mask=m, other=0.0)
    tl.store(dst + offs, tf32_grid_tile(v), mask=m)


@triton.jit
def _grid_many_kernel(s0, d0, n0, s1, d1, n1, s2, d2, n2, s3, d3, n3, s4, d4, n4, s5, d5, n5, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b0 = tl.cdiv(n0, BLOCK)
    b1 = b0 + tl.cdiv(n1, BLOCK)
    b2 = b1 + tl.cdiv(n2, BLOCK)
    b3 = b2 + tl.cdiv(n3, BLOCK)
    b4 = b3 + tl.cdiv(n4, BLOCK)
    if pid < b0:
        _grid_block(s0, d0, n0, pid, BLOCK)
    elif pid < b1:
        _grid_block(s1, d1, n1, pid - b0, BLOCK)
    elif pid < b2:
        _grid_block(s2, d2, n2, pid - b1, BLOCK)
    elif pid < b3:
        _grid_block(s3, d3, n3, pid - b2, BLOCK)
    elif pid < b4:
        _grid_block(s4, d4, n4, pid - b3, BLOCK)
    else:
        _grid_block(s5, d5, n5, pid - b4, BLOCK)


def tf32_grid_many(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``tf32_grid`` of up to six tensors in ONE launch (a launch costs more than the rounding of a weight): the launchers of the small
    inference kernels round their activation and weights together. A tensor that is not a contiguous fp32 one goes through ``tf32_grid``."""
    if not 1 <= len(tensors) <= _MAX_MANY:
        raise ValueError(f"tf32_grid_many takes 1..{_MAX_MANY} tensors, got {len(tensors)}")
    out: list[torch.Tensor | None] = [None] * len(tensors)
    args: list = []
    blocks = 0
    device = None
    for i, t in enumerate(tensors):
        flip = t.ndim == 2 and not t.is_contiguous() and t.t().is_contiguous()      # a transposed weight view: round the dense buffer
        s = t.t() if flip else t
        if s.dtype is torch.float32 and s.is_contiguous() and s.is_cuda and s.numel() < 2**31:
            o = torch.empty_like(s)
            args += [s, o, s.numel()]
            blocks += triton.cdiv(s.numel(), _BLOCK)
            device = s.device
            out[i] = o.t() if flip else o
        else:
            out[i] = tf32_grid(t)
    if not args:
        return tuple(out)
    pad = args[-3:]                       # unused slots: re-read the last tensor's pointers with n = 0
    pad = [pad[0], pad[1], 0]
    while len(args) < 3 * _MAX_MANY:
        args += pad
    with torch.cuda.device(device):
        _grid_many_kernel[(blocks,)](*args, BLOCK=_BLOCK)
    return tuple(out)


_CACHE: dict[tuple, tuple] = {}
_CACHE_MAX = 256


def tf32_cached(sources: tuple[torch.Tensor, ...], build):
    """``build()`` (a tuple of tensors made from the parameters ``sources``: their TF32 grid values, possibly stacked) memoised while torch's grad
    mode is off, i.e. for inference, where the weights do not change between calls: a launch per call costs more than the small kernels it
    feeds (ConditionedTransition / AdaLN inference at d768: +17 us on 102). The key is the identity, version and layout of the sources'
    base tensors, so an in-place update (an optimizer step, ``load_state_dict``) misses; nothing is stored during CUDA-graph capture (a
    hit is served there: warm up first). With grad enabled (training) the weights change every step and ``build()`` runs every time."""
    import weakref

    if torch.is_grad_enabled():
        return build()
    bases = [t._base if t._base is not None else t for t in sources]
    key = tuple(id(b) for b in bases)                       # one entry per set of parameters: a new version replaces the old one
    stamp = tuple((t._version, tuple(t.shape), t.stride(), t.storage_offset(), t.dtype) for t in sources)
    hit = _CACHE.get(key)
    if hit is not None and hit[1] == stamp and all(r() is b for r, b in zip(hit[0], bases, strict=True)):
        return hit[2]
    value = build()
    if not torch.cuda.is_current_stream_capturing():
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = (tuple(weakref.ref(b, lambda _, k=key: _CACHE.pop(k, None)) for b in bases), stamp, value)     # a dead source evicts its entry
    return value
