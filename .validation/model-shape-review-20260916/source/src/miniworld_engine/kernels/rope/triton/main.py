"""3D-RoPE application, fused.

ESMFold2's SWA atom attention rotates q and k by a batch-dependent ``cos``/``sin`` before the
windowed attention. The eager form (``modules/swa_atom_attention/apply_rotary_emb_3d``) is four
HBM round-trips and two temporaries per call: it ``repeat``s cos/sin to ``2*half``, gathers
``rotate_half`` with a ``chunk``+``cat``, promotes bf16 to fp32 against the fp32 angles, and
concatenates the rotated block back with the unrotated tail. This does it in one pass: a program
owns a tile of rows and the whole head dim, loads the row's cos/sin once, and writes the rotated
q (or k) with the tail passed through -- no repeat, no gather, no cat.

The rotation is the standard interleaved-half convention, matching `_rotate_half`:

    lo, hi = x[:half], x[half:2*half]                 # over the head dim
    x'[:half]      = lo*cos - hi*sin
    x'[half:2*half] = hi*cos + lo*sin
    x'[2*half:]    = x[2*half:]                        # unrotated tail

``cos``/``sin`` are (N, S, half): one angle set per (batch-row, position), shared across heads
and across the two halves. ``half`` is a constexpr so the split is resolved at compile time.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import atom_key
from miniworld_engine.kernels._compile import opaque


@triton.autotune(configs=configs_for("rope_fwd_triton"), key=["shape_key"])
@triton.jit
def rope_3d_kernel(
    X, COS, SIN, Y,
    stride_xr, stride_xh, stride_cr,
    M, H: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    shape_key,
):
    """``X``/``Y`` are (M, H, D) with M = N*S rows; COS/SIN are (M, HALF), shared over H."""
    row = tl.program_id(0).to(tl.int64)
    rows = row * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    row_mask = rows < M

    lo_c = tl.arange(0, HALF)
    cos = tl.load(COS + rows[:, None] * stride_cr + lo_c[None, :], mask=row_mask[:, None], other=0.0)
    sin = tl.load(SIN + rows[:, None] * stride_cr + lo_c[None, :], mask=row_mask[:, None], other=0.0)

    for h in range(H):
        base = X + rows[:, None] * stride_xr + h * stride_xh
        obase = Y + rows[:, None] * stride_xr + h * stride_xh
        lo = tl.load(base + lo_c[None, :], mask=row_mask[:, None], other=0.0).to(tl.float32)
        hi = tl.load(base + (HALF + lo_c)[None, :], mask=row_mask[:, None], other=0.0).to(tl.float32)
        # cos/sin are fp32 (angle precision); the rotation is done in fp32 and cast back, so q/k
        # keep the input dtype -- the eager path's `.to(x.dtype)`.
        rlo = (lo * cos - hi * sin).to(X.dtype.element_ty)
        rhi = (hi * cos + lo * sin).to(X.dtype.element_ty)
        tl.store(obase + lo_c[None, :], rlo, mask=row_mask[:, None])
        tl.store(obase + (HALF + lo_c)[None, :], rhi, mask=row_mask[:, None])
        # The unrotated tail (active rope frequencies underfill D/2): copy 2*HALF..D straight.
        if D > 2 * HALF:
            tail = 2 * HALF + tl.arange(0, D - 2 * HALF)
            t = tl.load(base + tail[None, :], mask=row_mask[:, None], other=0.0)
            tl.store(obase + tail[None, :], t, mask=row_mask[:, None])


def _rope_fake(x_2d, cos_2d, sin_2d, h, d, half, shape_key):
    """The rotated activation, same (M, H*D) shape and dtype as ``x_2d``."""
    return torch.empty_like(x_2d)


@opaque(fake=_rope_fake, name="rope_3d")
def _rope(x_2d: torch.Tensor, cos_2d: torch.Tensor, sin_2d: torch.Tensor,
          h: int, d: int, half: int, shape_key: int) -> torch.Tensor:
    """``(M, H*D)`` in, rotated ``(M, H*D)`` out. cos/sin are ``(M, half)``."""
    m = x_2d.shape[0]
    y = torch.empty_like(x_2d)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M1"]),)
    rope_3d_kernel[grid](
        x_2d, cos_2d, sin_2d, y,
        x_2d.stride(0), d, cos_2d.stride(0),
        m, h, d, half, shape_key=shape_key,
    )
    return y


class _RoPE3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        n, s, h, d = x.shape
        half = cos.shape[-1]
        x2 = x.reshape(n * s, h * d)
        if x2.stride(1) != 1:
            x2 = x2.contiguous()
        c2 = cos.reshape(n * s, half).contiguous()
        s2 = sin.reshape(n * s, half).contiguous()
        # level=atom: the atom count is the ROW count N*S (the kernel loops H internally), so it
        # keys through atom_key -- not both_key or length_of, which would read H from the 4-D
        # (N, S, H, D) shape.
        key = atom_key(n * s, D=d)
        y = _rope(x2, c2, s2, h, d, half, key)
        ctx.save_for_backward(c2, s2)
        ctx.shape, ctx.hdhalf, ctx.key = (n, s, h, d), (h, d, half), key
        return y.reshape(n, s, h, d)

    @staticmethod
    def backward(ctx, dy):
        # RoPE is a rotation: its Jacobian is the transpose, i.e. rotate by -angle. So the
        # backward is the SAME kernel with sin negated -- no new kernel, no saved activation
        # beyond cos/sin (which are step-invariant and tiny).
        c2, s2 = ctx.saved_tensors
        n, s, h, d = ctx.shape
        _h, _d, half = ctx.hdhalf
        dy2 = dy.reshape(n * s, h * d)
        if dy2.stride(1) != 1:
            dy2 = dy2.contiguous()
        dx = _rope(dy2, c2, -s2, h, d, half, ctx.key)
        return dx.reshape(n, s, h, d), None, None


def triton_rope_3d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 3D RoPE to ``x`` (N, S, H, D) with ``cos``/``sin`` (N, S, half), in one pass.

    Only the leading ``2*half`` channels of the head dim are rotated; a trailing block (when the
    active rope frequencies underfill ``D/2``) passes through. Replaces
    ``modules/swa_atom_attention/apply_rotary_emb_3d``, which is the same math in four passes.
    """
    return _RoPE3D.apply(x, cos, sin)
