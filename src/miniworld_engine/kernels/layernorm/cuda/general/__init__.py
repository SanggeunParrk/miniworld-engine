"""General LayerNorm on CUDA: any width (1 .. 65 536 bf16 / fp16, 32 768 fp32), any row count, bf16 / fp16 / fp32 activations, fp32
statistics, optional weight and bias (cast to fp32), forward and backward (dx and the affine gradients without atomics: a replay is
bit-identical). Plain CUDA (sm_80+), built on first use; every launch is an opaque op with a fake, so it traces under torch.compile
and captures in a CUDA graph. The same extension serves RMSNorm (`kernels/rmsnorm/cuda`): a template flag drops the mean and the bias.

Layout (``ln_common.cuh``): TPR threads per row, each holding CHUNKS vectors of VEC elements in registers; :func:`config` picks them
from the width: the fewest idle lanes, at most 32 values per thread, and TPR as close to 128 as that allows (measured on H100: 128
threads x 2-4 vectors beat 256-1024 threads x 1 vector by 10-40% at widths 2048-8192; 8 vectors per thread spill).

B200 (sm_100), bf16 / fp32 rows a (sub-)warp holds without idle lanes: ``ln_rows.cu`` (:func:`rows_launch`) -- small CTAs on a
resident grid, 16-byte vectors, shuffle reductions only, streaming loads / stores in the forward; per-CTA dw / db partials in a
fixed order. Measured against the Triton kernels and a plain copy of the same bytes (the HBM floor) in the commit that added it.

B200 (sm_100) launches its own layout (:func:`launch_config`): a (sub-)warp per row (TPR <= 32, shuffle reductions only, CHUNKS
up to 4 incl. 3: a 768-wide bf16 row is a warp of 3 vectors a lane) on a one-wave grid. The H100 layout (128 threads x one 16-byte
vector a row, shared-memory row reductions, a 4-wave grid) left a bf16 [48 L, 768] RMSNorm at ~2 TB/s.
The affine parameters are read as they are, fp32 or bf16 (no cast launch), and a bf16 parameter's gradient leaves the column pass
in bf16.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque

_dir = Path(__file__).parent

#: threads per row the builds have (4 / 8 / 16 only on the vector path)
TPRS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
CHUNKS = (1, 2, 4, 8)
DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def max_width(dtype: torch.dtype) -> int:
    return 1024 * 8 * (16 // torch.tensor([], dtype=dtype).element_size())


@functools.lru_cache(maxsize=4096)
def config(n: int, elem_bytes: int) -> tuple[int, int, int]:
    """(tpr, chunks, vec) for a row of ``n`` elements of ``elem_bytes`` bytes."""
    vmax = 16 // elem_bytes
    vec = vmax if n % vmax == 0 else 1
    nvec = math.ceil(n / vec)
    best = None
    for tpr in TPRS:
        if vec == 1 and tpr < 32:
            continue
        for ch in CHUNKS:
            if tpr * ch < nvec:
                continue
            waste = tpr * ch - nvec
            fits = ch * vec <= 32 or ch == 1
            key = (not fits, waste / (tpr * ch), abs(tpr.bit_length() - 8))
            if best is None or key < best[0]:
                best = (key, (tpr, ch, vec))
            break
    if best is None:
        raise ValueError(f"LayerNorm width {n} exceeds the general CUDA kernel's builds")
    return best[1]


def _sub_warp(n: int, elem_bytes: int, tprs=(4, 8, 16, 32), chunks=(1, 2, 3, 4)) -> tuple[int, int, int] | None:
    """(tpr, chunks, vec) of the vector path with TPR <= 32 and no idle lane-vector, or None."""
    vmax = 16 // elem_bytes
    if n % vmax:
        return None
    nvec = n // vmax
    best = None
    for tpr in tprs:
        for ch in chunks:
            if tpr * ch == nvec:
                key = (abs(ch - 2), -tpr)   # two vectors per thread first, then the wider row group
                if best is None or key < best[0]:
                    best = (key, (tpr, ch, vmax))
    return None if best is None else best[1]


@functools.lru_cache(maxsize=4096)
def config_sm100(n: int, elem_bytes: int, rows: int) -> tuple[int, int, int, int, int]:
    """(tpr, chunks, vec, rpt, waves) on B200 for ``rows`` rows of ``n`` elements: a (sub-)warp per row without idle lanes (two vectors a
    thread where the width allows) and a one-wave grid (each CTA walks its rows); the H100 layout where no such sub-warp layout exists.
    Measured (CUDA graph, B200) over N 64-3072 x bf16 / fp32 x 4K-590K rows against the H100 layout: e.g. bf16 [49152, 768] RMSNorm
    forward 45.6 -> 26.6 us, backward 65.4 -> 53.8 us; bf16 [589824, 128] LayerNorm 63.6 -> 56.3 / 106.6 -> 103.1 us."""
    sw = _sub_warp(n, elem_bytes)
    if sw is None:
        tpr, ch, vec = config(n, elem_bytes)
        return tpr, ch, vec, 1, 0
    return (*sw, 1, 1)


def rows_config(n: int, elem_bytes: int, backward: bool) -> tuple[int, int, int, int, int] | None:
    """(tpr, chunks, rows a thread, warps a CTA, CTAs a SM) of ``ln_rows.cu`` for rows of ``n`` elements, or None (no sub-warp layout
    without idle lanes). Measured on B200 (CUDA graphs) at [147456 / 589824, 128] bf16 / fp32, [147456, 256] bf16, [18432 / 49152, 768]
    bf16: the backward wants the most vectors a thread (up to 4) and two rows a thread on 4 two-warp CTAs a SM where a row group is
    under a warp, one row on 8 where it is a whole warp; the forward two vectors a thread (fp32: four), one row, 16 two-warp CTAs a SM."""
    vec = 16 // elem_bytes
    if n % vec:
        return None
    nvec = n // vec
    lays = [(t, c) for t in (4, 8, 16, 32) for c in (1, 2, 3, 4) if t * c == nvec and (c != 3 or t == 32)]
    if not lays:
        return None
    if backward:
        tpr, ch = max(lays, key=lambda tc: (tc[1], -tc[0]))
        return (tpr, ch, 2, 2, 4) if tpr < 32 else (tpr, ch, 1, 2, 8)
    want = 4 if elem_bytes == 4 else 2
    tpr, ch = min(lays, key=lambda tc: (abs(tc[1] - want), -tc[0]))
    return tpr, ch, 1, 2, 16


def rows_launch(x: torch.Tensor, backward: bool):
    """:func:`rows_config` for ``x`` [M, N] on B200 (bf16 / fp32, 16-byte aligned rows), or None (the general kernels)."""
    if not (x.is_cuda and x.dtype in (torch.bfloat16, torch.float32) and x.data_ptr() % 16 == 0):
        return None
    if _cc(x.device.index if x.device.index is not None else torch.cuda.current_device()) != (10, 0):
        return None
    return rows_config(x.shape[1], x.element_size(), backward)


def launch_config(n: int, elem_bytes: int, rows: int, device: torch.device) -> tuple[int, int, int, int, int]:
    """(tpr, chunks, vec, rpt, waves) for this card: :func:`config_sm100` on B200, :func:`config` (rpt 1) elsewhere."""
    if device.type == "cuda" and _cc(device.index if device.index is not None else torch.cuda.current_device()) == (10, 0):
        return config_sm100(n, elem_bytes, _bucket(rows))
    tpr, ch, vec = config(n, elem_bytes)
    return tpr, ch, vec, 1, 0


@functools.lru_cache(maxsize=16)
def _cc(index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(index)


def _bucket(rows: int) -> int:
    return 1 << max(0, rows.bit_length() - 1)


@functools.lru_cache(maxsize=None)
def _ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="layernorm_general_cuda_v5",
        sources=[str(_dir / "bind.cpp"), str(_dir / "ln_fwd.cu"), str(_dir / "ln_bwd.cu"), str(_dir / "ln_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "--expt-relaxed-constexpr", *gencodes("80", "90", "100")],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def supported(x: torch.Tensor) -> bool:
    """Whether the kernels take this activation (CUDA, a listed dtype, a width within the builds)."""
    return (
        x.is_cuda
        and x.dtype in DTYPES
        and x.dim() >= 1
        and 1 <= x.shape[-1] <= max_width(x.dtype)
        and (x.dtype != torch.float32 or x.shape[-1] <= 1024 * 8 * 4)
    )


def _affine(t: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    """The parameter as the kernels read it: fp32 or bf16 as it is (no cast launch), any other dtype cast to fp32."""
    if t is None:
        return x.new_empty((0,), dtype=torch.float32)
    t = t.detach()
    if t.dtype not in (torch.float32, torch.bfloat16) or t.device != x.device:
        t = t.to(device=x.device, dtype=torch.float32)
    return t.contiguous()


def _grad_bf16(*ts: torch.Tensor) -> bool:
    """The column pass writes the affine gradients in bf16 when every given parameter is bf16 (else fp32)."""
    ts = [t for t in ts if t.numel()]
    return bool(ts) and all(t.dtype is torch.bfloat16 for t in ts)


def _fwd_fake(x, w, b, eps, save):
    """y like x; mean / rstd fp32 [M] when save, else empty."""
    m = x.shape[0] if save else 0
    f32 = dict(dtype=torch.float32, device=x.device)
    return torch.empty_like(x), torch.empty((m,), **f32), torch.empty((m,), **f32)


@opaque(fake=_fwd_fake, name="layernorm_general_fwd")
def _fwd(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float, save: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm forward of contiguous x [M, N]; w / b fp32 or bf16 [N] or empty. (y, mean, rstd)."""
    rl = rows_launch(x, False)
    if rl is not None:
        return tuple(_ext().rows_fwd(x, w, b, float(eps), bool(save), False, *rl))
    tpr, ch, vec, rpt, waves = launch_config(x.shape[1], x.element_size(), x.shape[0], x.device)
    y, mean, rstd = _ext().fwd(x, w, b, float(eps), bool(save), False, tpr, ch, vec, rpt, waves)
    return y, mean, rstd


def _bwd_fake(dy, x, w, b, mean, rstd, wgrad):
    """dx like x; dw / db [N] (bf16 when w and b are, else fp32) when wgrad, else empty."""
    gdt = torch.bfloat16 if _grad_bf16(w, b) else torch.float32
    n = x.shape[1] if wgrad else 0
    return torch.empty_like(x), torch.empty((n,), dtype=gdt, device=x.device), torch.empty((n,), dtype=gdt, device=x.device)


@opaque(fake=_bwd_fake, name="layernorm_general_bwd")
def _bwd(dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor,
         wgrad: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm backward from the saved mean / rstd. (dx, dw, db); the affine gradients only when wgrad (``b`` only names their
    dtype)."""
    rl = rows_launch(x, True) if dy.data_ptr() % 16 == 0 else None
    if rl is not None:
        return tuple(_ext().rows_bwd(dy, x, w, mean, rstd, bool(wgrad), False, *rl, _grad_bf16(w, b)))
    tpr, ch, vec, rpt, _ = launch_config(x.shape[1], x.element_size(), x.shape[0], x.device)
    dx, dw, db = _ext().bwd(dy, x, w, mean, rstd, bool(wgrad), False, tpr, ch, vec, rpt, 0, _grad_bf16(w, b))
    return dx, dw, db


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, weight, bias, eps):
        w, b = _affine(weight, x2), _affine(bias, x2)
        y, mean, rstd = _fwd(x2, w, b, eps, True)
        ctx.save_for_backward(x2, w, b, mean, rstd)
        ctx.has = (weight is not None, bias is not None)
        ctx.dtypes = (None if weight is None else weight.dtype, None if bias is None else bias.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        x2, w, b, mean, rstd = ctx.saved_tensors
        has_w, has_b = ctx.has
        wgrad = (has_w and ctx.needs_input_grad[1]) or (has_b and ctx.needs_input_grad[2])
        dx, dw, db = _bwd(dy.contiguous(), x2, w, b, mean, rstd, bool(wgrad))
        gw = dw.to(ctx.dtypes[0]) if has_w and ctx.needs_input_grad[1] else None
        gb = db.to(ctx.dtypes[1]) if has_b and ctx.needs_input_grad[2] else None
        return dx, gw, gb, None


def layernorm_cuda(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm over the last dimension; ``y`` in ``x``'s dtype, fp32 statistics; weight / bias optional, any float dtype (fp32
    gradients handed back in the parameters' dtype). On H100 the rows ``pipe`` takes go to its pipelined kernels (``pipe.serves``), as
    in ``rmsnorm_cuda``: one CUDA LayerNorm entry, the general kernels for every other width or card."""
    from miniworld_engine.kernels.layernorm.cuda import pipe

    if pipe.serves(x):
        return pipe.layernorm_pipe(x, weight, bias, eps)
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, weight, bias))
    if not needs_grad:
        y, _, _ = _fwd(x2, _affine(weight, x2), _affine(bias, x2), float(eps), False)
        return y.reshape(shape)
    return _LayerNorm.apply(x2, weight, bias, float(eps)).reshape(shape)


__all__ = ["config", "config_sm100", "launch_config", "layernorm_cuda", "rows_config", "rows_launch", "supported"]
