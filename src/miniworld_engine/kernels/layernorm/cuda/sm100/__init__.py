"""B200 (sm_100a) LayerNorm / RMSNorm row kernels (``ln_rows_sm100.cu``): the sub-warp row kernels of the B200 port branch
(``origin/b200/ladder-final`` ``kernels/layernorm/cuda/general/ln_rows.cu``), built for sm_100a only (extension
``layernorm_rows_sm100``; no other card builds or runs them).

A row of N elements (a multiple of 16 bytes) is TPR <= 32 lanes x CH 16-byte vectors, no idle lane; a CTA is a few warps on a resident
grid (CTAs a SM), each thread loads its RPT rows before any reduction, shuffle reductions only. The forward reads x through the
non-coherent path (fp32: no L1 allocation, streaming stores). The backward writes dx and per-CTA fp32 partial rows of dw / db in a
fixed order, then one column pass adds them in a fixed order: a replay is bit-identical. fp32 statistics; the affine parameters are
read as they are (fp32 or bf16).

Skewed rows (``ln_skew_sm100.cuh``, 2026-11-17): bf16 / fp32 rows of an odd width (267 / 451 / 651 / 831 / 833) are not a whole number of
16-byte vectors. They run
the same register form on the aligned vectors a row spans (masked first / last vector), one row phase (row index mod 8 bf16 / 4 fp32) a CTA, the
affine parameters as a phase-shifted copy in shared memory: :func:`layernorm_skew`.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque

_dir = Path(__file__).parent
DTYPES = (torch.bfloat16, torch.float32)


@functools.lru_cache(maxsize=None)
def _ext():  # noqa: D103
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="layernorm_rows_sm100",
        sources=[str(_dir / "ln_rows_sm100.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_100a,code=sm_100a",
            f"-I{_dir}",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


#: (tpr, ch) layouts built (``MWLN_ROWS_LAYOUTS``)
LAYOUTS = (
    (4, 1),
    (4, 2),
    (4, 4),
    (8, 1),
    (8, 2),
    (8, 3),
    (8, 4),
    (16, 1),
    (16, 2),
    (16, 3),
    (16, 4),
    (32, 1),
    (32, 2),
    (32, 3),
    (32, 4),
    (32, 6),
    (160, 2),
)

#: launch overrides {(n, elem_bytes, backward): (tpr, ch, rpt, warps, ctas)} measured on the B200 (empty: :func:`rows_config` rule)
TUNED: dict[tuple[int, int, bool], tuple[int, int, int, int, int]] = {}

#: backward launches measured on the B200 (round 6, 2026-11-17; graph timing, L2 flushed, the training step forward + backward + column
#: pass; docs/b200/transition.md): two rows a thread on four-warp CTAs, four CTAs a SM, two 16-byte vectors a thread where the width
#: allows, e.g. fp32 [589824, 128] 249.8 (8 x 4 at 2 rows, 2-warp CTAs) -> 239.3 us with the forward below (Triton 237.3), fp32
#: [589824, 64] 132.9 -> 126.8 (Triton 128.9), bf16 [589824, 256] 274.2 -> 266.0, bf16 [262144, 64] 51.0 -> 43.0.
#: {(n, element bytes): (tpr, ch, rpt, warps, ctas)}
BWD_LAUNCH: dict[tuple[int, int], tuple[int, int, int, int, int]] = {
    (64, 4): (8, 2, 2, 4, 4),
    (128, 4): (16, 2, 2, 4, 4),
    (256, 4): (32, 2, 2, 4, 4),
    (384, 4): (32, 3, 4, 4, 2),
    (512, 4): (32, 4, 1, 4, 4),
    (64, 2): (4, 2, 2, 4, 4),
    (128, 2): (8, 2, 2, 4, 4),
    (256, 2): (16, 2, 2, 4, 4),
}

#: fp32 widths whose forward is the non-persistent grid on four-warp CTAs at every row count (round 6; the training forward that saves
#: the statistics: fp32 [589824, 128] 104 -> 93 us of the step, [147456, 512] 251.7 -> 239.9 the step)
FWD_DIRECT4: set[tuple[int, int]] = {(64, 4), (128, 4), (256, 4), (384, 4), (512, 4)}

#: bf16 widths whose inference forward from the given row count is the lean one-warp grid (``fwd_lean``, ``ctas`` -1) at the given rows
#: a thread (round 6, graph timing): bf16 [147456 / 589824, 384] 51.0 / 172.0 -> 46.9 / 155.5 us (row kernels' resident grid), [147456 /
#: 589824 / 1048576, 512] 59.2 / 204.7 / 356.4 -> 57.2 / 194.3 / 337.7. {(n, element bytes): (min rows, rpt)}
LEAN_FROM: dict[tuple[int, int], tuple[int, int]] = {(384, 2): (65_536, 4), (512, 2): (65_536, 2)}


@functools.lru_cache(maxsize=4096)
def rows_config(n: int, elem_bytes: int, backward: bool) -> tuple[int, int, int, int, int] | None:
    """(tpr, ch, rpt, warps, ctas) for rows of ``n`` elements, or None (no sub-warp layout without idle lanes). The branch's rule:
    the backward wants the most vectors a thread and two rows a thread on 4 two-warp CTAs a SM where a row group is under a warp,
    one row on 8 where it is a whole warp; the forward two vectors a thread (fp32: four), one row, 16 two-warp CTAs a SM. A row wider
    than a warp is one row a CTA (forward: a CTA a row; backward: four rows a thread, two CTAs a SM)."""
    t = TUNED.get((n, elem_bytes, backward)) or (BWD_LAUNCH.get((n, elem_bytes)) if backward else None)
    if t is not None:
        return t
    vec = 16 // elem_bytes
    if n % vec:
        return None
    nvec = n // vec
    lays = [(t, c) for t, c in LAYOUTS if t * c == nvec]
    if not lays:
        return None
    # a row wider than a warp (bf16 2560: 160 lanes x 2): one row a CTA, four rows a thread and one CTA a SM in the backward
    if lays[0][0] > 32:
        tpr, ch = lays[0]
        return (tpr, ch, 4, tpr // 32, 1) if backward else (tpr, ch, 1, tpr // 32, 0)
    if backward:
        tpr, ch = max(lays, key=lambda tc: (tc[1], -tc[0]))
        return (tpr, ch, 2, 2, 4) if tpr < 32 else (tpr, ch, 1, 2, 8)
    want = 4 if elem_bytes == 4 else 2
    tpr, ch = min(lays, key=lambda tc: (abs(tc[1] - want), -tc[0]))
    return tpr, ch, 1, 2, 16


#: forward row counts below which the non-persistent grid (a tile a CTA: ``ctas`` 0) is launched, {(n, elem_bytes): rows}. Measured on
#: the B200 (graph timing, L2 flushed; docs/b200/transition.md): fp32 768 token streams, [5, L, 768] 8.1 / 12.2 us at L 384 / 1024
#: against 8.3 / 12.2 on the resident grid, from 3 200 rows the resident grid (L 640 / 768: 10.1 against 10.7 / 12.0).
DIRECT_BELOW: dict[tuple[int, int], int] = {(768, 4): 3072}


#: (width, element bytes) whose resident forward keeps the next tile's loads in flight (fwd<PF>; built for the one-row two-warp launch of
#: the 16 x 3 layout). Measured on the B200 (inference, graph timing): bf16 384 [65536 / 147456 / 589824, 384]
#: 28.4 / 54.9 / 188 -> 26.5 / 51.0 / 172 us; bf16 64 / 128 / 256: no gain (a timer step slower at 1-4 of 5 sizes), not taken.
ROWS_PF: set[tuple[int, int]] = {(384, 2)}

#: fused backward (round 8, 2026-11-18): the column CTAs of ln_rows_sm100.cu (mwln::col_role, thread-block clusters of K, DSMEM fold) sum
#: dw / db inside the backward's launch (no partial rows, no column-pass launch; activations under 8 MB only). {(width, element bytes):
#: ((max rows, K), ...)}: the first span holding the row count gives K; none: the separate column pass. Measured on the B200 (training
#: steps, graph timing; docs/b200/transition.md round 8): fp32 [384 / 640 / 768, 768] 14.1 / 16.2 / 16.2 -> 12.1 / 14.1 / 14.1 us. Not taken
#: (slower): the skewed widths (bf16 / fp32 451-833: 14.1-16.2 -> 16.2-22.3), bf16 2560, bf16 / fp32 64-512 from 16 384 rows.
FUSE_K: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {(768, 4): ((4095, 16),)}


def fuse_groups(n: int, elem_bytes: int, rows: int) -> int:
    """The cluster size K of a fused backward of ``rows`` rows of width ``n`` (0: not fused, :data:`FUSE_K`)."""
    for max_rows, k in FUSE_K.get((n, elem_bytes), ()):
        if rows <= max_rows:
            return k
    return 0



def fwd_config(n: int, elem_bytes: int, rows: int) -> tuple[int, int, int, int, int] | None:
    """:func:`rows_config` of the forward for ``rows`` rows: the non-persistent grid below ``DIRECT_BELOW`` (unless ``TUNED``)."""
    cfg = rows_config(n, elem_bytes, False)
    if cfg is None or (n, elem_bytes, False) in TUNED:
        return cfg
    if (n, elem_bytes) in FWD_DIRECT4:
        return (cfg[0], cfg[1], 1, 4, 0)
    lean = LEAN_FROM.get((n, elem_bytes))
    if lean is not None and rows >= lean[0] and cfg[0] <= 32:
        return (cfg[0], cfg[1], lean[1], 1, -1)
    if rows < DIRECT_BELOW.get((n, elem_bytes), 0):
        return (*cfg[:4], 0)
    return cfg


def supported(x: torch.Tensor) -> bool:
    """Whether the row kernels take ``x``: CUDA bf16 / fp32 with a sub-warp layout for its width."""
    return (
        x.is_cuda and x.dtype in DTYPES and x.dim() >= 1 and x.numel() > 0 and rows_config(x.shape[-1], x.element_size(), False) is not None
    )


def _affine(t: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    """The parameter as the kernels read it: fp32 or bf16 as it is, any other dtype cast to fp32; empty when absent."""
    if t is None:
        return x.new_empty((0,), dtype=torch.float32)
    t = t.detach()
    if t.dtype not in (torch.float32, torch.bfloat16) or t.device != x.device:
        t = t.to(device=x.device, dtype=torch.float32)
    return t.contiguous()


def _grad_bf16(*ts: torch.Tensor) -> bool:
    ts2 = [t for t in ts if t.numel()]
    return bool(ts2) and all(t.dtype is torch.bfloat16 for t in ts2)


def _fwd_fake(x, w, b, eps, save, rms):
    m = x.shape[0] if save else 0
    f32 = dict(dtype=torch.float32, device=x.device)
    return torch.empty_like(x), torch.empty((0 if rms else m,), **f32), torch.empty((m,), **f32)


@opaque(fake=_fwd_fake, name="layernorm_rows_sm100_fwd")
def _fwd(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float, save: bool, rms: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward of contiguous x [M, N]: (y, mean, rstd); the statistics only when ``save`` (no mean for RMS)."""
    cfg = fwd_config(x.shape[1], x.element_size(), x.shape[0])
    assert cfg is not None
    y, mean, rstd = _ext().rows_fwd(x, w, b, float(eps), bool(save), bool(rms), *cfg, (x.shape[1], x.element_size()) in ROWS_PF)
    return y, mean, rstd


def _bwd_fake(dy, x, w, b, mean, rstd, wgrad, rms):
    """dx like x; dw [N] (bf16 when every parameter is, else fp32) when wgrad, db too unless RMS; empty fp32 otherwise."""
    gdt = torch.bfloat16 if _grad_bf16(w, b) else torch.float32
    n = x.shape[1]
    dw = torch.empty((n,), dtype=gdt, device=x.device) if wgrad else torch.empty((0,), dtype=torch.float32, device=x.device)
    db = torch.empty((n,), dtype=gdt, device=x.device) if wgrad and not rms else torch.empty((0,), dtype=torch.float32, device=x.device)
    return torch.empty_like(x), dw, db


@opaque(fake=_bwd_fake, name="layernorm_rows_sm100_bwd")
def _bwd(
    dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, wgrad: bool, rms: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward from the saved statistics: (dx, dw, db); the affine gradients only when ``wgrad`` (``b`` names their dtype)."""
    cfg = rows_config(x.shape[1], x.element_size(), True)
    assert cfg is not None
    dx, dw, db = _ext().rows_bwd(dy, x, w, mean, rstd, bool(wgrad), bool(rms), *cfg, _grad_bf16(w, b), False, fuse_groups(x.shape[1], x.element_size(), x.shape[0]))
    return dx, dw, db


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, weight, bias, eps, rms):
        w, b = _affine(weight, x2), _affine(bias, x2)
        y, mean, rstd = _fwd(x2, w, b, eps, True, rms)
        ctx.save_for_backward(x2, w, b, mean, rstd)
        ctx.cfg = (
            rms,
            weight is not None,
            bias is not None,
            None if weight is None else weight.dtype,
            None if bias is None else bias.dtype,
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        x2, w, b, mean, rstd = ctx.saved_tensors
        rms, has_w, has_b, wdt, bdt = ctx.cfg
        wgrad = (has_w and ctx.needs_input_grad[1]) or (has_b and ctx.needs_input_grad[2])
        dy = dy.contiguous()
        if dy.data_ptr() % 16:
            dy = dy.clone()
        dx, dw, db = _bwd(dy, x2, w, b, mean, rstd, bool(wgrad), rms)
        gw = dw.to(wdt) if has_w and ctx.needs_input_grad[1] else None
        gb = db.to(bdt) if has_b and ctx.needs_input_grad[2] else None
        return dx, gw, gb, None, None


def _needs_grad(*ts: torch.Tensor | None) -> bool:
    return torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in ts)


def _x2(x: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    return x2 if x2.data_ptr() % 16 == 0 else x2.clone()


def layernorm(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm over the last dimension (``supported(x)``); ``y`` in ``x``'s dtype, fp32 statistics."""
    shape = x.shape
    x2 = _x2(x)
    if not _needs_grad(x, weight, bias):
        return _fwd(x2, _affine(weight, x2), _affine(bias, x2), float(eps), False, False)[0].reshape(shape)
    return _Norm.apply(x2, weight, bias, float(eps), False).reshape(shape)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension (``supported(x)``); weight optional."""
    shape = x.shape
    x2 = _x2(x)
    if not _needs_grad(x, weight):
        return _fwd(x2, _affine(weight, x2), _affine(None, x2), float(eps), False, True)[0].reshape(shape)
    return _Norm.apply(x2, weight, None, float(eps), True).reshape(shape)


# ---- skewed rows: bf16 rows of an odd width (267 / 651: a row is not a whole number of 16-byte vectors) -------------------------------

#: (tpr, ch, warps, ctas) launches of the skewed-row kernels (``MWLN_SKEW_LAYOUTS``), the first that holds the row: a row of n elements
#: spans up to ``ceil((n + 7) / 8)`` 16-byte vectors, the partial ones in the first / last chunk only (``skew::covers``). Measured on
#: the B200 (kernel durations, L2 flushed, [L^2, 267] / [L^2, 651], L 128-768; docs/b200/transition.md): 8 x 5 at 12 (forward) / 4
#: (backward) CTAs a SM for 267; 651 forward 8 x 11 at 6 (16 x 6 and 32 x 3 are 5-7 % behind from L 384), backward 32 x 3 at 6 (16 x 6
#: 12 % behind, 8 x 11 2x: its dw / db sums do not fit the registers).
#: 2026-11-17 (part 2): the kernels take fp32 rows too (4 elements a vector, 4 row phases) and the 451 / 831 / 833 token streams:
#: {element bytes: launches}, bf16 451 on 16 x 4, 831 / 833 on 32 x 4; fp32 267 on 8 x 9, 451 on 32 x 4, 651 on 32 x 6 (forward at 8 CTAs a SM with the
#: next-tile prefetch: [65536, 651] 77.7 -> 63.4 us), 831 / 833 on 32 x 7.
SKEW_FWD = {2: ((8, 5, 2, 12), (16, 4, 2, 8), (8, 11, 2, 6), (32, 4, 2, 8)), 4: ((8, 9, 2, 8), (32, 4, 2, 8), (32, 6, 2, 8), (32, 7, 2, 6))}
SKEW_BWD = {2: ((8, 5, 2, 4), (16, 4, 2, 4), (32, 3, 2, 6), (32, 4, 2, 4)), 4: ((8, 9, 2, 4), (32, 4, 2, 4), (32, 6, 2, 4), (32, 7, 2, 4))}

#: widths whose multi-tile skewed forward keeps the next tile's loads in flight (measured on the B200, inference, graph timing: 651 bf16
#: [65536 / 262144, 651] 46.8 / 145 -> 42.9 / 131 us, fp32 training [147456, 651] 379 -> 364; 267 at its best launch: no change)
SKEW_PF: set[tuple[int, int]] = {(651, 2), (651, 4)}

#: launch overrides {(n, element bytes, backward): (tpr, ch, warps, ctas)} (empty: :func:`skew_config`'s rule)
SKEW_TUNED: dict[tuple[int, int, bool], tuple[int, int, int, int]] = {}


@functools.lru_cache(maxsize=4096)
def skew_config(n: int, elem_bytes: int, backward: bool) -> tuple[int, int, int, int] | None:
    """(tpr, ch, warps, ctas) of the skewed-row kernels for bf16 / fp32 rows of ``n`` elements, or None (even width, or no launch of
    ``SKEW_FWD`` / ``SKEW_BWD`` holds the row)."""
    if n % 2 == 0 or elem_bytes not in SKEW_FWD:
        return None
    t = SKEW_TUNED.get((n, elem_bytes, backward))
    if t is not None:
        return t
    vec = 16 // elem_bytes
    for tpr, ch, warps, ctas in (SKEW_BWD if backward else SKEW_FWD)[elem_bytes]:
        if tpr * ch * vec >= n + vec - 1 and (n - 1) // vec >= (ch - 1) * tpr:
            return tpr, ch, warps, ctas
    return None


def skew_supported(x: torch.Tensor) -> bool:
    """Whether the skewed-row kernels take ``x``: CUDA bf16 / fp32 of an odd width with a forward and a backward launch."""
    n = x.shape[-1] if x.dim() else 0
    es = x.element_size()
    return (
        x.is_cuda and x.dtype in DTYPES and x.numel() > 0 and skew_config(n, es, False) is not None and skew_config(n, es, True) is not None
    )


def _skew_rows(x: torch.Tensor) -> torch.Tensor:
    """``x`` as contiguous, 16-byte aligned rows whose last 16-byte vector lies inside the storage (the kernels read whole vectors): a
    tensor that ends mid-vector at the very end of its storage is copied into a padded one."""
    x2 = _x2(x)
    es = x2.element_size()
    vec = 16 // es
    rest = x2.numel() % vec
    if rest and x2.untyped_storage().nbytes() - (x2.storage_offset() + x2.numel()) * es < (vec - rest) * es:
        buf = x2.new_empty(x2.numel() + vec - rest)
        buf[: x2.numel()] = x2.reshape(-1)
        x2 = buf[: x2.numel()].view(x2.shape)
    return x2


def _sfwd_fake(x, w, b, eps, save):
    m = x.shape[0] if save else 0
    f32 = dict(dtype=torch.float32, device=x.device)
    return torch.empty_like(x), torch.empty((m,), **f32), torch.empty((m,), **f32)


@opaque(fake=_sfwd_fake, name="layernorm_skew_sm100_fwd")
def _sfwd(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float, save: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward of contiguous bf16 / fp32 x [M, N], N odd: (y, mean, rstd); the statistics only when ``save``."""
    cfg = skew_config(x.shape[1], x.element_size(), False)
    assert cfg is not None
    y, mean, rstd = _ext().skew_fwd(x, w, b, float(eps), bool(save), *cfg, (x.shape[1], x.element_size()) in SKEW_PF)
    return y, mean, rstd


def _sbwd_fake(dy, x, w, b, mean, rstd, wgrad):
    return _bwd_fake(dy, x, w, b, mean, rstd, wgrad, False)


@opaque(fake=_sbwd_fake, name="layernorm_skew_sm100_bwd")
def _sbwd(
    dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, wgrad: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward from the saved statistics: (dx, dw, db); the affine gradients only when ``wgrad`` (``b`` names their dtype)."""
    cfg = skew_config(x.shape[1], x.element_size(), True)
    assert cfg is not None
    dx, dw, db = _ext().skew_bwd(dy, x, w, mean, rstd, bool(wgrad), *cfg, _grad_bf16(w, b), fuse_groups(x.shape[1], x.element_size(), x.shape[0]))
    return dx, dw, db


class _SkewNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, weight, bias, eps):
        w, b = _affine(weight, x2), _affine(bias, x2)
        y, mean, rstd = _sfwd(x2, w, b, eps, True)
        ctx.save_for_backward(x2, w, b, mean, rstd)
        ctx.cfg = (weight is not None, bias is not None, None if weight is None else weight.dtype, None if bias is None else bias.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        x2, w, b, mean, rstd = ctx.saved_tensors
        has_w, has_b, wdt, bdt = ctx.cfg
        wgrad = (has_w and ctx.needs_input_grad[1]) or (has_b and ctx.needs_input_grad[2])
        dy = _skew_rows(dy)
        dx, dw, db = _sbwd(dy, x2, w, b, mean, rstd, bool(wgrad))
        gw = dw.to(wdt) if has_w and ctx.needs_input_grad[1] else None
        gb = db.to(bdt) if has_b and ctx.needs_input_grad[2] else None
        return dx, gw, gb, None


def layernorm_skew(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm over the last dimension on the skewed-row kernels (``skew_supported(x)``); ``y`` in ``x``'s dtype, fp32 statistics."""
    shape = x.shape
    x2 = _skew_rows(x)
    if not _needs_grad(x, weight, bias):
        return _sfwd(x2, _affine(weight, x2), _affine(bias, x2), float(eps), False)[0].reshape(shape)
    return _SkewNorm.apply(x2, weight, bias, float(eps)).reshape(shape)


__all__ = ["layernorm", "layernorm_skew", "rmsnorm", "rows_config", "skew_config", "skew_supported", "supported"]
