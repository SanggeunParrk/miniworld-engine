"""m-major-specialized LayerNorm backward for the trimul LN_out (te_style) path.

Replaces te_style's atomic `_ln_bwd_kernel`. The LN_out backward operates on (M, N)
tensors that are **m-major** (features N have stride M; the M tokens are unit-stride /
contiguous) — dx_normed, x, and dx all share the SAME m-major strides (1, M).

Two lessons from the layernorm CUDA / persistent work, applied to THIS layout:
  1. ATOMIC-FREE dγ/dβ: te_style's per-CTA `atomic_add(DG/DB)` (one atomic per tile per
     feature -> ~cdiv(M,BLOCK_M1) atomics = L2 contention, the profiled bottleneck). Here a
     PERSISTENT grid of NUM_SM*WAVES blocks grid-strides over the M-tiles, carries dγ/dβ in
     fp32 REGISTERS across the whole stride loop, and writes exactly ONE partial row each ->
     partial buffer is [G, N] (~few hundred rows); a tiny final reduce sums it. No atomics.
  2. M-CONTIGUOUS VECTOR LOADS: m-major => M is unit-stride. `tl.max_contiguous/multiple_of`
     on the row index + a compile-time unit row-stride let triton emit wide (uint4) vector
     loads along M (many tokens per transaction), amortizing the strided-N (stride M) reads.

dx/dγ/dβ math + fp32 accumulation are IDENTICAL to te_style (bit-exact). For small M (few
tiles -> persistent grid underfilled + reduce-launch overhead dominates) the wrapper falls
back to the te_style atomic kernel, which is faster there.
"""
from __future__ import annotations
from miniworld_engine.autotune.configs import configs_for


import torch
import triton
import triton.language as tl


from miniworld_engine.autotune.shape_key import both_key, pack  # kernels here are level=both
from ...layernorm.triton.persistent import _ln_bwd_persistent as _ln_bwd_persistent_jit
from .te_style import _ln_bwd_kernel  # atomic small-M fallback
from miniworld_engine import settings

# Escape hatch / A-B control: force one path. "atomic" = the old te_style plain-atomic kernel.
def _override() -> str | None:
    """Read at call time; see layernorm.compile_native._ln_bwd_override."""
    return settings.current().layernorm_out_bwd_path

_WAVES = 2

# BLOCK_N used to arrive as next_pow2(K) from the launcher; it is the REDUCE axis (the c1/c2 row
# sums and the dγ/dβ column sums), so it is a CSV tile, and
# the previous single-column-tile schedule stays in the sweep. BLOCK_M1 takes the canonical 2-D row
# tile (the old list started at 4, below the 16 floor the grid module defines).


# fmt: off


# shape_key is keyed: the grid is FIXED at NP programs, so BLOCK_M1 alone sets how many M-tiles each
# program grid-strides over. The atomic fallback in this file already bucketed the row count.
# VEC_HINT is deliberately NOT keyed even though it gates a code path (the tl.max_contiguous vector
# hint): the sole launcher (`_ln_bwd_persistent_new`) passes `N=K` and `VEC_HINT=(K <= 128)`, so it
# is a pure function of N -- already in the key. Keying it too would only duplicate a partition the
# cache makes anyway.
@triton.autotune(configs=configs_for("layernorm_bwd_split_mmajor_triton"), key=['shape_key'])
@triton.jit
def _ln_bwd_mmajor_kernel(
    DX, PDG, PDB, DXn, X, G, Mean, Rstd,
    stride_part, sc,                       # sc = feature stride (= M); row stride is 1 (m-major)
    M, N: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, VEC_HINT: tl.constexpr, shape_key,
):
    # Grid is 2-D: axis 0 = the persistent programs that grid-stride over M-tiles (the axis
    # PDG/PDB are indexed by), axis 1 = the feature tile this program owns. The feature axis must
    # be a GRID axis and not an inner loop for the same reason as the canonical persistent kernel:
    # dγ/dβ are carried per FEATURE in fp32 registers across the whole stride loop (that register
    # accumulator, in place of atomics, is lesson 1 in this module's docstring) and it can only be
    # BLOCK_K wide. Each program then owns a disjoint slice of its [NP, N] partial row.
    pid = tl.program_id(0).to(tl.int64)
    NP = tl.num_programs(0)
    pid_n = tl.program_id(1).to(tl.int64)

    offs_n = pid_n * BLOCK_K + tl.arange(0, BLOCK_K)
    cmask = offs_n < N
    g = tl.load(G + offs_n, mask=cmask, other=0.0).to(tl.float32)

    acc_dg = tl.zeros([BLOCK_K], dtype=tl.float32)
    acc_db = tl.zeros([BLOCK_K], dtype=tl.float32)

    num_tiles = tl.cdiv(M, BLOCK_M1)
    for tile in range(pid, num_tiles, NP):
        rows = tile * BLOCK_M1 + tl.arange(0, BLOCK_M1)
        if VEC_HINT:
            rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_M1), BLOCK_M1)  # M unit-stride vec hint
        rmask = rows < M
        mean = tl.load(Mean + rows, mask=rmask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=rmask, other=0.0).to(tl.float32)

        # COVERING TILE (BLOCK_K >= N): the c1/c2 gather loop below re-reads DXn and X for the
        # WHOLE feature row, and the dx/dγ/dβ tail then reads this program's own feature tile
        # again. At a covering tile those are the SAME addresses (the launcher's grid is
        # `(NP, cdiv(K, BLOCK_K))`, so the feature axis is exactly ONE block wide and pid_n == 0),
        # but they are not CSE'd -- the gather is its own scf.for region and the DX tl.store lands
        # between the two, and Triton cannot prove the raw pointers do not alias. So the covering
        # config read DXn and X twice per M-tile. Both N and BLOCK_K are tl.constexpr, so the guard
        # is resolved at TRACE time and only ONE branch is emitted; the grid collapse that makes
        # pid_n provably 0 is the launcher's cdiv, not an assumption -- and offs_n stays pid_n-
        # relative, so a wider grid would mask everything off rather than compute a wrong row. The
        # fp32 register accumulators acc_dg/acc_db (lesson 1 in the module docstring: no atomics)
        # are fed exactly as in the feature-grid path, so the [NP, N] partial contract is unchanged.
        if BLOCK_K >= N:
            # m-major addressing: row stride == 1 (contiguous), feature stride == sc (== M)
            p = rows[:, None] + offs_n[None, :] * sc
            mask = rmask[:, None] & cmask[None, :]
            dxn = tl.load(DXn + p, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(X + p, mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
            dxhat = tl.where(mask, dxn * g[None, :], 0.0)
            c2 = tl.sum(dxhat, axis=1) / N
            c1 = tl.sum(dxhat * xhat, axis=1) / N
            dx = (dxhat - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
            tl.store(DX + p, dx, mask=mask)

            acc_dg += tl.sum(dxn * xhat, axis=0)
            acc_db += tl.sum(dxn, axis=0)
        else:
            # c1/c2 reduce over the WHOLE feature row, so they are gathered over every feature tile
            # before any dx element of this program's tile can be written.
            c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
            c2 = tl.zeros([BLOCK_M1], dtype=tl.float32)
            for n0 in range(0, N, BLOCK_K):
                cn = n0 + tl.arange(0, BLOCK_K)
                cm = cn < N
                m2 = rmask[:, None] & cm[None, :]
                q = rows[:, None] + cn[None, :] * sc
                dxn_r = tl.load(DXn + q, mask=m2, other=0.0).to(tl.float32)
                x_r = tl.load(X + q, mask=m2, other=0.0).to(tl.float32)
                g_r = tl.load(G + cn, mask=cm, other=0.0).to(tl.float32)
                xhat_r = tl.where(m2, (x_r - mean[:, None]) * rstd[:, None], 0.0)
                dxhat_r = tl.where(m2, dxn_r * g_r[None, :], 0.0)
                c2 += tl.sum(dxhat_r, axis=1)
                c1 += tl.sum(dxhat_r * xhat_r, axis=1)
            c1 = c1 / N
            c2 = c2 / N

            # m-major addressing: row stride == 1 (contiguous), feature stride == sc (== M)
            p = rows[:, None] + offs_n[None, :] * sc
            mask = rmask[:, None] & cmask[None, :]
            dxn = tl.load(DXn + p, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(X + p, mask=mask, other=0.0).to(tl.float32)
            xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
            dxhat = tl.where(mask, dxn * g[None, :], 0.0)
            dx = (dxhat - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
            tl.store(DX + p, dx, mask=mask)

            acc_dg += tl.sum(dxn * xhat, axis=0)
            acc_db += tl.sum(dxn, axis=0)

    part = pid * stride_part + offs_n
    tl.store(PDG + part, acc_dg, mask=cmask)
    tl.store(PDB + part, acc_db, mask=cmask)
# fmt: on


def _persistent_grid(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count * _WAVES


# Persistent wins once there are enough M-tiles to fill the grid AND amortize the reduce;
# below this, the te_style atomic kernel is faster (measured). M threshold, not tile count,
# keeps the decision graph-capture-safe (no host-side tile math per call).
_PERSIST_MIN_M = 300_000


def _ln_bwd_atomic(dxn, x, gamma, mean, rstd, dx_strides, *, shape_key: int | None = None):
    """Exact te_style atomic path (small-M fallback).

    ``shape_key`` is ``both_key(L)`` from the caller: x is the flattened (M, K) matrix here and M
    alone cannot say which L produced it. None -> smallest bucket (bench/driver entry only)."""
    M, K = x.shape
    dx = torch.empty_strided((M, K), dx_strides, device=x.device, dtype=dxn.dtype)
    dg = torch.zeros(K, dtype=torch.float32, device=x.device)
    db = torch.zeros(K, dtype=torch.float32, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]),)  # noqa: E731
    _ln_bwd_kernel[grid](
        dxn, x, gamma, mean, rstd, dx, dg, db, M, K,
        dxn.stride(0), dxn.stride(1), x.stride(0), x.stride(1),
        dx.stride(0), dx.stride(1),
        N_PAD=triton.next_power_of_2(K),
        shape_key=both_key(0, N=K) if shape_key is None else pack(shape_key, N=K),
    )
    return dx, dg, db


def _ln_bwd_persistent_new(dxn, x, gamma, mean, rstd, dx_strides, *,
                           shape_key: int | None = None):
    """Atomic-free persistent m-major path via THIS module's specialized kernel. Fastest at
    narrow N (N=128: the M-contiguous vector hint + compile-time unit row-stride let triton widen
    the load — 1.20x vs te_style atomic at L=1024). VEC_HINT is neutral at wide N.

    ``shape_key`` is ``both_key(L)`` from the caller (see ``_ln_bwd_atomic``)."""
    M, K = x.shape
    dx = torch.empty_strided((M, K), dx_strides, device=x.device, dtype=dxn.dtype)
    NP = _persistent_grid(x.device)
    pdg = torch.empty((NP, K), dtype=torch.float32, device=x.device)
    pdb = torch.empty((NP, K), dtype=torch.float32, device=x.device)
    # grid axis 1 = feature tiles (see the kernel note); the partial buffer stays [NP, K].
    grid = lambda meta: (NP, triton.cdiv(K, meta["BLOCK_K"]))  # noqa: E731
    _ln_bwd_mmajor_kernel[grid](
        dx, pdg, pdb, dxn, x, gamma, mean, rstd,
        pdg.stride(0), x.stride(1),      # feature stride (= M); row stride assumed 1
        M, N=K, VEC_HINT=(K <= 128),
        shape_key=both_key(0, N=K) if shape_key is None else pack(shape_key, N=K),
    )
    return dx, pdg.sum(0), pdb.sum(0)


def _ln_bwd_persistent_canonical(dxn, x, gamma, mean, rstd, dx_strides, *,
                                 shape_key: int | None = None):
    """Atomic-free persistent m-major path via the canonical layernorm persistent kernel
    (stride-generic; both operands share m-major strides). Fastest at wide N (N=256: 1.21x vs
    te_style atomic at L=1024).

    ``shape_key`` is ``both_key(L)`` from the caller (see ``_ln_bwd_atomic``) and IS forwarded.
    This docstring used to say ``_ln_bwd_persistent`` was not autotuned on it, so it was dropped;
    that stopped being true when the shape-key unification added ``shape_key`` to that kernel's
    signature and ``key=['N', 'shape_key']``. The two callers inside persistent.py were updated
    and this one, in another file, was not -- so every launch down this path raised
    ``dynamic_func() missing 1 required positional argument: 'shape_key'``. It is the wide-N
    large-M branch, so the trimul backward died for any L with L*L >= 300_000 (L >= 548) at
    d_hidden > 128."""
    M, K = x.shape
    dx = torch.empty_strided((M, K), dx_strides, device=x.device, dtype=dxn.dtype)
    NP = _persistent_grid(x.device)
    pdw = torch.empty((NP, K), dtype=torch.float32, device=x.device)
    pdb = torch.empty((NP, K), dtype=torch.float32, device=x.device)
    grid = lambda meta: (NP, triton.cdiv(K, meta["BLOCK_K"]))  # noqa: E731
    _ln_bwd_persistent_jit[grid](
        dx, pdw, pdb, dxn, x, gamma, mean, rstd,
        pdw.stride(0), x.stride(0), x.stride(1),
        M, N=K,
        shape_key=both_key(0, N=K) if shape_key is None else pack(shape_key, N=K),
    )
    return dx, pdw.sum(0), pdb.sum(0)


def ln_bwd_mmajor(dxn, x, gamma, mean, rstd, dx_strides, *, shape_key: int | None = None):
    """Drop-in for te_style `_ln_bwd`: dx (m-major, at dx_strides) + dγ + dβ (fp32 (N,)).

    Size-adaptive (all bit-exact vs te_style atomic):
      • small M  -> te_style atomic kernel (persistent grid underfilled + reduce launch dominates).
      • large M, N<=128 -> this module's specialized kernel (M-contiguous vector hint wins).
      • large M, N>128  -> canonical layernorm persistent kernel (wins at wide N).
    Requires m-major inputs (row stride 1); falls back to atomic otherwise. Env
    settings.layernorm_out_bwd_path forces one path (A-B / debug).

    ``shape_key`` is ``both_key(L)`` computed by the caller (te_style ``_ln_bwd``) and passed
    through unchanged -- the dispatch below picks a kernel, never a key."""
    M, K = x.shape
    if _override() == "atomic" or x.stride(0) != 1 or M < _PERSIST_MIN_M:
        return _ln_bwd_atomic(dxn, x, gamma, mean, rstd, dx_strides, shape_key=shape_key)
    if _override() == "canonical":
        return _ln_bwd_persistent_canonical(dxn, x, gamma, mean, rstd, dx_strides,
                                            shape_key=shape_key)
    if _override() == "persistent" or K <= 128:
        return _ln_bwd_persistent_new(dxn, x, gamma, mean, rstd, dx_strides, shape_key=shape_key)
    return _ln_bwd_persistent_canonical(dxn, x, gamma, mean, rstd, dx_strides,
                                        shape_key=shape_key)
