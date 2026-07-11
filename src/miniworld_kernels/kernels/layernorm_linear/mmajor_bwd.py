"""m-major-specialized LayerNorm backward for the trimul LN_out (te_style) path.

Replaces te_style's atomic `_ln_bwd_kernel`. The LN_out backward operates on (M, N)
tensors that are **m-major** (features N have stride M; the M tokens are unit-stride /
contiguous) — dx_normed, x, and dx all share the SAME m-major strides (1, M).

Two lessons from the layernorm CUDA / persistent work, applied to THIS layout:
  1. ATOMIC-FREE dγ/dβ: te_style's per-CTA `atomic_add(DG/DB)` (one atomic per tile per
     feature -> ~cdiv(M,BLOCK_M) atomics = L2 contention, the profiled bottleneck). Here a
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

import os

import torch
import triton
import triton.language as tl

from ..layernorm.triton.main import get_seq_group
from ..layernorm.triton.persistent import _ln_bwd_persistent as _ln_bwd_persistent_jit
from .te_style import _ln_bwd_kernel  # atomic small-M fallback

# Escape hatch / A-B control: force one path. "atomic" = the old te_style plain-atomic kernel.
_OVERRIDE = (os.environ.get("MINIWORLD_LNOUT_BWD") or "").strip().lower() or None

_WAVES = 2

_configs = [
    triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns)
    for bm in (4, 8, 16, 32, 64)
    for nw in (4, 8, 16)
    for ns in (1, 2, 3)
]


# fmt: off
@triton.autotune(configs=_configs, key=["N"])
@triton.jit
def _ln_bwd_mmajor_kernel(
    DX, PDG, PDB, DXn, X, G, Mean, Rstd,
    stride_part, sc,                       # sc = feature stride (= M); row stride is 1 (m-major)
    M, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, VEC_HINT: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NP = tl.num_programs(0)

    offs_n = tl.arange(0, BLOCK_N)
    cmask = offs_n < N
    g = tl.load(G + offs_n, mask=cmask, other=0.0).to(tl.float32)

    acc_dg = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_db = tl.zeros([BLOCK_N], dtype=tl.float32)

    num_tiles = tl.cdiv(M, BLOCK_M)
    for tile in range(pid, num_tiles, NP):
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        if VEC_HINT:
            rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_M), BLOCK_M)  # M unit-stride vec hint
        rmask = rows < M
        # m-major addressing: row stride == 1 (contiguous), feature stride == sc (== M)
        p = rows[:, None] + offs_n[None, :] * sc
        mask = rmask[:, None] & cmask[None, :]

        dxn = tl.load(DXn + p, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + p, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean + rows, mask=rmask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=rmask, other=0.0).to(tl.float32)

        xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        dxhat = tl.where(mask, dxn * g[None, :], 0.0)
        c2 = tl.sum(dxhat, axis=1) / N
        c1 = tl.sum(dxhat * xhat, axis=1) / N
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


def _ln_bwd_atomic(dxn, x, gamma, mean, rstd, dx_strides):
    """Exact te_style atomic path (small-M fallback)."""
    M, K = x.shape
    dx = torch.empty_strided((M, K), dx_strides, device=x.device, dtype=dxn.dtype)
    dg = torch.zeros(K, dtype=torch.float32, device=x.device)
    db = torch.zeros(K, dtype=torch.float32, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)  # noqa: E731
    _ln_bwd_kernel[grid](
        dxn, x, gamma, mean, rstd, dx, dg, db, M, K,
        dxn.stride(0), dxn.stride(1), x.stride(0), x.stride(1),
        dx.stride(0), dx.stride(1),
        BLOCK_N=triton.next_power_of_2(K), GROUP_M=get_seq_group(M), DT=x.element_size(),
    )
    return dx, dg, db


def _ln_bwd_persistent_new(dxn, x, gamma, mean, rstd, dx_strides):
    """Atomic-free persistent m-major path via THIS module's specialized kernel. Fastest at
    narrow N (N=128: the M-contiguous vector hint + compile-time unit row-stride let triton widen
    the load — 1.20x vs te_style atomic at L=1024). VEC_HINT is neutral at wide N."""
    M, K = x.shape
    dx = torch.empty_strided((M, K), dx_strides, device=x.device, dtype=dxn.dtype)
    NP = _persistent_grid(x.device)
    pdg = torch.empty((NP, K), dtype=torch.float32, device=x.device)
    pdb = torch.empty((NP, K), dtype=torch.float32, device=x.device)
    _ln_bwd_mmajor_kernel[(NP,)](
        dx, pdg, pdb, dxn, x, gamma, mean, rstd,
        pdg.stride(0), x.stride(1),      # feature stride (= M); row stride assumed 1
        M, N=K, BLOCK_N=triton.next_power_of_2(K), VEC_HINT=(K <= 128),
    )
    return dx, pdg.sum(0), pdb.sum(0)


def _ln_bwd_persistent_canonical(dxn, x, gamma, mean, rstd, dx_strides):
    """Atomic-free persistent m-major path via the canonical layernorm persistent kernel
    (stride-generic; both operands share m-major strides). Fastest at wide N (N=256: 1.21x vs
    te_style atomic at L=1024)."""
    M, K = x.shape
    dx = torch.empty_strided((M, K), dx_strides, device=x.device, dtype=dxn.dtype)
    NP = _persistent_grid(x.device)
    pdw = torch.empty((NP, K), dtype=torch.float32, device=x.device)
    pdb = torch.empty((NP, K), dtype=torch.float32, device=x.device)
    _ln_bwd_persistent_jit[(NP,)](
        dx, pdw, pdb, dxn, x, gamma, mean, rstd,
        pdw.stride(0), x.stride(0), x.stride(1),
        M, N=K, BLOCK_N=triton.next_power_of_2(K),
    )
    return dx, pdw.sum(0), pdb.sum(0)


def ln_bwd_mmajor(dxn, x, gamma, mean, rstd, dx_strides):
    """Drop-in for te_style `_ln_bwd`: dx (m-major, at dx_strides) + dγ + dβ (fp32 (N,)).

    Size-adaptive (all bit-exact vs te_style atomic):
      • small M  -> te_style atomic kernel (persistent grid underfilled + reduce launch dominates).
      • large M, N<=128 -> this module's specialized kernel (M-contiguous vector hint wins).
      • large M, N>128  -> canonical layernorm persistent kernel (wins at wide N).
    Requires m-major inputs (row stride 1); falls back to atomic otherwise. Env
    MINIWORLD_LNOUT_BWD=atomic|persistent|canonical forces one path (A-B / debug)."""
    M, K = x.shape
    if _OVERRIDE == "atomic" or x.stride(0) != 1 or M < _PERSIST_MIN_M:
        return _ln_bwd_atomic(dxn, x, gamma, mean, rstd, dx_strides)
    if _OVERRIDE == "canonical":
        return _ln_bwd_persistent_canonical(dxn, x, gamma, mean, rstd, dx_strides)
    if _OVERRIDE == "persistent" or K <= 128:
        return _ln_bwd_persistent_new(dxn, x, gamma, mean, rstd, dx_strides)
    return _ln_bwd_persistent_canonical(dxn, x, gamma, mean, rstd, dx_strides)
