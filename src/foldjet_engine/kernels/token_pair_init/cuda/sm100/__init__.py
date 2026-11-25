"""sm_100a build and launch of the token-pair initialisation kernels (``token_pair_init.cu``).

The cubin is built on first use by the newest nvcc here that knows sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``)
and cached under ``settings.engine_jit_root``; launches go through the CUDA driver on torch's current stream (CUDA-graph
capturable), the way the ``swa_dit`` sm_100a stages do. Plain CUDA-core kernels -- no tensor cores: the op is bandwidth bound.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent
P = 128
NWARP = 8
W_WARPS = 16  # warps per CTA of tpi_fwd_w (TPI_W_WARPS in token_pair_init.cu)
BC = 32  # positions per unit of the backward (TPI_BC)
BSW = 4  # warps per stream of the backward (4 / TPI_BCL: a lane holds one channel)
BST = 3  # streams per CTA of the backward at most (TPI_BST)
_SMEM_MAX = 227 * 1024  # shared memory a CTA can ask for on sm_100
_plan_force: tuple[int, int] | None = None  # (grid, streams per CTA) instead of _bwd_plan's choice: measurements only


def table_rows(r_max: int, s_max: int) -> int:
    """Rows of a backward stream's class table: the merged residue / token class (2 (2 r_max + 2) - 1) and chain class x bond (2 (2 s_max + 2))."""
    return 2 * (2 * r_max + 2) - 1 + 2 * (2 * s_max + 2)


def fwd_rows(L: int) -> int:
    """Rows of z per forward tile (one register copy of right[j] serves them): 2, 1 where the tile's packed words would not fit the
    shared memory. Measured (CUDA graph, B200): rows 1 / 2 / 4 x 2 / 4 / 8 positions in flight x 4 / 8 CTAs a SM at B16 L128-384, B48
    L128-256, B5 L768, B1 L2048: 2 rows, 4 positions, 8 CTAs is the best or within 3% of it everywhere."""
    return 2 if 2 * L * 4 <= 200 * 1024 else 1


def check_packable(r_max: int, s_max: int) -> None:
    """The kernels pack the three relative-position bins into 8 bits each: the largest is bin3 = 4 r_max + 2 s_max + 5 = n_rel - 2."""
    if n_rel(r_max, s_max) - 2 > 255:
        raise ValueError(f"token_pair_init packs its bins into 8 bits: r_max={r_max}, s_max={s_max} needs more")


def n_rel(r_max: int, s_max: int) -> int:
    """Width of the relative-position one-hot: 2 (2 r_max + 2) + (2 s_max + 2) + 1 (139 for r_max 32, s_max 2)."""
    return 2 * (2 * r_max + 2) + (2 * s_max + 2) + 1


@functools.lru_cache(maxsize=None)
def cubin(rf: int = 2) -> str:
    """Path of the cubin (``rf``: rows of z per tile of ``tpi_fwd``), built on first use; rebuilt only when the source, a flag or the toolchain changes."""
    from foldjet_engine.kernels import _nvcc

    flags = ("-std=c++17", "-O3", "-arch=sm_100a", "-cubin", "-lineinfo", f"-DTPI_RF={rf}")
    return str(_nvcc.cubin(f"token_pair_init_sm100/token_pair_init_rf{rf}", flags, source=_dir / "token_pair_init.cu", toolchain="sm100"))


@functools.lru_cache(maxsize=16)
def _kernel_f(index: int, L: int):
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    with torch.cuda.device(index):
        return driver.Kernel(cubin(fwd_rows(L)), "tpi_fwd", fwd_rows(L) * L * 4)


@functools.lru_cache(maxsize=16)
def _kernel_w(index: int, r_max: int, s_max: int):
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    with torch.cuda.device(index):
        return driver.Kernel(cubin(), "tpi_fwd_w", fwd_w_smem(r_max, s_max))


def fwd_w_smem(r_max: int, s_max: int) -> int:
    """Shared memory of ``tpi_fwd_w``: the merged residue x token rows (2 (2 r_max + 2) - 1) and the combined rows (4 (2 s_max + 2)), 128 fp32
    each, then the raw weights the bulk copies bring (w_rel [128][n_rel], w_bond [128][2]) and their mbarrier."""
    nr, ns = 2 * r_max + 2, 2 * s_max + 2
    return ((2 * nr - 1) + 4 * ns + n_rel(r_max, s_max) + 2) * P * 4 + 16


def fwd_w_pipe(B: int, L: int, sms: int) -> bool:
    """``tpi_fwd_w``'s branch-free item path: where the grid has at least two warps per work item (a warp's one item is latency, not bandwidth:
    2026-11-17, a launch back to back, us: L128 8.8 -> 6.9 with it; L256 10.7 -> 11.0 and L384 17.8 -> 18.4 without it is better)."""
    return 2 * B * ((L + 1) // 2) * ((L + 15) // 16) <= sms * W_WARPS


def forward_w(left, right, w_rel, w_bond, ids, bond, r_max: int, s_max: int) -> torch.Tensor:
    """Forward only, ONE launch: z [B, L, L, 128] fp32 from left / right [B, L, 128] (both fp32 or both bf16, read as they are), the weights w_rel [128, n_rel] / w_bond [128, 2] fp32 as they
    are (the table rows are built in the kernel from a bulk copy of them), the five id tensors [B * L] int32 or int64 (asym, residue, token, entity, sym) and bond [B*L*L] uint8.
    The z of :func:`forward` over the prep kernel's ids and table up to the order of the table rows' sum (the residue / token rows and the chain /
    entity / bond rows are added to each other first: ``token_pair_init.cu``)."""
    B, L, _ = left.shape
    check_packable(r_max, s_max)
    assert left.dtype == right.dtype and left.dtype in (torch.float32, torch.bfloat16), (left.dtype, right.dtype)
    assert B * ((L + 1) // 2) * ((L + 15) // 16) < 2**31, (B, L)  # the kernel's work items (row pair, 16 positions) in 32-bit arithmetic
    from foldjet_engine.kernels._device import num_sms

    k = _kernel_w(_dev(left), r_max, s_max)
    sms = num_sms(_dev(left))
    # the kernel bulk-copies the weights as they are: contiguous, 16-byte aligned (a parameter is; anything else is copied once here)
    w_rel, w_bond = (w if w.is_contiguous() and w.data_ptr() % 16 == 0 else w.contiguous().clone() for w in (w_rel, w_bond))
    out = torch.empty(B, L, L, P, device=left.device, dtype=torch.float32)
    wide = sum(1 << q for q, t in enumerate(ids) if t.dtype == torch.int64)
    with torch.cuda.device(left.device):
        k(
            (sms, 1, 1),
            (32 * W_WARPS, 1, 1),
            left,
            right,
            w_rel,
            w_bond,
            *ids,
            int(wide),
            bond,
            out,
            int(B),
            int(L),
            int(r_max),
            int(s_max),
            int(left.dtype == torch.bfloat16),
            int(fwd_w_pipe(B, L, sms)),
        )
    return out


def _dev(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def forward(left, right, tbl, ids, bond, r_max: int, s_max: int) -> torch.Tensor:
    """z [B, L, L, 128] fp32 from left / right [B, L, 128], the table [nbin, 128], ids [5, B*L] int32, bond [B*L*L] uint8."""
    B, L, _ = left.shape
    check_packable(r_max, s_max)
    fwd = _kernel_f(_dev(left), L)
    _kernel_bw(_dev(left), r_max, s_max, "tpi_bwd")  # the backward of this forward: loaded here, not on autograd's device thread
    out = torch.empty(B, L, L, P, device=left.device, dtype=torch.float32)
    tiles = B * -(-L // fwd_rows(L))
    from foldjet_engine.kernels._device import num_sms

    with torch.cuda.device(left.device):
        fwd(
            (min(tiles, 8 * num_sms(_dev(left))), 1, 1),
            (32 * NWARP, 1, 1),
            left,
            right,
            tbl,
            ids,
            bond,
            out,
            int(B),
            int(L),
            int(r_max),
            int(s_max),
        )
    return out


def _bwd_plan(B: int, L: int, r_max: int, s_max: int, sms: int) -> tuple[int, int, int]:
    """``(grid, streams per CTA, K)`` of a ``tpi_bwd`` launch: one CTA per SM, as many streams (private class tables, ``table_rows`` x 128
    fp32 each) as its shared memory holds, at most BST; fewer where the call has fewer units (a unit: BC positions of one row) than
    streams. ``K``: the column chunks a stream's run of units can touch (the size of its column-sum slots in ``cpart``)."""
    units = B * -(-L // BC) * L
    assert units < 2**31, (B, L)
    nst = max(1, min(BST, _SMEM_MAX // (table_rows(r_max, s_max) * P * 4)))
    grid = sms
    if _plan_force is not None:
        grid, nst = _plan_force
    while grid * nst > units and nst > 1:
        nst -= 1
    grid = min(grid, units)
    per = -(-units // (grid * nst))
    return grid, nst, -(-per // L) + 1


def _bwd_launch(g, k, red, id_args, bond, r_max: int, s_max: int, dleft_dtype, dright_dtype):
    """The two launches of the backward: the streams (``k``: tpi_bwd / tpi_bwd_w with its id arguments) and the fixed-order reduce."""
    from foldjet_engine.kernels._device import num_sms

    B, L = g.shape[:2]
    nbin = n_rel(r_max, s_max) + 2
    grid, nst, K = _bwd_plan(B, L, r_max, s_max, num_sms(_dev(g)))
    units = B * -(-L // BC) * L
    f32 = torch.float32
    rpart = torch.empty(units * P, device=g.device, dtype=f32)  # the row sum of every unit
    cpart = torch.empty(grid * nst * K * BC * P, device=g.device, dtype=f32)  # the column sums of every (stream, column chunk)
    wpart = torch.empty(grid * nbin * P, device=g.device, dtype=f32)  # the class-row sums of every CTA
    dleft = torch.empty(B, L, P, device=g.device, dtype=dleft_dtype)
    dright = torch.empty(B, L, P, device=g.device, dtype=dright_dtype)
    dtbl = torch.empty(nbin, P, device=g.device, dtype=f32)
    with torch.cuda.device(g.device):
        k((grid, 1, 1), (32 * BSW * nst, 1, 1), g, *id_args, bond, rpart, cpart, wpart, int(B), int(L), int(r_max), int(s_max), int(nst), int(K))
        red(
            (2 * B * L + nbin, 1, 1),
            (P, 1, 1),
            rpart,
            cpart,
            wpart,
            dleft,
            dright,
            dtbl,
            int(B),
            int(L),
            int(nbin),
            int(grid),
            int(nst),
            int(K),
            int(dleft_dtype == torch.bfloat16),
            int(dright_dtype == torch.bfloat16),
        )
    return dleft, dright, dtbl


@functools.lru_cache(maxsize=16)
def _kernel_bw(index: int, r_max: int, s_max: int, name: str):
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    with torch.cuda.device(index):
        smem = max(1, min(BST, _SMEM_MAX // (table_rows(r_max, s_max) * P * 4))) * table_rows(r_max, s_max) * P * 4
        return driver.Kernel(cubin(), name, smem), driver.Kernel(cubin(), "tpi_bwd_red", 0)


def backward(g, ids, bond, nbin: int, r_max: int, s_max: int):
    """(dleft [B, L, 128], dright [B, L, 128], dtbl [nbin, 128]) of :func:`forward` from dz = g [B, L, L, 128] fp32 (``nbin`` = table rows = n_rel + 2:
    the H100 launcher's signature; checked against r_max / s_max). Two launches (``tpi_bwd`` + ``tpi_bwd_red``), bitwise repeatable."""
    check_packable(r_max, s_max)
    if nbin != n_rel(r_max, s_max) + 2:
        raise ValueError(f"token_pair_init: {nbin} table rows, r_max={r_max} / s_max={s_max} make {n_rel(r_max, s_max) + 2}")
    k, red = _kernel_bw(_dev(g), r_max, s_max, "tpi_bwd")
    return _bwd_launch(g, k, red, (ids,), bond, r_max, s_max, torch.float32, torch.float32)


def backward_w_ready(left, r_max: int, s_max: int) -> None:
    """Load ``backward_w``'s kernels for (device, r_max, s_max) now, on the forward's thread: the backward runs on autograd's device thread,
    where no CUDA driver context is current for a module load (cuModuleLoadData: CUDA_ERROR_INVALID_CONTEXT)."""
    _kernel_bw(_dev(left), r_max, s_max, "tpi_bwd_w")


def backward_w(g, ids, bond, r_max: int, s_max: int, dleft_dtype: torch.dtype, dright_dtype: torch.dtype):
    """The backward of :func:`forward_w` on the caller's five id tensors as they are ([B * L] int32 or int64; no ids prep), two launches
    (``tpi_bwd_w`` + ``tpi_bwd_red``; nothing zero-filled): ``(dleft, dright [B, L, 128] in their dtypes (fp32 or bf16: the fp32 sums rounded
    once), dtbl [nbin, 128] fp32 = [dW_rel^T ; dW_bond^T])`` from dz = g [B, L, L, 128] fp32. Bitwise repeatable (no atomics: private class
    tables and partial sums added in a fixed order)."""
    check_packable(r_max, s_max)
    assert dleft_dtype in (torch.float32, torch.bfloat16) and dright_dtype in (torch.float32, torch.bfloat16), (dleft_dtype, dright_dtype)
    k, red = _kernel_bw(_dev(g), r_max, s_max, "tpi_bwd_w")
    wide = sum(1 << q for q, t in enumerate(ids) if t.dtype == torch.int64)
    return _bwd_launch(g, k, red, (*ids, int(wide)), bond, r_max, s_max, dleft_dtype, dright_dtype)
