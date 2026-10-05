"""A100 (sm_80) adaptive LayerNorm: ``y = sigmoid(scale) LN(x) + bias`` with ``[scale | bias] = (LN(cond) w) [Ws | Wb]^T + [sb | 0]`` -- row passes in hand CUDA
(``sm80/adaln_rows.cuh``), the GEMMs between them cuBLAS (``integrations/adaln_sm80.py`` composes them).

    cond_ln      aff = LN(cond) w                          [P, dc], and the (mean, rstd) of every cond row
    epilogue     y = sigmoid(S + sb) LN(x) + B             row r of x reads row r % period of the [period, 2 d] GEMM output S | B
    bwd_x        D = [dscale | dy], dx, the column sums of dscale
    cond_bwd     dcond = LN-backward(dcond_aff), the column sums of dcond_aff cond_hat

Rows are fp32 or bf16 (statistics and sums always fp32), widths 128 / 384 / 768.  Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

import torch

from ..._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"

#: row widths of the row kernels, and the rows one block takes (``adaln_rows.cuh``)
ROWS_PER_BLOCK = {128: 8, 384: 2, 768: 1}
WIDTHS = tuple(ROWS_PER_BLOCK)


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    # MINIWORLD_ADALN_SM80_FLAGS: extra -D / nvcc flags for A/B experiments (their own build)
    extra = os.environ.get("MINIWORLD_ADALN_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"adaln_sm80{tag}",
        sources=[str(_dir / "ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                           "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


def available() -> bool:
    """The extension builds (a failure is the caller's to report once)."""
    _ext()
    return True


@functools.lru_cache(maxsize=8)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def blocks_for(rows: int, width: int, device: torch.device) -> int:
    """Blocks of a column-sum kernel over ``rows`` rows of ``width`` columns: one per row chunk, at most 8 per SM (each leaves one row of the partial buffer)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return max(1, min(-(-rows // ROWS_PER_BLOCK[width]), 8 * _sm_count(index)))


#: the warp-per-row kernels (``adaln_rows.cuh``): warps (rows) per block, and blocks per SM of the column-sum kernels (each block leaves one row of the partial buffer)
WARPS_PER_BLOCK = 4
WARP_BLOCKS_PER_SM = int(os.environ.get("MINIWORLD_ADALN_WARP_BLOCKS", "4"))


def warp_blocks_for(rows: int, device: torch.device) -> int:
    """Blocks of a warp-per-row column-sum kernel over ``rows`` rows."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return max(1, min(-(-rows // WARPS_PER_BLOCK), WARP_BLOCKS_PER_SM * _sm_count(index)))


def cond_ln(cond: torch.Tensor, lnw: torch.Tensor, out_dtype: torch.dtype, eps: float, stats: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``aff = LN(cond) lnw`` as [P, dc] ``out_dtype`` (cond [P, dc] fp32 or bf16 with unit column stride, lnw [dc] fp32); with ``stats`` also (mean, rstd) [P, 2] fp32."""
    p, dc = cond.shape
    aff = torch.empty(p, dc, device=cond.device, dtype=out_dtype)
    st = torch.empty(p, 2, device=cond.device, dtype=torch.float32) if stats else None
    _ext().cond_ln(cond, lnw, aff, st, float(eps))
    return aff, st


def epilogue(x: torch.Tensor, sbt: torch.Tensor, sb: torch.Tensor, eps: float, stats: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``y = sigmoid(S + sb) LN(x) + B`` for x [M, d], sbt = [S | B] [period, 2 d] (row r of x reads row r % period), sb [d] in x's dtype; with ``stats`` also
    the (mean, rstd) of every x row [M, 2] fp32."""
    m, d = x.shape
    y = torch.empty(m, d, device=x.device, dtype=x.dtype)
    st = torch.empty(m, 2, device=x.device, dtype=torch.float32) if stats else None
    _ext().adaln_epi(x, sbt, sb, y, st, sbt.shape[0], float(eps))
    return y, st


def bwd_x(dy: torch.Tensor, x: torch.Tensor, xst: torch.Tensor, sbt: torch.Tensor, sb: torch.Tensor, dres: torch.Tensor | None,
          operand_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(D, dx, psb)``: D [M, 2 d] ``operand_dtype`` = [dscale | dy], dx [M, d] = the LayerNorm backward of dy sigmoid(S + sb) (+ ``dres``), psb [blocks, d] fp32 =
    the column sums of dscale (add the rows for d sb).  dy / x contiguous [M, d] of one dtype."""
    m, d = x.shape
    dm = torch.empty(m, 2 * d, device=x.device, dtype=operand_dtype)
    dx = torch.empty(m, d, device=x.device, dtype=x.dtype)
    psb = torch.empty(warp_blocks_for(m, x.device), d, device=x.device, dtype=torch.float32)
    _ext().adaln_bwd_x(dy, x, xst, sbt, sb, dres, dm, dx, psb, sbt.shape[0])
    return dm, dx, psb


def cond_bwd(dca: torch.Tensor, cond: torch.Tensor, cst: torch.Tensor, lnw: torch.Tensor,
             dextra: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(dcond, pw)``: dcond [M, dc] like cond = the LayerNorm backward of dca [M, dc] fp32 (the gradient of ``LN(cond) lnw``) (+ ``dextra``), pw [blocks, dc]
    fp32 = the column sums of dca cond_hat (add the rows for d lnw)."""
    m, dc = cond.shape
    dcond = torch.empty(m, dc, device=cond.device, dtype=cond.dtype)
    pw = torch.empty(warp_blocks_for(m, cond.device), dc, device=cond.device, dtype=torch.float32)
    _ext().cond_ln_bwd(dca, cond, cst, lnw, dextra, dcond, pw)
    return dcond, pw


# ------------------------------------------------------------------------------------------------------------ atom width (128), bf16
ATOM_WIDTH = 128
ATOM_FWD_PREFETCH_CFG, ATOM_FWD_PREFETCH_ROWS, ATOM_FWD_SMALL_ROWS = 2, 24576, 8192      # the forward's config 2 (8 warps x 1 CTA / SM + the cp.async prefetch of the next tile) from this many rows on, and below the small count


def atom_supported(x: torch.Tensor, cond: torch.Tensor) -> bool:
    """The fused atom kernels' contract: bf16 rows, d_hidden = d_cond = 128."""
    return x.dtype is torch.bfloat16 and cond.dtype is torch.bfloat16 and x.shape[-1] == ATOM_WIDTH and cond.shape[-1] == ATOM_WIDTH and os.environ.get("MINIWORLD_ADALN_ATOM", "1") != "0"


def atom_fwd(x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, ws: torch.Tensor, wb: torch.Tensor, sb: torch.Tensor, eps_x: float, eps_c: float,
             stats: bool = False, cfg: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """The whole forward in one kernel (``adaln_atom_fwd.cuh``): ``(y, xst, cst)`` for x [M, 128], cond [P, 128] bf16 (row r reads cond row r % P), lnw [128] fp32, Ws / Wb [128, 128]
    and sb [128] bf16; with ``stats`` the (mean, rstd) of every x row and every cond row (P == M) [M, 2] fp32 for the backward."""
    m = x.shape[0]
    y = torch.empty_like(x)
    xst = torch.empty(m, 2, device=x.device, dtype=torch.float32) if stats else None
    cst = torch.empty(cond.shape[0], 2, device=x.device, dtype=torch.float32) if stats else None
    if cfg is None:
        cfg = int(os.environ.get("MINIWORLD_ADALN_ATOM_CFG", "-1"))
    if cfg < 0:
        cfg = ATOM_FWD_PREFETCH_CFG if (m >= ATOM_FWD_PREFETCH_ROWS or m < ATOM_FWD_SMALL_ROWS) else 0      # a warp with a second tile in sight prefetches it; at 8192 .. 24575 rows one tile a warp: two CTAs of 8 warps hide the latency better; at 5120 rows (the registry's N = 1024, A = 5) the prefetch kernel is 12.9 against 13.7 us, level with Triton's 12.9
    _ext().adaln_atom_fwd(x, cond, lnw, ws, wb, sb, y, xst, cst, float(eps_x), float(eps_c), cfg)
    return y, xst, cst


def atom_tf32_supported(x: torch.Tensor, cond: torch.Tensor) -> bool:
    """The fp32 (TF32) fused forward's contract: fp32 rows, d_hidden = d_cond = 128 (inference only: nothing is saved for a backward)."""
    return (x.dtype is torch.float32 and cond.dtype is torch.float32 and x.shape[-1] == ATOM_WIDTH and cond.shape[-1] == ATOM_WIDTH
            and os.environ.get("MINIWORLD_ADALN_ATOM", "1") != "0" and os.environ.get("MINIWORLD_ADALN_ATOM_TF32", "1") != "0")


def atom_fwd_tf32(x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, ws: torch.Tensor, wb: torch.Tensor, sb: torch.Tensor, eps_x: float, eps_c: float) -> torch.Tensor:
    """The whole forward in one kernel on TF32 tensor cores (``adaln_atom_fwd_tf32.cuh``): y [M, 128] for fp32 x [M, 128], cond [P, 128] (row r reads cond row r % P), lnw [128], Ws / Wb [128, 128], sb [128]."""
    y = torch.empty_like(x)
    _ext().adaln_atom_fwd_tf32(x, cond, lnw, ws, wb, sb, y, float(eps_x), float(eps_c))
    return y


def atom_bwd(dy: torch.Tensor, x: torch.Tensor, cond: torch.Tensor, xst: torch.Tensor, cst: torch.Tensor, lnw: torch.Tensor, ws: torch.Tensor, wb: torch.Tensor,
             sb: torch.Tensor, dres: torch.Tensor | None = None, dextra: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    """The backward in one kernel (``adaln_atom_bwd.cuh``): ``(dx, dcond, dscale, aff, psb, plnw)`` for dy / x / cond [M, 128] bf16 (cond one row per row), the forward's statistics
    xst / cst [M, 2] fp32, lnw [128] fp32, Ws / Wb [128, 128] and sb [128] bf16; ``dres`` is added to dx, ``dextra`` to dcond.  dscale and aff are the operands of the weight gradients
    (dWs = dscale^T aff, dWb = dy^T aff: cuBLAS); psb / plnw [rows, 128] fp32 are the per-CTA partial column sums of dscale and of the cond norm's weight gradient (``finish`` adds the rows)."""
    m = x.shape[0]
    dx, dcond, dsc, aff = (torch.empty_like(x) for _ in range(4))
    rows = _ext().atom_bwd_grid(m)
    psb = torch.empty(rows, ATOM_WIDTH, device=x.device, dtype=torch.float32)
    plw = torch.empty_like(psb)
    _ext().adaln_atom_bwd(dy, x, cond, xst, cst, lnw, ws, ws.t().contiguous(), wb.t().contiguous(), sb, dres, dextra, dx, dcond, dsc, aff, psb, plw)
    return dx, dcond, dsc, aff, psb, plw


def finish(sums: list[torch.Tensor], sum_like: list[torch.Tensor], casts: list[torch.Tensor], cast_like: list[torch.Tensor]) -> list[torch.Tensor]:
    """The closing pass of a backward in ONE launch (``adaln_finish.cuh``): ``sums[i]`` [rows, n] fp32 partial rows -> their column sums [n] in ``sum_like[i]``'s dtype, ``casts[i]`` an fp32
    matrix (any strides) -> a fresh contiguous matrix in ``cast_like[i]``'s dtype; returns the sums, then the casts.  Fixed order: bit-reproducible."""
    return _ext().finish(list(sums), list(sum_like), list(casts), list(cast_like))


__all__ = ["ATOM_WIDTH", "ROWS_PER_BLOCK", "WIDTHS", "atom_bwd", "atom_fwd", "atom_fwd_tf32", "atom_supported", "atom_tf32_supported", "available", "blocks_for", "bwd_x", "cond_bwd", "cond_ln", "epilogue", "finish"]
