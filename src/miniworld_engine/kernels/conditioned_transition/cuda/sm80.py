"""A100 (sm_80) ConditionedTransition tail: ``y = x + sigmoid(cond Wsc^T + bsc) * (silu(a) b) Ws^T`` with ``[a | b] = xa [Wa; Wb]^T`` -- the row passes in hand CUDA
(``sm80/ct_rows.cuh``), the GEMMs between them cuBLAS (``integrations/conditioned_transition_sm80.py`` composes them with the AdaLN of ``kernels/adaln/cuda/sm80``).

    swiglu_fwd / swiglu_bwd     h = silu(a) b;  da | db and h (recomputed for the squeeze's weight gradient)
    gate_res_fwd / gate_res_bwd y = x + sigmoid(g) z (x optional);  dz, dg and the column sums of dg

Rows are fp32 or bf16.  Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

import torch

from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from ...adaln.cuda.sm80 import ROWS_PER_BLOCK, WIDTHS, blocks_for

_dir = Path(__file__).parent / "sm80"
_adaln_dir = Path(__file__).parents[2] / "adaln" / "cuda" / "sm80"       # adaln_common.cuh


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    # MINIWORLD_CONDTRANS_SM80_FLAGS: extra -D / nvcc flags for A/B experiments (their own build)
    extra = os.environ.get("MINIWORLD_CONDTRANS_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"conditioned_transition_sm80{tag}",
        sources=[str(_dir / "ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                           "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", f"-I{_adaln_dir}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


def available() -> bool:
    """The extension builds (a failure is the caller's to report once)."""
    _ext()
    return True


def swiglu_fwd(ab: torch.Tensor) -> torch.Tensor:
    """``h [M, N] = silu(a) b`` for ``ab = [a | b]`` [M, 2 N]."""
    h = torch.empty(ab.shape[0], ab.shape[1] // 2, device=ab.device, dtype=ab.dtype)
    _ext().swiglu_fwd(ab, h)
    return h


def swiglu_bwd(dh: torch.Tensor, ab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(dab [M, 2 N] = [da | db], h [M, N])`` of ``h = silu(a) b`` from dh [M, N] and ab [M, 2 N]."""
    m, n = dh.shape
    dab = torch.empty(m, 2 * n, device=dh.device, dtype=dh.dtype)
    h = torch.empty(m, n, device=dh.device, dtype=dh.dtype)
    _ext().swiglu_bwd(dh, ab, dab, h)
    return dab, h


def gate_res_fwd(x: torch.Tensor | None, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """``y [M, d] = x + sigmoid(g[r % P]) z`` (x None: the update alone), z [M, d], g [P, d] (the gate logits), contiguous."""
    y = torch.empty_like(z)
    _ext().gate_res_fwd(x, z, g, y)
    return y


def gate_res_bwd(dy: torch.Tensor, z: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(dz, dg, pb)``: dz = sigmoid(g) dy, dg = dy z s (1 - s) [M, d] and pb [blocks, d] fp32 = the column sums of dg (add the rows for d bsc); one gate row per row."""
    m, d = z.shape
    dz, dg = torch.empty_like(z), torch.empty_like(z)
    pb = torch.empty(blocks_for(m, d, z.device), d, device=z.device, dtype=torch.float32)
    _ext().gate_res_bwd(dy, z, g, dz, dg, pb)
    return dz, dg, pb



# ------------------------------------------------------------------------------------------------------------ atom width (128), bf16
def atom_supported(x: torch.Tensor, cond: torch.Tensor, wa: torch.Tensor) -> bool:
    """The fused atom kernels' contract: bf16 rows, d_hidden = d_cond = 128, expansion 2 (hidden 256)."""
    return (x.dtype is torch.bfloat16 and cond.dtype is torch.bfloat16 and x.shape[-1] == 128 and cond.shape[-1] == 128 and tuple(wa.shape) == (256, 128)
            and os.environ.get("MINIWORLD_CONDTRANS_ATOM", "1") != "0")


def atom_fwd(xa: torch.Tensor, xin: torch.Tensor | None, cond: torch.Tensor, wab: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor,
             save_z: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The tail in one kernel (``ct_atom_fwd.cuh``): ``(y, z)`` = x + sigmoid(cond Wsc^T + bsc) (silu(xa Wa^T) (xa Wb^T)) Ws^T for xa [M, 128] (the AdaLN's output), x [M, 128] or
    None (the update alone), cond [P, 128] (row r reads row r % P), wab [512, 128] = [Wa; Wb], Ws [128, 256], Wsc [128, 128], bsc [128], all bf16; with ``save_z`` also rn(z) [M, 128]."""
    y = torch.empty_like(xa)
    z = torch.empty_like(xa) if save_z else None
    _ext().ct_atom_fwd(xa, xin, cond, wab, ws, wsc, bsc, y, z)
    return y, z



def atom_bwd_gate(dy: torch.Tensor, z: torch.Tensor, cond: torch.Tensor, xa: torch.Tensor, wab: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor,
                  bsc: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The gate / FFN-backward kernel (``ct_atom_bwd.cuh``): ``(dz, dg, dcond2, hh, dab, pbsc)`` for dy / z (rn(z)) / cond (one row per row) / xa [M, 128], wab [512, 128] = [Wa; Wb],
    Ws [128, 256], Wsc [128, 128], bsc [128], all bf16: dz = g dy, dg = dy z g (1 - g), dcond2 = dg Wsc, hh = h (the weight gradient of Ws's operand), dab = da | db [M, 512];
    pbsc [rows, 128] fp32 holds the per-CTA partial column sums of dg (``adaln sm80 finish`` adds the rows for d bsc)."""
    m = xa.shape[0]
    dz, dg, dcond2 = (torch.empty_like(xa) for _ in range(3))
    hh = torch.empty(m, 256, device=xa.device, dtype=xa.dtype)
    dab = torch.empty(m, 512, device=xa.device, dtype=xa.dtype)
    pbsc = torch.empty(_ext().ct_bwd_grid(m), 128, device=xa.device, dtype=torch.float32)
    _ext().ct_atom_bwd_gate(dy, z, cond, xa, wab, ws.t().contiguous(), wsc, wsc.t().contiguous(), bsc, dz, dg, dcond2, hh, dab, pbsc)
    return dz, dg, dcond2, hh, dab, pbsc


def atom_tf32_supported(x: torch.Tensor, cond: torch.Tensor, wa: torch.Tensor) -> bool:
    """The fp32 (TF32) fused tail's contract: fp32 rows, d_hidden = d_cond = 128, expansion 2 (inference only)."""
    return (x.dtype is torch.float32 and cond.dtype is torch.float32 and x.shape[-1] == 128 and cond.shape[-1] == 128 and tuple(wa.shape) == (256, 128)
            and os.environ.get("MINIWORLD_CONDTRANS_ATOM", "1") != "0" and os.environ.get("MINIWORLD_CONDTRANS_ATOM_TF32", "1") != "0")


def tf32_round(t: torch.Tensor) -> torch.Tensor:
    """``t`` (fp32) rounded to TF32 as ``cvt.rna.tf32.f32`` does (nearest, ties away from zero; the low 13 bits cleared), in an fp32 container."""
    return ((t.contiguous().view(torch.int32) + 0x1000) & -8192).view(torch.float32)


def _pack_indices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(out_perm [128], unit_perm [256]): row s of the packed squeeze weight is output channel ``qf_channel(s)`` (the f1 row order), and column 32 c + 8 q + idx of a row is hidden unit
    32 c + 8 (2 (idx >> 2) + ((idx >> 1) & 1)) + 2 q + (idx & 1) (``ct_atom_fwd_tf32.cuh``: the units a lane's B fragments of chunk c need, contiguous)."""
    s = torch.arange(128, device=device)
    out_perm = (s & ~63) + 32 * ((s >> 5) & 1) + 8 * ((s >> 1) & 3) + 2 * ((s >> 3) & 3) + (s & 1)
    pos = torch.arange(256, device=device)
    c, q, idx = pos >> 5, (pos >> 3) & 3, pos & 7
    unit_perm = 32 * c + 8 * (2 * (idx >> 2) + ((idx >> 1) & 1)) + 2 * q + (idx & 1)
    return out_perm, unit_perm


def pack_tf32(wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(wab [512, 128], wsp [128, 256])`` for ``atom_fwd_tf32``: [Wa; Wb] and the squeeze weight Ws [128, 256], TF32-rounded, the squeeze weight permuted for the kernel."""
    out_perm, unit_perm = _pack_indices(ws.device)
    # the 8 inputs a lane reads of a row for 32 channels are stored as the pairs (e0, e4) (e1, e5) | (e2, e6) (e3, e7), as the AdaLN kernel's weights are (``adaln_tf32.cuh: load_weight128_tf32``)
    col_perm = torch.arange(128, device=wa.device).view(16, 8)[:, [0, 4, 1, 5, 2, 6, 3, 7]].reshape(-1)
    return tf32_round(torch.cat([wa, wb])[:, col_perm]), tf32_round(ws[out_perm][:, unit_perm])


def atom_fwd_tf32(xa: torch.Tensor, xin: torch.Tensor | None, cond: torch.Tensor, wab: torch.Tensor, wsp: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor) -> torch.Tensor:
    """The tail on TF32 tensor cores (``ct_atom_fwd_tf32.cuh``, two kernels): ``y = x + sigmoid(cond Wg^T + bg) (silu(xa Wa^T) (xa Wb^T)) Ws^T`` for fp32 xa [M, 128] (the AdaLN's output), x [M, 128] or None,
    cond [P, 128] (row r reads row r % P), ``wab`` / ``wsp`` from ``pack_tf32``, Wg [128, 128], bg [128]."""
    z = torch.empty_like(xa)
    y = torch.empty_like(xa)
    _ext().ct_tail_tf32(xa, wab, wsp, z)
    _ext().ct_gate_tf32(z, xin, cond, wsc, bsc, y)
    return y


def atom_bwd_dxa(dab: torch.Tensor, wab: torch.Tensor) -> torch.Tensor:
    """``dxa [M, 128] = dab [M, 512] [Wa; Wb]`` (bf16) for wab [512, 128] = [Wa; Wb]."""
    dxa = torch.empty(dab.shape[0], 128, device=dab.device, dtype=dab.dtype)
    _ext().ct_atom_bwd_dxa(dab, wab.t().contiguous(), dxa)
    return dxa


__all__ = ["ROWS_PER_BLOCK", "WIDTHS", "atom_bwd_dxa", "atom_bwd_gate", "atom_fwd", "atom_fwd_tf32", "atom_supported", "atom_tf32_supported", "available", "blocks_for", "gate_res_bwd", "gate_res_fwd", "swiglu_bwd", "swiglu_fwd"]
