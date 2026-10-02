"""sm_100a build and launch of the token-pair initialisation kernels (``token_pair_init.cu``).

The cubin is built on first use by the newest nvcc here that knows sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``)
and cached under ``MINIWORLD_ENGINE_JIT_ROOT``; launches go through the CUDA driver on torch's current stream (CUDA-graph
capturable), the way the ``swa_dit`` sm_100a stages do. Plain CUDA-core kernels -- no tensor cores: the op is bandwidth bound.
"""

from __future__ import annotations

import functools
import hashlib
import os
import subprocess
from pathlib import Path

import torch

_dir = Path(__file__).parent
P = 128
NWARP = 8
def bwd_rows(L: int) -> int:
    """Rows of dz per CTA in the backward: 1 up to L 256 (more CTAs for the small grids), else 2 (measured L128-L768)."""
    return 1 if L <= 256 else 2


def fwd_rows(L: int) -> int:
    """Rows of z per CTA in the forward: 2 from L 512 (the 72 KB table load is amortised), else 1 (measured L128-L768)."""
    return 2 if L >= 512 else 1


def check_packable(r_max: int, s_max: int) -> None:
    """The kernels pack the three relative-position bins into 8 bits each: the largest is bin3 = 4 r_max + 2 s_max + 5 = n_rel - 2."""
    if n_rel(r_max, s_max) - 2 > 255:
        raise ValueError(f"token_pair_init packs its bins into 8 bits: r_max={r_max}, s_max={s_max} needs more")


def n_rel(r_max: int, s_max: int) -> int:
    """Width of the relative-position one-hot: 2 (2 r_max + 2) + (2 s_max + 2) + 1 (139 for r_max 32, s_max 2)."""
    return 2 * (2 * r_max + 2) + (2 * s_max + 2) + 1


@functools.lru_cache(maxsize=None)
def cubin(ri: int = 2) -> str:
    """Path of the cubin, built on first use; rebuilt only when the source, a flag or the toolchain changes."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    nvcc, rel, host = kernel_toolchain()
    flags = (*host, "-std=c++17", "-O3", "-arch=sm_100a", "-cubin", "-lineinfo", f"-DTPI_RI={ri}")
    h = hashlib.sha256(" ".join((nvcc, str(rel), *flags)).encode())
    src = _dir / "token_pair_init.cu"
    h.update(src.read_bytes())
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    out = root / "token_pair_init_sm100" / f"token_pair_init_ri{ri}_{h.hexdigest()[:16]}.cubin"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        res = subprocess.run([nvcc, *flags, str(src), "-o", str(tmp)], capture_output=True, text=True, timeout=900, check=False)
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"nvcc {rel[0]}.{rel[1]} failed on {src.name}:\n{res.stderr[-4000:]}")
        os.replace(tmp, out)
    return str(out)


@functools.lru_cache(maxsize=16)
def _kernels(index: int, nbin: int, L: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    with torch.cuda.device(index):
        ri = bwd_rows(L)
        fwd = driver.Kernel(cubin(ri), "tpi_fwd", nbin * P * 4 + L * 4)
        bwd = driver.Kernel(cubin(ri), "tpi_bwd", nbin * P * 4 + NWARP * ri * P * 4 + ri * L * 4)
    return fwd, bwd


def _dev(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def forward(left, right, tbl, ids, bond, r_max: int, s_max: int) -> torch.Tensor:
    """z [B, L, L, 128] fp32 from left / right [B, L, 128], the table [nbin, 128], ids [5, B*L] int32, bond [B*L*L] uint8."""
    B, L, _ = left.shape
    check_packable(r_max, s_max)
    nbin = n_rel(r_max, s_max) + 2
    fwd, _ = _kernels(_dev(left), nbin, L)
    out = torch.empty(B, L, L, P, device=left.device, dtype=torch.float32)
    rows = fwd_rows(L)
    tiles = (L + rows - 1) // rows
    with torch.cuda.device(left.device):
        fwd((B * tiles, 1, 1), (32 * NWARP, 1, 1), left, right, tbl, ids, bond, out, int(B), int(L), int(r_max), int(s_max), rows)
    return out


def backward(g, ids, bond, r_max: int, s_max: int):
    """(dleft [B, L, 128], dright [B, L, 128], dtbl [nbin, 128]) of :func:`forward` from dz = g [B, L, L, 128] fp32."""
    B, L = g.shape[:2]
    check_packable(r_max, s_max)
    nbin = n_rel(r_max, s_max) + 2
    _, bwd = _kernels(_dev(g), nbin, L)
    dleft = torch.empty(B, L, P, device=g.device, dtype=torch.float32)
    dright = torch.zeros(B, L, P, device=g.device, dtype=torch.float32)
    dtbl = torch.zeros(nbin, P, device=g.device, dtype=torch.float32)
    tiles = (L + bwd_rows(L) - 1) // bwd_rows(L)
    with torch.cuda.device(g.device):
        bwd((B * tiles, 1, 1), (32 * NWARP, 1, 1), g, ids, bond, dleft, dright, dtbl, int(B), int(L), int(r_max), int(s_max))
    return dleft, dright, dtbl
