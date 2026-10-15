"""sm_100a / sm_90a build and launch of the smooth lDDT kernels (``smooth_lddt.cu``).

The cubin is built on first use by the newest nvcc here that knows sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``) and cached
under ``settings.engine_jit_root()``; launches go through the CUDA driver on torch's current stream (CUDA-graph capturable), the way the
``token_pair_init`` and ``swa_dit`` sm_100a stages do. Plain CUDA-core kernels: the op is bound by the exponentials of the pair scores.
"""
from __future__ import annotations

import functools
import hashlib
import os
import subprocess
from pathlib import Path

import torch

from miniworld_engine import settings

_dir = Path(__file__).parent
BLK = 128   # atoms per tile / threads per CTA (slddt_*: one thread per atom row)
FBLK = 512  # threads per CTA of the whole-structure kernels (slddt_*_full)
FULL_SMEM = 110 * 1024   # whole-structure kernels need N * 32 B of shared memory; two CTAs per SM (N <= 3520 atoms) beat the tile kernels
FORCE_TILE = False  # tests set it to compare the tile kernel with the whole-structure one


def _full(n: int) -> bool:
    return not FORCE_TILE and n * 32 <= FULL_SMEM


def _arch(index: int) -> str:
    """nvcc arch of the device: sm_100a (B200) or sm_90a (H100). The kernels are plain CUDA-core code, the same source serves both."""
    major, _ = torch.cuda.get_device_capability(index)
    return "sm_90a" if major == 9 else "sm_100a"


@functools.lru_cache(maxsize=None)
def cubin(arch: str = "sm_100a") -> str:
    """Path of the cubin, built on first use; rebuilt only when the source, a flag or the toolchain changes."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    nvcc, rel, host = kernel_toolchain()
    flags = (*host, "-std=c++17", "-O3", f"-arch={arch}", "-cubin", "-lineinfo", "-ftz=true", "-prec-sqrt=false", "-prec-div=false")
    h = hashlib.sha256(" ".join((nvcc, str(rel), *flags)).encode())
    src = _dir / "smooth_lddt.cu"
    h.update(src.read_bytes())
    root = settings.engine_jit_root()
    out = root / f"smooth_lddt_{arch[3:]}" / f"smooth_lddt_{h.hexdigest()[:16]}.cubin"
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
def _kernels(index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    with torch.cuda.device(index):
        # the whole-structure kernels take N * 32 B of dynamic shared memory: the attribute is raised once, the launch size is set per call
        return tuple(driver.Kernel(cubin(_arch(index)), name, FULL_SMEM if name.endswith("_full") else 0)
                     for name in ("slddt_prep", "slddt_fwd", "slddt_bwd", "slddt_fwd_full", "slddt_bwd_full"))


def _dev(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def prep(gt, nuc, mask):
    """(tile [B, NB, NB] int8, row_count [B, N] int32, FM, UM [B, N, N / 32] int32 bit masks) of the ground truth gt [B, N, 3] fp32;
    nuc / mask [B, N] int8; N % 128 == 0. tile bit 0: FM has a bit in the tile, bit 1: UM has one."""
    B, N = mask.shape
    nb = N // BLK
    prep_k = _kernels(_dev(gt))[0]
    tile = torch.empty(B, nb, nb, device=gt.device, dtype=torch.int8)
    cnt = torch.zeros(B, N, device=gt.device, dtype=torch.int32)   # the kernel adds the per-tile counts with integer atomics
    fm = torch.empty(B, N, N // 32, device=gt.device, dtype=torch.int32)   # every word is written
    um = torch.empty_like(fm)
    with torch.cuda.device(gt.device):
        prep_k((B, nb, nb), (BLK, 1, 1), gt, nuc, mask, tile, cnt, fm, um, int(B), int(N), int(nb))
    return tile, cnt, fm, um


def forward(x, gt, tile, fm):
    """Per (augment * batch, row block) partial sums of the pair scores, [A * B, NB] fp32."""
    A, B, N, _ = x.shape
    nb = N // BLK
    ks = _kernels(_dev(x))
    out = torch.empty(A * B, nb, device=x.device, dtype=torch.float32)
    with torch.cuda.device(x.device):
        if _full(N):
            k = ks[3]
            k.smem = N * 32
            k((A * B, -(-N // FBLK), 1), (FBLK, 1, 1), x, gt, fm, out, int(B), int(N), int(nb))
        else:
            ks[1]((A * B, nb, 1), (BLK, 1, 1), x, gt, tile, fm, out, int(B), int(N), int(nb))
    return out


def backward(x, gt, nuc, tile, um, go):
    """d(sum of the pair scores)/dx [A, B, N, 3] fp32 for the upstream gradient go [A * B] fp32."""
    A, B, N, _ = x.shape
    nb = N // BLK
    ks = _kernels(_dev(x))
    gx = torch.empty_like(x)
    with torch.cuda.device(x.device):
        if _full(N):
            k = ks[4]
            k.smem = N * 32
            k((A * B, -(-N // FBLK), 1), (FBLK, 1, 1), x, gt, nuc, um, go, gx, int(B), int(N), int(nb))
        else:
            ks[2]((A * B, nb, 1), (BLK, 1, 1), x, gt, nuc, tile, um, go, gx, int(B), int(N), int(nb))
    return gx
