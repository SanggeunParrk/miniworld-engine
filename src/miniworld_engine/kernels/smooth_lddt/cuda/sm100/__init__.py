"""sm_100a build and launch of the smooth lDDT kernels (``smooth_lddt.cu``).

The cubin is built on first use by the newest nvcc here that knows sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``) and cached
under ``MINIWORLD_ENGINE_JIT_ROOT``; launches go through the CUDA driver on torch's current stream (CUDA-graph capturable), the way the
``token_pair_init`` and ``swa_dit`` sm_100a stages do. Plain CUDA-core kernels: the op is bound by the exponentials of the pair scores.
"""
from __future__ import annotations

import functools
import hashlib
import os
import subprocess
from pathlib import Path

import torch

_dir = Path(__file__).parent
BLK = 128   # atoms per tile / threads per CTA (slddt_*: one thread per atom row)
GA = int(os.environ.get("MW_SLDDT_GA", "1"))   # augments per thread: also the -DGA of the cubin


@functools.lru_cache(maxsize=None)
def cubin() -> str:
    """Path of the cubin, built on first use; rebuilt only when the source, a flag or the toolchain changes."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    nvcc, rel, host = kernel_toolchain()
    flags = (*host, "-std=c++17", "-O3", "-arch=sm_100a", "-cubin", "-lineinfo", f"-DGA={GA}")
    h = hashlib.sha256(" ".join((nvcc, str(rel), *flags)).encode())
    src = _dir / "smooth_lddt.cu"
    h.update(src.read_bytes())
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    out = root / "smooth_lddt_sm100" / f"smooth_lddt_{h.hexdigest()[:16]}.cubin"
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
        return tuple(driver.Kernel(cubin(), name, 0) for name in ("slddt_prep", "slddt_fwd", "slddt_bwd"))


def _dev(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def prep(gt, nuc, mask):
    """(tile [B, NB, NB] int8, row_count [B, N] int32) of the ground truth gt [B, N, 3] fp32; nuc / mask [B, N] int8; N % 128 == 0."""
    B, N = mask.shape
    nb = N // BLK
    prep_k, _, _ = _kernels(_dev(gt))
    tile = torch.empty(B, nb, nb, device=gt.device, dtype=torch.int8)
    cnt = torch.zeros(B, N, device=gt.device, dtype=torch.int32)   # the kernel adds the per-tile counts with integer atomics
    with torch.cuda.device(gt.device):
        prep_k((B, nb, nb), (BLK, 1, 1), gt, nuc, mask, tile, cnt, int(B), int(N), int(nb))
    return tile, cnt


def forward(x, gt, nuc, mask, tile):
    """Per (augment * batch, row block) partial sums of the pair scores, [A * B, NB] fp32."""
    A, B, N, _ = x.shape
    nb = N // BLK
    _, fwd_k, _ = _kernels(_dev(x))
    out = torch.empty(A * B, nb, device=x.device, dtype=torch.float32)
    with torch.cuda.device(x.device):
        fwd_k((-(-A // GA) * B, nb, 1), (BLK, 1, 1), x, gt, nuc, mask, tile, out, int(A), int(B), int(N), int(nb))
    return out


def backward(x, gt, nuc, mask, tile, go):
    """d(sum of the pair scores)/dx [A, B, N, 3] fp32 for the upstream gradient go [A * B] fp32."""
    A, B, N, _ = x.shape
    nb = N // BLK
    _, _, bwd_k = _kernels(_dev(x))
    gx = torch.empty_like(x)
    with torch.cuda.device(x.device):
        bwd_k((-(-A // GA) * B, nb, 1), (BLK, 1, 1), x, gt, nuc, mask, tile, go, gx, int(A), int(B), int(N), int(nb))
    return gx
