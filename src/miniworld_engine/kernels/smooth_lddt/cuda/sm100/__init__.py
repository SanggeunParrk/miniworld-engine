"""sm_100a / sm_90a build and launch of the smooth lDDT kernels (``smooth_lddt.cu``).

The cubin is built on first use by the newest nvcc here that knows sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``) and cached
under ``settings.engine_jit_root()``; launches go through the CUDA driver on torch's current stream (CUDA-graph capturable), the way the
``token_pair_init`` and ``swa_dit`` sm_100a stages do. Plain CUDA-core kernels: the op is bound by the exponentials of the pair scores.
"""
from __future__ import annotations

import functools
from pathlib import Path

import torch


_dir = Path(__file__).parent
BLK = 128   # atoms per tile / threads per CTA (slddt_*: one thread per atom row)
FBLK = 512  # threads per CTA of the whole-structure kernels (slddt_*_full) on B200; H100 picks 128 / 256 / 512 (:func:`_fblk`)
FULL_SMEM = 110 * 1024   # whole-structure kernels need N * 32 B of shared memory; two CTAs per SM (N <= 3520 atoms) beat the tile kernels
FORCE_TILE = False  # tests set it to compare the tile kernel with the whole-structure one


def _full(n: int) -> bool:
    return not FORCE_TILE and n * 32 <= FULL_SMEM


FORCE_FWD = ""   # tests / probes: a fixed forward variant "<threads>s<threads per row>" of FWD_VARIANTS (the whole-structure kernel)
FORCE_FB = 0     # ... and a fixed backward CTA size (128 / 256 / 512)
FWD_VARIANTS = ("128s1", "256s1", "512s1", "256s2", "512s2", "512s4")
FBLK_FILL = 2   # H100 backward: the largest CTA (512 / 256 / 128 threads) whose grid keeps FBLK_FILL CTAs per SM (else the smallest)


@functools.lru_cache(maxsize=None)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _fblk(x: torch.Tensor, n_augment_batch: int, n: int) -> int:
    """Threads per CTA of the whole-structure kernels. B200 keeps 512. On H100 a launch of A * B * N / 512 CTAs can be smaller than the SM count (A48 x N1024:
    96 CTAs on 132 SMs, the other SMs idle, 16 warps on the rest): take the largest CTA whose grid has at least ``FBLK_FILL`` CTAs per SM (else the smallest)."""
    if FORCE_FB:
        return FORCE_FB
    if _arch(_dev(x)) != "sm_90a":
        return FBLK
    sms = _sm_count(_dev(x))
    for fb in (512, 256, 128):
        if n_augment_batch * -(-n // fb) >= FBLK_FILL * sms:
            return fb
    return 128


def _fwd_variant(x: torch.Tensor, n_augment_batch: int, n: int) -> str:
    """Forward whole-structure variant "<threads>s<threads per row>": B200 keeps 512s1; H100 deals each row to S = 2 threads (a launch has only A * B * N rows: A48 x N1024 is 11 warps per SM) and sizes the CTA to the grid."""
    if FORCE_FWD:
        return FORCE_FWD
    if _arch(_dev(x)) != "sm_90a":
        return "512s1"
    sms = _sm_count(_dev(x))
    for v, rows in (("512s2", 256), ("256s2", 128)):   # the largest CTA whose grid still has 1.5 CTAs per SM (measured A16 / A48, N512 - N2048)
        if n_augment_batch * -(-n // rows) >= 1.5 * sms:
            return v
    return "256s2"


def _arch(index: int) -> str:
    """nvcc arch of the device: sm_100a (B200) or sm_90a (H100). The kernels are plain CUDA-core code, the same source serves both."""
    major, _ = torch.cuda.get_device_capability(index)
    return "sm_90a" if major == 9 else "sm_100a"


@functools.lru_cache(maxsize=None)
def cubin(arch: str = "sm_100a") -> str:
    """Path of the cubin, built on first use; rebuilt only when the source, a flag or the toolchain changes."""
    from miniworld_engine.kernels import _nvcc

    flags = ("-std=c++17", "-O3", f"-arch={arch}", "-cubin", "-lineinfo", "-ftz=true", "-prec-sqrt=false", "-prec-div=false")
    return str(_nvcc.cubin(f"smooth_lddt_{arch[3:]}/smooth_lddt", flags, source=_dir / "smooth_lddt.cu", toolchain="sm100"))


@functools.lru_cache(maxsize=16)
def _kernels(index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    with torch.cuda.device(index):
        # the whole-structure kernels take N * 32 B of dynamic shared memory: the attribute is raised once, the launch size is set per call
        names = ("slddt_prep", "slddt_fwd", "slddt_bwd", *(f"slddt_fwd_full_{v}" for v in FWD_VARIANTS), *(f"slddt_bwd_full_{fb}" for fb in (128, 256, 512)))
        return {name: driver.Kernel(cubin(_arch(index)), name, FULL_SMEM if "_full" in name else 0) for name in names}


def _dev(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def prep(gt, nuc, mask):
    """(tile [B, NB, NB] int8, row_count [B, N] int32, FM, UM [B, N, N / 32] int32 bit masks) of the ground truth gt [B, N, 3] fp32;
    nuc / mask [B, N] int8; N % 128 == 0. tile bit 0: FM has a bit in the tile, bit 1: UM has one."""
    B, N = mask.shape
    nb = N // BLK
    prep_k = _kernels(_dev(gt))["slddt_prep"]
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
            v = _fwd_variant(x, A * B, N)
            fb, s = (int(p) for p in v.split("s"))
            k = ks[f"slddt_fwd_full_{v}"]
            k.smem = N * 32
            k((A * B, -(-N // (fb // s)), 1), (fb, 1, 1), x, gt, fm, out, int(B), int(N), int(nb))
        else:
            ks["slddt_fwd"]((A * B, nb, 1), (BLK, 1, 1), x, gt, tile, fm, out, int(B), int(N), int(nb))
    return out


def backward(x, gt, nuc, tile, um, go):
    """d(sum of the pair scores)/dx [A, B, N, 3] fp32 for the upstream gradient go [A * B] fp32."""
    A, B, N, _ = x.shape
    nb = N // BLK
    ks = _kernels(_dev(x))
    gx = torch.empty_like(x)
    with torch.cuda.device(x.device):
        if _full(N):
            fb = _fblk(x, A * B, N)
            k = ks[f"slddt_bwd_full_{fb}"]
            k.smem = N * 32
            k((A * B, -(-N // fb), 1), (fb, 1, 1), x, gt, nuc, um, go, gx, int(B), int(N), int(nb))
        else:
            ks["slddt_bwd"]((A * B, nb, 1), (BLK, 1, 1), x, gt, nuc, tile, um, go, gx, int(B), int(N), int(nb))
    return gx
