"""Own-CUDA plane GEMM (``sm80/contract_tuned.cu``): ``out[c] = op(a[c]) op(b[c])`` over square bf16 planes, fp32 accumulation, bf16 out.

On the A100 it is a candidate only (the callers keep cuBLAS there). On the GeForce RTX 5090 (the same source built for sm_120a) it is the
contraction of the wide TriMul path (``sm80_wide``, forward and backward): cuBLAS picks 32 x 32 WMMA kernels for many of these batched
plane shapes on sm_120 (2-3x off the tensor-core rate); the tile per (mode, channels, length) comes from ``SM120_TILES``."""

import functools
from pathlib import Path

from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    return load_extension(
        name="trimul_contract_sm80",
        sources=[str(directory / "contract_tuned.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_80,code=sm_80",
            f"-I{directory}",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def _ext120():
    """The same source for the GeForce RTX 5090 (sm_120a)."""
    from . import sm80

    return sm80._build("trimul_contract_sm80", "contract_tuned.cu", 120)


def run(a, b, out, h, *, mode=-1, config=0, grid=0):
    """Write square BF16 planes: mode -1 mixed NT/TN, 0 NT, 1 TN, 2 NN, 3 TT."""
    from . import sm80

    (_ext120() if sm80.on_sm120() else _ext()).run(a, b, out, h, mode, config, grid)


#: RTX 5090 (sm_120a): the measured tile (``contract_tuned.cu`` config) per (mode, channels, length); a shape that is not listed keeps cuBLAS
#: (``bench_contract``: own tile vs cuBLAS bmm, L2 flushed, median). Modes: 0 NT (out = a b^T), 1 TN (a^T b), 2 NN (a b).
SM120_TILES: dict[tuple[int, int, int], int] = {  # 2026-11-12, every listed shape 1.0-3.0x faster than cuBLAS bmm (all 90 measured shapes)
    (0, 64, 128): 2, (1, 64, 128): 4, (2, 64, 128): 4, (0, 64, 256): 1, (1, 64, 256): 2, (2, 64, 256): 1,
    (0, 64, 384): 13, (1, 64, 384): 2, (2, 64, 384): 2, (0, 64, 512): 2, (1, 64, 512): 2, (2, 64, 512): 2,
    (0, 64, 640): 1, (1, 64, 640): 3, (2, 64, 640): 14, (0, 64, 768): 14, (1, 64, 768): 1, (2, 64, 768): 2,
    (0, 128, 128): 4, (1, 128, 128): 0, (2, 128, 128): 0, (0, 128, 256): 1, (1, 128, 256): 1, (2, 128, 256): 1,
    (0, 128, 384): 1, (1, 128, 384): 2, (2, 128, 384): 2, (0, 128, 512): 2, (1, 128, 512): 2, (2, 128, 512): 14,
    (0, 128, 640): 1, (1, 128, 640): 1, (2, 128, 640): 1, (0, 128, 768): 1, (1, 128, 768): 3, (2, 128, 768): 2,
    (0, 256, 128): 1, (1, 256, 128): 1, (2, 256, 128): 1, (0, 256, 256): 0, (1, 256, 256): 2, (2, 256, 256): 0,
    (0, 256, 384): 1, (1, 256, 384): 1, (2, 256, 384): 1, (0, 256, 512): 1, (1, 256, 512): 2, (2, 256, 512): 2,
    (0, 256, 640): 1, (1, 256, 640): 1, (2, 256, 640): 1, (0, 256, 768): 1, (1, 256, 768): 1, (2, 256, 768): 1,
    (0, 384, 128): 1, (1, 384, 128): 0, (2, 384, 128): 2, (0, 384, 256): 0, (1, 384, 256): 2, (2, 384, 256): 2,
    (0, 384, 384): 1, (1, 384, 384): 2, (2, 384, 384): 1, (0, 384, 512): 1, (1, 384, 512): 2, (2, 384, 512): 1,
    (0, 384, 640): 14, (1, 384, 640): 1, (2, 384, 640): 1, (0, 384, 768): 1, (1, 384, 768): 1, (2, 384, 768): 1,
    (0, 512, 128): 1, (1, 512, 128): 0, (2, 512, 128): 1, (0, 512, 256): 1, (1, 512, 256): 3, (2, 512, 256): 2,
    (0, 512, 384): 1, (1, 512, 384): 2, (2, 512, 384): 1, (0, 512, 512): 1, (1, 512, 512): 1, (2, 512, 512): 1,
    (0, 512, 640): 1, (1, 512, 640): 1, (2, 512, 640): 1, (0, 512, 768): 1, (1, 512, 768): 1, (2, 512, 768): 1,
}


def _cublas(mode: int, a, b, out) -> None:
    import torch

    if mode == 0:
        torch.bmm(a, b.transpose(1, 2), out=out)
    elif mode == 1:
        torch.bmm(a.transpose(1, 2), b, out=out)
    elif mode == 2:
        torch.bmm(a, b, out=out)
    else:
        torch.bmm(a.transpose(1, 2), b.transpose(1, 2), out=out)


def planes(mode: int, a, b, out) -> None:
    """``out[c] = op(a[c]) op(b[c])`` over [C, n, n] planes (mode 0 NT, 1 TN, 2 NN, 3 TT): the own kernel where ``SM120_TILES`` names a tile for this
    shape on an RTX 5090 (bf16, contiguous), cuBLAS otherwise (every other card: the A100 / H100 paths are unchanged)."""
    import torch

    from . import sm80

    if a.dtype is torch.bfloat16 and a.is_cuda and sm80.on_sm120():
        cfg = SM120_TILES.get((mode, a.shape[0], a.shape[1]))
        if cfg is not None and a.is_contiguous() and b.is_contiguous() and out.is_contiguous():
            _ext120().run(a, b, out, 0, mode, cfg, 0)
            return
    _cublas(mode, a, b, out)
