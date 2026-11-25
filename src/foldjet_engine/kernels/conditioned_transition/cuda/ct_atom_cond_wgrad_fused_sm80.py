"""CUDA conditioning weight gradients from physical BF16 fields, FP32 output."""

import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent
    return load_extension(
        name="ct_atom_cond_wgrad_fused_sm80",
        sources=[str(d / "ct_atom_cond_wgrad_fused_sm80.cu")],
        extra_include_paths=[str(d)],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )


def run(a0, a1, a2, b0, b1, partial, out, cfg=0, splits=32, planes=3):
    _ext().run(a0, a1, a2, b0, b1, partial, out, cfg, splits, planes)


def result(dsc, dxa, dg, aff, cond):
    """Return dWs1, dWb1, dWsc without packing the physical row operands."""
    rows = dsc.shape[0]
    if rows == 48 * 4096:
        cfg, splits = 2, 16
    elif rows == 48 * 8192:
        cfg, splits = 3, 32
    else:
        raise ValueError("qualified atom conditioning gradient dimensions required")
    partial = torch.empty((splits, 384, 128), dtype=torch.float32, device=dsc.device)
    out = torch.empty((384, 128), dtype=torch.float32, device=dsc.device)
    run(dsc, dxa, dg, aff, cond, partial, out, cfg, splits, 3)
    return out[:128], out[128:256], out[256:]


def result_two(dsc, dy, aff):
    """AdaLN's two exact physical weight gradients, no operand pack or BF rounding."""
    rows = dsc.shape[0]
    if rows not in (48 * 4096, 48 * 8192):
        raise ValueError("qualified atom AdaLN weight-gradient dimensions required")
    partial = torch.empty((32, 256, 128), dtype=torch.float32, device=dsc.device)
    out = torch.empty((256, 128), dtype=torch.float32, device=dsc.device)
    # The third-plane fields are unused in planes=2; aliases add no copies.
    run(dsc, dy, dsc, aff, aff, partial, out, 8, 32, 2)
    return out[:128], out[128:]
