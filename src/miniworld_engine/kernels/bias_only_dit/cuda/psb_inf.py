"""Inference attention cores on a pair bias of its own per sample (``bo_psb32_inf.cu``): ``softmax(bias + key mask) v`` with the sigmoid gate,
without the backward bookkeeping of the training cores (``bo_psb.cu`` / ``bo_psb32.cu``). Built on first use."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent
#: Longest L and head widths of the fp32 inference core (v of the (head, sample) stays resident in shared memory)
PSB32I_MAX_L = 512
PSB32I_DH = (32, 48)


@functools.lru_cache(maxsize=None)
def psb32i_ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_psb32_inf_sm90a",
        sources=[str(_dir / "bo_psb32_inf.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a")],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def psb32i_served(L: int, dh: int) -> bool:
    return L <= PSB32I_MAX_L and L % 128 == 0 and dh in PSB32I_DH


def psb32i_fwd(bias: torch.Tensor, mask: torch.Tensor | None, vg: torch.Tensor, og: torch.Tensor, samples: int, heads: int, dh: int) -> None:
    """og = sigmoid(g) (softmax(bias + key mask) v), fp32 on TF32 tensor cores. bias [H A L, L] head-major; vg [A L, 2 H dh] = v | g; og [A L, H dh];
    mask bool [L] or [A, L] or None."""
    psb32i_ext().psb32i_fwd(bias, mask, vg, og, int(samples), int(heads), int(dh))
