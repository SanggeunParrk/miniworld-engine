"""Separate own-CUDA contraction/gate/projection candidate; no default dispatch."""

import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from miniworld_engine.kernels import _msa_card as _card


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    source = Path(__file__).parent / "sm80" / "contract_gate_fused.cu"
    return load_extension(
        name=_card.name("pwa_contract_gate_fused_sm80"),
        sources=[str(source)],
        extra_cuda_cflags=_card.cflags([
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
        ]),
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def result(m, w, v, gamma, beta, wg, wo, keep=None, eps=1e-5, dscale=1.0, save_o=False, mode=0):
    out = torch.empty_like(m)
    saved = torch.empty_like(v) if save_o else None
    _ext().run(m, w, v, gamma, beta, wg, wo, keep, out, saved, eps, dscale, mode)
    return out, saved
