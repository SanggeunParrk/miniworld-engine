"""A100 (sm_80) attention core of the bias-only token DiT: ``out = sigmoid(g) * (P v)`` per head and sample, hand CUDA (``mma.sync`` / ``ldmatrix`` / ``cp.async``).

The inference runner of ``cuda/runner.py`` is cuBLAS and the family's CUDA row kernels (``bias_only_dit_rows.cu``, built for sm_80 too) around one attention core; on B200 that core is the
tcgen05 ``pv_gate_inf`` (``PvGateCore``), here it is :class:`PvGate`: the same call, the same layouts, bf16 operands with fp32 accumulation, the gate and one rounding to bf16 in the epilogue.
Built on first use (``load_extension``), never at import.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent
#: head widths the kernel is instantiated for
HEAD_WIDTHS = (32, 48, 64)
#: depth of the cp.async ring (MINIWORLD_BIAS_ONLY_DIT_SM80_STAGES overrides it, for A/B runs)
STAGES = int(os.environ.get("MINIWORLD_BIAS_ONLY_DIT_SM80_STAGES", "2"))


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_sm80",
        sources=[str(_dir / "ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}"],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


def pick_group(S: int, L: int, nh: int, nsm: int) -> int:
    """Samples per CTA (1 .. 4): the fewest CTAs that still fill the card -- a group reads P once for all its samples, and a CTA alone on its SM is request-bound -- i.e. the smallest g with
    ``nh (L / 128) ceil(S / g) <= nsm``, else 4 when even that leaves a second wave."""
    forced = os.environ.get("MINIWORLD_BIAS_ONLY_DIT_SM80_SG")
    if forced:
        return max(1, min(int(forced), 4))
    tiles = nh * (L // 128)
    for g in (1, 2, 3, 4):
        if tiles * -(-S // g) <= nsm:
            return g
    return 4


class PvGate:
    """The A100 twin of ``cuda.PvGateCore``: ``out [S L, W] = sigmoid(g) * (P v)`` -- or, without ``g``, ``P v`` -- per head and sample, W = nh dh."""

    def __init__(self, nh: int, dh: int):
        assert dh in HEAD_WIDTHS, dh
        self.nh, self.dh = nh, dh
        self.nsm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    def __call__(self, v: torch.Tensor, P: torch.Tensor, out: torch.Tensor, S: int, g: torch.Tensor | None = None, stages: int | None = None, sg: int | None = None) -> torch.Tensor:
        """v [S L, W] view, P [nh L, L] contiguous, out [S L, W] view, g [S L, W] view or None; all bf16 (v, g, out: any row stride, unit column stride).
        ``stages`` (2 or 3; default :data:`STAGES`) is the depth of the cp.async ring, ``sg`` (1 .. 4; default :func:`pick_group`) the samples a CTA takes."""
        L = P.shape[1]
        _ext().pv_gate(v, P, out, g if g is not None else v.new_empty(0), S, self.nh, self.dh, STAGES if stages is None else stages,
                       pick_group(S, L, self.nh, self.nsm) if sg is None else sg)
        return out


#: schedule of the bias-gradient core (``dpb``): 0 two stages x one sample with two CTAs per SM, 1 three x one, 2 two x two, 3 three x two (MINIWORLD_BIAS_ONLY_DIT_SM80_DPB overrides)
DPB_CFG = os.environ.get("MINIWORLD_BIAS_ONLY_DIT_SM80_DPB")


class Dpb:
    """The A100 twin of ``cuda.DpbKernel``: ``dbias [nh L, L] = P o (sum_a do v^T - D)`` per head, ``D[h, i] = sum_a dd[a, h, i]`` -- ``do`` / ``v`` bf16 ``[A L, W]`` views (any row stride),
    ``P`` / ``dbias`` bf16 ``[nh L, L]`` contiguous, ``dd`` fp32 ``[A, nh, L]``; a masked key (P = 0) gets 0."""

    def __init__(self, nh: int, dh: int):
        assert dh in HEAD_WIDTHS, dh
        self.nh, self.dh = nh, dh

    def __call__(self, do: torch.Tensor, v: torch.Tensor, P: torch.Tensor, dd: torch.Tensor, out: torch.Tensor, A: int, cfg: int | None = None) -> torch.Tensor:
        _ext().dpb(do, v, P, dd, out, A, self.nh, self.dh, (int(DPB_CFG) if DPB_CFG else 0) if cfg is None else cfg)
        return out


# ------------------------------------------------------------------------------------------------- head planes (the bias-only ATTENTION family, ``kernels/bias_only_attention``)
def pv_planes(v: torch.Tensor, P: torch.Tensor, out: torch.Tensor, samples: int, planes: int, dh: int, stages: int | None = None, sg: int | None = None) -> torch.Tensor:
    """``out = P v`` per head plane for ``samples`` problems that share the plane's ``P``: ``v`` / ``out`` bf16 ``[planes samples L, dh]`` contiguous (plane h = ``[samples L][dh]``, one after
    the other: the bias-only attention's ``[B, H, L (t), L (n), D]``), ``P`` bf16 ``[planes L, L]``."""
    L = P.shape[1]
    nsm = torch.cuda.get_device_properties(v.device).multi_processor_count
    _ext().pv_planes(v, P, out, samples, planes, dh, STAGES if stages is None else stages, pick_group(samples, L, planes, nsm) if sg is None else sg)
    return out


def delta_planes(do: torch.Tensor, o: torch.Tensor, dd: torch.Tensor, samples: int, length: int, planes: int, dh: int) -> torch.Tensor:
    """``dd[a, h, i] = sum_d do o`` over bf16 ``[planes samples L, dh]`` operands (``dd`` fp32 ``[samples, planes, L]``): the backward's row term."""
    _ext().delta_planes(do, o, dd, samples, length, planes, dh)
    return dd


def dpb_planes(do: torch.Tensor, v: torch.Tensor, P: torch.Tensor, dd: torch.Tensor, out: torch.Tensor, samples: int, planes: int, dh: int, cfg: int | None = None) -> torch.Tensor:
    """``out [planes L, L] = P (sum_a do v^T - D)`` per head plane, ``D[h, i] = sum_a dd[a, h, i]``: ``do`` / ``v`` bf16 ``[planes samples L, dh]``, ``P`` / ``out`` bf16 ``[planes L, L]``."""
    _ext().dpb_planes(do, v, P, dd, out, samples, planes, dh, (int(DPB_CFG) if DPB_CFG else 0) if cfg is None else cfg)
    return out
