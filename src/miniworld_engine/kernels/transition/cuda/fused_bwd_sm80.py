"""The fused sm_80 Transition backward at D = 64 / 128 (the PW and X roles of ``fused_sm80`` generalised over (D, H), ``sm80/tr_bwd_g_sm80.cuh`` + ``tr_bwd_sm80.cuh``).

Two kernels and a reduction: PW (one CTA per hidden slice of 8192 / D units x row replica: a, b and dh recomputed from the saved xn and dy, the SwiGLU backward, dA | dB handed
to X as fragment-native 16 x 16 blocks, dWa / dWb / dWs^T accumulated in registers over the replica's rows) and X (d_xn = [dA | dB] [Wa; Wb] with the LayerNorm backward + residual
in the accumulator layout -> dx and per-CTA dgamma / dbeta partials; the bare FFN: dx = d_xn); the partial sums are reduced in a fixed order in one launch (no atomics).  Needs
whole 256-row tiles.  It consumes what the wide forward and the fused forward save (xn, mean / rstd), so either forward can feed it.
"""

import functools
import hashlib
import os
import warnings
from pathlib import Path

import torch

from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from .fused_fwd_sm80 import SHAPES, _o_perm

_dir = Path(__file__).parent / "sm80"

#: Fewest rows for which the fused backward beats the wide chain (the PW / X tiles are sized for the whole card; CUDA graph, probes/tiny_graph.py, docs page): the kernel
#: takes ~41 us at (64, 256) from 1 k rows on against the chain's 45 / 53 / 48 us at 3 k / 5 k / 9 k rows; the other builds' crossover is at 8 k rows.
MIN_ROWS = {(64, 128): 8192, (64, 256): 2048, (128, 256): 8192, (128, 512): 8192}


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    extra = os.environ.get("MINIWORLD_TRANSITION_BWD_SM80_FLAGS", "").split()      # experiments: extra nvcc flags (their own build)
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"transition_bwd_sm80{tag}", sources=[str(_dir / "transition_bwd_sm80.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", "-DPW_XN", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=bool(extra),
    )


_BUILD_FAILED = False


@torch.compiler.assume_constant_result
def loads() -> bool:
    """The extension builds / loads; a failure warns once (the wide backward then serves).  Constant for dynamo."""
    global _BUILD_FAILED
    if _BUILD_FAILED:
        return False
    try:
        _ext()
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED = True
        warnings.warn(f"fused sm80 Transition backward unavailable, keeping the wide path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def supports(d: int, h: int, rows: int) -> bool:
    """A build exists for (d, h), whole 256-row tiles, enough rows to fill the card, the switch is not off."""
    if (d, h) not in SHAPES or rows % 256 or rows < MIN_ROWS[(d, h)]:
        return False
    return os.environ.get("MINIWORLD_TRANSITION_FUSED_BWD_SM80", "1") != "0"


def _layout_bwd(wa, wb, ws, dev, d, h):
    """PW: per slice of SU = 8192 / D hidden units [W1s: D / 8 k-granules x 2 SU rows (0.5 Wa | Wb of the 16-unit steps) | W3s: D / 8 x SU rows (Ws^T)];
    X: [Wa | Wb] per 32-unit chunk, output columns in the forward's permutation.  On element codes."""
    su = 8192 // d
    nsl, nchunk = h // su, h // 32
    op = _o_perm(dev, d)
    w1s = torch.stack([wa.view(nsl, su // 16, 16, d), wb.view(nsl, su // 16, 16, d)], 2).reshape(nsl, 2 * su, d).view(nsl, 2 * su, d // 8, 8).transpose(1, 2)
    w3s = ws.t().reshape(nsl, su, d // 8, 8).transpose(1, 2)
    wdw = torch.cat([w1s.reshape(nsl, -1), w3s.reshape(nsl, -1)], 1).reshape(-1)
    xa = wa[:, op].reshape(nchunk, 32, d // 8, 8).transpose(1, 2).reshape(nchunk, -1)
    xb = wb[:, op].reshape(nchunk, 32, d // 8, 8).transpose(1, 2).reshape(nchunk, -1)
    return wdw, torch.cat([xa, xb], 1).reshape(-1)


@functools.lru_cache(maxsize=16)
def _tables(dev, d, h):
    """Gather table of the pack kernel for [wdw | wx] (element | source << 26 (Wa, Wb, Ws) | x0.5 << 28) and the length of the wdw part."""
    n = h * d
    code = lambda src: (torch.arange(n, device=dev, dtype=torch.int64) | (src << 26))  # noqa: E731
    wa, wb, ws = code(0).view(h, d), code(1).view(h, d), code(2).view(d, h)
    wdw, wx = _layout_bwd(wa | (1 << 28), wb, ws, dev, d, h)                           # Wa carries the 1/2 in both roles (PW's dA tile holds 2 dA)
    return torch.cat([wdw, wx]).to(torch.int32).contiguous(), wdw.numel()


def backward(dy, x, xn, stats, gamma, wa, wb, ws, ln: bool, gdt: torch.dtype = torch.float32, wdt: torch.dtype = torch.bfloat16):
    """(dx, dgamma, dbeta [``gdt``; empty without ``ln``], [dWa; dWb] [2H, D], dWs [D, H] in ``wdt``: bf16, or f32 unrounded for an fp32 master) of x [M, D] bf16; ``xn`` is the saved LayerNorm output (the FFN: unused,
    x feeds PW); gamma f32."""
    d, h = x.shape[1], wa.shape[0]
    idx16, n_wdw = _tables(wa.device, d, h)
    packed = torch.empty(idx16.numel(), dtype=torch.bfloat16, device=wa.device)
    ext = _ext()
    ext.pack(wa, wb, ws, idx16, packed)
    nrep = int(os.environ.get("MINIWORLD_TRANSITION_BWD_SM80_NREP", "0"))
    return tuple(ext.bwd(d, h, ln, dy, x, xn if ln else x, stats, packed[:n_wdw], packed[n_wdw:], gamma, nrep, gdt, wdt))
