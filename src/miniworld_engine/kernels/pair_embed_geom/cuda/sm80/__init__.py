"""sm_80 (A100) build and launch of the backbone pair-embedding kernels: ``pair_embed_geom_sm80.cu`` compiles ``sm90/pair_embed_geom.cu`` unchanged with
the sm_80 forms of its sm_90-only helpers (``pe_sm80.cuh``: the table gradient's warpgroup product as mma.sync). The launchers are the H100 ones
(``sm90/__init__.py``: the same arguments, scratch sizes and implementation choices) bound to this build. Built on first use, never at import."""

from __future__ import annotations

import functools
import sys
import types
from pathlib import Path

import torch

from miniworld_engine.kernels.pair_embed_geom.cuda import sm90 as _h100

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags
    from miniworld_engine.kernels._sm120 import sm80_load_extension as load_extension  # RTX 5090: the same source for sm_120a (kernels/_sm120.py)

    ensure_cuda_home()
    return load_extension(
        name="pair_embed_geom_sm80",
        sources=[str(_dir / "pair_embed_geom_sm80.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("80"), f"-I{_dir}"],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


#: the launchers' module globals: the H100 launcher module's, with ``ext`` this build (and this card's choices, set below)
_G = {**vars(_h100), "ext": ext}


def _on_this_build(fn):
    """The H100 launcher ``fn`` reading ``_G``."""
    return types.FunctionType(fn.__code__, _G, fn.__name__, fn.__defaults__, fn.__closure__)


forward = _on_this_build(_h100.forward)
forward_ca = _on_this_build(_h100.forward_ca)
ln_backward = _on_this_build(_h100.ln_backward)
#: the table gradient per mode on this card (measured L128-L384, L2 flushed): the encoder table (mode 0, 317 rows) on the warpgroup kernel
#: (impl 1, here mma.sync on its swizzled operands), the decoder's (mode 1, 157 rows) on the mma.sync kernel over KS concurrent pair chunks
#: (impl 3, pair_embed_geom_sm80.cu: the H100 grid of 132 CTAs left the A100's SMs too few warps)
BWD_IMPL = {0: 1, 1: 3}
#: RTX 5090 (sm_120a build): the encoder table's warpgroup kernel (impl 1) needs 126 KB of shared memory, above the card's 99 KB a block; both
#: tables take the packed-row mma.sync kernel pg120_bwd (pair_embed_geom_sm80.cu, sm_120a only): the encoder's at 64 channels a CTA (impl 4,
#: 10 warps), the decoder's at 128 (impl 5, 10 warps). Measured L128-L384, L2 flushed: 1.3x / 1.5x the Triton table gradient at L384, where
#: impl 3 (the A100's choice: pg_bwd_mma's 32-row feature groups, 416 / 160 rows, on 4 x 33 x KS CTAs) lost to it on the 170 SMs.
BWD_IMPL_SM120 = {0: 4, 1: 5}
KS = 2  # pair_embed_geom_sm80.cu
#: impl 4 / 5: the pair slices (= fp32 partials) of pg120_bwd per mode: 4 x 42 = 168 CTAs (one per SM) / 2 x 85 = 170
NB_SM120 = {0: 42, 1: 85}


def parts_of(impl, mode):
    """The partial count of the table gradient's fp32 scratch for ``impl``."""
    if impl == 3:
        return KS * _h100.NBLK
    return NB_SM120[mode] if impl >= 4 else _h100.NBLK


def backward(g, codes, mode):
    """Table gradient [N_ROWS[mode], 256] bf16 from d pair = g [B, n, n, 256] bf16 (the H100 launcher's arguments and scratch)."""
    nt = _h100.N_ROWS[mode]
    impl = (BWD_IMPL_SM120 if torch.cuda.get_device_capability(g.device) == (12, 0) else BWD_IMPL)[mode]
    parts = parts_of(impl, mode)
    tpart = torch.empty(parts * nt * 256, device=g.device, dtype=torch.float32)
    dtbl = torch.empty(nt, 256, device=g.device, dtype=torch.bfloat16)
    ext().backward(g, codes, tpart, dtbl, int(mode), impl)
    return dtbl


#: RTX 5090 (sm_120a build): the LayerNorm backward fused into the table gradient (pair_embed_geom_sm80.cu ln_table_grad_sm120, reached as
#: ln_backward's mode 2 / 3 / 5): the d pair is never written and read back (2 x 2 bytes x 256 channels a pair less traffic). Per mode its
#: ln_backward code (encoder 2: 64 channels a CTA; decoder 3: 128) and its pair slices (one CTA per SM).
LN_FUSED_SM120 = {0: 2, 1: 3}
NB_LN_SM120 = {0: 42, 1: 85}


def ln_table_grad(tbl, codes, dh, mode, eps):
    """RTX 5090: the table gradient [N_ROWS[mode], 256] bf16 straight from d LayerNorm(pair) = ``dh`` [B, n, n, 256] bf16 (the LayerNorm of
    ``eps``): per-pair statistics, then the product with each tile's d pair recomputed in shared memory. The result is a view of the fp32 scratch."""
    nt = _h100.N_ROWS[mode]
    pairs = dh.shape[0] * dh.shape[1] * dh.shape[2]
    nb = NB_LN_SM120[mode]
    head = nb * nt * 256 + 4 * pairs
    scratch = torch.empty(head + nt * 128, device=dh.device, dtype=torch.float32)
    ext().ln_backward(tbl, codes, dh, scratch, LN_FUSED_SM120[mode], float(eps))
    return scratch[head:].view(torch.bfloat16).view(nt, 256)


#: the modules' bin limits per (device, kind), made once outside a CUDA-graph capture (the values of modules/pair_embed.py's torch.linspace calls)
_LIMITS: dict = {}


def _limits(device, kind: str) -> torch.Tensor:
    lim = _LIMITS.get((device, kind))
    if lim is None:
        if kind == "dist":
            lim = torch.linspace(0.1, 2, 20, device=device)
        elif kind == "angle":
            lim = torch.linspace(-torch.pi, torch.pi, 20, device=device)
        else:
            lim = torch.linspace(0.1, 3.0, 29, device=device)
        if not torch.cuda.is_current_stream_capturing():
            _LIMITS[(device, kind)] = lim
    return lim


def _bytes(m: torch.Tensor) -> torch.Tensor:
    """A mask as bytes: a contiguous bool mask as it is (read as uint8), anything else cast."""
    return m if m.dtype in (torch.bool, torch.uint8) and m.is_contiguous() else m.to(torch.uint8).contiguous()


def _indices(*ts: torch.Tensor) -> tuple:
    """Integer inputs as they are when they share one of int32 / int64 and are contiguous, else int32 copies."""
    if ts[0].dtype in (torch.int32, torch.int64) and all(t.dtype is ts[0].dtype and t.is_contiguous() for t in ts):
        return ts
    return tuple(t.to(torch.int32).contiguous() for t in ts)


def forward_eval(weight, bb, ca_mask, has_cb, mask, residue_idx, chains, ln_eps):
    """RTX 5090 inference: the encoder pair [B, n, n, 256] bf16 (its LayerNorm when ``ln_eps`` >= 0) from the module's own inputs and its weight
    [256, 317] (no per-call preparation kernels: the extension's gt-2 launch, pair_embed_geom_sm80.cu)."""
    B, n = mask.shape
    dev = bb.device
    idx, ch = _indices(residue_idx, chains)
    out = torch.empty(B, n, n, 256, device=dev, dtype=torch.bfloat16)
    none = torch.empty(0, device=dev, dtype=torch.int32)
    ext().forward(
        weight.detach().contiguous(), bb.float().contiguous(), _bytes(ca_mask), _bytes(has_cb), _bytes(mask), idx, ch, _limits(dev, "dist"),
        _limits(dev, "angle"), out, none, int(ln_eps >= 0), float(max(ln_eps, 0.0)), 0, 2,
    )
    return out


def forward_ca_eval(weight, ca, mask, residue_idx, ln_eps):
    """RTX 5090 inference: the decoder pair [B, n, n, 256] bf16 (its LayerNorm when ``ln_eps`` >= 0) from the module's inputs and its weight [256, 157]."""
    B, n = mask.shape
    dev = ca.device
    if residue_idx is None:
        residue_idx = torch.arange(1, n + 1, device=dev, dtype=torch.int32).expand(B, n).contiguous()
    (idx,) = _indices(residue_idx)
    out = torch.empty(B, n, n, 256, device=dev, dtype=torch.bfloat16)
    none = torch.empty(0, device=dev, dtype=torch.int32)
    ext().forward_ca(
        weight.detach().contiguous(), ca.float().contiguous(), _bytes(mask), idx, _limits(dev, "ca"), out, none, int(ln_eps >= 0),
        float(max(ln_eps, 0.0)), 0, 2,
    )
    return out


def device_build():
    """The launchers of the current device's build: this one on an A100 (and its sm_120a build on an RTX 5090), else the H100 (sm_90a) one."""
    return sys.modules[__name__] if torch.cuda.get_device_capability() in ((8, 0), (12, 0)) else _h100
